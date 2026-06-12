import os
import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Set up Gemini
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up a simple SQLite database to store insights and outcomes
def init_db():
    conn = sqlite3.connect("asha.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            week TEXT,
            category TEXT,
            observation TEXT,
            recommendation TEXT,
            framing_style TEXT,
            dollar_amount REAL,
            created_at TEXT,
            outcome TEXT DEFAULT 'pending',
            judge_score REAL DEFAULT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            description TEXT,
            amount REAL,
            category TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    print("Database ready.")

def categorize_transactions(transactions: list[dict]) -> list[dict]:
    """Use Gemini to categorize a list of transactions."""
    
    tx_text = "\n".join([
        f"- {t['date']}: {t['description']} ${t['amount']}"
        for t in transactions
    ])
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"""
        Categorize each transaction into one of these categories:
        Food, Transport, Subscriptions, Shopping, Utilities, Groceries, Entertainment, Other
        
        Transactions:
        {tx_text}
        
        Return ONLY a JSON array like this, no other text:
        [{{"description": "Uber Eats", "category": "Food"}}, ...]
        """
    )
    
    try:
        text = response.text.strip()
        # Clean up any markdown formatting
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        categories = json.loads(text.strip())
        
        # Map categories back to transactions
        cat_map = {c["description"]: c["category"] for c in categories}
        for t in transactions:
            t["category"] = cat_map.get(t["description"], "Other")
        
        return transactions
    except Exception as e:
        print(f"Categorization error: {e}")
        for t in transactions:
            t["category"] = "Other"
        return transactions

def generate_insight(transactions: list[dict], past_bias_patterns: str = "") -> dict:
    """Generate a spending insight using Gemini, informed by past failures."""
    
    # Summarize spending by category
    df = pd.DataFrame(transactions)
    if df.empty:
        return {}
    
    summary = df.groupby("category")["amount"].sum().to_dict()
    summary_text = "\n".join([f"- {k}: ${v:.2f}" for k, v in summary.items()])
    
    # Build the self-improvement prompt injection
    bias_context = ""
    if past_bias_patterns:
        bias_context = f"""
        IMPORTANT - Learn from past failures:
        Your previous insights used these patterns that were IGNORED by the user:
        {past_bias_patterns}
        
        Avoid these patterns. Instead use specific dollar amounts and concrete goals.
        """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"""
        You are ASHA, a personal spending advisor. Analyze this week's spending and give ONE insight.
        
        Spending this week:
        {summary_text}
        
        {bias_context}
        
        Return ONLY a JSON object like this, no other text:
        {{
            "category": "Food",
            "observation": "You spent $284 on food delivery this week",
            "recommendation": "Set a $150 weekly limit on delivery to save $134",
            "framing_style": "dollar",
            "dollar_amount": 284.0
        }}
        
        Rules:
        - framing_style must be "dollar" or "percentage"  
        - Always prefer dollar amounts over percentages
        - Be specific, not vague
        - observation should state exactly what happened
        - recommendation should give a concrete dollar goal
        """
    )
    
    try:
        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        insight = json.loads(text.strip())
        return insight
    except Exception as e:
        print(f"Insight generation error: {e}")
        return {}

def judge_insight(insight: dict) -> float:
    """Use Gemini as a judge to score the insight quality 1-10."""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"""
        You are an evaluator scoring a financial insight for quality.
        
        Insight:
        Observation: {insight.get('observation', '')}
        Recommendation: {insight.get('recommendation', '')}
        Framing style: {insight.get('framing_style', '')}
        
        Score it 1-10 on these criteria:
        1. Specificity — does it mention exact dollar amounts?
        2. Actionability — is the recommendation concrete and achievable?
        3. Motivational framing — does it feel encouraging not shaming?
        
        Return ONLY a JSON object, no other text:
        {{"specificity": 8, "actionability": 7, "motivation": 9, "mean": 8.0}}
        """
    )
    
    try:
        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        scores = json.loads(text.strip())
        return scores.get("mean", 5.0)
    except Exception as e:
        print(f"Judge error: {e}")
        return 5.0

def get_bias_patterns() -> str:
    """Query the database for ignored insights and extract failure patterns."""
    
    conn = sqlite3.connect("asha.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT observation, recommendation, framing_style 
        FROM insights 
        WHERE outcome = 'ignored'
        ORDER BY created_at DESC
        LIMIT 10
    """)
    
    ignored = cursor.fetchall()
    conn.close()
    
    if not ignored:
        return ""
    
    patterns_text = "\n".join([
        f"- Framing: {row[2]} | Said: '{row[0]}' | Recommended: '{row[1]}'"
        for row in ignored
    ])
    
    if len(ignored) < 2:
        return patterns_text
    
    # Use Gemini to summarize the failure patterns
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"""
        These are financial insights the user IGNORED. 
        What patterns made them ineffective? Be brief and specific.
        
        {patterns_text}
        
        Return 2-3 bullet points only, no other text.
        """
    )
    
    return response.text.strip()

def save_insight(insight: dict, week: str, score: float):
    """Save an insight to the database."""
    conn = sqlite3.connect("asha.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO insights (week, category, observation, recommendation, framing_style, dollar_amount, created_at, judge_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        week,
        insight.get("category", ""),
        insight.get("observation", ""),
        insight.get("recommendation", ""),
        insight.get("framing_style", "dollar"),
        insight.get("dollar_amount", 0),
        datetime.now().isoformat(),
        score
    ))
    conn.commit()
    conn.close()

def update_outcome(insight_id: int, outcome: str):
    """Mark an insight as acted_on or ignored."""
    conn = sqlite3.connect("asha.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE insights SET outcome = ? WHERE id = ?", (outcome, insight_id))
    conn.commit()
    conn.close()

def get_all_insights() -> list[dict]:
    """Get all insights for the Loop page."""
    conn = sqlite3.connect("asha.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM insights ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": r[0], "week": r[1], "category": r[2],
            "observation": r[3], "recommendation": r[4],
            "framing_style": r[5], "dollar_amount": r[6],
            "created_at": r[7], "outcome": r[8], "judge_score": r[9]
        }
        for r in rows
    ]

if __name__ == "__main__":
    init_db()
    print("ASHA agent initialized.")