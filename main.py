from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
import pandas as pd
import io
import os
from agent import (
    init_db,
    categorize_transactions,
    generate_insight,
    judge_insight,
    get_bias_patterns,
    save_insight,
    update_outcome,
    get_all_insights,
)

load_dotenv()
init_db()

app = FastAPI(title="ASHA Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ASHA backend is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Parse a bank CSV and categorize transactions with Gemini."""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), skiprows=1)

        # Normalize column names — handle different bank formats
        df.columns = [c.strip().lower() for c in df.columns]

        # Try to find date, description, amount columns
        date_col = next((c for c in df.columns if "date" in c), None)
        desc_col = next((c for c in df.columns if any(x in c for x in ["desc", "name", "merchant", "payee", "transaction"])), None)
        amt_col = next((c for c in df.columns if any(x in c for x in ["amount", "amt", "debit", "credit"])), None)

        if not all([date_col, desc_col, amt_col]):
            raise HTTPException(status_code=400, detail=f"Could not find required columns. Found: {list(df.columns)}")

        transactions = []
        for _, row in df.iterrows():
            try:
                amount = float(str(row[amt_col]).replace("$", "").replace(",", "").replace("-", ""))
                transactions.append({
                    "date": str(row[date_col]),
                    "description": str(row[desc_col]),
                    "amount": amount,
                })
            except Exception:
                continue

        # Categorize with Gemini
        categorized = categorize_transactions(transactions[:50])

        return {
            "success": True,
            "transaction_count": len(categorized),
            "transactions": categorized[:20],
            "summary": pd.DataFrame(categorized).groupby("category")["amount"].sum().to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Full ASHA pipeline — upload CSV, generate insight, judge it, save it."""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), skiprows=1)
        df.columns = [c.strip().lower() for c in df.columns]

        date_col = next((c for c in df.columns if "date" in c), None)
        desc_col = next((c for c in df.columns if any(x in c for x in ["desc", "name", "merchant", "payee", "transaction"])), None)
        amt_col = next((c for c in df.columns if any(x in c for x in ["amount", "amt", "debit", "credit"])), None)

        transactions = []
        for _, row in df.iterrows():
            try:
                amount = float(str(row[amt_col]).replace("$", "").replace(",", "").replace("-", ""))
                transactions.append({
                    "date": str(row[date_col]),
                    "description": str(row[desc_col]),
                    "amount": amount,
                })
            except Exception:
                continue

        # Step 1 — categorize
        categorized = categorize_transactions(transactions[:50])

        # Step 2 — get bias patterns from past ignored insights (the self-improvement loop)
        bias_patterns = get_bias_patterns()

        # Step 3 — generate insight informed by past failures
        insight = generate_insight(categorized, bias_patterns)

        # Step 4 — judge the insight quality
        score = judge_insight(insight)

        # Step 5 — save to database
        from datetime import datetime
        week = datetime.now().strftime("Wk %W")
        save_insight(insight, week, score)

        return {
            "success": True,
            "insight": insight,
            "judge_score": score,
            "bias_patterns_used": bias_patterns,
            "transaction_count": len(categorized),
            "summary": pd.DataFrame(categorized).groupby("category")["amount"].sum().to_dict()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class OutcomeUpdate(BaseModel):
    insight_id: int
    outcome: str

@app.post("/outcome")
def record_outcome(body: OutcomeUpdate):
    """Record whether the user acted on or ignored an insight."""
    if body.outcome not in ["acted_on", "ignored"]:
        raise HTTPException(status_code=400, detail="outcome must be acted_on or ignored")
    update_outcome(body.insight_id, body.outcome)
    return {"success": True, "message": f"Insight {body.insight_id} marked as {body.outcome}"}

@app.get("/insights")
def get_insights():
    """Get all insights for the Loop page."""
    insights = get_all_insights()
    return {"insights": insights}

@app.get("/loop-stats")
def loop_stats():
    """Get stats for the ASHA Loop page."""
    insights = get_all_insights()
    
    total = len(insights)
    ignored = len([i for i in insights if i["outcome"] == "ignored"])
    acted = len([i for i in insights if i["outcome"] == "acted_on"])
    
    scored = [i for i in insights if i["judge_score"] is not None]
    avg_score = sum(i["judge_score"] for i in scored) / len(scored) if scored else 0
    
    return {
        "total_traces": total,
        "ignored_insights": ignored,
        "acted_on": acted,
        "avg_quality_score": round(avg_score, 1),
        "insights": insights
    }