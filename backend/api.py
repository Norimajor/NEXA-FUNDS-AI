import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.engine.strategy_analysis.analysis_service import StrategyAnalysisService

app = FastAPI(title="NEXA FUNDS AI")

frontend_origin = os.getenv(
    "NEXAFUNDS_FRONTEND_ORIGIN",
    "https://nexafunds-steel.vercel.app",
).rstrip("/")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        frontend_origin,
        "https://nexafunds-steel.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


class StrategyRequest(BaseModel):
    prompt: str


@app.get("/")
def root():
    return {
        "success": True,
        "service": "NEXA FUNDS AI",
        "status": "online",
    }


@app.get("/health")
def health():
    return {
        "success": True,
        "status": "healthy",
    }


@app.post("/analyze")
def analyze_strategy(request: StrategyRequest):

    prompt = request.prompt.strip()

    if not prompt:
        raise HTTPException(
            status_code=400,
            detail="Strategy description is required.",
        )

    if len(prompt) > 4000:
        raise HTTPException(
            status_code=400,
            detail="Strategy description is too long.",
        )

    try:
        service = StrategyAnalysisService()
        result = service.analyze(prompt)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error") or "Strategy analysis failed.")
        return result
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )
