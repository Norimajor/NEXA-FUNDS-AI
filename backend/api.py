import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.engine.strategy_analysis.analysis_service import StrategyAnalysisService
from backend.analysis_persistence import AnalysisPersistenceError, AnalysisStore
from backend.combination_ranking import CombinationRankingService
from backend.llm.provider import LLMProviderError, get_llm_provider
from backend.conversation_assistant import ConversationalAssistant
from backend.backtest_jobs import backtest_jobs

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


class RecommendationRequest(BaseModel):
    fingerprint: str | None = None
    limit: int = Field(default=10, ge=1, le=100)


class ChatRequest(BaseModel):
    message: str
    user_id: str | None = None


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


@app.post("/chat")
def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required.")
    if len(message) > 4000:
        raise HTTPException(status_code=400, detail="Message is too long.")
    try:
        intent, entities = ConversationalAssistant().classifier.classify(message)
        if intent == "combination_search" and entities.get("all_downloaded_data"):
            job = backtest_jobs.submit(message, request.user_id or "anonymous")
            return {"success": True, "intent": intent, "status": job["status"], "job_id": job["job_id"]}
        result = ConversationalAssistant().respond(message, request.user_id or "anonymous")
        return {"success": True, **result}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/backtest/jobs")
def backtest_job_list():
    return {"success": True, "jobs": backtest_jobs.list()}


@app.get("/backtest/jobs/{job_id}")
def backtest_job(job_id: str):
    try:
        return {"success": True, **backtest_jobs.status(job_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Backtest job not found.")


def _analysis_store() -> AnalysisStore:
    return AnalysisStore()


@app.get("/history")
@app.get("/analysis/history")
def analysis_history(limit: int = Query(default=20, ge=1, le=100)):
    try:
        return {"success": True, "history": _analysis_store().recent(limit)}
    except AnalysisPersistenceError as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/recommendations")
def recommendations(request: RecommendationRequest):
    try:
        records = _analysis_store().recent(100)
        if request.fingerprint:
            records = [record for record in records if record["strategy_fingerprint"] == request.fingerprint]
        ranked = CombinationRankingService().rank(records, request.limit)
        return {"success": True, "recommendations": ranked}
    except (AnalysisPersistenceError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error))


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
        try:
            strategy = get_llm_provider().interpret_strategy(request.prompt)
        except LLMProviderError:
            raise HTTPException(
                status_code=502,
                detail="Strategy interpretation failed. Please try again.",
            )
        except ValueError:
            raise HTTPException(
                status_code=500,
                detail="LLM provider configuration error.",
            )

        service = StrategyAnalysisService()
        result = service.analyze(prompt, strategy=strategy)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error") or "Strategy analysis failed.")
        try:
            _analysis_store().insert(prompt, result)
        except AnalysisPersistenceError as error:
            raise HTTPException(status_code=500, detail=str(error))
        return result
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )
