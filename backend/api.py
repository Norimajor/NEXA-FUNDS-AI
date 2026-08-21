from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.engine.strategy_interpreter.parser import StrategyParser

app = FastAPI(title="NEXA FUNDS AI")


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
        parser = StrategyParser()

        strategy = parser.parse(
            prompt
        )

        return {
            "success": True,
            "strategy": {
                "name": strategy.name,
                "symbol": strategy.symbol,
                "timeframe": strategy.timeframe,
                "direction": strategy.direction,
                "entry_conditions": [
                    {
                        "indicator": condition.indicator,
                        "operator": condition.operator,
                        "value": condition.value,
                        "period": condition.period,
                        "timeframe": condition.timeframe,
                    }
                    for condition in strategy.entry_conditions
                ],
            },
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )