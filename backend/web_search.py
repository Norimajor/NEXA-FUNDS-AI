import json
import os
import re
from typing import Any
from urllib import parse, request


class WebSearchError(RuntimeError):
    """Raised when a configured web research provider cannot fetch live results."""


def needs_web_research(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False

    lowered = text.lower()

    if re.search(r"\b(?:search the web|look this up|research this|research online|what does the internet say|find current information)\b", lowered):
        return True

    if re.search(r"\b(?:joke|tell me a joke|hello|hi|hey|how are you|good morning|good evening|what's up)\b", lowered):
        return False

    if re.search(r"\b(?:backtest|back test|strategy.*backtest|walk[- ]forward|analy[sz]e.*strategy|simulate.*strategy)\b", lowered):
        return False

    if re.search(r"\b(?:what is rsi|what is macd|difference between rsi and macd|explain countercurrent multiplication|what is the difference between rsi and macd)\b", lowered):
        return False

    if re.search(r"\b(?:current|latest|recent|today|yesterday|this week|this month|now|just now|breaking|latest developments)\b", lowered):
        if re.search(r"\b(?:rate|rates|fed|federal funds|inflation|gold|xauusd|economy|market|news|policy|jobs|unemployment|gdp|price|prices)\b", lowered):
            return True

    if re.search(r"\b(?:gold|xauusd|fed|federal funds|inflation|jobs|gdp|economy|rate|rates|policy|market news)\b", lowered):
        if re.search(r"\b(?:today|recent|latest|current|what happened|news|outlook|update|release|report)\b", lowered):
            return True

    return False


def _provider_name() -> str:
    return (os.getenv("WEB_SEARCH_PROVIDER", "tavily") or "tavily").strip().lower()


def _api_key() -> str:
    for env_name in ("TAVILY_API_KEY", "WEB_SEARCH_API_KEY"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return ""


def _coerce_results(payload: Any) -> list[dict[str, str]]:
    results = payload.get("results") if isinstance(payload, dict) else []
    if not isinstance(results, list):
        raise WebSearchError("The search provider returned no usable results.")
    normalized: list[dict[str, str]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or item.get("name") or ("Source" if url else "Untitled")).strip()
        snippet = str(item.get("content") or item.get("snippet") or "").strip()
        source = str(item.get("source") or (parse.urlparse(url).netloc if url else "unknown")).strip()
        published_date = str(item.get("published_date") or item.get("date") or item.get("published") or "").strip() or None
        if not url and not snippet:
            continue
        normalized.append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "source": source,
            "published_date": published_date,
        })
    if not normalized:
        raise WebSearchError("The search provider returned no usable results.")
    return normalized


def _tavily_search(query: str, max_results: int) -> dict[str, Any]:
    api_key = _api_key()
    if not api_key:
        raise WebSearchError("Web research is not configured on this deployment: set TAVILY_API_KEY or WEB_SEARCH_API_KEY.")

    payload = json.dumps({
        "query": query,
        "max_results": max(1, min(int(max_results or 5), 10)),
        "search_depth": "basic",
        "include_answer": True,
        "include_raw_content": False,
    }).encode("utf-8")

    request_obj = request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(request_obj, timeout=20) as response:
            body = response.read().decode("utf-8")
    except Exception as exc:  # pragma: no cover - runtime network error path
        raise WebSearchError(f"Web research request failed: {exc}") from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:  # pragma: no cover - provider payload issue
        raise WebSearchError("Web research provider returned invalid JSON.") from exc

    return {
        "query": query,
        "provider": "tavily",
        "results": _coerce_results(data),
        "answer": data.get("answer"),
    }


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise WebSearchError("Search query cannot be empty.")

    provider_name = _provider_name()
    if provider_name in {"", "none", "disabled", "off"}:
        raise WebSearchError("Web research is not configured on this deployment.")
    if provider_name != "tavily":
        raise WebSearchError(f"Unsupported web search provider '{provider_name}'.")
    return _tavily_search(query.strip(), max_results=max_results)
