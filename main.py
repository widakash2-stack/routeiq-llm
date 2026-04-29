import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from routing_engine import route_transaction_with_trace

app = FastAPI(title="RouteIQ API", version="1.0.0")


# ── REQUEST / RESPONSE MODELS ────────────────────────────────────────────────

class RouteRequest(BaseModel):
    country: str
    payment_method: str
    amount: float = 500


class RouteResponse(BaseModel):
    recommended_psp: str
    confidence: float
    reason: str
    fallback_psp: str
    processing_time_ms: int | str


# ── ENDPOINT ─────────────────────────────────────────────────────────────────

@app.post("/route", response_model=RouteResponse)
def route(request: RouteRequest):
    start = time.time()

    transaction = {
        "country":        request.country.upper(),
        "payment_method": request.payment_method.upper().replace(" ", "_"),
        "amount":         request.amount,
        "time_bucket":    "afternoon",
    }

    selected_psp, trace = route_transaction_with_trace(transaction)

    if not selected_psp:
        raise HTTPException(
            status_code=422,
            detail=f"No PSPs available for {request.country} / {request.payment_method}"
        )

    ranking = trace.get("psp_ranking", [])

    # Confidence = normalised final_score of winner (0–1 already, round to 2dp)
    top_score = ranking[0]["final_score"] if ranking else 0.0
    confidence = round(min(top_score, 1.0), 2)

    # Fallback = second-ranked PSP (if any), "none" if only one PSP available
    fallback_psp = ranking[1]["psp"] if len(ranking) > 1 else "none"

    elapsed_ms = round((time.time() - start) * 1000)
    processing_time_ms = "< 1" if elapsed_ms == 0 else elapsed_ms

    return RouteResponse(
        recommended_psp    = selected_psp,
        confidence         = confidence,
        reason             = trace.get("why_winner", ""),
        fallback_psp       = fallback_psp,
        processing_time_ms = processing_time_ms,
    )


# ── RUN ───────────────────────────────────────────────────────────────────────
# uvicorn api:app --reload --port 8000
