from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.intelligence.client_api.schemas import ClientIntelligenceRequest, ClientIntelligenceResponse
from app.intelligence.client_api.service import ClientIntelligenceService


router = APIRouter(
    prefix="/api/v1/client",
    tags=["Client Intelligence"],
)

_service = ClientIntelligenceService()


@router.get("/health")
def client_intelligence_health() -> dict:
    return {
        "service": "Restaurant Demand AI",
        "component": "Client JSON Intelligence API",
        "status": "ready",
        "input": "client-provided JSON",
        "synthetic_training": False,
    }


@router.post("/intelligence", response_model=ClientIntelligenceResponse)
def client_intelligence(request: ClientIntelligenceRequest):
    try:
        return _service.generate(request.model_dump(mode="json"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Client intelligence processing failed") from exc
