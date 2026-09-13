from __future__ import annotations

from fastapi import FastAPI


from app.api.autonomous import (
    router as autonomous_router,
)

from app.api.daily_intelligence import (
    router as daily_intelligence_router,
)


app = FastAPI(
    title="Restaurant Demand AI",
    description=(
        "Production-oriented restaurant, bar, "
        "cloud-kitchen demand and supply intelligence API."
    ),
    version="1.0.0",
)


app.include_router(
    autonomous_router
)

app.include_router(
    daily_intelligence_router
)


@app.get("/")
def root():
    return {
        "service": "Restaurant Demand AI",
        "status": "online",
        "version": "1.0.0",
        "api": "/api/v1",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "restaurant-demand-ai",
    }