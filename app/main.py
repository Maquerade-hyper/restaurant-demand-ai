from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.autonomous import router as autonomous_router
from app.api.client_intelligence import router as client_intelligence_router
from app.api.daily_intelligence import router as daily_intelligence_router


app = FastAPI(
    title="Restaurant Demand AI",
    description=(
        "Production-oriented restaurant, bar, "
        "cloud-kitchen demand and supply intelligence API."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS
# Allows the standalone client dashboard to call the API.
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# API ROUTES
# ---------------------------------------------------------
app.include_router(autonomous_router)
app.include_router(daily_intelligence_router)
app.include_router(client_intelligence_router)


# ---------------------------------------------------------
# ROOT
# ---------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Restaurant Demand AI",
        "status": "online",
        "version": "1.0.0",
        "api": "/api/v1",
    }


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "restaurant-demand-ai",
    }