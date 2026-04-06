"""
api/main.py — FastAPI application entry point.

Architecture
------------
- All spike_discrim logic lives in src/; the API ONLY reads files from data/.
- Frontend lives in frontend/ and is served as static files at /.
- Three routers:
    /api/runs      — read existing benchmark results
    /api/pipeline  — trigger a new benchmark run (POST + poll)
    /api/validate  — objective proof endpoints (checksums, metric recomputation)

CORS is wide-open for development (localhost:*).  Lock down origins in prod.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routers import runs, pipeline, validate, guide, ephys_eval

# ── Paths ──────────────────────────────────────────────────────────────────── #
BASE_DIR     = Path(__file__).resolve().parent.parent  # project root
FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIR.mkdir(exist_ok=True)

# ── App ────────────────────────────────────────────────────────────────────── #
app = FastAPI(
    title       = "spike_discrim API",
    description = "Feature benchmarking for spike discrimination input layers.",
    version     = "0.1.0",
    docs_url    = "/api/docs",
    redoc_url   = "/api/redoc",
    openapi_url = "/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── API routers ────────────────────────────────────────────────────────────── #
app.include_router(runs.router,        prefix="/api")
app.include_router(pipeline.router,    prefix="/api")
app.include_router(validate.router,    prefix="/api")
app.include_router(guide.router,       prefix="/api")
app.include_router(ephys_eval.router,  prefix="/api")

# ── Static frontend (must be mounted LAST so /api routes take priority) ────── #
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ── Entry point for `spike-api` console script ────────────────────────────── #
def start() -> None:
    import uvicorn
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8099"))
    uvicorn.run("api.main:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    start()
