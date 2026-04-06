"""
api/routers/pipeline.py — Trigger and monitor benchmark pipeline runs.

POST /api/pipeline/run   → launches run_benchmark.py as a subprocess background task
                           returns {"job_id": "<uuid>"}

GET  /api/pipeline/status/{job_id}  → {"status": "running|done|failed",
                                        "run_id": "...",
                                        "log_tail": [...last 20 lines...]}

GET  /api/pipeline/jobs  → list all known jobs (newest first)

Jobs are tracked in an in-memory dict (sufficient for single-process dev server).
"""
from __future__ import annotations

import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root
PYTHON   = str(BASE_DIR / ".venv" / "bin" / "python")
SCRIPT   = str(BASE_DIR / "scripts" / "run_benchmark.py")

router = APIRouter(tags=["pipeline"])

# ── In-memory job registry ─────────────────────────────────────────────────── #

JobStatus = Literal["queued", "running", "done", "failed"]

_jobs: dict[str, dict] = {}    # job_id → job record
_lock = threading.Lock()


# ── Request / Response models ──────────────────────────────────────────────── #

class RunRequest(BaseModel):
    tier:          int  = 2
    data_dir:      str  = "data/synthetic"
    no_profile:    bool = False
    use_real_data: bool = False
    """When True, trains and benchmarks on waveforms_real.npz (real OSort data)
    instead of the default synthetic waveforms.npz."""


class JobResponse(BaseModel):
    job_id:   str
    status:   JobStatus
    run_id:   str | None = None
    log_tail: list[str]  = []
    started_at: str      = ""
    finished_at: str | None = None


# ── Background worker ─────────────────────────────────────────────────────── #

def _run_subprocess(job_id: str, cmd: list[str]) -> None:
    lines: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(BASE_DIR),
        )
        with _lock:
            _jobs[job_id]["pid"] = proc.pid
            _jobs[job_id]["status"] = "running"

        for line in proc.stdout:          # type: ignore[union-attr]
            line = line.rstrip()
            lines.append(line)
            with _lock:
                _jobs[job_id]["log"] = lines[-200:]  # keep last 200 lines

        proc.wait()
        finished = datetime.now(timezone.utc).isoformat()

        with _lock:
            if proc.returncode == 0:
                _jobs[job_id]["status"] = "done"
                # Extract run_id from last "Results in: data/results/XXXXXXXX" line
                for l in reversed(lines):
                    if "Results in:" in l:
                        _jobs[job_id]["run_id"] = l.split("Results in:")[-1].strip().split("/")[-1]
                        break
            else:
                _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["finished_at"] = finished

    except Exception as exc:
        with _lock:
            _jobs[job_id]["status"]      = "failed"
            _jobs[job_id]["log"]         = lines + [f"EXCEPTION: {exc}"]
            _jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()


# ── Endpoints ─────────────────────────────────────────────────────────────── #

@router.post("/pipeline/run", response_model=JobResponse)
def trigger_run(req: RunRequest) -> JobResponse:
    """Launch a new benchmark pipeline run as a background process."""
    job_id = str(uuid.uuid4())[:8]
    cmd = [PYTHON, SCRIPT,
           "--data-dir", req.data_dir,
           "--tier",     str(req.tier)]
    if req.use_real_data:
        real_npz = str(Path(BASE_DIR) / "data" / "real_units" / "waveforms_real.npz")
        cmd += ["--waveforms-file", real_npz]
    if req.no_profile:
        cmd.append("--no-profile")

    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        _jobs[job_id] = {
            "job_id":     job_id,
            "status":     "queued",
            "run_id":     None,
            "log":        [],
            "started_at": now,
            "finished_at": None,
            "cmd":        cmd,
        }

    t = threading.Thread(target=_run_subprocess, args=(job_id, cmd), daemon=True)
    t.start()

    return JobResponse(job_id=job_id, status="queued", started_at=now)


@router.get("/pipeline/status/{job_id}", response_model=JobResponse)
def job_status(job_id: str) -> JobResponse:
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobResponse(
        job_id      = job["job_id"],
        status      = job["status"],
        run_id      = job.get("run_id"),
        log_tail    = job["log"][-20:],
        started_at  = job["started_at"],
        finished_at = job.get("finished_at"),
    )


@router.get("/pipeline/jobs")
def list_jobs() -> list[dict]:
    with _lock:
        jobs = list(_jobs.values())
    return sorted(jobs, key=lambda j: j["started_at"], reverse=True)
