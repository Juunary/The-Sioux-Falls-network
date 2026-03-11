# ============================================================
# Job manager — spawns training_worker subprocesses and
# streams metrics.jsonl changes to WebSocket clients.
#
# Architecture (from plan Section 9):
#   FastAPI event loop (asyncio)
#     ├── JobManager: subprocess spawn, status tracking
#     ├── asyncio file watcher: metrics.jsonl → WS broadcast
#     └── training_worker.py (subprocess, independent process)
#          ├── MaskablePPO training
#          ├── metrics.jsonl append
#          └── checkpoint saves
# ============================================================

from __future__ import annotations

import asyncio
import json
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

_JOBS: dict[str, "Job"] = {}

# Maximum number of concurrently running training subprocesses.
# Prevents resource exhaustion from rapid POST /training/start calls.
MAX_CONCURRENT_JOBS: int = 5


@dataclass
class Job:
    job_id: str
    config: dict
    status: str = "pending"    # pending / running / completed / failed
    pid: Optional[int] = None
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    error: Optional[str] = None
    _proc: Optional[subprocess.Popen] = field(default=None, repr=False, compare=False)
    _watchers: list[asyncio.Queue] = field(default_factory=list, repr=False, compare=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_proc", None)
        d.pop("_watchers", None)
        return d


# ============================================================
# Job Manager
# ============================================================

class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = _JOBS

    def running_count(self) -> int:
        """Return the number of currently running jobs."""
        return sum(1 for j in self._jobs.values() if j.status == "running")

    def start_job(self, job_id: str, config: dict) -> Job:
        """Spawn training_worker.py as a subprocess."""
        if job_id in self._jobs and self._jobs[job_id].status == "running":
            raise ValueError(f"Job '{job_id}' is already running")
        if self.running_count() >= MAX_CONCURRENT_JOBS:
            raise RuntimeError(
                f"Too many concurrent training jobs (limit: {MAX_CONCURRENT_JOBS}). "
                "Stop an existing job before starting a new one."
            )

        job = Job(job_id=job_id, config=config, status="running")

        proc = subprocess.Popen(
            [
                sys.executable, "-m", "backend.trainer.training_worker",
                "--job-id", job_id,
                "--config", json.dumps(config),
            ],
            # stdout/stderr go to the parent process's console — avoids pipe buffer
            # saturation that would block the training subprocess if output is not drained.
            stdout=None,
            stderr=None,
        )
        job._proc = proc
        job.pid = proc.pid
        self._jobs[job_id] = job

        # Start background monitor
        asyncio.create_task(self._monitor_job(job))
        asyncio.create_task(self._watch_metrics(job))

        return job

    def stop_job(self, job_id: str) -> bool:
        """Send SIGTERM to the worker process."""
        job = self._jobs.get(job_id)
        if job is None or job._proc is None:
            return False
        job._proc.terminate()
        job.status = "failed"
        job.error = "stopped by user"
        return True

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict]:
        return [j.to_dict() for j in self._jobs.values()]

    def subscribe(self, job_id: str) -> asyncio.Queue:
        """Return an asyncio.Queue that receives metrics lines as dicts."""
        queue: asyncio.Queue = asyncio.Queue()
        job = self._jobs.get(job_id)
        if job:
            job._watchers.append(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        job = self._jobs.get(job_id)
        if job and queue in job._watchers:
            job._watchers.remove(queue)

    # ---- Background tasks ----

    async def _monitor_job(self, job: Job) -> None:
        """Wait for subprocess to finish and update status."""
        if job._proc is None:
            return
        ret = await asyncio.to_thread(job._proc.wait)
        job.ended_at = time.time()
        if ret == 0:
            job.status = "completed"
        elif job.status != "failed":   # not manually stopped
            job.status = "failed"
            job.error = f"exit code {ret}"

    async def _watch_metrics(self, job: Job) -> None:
        """
        Tail metrics.jsonl and broadcast new lines to all WebSocket subscribers.
        Polls every 0.5 s until job is no longer running.
        """
        metrics_path = pathlib.Path("data") / "training" / job.job_id / "metrics.jsonl"
        last_size = 0

        while job.status == "running":
            await asyncio.sleep(0.5)
            try:
                if not metrics_path.exists():
                    continue
                current_size = metrics_path.stat().st_size
                if current_size <= last_size:
                    continue

                with open(metrics_path, encoding="utf-8") as f:
                    f.seek(last_size)
                    new_content = f.read()
                last_size = current_size

                for line in new_content.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for q in list(job._watchers):
                        await q.put(record)

            except Exception:
                pass  # file not yet created or read error — retry


# Singleton instance (shared across FastAPI app)
job_manager = JobManager()
