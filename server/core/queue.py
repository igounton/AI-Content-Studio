import uuid
import threading
import time
import logging
from typing import Dict, Any, List, Optional
import contextvars
import datetime
import asyncio
from fastapi import WebSocket
import json
from concurrent.futures import ThreadPoolExecutor

from server.core.database import SessionLocal, Job as DBJob, JobLog as DBJobLog, JobStage as DBJobStage

# In-memory fast cache synchronized with SQLite database
_JOBS: Dict[str, Dict[str, Any]] = {}

current_job_id = contextvars.ContextVar("current_job_id", default=None)

# ---------------------------------------------------------------------------
# Worker Pool Configuration
# We limit to 2 concurrent generation tasks to prevent resource exhaustion
# ---------------------------------------------------------------------------
worker_pool = ThreadPoolExecutor(max_workers=2)

def submit_to_queue(func, *args, **kwargs):
    """Submits a job to the background worker pool."""
    return worker_pool.submit(func, *args, **kwargs)

# ---------------------------------------------------------------------------
# WebSocket Connection Manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.loop = None 

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
        if self.loop is None:
            self.loop = asyncio.get_running_loop()
        
        job_state = get_job(job_id)
        if job_state:
            await websocket.send_text(json.dumps({"type": "full_state", "data": job_state}))

    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            if websocket in self.active_connections[job_id]:
                self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

    async def _send_message_async(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            for connection in list(self.active_connections[job_id]):
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    logging.error(f"WebSocket send error: {e}")
                    self.disconnect(connection, job_id)

    def broadcast(self, job_id: str, message: dict):
        """Thread-safe method to send messages to WebSockets from sync code."""
        if not self.active_connections.get(job_id):
            return
            
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._send_message_async(job_id, message))
        except RuntimeError:
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(self._send_message_async(job_id, message), self.loop)

manager = ConnectionManager()

# ---------------------------------------------------------------------------
# Custom Logging Handler to Broadcast via WS & Persist
# ---------------------------------------------------------------------------
class JobLogHandler(logging.Handler):
    def emit(self, record):
        job_id = current_job_id.get()
        if job_id:
            msg = self.format(record)
            add_log(job_id, msg, level=record.levelname)

job_handler = JobLogHandler()
job_handler.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger().addHandler(job_handler)
logging.getLogger().setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Job Management Methods (SQLite Backed)
# ---------------------------------------------------------------------------
def create_job(topic: str, job_type: str = "pipeline", params: Optional[Dict[str, Any]] = None) -> str:
    job_id = str(uuid.uuid4())
    params_str = json.dumps(params) if params else None
    
    # In-memory state
    _JOBS[job_id] = {
        "id": job_id,
        "topic": topic,
        "job_type": job_type,
        "status": "queued",
        "progress": 0.0,
        "step": "Waiting in queue...",
        "result": None,
        "error": None,
        "logs": [],
        "stop_event": threading.Event()
    }
    
    # Persist to SQLite
    db = SessionLocal()
    try:
        db_job = DBJob(
            id=job_id,
            topic=topic,
            job_type=job_type,
            status="queued",
            progress=0,
            step="Waiting in queue...",
            params_json=params_str
        )
        db.add(db_job)
        db.commit()
    except Exception as e:
        logging.error(f"Failed to persist new job {job_id} to database: {e}")
        db.rollback()
    finally:
        db.close()
        
    return job_id

def update_job(
    job_id: str, 
    status: Optional[str] = None, 
    progress: Optional[float] = None, 
    step: Optional[str] = None, 
    result: Optional[str] = None, 
    error: Optional[str] = None
):
    if job_id not in _JOBS:
        # Check if job exists in database
        loaded = get_job(job_id)
        if not loaded:
            return
        _JOBS[job_id] = {
            **loaded,
            "stop_event": threading.Event()
        }
        
    job = _JOBS[job_id]
    updated = False
    
    if status is not None and job["status"] != status: 
        job["status"] = status
        updated = True
    if progress is not None and job["progress"] != progress: 
        job["progress"] = progress
        updated = True
    if step is not None and job["step"] != step: 
        job["step"] = step
        updated = True
    if result is not None: 
        job["result"] = result
        updated = True
    if error is not None: 
        job["error"] = error
        updated = True
        
    if updated:
        # Broadcast minimal state update
        manager.broadcast(job_id, {
            "type": "update",
            "status": job["status"],
            "progress": job["progress"],
            "step": job["step"],
            "result": job["result"],
            "error": job["error"],
            "data": {
                "status": job["status"],
                "progress": job["progress"],
                "step": job["step"],
                "result": job["result"],
                "error": job["error"]
            }
        })
        
        # Persist to database asynchronously or with quick session
        db = SessionLocal()
        try:
            db_job = db.query(DBJob).filter(DBJob.id == job_id).first()
            if db_job:
                if status is not None: db_job.status = status
                if progress is not None: db_job.progress = int(progress * 100) if progress <= 1.0 else int(progress)
                if step is not None: db_job.step = step
                if result is not None: db_job.result = result
                if error is not None: db_job.error = error
                db.commit()
        except Exception as e:
            logging.error(f"Database error updating job {job_id}: {e}")
            db.rollback()
        finally:
            db.close()

def add_log(job_id: str, msg: str, level: str = "INFO"):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_line = f"[{timestamp}] {msg}"
    
    current_prog = 0.0
    if job_id in _JOBS:
        _JOBS[job_id].setdefault("logs", []).append(log_line)
        current_prog = _JOBS[job_id].get("progress", 0.0)
    manager.broadcast(job_id, {
        "type": "log",
        "data": log_line,
        "message": msg,
        "progress": current_prog
    })
    
    # Save log to database
    db = SessionLocal()
    try:
        log_entry = DBJobLog(
            job_id=job_id,
            level=level,
            message=msg
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        db.rollback()
    finally:
        db.close()

def save_stage_checkpoint(job_id: str, stage_name: str, stage_data: Dict[str, Any], status: str = "completed"):
    """Saves checkpoint data for a specific pipeline stage."""
    db = SessionLocal()
    try:
        stage = db.query(DBJobStage).filter(DBJobStage.job_id == job_id, DBJobStage.stage_name == stage_name).first()
        if not stage:
            stage = DBJobStage(
                job_id=job_id,
                stage_name=stage_name,
                status=status,
                stage_data_json=json.dumps(stage_data)
            )
            db.add(stage)
        else:
            stage.status = status
            stage.stage_data_json = json.dumps(stage_data)
        db.commit()
    except Exception as e:
        logging.error(f"Failed to save stage checkpoint for {job_id}/{stage_name}: {e}")
        db.rollback()
    finally:
        db.close()

def get_stage_checkpoint(job_id: str, stage_name: str) -> Optional[Dict[str, Any]]:
    """Retrieves checkpoint data for a specific pipeline stage."""
    db = SessionLocal()
    try:
        stage = db.query(DBJobStage).filter(DBJobStage.job_id == job_id, DBJobStage.stage_name == stage_name).first()
        if stage and stage.stage_data_json:
            return json.loads(stage.stage_data_json)
        return None
    except Exception as e:
        logging.error(f"Failed to load stage checkpoint {job_id}/{stage_name}: {e}")
        return None
    finally:
        db.close()

def get_job_stages(job_id: str) -> List[Dict[str, Any]]:
    """Retrieves all stages recorded for a given job."""
    db = SessionLocal()
    try:
        stages = db.query(DBJobStage).filter(DBJobStage.job_id == job_id).all()
        return [
            {
                "stage_name": s.stage_name,
                "status": s.status,
                "data": json.loads(s.stage_data_json) if s.stage_data_json else None
            }
            for s in stages
        ]
    except Exception as e:
        logging.error(f"Failed to load stages for job {job_id}: {e}")
        return []
    finally:
        db.close()

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    # 1. Check in-memory cache
    if job_id in _JOBS:
        return {k: v for k, v in _JOBS[job_id].items() if k != "stop_event"}
        
    # 2. Check SQLite database
    db = SessionLocal()
    try:
        db_job = db.query(DBJob).filter(DBJob.id == job_id).first()
        if not db_job:
            return None
            
        # Fetch logs
        logs = db.query(DBJobLog).filter(DBJobLog.job_id == job_id).order_by(DBJobLog.timestamp.asc()).all()
        log_lines = [f"[{l.timestamp.strftime('%H:%M:%S')}] {l.message}" for l in logs]
        
        job_dict = {
            "id": db_job.id,
            "topic": db_job.topic,
            "job_type": db_job.job_type,
            "status": db_job.status,
            "progress": (db_job.progress / 100.0) if db_job.progress > 1 else float(db_job.progress),
            "step": db_job.step,
            "result": db_job.result,
            "error": db_job.error,
            "logs": log_lines,
        }
        
        # Populate in memory cache
        _JOBS[job_id] = {
            **job_dict,
            "stop_event": threading.Event()
        }
        return job_dict
    except Exception as e:
        logging.error(f"Error fetching job {job_id} from database: {e}")
        return None
    finally:
        db.close()

def stop_job(job_id: str):
    if job_id in _JOBS:
        _JOBS[job_id]["stop_event"].set()
        update_job(job_id, status="cancelled")
    else:
        # Mark cancelled in DB directly
        update_job(job_id, status="cancelled")

def recover_interrupted_jobs():
    """Identifies and marks any jobs that were abruptly terminated during server restart."""
    db = SessionLocal()
    try:
        interrupted = db.query(DBJob).filter(DBJob.status.in_(["running", "processing"])).all()
        for j in interrupted:
            j.status = "interrupted"
            j.error = "Job execution was interrupted by server restart. You can safely retry this job."
            logging.info(f"Recovered interrupted job {j.id} (topic: {j.topic})")
        db.commit()
    except Exception as e:
        logging.error(f"Error recovering interrupted jobs: {e}")
        db.rollback()
    finally:
        db.close()

def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent jobs from the persistent database."""
    db = SessionLocal()
    try:
        jobs = db.query(DBJob).order_by(DBJob.created_at.desc()).limit(limit).all()
        return [
            {
                "id": j.id,
                "topic": j.topic,
                "job_type": j.job_type,
                "status": j.status,
                "progress": (j.progress / 100.0) if j.progress > 1 else float(j.progress),
                "step": j.step,
                "result": j.result,
                "error": j.error,
                "created_at": j.created_at.isoformat() if j.created_at else None,
                "updated_at": j.updated_at.isoformat() if j.updated_at else None
            }
            for j in jobs
        ]
    except Exception as e:
        logging.error(f"Error listing jobs: {e}")
        return []
    finally:
        db.close()

# Automatically run recovery when module loads
recover_interrupted_jobs()
