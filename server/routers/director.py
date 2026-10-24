import os
import json
import traceback
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from server.core.queue import create_job, submit_to_queue, update_job, manager, current_job_id
from server.core.director_engine import run_director_pipeline
from server.core.orchestrator import AutonomousStudioOrchestrator
from server.core.agent_system import TrendDiscoveryAgent

router = APIRouter()

class DirectorRequest(BaseModel):
    url: str
    remove_silence: bool = True
    add_punch_ins: bool = True
    add_sfx: bool = True

class AutonomousPlanRequest(BaseModel):
    topic: str
    duration_sec: int = 45
    style: str = "Cinematic Documentary"
    region: str = "Global"
    auto_approve: bool = False

class RenderPlanRequest(BaseModel):
    job_id: str
    scenes: Optional[List[Dict[str, Any]]] = None

def run_director_task(job_id: str, req: DirectorRequest):
    current_job_id.set(job_id)
    def update_callback(j_id, step, progress, result=None, error=None):
        status = "processing"
        if error: status = "failed"
        elif progress >= 1.0: status = "completed"
        update_job(j_id, status=status, progress=progress, step=step, result=result, error=error)
        manager.broadcast(j_id, {"type": "log", "data": step, "message": step, "progress": progress})

    try:
        run_director_pipeline(
            req.url, 
            job_id, 
            update_callback,
            remove_silence=req.remove_silence,
            add_punch_ins=req.add_punch_ins,
            add_sfx=req.add_sfx
        )
    except Exception as e:
        traceback.print_exc()
        update_job(job_id, status="failed", error=str(e))
        manager.broadcast(job_id, {"type": "log", "data": f"Error: {str(e)}", "message": f"Error: {str(e)}", "progress": 1.0})

def run_autonomous_orchestrator_task(job_id: str, req: AutonomousPlanRequest):
    current_job_id.set(job_id)
    try:
        orchestrator = AutonomousStudioOrchestrator(job_id=job_id)
        result = orchestrator.produce_content_pipeline(
            topic=req.topic,
            duration_sec=req.duration_sec,
            style=req.style,
            region=req.region,
            auto_approve=req.auto_approve
        )
        manager.broadcast(job_id, {"type": "plan_ready", "data": result})
        if req.auto_approve:
            orchestrator.render_approved_storyboard()
    except Exception as e:
        traceback.print_exc()
        update_job(job_id, status="failed", error=str(e))
        manager.broadcast(job_id, {"type": "log", "data": f"Error: {str(e)}", "message": f"Error: {str(e)}", "progress": 1.0})

def run_render_plan_task(job_id: str, scenes: Optional[List[Dict[str, Any]]] = None):
    current_job_id.set(job_id)
    try:
        update_job(job_id, status="running", step="Starting Master Video Render", progress=0.75)
        orchestrator = AutonomousStudioOrchestrator(job_id=job_id)
        orchestrator.render_approved_storyboard(scenes_override=scenes)
    except Exception as e:
        traceback.print_exc()
        update_job(job_id, status="failed", error=str(e))
        manager.broadcast(job_id, {"type": "log", "data": f"Error: {str(e)}", "message": f"Error: {str(e)}", "progress": 1.0})

class RegenerateSceneRequest(BaseModel):
    job_id: str
    scene_number: int
    instruction: str = "Make it more visually cinematic and punchy"

class UpdateScenesRequest(BaseModel):
    job_id: str
    scenes: List[Dict[str, Any]]

@router.post("/process")
def start_director(req: DirectorRequest):
    job_id = create_job(f"director_{req.url[:20]}")
    submit_to_queue(run_director_task, job_id, req)
    return {"job_id": job_id, "message": "Director job queued successfully."}

@router.post("/autonomous-plan")
def start_autonomous_planning(req: AutonomousPlanRequest):
    job_id = create_job(f"plan_{req.topic[:25]}", job_type="plan")
    submit_to_queue(run_autonomous_orchestrator_task, job_id, req)
    return {"job_id": job_id, "message": "Autonomous creative planning queued successfully."}

@router.post("/render-plan")
def render_approved_plan(req: RenderPlanRequest):
    """Triggers master video rendering for an approved storyboard plan."""
    submit_to_queue(run_render_plan_task, req.job_id, req.scenes)
    return {"job_id": req.job_id, "message": "Master video render started."}

@router.get("/plan/{job_id}")
def get_storyboard_plan(job_id: str):
    """Retrieves current storyboard plan and checkpoint stages for a job."""
    from server.core.queue import get_stage_checkpoint, get_job
    checkpoint = get_stage_checkpoint(job_id, "storyboard")
    if not checkpoint:
        job = get_job(job_id)
        if job and job.get("result"):
            try:
                checkpoint = json.loads(job["result"])
            except Exception:
                pass
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Storyboard not found for job")
    return {"job_id": job_id, "plan": checkpoint}

@router.post("/regenerate-scene")
def regenerate_scene_endpoint(req: RegenerateSceneRequest):
    """Regenerates a single scene in the storyboard according to user instruction."""
    try:
        updated_plan = AutonomousStudioOrchestrator.regenerate_single_scene(
            job_id=req.job_id,
            scene_number=req.scene_number,
            instruction=req.instruction
        )
        return {"status": "success", "plan": updated_plan}
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update-scenes")
def update_scenes_endpoint(req: UpdateScenesRequest):
    """Saves updated scenes (locks, narration, descriptions) to the storyboard."""
    try:
        updated_plan = AutonomousStudioOrchestrator.update_storyboard_scenes(
            job_id=req.job_id,
            updated_scenes=req.scenes
        )
        return {"status": "success", "plan": updated_plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trends")
def get_trend_opportunities(niche: str = Query("technology"), region: str = Query("Global"), count: int = Query(4)):
    """Discovers high-velocity topic opportunities using the Trend Discovery Agent."""
    agent = TrendDiscoveryAgent()
    opportunities = agent.discover_opportunities(niche=niche, region=region, count=count)
    return {"opportunities": [o.model_dump() for o in opportunities]}
