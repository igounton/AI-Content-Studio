import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from server.routers import (
    generation, publish, tools, brand, studio,
    campaigns, clipper, director, videofx,
    explainer, social, style, timeline
)
from server.core.scheduler import start_scheduler
from server.core.queue import stop_job
from server.core.model_gateway import gateway
from config import load_config, save_config

# Ensure workspace and output directories exist
for d in ("workspace", "outputs", "agent_workspace", "assets", "videofx_outputs"):
    os.makedirs(d, exist_ok=True)

app = FastAPI(
    title="AI Content Studio Engine",
    description="Scalable backend API for AI video generation",
    version="2.0.0"
)

@app.on_event("startup")
def on_startup():
    start_scheduler()

# Restricted CORS configuration
cors_origins_env = os.getenv("CORS_ORIGINS", "")
if cors_origins_env:
    allowed_origins = [orig.strip() for orig in cors_origins_env.split(",") if orig.strip()]
else:
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8008",
        "http://127.0.0.1:8008"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.middleware("http")
async def api_token_auth_middleware(request: Request, call_next):
    """
    Optional API token enforcement: if STUDIO_API_TOKEN or API_AUTH_TOKEN is configured,
    verifies X-API-Key or Bearer token on protected /api/ endpoints.
    """
    if request.method == "OPTIONS" or request.url.path in ("/health", "/api/social/youtube/callback"):
        return await call_next(request)

    required_token = os.getenv("STUDIO_API_TOKEN") or load_config().get("API_AUTH_TOKEN", "")
    if required_token and request.url.path.startswith("/api/"):
        auth_header = request.headers.get("Authorization", "")
        api_key_header = request.headers.get("X-API-Key", "")
        query_token = request.query_params.get("token", "")
        bearer_token = auth_header.replace("Bearer ", "").strip() if auth_header.startswith("Bearer ") else ""
        if required_token not in (api_key_header, bearer_token, query_token):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized: Invalid or missing API token."})

    return await call_next(request)

# Include routers
app.include_router(generation.router, prefix="/api/generate", tags=["Generation"])
app.include_router(clipper.router, prefix="/api/clipper", tags=["Clipper"])
app.include_router(videofx.router, prefix="/api/videofx", tags=["VideoFX Flow"])
app.include_router(director.router, prefix="/api/director", tags=["Director"])
app.include_router(explainer.router, prefix="/api/explainer", tags=["Explainer"])
app.include_router(brand.router, prefix="/api/brand", tags=["Brand"])
app.include_router(tools.router, prefix="/api/tools", tags=["Tools"])
app.include_router(publish.router, prefix="/api/publish", tags=["Publish"])
app.include_router(studio.router, prefix="/api/studio", tags=["Studio"])
app.include_router(campaigns.router, prefix="/api/campaigns", tags=["Campaigns"])
app.include_router(social.router, prefix="/api/social", tags=["Social"])
# Mount both /api/styles and /api/style so singular & plural frontend calls succeed
app.include_router(style.router, prefix="/api/styles", tags=["Style Engine"])
app.include_router(style.router, prefix="/api/style", tags=["Style Engine"])
app.include_router(timeline.router, prefix="/api/timeline", tags=["Timeline Editor"])

@app.post("/api/cancel/{job_id}")
def cancel_job_alias(job_id: str):
    """Alias route for job cancellation."""
    stop_job(job_id)
    return {"message": f"Cancellation requested for {job_id}"}

@app.get("/api/config")
def get_config():
    # Never expose unmasked secrets to the browser
    return load_config(masked=True)

@app.post("/api/config")
def update_config(new_config: dict):
    # Load current config to merge updates safely without clobbering with masked placeholders
    current = load_config()
    current.update(new_config)
    save_config(current)
    gateway.refresh_config()
    return {"message": "Configuration updated successfully."}

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "AI Content Studio Engine"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8008, reload=True)
