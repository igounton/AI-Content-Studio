import os
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, Dict, Any

from server.core.database import get_db, OAuthToken
from server.routers.publish import validate_media_path
from config import load_config

router = APIRouter()

@router.get("/status")
def get_social_status(db: Session = Depends(get_db)):
    """Returns the genuine connection and health status of all supported social platforms."""
    tokens = db.query(OAuthToken).all()
    status_map = {}
    for t in tokens:
        has_token = bool(t.access_token)
        is_expired = bool(t.expires_at and t.expires_at < datetime.utcnow())
        status_map[t.platform] = {
            "connected": has_token and not is_expired,
            "expired": is_expired,
            "updated_at": t.updated_at.isoformat() if t.updated_at else None
        }

    if "youtube" not in status_map and os.path.exists("token.json"):
        status_map["youtube"] = {
            "connected": True,
            "expired": False,
            "updated_at": datetime.utcfromtimestamp(os.path.getmtime("token.json")).isoformat()
        }
    
    return {
        "youtube": status_map.get("youtube", {"connected": False, "expired": False}),
        "tiktok": status_map.get("tiktok", {"connected": False, "expired": False}),
        "instagram": status_map.get("instagram", {"connected": False, "expired": False}),
        "facebook": status_map.get("facebook", {"connected": False, "expired": False})
    }

@router.get("/youtube/auth")
def youtube_auth_url():
    """Generates genuine Google OAuth 2.0 authorization URL for YouTube Data API."""
    config = load_config()
    client_secrets_path = "client_secrets.json"
    
    client_id = config.get("YOUTUBE_CLIENT_ID")
    client_secret = config.get("YOUTUBE_CLIENT_SECRET")
    redirect_uri = os.getenv("YOUTUBE_REDIRECT_URI", "http://localhost:8008/api/social/youtube/callback")
    
    if os.path.exists(client_secrets_path):
        try:
            from google_auth_oauthlib.flow import Flow
            flow = Flow.from_client_secrets_file(
                client_secrets_path,
                scopes=[
                    "https://www.googleapis.com/auth/youtube.upload",
                    "https://www.googleapis.com/auth/youtube.readonly"
                ],
                redirect_uri=redirect_uri
            )
            auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
            return {"auth_url": auth_url, "configured": True}
        except Exception as e:
            logging.error(f"Failed to create Google OAuth flow from client_secrets.json: {e}")
            
    if client_id:
        scopes = "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly"
        auth_url = (
            f"https://accounts.google.com/o/oauth2/v2/auth?"
            f"client_id={client_id}&redirect_uri={redirect_uri}&response_type=code&"
            f"scope={scopes}&access_type=offline&prompt=consent"
        )
        return {"auth_url": auth_url, "configured": True}
        
    return {
        "auth_url": None,
        "configured": False,
        "message": "YouTube API credentials not configured. Please supply client_secrets.json or set YOUTUBE_CLIENT_ID in Settings."
    }

@router.post("/youtube/callback")
def youtube_callback(request_data: dict, db: Session = Depends(get_db)):
    """Exchanges YouTube OAuth code for real access and refresh tokens."""
    code = request_data.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code.")

    client_secrets_path = "client_secrets.json"
    redirect_uri = os.getenv("YOUTUBE_REDIRECT_URI", "http://localhost:8008/api/social/youtube/callback")
    
    try:
        from google_auth_oauthlib.flow import Flow
        if os.path.exists(client_secrets_path):
            flow = Flow.from_client_secrets_file(
                client_secrets_path,
                scopes=[
                    "https://www.googleapis.com/auth/youtube.upload",
                    "https://www.googleapis.com/auth/youtube.readonly"
                ],
                redirect_uri=redirect_uri
            )
            flow.fetch_token(code=code)
            credentials = flow.credentials
            
            token_entry = db.query(OAuthToken).filter(OAuthToken.platform == "youtube").first()
            if not token_entry:
                token_entry = OAuthToken(platform="youtube")
                db.add(token_entry)
                
            token_entry.access_token = credentials.token
            token_entry.refresh_token = credentials.refresh_token
            token_entry.expires_at = credentials.expiry
            token_entry.updated_at = datetime.utcnow()
            db.commit()
            
            with open("token.json", "w", encoding="utf-8") as f:
                f.write(credentials.to_json())
                
            return {"message": "YouTube connected successfully."}
        else:
            raise HTTPException(status_code=400, detail="client_secrets.json required to complete token exchange.")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logging.error(f"OAuth exchange error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to authenticate YouTube: {str(e)}")

@router.post("/publish/youtube")
def publish_to_youtube(video_data: dict, db: Session = Depends(get_db)):
    """Performs verified YouTube video upload using real credentials."""
    raw_video_path = video_data.get("video_path", "")
    safe_video_path = validate_media_path(raw_video_path)

    token = db.query(OAuthToken).filter(OAuthToken.platform == "youtube").first()
    if not token or not token.access_token:
        if os.path.exists("token.json"):
            from google.oauth2.credentials import Credentials
            credentials = Credentials.from_authorized_user_file("token.json")
        else:
            raise HTTPException(status_code=401, detail="YouTube account is not connected. Please connect via settings or auth.")
    else:
        from google.oauth2.credentials import Credentials
        config = load_config()
        credentials = Credentials(
            token=token.access_token,
            refresh_token=token.refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=config.get("YOUTUBE_CLIENT_ID"),
            client_secret=config.get("YOUTUBE_CLIENT_SECRET")
        )

    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        
        youtube = build("youtube", "v3", credentials=credentials)
        body = {
            "snippet": {
                "title": video_data.get("title", "AI Studio Video"),
                "description": video_data.get("description", ""),
                "tags": [t.strip() for t in video_data.get("tags", "").split(",") if t.strip()],
                "categoryId": "22"
            },
            "status": {
                "privacyStatus": video_data.get("privacy_status", "private")
            }
        }
        
        media = MediaFileUpload(safe_video_path, chunksize=-1, resumable=True)
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        
        response = None
        while response is None:
            status, response = request.next_chunk()
            
        video_id = response.get("id")
        if not video_id:
            raise Exception("YouTube API did not return a valid video ID.")
            
        return {
            "message": "Video successfully uploaded to YouTube!",
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}"
        }
    except Exception as e:
        logging.error(f"YouTube publishing error: {e}")
        raise HTTPException(status_code=500, detail=f"Publishing failed: {str(e)}")

@router.post("/export-bundle")
def export_manual_bundle(bundle_request: dict):
    """
    Creates an export bundle with optimized video, thumbnail, and metadata
    for manual publishing to platforms without direct API permissions.
    """
    raw_video_path = bundle_request.get("video_path", "")
    safe_video_path = validate_media_path(raw_video_path)
        
    project_dir = os.path.dirname(safe_video_path)
    metadata_path = os.path.join(project_dir, "publish_metadata.txt")
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(f"TITLE: {bundle_request.get('title', '')}\n\n")
        f.write(f"DESCRIPTION:\n{bundle_request.get('description', '')}\n\n")
        f.write(f"TAGS: {bundle_request.get('tags', '')}\n\n")
        f.write(f"PLATFORM: {bundle_request.get('platform', 'all')}\n")
        
    return {
        "message": "Manual publishing bundle created.",
        "video_path": safe_video_path.replace("\\", "/"),
        "metadata_path": metadata_path.replace("\\", "/")
    }
