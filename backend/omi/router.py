"""FastAPI router for Omi voice input streams and webhooks."""

from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.omi.listener import OmiVoiceListener, get_omi_listener

router = APIRouter()


class VoiceInputRequest(BaseModel):
    """Request payload for incoming spoken user instruction."""

    transcript: str = Field(..., description="Transcribed voice text spoken by user")
    speaker: str = Field(default="user", description="Speaker identity or user name")
    session_id: Optional[str] = Field(default=None, description="Optional audio session identifier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Device or audio metadata")


@router.get("/status", tags=["Omi Voice"])
async def omi_status(
    listener: OmiVoiceListener = Depends(get_omi_listener),
) -> Dict[str, Any]:
    """Check Omi voice listener status."""
    return {
        "status": "active",
        "service": "OmiVoiceListener",
        "has_app_id": bool(listener.app_id),
    }


@router.post("/voice", tags=["Omi Voice"])
async def process_voice(
    request: VoiceInputRequest,
    listener: OmiVoiceListener = Depends(get_omi_listener),
) -> Dict[str, Any]:
    """Process incoming voice instruction and trigger the autonomous multi-agent mesh."""
    try:
        result = listener.process_voice_transcript(
            transcript=request.transcript,
            speaker=request.speaker,
            session_id=request.session_id,
            metadata=request.metadata,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process voice instruction: {str(e)}",
        )


@router.post("/webhook", tags=["Omi Voice"])
async def omi_webhook(
    payload: Dict[str, Any],
    listener: OmiVoiceListener = Depends(get_omi_listener),
) -> Dict[str, Any]:
    """Receive real-time webhooks from Omi hardware / cloud services."""
    try:
        result = listener.handle_webhook_event(payload)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error handling Omi webhook: {str(e)}",
        )

from fastapi import WebSocket, WebSocketDisconnect
import json

@router.websocket("/stream")
async def omi_websocket_stream(
    websocket: WebSocket,
    listener: OmiVoiceListener = Depends(get_omi_listener)
):
    """Real-time voice input stream via WebSockets for Omi hardware."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                transcript = payload.get("transcript", "")
                speaker = payload.get("speaker", "user")
                session_id = payload.get("session_id", "websocket_session")
                
                if transcript:
                    result = listener.process_voice_transcript(
                        transcript=transcript,
                        speaker=speaker,
                        session_id=session_id
                    )
                    await websocket.send_json(result)
                else:
                    await websocket.send_json({"status": "ignored", "message": "No transcript"})
            except json.JSONDecodeError:
                await websocket.send_json({"status": "error", "message": "Invalid JSON payload"})
    except WebSocketDisconnect:
        pass

