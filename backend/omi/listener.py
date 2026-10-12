"""Omi real-time voice input listener.

Handles real-time audio streams, transcripts, and webhook payloads from Omi,
routes user voice instructions to the Orchestrator agent, and feeds contextual
voice insights into persistent Qdrant memory.
"""

import logging
from typing import Any, Dict, Optional

from backend.agents.orchestrator import OrchestratorAgent, get_orchestrator_agent
from backend.config import settings

logger = logging.getLogger(__name__)


class OmiVoiceListener:
    """Listener service for real-time voice input from Omi."""

    def __init__(
        self,
        orchestrator: Optional[OrchestratorAgent] = None,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
    ):
        self.orchestrator = orchestrator or get_orchestrator_agent()
        self.app_id = app_id or settings.OMI_APP_ID
        self._app_secret = app_secret or settings.OMI_APP_SECRET

    def process_voice_transcript(
        self,
        transcript: str,
        speaker: str = "user",
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process incoming voice transcript and trigger the autonomous multi-agent mesh."""
        cleaned_transcript = transcript.strip()
        if not cleaned_transcript:
            return {
                "status": "ignored",
                "message": "Empty voice transcript received.",
            }

        logger.info("Processing voice instruction from Omi: '%s' (speaker: %s)", cleaned_transcript, speaker)

        # Trigger multi-agent execution loop
        result = self.orchestrator.run_workflow(
            task=cleaned_transcript,
            speaker=speaker,
            source="omi_voice",
            auto_persist_memory=True,
        )

        return {
            "status": "success",
            "source": "omi_voice",
            "session_id": session_id,
            "transcript": cleaned_transcript,
            "orchestrator_result": result,
            "speech_response": result.get("final_response"),
        }

    def handle_webhook_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle standard Omi webhook events (e.g. conversation events or transcript chunks)."""
        event_type = payload.get("event") or payload.get("type", "transcript")
        data = payload.get("data") or payload

        transcript = (
            data.get("transcript")
            or data.get("text")
            or data.get("content")
            or data.get("message", "")
        )
        speaker = data.get("speaker") or data.get("user") or "user"
        session_id = data.get("session_id") or data.get("conversation_id")

        if not transcript:
            return {
                "status": "ignored",
                "event_type": event_type,
                "message": "No transcript text found in payload.",
            }

        return self.process_voice_transcript(
            transcript=transcript,
            speaker=speaker,
            session_id=session_id,
            metadata=payload,
        )


# Global singleton helper
_omi_listener: Optional[OmiVoiceListener] = None


def get_omi_listener() -> OmiVoiceListener:
    """Dependency helper returning active OmiVoiceListener instance."""
    global _omi_listener
    if _omi_listener is None:
        _omi_listener = OmiVoiceListener()
    return _omi_listener
