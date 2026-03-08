import json
import logging

from fastapi import APIRouter, HTTPException, WebSocket, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.services import gemini
from app.services.teler import TelerClient

logger = logging.getLogger(__name__)
router = APIRouter()

current_system_prompt = settings.gemini_system_message
current_initial_prompt = settings.gemini_initial_prompt


class CallFlowRequest(BaseModel):
    call_id: str
    account_id: str
    from_number: str
    to_number: str


class CallRequest(BaseModel):
    from_number: str
    to_number: str
    system_prompt: str = ""
    initial_prompt: str = ""


@router.post("/flow", status_code=status.HTTP_200_OK, include_in_schema=False)
async def stream_flow(payload: CallFlowRequest):
    if not settings.google_api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")

    return JSONResponse({
        "action": "stream",
        "ws_url": f"wss://{settings.server_domain}/api/v1/calls/media-stream",
        "chunk_size": 500,
        "sample_rate": "16k",
        "record": True,
    })


@router.post("/initiate-call", status_code=status.HTTP_200_OK)
async def initiate_call(call_request: CallRequest):
    global current_system_prompt, current_initial_prompt

    try:
        if not settings.google_api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured")

        current_system_prompt = (
            settings.gemini_system_message
            if not call_request.system_prompt or call_request.system_prompt.lower() == "n/a"
            else call_request.system_prompt
        )
        current_initial_prompt = (
            ""
            if call_request.initial_prompt.lower() == "n/a"
            else call_request.initial_prompt or settings.gemini_initial_prompt
        )

        logger.info(f"Prompts — system: {current_system_prompt[:50]}..., initial: {current_initial_prompt[:50]}...")

        call = await TelerClient(api_key=settings.teler_api_key).create_call(
            from_number=call_request.from_number,
            to_number=call_request.to_number,
            flow_url=f"https://{settings.server_domain}/api/v1/calls/flow",
            status_callback_url=f"https://{settings.server_domain}/api/v1/webhooks/receiver",
            record=True,
        )
        logger.info(f"Call created: {call}")

        if hasattr(call, '__dict__'):
            call_data = {}
            for k, v in call.__dict__.items():
                try:
                    json.dumps(v)
                    call_data[k] = v
                except (TypeError, ValueError):
                    call_data[k] = str(v)
        elif isinstance(call, dict):
            call_data = call
        else:
            call_data = {"call_info": str(call)}

        return JSONResponse(content={"success": True, "call": call_data})

    except Exception as e:
        logger.error(f"Failed to create call: {e}")
        raise HTTPException(status_code=500, detail="Failed to create call.")


@router.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    await websocket.accept()
    await gemini.run_session(websocket, current_system_prompt, current_initial_prompt)
