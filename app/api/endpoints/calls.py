import asyncio
import base64
import json
import logging

from fastapi import (APIRouter, HTTPException, WebSocket, WebSocketDisconnect,
                     status)
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocketState
from pydantic import BaseModel

from app.core.config import settings
from app.utils.gemini_client import GeminiClient

logger = logging.getLogger(__name__)
router = APIRouter()

class CallFlowRequest(BaseModel):
    call_id: str
    account_id: str
    from_number: str
    to_number: str

class CallRequest(BaseModel):
    from_number: str
    to_number: str

@router.post("/flow", status_code=status.HTTP_200_OK, include_in_schema=False)
async def stream_flow(payload: CallFlowRequest):
    """
    Build and return Stream flow.
    """
    if not settings.google_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GOOGLE_API_KEY not configured"
        )

    stream_flow = {
        "action": "stream",
        "ws_url": f"wss://{settings.server_domain}/api/v1/calls/media-stream",
        "chunk_size": 500,
        "sample_rate": "16k",
        "record": True
    }

    return JSONResponse(stream_flow)

@router.post("/initiate-call", status_code=status.HTTP_200_OK)
async def initiate_call(call_request: CallRequest):
    """
    Initiate a call using Teler SDK.
    """
    try:
        from app.utils.teler_client import TelerClient
        
        if not settings.google_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GOOGLE_API_KEY not configured"
            )
        
        teler_client = TelerClient(api_key=settings.teler_api_key)
        call = await teler_client.create_call(
            from_number=call_request.from_number,
            to_number=call_request.to_number,
            flow_url=f"https://{settings.server_domain}/api/v1/calls/flow",
            status_callback_url=f"https://{settings.server_domain}/api/v1/webhooks/receiver",
            record=True,
        )
        logger.info(f"Call created: {call}")
        return JSONResponse(content={"success": True, "call": call})
    except Exception as e:
        logger.error(f"Failed to create call: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Failed to create call."
        )

@router.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle media streaming"""
    await websocket.accept()
    logger.info("WebSocket connected.")
    
    try:
        # Validate configuration
        if not settings.google_api_key:
            await websocket.close(code=1008, reason="GOOGLE_API_KEY not configured")
            return
        
        # Initialize Gemini client
        gemini_client = GeminiClient(
            api_key=settings.google_api_key,
            model=settings.gemini_model,
            system_message=settings.gemini_system_message
        )
        
        # Create Gemini session directly using the client
        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": settings.gemini_system_message,
        }
        
        async with gemini_client.client.aio.live.connect(model=settings.gemini_model, config=config) as session:
            logger.info("Successfully connected to Gemini Live session")
            
            # Simple list to buffer audio chunks from Gemini
            gemini_audio_chunks = []
            chunk_id = 1

            async def handle_media_stream():
                """Handle bidirectional audio streaming between Teler and Gemini"""
                nonlocal chunk_id, gemini_audio_chunks
                
                async def process_teler_to_gemini():
                    """Receive audio from Teler and send to Gemini"""
                    while True:
                        data = json.loads(await websocket.receive_text())
                        if data.get("type") == "audio":
                            audio_b64 = data["data"]["audio_b64"]
                            logger.debug("Received audio chunk from Teler")

                            try:
                                # Decode PCM audio from Teler (16kHz, 16-bit)
                                pcm_data = base64.b64decode(audio_b64)
                                
                                # Send to Gemini (no resampling needed since both are 16kHz)
                                await gemini_client.send_audio_to_gemini(session, pcm_data)

                            except Exception as e:
                                logger.error(f"Audio processing error: {e}")

                async def process_gemini_to_teler():
                    """Receive audio from Gemini and send to Teler"""
                    nonlocal chunk_id, gemini_audio_chunks
                    
                    async for audio_data in gemini_client.receive_audio_from_gemini(session):
                        # Add chunk to buffer
                        gemini_audio_chunks.append(audio_data)
                        
                        # Check if we have enough chunks to send
                        if len(gemini_audio_chunks) >= settings.gemini_audio_chunk_count:
                            try:
                                # Combine all buffered chunks
                                combined_audio = b"".join(gemini_audio_chunks)

                                # Downsample from 24kHz to 8kHz for Teler
                                downsampled_data = gemini_client.audio_resampler.downsample(combined_audio, 24000)

                                # Encode to base64 for Teler
                                downsampled_b64 = base64.b64encode(downsampled_data).decode('utf-8')

                                await websocket.send_json({
                                    "type": "audio",
                                    "audio_b64": downsampled_b64,
                                    "chunk_id": chunk_id
                                })
                                logger.debug(f"Sent downsampled PCM16 audio to Teler (chunk {chunk_id}, size: {len(downsampled_data)} bytes)")
                                
                                # Reset buffer and increment chunk ID
                                gemini_audio_chunks = []
                                chunk_id += 1
                                
                            except Exception as e:
                                logger.error(f"Error processing chunks: {e}")

                # Start both tasks concurrently
                teler_to_gemini_task = asyncio.create_task(process_teler_to_gemini())
                gemini_to_teler_task = asyncio.create_task(process_gemini_to_teler())

                await asyncio.gather(teler_to_gemini_task, gemini_to_teler_task)

            await handle_media_stream()
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"Error in media stream: {e}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()
    finally:
        logger.info("Connection closed.")
