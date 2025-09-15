import asyncio
import base64
import contextlib
import json
import logging

from fastapi import (APIRouter, HTTPException, WebSocket, WebSocketDisconnect,
                     status)
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocketState
from google import genai
from google.genai import types
from pydantic import BaseModel

from app.core.config import settings
from app.utils.audio_resampler import AudioResampler

logger = logging.getLogger(__name__)
router = APIRouter()
audio_resampler = AudioResampler()

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

        gemini_client = genai.Client(api_key=settings.google_api_key)
        
        # Create Gemini session directly using the client
        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": settings.gemini_system_message,
        }
        
        async with gemini_client.aio.live.connect(model=settings.gemini_model, config=config) as session:
            logger.info("Successfully connected to Gemini Live session")
            
            # Audio buffering for Gemini output
            gemini_audio_chunks = []
            chunk_id = 1

            async def teler_stream():
                """Receive audio from Teler and send to Gemini"""
                try:
                    async for message in websocket.iter_text():
                        data = json.loads(message)
                        
                        if data.get("type") == "audio":
                            try:
                                # Decode and send PCM audio to Gemini (16kHz)
                                pcm_data = base64.b64decode(data["data"]["audio_b64"])
                                await session.send_realtime_input(
                                    audio=types.Blob(data=pcm_data, mime_type="audio/pcm;rate=16000")
                                )
                                logger.debug(f"Sent audio to Gemini ({len(pcm_data)} bytes)")
                            except Exception as e:
                                logger.error(f"Error sending audio to Gemini: {e}")
                                
                except WebSocketDisconnect:
                    logger.info("Teler WebSocket disconnected")
                except Exception as e:
                    logger.error(f"Error in teler stream: {e}")

            async def gemini_stream():
                """Receive audio from Gemini and send to Teler"""
                nonlocal chunk_id, gemini_audio_chunks
                
                try:
                    while True:  # Keep the stream alive indefinitely
                        try:
                            async for response in session.receive():
                                # Process audio data
                                if response.data is not None:
                                    gemini_audio_chunks.append(response.data)
                                    logger.debug(f"Received audio chunk ({len(response.data)} bytes)")
                                    
                                    # Send buffered audio when we have enough chunks
                                    if len(gemini_audio_chunks) >= settings.gemini_audio_chunk_count:
                                        try:
                                            # Combine, downsample, and send audio
                                            combined_audio = b"".join(gemini_audio_chunks)
                                            downsampled_data = audio_resampler.downsample(combined_audio, 24000)
                                            downsampled_b64 = base64.b64encode(downsampled_data).decode('utf-8')

                                            await websocket.send_json({
                                                "type": "audio",
                                                "audio_b64": downsampled_b64,
                                                "chunk_id": chunk_id
                                            })
                                            logger.debug(f"Sent audio to Teler (chunk {chunk_id})")
                                            
                                            # Reset buffer and increment chunk ID
                                            gemini_audio_chunks = []
                                            chunk_id += 1
                                            
                                        except Exception as e:
                                            logger.error(f"Error processing audio chunks: {e}")
                                
                                # Handle turn completion - continue waiting for next turn
                                if (response.server_content and 
                                    (getattr(response.server_content, 'turn_complete', False) or 
                                     getattr(response.server_content, 'generation_complete', False))):
                                    logger.debug("Turn/generation completed - waiting for next")
                                    
                        except Exception as session_error:
                            logger.debug(f"Session iteration ended: {session_error}")
                            await asyncio.sleep(0.1)  # Brief pause before continuing
                            continue
                            
                except Exception as e:
                    logger.error(f"Error in gemini stream: {e}")

            # Run both streams concurrently
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(teler_stream()),
                    asyncio.create_task(gemini_stream()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            logger.info("Bridge closed")
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"Error in media stream: {e}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()
    finally:
        logger.info("Connection closed.")
