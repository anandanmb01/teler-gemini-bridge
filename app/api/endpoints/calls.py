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

# Global variables for dynamic prompts
current_system_prompt = settings.gemini_system_message
current_initial_prompt = ""

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
    global current_system_prompt, current_initial_prompt

    try:
        from app.utils.teler_client import TelerClient

        if not settings.google_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GOOGLE_API_KEY not configured"
            )

        # Handle system prompt updates
        if call_request.system_prompt:
            if call_request.system_prompt.lower() == "n/a":
                # Use default system prompt when "n/a" is specified
                current_system_prompt = settings.gemini_system_message
            else:
                # Update with provided system prompt
                current_system_prompt = call_request.system_prompt
        # If empty
        else:
            current_system_prompt = settings.gemini_system_message
        

        # Handle initial prompt updates
        if call_request.initial_prompt:
            if call_request.initial_prompt.lower() == "n/a":
                # Clear initial prompt when "n/a" is specified
                current_initial_prompt = ""
            else:
                # Update with provided initial prompt
                current_initial_prompt = call_request.initial_prompt
        else:
            current_system_prompt = "hello who is this"

        logger.info(f"Updated prompts - System: {current_system_prompt[:50]}..., Initial: {current_initial_prompt[:50]}...")
        
        teler_client = TelerClient(api_key=settings.teler_api_key)
        call = await teler_client.create_call(
            from_number=call_request.from_number,
            to_number=call_request.to_number,
            flow_url=f"https://{settings.server_domain}/api/v1/calls/flow",
            status_callback_url=f"https://{settings.server_domain}/api/v1/webhooks/receiver",
            record=True,
        )
        logger.info(f"Call created: {call}")

        # Serialize call object by parsing all possible keys
        call_data = {}
        if hasattr(call, '__dict__'):
            # Parse object attributes
            for key, value in call.__dict__.items():
                try:
                    # Test if value is JSON serializable
                    json.dumps(value)
                    call_data[key] = value
                except (TypeError, ValueError):
                    # Convert non-serializable values to string
                    call_data[key] = str(value)
        elif isinstance(call, dict):
            # Handle dictionary
            call_data = call
        else:
            # Fallback: convert to string
            call_data = {"call_info": str(call)}

        return JSONResponse(content={"success": True, "call": call_data})
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

        # Define hangup function tool
        hangup_call = {
            "name": "hangup_call",
            "description": "Immediately terminate and end the current phone call conversation. Use this when you are done with the purpose of the call or user explicitly requests to hang up, end the call, or disconnect. This will close the connection and stop all audio streaming."
        }
        tools = [{"function_declarations": [hangup_call]}]

        # Create Gemini session directly using the client
        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": current_system_prompt,
            "speech_config": {
                "voice_config": {"prebuilt_voice_config": {"voice_name": "Aoede"}}
            },
            "tools": tools
        }
        
        async with gemini_client.aio.live.connect(model=settings.gemini_model, config=config) as session: # type: ignore
            logger.info("Successfully connected to Gemini Live session")

            # Send initial prompt if available
            if current_initial_prompt:
                await session.send_client_content(turns={"parts": [{"text": current_initial_prompt}]})
                logger.info(f"Sent initial prompt: {current_initial_prompt[:50]}...")

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
                                # Handle tool calls (like hangup)
                                if response.tool_call and response.tool_call.function_calls:
                                    function_responses = []
                                    hangup_requested = False

                                    for fc in response.tool_call.function_calls:
                                        if fc.name == "hangup_call":
                                            function_response = types.FunctionResponse(
                                                id=fc.id,
                                                name=fc.name,
                                                response={"result": "Call ended successfully"}
                                            )
                                            logger.info("Call hangup executed via tool call - will terminate WebSocket")
                                            hangup_requested = True
                                        else:
                                            function_response = types.FunctionResponse(
                                                id=fc.id,
                                                name=fc.name,
                                                response={"result": "Unknown function"}
                                            )
                                        function_responses.append(function_response)

                                    # Send tool responses
                                    await session.send_tool_response(function_responses=function_responses)

                                    # If hangup was requested, close the connection
                                    if hangup_requested:
                                        await websocket.close(code=1000, reason="Call ended")
                                        return

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

            # initial user question
            await session.send_realtime_input(text="hello who is this")

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
