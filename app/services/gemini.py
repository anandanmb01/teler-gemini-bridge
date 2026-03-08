import asyncio
import base64
import contextlib
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from google import genai
from google.genai import types

from app.config import settings
from app.utils import AudioResampler

logger = logging.getLogger(__name__)
_audio_resampler = AudioResampler()

_HANGUP_TOOL = {
    "name": "hangup_call",
    "description": (
        "Immediately terminate and end the current phone call conversation. "
        "Use this when you are done with the purpose of the call or user explicitly "
        "requests to hang up, end the call, or disconnect."
    )
}


async def run_session(websocket: WebSocket, system_prompt: str, initial_prompt: str) -> None:
    """Run a Gemini Live session bridged to a Teler WebSocket."""
    logger.info("WebSocket connected.")
    try:
        if not settings.google_api_key:
            await websocket.close(code=1008, reason="GOOGLE_API_KEY not configured")
            return

        client = genai.Client(api_key=settings.google_api_key)
        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": system_prompt,
            "speech_config": {
                "voice_config": {"prebuilt_voice_config": {"voice_name": "Aoede"}}
            },
            "tools": [{"function_declarations": [_HANGUP_TOOL]}],
        }

        async with client.aio.live.connect(model=settings.gemini_model, config=config) as session:  # type: ignore
            logger.info("Connected to Gemini Live session")

            if initial_prompt:
                await session.send_client_content(turns={"parts": [{"text": initial_prompt}]})
                logger.info(f"Sent initial prompt: {initial_prompt[:50]}...")

            audio_chunks: list[bytes] = []
            chunk_id = 1

            async def _teler_to_gemini():
                try:
                    async for message in websocket.iter_text():
                        data = json.loads(message)
                        if data.get("type") == "audio":
                            try:
                                pcm = base64.b64decode(data["data"]["audio_b64"])
                                await session.send_realtime_input(
                                    audio=types.Blob(data=pcm, mime_type="audio/pcm;rate=16000")
                                )
                                logger.debug(f"Sent audio to Gemini ({len(pcm)} bytes)")
                            except Exception as e:
                                logger.error(f"Error sending audio to Gemini: {e}")
                except WebSocketDisconnect:
                    logger.info("Teler WebSocket disconnected")
                except Exception as e:
                    logger.error(f"Error in teler→gemini stream: {e}")

            async def _gemini_to_teler():
                nonlocal chunk_id, audio_chunks
                try:
                    while True:
                        try:
                            async for response in session.receive():
                                if response.tool_call and response.tool_call.function_calls:
                                    responses = []
                                    hangup = False
                                    for fc in response.tool_call.function_calls:
                                        result = "Call ended successfully" if fc.name == "hangup_call" else "Unknown function"
                                        if fc.name == "hangup_call":
                                            hangup = True
                                            logger.info("Hangup requested via tool call")
                                        responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response={"result": result}))
                                    await session.send_tool_response(function_responses=responses)
                                    if hangup:
                                        await websocket.close(code=1000, reason="Call ended")
                                        return

                                if response.data is not None:
                                    audio_chunks.append(response.data)
                                    logger.debug(f"Received Gemini audio chunk ({len(response.data)} bytes)")

                                    if len(audio_chunks) >= settings.gemini_audio_chunk_count:
                                        try:
                                            combined = b"".join(audio_chunks)
                                            downsampled = _audio_resampler.downsample(combined, 24000)
                                            await websocket.send_json({
                                                "type": "audio",
                                                "audio_b64": base64.b64encode(downsampled).decode(),
                                                "chunk_id": chunk_id,
                                            })
                                            logger.debug(f"Sent audio to Teler (chunk {chunk_id})")
                                            audio_chunks = []
                                            chunk_id += 1
                                        except Exception as e:
                                            logger.error(f"Error processing audio chunks: {e}")

                                if response.server_content and (
                                    getattr(response.server_content, 'turn_complete', False) or
                                    getattr(response.server_content, 'generation_complete', False)
                                ):
                                    logger.debug("Turn complete — waiting for next")

                        except Exception as e:
                            logger.debug(f"Session iteration ended: {e}")
                            await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error in gemini→teler stream: {e}")

            done, pending = await asyncio.wait(
                [asyncio.create_task(_teler_to_gemini()), asyncio.create_task(_gemini_to_teler())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            logger.info("Bridge session closed")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"Error in media stream: {e}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()
    finally:
        logger.info("Connection closed.")
