import asyncio
import base64
import contextlib
import json
import logging
import uuid
import wave
from pathlib import Path

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from google import genai
from google.genai import types
from scipy.signal import resample_poly

from app.config import settings
from app.utils.audio import AudioResampler

_RECORD_DIR = Path("/tmp/audiorecord")
_TRANSCRIPT_DIR = Path("/tmp/transcript")

logger = logging.getLogger(__name__)

_HANGUP_TOOL = {
    "name": "hangup_call",
    "description": (
        "Terminate and disconnect the current phone call. "
        "You MUST call this tool to end the call — do NOT just say goodbye without calling it. "
        "Call this when: (1) the conversation purpose is complete, (2) the caller asks to hang up, "
        "end the call, disconnect, or says goodbye, (3) you have nothing more to assist with. "
        "After calling this tool the call will be disconnected immediately."
    )
}


def _save_recording(caller_chunks: list[bytes], gemini_chunks: list[bytes], path: Path) -> None:
    """Mix caller (16kHz) and Gemini (24kHz) PCM into a mono 8kHz WAV file."""
    OUT_RATE = 8000
    caller_pcm = b"".join(caller_chunks)
    gemini_pcm = b"".join(gemini_chunks)

    caller = resample_poly(np.frombuffer(caller_pcm, dtype=np.int16).astype(np.float32), 1, 2) if caller_pcm else np.array([], dtype=np.float32)  # 16k -> 8k
    gemini = resample_poly(np.frombuffer(gemini_pcm, dtype=np.int16).astype(np.float32), 1, 3) if gemini_pcm else np.array([], dtype=np.float32)  # 24k -> 8k

    max_len = max(len(caller), len(gemini))
    caller = np.pad(caller, (0, max_len - len(caller)))
    gemini = np.pad(gemini, (0, max_len - len(gemini)))

    mixed = np.clip((caller + gemini) / 2, -32768, 32767).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(OUT_RATE)
        wf.writeframes(mixed.tobytes())
    logger.info(f"Recording saved: {path}")


def _save_transcript(lines: list[str], path: Path) -> None:
    """Write transcript lines to a text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Transcript saved: {path}")


async def run_session(websocket: WebSocket, system_prompt: str, initial_prompt: str) -> None:
    """Run a Gemini Live session bridged to a Teler WebSocket."""
    resampler = AudioResampler()
    session_id = str(uuid.uuid4())
    record_path = _RECORD_DIR / f"{session_id}.wav"
    transcript_path = _TRANSCRIPT_DIR / f"{session_id}.txt"
    caller_audio: list[bytes] = []
    gemini_audio: list[bytes] = []
    transcript_lines: list[str] = []
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
            "input_audio_transcription": {},
            "output_audio_transcription": {},
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
                                caller_audio.append(pcm)
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
                                    gemini_audio.append(response.data)
                                    logger.debug(f"Received Gemini audio chunk ({len(response.data)} bytes)")

                                    if len(audio_chunks) >= settings.gemini_audio_chunk_count:
                                        try:
                                            combined = b"".join(audio_chunks)
                                            downsampled = resampler.downsample(combined)
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

                                if response.server_content:
                                    sc = response.server_content
                                    if getattr(sc, 'input_transcription', None) and sc.input_transcription.text:
                                        line = f"[CALLER]: {sc.input_transcription.text.strip()}"
                                        transcript_lines.append(line)
                                        logger.debug(line)
                                    if getattr(sc, 'output_transcription', None) and sc.output_transcription.text:
                                        line = f"[GEMINI]: {sc.output_transcription.text.strip()}"
                                        transcript_lines.append(line)
                                        logger.debug(line)
                                    if getattr(sc, 'turn_complete', False) or getattr(sc, 'generation_complete', False):
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

            try:
                _save_recording(caller_audio, gemini_audio, record_path)
            except Exception as e:
                logger.error(f"Failed to save recording: {e}")
            try:
                _save_transcript(transcript_lines, transcript_path)
            except Exception as e:
                logger.error(f"Failed to save transcript: {e}")

            logger.info("Bridge session closed")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"Error in media stream: {e}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()
    finally:
        logger.info("Connection closed.")
