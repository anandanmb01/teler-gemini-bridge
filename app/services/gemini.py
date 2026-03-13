import asyncio
import base64
import contextlib
import json
import logging
import os
import queue as _queue
import threading
import uuid
import wave

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from google import genai
from google.genai import types
from scipy.signal import resample_poly

from app.config import settings

logger = logging.getLogger(__name__)

_AUDIO_REC_DIR = "/tmp/audio_rec"
_TRANSCRIPT_DIR = "/tmp/transcription"
os.makedirs(_AUDIO_REC_DIR, exist_ok=True)
os.makedirs(_TRANSCRIPT_DIR, exist_ok=True)

# FFmpeg read size: 20ms of audio per direction
_G2T_READ_SIZE = 320   # 8kHz  16-bit mono → 160 samples → 20ms
_T2G_READ_SIZE = 640   # 16kHz 16-bit mono → 320 samples → 20ms


def _run_transcript_writer(session_id: str, q: "_queue.Queue[str | None]") -> None:
    """Background thread: drains transcript queue and writes lines to <uuid>.txt."""
    path = os.path.join(_TRANSCRIPT_DIR, f"{session_id}.txt")
    try:
        os.makedirs(_TRANSCRIPT_DIR, exist_ok=True)
        with open(path, "w") as f:
            logger.info(f"Transcript file opened: {path}")
            while True:
                line = q.get()
                if line is None:
                    break
                f.write(line)
                f.flush()
        logger.info(f"Transcript saved: {path}")
    except Exception as e:
        logger.error(f"Transcript write failed: {e}")


def _save_recording(session_id: str, teler_pcm: bytes, gemini_pcm: bytes) -> None:
    """Mix both audio streams to mono 16kHz WAV and write to disk (background thread)."""
    def _worker():
        try:
            os.makedirs(_AUDIO_REC_DIR, exist_ok=True)
            logger.info(f"Saving recording session={session_id} teler={len(teler_pcm)}B gemini={len(gemini_pcm)}B")
            teler = np.frombuffer(teler_pcm, dtype=np.int16).astype(np.float32)
            gemini_raw = np.frombuffer(gemini_pcm, dtype=np.int16).astype(np.float32)
            gemini = resample_poly(gemini_raw, 2, 3)  # 24kHz → 16kHz
            n = max(len(teler), len(gemini))
            teler = np.pad(teler, (0, n - len(teler)))
            gemini = np.pad(gemini, (0, n - len(gemini)))
            mixed = np.clip((teler + gemini) / 2, -32768, 32767).astype(np.int16)
            path = os.path.join(_AUDIO_REC_DIR, f"{session_id}.wav")
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(mixed.tobytes())
            logger.info(f"Saved recording: {path}")
        except Exception as e:
            logger.error(f"Recording save failed: {e}")

    threading.Thread(target=_worker, daemon=True).start()


_HANGUP_TOOL = types.FunctionDeclaration(
    name="hangup_call",
    description=(
        "Terminate and disconnect the current phone call. "
        "You MUST call this tool to end the call — do NOT just say goodbye without calling it. "
        "Call this when: (1) the conversation purpose is complete, (2) the caller asks to hang up, "
        "end the call, disconnect, or says goodbye, (3) you have nothing more to assist with. "
        "After calling this tool the call will be disconnected immediately."
    ),
)


async def _start_ffmpeg(in_rate: int, out_rate: int) -> asyncio.subprocess.Process:
    """Start an FFmpeg process that converts s16le PCM from in_rate to out_rate via stdio."""
    return await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "quiet",
        "-f", "s16le", "-ar", str(in_rate), "-ac", "1", "-i", "pipe:0",
        "-f", "s16le", "-ar", str(out_rate), "-ac", "1", "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )


async def _kill(proc: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(Exception):
        proc.stdin.close()  # type: ignore[union-attr]
    with contextlib.suppress(Exception):
        proc.terminate()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(proc.wait(), timeout=2)


async def run_session(websocket: WebSocket, system_prompt: str, initial_prompt: str) -> None:
    """Run a Gemini Live session bridged to a Teler WebSocket via FFmpeg audio pipelines."""
    session_id = str(uuid.uuid4())
    teler_buf: bytearray = bytearray()
    gemini_buf: bytearray = bytearray()
    transcript_q: "_queue.Queue[str | None]" = _queue.Queue()
    transcript_thread = threading.Thread(
        target=_run_transcript_writer, args=(session_id, transcript_q), daemon=True
    )
    transcript_thread.start()
    logger.info(f"WebSocket connected. session={session_id}")

    g2t_proc = t2g_proc = None  # FFmpeg processes — kept for cleanup in finally

    try:
        if not settings.google_api_key:
            await websocket.close(code=1008, reason="GOOGLE_API_KEY not configured")
            return

        # Gemini 24kHz → 8kHz → Teler
        g2t_proc = await _start_ffmpeg(in_rate=24000, out_rate=8000)
        # Teler 16kHz → 16kHz → Gemini  (passthrough; ensures clean s16le format)
        t2g_proc = await _start_ffmpeg(in_rate=16000, out_rate=16000)
        logger.info("FFmpeg pipelines started")

        client = genai.Client(
            api_key=settings.google_api_key,
            http_options={"api_version": "v1beta"},
        )
        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=system_prompt,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
                )
            ),
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=104857,
                sliding_window=types.SlidingWindow(target_tokens=52428),
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            tools=[types.Tool(function_declarations=[_HANGUP_TOOL])],
        )

        async with client.aio.live.connect(model=settings.gemini_model, config=config) as session:
            logger.info("Connected to Gemini Live session")

            if initial_prompt:
                await session.send_client_content(turns={"parts": [{"text": initial_prompt}]})
                logger.info(f"Sent initial prompt: {initial_prompt[:50]}...")

            gemini_speaking = False
            chunk_id = 1

            # ── Teler → t2g FFmpeg stdin ─────────────────────────────────────
            async def _teler_to_t2g():
                try:
                    async for message in websocket.iter_text():
                        data = json.loads(message)
                        if data.get("type") == "audio":
                            try:
                                pcm = base64.b64decode(data["data"]["audio_b64"])
                                teler_buf.extend(pcm)
                                if not gemini_speaking:
                                    t2g_proc.stdin.write(pcm)  # type: ignore[union-attr]
                                    await t2g_proc.stdin.drain()  # type: ignore[union-attr]
                                else:
                                    logger.debug("Echo gate: dropping Teler audio while Gemini is speaking")
                            except Exception as e:
                                logger.error(f"Error writing to t2g FFmpeg: {e}")
                except WebSocketDisconnect:
                    logger.info("Teler WebSocket disconnected")
                except Exception as e:
                    logger.error(f"Error in teler→t2g: {e}")
                finally:
                    with contextlib.suppress(Exception):
                        t2g_proc.stdin.close()  # type: ignore[union-attr]

            # ── t2g FFmpeg stdout → Gemini ────────────────────────────────────
            async def _t2g_to_gemini():
                try:
                    while True:
                        pcm = await t2g_proc.stdout.read(_T2G_READ_SIZE)  # type: ignore[union-attr]
                        if not pcm:
                            break
                        await session.send_realtime_input(
                            audio=types.Blob(data=pcm, mime_type="audio/pcm")
                        )
                        logger.debug(f"Sent {len(pcm)}B to Gemini")
                except Exception as e:
                    logger.error(f"Error in t2g→gemini: {e}")

            # ── Gemini → g2t FFmpeg stdin ─────────────────────────────────────
            async def _gemini_to_g2t():
                nonlocal gemini_speaking
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
                                    gemini_speaking = True
                                    gemini_buf.extend(response.data)
                                    g2t_proc.stdin.write(response.data)  # type: ignore[union-attr]
                                    await g2t_proc.stdin.drain()  # type: ignore[union-attr]
                                    logger.debug(f"Wrote {len(response.data)}B to g2t FFmpeg")

                                if response.server_content:
                                    sc = response.server_content
                                    if sc.input_transcription and sc.input_transcription.text:
                                        transcript_q.put(f"[Caller]: {sc.input_transcription.text}\n")
                                    if sc.output_transcription and sc.output_transcription.text:
                                        transcript_q.put(f"[Agent]: {sc.output_transcription.text}\n")
                                    if getattr(sc, "turn_complete", False):
                                        gemini_speaking = False
                                        logger.debug("Turn complete — echo gate open")

                        except Exception as e:
                            logger.debug(f"Session iteration ended: {e}")
                            await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error in gemini→g2t: {e}")
                finally:
                    with contextlib.suppress(Exception):
                        g2t_proc.stdin.close()  # type: ignore[union-attr]

            # ── g2t FFmpeg stdout → Teler ─────────────────────────────────────
            async def _g2t_to_teler():
                nonlocal chunk_id
                try:
                    while True:
                        pcm = await g2t_proc.stdout.read(_G2T_READ_SIZE)  # type: ignore[union-attr]
                        if not pcm:
                            break
                        await websocket.send_json({
                            "type": "audio",
                            "audio_b64": base64.b64encode(pcm).decode(),
                            "chunk_id": chunk_id,
                        })
                        chunk_id += 1
                        logger.debug(f"Sent {len(pcm)}B to Teler (chunk {chunk_id})")
                except Exception as e:
                    logger.error(f"Error in g2t→teler: {e}")

            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(_teler_to_t2g()),
                    asyncio.create_task(_t2g_to_gemini()),
                    asyncio.create_task(_gemini_to_g2t()),
                    asyncio.create_task(_g2t_to_teler()),
                ],
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
        if g2t_proc:
            await _kill(g2t_proc)
        if t2g_proc:
            await _kill(t2g_proc)
        transcript_q.put(None)
        _save_recording(session_id, bytes(teler_buf), bytes(gemini_buf))
        logger.info("Connection closed.")
