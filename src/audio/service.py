import asyncio
import base64
import os
from collections import deque
from pathlib import Path
from typing import AsyncIterator, Deque, Dict, Iterable, Optional

import numpy as np
from dotenv import load_dotenv
from elevenlabs import (
    AudioFormat,
    CommitStrategy,
    ElevenLabs,
    RealtimeAudioOptions,
    VoiceSettings,
)
from elevenlabs.realtime.connection import RealtimeConnection, RealtimeEvents

from filter.service import FilterService

# Ensure environment variables are loaded when executed from any working directory
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam pre-made voice
MODEL_ID = "eleven_multilingual_v2"
PCM_SAMPLE_RATE = 16_000
MUTE_TAIL_SECONDS = 0.25

_client: Optional[ElevenLabs] = None

keyword_filter = FilterService(patterns=[r"\bclave\b"])


def _pcm16_bytes_to_float32(chunk: bytes) -> np.ndarray:
    if not chunk:
        return np.zeros(0, dtype=np.float32)
    pcm = np.frombuffer(chunk, dtype=np.int16)
    return pcm.astype(np.float32) / float(np.iinfo(np.int16).max)


def _float32_to_pcm16(samples: np.ndarray) -> bytes:
    if samples.size == 0:
        return b""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * np.iinfo(np.int16).max).astype(np.int16)
    return pcm.tobytes()


def _get_client() -> ElevenLabs:
    global _client
    if _client is None:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY is not set. Provide it via environment or .env file."
            )
        _client = ElevenLabs(api_key=api_key)
    return _client


def pcm_stream_for_text(text: str) -> Iterable[bytes]:
    """Yield raw PCM16 chunks for the supplied text (used for local testing)."""
    response = _get_client().text_to_speech.stream(
        voice_id=VOICE_ID,
        text=text,
        model_id=MODEL_ID,
        output_format="pcm_16000",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
            speed=1.0,
        ),
    )
    for chunk in response:
        if chunk:
            yield chunk


class AudioService:
    """
    Accepts realtime audio chunks, forwards them to the ElevenLabs transcription
    websocket, and returns delayed audio with sensitive sections muted.
    """

    def __init__(
        self,
        *,
        sample_rate: int = PCM_SAMPLE_RATE,
        delay_seconds: float = 1.5,
        mute_tail_seconds: float = MUTE_TAIL_SECONDS,
    ) -> None:
        if sample_rate != PCM_SAMPLE_RATE:
            raise ValueError(
                f"AudioService currently requires {PCM_SAMPLE_RATE} Hz audio, "
                f"received {sample_rate} Hz."
            )

        self.sample_rate = sample_rate
        self.delay_seconds = max(delay_seconds, 0.0)
        self.delay_samples = int(self.delay_seconds * self.sample_rate)
        self._mute_tail_samples = int(max(mute_tail_seconds, 0.0) * self.sample_rate)

        self._connection: Optional[RealtimeConnection] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = asyncio.Lock()

        self._delay_queue: Deque[np.ndarray] = deque()
        self._buffered_samples = 0
        self._mute_remaining_samples = 0
        self._closed = False

    async def start(self) -> None:
        if self._connection is not None:
            return

        self._loop = asyncio.get_running_loop()
        self._connection = await _get_client().speech_to_text.realtime.connect(
            RealtimeAudioOptions(
                model_id="scribe_v2_realtime",
                language_code="es",
                audio_format=AudioFormat.PCM_16000,
                sample_rate=self.sample_rate,
                commit_strategy=CommitStrategy.VAD,
                include_timestamps=False,
            )
        )

        self._connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, self._handle_partial)
        self._connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, self._handle_committed)
        self._connection.on(RealtimeEvents.ERROR, self._handle_error)

    async def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        if self._closed:
            raise RuntimeError("AudioService has been closed.")
        if self._connection is None:
            raise RuntimeError(
                "AudioService.start() must be awaited before processing audio."
            )

        samples = np.asarray(chunk, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("AudioService expects mono float32 audio chunks.")

        pcm_payload = _float32_to_pcm16(samples)
        if pcm_payload:
            payload = base64.b64encode(pcm_payload).decode("ascii")
            await self._connection.send({"audio_base_64": payload})

        async with self._lock:
            self._enqueue(samples)
            output = self._dequeue(len(samples))

        return output

    async def flush(self) -> None:
        """
        Request a final commit from the transcription engine so buffered
        transcripts are emitted. Delayed audio is released by streaming silence.
        """
        if self._connection is None:
            return
        try:
            await self._connection.commit()
        except Exception:
            # Commit is best-effort; transient errors shouldn't bring the pipeline down.
            pass

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._connection is not None:
            await self._connection.close()
            self._connection = None

        async with self._lock:
            self._delay_queue.clear()
            self._buffered_samples = 0
            self._mute_remaining_samples = 0

    async def stream_text(
        self, text: str, chunk_samples: int = 3200
    ) -> AsyncIterator[np.ndarray]:
        """
        Convenience helper that synthesizes text using TTS and streams it through
        the anonymizer. Intended for manual testing of the streaming pipeline.
        """
        if self._connection is None:
            await self.start()

        tail_samples = int(self.delay_seconds * self.sample_rate)
        pending_tail = tail_samples

        try:
            for pcm_chunk in pcm_stream_for_text(text):
                float_chunk = _pcm16_bytes_to_float32(pcm_chunk)
                processed = await self.process_chunk(float_chunk)
                yield processed

            # Stream silence to flush the delay line.
            if tail_samples <= 0:
                return

            while pending_tail > 0:
                take = min(chunk_samples, pending_tail)
                pending_tail -= take
                silent = np.zeros(take, dtype=np.float32)
                processed = await self.process_chunk(silent)
                yield processed
        finally:
            await self.flush()

    def _enqueue(self, samples: np.ndarray) -> None:
        copied = np.array(samples, dtype=np.float32, copy=True)
        self._delay_queue.append(copied)
        self._buffered_samples += copied.size

    def _dequeue(self, requested: int) -> np.ndarray:
        if requested <= 0:
            return np.zeros(0, dtype=np.float32)

        if self.delay_samples == 0:
            output, filled = self._drain(requested)
            return self._apply_mute(output, filled)

        if self._buffered_samples <= self.delay_samples:
            return np.zeros(requested, dtype=np.float32)

        available = self._buffered_samples - self.delay_samples
        to_release = min(requested, available)
        output = np.zeros(requested, dtype=np.float32)
        filled = 0

        while to_release > 0 and self._delay_queue:
            block = self._delay_queue[0]
            block_len = block.size
            if block_len <= to_release:
                output[filled : filled + block_len] = block
                filled += block_len
                to_release -= block_len
                self._delay_queue.popleft()
            else:
                output[filled : filled + to_release] = block[:to_release]
                self._delay_queue[0] = block[to_release:]
                filled += to_release
                to_release = 0

        self._buffered_samples -= filled
        return self._apply_mute(output, filled)

    def _drain(self, requested: int) -> tuple[np.ndarray, int]:
        output = np.zeros(requested, dtype=np.float32)
        filled = 0
        to_release = min(requested, self._buffered_samples)

        while to_release > 0 and self._delay_queue:
            block = self._delay_queue[0]
            block_len = block.size
            if block_len <= to_release:
                output[filled : filled + block_len] = block
                filled += block_len
                to_release -= block_len
                self._delay_queue.popleft()
            else:
                output[filled : filled + to_release] = block[:to_release]
                self._delay_queue[0] = block[to_release:]
                filled += to_release
                to_release = 0

        self._buffered_samples -= filled
        return output, filled

    def _apply_mute(self, output: np.ndarray, filled: int) -> np.ndarray:
        if filled <= 0 or self._mute_remaining_samples <= 0:
            return output

        mute = min(self._mute_remaining_samples, filled)
        if mute > 0:
            output[filled - mute : filled] = 0.0
            self._mute_remaining_samples -= mute
        return output

    async def _mute_buffer(self) -> None:
        async with self._lock:
            for block in self._delay_queue:
                block.fill(0.0)
            if self._mute_tail_samples > 0:
                self._mute_remaining_samples = max(
                    self._mute_remaining_samples, self._mute_tail_samples
                )

    def _schedule_mute(self) -> None:
        if self._loop is None or self._closed:
            return
        self._loop.call_soon(asyncio.create_task, self._mute_buffer())

    def _handle_partial(self, data: Dict) -> None:
        transcript = data.get("transcript") or data.get("text")
        if not transcript:
            return
        if keyword_filter.filter(transcript):
            self._schedule_mute()

    def _handle_committed(self, data: Dict) -> None:
        transcript = data.get("transcript") or data.get("text")
        if not transcript:
            return
        if keyword_filter.filter(transcript):
            self._schedule_mute()

    def _handle_error(self, data: Dict) -> None:
        message = data.get("error") or str(data)
        print(f"Realtime transcription error: {message}")


__all__ = ["AudioService", "pcm_stream_for_text"]
