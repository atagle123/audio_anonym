import asyncio
import base64
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Deque, Dict, Iterable, List, Optional, Tuple

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


@dataclass
class BufferedChunk:
    samples: np.ndarray
    start: int
    flagged: bool = False


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

        self._delay_queue: Deque[BufferedChunk] = deque()
        self._buffered_samples = 0
        self._total_input_samples = 0
        self._pending_mute_ranges: List[Tuple[int, int]] = []
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
                include_timestamps=True,
                vad_silence_threshold_secs=0.8,  # Balanced threshold for better accuracy while maintaining responsiveness
            )
        )

        self._connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, self._handle_partial)
        self._connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, self._handle_committed)
        self._connection.on(
            RealtimeEvents.COMMITTED_TRANSCRIPT_WITH_TIMESTAMPS,
            self._handle_committed_with_timestamps,
        )
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
            self._total_input_samples = 0
            self._pending_mute_ranges.clear()

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
        block = BufferedChunk(samples=copied, start=self._total_input_samples)
        self._delay_queue.append(block)
        self._buffered_samples += block.samples.size
        self._total_input_samples += block.samples.size
        self._apply_pending_mutes()

    def _dequeue(self, requested: int) -> np.ndarray:
        if requested <= 0:
            return np.zeros(0, dtype=np.float32)

        if self.delay_samples == 0:
            output, _ = self._drain(requested)
            return output

        if self._buffered_samples <= self.delay_samples:
            return np.zeros(requested, dtype=np.float32)

        available = self._buffered_samples - self.delay_samples
        to_release = min(requested, available)
        output = np.zeros(requested, dtype=np.float32)
        filled = self._fill_from_queue(output, to_release)

        self._buffered_samples -= filled
        return output

    def _drain(self, requested: int) -> tuple[np.ndarray, int]:
        output = np.zeros(requested, dtype=np.float32)
        to_release = min(requested, self._buffered_samples)
        filled = self._fill_from_queue(output, to_release)
        self._buffered_samples -= filled
        return output, filled

    def _fill_from_queue(self, target: np.ndarray, to_release: int) -> int:
        filled = 0

        while to_release > 0 and self._delay_queue:
            block = self._delay_queue[0]
            block_len = block.samples.size
            if block_len == 0:
                self._delay_queue.popleft()
                continue

            if block_len <= to_release:
                if not block.flagged:
                    target[filled : filled + block_len] = block.samples
                filled += block_len
                to_release -= block_len
                self._delay_queue.popleft()
            else:
                if not block.flagged:
                    target[filled : filled + to_release] = block.samples[:to_release]
                block.samples = block.samples[to_release:]
                block.start += to_release
                filled += to_release
                to_release = 0

        return filled

    def _flag_range_samples(self, start_sample: int, end_sample: int) -> None:
        if start_sample >= end_sample:
            return

        earliest_buffered = self._total_input_samples - self._buffered_samples
        if end_sample <= earliest_buffered:
            return

        capped_end = min(end_sample, self._total_input_samples)
        if capped_end > start_sample:
            self._apply_flag_to_queue(start_sample, capped_end)

        if end_sample > self._total_input_samples:
            self._add_pending_range(
                max(start_sample, self._total_input_samples), end_sample
            )

    def _apply_flag_to_queue(self, start_sample: int, end_sample: int) -> None:
        if start_sample >= end_sample or not self._delay_queue:
            return

        earliest_buffered = self._total_input_samples - self._buffered_samples
        if end_sample <= earliest_buffered:
            return

        start_sample = max(start_sample, earliest_buffered)

        new_queue: Deque[BufferedChunk] = deque()
        new_total = 0

        for block in self._delay_queue:
            block_start = block.start
            block_end = block_start + block.samples.size

            if block.flagged or block_end <= start_sample or block_start >= end_sample:
                new_queue.append(block)
                new_total += block.samples.size
                continue

            before_len = max(0, min(block.samples.size, start_sample - block_start))
            overlap_start = block_start + before_len
            overlap_end = min(block_end, end_sample)
            overlap_len = max(0, overlap_end - overlap_start)
            after_len = block.samples.size - before_len - overlap_len

            if before_len > 0:
                before_samples = block.samples[:before_len].copy()
                new_queue.append(
                    BufferedChunk(samples=before_samples, start=block_start, flagged=False)
                )
                new_total += before_len

            if overlap_len > 0:
                overlap_samples = block.samples[
                    before_len : before_len + overlap_len
                ].copy()
                new_queue.append(
                    BufferedChunk(
                        samples=overlap_samples, start=overlap_start, flagged=True
                    )
                )
                new_total += overlap_len

            if after_len > 0:
                after_samples = block.samples[-after_len:].copy()
                new_queue.append(
                    BufferedChunk(
                        samples=after_samples, start=overlap_end, flagged=False
                    )
                )
                new_total += after_len

        self._delay_queue = new_queue
        self._buffered_samples = new_total

    def _add_pending_range(self, start: int, end: int) -> None:
        if start >= end:
            return
        self._pending_mute_ranges.append((start, end))
        self._pending_mute_ranges.sort(key=lambda rng: rng[0])

        merged: List[Tuple[int, int]] = []
        for current in self._pending_mute_ranges:
            if not merged or current[0] > merged[-1][1]:
                merged.append(list(current))
            else:
                merged[-1][1] = max(merged[-1][1], current[1])

        # Convert back to tuples to keep typing happy.
        self._pending_mute_ranges = [(start, end) for start, end in merged]

    def _apply_pending_mutes(self) -> None:
        if not self._pending_mute_ranges or not self._delay_queue:
            return

        available_end = self._total_input_samples
        earliest_buffered = available_end - self._buffered_samples

        remaining: List[Tuple[int, int]] = []
        for start, end in self._pending_mute_ranges:
            if end <= earliest_buffered:
                continue
            if end <= available_end:
                self._apply_flag_to_queue(start, end)
            else:
                if start < available_end:
                    self._apply_flag_to_queue(start, available_end)
                remaining.append((max(start, available_end), end))

        self._pending_mute_ranges = remaining

    def _schedule_flag(self, start_sample: int, end_sample: int) -> None:
        if self._loop is None or self._closed:
            return
        self._loop.call_soon(
            asyncio.create_task, self._flag_range_async(start_sample, end_sample)
        )

    async def _flag_range_async(self, start_sample: int, end_sample: int) -> None:
        async with self._lock:
            self._flag_range_samples(start_sample, end_sample)

    def _schedule_fallback_flag(self) -> None:
        if self._loop is None or self._closed:
            return
        self._loop.call_soon(asyncio.create_task, self._flag_recent_async())

    async def _flag_recent_async(self) -> None:
        window = self._mute_tail_samples
        if window <= 0:
            return
        async with self._lock:
            end = self._total_input_samples
            if end <= 0:
                return
            start = max(0, end - window)
            self._flag_range_samples(start, end)

    def _process_transcript_event(self, data: Dict[str, Any]) -> None:
        transcript = data.get("transcript") or data.get("text")
        if not transcript:
            return
        if not keyword_filter.filter(transcript):
            return

        print(f"[FILTERED] Filtered transcript: {transcript}")

        ranges = self._extract_flagged_ranges(data)
        if ranges:
            for start_sample, end_sample in ranges:
                self._schedule_flag(start_sample, end_sample)
        else:
            self._schedule_fallback_flag()

    def _extract_flagged_ranges(self, data: Dict[str, Any]) -> List[Tuple[int, int]]:
        ranges: List[Tuple[int, int]] = []

        for word_entry in self._iter_word_entries(data):
            word_text = self._extract_text(word_entry)
            if not word_text or not keyword_filter.filter(word_text):
                continue

            time_range = self._extract_time_range(word_entry)
            if time_range is None:
                continue

            start_sec, end_sec = time_range
            start_sample, end_sample = self._time_range_to_samples(start_sec, end_sec)
            ranges.append((start_sample, end_sample))

        if ranges:
            return ranges

        transcript_range = self._extract_time_range(data)
        if transcript_range is None:
            return []

        start_sec, end_sec = transcript_range
        start_sample, end_sample = self._time_range_to_samples(start_sec, end_sec)
        return [(start_sample, end_sample)]

    def _iter_word_entries(self, data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        candidates = []
        for key in ("words", "word_timestamps", "segments", "items", "tokens"):
            value = data.get(key)
            if isinstance(value, list):
                candidates.extend(value)

        timestamps = data.get("timestamps")
        if isinstance(timestamps, dict):
            for key in ("words", "segments", "items"):
                value = timestamps.get(key)
                if isinstance(value, list):
                    candidates.extend(value)

        for entry in candidates:
            if isinstance(entry, dict):
                yield entry

    @staticmethod
    def _extract_text(entry: Dict[str, Any]) -> Optional[str]:
        for key in ("word", "text", "token", "value"):
            value = entry.get(key)
            if isinstance(value, str):
                return value
        return None

    def _extract_time_range(self, obj: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        if not isinstance(obj, dict):
            return None

        lower = {key.lower(): value for key, value in obj.items()}

        start_ms = self._find_numeric(
            lower,
            [
                "audio_start_offset_ms",
                "start_ms",
                "start_millis",
                "offset_start_ms",
                "offset_ms_start",
            ],
        )
        end_ms = self._find_numeric(
            lower,
            [
                "audio_end_offset_ms",
                "end_ms",
                "end_millis",
                "offset_end_ms",
                "offset_ms_end",
            ],
        )
        if start_ms is not None and end_ms is not None:
            return float(start_ms) / 1000.0, float(end_ms) / 1000.0

        start_sec = self._find_numeric(
            lower,
            [
                "audio_start_offset_s",
                "audio_start_offset_sec",
                "audio_start_offset_secs",
                "start_second",
                "start_seconds",
                "start_time",
                "start_timestamp",
                "start_sec",
                "start_secs",
            ],
        )
        end_sec = self._find_numeric(
            lower,
            [
                "audio_end_offset_s",
                "audio_end_offset_sec",
                "audio_end_offset_secs",
                "end_second",
                "end_seconds",
                "end_time",
                "end_timestamp",
                "end_sec",
                "end_secs",
            ],
        )
        if start_sec is not None and end_sec is not None:
            return float(start_sec), float(end_sec)

        if "start" in lower and "end" in lower:
            start_val = lower["start"]
            end_val = lower["end"]
            if isinstance(start_val, (int, float)) and isinstance(end_val, (int, float)):
                start_val = float(start_val)
                end_val = float(end_val)
                if max(abs(start_val), abs(end_val)) > 1000:
                    return start_val / 1000.0, end_val / 1000.0
                return start_val, end_val

        for value in obj.values():
            if isinstance(value, dict):
                nested = self._extract_time_range(value)
                if nested is not None:
                    return nested

        return None

    @staticmethod
    def _find_numeric(
        mapping: Dict[str, Any], keys: Iterable[str]
    ) -> Optional[float]:
        for key in keys:
            if key in mapping and isinstance(mapping[key], (int, float)):
                return float(mapping[key])
        return None

    def _time_range_to_samples(self, start_seconds: float, end_seconds: float) -> Tuple[int, int]:
        start_sample = max(0, int(math.floor(start_seconds * self.sample_rate)))
        end_sample = int(math.ceil(end_seconds * self.sample_rate))
        if end_sample <= start_sample:
            end_sample = start_sample + 1
        return start_sample, end_sample

    def _handle_partial(self, data: Dict[str, Any]) -> None:
        transcript = data.get("transcript") or data.get("text")
        if transcript:
            print(f"[PARTIAL] Transcribed text: {transcript}")
        self._process_transcript_event(data)

    def _handle_committed(self, data: Dict[str, Any]) -> None:
        transcript = data.get("transcript") or data.get("text")
        if transcript:
            print(f"[COMMITTED] Transcribed text: {transcript}")
        self._process_transcript_event(data)

    def _handle_committed_with_timestamps(self, data: Dict[str, Any]) -> None:
        transcript = data.get("transcript") or data.get("text")
        if transcript:
            print(f"[COMMITTED WITH TIMESTAMPS] Transcribed text: {transcript}")
        self._process_transcript_event(data)

    def _handle_error(self, data: Dict) -> None:
        message = data.get("error") or str(data)
        print(f"Realtime transcription error: {message}")


__all__ = ["AudioService", "pcm_stream_for_text"]
