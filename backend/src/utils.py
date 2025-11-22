from __future__ import annotations

import asyncio
import os
from typing import Optional

import numpy as np

from audio.service import AudioService

_INT16_MAX = np.iinfo(np.int16).max

_audio_service: Optional[AudioService] = None
_init_lock = asyncio.Lock()

try:
    _default_delay_seconds = float(os.getenv("AUDIO_DELAY_SECONDS", "1.5"))
except ValueError:
    _default_delay_seconds = 1.5


def decode_audio(data: bytes) -> np.ndarray:
    """Decode little-endian PCM16 audio into a float32 numpy array."""
    if not data:
        return np.zeros(0, dtype=np.float32)
    pcm = np.frombuffer(data, dtype=np.int16)
    return pcm.astype(np.float32) / float(_INT16_MAX)


def encode_audio(samples: np.ndarray) -> bytes:
    """Encode a float32 numpy array into little-endian PCM16 bytes."""
    if samples.size == 0:
        return b""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * _INT16_MAX).astype(np.int16)
    # Ensure little-endian byte order for PCM16
    return pcm.astype("<i2").tobytes()


async def process_audio_chunk(chunk: np.ndarray) -> np.ndarray:
    """Delay and filter the provided audio chunk."""
    service = await _get_audio_service()
    return await service.process_chunk(chunk)


async def shutdown_audio_service() -> None:
    global _audio_service
    if _audio_service is None:
        return
    await _audio_service.close()
    _audio_service = None


async def shutdown_anonymizer() -> None:
    await shutdown_audio_service()


async def _get_audio_service() -> AudioService:
    global _audio_service
    if _audio_service is not None:
        return _audio_service

    async with _init_lock:
        if _audio_service is None:
            service = AudioService(delay_seconds=_default_delay_seconds)
            await service.start()
            _audio_service = service

    return _audio_service


__all__ = [
    "decode_audio",
    "encode_audio",
    "process_audio_chunk",
    "shutdown_audio_service",
    "shutdown_anonymizer",
]
