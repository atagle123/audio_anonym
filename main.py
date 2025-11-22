from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import wave

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import encode_audio, _default_delay_seconds  # type: ignore  # noqa: E402
from audio.service import AudioService  # type: ignore  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream an input WAV file through the realtime anonymizer pipeline and "
            "write the delayed, redacted audio to a new WAV file."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a mono PCM16 WAV file sampled at 16 kHz.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to write the processed mono PCM16 WAV file.",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=0.2,
        help="Chunk size (in seconds) forwarded to the realtime service. Default: 0.2",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=None,
        help=(
            "Playback delay to apply before releasing audio. "
            "Defaults to AUDIO_DELAY_SECONDS environment variable or 1.5 seconds."
        ),
    )
    parser.add_argument(
        "--tail-seconds",
        type=float,
        default=None,
        help=(
            "Additional silence streamed after the file to flush the delay buffer. "
            "Defaults to the effective delay."
        ),
    )
    return parser.parse_args(argv)


async def run_pipeline(
    input_path: Path,
    output_path: Path,
    *,
    chunk_duration: float,
    delay_seconds: float | None,
    tail_seconds: float | None,
) -> None:
    sample_rate, samples = read_wav_mono_pcm16(input_path)

    if chunk_duration <= 0:
        raise ValueError("--chunk-duration must be positive.")

    # Resolve delay configuration with the same default logic as the FastAPI server.
    if delay_seconds is None:
        delay_seconds = float(_default_delay_seconds)

    audio_service = AudioService(sample_rate=sample_rate, delay_seconds=delay_seconds)
    await audio_service.start()

    processed_blocks: List[np.ndarray] = []
    chunk_size = max(int(chunk_duration * sample_rate), 1)

    try:
        for block in iter_blocks(samples, chunk_size):
            processed = await audio_service.process_chunk(block)
            processed_blocks.append(processed)

        effective_tail = tail_seconds if tail_seconds is not None else delay_seconds
        tail_samples = int(max(effective_tail, 0) * sample_rate)
        if tail_samples:
            padding = np.zeros(tail_samples, dtype=np.float32)
            for block in iter_blocks(padding, chunk_size):
                processed = await audio_service.process_chunk(block)
                processed_blocks.append(processed)

        await audio_service.flush()
    finally:
        await audio_service.close()

    if processed_blocks:
        processed_audio = np.concatenate(processed_blocks)
    else:
        processed_audio = np.zeros(0, dtype=np.float32)

    write_wav_mono_pcm16(output_path, processed_audio, sample_rate)


def read_wav_mono_pcm16(path: Path) -> Tuple[int, np.ndarray]:
    with contextlib.closing(wave.open(str(path), "rb")) as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError(f"{path} is not mono audio.")
        if wav_file.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit PCM audio.")

        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / np.iinfo(np.int16).max
    return sample_rate, samples


def iter_blocks(samples: np.ndarray, block_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(samples), block_size):
        yield samples[start : start + block_size].astype(np.float32, copy=False)


def write_wav_mono_pcm16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm_bytes = encode_audio(samples)

    with contextlib.closing(wave.open(str(path), "wb")) as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    asyncio.run(
        run_pipeline(
            args.input,
            args.output,
            chunk_duration=args.chunk_duration,
            delay_seconds=args.delay_seconds,
            tail_seconds=args.tail_seconds,
        )
    )


if __name__ == "__main__":
    main()
