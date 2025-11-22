from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Iterable, List

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api import app  # noqa: E402
from audio.service import AudioService, _float32_to_pcm16  # noqa: E402
from utils import decode_audio  # noqa: E402


async def create_text_to_audio(
    text: str, *, delay_seconds: float = 1.5, chunk_samples: int = 3200
) -> bytes:
    """
    Utility that synthesizes text and runs it through the anonymizer pipeline.
    Returns PCM16 encoded audio suitable for websocket testing.
    """
    service = AudioService(delay_seconds=delay_seconds)
    await service.start()

    output_chunks: List[bytes] = []
    try:
        async for processed in service.stream_text(text, chunk_samples=chunk_samples):
            output_chunks.append(_float32_to_pcm16(processed))
    finally:
        await service.close()

    return b"".join(output_chunks)


def _chunk_bytes(payload: bytes, chunk_size: int) -> Iterable[bytes]:
    for start in range(0, len(payload), chunk_size):
        yield payload[start : start + chunk_size]


def test_audio_filter_websocket(audio_payload: bytes, chunk_size: int = 3200 * 2) -> List[bytes]:
    """
    Stream synthesized audio through the /ws/audio_filter websocket and collect responses.
    """
    responses: List[bytes] = []
    with TestClient(app) as client:
        with client.websocket_connect("/ws/audio_filter") as websocket:
            for chunk in _chunk_bytes(audio_payload, chunk_size):
                websocket.send_bytes(chunk)
                responses.append(websocket.receive_bytes())
    return responses


def test_audio_flag_websocket(iterations: int = 5) -> List[dict]:
    """
    Send placeholder payloads through the /ws/audio_flag websocket and collect JSON replies.
    """
    responses: List[dict] = []
    with TestClient(app) as client:
        with client.websocket_connect("/ws/audio_flag") as websocket:
            for _ in range(iterations):
                websocket.send_bytes(b"\x00")
                responses.append(websocket.receive_json())
    return responses


async def main() -> None:
    synthesized = await create_text_to_audio(
        "Esta es una prueba de anonimización para el cortafuegos de audio."
    )

    processed_chunks = await asyncio.to_thread(test_audio_filter_websocket, synthesized)
    flag_results = await asyncio.to_thread(test_audio_flag_websocket, 5)

    total_input_samples = len(synthesized) // 2
    total_output_samples = sum(len(chunk) for chunk in processed_chunks) // 2

    print(f"Input samples: {total_input_samples}")
    print(f"Returned chunks: {len(processed_chunks)}")
    print(f"Output samples: {total_output_samples}")

    if processed_chunks:
        decoded = decode_audio(processed_chunks[0])
        print(f"First chunk decoded sample count: {decoded.size}")

    print("Flag websocket responses:")
    for index, payload in enumerate(flag_results, start=1):
        print(f"  [{index}] {payload}")


if __name__ == "__main__":
    asyncio.run(main())
