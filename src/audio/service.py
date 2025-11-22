import asyncio
import base64
import os
import re
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from elevenlabs import (
    AudioFormat,
    CommitStrategy,
    ElevenLabs,
    RealtimeAudioOptions,
    VoiceSettings,
)
from elevenlabs.play import play
from elevenlabs.realtime.connection import RealtimeEvents

# Ensure environment variables are loaded when executed from any working directory
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "ELEVENLABS_API_KEY is not set. Please provide it in your environment or .env file."
    )


VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam pre-made voice
MODEL_ID = "eleven_multilingual_v2"
PCM_SAMPLE_RATE = 16_000
counter = 0

client = ElevenLabs(api_key=API_KEY)


def pcm_stream_for_text(text: str):
    """Yield 16 kHz PCM chunks for the provided text."""
    response = client.text_to_speech.stream(
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


async def stream_into_realtime(text: str) -> str:
    """Stream TTS PCM audio into the realtime transcription websocket and return the transcript."""
    connection = await client.speech_to_text.realtime.connect(
        RealtimeAudioOptions(
            model_id="scribe_v2_realtime",
            language_code="es",
            audio_format=AudioFormat.PCM_16000,
            sample_rate=PCM_SAMPLE_RATE,
            commit_strategy=CommitStrategy.MANUAL,
            include_timestamps=False,
        )
    )

    transcripts: List[str] = []
    transcript_ready = asyncio.Event()

    def handle_committed(data: Dict):
        transcript = data.get("transcript") or data.get("text")
        if transcript:
            filtered = filter_transcript(transcript)
            if filtered:
                transcripts.append(filtered)
        transcript_ready.set()

    def handle_error(data: Dict):
        message = data.get("error") or str(data)
        if not transcript_ready.is_set():
            transcript_ready.set()
        print(f"Realtime transcription error: {message}")

    connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, handle_committed)
    connection.on(RealtimeEvents.ERROR, handle_error)

    try:
        for chunk in pcm_stream_for_text(text):
            chunk_base64 = base64.b64encode(chunk).decode("ascii")
            await connection.send({"audio_base_64": chunk_base64})

        await connection.commit()
        await asyncio.wait_for(transcript_ready.wait(), timeout=15)
    finally:
        await connection.close()

    return " ".join(transcripts).strip()


def filter_transcript(text: str) -> str | None:
    """
    Anonymizes text after the keyword 'clave' is detected.

    Behavior:
    - On detecting 'clave', anonymize immediately and activate counter.
    - While counter > 0, anonymize everything.
    - Counter decreases each received transcript chunk.
    """

    global counter
    lowered = text.lower().strip()

    # --- 1. If counter is active -> anonymize everything ---
    if counter > 0:
        counter -= 1
        cleaned = re.sub(r"\w+", "zzz", text)
        return cleaned

    # --- 2. Detect trigger word ---
    if "clave" in lowered:
        counter = 10  # activate anonymization mode
        cleaned = re.sub(r"\w+", "zzz", text)
        return cleaned

    # --- 3. Normal case (no anonymization) ---
    return text


def synthesize_and_play(text: str) -> None:
    """Generate speech for the provided text and play it locally."""
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id="eleven_flash_v2_5",
        output_format="mp3_44100_128",
    )
    play(audio)


async def service():  # TODO en teoria entra un audio
    seed_text = "Te comparto mi clave es 1234."
    print(f"Seed text: {seed_text}")

    transcript = await stream_into_realtime(seed_text)
    if not transcript:
        raise RuntimeError("No transcript received from realtime transcription.")

    print(f"Realtime transcript: {transcript}")
    synthesize_and_play(transcript)


if __name__ == "__main__":
    asyncio.run(service())
