## Audio Anonymizer Service

This project exposes a websocket that accepts realtime PCM16 audio, delays the
playback by a configurable buffer, and uses ElevenLabs realtime speech-to-text
to mute any segment containing the keyword `"clave"`. While the keyword is being
spoken, the delayed stream outputs silence so that sensitive data never leaves
the service.

### Prerequisites

- Python 3.12+
- An ElevenLabs API key with access to realtime transcription

Create a `.env` file next to `pyproject.toml` and set the API key:

```env
ELEVENLABS_API_KEY=your_key_here
# Optional: override the default 1.5 second playback delay
AUDIO_DELAY_SECONDS=2.0
```

Install dependencies with `uv` or `pip`:

```bash
uv sync
# or
pip install -e .
```

Alternatively, you can build and run the service with Docker (see below).

### Running the websocket API

Launch the FastAPI application:

```bash
uvicorn src.api:app --reload
```

Open a websocket client on `ws://localhost:8000/ws/audio` and stream
little-endian PCM16 frames (float32 conversion handled in-app). The server will:

1. Forward each chunk through the realtime transcription service.
2. Delay the outbound audio by roughly 1.5 seconds.
3. Drop/replace any buffered samples that align with the keyword `"clave"`.
4. Continue streaming once the word finishes.

### Docker

Ensure your `.env` file (containing `ELEVENLABS_API_KEY`) is present in the project root before building; it will be copied into the container image.

Build the container image from the project root:

```bash
docker build -t audio-anonym .
```

Run the websocket service, exposing port 8000. The ElevenLabs API key must be provided at runtime:

```bash
docker run --rm -p 8000:8000 \
  -e ELEVENLABS_API_KEY=your_key_here \
  audio-anonym
```

To process a WAV file through the CLI inside the container, mount the directory that contains your audio files:

```bash
docker run --rm \
  -e ELEVENLABS_API_KEY=your_key_here \
  -v "$(pwd)/data:/data" \
  audio-anonym \
  python main.py /data/input.wav /data/output.wav
```

### Development Notes

- Keyword handling uses timestamps when available; otherwise it falls back to a
  short window around the detection point.
- A single `AudioAnonymizer` instance is reused across websocket sessions and is
  closed automatically when the FastAPI app shuts down.
