import random

from fastapi import FastAPI, WebSocket

from utils import decode_audio, encode_audio, process_audio_chunk, shutdown_anonymizer

app = FastAPI(title="Realtime Audio Firewall")


@app.websocket("/ws/audio_filter")
async def audio_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # Receive audio chunk as bytes
            chunk_bytes = await ws.receive_bytes()

            # Decode bytes to numpy float32 array
            audio_chunk = decode_audio(chunk_bytes)

            # Process audio (replace with ML)
            processed_chunk = await process_audio_chunk(audio_chunk)

            # Encode back to bytes
            out_bytes = encode_audio(processed_chunk)

            # Send processed audio back
            await ws.send_bytes(out_bytes)

    except Exception as e:
        print(f"Connection closed: {e}")


@app.websocket("/ws/audio_flag")
async def audio_flag(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # Receive any streaming bytes from client
            chunk = await ws.receive_bytes()

            # Random 1/20 chance of returning false
            ok = random.randint(1, 20) != 1

            await ws.send_json({"ok": ok})

    except Exception:
        await ws.close()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await shutdown_anonymizer()
