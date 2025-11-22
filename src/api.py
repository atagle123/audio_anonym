from fastapi import FastAPI, WebSocket
from utils import process_audio_chunk, encode_audio, decode_audio
import numpy as np

app = FastAPI(title="Realtime Audio Firewall")


@app.websocket("/ws/audio")
async def audio_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # Receive audio chunk as bytes
            chunk_bytes = await ws.receive_bytes()
            
            # Decode bytes to numpy float32 array
            audio_chunk = decode_audio(chunk_bytes)
            
            # Process audio (replace with ML)
            processed_chunk = process_audio_chunk(audio_chunk)
            
            # Encode back to bytes
            out_bytes = encode_audio(processed_chunk)
            
            # Send processed audio back
            await ws.send_bytes(out_bytes)
            
    except Exception as e:
        print(f"Connection closed: {e}")
