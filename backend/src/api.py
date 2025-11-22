import json
import random
import sys
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Ensure src directory is in path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import encode_audio, process_audio_chunk, shutdown_anonymizer
import numpy as np

app = FastAPI(title="Realtime Audio Firewall")

# Target sample rate for ElevenLabs transcription
TARGET_SAMPLE_RATE = 16_000


def decode_audio_float32(data: bytes) -> np.ndarray:
    """Decode Float32 audio bytes into a float32 numpy array."""
    if not data:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(data, dtype=np.float32)


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio from source_rate to target_rate using linear interpolation."""
    if source_rate == target_rate or samples.size == 0:
        return samples
    
    # Calculate the number of output samples
    ratio = target_rate / source_rate
    output_length = int(len(samples) * ratio)
    
    if output_length == 0:
        return np.zeros(0, dtype=np.float32)
    
    # Create indices for interpolation
    source_indices = np.arange(len(samples), dtype=np.float32)
    target_indices = np.linspace(0, len(samples) - 1, output_length, dtype=np.float32)
    
    # Linear interpolation
    resampled = np.interp(target_indices, source_indices, samples)
    
    return resampled.astype(np.float32)


@app.websocket("/ws/audio_filter")
async def audio_stream(ws: WebSocket):
    await ws.accept()
    try:
        # Handle initial config message (JSON text)
        sample_rate = None
        
        while True:
            try:
                # Check if message is text (config) or bytes (audio)
                message = await ws.receive()
            except (WebSocketDisconnect, RuntimeError) as e:
                # RuntimeError can occur when trying to receive after disconnect
                error_msg = str(e)
                if "disconnect" in error_msg.lower() or "receive" in error_msg.lower():
                    print("Client disconnected")
                    break
                # Re-raise if it's a different RuntimeError
                raise
            
            # Check for disconnect in message
            if "type" in message and message["type"] == "websocket.disconnect":
                print("Client disconnected")
                break
            
            if "text" in message:
                # Handle JSON config message
                try:
                    config = json.loads(message["text"])
                    if config.get("type") == "config":
                        sample_rate = config.get("sampleRate")
                        print(f"Received sample rate from client: {sample_rate} Hz")
                        # Don't send JSON response - frontend doesn't handle it
                        # Just continue to process audio chunks
                        continue
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing config: {e}")
                    continue
            
            elif "bytes" in message:
                # Handle audio chunk as bytes
                chunk_bytes = message["bytes"]

                # Decode bytes to numpy float32 array
                # Frontend sends Float32Array directly, so read as float32
                audio_chunk = decode_audio_float32(chunk_bytes)

                # Resample to target sample rate if needed
                if sample_rate and sample_rate != TARGET_SAMPLE_RATE:
                    audio_chunk = resample_audio(audio_chunk, sample_rate, TARGET_SAMPLE_RATE)

                # Process audio (replace with ML)
                processed_chunk = await process_audio_chunk(audio_chunk)

                # Resample output back to original sample rate if needed
                if sample_rate and sample_rate != TARGET_SAMPLE_RATE:
                    processed_chunk = resample_audio(processed_chunk, TARGET_SAMPLE_RATE, sample_rate)

                # Encode back to bytes
                out_bytes = encode_audio(processed_chunk)

                # Send processed audio back
                try:
                    await ws.send_bytes(out_bytes)
                except (WebSocketDisconnect, RuntimeError) as e:
                    error_msg = str(e)
                    if "disconnect" in error_msg.lower():
                        print("Client disconnected during send")
                        break
                    raise
            else:
                # Unknown message type
                continue

    except (WebSocketDisconnect, RuntimeError) as e:
        error_msg = str(e)
        if "disconnect" in error_msg.lower() or "receive" in error_msg.lower():
            print("Client disconnected")
        else:
            print(f"Connection closed: {e}")
    except Exception as e:
        print(f"Connection closed: {e}")


@app.websocket("/ws/audio_flag")
async def audio_flag(ws: WebSocket): # TODO: Remove or use
    print("Running audio_flag websocket")
    await ws.accept()
    try:
        while True:
            # Receive any streaming bytes from client
            _chunk = await ws.receive_bytes()

            # Random 1/20 chance of returning false
            ok = random.randint(1, 20) != 1

            await ws.send_json({"ok": ok})

    except WebSocketDisconnect:
        print("Client disconnected from audio_flag")
    except Exception as e:
        print(f"Connection closed in audio_flag: {e}")
        await ws.close()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await shutdown_anonymizer()
