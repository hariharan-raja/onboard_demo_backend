import asyncio
import io
import wave
import struct
import numpy as np
import json
import websockets
from dotenv import load_dotenv

load_dotenv()

TARGET_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2


def load_and_convert_wav(path: str) -> bytes:
    with wave.open(path, 'rb') as wf:
        src_channels = wf.getnchannels()
        src_rate = wf.getframerate()
        src_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    print(f"Source WAV: {src_channels}ch, {src_rate}Hz, {src_width*8}bit")

    # Decode to int16
    samples = np.frombuffer(frames, dtype=np.int16)

    # Mix down to mono if stereo
    if src_channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)

    # Resample to 16kHz if needed
    if src_rate != TARGET_RATE:
        num_target = int(len(samples) * TARGET_RATE / src_rate)
        samples = np.interp(
            np.linspace(0, len(samples) - 1, num_target),
            np.arange(len(samples)),
            samples
        ).astype(np.int16)
        print(f"Resampled {src_rate}Hz -> {TARGET_RATE}Hz ({len(samples)} samples)")

    return samples.tobytes()


async def run_test():
    pcm_bytes = load_and_convert_wav("test_tts.wav")

    # Pad with silence to exceed the server's 6-second SEGMENT_SIZE
    SEGMENT_SIZE = TARGET_RATE * 6 * SAMPLE_WIDTH  # 192000 bytes
    if len(pcm_bytes) < SEGMENT_SIZE:
        padding = b'\x00' * (SEGMENT_SIZE - len(pcm_bytes) + TARGET_RATE * SAMPLE_WIDTH)
        pcm_bytes = pcm_bytes + padding
        print(f"Padded to {len(pcm_bytes)} bytes to exceed segment threshold")

    print(f"PCM bytes to send: {len(pcm_bytes)}")

    uri = "ws://localhost:8000/ws/audio"
    print(f"Connecting to {uri} ...")

    async with websockets.connect(uri, open_timeout=10) as ws:
        print("Connected. Sending audio in chunks...")

        # Send in 6-second chunks (matching SEGMENT_SIZE on server)
        chunk_size = TARGET_RATE * 6 * SAMPLE_WIDTH
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i:i + chunk_size]
            await ws.send(chunk)
            print(f"  Sent chunk {i//chunk_size + 1}: {len(chunk)} bytes")

            try:
                response = await asyncio.wait_for(ws.recv(), timeout=60)
                result = json.loads(response)
                print("\n=== SERVER RESPONSE ===")
                print(json.dumps(result, indent=2))
                break  # got the result, done
            except asyncio.TimeoutError:
                print("  (no response within 60s)")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(run_test())
