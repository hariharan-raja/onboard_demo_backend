import asyncio
import io
import wave
import math
import struct
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

AZURE_WHISPER_KEY = os.getenv("AZURE_WHISPER_KEY")
TRANSCRIPTION_URL = "https://oceanai-openai-europe.openai.azure.com/openai/deployments/whisper/audio/transcriptions?api-version=2024-06-01"

RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2


def generate_test_wav() -> bytes:
    # Generate 3 seconds of a 440 Hz sine wave (audible tone)
    duration = 3
    num_samples = RATE * duration
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(RATE)
        samples = []
        for i in range(num_samples):
            sample = int(32767 * 0.5 * math.sin(2 * math.pi * 440 * i / RATE))
            samples.append(struct.pack('<h', sample))
        wf.writeframes(b''.join(samples))
    buf.seek(0)
    return buf.read()


async def test_transcription():
    print("Generating test WAV...")
    wav_bytes = generate_test_wav()
    print(f"WAV size: {len(wav_bytes)} bytes")

    buf = io.BytesIO(wav_bytes)
    files = {"file": ("test.wav", buf, "audio/wav")}
    headers = {"api-key": AZURE_WHISPER_KEY}
    data = {"response_format": "verbose_json"}

    print("Sending to Azure Whisper...")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            TRANSCRIPTION_URL,
            headers=headers,
            files=files,
            data=data
        )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("SUCCESS")
        print("Text:", result.get("text", "(empty — sine wave has no speech)"))
        print("Segments:", len(result.get("segments", [])))
    else:
        print("FAILED")
        print(response.text)


if __name__ == "__main__":
    asyncio.run(test_transcription())
