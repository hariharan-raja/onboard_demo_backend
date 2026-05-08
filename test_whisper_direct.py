import asyncio
import io
import wave
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

AZURE_WHISPER_KEY = os.getenv("AZURE_WHISPER_KEY")
TRANSCRIPTION_URL = "https://oceanai-openai-europe.openai.azure.com/openai/deployments/whisper/audio/transcriptions?api-version=2024-06-01"


async def transcribe(wav_path):
    with open(wav_path, "rb") as f:
        wav_bytes = f.read()

    buf = io.BytesIO(wav_bytes)
    files = {"file": ("test.wav", buf, "audio/wav")}
    headers = {"api-key": AZURE_WHISPER_KEY}
    data = {"response_format": "verbose_json"}

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(TRANSCRIPTION_URL, headers=headers, files=files, data=data)

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("Transcribed text:", result.get("text", "(empty)"))
        print("Segments:", len(result.get("segments", [])))
    else:
        print("Error:", response.text)


if __name__ == "__main__":
    asyncio.run(transcribe("test_tts.wav"))
