import os
import io
import wave
import json
from datetime import datetime
from collections import deque
from itertools import islice
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from new_helper import *

# Constants
api_key = os.getenv("OPENAI_API_KEY")
TRANSCRIPTION_URL = "https://api.openai.com/v1/audio/transcriptions"
RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
SEGMENT_DURATION_SEC = 6
OVERLAP_DURATION_SEC = 2
SEGMENT_SIZE = RATE * SEGMENT_DURATION_SEC * SAMPLE_WIDTH
OVERLAP_SIZE = RATE * OVERLAP_DURATION_SEC * SAMPLE_WIDTH
MAX_BUFFER_SIZE = RATE * 30 * SAMPLE_WIDTH  # 30 seconds max buffer

client_histories = {}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)



@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    print("🎙️ Client connected")

    buffer = deque(maxlen=MAX_BUFFER_SIZE)
    client_id = id(websocket)
    client_histories[client_id] = []
    previous_segments = []

    try:
        while True:
            if websocket.client_state != WebSocketState.CONNECTED:
                print("🔌 WebSocket not connected")
                break

            try:
                chunk = await websocket.receive_bytes()
            except Exception as e:
                print(f"⚠️ Error receiving bytes: {e}")
                break

            buffer.extend(chunk)
            print(f"🧠 Buffer size: {len(buffer)} bytes")

            if len(buffer) >= SEGMENT_SIZE:
                segment = bytes(islice(buffer, SEGMENT_SIZE))
                for _ in range(SEGMENT_SIZE - OVERLAP_SIZE):
                    buffer.popleft()

                try:
                    transcription_result = await transcribe_audio_from_bytes(segment)
                except Exception as e:
                    print(f"❌ Transcription error: {e}")
                    if websocket.client_state == WebSocketState.CONNECTED:
                        try:
                            await websocket.send_text(json.dumps({"error": "Transcription failed ==> ${e}"}))
                        except:
                            pass
                    continue

                if transcription_result:
                    current_text = transcription_result.get("text", "")
                    current_segments = transcription_result.get("segments", [])

                    try:
                        if previous_segments:
                            current_text = merge_transcriptions_with_timestamps(previous_segments, current_segments)

                        print("📝 TRANSCRIPTION:", current_text)

                        client_histories[client_id].append(current_text)
                        full_history = " ".join(client_histories[client_id])

                        result_json = await process_transcript(full_history)
                        result_json = await convert_non_null_values_to_text(result_json)

                        if websocket.client_state == WebSocketState.CONNECTED:
                            try:
                                await websocket.send_text(json.dumps(result_json))
                            except Exception as e:
                                print(f"⚠️ Failed to send result: {e}")

                        previous_segments = current_segments
                    except Exception as e:
                        print(f"🚨 Error processing transcript: {e}")
                        if websocket.client_state == WebSocketState.CONNECTED:
                            try:
                                await websocket.send_text(json.dumps({"error": "Processing failed"}))
                            except:
                                pass
                else:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        try:
                            await websocket.send_text(json.dumps({"error": "Transcription failed {e}"}))
                        except:
                            pass

    except WebSocketDisconnect as e:
        print(f"❌ Client disconnected (code={e.code})")

    except Exception as e:
        print(f"🔥 WebSocket error: {e}")

    finally:
        client_histories.pop(client_id, None)
        print("🧹 Cleaned up client history")


# async def transcribe_audio_from_bytes(audio_bytes: bytes) -> dict:
#     try:
#         timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#         filename = f"temp_{timestamp}.wav"

#         with io.BytesIO() as buffer:
#             with wave.open(buffer, 'wb') as wf:
#                 wf.setnchannels(CHANNELS)
#                 wf.setsampwidth(SAMPLE_WIDTH)
#                 wf.setframerate(RATE)
#                 wf.writeframes(audio_bytes)
#             buffer.seek(0)

#             files = {"file": (filename, buffer, "audio/wav")}
#             headers = {"Authorization": f"Bearer {api_key}"}
#             data = {
#                 "model": "whisper-1",
#                 "response_format": "verbose_json"
#             }

#             async with httpx.AsyncClient(timeout=60) as client:
#                 response = await client.post(
#                     TRANSCRIPTION_URL,
#                     headers=headers,
#                     files=files,
#                     data=data
#                 )

#             if response.status_code == 200:
#                 whisper_json = response.json()
#                 segments = whisper_json.get("segments", [])
#                 full_text = " ".join([seg.get("text", "").strip() for seg in segments])
#                 return {
#                     "text": full_text,
#                     "segments": segments
#                 }
#             else:
#                 print("❌ Transcription API error:", response.text)
#                 return None

#     except Exception as e:
#         print("❗ Error during transcription:", e)
#         return None

async def transcribe_audio_from_bytes(audio_bytes: bytes) -> dict:
    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"temp_{timestamp}.wav"

        with io.BytesIO() as buffer:
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH)
                wf.setframerate(RATE)
                wf.writeframes(audio_bytes)

            buffer.seek(0)

            files = {
                "file": (filename, buffer, "audio/wav")
            }

            headers = {
                "Authorization": f"Bearer {api_key}"
            }

            data = {
                "model": "gpt-4o-mini-transcribe"
            }

            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data
                )

            if response.status_code == 200:
                result = response.json()

                return {
                    "text": result.get("text", ""),
                    "raw_response": result
                }

            else:
                print("❌ Transcription API error:", response.text)
                return None

    except Exception as e:
        print("❗ Error during transcription:", e)
        return None

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI WebSocket server on ws://localhost:8000/ws/audio")
    uvicorn.run(app, host="0.0.0.0", port=8000)
