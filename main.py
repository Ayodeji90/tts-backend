from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from TTS.api import TTS
import os
import uuid

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000", # For local development
    # Add your deployed frontend URL here after deployment, e.g., "https://your-frontend-domain.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a Pydantic model for the request body
class TTSRequest(BaseModel):
    text: str
    model_name: str

# Directory to save generated audio files
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Initialize TTS models (this will download models if not present)
# You might want to load these lazily or in a more robust way for production
tts_models = {
    "Tacotron2": TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False),
    "FastSpeech": TTS(model_name="tts_models/en/ljspeech/fastspeech", progress_bar=False, gpu=False),
    "VITS": TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=False),
    "TransformerTTS": TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC_ph", progress_bar=False, gpu=False), # Using a similar model as TransformerTTS is not directly available as a simple model name
}

@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    text = request.text
    model_name = request.model_name

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if model_name not in tts_models:
        raise HTTPException(status_code=400, detail=f"Invalid model name: {model_name}. Available models: {', '.join(tts_models.keys())}")

    try:
        tts = tts_models[model_name]
        audio_filename = f"{uuid.uuid4()}.wav"
        audio_filepath = os.path.join(AUDIO_DIR, audio_filename)
        tts.tts_to_file(text=text, file_path=audio_filepath)
        return {"audio_url": f"/{AUDIO_DIR}/{audio_filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS conversion failed: {str(e)}")

# To run this: uvicorn main:app --reload --port 8000
