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

# Dictionary to hold model paths
# We are not loading them into memory on startup anymore.
model_paths = {
    "Tacotron2": "tts_models/en/ljspeech/tacotron2-DDC",
    "FastSpeech": "tts_models/en/ljspeech/fastspeech",
    "VITS": "tts_models/en/ljspeech/vits",
    "TransformerTTS": "tts_models/en/ljspeech/tacotron2-DDC_ph",
}

# Cache for loaded models to avoid reloading them on every request
loaded_models = {}

def get_tts_model(model_name: str):
    """Loads a TTS model into memory or returns the cached model."""
    if model_name not in loaded_models:
        print(f"Loading model: {model_name}...")
        # Load the model and cache it
        loaded_models[model_name] = TTS(model_name=model_paths[model_name], progress_bar=False, gpu=False)
        print(f"Model {model_name} loaded.")
    return loaded_models[model_name]

@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    text = request.text
    model_name = request.model_name

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if model_name not in model_paths:
        raise HTTPException(status_code=400, detail=f"Invalid model name: {model_name}. Available models: {', '.join(model_paths.keys())}")

    try:
        # Get the model (loads it if it's not already in memory)
        tts = get_tts_model(model_name)
        
        audio_filename = f"{uuid.uuid4()}.wav"
        audio_filepath = os.path.join(AUDIO_DIR, audio_filename)
        
        print(f"Generating audio for '{text[:30]}...' with {model_name}")
        tts.tts_to_file(text=text, file_path=audio_filepath)
        print(f"Audio file saved to {audio_filepath}")
        
        return {"audio_url": f"/{AUDIO_DIR}/{audio_filename}"}
    except Exception as e:
        # Log the full error for debugging
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"TTS conversion failed: {str(e)}")

# To run this: uvicorn main:app --reload --port 8000
