
import streamlit as st
from TTS.api import TTS
import os
import uuid

# --- Page Configuration ---
st.set_page_config(
    page_title="Text-to-Speech Synthesizer",
    page_icon="️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Application Title and Description ---
st.title("Text-to-Speech Synthesizer")
st.markdown(
    """
    This application converts your text into high-quality speech using a selection of cutting-edge TTS models. 
    Simply enter your text, choose a model, and listen to the generated audio.
    """
)

# --- Model Selection ---
st.header("Choose a Model")
model_name = st.selectbox(
    "Select a TTS model:",
    ("Tacotron2", "FastSpeech", "VITS", "TransformerTTS"),
    help="Each model offers a unique voice and style. Experiment to find your favorite!",
)

# --- Text Input ---
st.header("Enter Your Text")
text = st.text_area(
    "Enter the text you want to convert:",
    "Hello, this is a test of the text-to-speech system.",
    height=150,
    help="The longer the text, the longer the audio generation will take.",
)

# --- Audio Generation ---
if st.button("Generate Audio"):
    if not text:
        st.error("Please enter some text to generate audio.")
    else:
        with st.spinner(f"Generating audio with {model_name}... This may take a moment."):
            try:
                # --- Model Loading (with caching) ---
                @st.cache_resource
                def get_tts_model(model_name):
                    model_paths = {
                        "Tacotron2": "tts_models/en/ljspeech/tacotron2-DDC",
                        "FastSpeech": "tts_models/en/ljspeech/fastspeech",
                        "VITS": "tts_models/en/ljspeech/vits",
                        "TransformerTTS": "tts_models/en/ljspeech/tacotron2-DDC_ph",
                    }
                    return TTS(model_name=model_paths[model_name], progress_bar=False, gpu=False)

                tts = get_tts_model(model_name)

                # --- Audio File Generation ---
                AUDIO_DIR = "audio"
                os.makedirs(AUDIO_DIR, exist_ok=True)
                audio_filename = f"{uuid.uuid4()}.wav"
                audio_filepath = os.path.join(AUDIO_DIR, audio_filename)

                tts.tts_to_file(text=text, file_path=audio_filepath)

                # --- Display Audio ---
                st.success("Audio generated successfully!")
                st.audio(audio_filepath, format="audio/wav")

            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Sidebar Information ---
st.sidebar.header("About")
st.sidebar.info(
    """
    This application is powered by the Coqui TTS library and Streamlit.
    """
)
st.sidebar.header("Models")
st.sidebar.markdown(
    """
    - **Tacotron2:** A popular and robust model.
    - **FastSpeech:** A faster alternative to Tacotron2.
    - **VITS:** A high-quality, end-to-end model.
    - **TransformerTTS:** A model based on the Transformer architecture.
    """
)
