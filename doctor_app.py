import streamlit as st
from PIL import Image
from google import genai
from google.api_core import exceptions
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
import io
import os
import base64
from faster_whisper import WhisperModel

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="PocketDoc AI", page_icon="🩺", layout="wide")

# --- 2. LOCAL AI MODELS LOADING (STT) ---
@st.cache_resource
def load_stt_model():
    # 'tiny' is best for Streamlit Cloud CPUs
    return WhisperModel("tiny", device="cpu", compute_type="int8")

stt_model = load_stt_model()

# --- 3. CUSTOM CSS (LangChain / Futuristic Theme) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono&display=swap');

    .stApp {
        background-color: #050505 !important;
        color: #e2e8f0 !important;
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 80px 20px;
        background: radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.08) 0%, rgba(5, 5, 5, 1) 70%);
    }

    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 4rem !important;
        font-weight: 800;
        background: linear-gradient(to right, #ffffff, #64748b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Bento-Style Sections */
    .feature-section {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(12px);
        padding: 40px !important;
        border-radius: 24px !important;
        margin: 20px auto !important;
        max-width: 900px;
    }

    /* AI Response Card */
    .status-card {
        background-color: #020617 !important;
        color: #38bdf8 !important;
        font-family: 'JetBrains Mono', monospace;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #1e293b;
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.1);
        margin-top: 20px;
    }

    /* Buttons & Toggles */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #2563eb) !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 12px 40px !important;
        border: none !important;
        font-weight: 600 !important;
        width: 100%;
    }

    /* Hide redundant UI */
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 4. UTILITY FUNCTIONS ---
def autoplay_audio(text):
    """Generates speech and injects HTML for automatic playback."""
    tts = gTTS(text=text[:500], lang='en')
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    b64 = base64.b64encode(audio_fp.getvalue()).decode()
    md = f"""
        <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

# --- 5. API INITIALIZATION ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
except Exception:
    st.error("Missing GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

# --- 6. LANDING PAGE SECTIONS ---

# HERO
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">PocketDoc AI</h1>
        <p style="font-size:1.2rem; color:#94a3b8;">Futuristic Multimodal Triage for Modern Clinical Analysis.</p>
    </div>
    """, unsafe_allow_html=True)

# EXPLANATION SECTION
st.markdown('<div class="feature-section">', unsafe_allow_html=True)
st.write("### 🧠 The Intelligence Layer")
st.write("Our system utilizes a **Chained Multimodal Inference** process:")
col1, col2, col3 = st.columns(3)
col1.metric("Vision", "Gemini 2.0")
col2.metric("Voice", "Faster-Whisper")
col3.metric("Logic", "Cross-Modal")
st.markdown('</div>', unsafe_allow_html=True)

# INTERACTIVE SECTION
st.markdown("<h2 style='text-align:center; margin-top:50px;'>🚀 Start Analysis</h2>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="feature-section">', unsafe_allow_html=True)
    
    # WEBCAM TOGGLE
    st.write("### 📷 01. Visual Input")
    cam_on = st.toggle("Activate Clinical Camera", value=False)
    img_file = None
    if cam_on:
        img_file = st.camera_input("Capture symptom morphology")
    else:
        st.info("Webcam is offline. Toggle above to enable visual scanning.")

    st.divider()

    # VOICE INPUT
    st.write("### 💬 02. Voice Narrative")
    audio = mic_recorder(start_prompt="⏺️ Record Symptoms", stop_prompt="⏹️ Process Voice", key='recorder')
    
    # Process voice locally
    if audio:
        with st.spinner("🤖 Local STT: Transcribing narrative..."):
            with open("temp.wav", "wb") as f:
                f.write(audio['bytes'])
            segments, _ = stt_model.transcribe("temp.wav")
            st.session_state.symptoms_text = " ".join([s.text for s in segments])
            os.remove("temp.wav")

    # Manual edit/view of transcription
    symptoms_input = st.text_area(
        "Narrative Transcript:",
        value=st.session_state.get('symptoms_text', ""),
        placeholder="Waiting for voice input or manual typing...",
        height=100
    )

    st.divider()

    # TRIGGER ANALYSIS
    if st.button("RUN MULTIMODAL INFERENCE"):
        if not img_file or not symptoms_input:
            st.warning("Critical Error: Both Visual and Narrative data streams are required.")
        else:
            with st.spinner("Executing Intelligence Chain..."):
                try:
                    image = Image.open(img_file)
                    prompt = f"""
                    SYSTEM: Professional Medical Triage Prototype. 
                    NARRATIVE: {symptoms_input}
                    TASK: Analyze image + narrative. Provide Triage Category and Clinical Steps.
                    """
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[prompt, image]
                    )

                    # Result Rendering
                    st.markdown(f'<div class="status-card">{response.text}</div>', unsafe_allow_html=True)
                    
                    # Voice Playback
                    autoplay_audio(response.text)

                except exceptions.ResourceExhausted:
                    st.error("API Limit reached. Please wait 60s.")
                except Exception as e:
                    st.error(f"Inference Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #475569; padding-bottom: 50px;">
        <p><strong>RESEARCH PROTOTYPE: NOT FOR MEDICAL DIAGNOSIS</strong></p>
        <p>© 2026 PocketDoc Labs // Multimodal AI Triage</p>
    </div>
    """, unsafe_allow_html=True)
