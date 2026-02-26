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
    return WhisperModel("tiny", device="cpu", compute_type="int8")

stt_model = load_stt_model()

# --- 3. REFINED FUTURISTIC CSS (FIXED SPACING) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&family=JetBrains+Mono&display=swap');

    /* Global Typography & Background */
    .stApp {
        background-color: #050505 !important;
        color: #e2e8f0 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Hero Styling - Reduced Padding */
    .hero-container {
        text-align: center;
        padding: 60px 20px 20px 20px;
        background: radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.1) 0%, rgba(5, 5, 5, 1) 80%);
    }

    .hero-title {
        font-size: 5rem !important;
        font-weight: 800;
        letter-spacing: -2px;
        background: linear-gradient(to right, #ffffff 40%, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    .hero-tagline {
        font-size: 1.3rem;
        color: #94a3b8;
        margin-top: 10px;
        margin-bottom: 40px;
    }

    /* Unified Section Heading */
    .section-header {
        font-size: 2.2rem !important;
        font-weight: 700;
        text-align: center;
        margin-bottom: 30px !important;
        color: #ffffff;
    }

    /* The Unified Interaction Cockpit Card */
    .interaction-card {
    background: linear-gradient(
        180deg,
        rgba(15, 23, 42, 0.65),
        rgba(2, 6, 23, 0.75)
    ) !important;

    border-radius: 28px !important;
    padding: 48px 44px !important;
    max-width: 880px;
    margin: 0 auto 80px auto !important;

    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);

    border: 1px solid rgba(148, 163, 184, 0.18);
    box-shadow:
        0 0 0 1px rgba(59,130,246,0.15),
        0 30px 80px rgba(0,0,0,0.75);

    position: relative;
}

    /* About Text - Tighter Integration */
    .about-box {
        text-align: center;
        max-width: 950px;
        margin: 0 auto 50px auto;
        padding: 20px;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.03);
    }

    .about-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #cbd5e1;
    }

    /* Result Card (Output) */
    .status-card {
        background-color: #020617 !important;
        color: #38bdf8 !important;
        font-family: 'JetBrains Mono', monospace;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #1e293b;
        margin-top: 25px;
    }

    /* Unified Button */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #1d4ed8) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 18px !important;
        font-weight: 700 !important;
        width: 100%;
        border: none !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Spacing Fixes */
    .block-container { padding-top: 2rem !important; }
    footer, header { visibility: hidden; }
    hr { margin: 2rem 0 !important; opacity: 0.1; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. UTILITY FUNCTIONS ---
def autoplay_audio(text):
    tts = gTTS(text=text[:500], lang='en')
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    b64 = base64.b64encode(audio_fp.getvalue()).decode()
    md = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    st.markdown(md, unsafe_allow_html=True)

# --- 5. API INITIALIZATION ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
except:
    st.error("Missing GEMINI_API_KEY")
    st.stop()

# --- 6. PAGE LAYOUT ---

# HERO SECTION
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">PocketDoc AI</h1>
        <p class="hero-tagline">Autonomous Multimodal Clinical Triage</p>
    </div>
    """, unsafe_allow_html=True)

# ABOUT SECTION (Condensed)
st.markdown("""
    <div class="about-box">
        <p class="about-text">
            PocketDoc AI is a research prototype that combines images, voice, and text to help analyze health-related information in real time. It demonstrates how multimodal AI can assist in early clinical triage by understanding what a person shows, says, and types. This project is built to explore how different input types can work together to support faster, more intuitive health assessments in a single interface.
        </p>
    </div>
    """, unsafe_allow_html=True)

# UNIFIED INTERACTION CARD
st.markdown('<h2 class="section-header">Start Analysis</h2>', unsafe_allow_html=True)

with st.container():

    # CARD TITLE
    st.markdown("###  Multimodal Input Panel")

    # --- VISUAL CAPTURE ---
    st.markdown("#### 📸 Visual Capture")
    cam_on = st.toggle("Enable Triage Camera Feed", value=False)

    img_file = None
    if cam_on:
        img_file = st.camera_input(
            "Camera Active",
            label_visibility="collapsed"
        )
        st.success("Webcam online. Capture when ready.")
    else:
        st.info("Webcam offline. Enable to capture visual symptoms.")

    st.divider()

    # --- VOICE INPUT ---
    st.markdown("#### 🎙️ Voice Narrative")

    audio = mic_recorder(
        start_prompt="⏺️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        key="recorder"
    )

    if audio:
        with st.spinner("Transcribing voice input..."):
            with open("temp.wav", "wb") as f:
                f.write(audio["bytes"])
            segments, _ = stt_model.transcribe("temp.wav")
            st.session_state.symptoms_text = " ".join(s.text for s in segments)
            os.remove("temp.wav")

    st.divider()

    # --- TEXT INPUT ---
    st.markdown("####  Clinical Narrative")

    symptoms_input = st.text_area(
        "Transcript / Additional Details",
        value=st.session_state.get("symptoms_text", ""),
        placeholder="Voice transcript appears here. You may edit or add details...",
        height=140,
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- RUN BUTTON ---
    run = st.button("Run Multimodal Inference")

    if run:
        if not img_file or not symptoms_input:
            st.warning("Data Incomplete: Image and Narrative are required.")
        else:
            with st.spinner("Running Neural Inference Chain..."):
                try:
                    image = Image.open(img_file)
                    prompt = f"Professional Triage. Narrative: {symptoms_input}."
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[prompt, image]
                    )

                    st.markdown(
                        f'<div class="status-card">{response.text}</div>',
                        unsafe_allow_html=True
                    )
                    autoplay_audio(response.text)

                except exceptions.ResourceExhausted:
                    st.error("API Limit Reached.")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown("""
    <div style="text-align: center; color: #475569; padding: 60px 0;">
        <p style="font-weight:600; font-size: 0.9rem;">RESEARCH PROTOTYPE // NON-DIAGNOSTIC</p>
        <p style="font-size:0.75rem; opacity: 0.5;">© 2026 PocketDoc Labs</p>
    </div>
    """, unsafe_allow_html=True)
