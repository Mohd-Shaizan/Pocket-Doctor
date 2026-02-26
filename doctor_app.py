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

# --- 3. REFINED FRONTEND UI & CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&family=JetBrains+Mono&display=swap');

    .stApp {
        background-color: #050505 !important;
        color: #e2e8f0 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Hero Styling */
    .hero-container {
        text-align: center;
        padding: 80px 20px 20px 20px;
        background: radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.12) 0%, rgba(5, 5, 5, 1) 80%);
    }

    .hero-title {
        font-size: 5.5rem !important;
        font-weight: 800;
        letter-spacing: -2px;
        background: linear-gradient(to right, #ffffff, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    .hero-tagline {
        font-size: 1.4rem;
        color: #94a3b8;
        margin-top: 10px;
        margin-bottom: 40px;
    }

    /* THE CORE INTERACTION CARD (Centered & Unified) */
    .interaction-card {
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(25px);
        padding: 50px !important;
        border-radius: 32px !important;
        margin: 40px auto !important; /* Centering */
        max-width: 800px; /* Desktop Friendly Width */
        box-shadow: 0 0 40px rgba(59, 130, 246, 0.05), 0 25px 50px -12px rgba(0, 0, 0, 0.8);
    }

    .card-header {
        font-size: 1.8rem !important;
        font-weight: 700;
        text-align: center;
        margin-bottom: 30px;
        color: #ffffff;
        letter-spacing: 0.5px;
    }

    /* Section Heading Styling */
    .section-header {
        font-size: 2.2rem !important;
        font-weight: 700;
        text-align: center;
        margin-top: 40px !important;
        color: #ffffff;
    }

    .about-box {
        text-align: center;
        max-width: 700px;
        margin: 0 auto 20px auto;
        padding: 20px;
        color: #cbd5e1;
        font-size: 1.1rem;
        line-height: 1.6;
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

    /* Full-width Centered Button */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #1d4ed8) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 20px !important;
        font-weight: 700 !important;
        width: 100%;
        border: none !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 20px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.01);
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
    }

    /* Clean Up streamlit elements inside the card */
    div[data-testid="stVerticalBlock"] > div { width: 100%; }
    footer, header { visibility: hidden; }
    hr { margin: 2.5rem 0 !important; opacity: 0.15; }
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
        PocketDoc AI synthesizes <b>visual markers, vocal narratives, and text data</b> 
        into a real-time clinical assessment. A research prototype demonstrating multimodal 
        AI in healthcare environments.
    </div>
    """, unsafe_allow_html=True)

# START ANALYSIS (The Single Centered Card)
st.markdown('<h2 class="section-header">Start Analysis</h2>', unsafe_allow_html=True)

# Main Container for Horizontal Centering
outer_col_1, outer_col_2, outer_col_3 = st.columns([1, 6, 1])

with outer_col_2:
    st.markdown('<div class="interaction-card">', unsafe_allow_html=True)
    st.markdown('<p class="card-header">Clinical Input Panel</p>', unsafe_allow_html=True)
    
    # 01. VISUAL CAPTURE
    st.markdown("#### 📸 Visual Capture")
    cam_on = st.toggle("Initialize Triage Camera", value=False)
    img_file = None
    if cam_on:
        img_file = st.camera_input("Scanner Active", label_visibility="collapsed")
    else:
        st.info("Visual sensor offline. Toggle above to begin scan.")

    st.divider()
    
    # 02. VOICE NARRATIVE
    st.markdown("#### 💬 Voice Narrative")
    audio = mic_recorder(start_prompt="⏺️ Record Narrative", stop_prompt="⏹️ Stop & Transcribe", key='recorder')
    
    if audio:
        with st.spinner("Decoding vocal stream..."):
            with open("temp.wav", "wb") as f: f.write(audio['bytes'])
            segments, _ = stt_model.transcribe("temp.wav")
            st.session_state.symptoms_text = " ".join([s.text for s in segments])
            os.remove("temp.wav")

    st.markdown("<br>", unsafe_allow_html=True)

    # 03. TEXT INPUT
    symptoms_input = st.text_area(
        "Narrative Transcript",
        value=st.session_state.get('symptoms_text', ""),
        placeholder="Transcript will appear here automatically, or enter symptoms manually...",
        height=120,
        label_visibility="visible"
    )

    st.divider()

    # 04. RUN BUTTON (Inside Card)
    if st.button("Run Multimodal Inference"):
        if not img_file or not symptoms_input:
            st.warning("Prerequisite Missing: Both visual capture and narrative are required.")
        else:
            with st.spinner("Executing Neural Inference..."):
                try:
                    image = Image.open(img_file)
                    prompt = f"Professional Triage. Narrative: {symptoms_input}."
                    response = client.models.generate_content(model="gemini-2.5-flash", contents=[prompt, image])
                    
                    st.markdown(f'<div class="status-card">{response.text}</div>', unsafe_allow_html=True)
                    autoplay_audio(response.text)
                except exceptions.ResourceExhausted:
                    st.error("Quota Exceeded.")
                except Exception as e:
                    st.error(f"Inference Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown("""
    <div style="text-align: center; color: #475569; padding: 60px 0;">
        <p style="font-weight:600; font-size: 0.9rem; letter-spacing: 1px;">RESEARCH PROTOTYPE // NON-DIAGNOSTIC</p>
        <p style="font-size:0.75rem; opacity: 0.5;">© 2026 Better Mind Labs</p>
    </div>
    """, unsafe_allow_html=True)
