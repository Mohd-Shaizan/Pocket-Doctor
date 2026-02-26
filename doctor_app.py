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

# --- 3. REFINED FUTURISTIC CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&family=JetBrains+Mono&display=swap');

    /* Global Typography & Background */
    .stApp {
        background-color: #050505 !important;
        color: #e2e8f0 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Hero Styling */
    .hero-container {
        text-align: center;
        padding: 120px 20px 60px 20px;
        background: radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.12) 0%, rgba(5, 5, 5, 1) 80%);
    }

    .hero-title {
        font-size: 5.5rem !important;
        font-weight: 800;
        letter-spacing: -2px;
        background: linear-gradient(to right, #ffffff 30%, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    /* Section Headings */
    .section-header {
        font-size: 2.8rem !important;
        font-weight: 700;
        text-align: center;
        margin-top: 80px !important;
        margin-bottom: 40px !important;
        background: linear-gradient(to right, #ffffff, #64748b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Core Interaction Card (Glassmorphism) */
    .interaction-card {
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(20px);
        padding: 50px !important;
        border-radius: 32px !important;
        margin: 0 auto !important;
        max-width: 900px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
    }

    /* About Section Text */
    .about-text {
        font-size: 1.15rem;
        line-height: 1.8;
        color: #94a3b8;
        text-align: center;
        max-width: 700px;
        margin: 0 auto;
    }

    /* AI Response Card */
    .status-card {
        background-color: #020617 !important;
        color: #38bdf8 !important;
        font-family: 'JetBrains Mono', monospace;
        padding: 2.5rem;
        border-radius: 20px;
        border: 1px solid #1e293b;
        box-shadow: 0 0 40px rgba(56, 189, 248, 0.08);
        margin-top: 30px;
    }

    /* Buttons */
    .stButton>button {
        background: #3b82f6 !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 16px 40px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
        border: none !important;
        margin-top: 20px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(59, 130, 246, 0.3);
    }

    /* Clean Up Streamlit Defaults */
    div[data-testid="stMetric"] { background: rgba(255,255,255,0.03); padding: 20px; border-radius: 15px; }
    footer, header {visibility: hidden;}
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

# --- 6. LANDING PAGE EXPERIENCE ---

# HERO SECTION
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">PocketDoc AI</h1>
        <p style="font-size:1.4rem; color:#94a3b8; margin-top:15px;">Clinical Triage, Reimagined.</p>
    </div>
    """, unsafe_allow_html=True)

# ABOUT SECTION (Replacement for Intelligence Layer)
st.markdown('<h2 class="section-header">Next-Gen Multimodal Triage</h2>', unsafe_allow_html=True)
st.markdown("""
    <p class="about-text">
        PocketDoc AI is a high-performance research prototype designed for automated clinical triage. 
        By synthesizing <b>live visual data, voice narratives, and text inputs</b>, our multimodal engine 
        provides instantaneous risk assessment. Built for the future of healthcare—designed for 
        preliminary triage only.
    </p>
    """, unsafe_allow_html=True)

# START ANALYSIS SECTION (Consolidated Interaction Card)
st.markdown('<h2 class="section-header">System Interface</h2>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="interaction-card">', unsafe_allow_html=True)
    
    # 01. CAMERA INPUT
    st.subheader("📸 Visual Capture")
    cam_on = st.toggle("Initialize Clinical Camera Feed", value=False)
    img_file = None
    if cam_on:
        img_file = st.camera_input("Scanner Active: Position symptoms clearly")
    else:
        st.info("Sensor Offline: Toggle the switch above to initialize visual input.")

    st.divider()

    # 02. VOICE NARRATIVE
    st.subheader("💬 Patient Narrative")
    col_v1, col_v2 = st.columns([1, 1])
    with col_v1:
        st.write("Voice Recording")
        audio = mic_recorder(start_prompt="⏺️ Begin Recording", stop_prompt="⏹️ End & Process", key='recorder')
    
    if audio:
        with st.spinner("Decoding vocal streams..."):
            with open("temp.wav", "wb") as f: f.write(audio['bytes'])
            segments, _ = stt_model.transcribe("temp.wav")
            st.session_state.symptoms_text = " ".join([s.text for s in segments])
            os.remove("temp.wav")

    # 03. TEXT INPUT
    symptoms_input = st.text_area(
        "Vocal Transcript / Manual Entry",
        value=st.session_state.get('symptoms_text', ""),
        placeholder="Narrative will appear here, or type manually...",
        height=120
    )

    st.divider()

    # 04. EXECUTION
    if st.button("RUN MULTIMODAL INFERENCE"):
        if not img_file or not symptoms_input:
            st.warning("Incomplete Data: Please provide both visual capture and patient narrative.")
        else:
            with st.spinner("Processing Intelligence Chain..."):
                try:
                    image = Image.open(img_file)
                    prompt = f"SYSTEM: Professional Triage. NARRATIVE: {symptoms_input}. TASK: Analyze and provide Triage Category + Steps."
                    response = client.models.generate_content(model="gemini-2.5-flash", contents=[prompt, image])
                    
                    # Rendering Output within the flow
                    st.markdown(f'<div class="status-card">{response.text}</div>', unsafe_allow_html=True)
                    autoplay_audio(response.text)

                except exceptions.ResourceExhausted:
                    st.error("API Rate Limit Reached.")
                except Exception as e:
                    st.error(f"Execution Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #475569; padding-bottom: 80px; border-top: 1px solid rgba(255,255,255,0.05); padding-top:40px;">
        <p style="letter-spacing:1px; font-weight:600; color:#64748b;">RESEARCH PROTOTYPE // NON-DIAGNOSTIC</p>
        <p style="font-size:0.8rem;">© 2026 PocketDoc Labs. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)
