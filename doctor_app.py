import streamlit as st
from PIL import Image
from google import genai
from google.api_core import exceptions
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="PocketDoc AI", page_icon="🩺", layout="wide")

# --- LANGCHAIN-INSPIRED DARK UI CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono&display=swap');

    .stApp {
        background-color: #050505 !important;
        color: #e2e8f0 !important;
    }

    /* Hero Section Glow */
    .hero-container {
        text-align: center;
        padding: 100px 20px;
        background: radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.1) 0%, rgba(5, 5, 5, 1) 70%);
    }

    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 4.5rem !important;
        font-weight: 800;
        background: linear-gradient(to right, #ffffff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    .hero-tagline {
        font-size: 1.4rem;
        color: #94a3b8;
        max-width: 800px;
        margin: 0 auto 30px;
    }

    /* Section Cards (Bento Style) */
    .feature-section {
        background: rgba(30, 41, 59, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px);
        padding: 40px !important;
        border-radius: 24px !important;
        margin: 20px auto !important;
        max-width: 1000px;
    }

    /* Terminal/Mono styling for output */
    .status-card {
        background-color: #0f172a !important;
        color: #38bdf8 !important;
        font-family: 'JetBrains Mono', monospace;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #1e293b;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.1);
    }

    /* Buttons */
    .stButton>button {
        background: #3b82f6 !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 12px 30px !important;
        border: none !important;
        font-weight: 600 !important;
    }

    /* Hide Streamlit components that break the landing page feel */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- API INITIALIZATION ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
except Exception:
    st.error("Missing API Key in Secrets.")
    st.stop()

# --- 1. HERO SECTION ---
st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">PocketDoc AI</h1>
        <p class="hero-tagline">Advanced multimodal triage powered by Gemini 2.0. Correlate visual markers with patient narrative in real-time.</p>
    </div>
    """, unsafe_allow_html=True)

# --- 2. MODEL DESCRIPTION SECTION ---
st.markdown('<div class="feature-section">', unsafe_allow_html=True)
st.markdown("## 🧠 The Intelligence Engine")
col_desc1, col_desc2 = st.columns(2)
with col_desc1:
    st.write("### Multimodal Input")
    st.write("- **Visual:** High-fidelity morphology analysis.")
    st.write("- **Audio:** Natural language symptom processing.")
    st.write("- **Reasoning:** Cross-modal contextual correlation.")
with col_desc2:
    st.write("### Clinical Triage")
    st.write("- **Urgency Scoring:** High/Medium/Low prioritization.")
    st.write("- **SOAP Support:** Structured assessment formats.")
    st.write("- **Research Grade:** Designed for triage prototype analysis.")
st.markdown('</div>', unsafe_allow_html=True)

# --- 3. INTERACTIVE AI SECTION ---
st.markdown("<h2 style='text-align:center;'>🚀 Experience the Interface</h2>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="feature-section">', unsafe_allow_html=True)
    
    # --- Webcam Logic ---
    st.write("### 📸 Step 1: Visual Feed")
    cam_on = st.toggle("Enable Triage Camera", value=False)
    img_file = None
    if cam_on:
        img_file = st.camera_input("Position visible symptoms in frame")
    else:
        st.info("Webcam is currently disabled. Toggle the switch above to start.")

    st.divider()

    # --- Voice & Text Logic ---
    st.write("### 💬 Step 2: Symptom Dialogue")
    
    col_v1, col_v2 = st.columns([1, 2])
    with col_v1:
        st.write("Record Symptoms:")
        audio = mic_recorder(start_prompt="⏺️ Record", stop_prompt="⏹️ Stop", key='recorder')
    
    # Logic to handle speech-to-text or manual text
    symptoms_text = ""
    if audio:
        # In a production app, you would send 'audio['bytes']' to a STT API (like OpenAI Whisper or Google STT)
        # For this prototype, we will assume text input as the fallback or prompt the user.
        st.success("Audio captured! (Connect STT API here for real-time transcription)")
    
    symptoms_text = st.text_area(
        "Narrative Input:",
        placeholder="e.g. 'Red circular rash on left arm, itching for 48 hours...'",
        height=100
    )

    # --- ANALYSIS ENGINE ---
    if st.button("RUN TRIAGE ANALYSIS", use_container_width=True):
        if not img_file or not symptoms_text:
            st.warning("Analysis requires both a visual capture and a narrative input.")
        else:
            with st.spinner("Executing Inference Chain..."):
                try:
                    image = Image.open(img_file)
                    prompt = f"""
                    SYSTEM: Professional Triage Assistant Prototype.
                    SYMPTOMS: {symptoms_text}
                    TASK: Analyze visual signs + verbal context. Determine Risk Category and Next Steps.
                    """
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=[prompt, image]
                    )

                    # Output Rendering
                    st.markdown(f'<div class="status-card">{response.text}</div>', unsafe_allow_html=True)
                    
                    # --- TTS Playback ---
                    tts = gTTS(text=response.text[:500], lang='en') # Limit TTS length for speed
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    st.audio(audio_fp, format='audio/mp3')

                except exceptions.ResourceExhausted:
                    st.error("Quota Exceeded. Please retry in 60s.")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #64748B; padding: 40px;">
        <p>⚠️ <strong>RESEARCH PROTOTYPE ONLY</strong></p>
        <p>This is a demonstration of AI triage logic. It is not a medical device and does not provide diagnosis.</p>
        <p>© 2026 PocketDoc AI Labs</p>
    </div>
    """, unsafe_allow_html=True)
