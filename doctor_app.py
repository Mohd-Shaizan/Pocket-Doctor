import streamlit as st
from PIL import Image
from google import genai
from google.api_core import exceptions

# --- PAGE CONFIG ---
st.set_page_config(page_title="PocketDoc AI", page_icon="🩺", layout="wide")

# --- HIGH-VISIBILITY CLINICAL CSS ---
st.markdown("""
    <style>
    /* Force the background color on the main app area */
    .stApp {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%) !important;
    }

    /* Professional Card Styling with forced white background */
    [data-testid="stVerticalBlock"] > div > div > div[data-testid="column"] {
        background-color: white !important;
        padding: 2rem !important;
        border-radius: 20px !important;
        border: 1px solid #d1d5db !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1) !important;
    }

    /* Custom Header */
    .main-header {
        text-align: center;
        color: #1e3a8a;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }
    
    .sub-header {
        text-align: center;
        color: #475569;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* Result Card */
    .status-card {
        background-color: #ffffff !important;
        color: #1e293b !important;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 8px solid #3b82f6;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.02);
    }

    /* Adjusting the Camera Input spacing */
    div[data-testid="stCameraInput"] {
        padding: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- API INITIALIZATION ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
except Exception:
    st.error("Missing API Key in Secrets.")
    st.stop()

# --- HEADER ---
st.markdown("<h1 class='main-header'>🩺 PocketDoc AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Research Prototype: Multimodal Clinical Triage</p>", unsafe_allow_html=True)

# --- MAIN LAYOUT ---
col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.markdown("### 📥 Diagnostic Inputs")
    
    # 1. Visual Capture
    st.write("**Step 1: Visual Examination**")
    img_file = st.camera_input("Capture visible symptoms")
    
    # 2. Symptom Description
    st.write("**Step 2: Patient Narrative**")
    symptoms = st.text_area(
        "Describe your symptoms (Voice-to-Text):",
        placeholder="e.g. 'I have a red, itchy patch on my forearm that appeared this morning...'",
        height=150
    )

with col_out:
    st.markdown("### 📋 AI Triage Report")
    
    if st.button("🚀 Run Multimodal Analysis", type="primary", use_container_width=True):
        if not img_file or not symptoms:
            st.warning("Please provide both an image and a symptom description.")
        else:
            with st.spinner("Analyzing hybrid data streams..."):
                try:
                    image = Image.open(img_file)
                    
                    prompt = f"""
                    SYSTEM: Professional Triage Assistant.
                    SYMPTOMS: {symptoms}
                    TASK: Analyze image + text. Provide Triage Category and Next Steps.
                    DISCLAIMER: This is not a diagnosis.
                    """

                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=[prompt, image]
                    )

                    st.markdown(f'<div class="status-card">{response.text}</div>', unsafe_allow_html=True)
                
                except exceptions.ResourceExhausted:
                    st.error("Quota exceeded. Please wait 60s.")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        # Placeholder when no analysis has run
        st.info("Captured data will be analyzed here in real-time.")

# --- FOOTER ---
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #64748B; font-size: 0.8rem;'>
        <strong>RESEARCH PROTOTYPE ONLY</strong><br>
        Not for medical use. In case of emergency, call 911 immediately.
    </div>
    """, unsafe_allow_html=True)
