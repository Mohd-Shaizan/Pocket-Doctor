import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Doctor in Your Pocket | AI Triage",
    page_icon="🩺",
    layout="wide"
)

# --- ADVANCED CLINICAL UI STYLING (Dr.AI Inspired) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #F0F4F8;
    }

    /* Professional Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }

    /* Glassmorphism Input Cards */
    div[data-testid="column"] {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }

    /* Result Box Styling */
    .status-card {
        background-color: #FFFFFF;
        color: #1E293B;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        border-left: 6px solid #3B82F6;
        line-height: 1.6;
    }

    /* Header styling */
    .header-container {
        text-align: center;
        padding: 2rem 0;
    }
    
    .header-text { 
        color: #0F172A; 
        font-size: 2.5rem !important;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle-text {
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Buttons */
    .stButton>button {
        background-color: #3B82F6 !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #2563EB !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- API INITIALIZATION ---
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
except Exception:
    st.error("API Key not found. Please configure GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

# --- HEADER SECTION ---
st.markdown("""
    <div class="header-container">
        <h1 class='header-text'>🩺 PocketDoc AI</h1>
        <p class='subtitle-text'>Next-Generation Multimodal Triage Assistant</p>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN INTERFACE LAYOUT ---
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### 📥 Diagnostics Input")
    
    # Visual Input with a cleaner container
    st.write("**Visual Examination**")
    img_file = st.camera_input("Position camera over visible symptoms")
    
    st.divider()
    
    # Voice/Text Input with more professional labeling
    st.write("**Patient History & Symptoms**")
    symptoms = st.text_area(
        label="Voice Input (Transcribed)",
        placeholder="Describe onset, duration, and sensation (e.g., 'Burning sensation on arm for 2 hours...')",
        height=180,
        label_visibility="collapsed"
    )

with col_output:
    st.markdown("### 📋 Clinical Summary")
    
    if st.button("Generate AI Triage Report", type="primary", use_container_width=True):
        if not img_file or not symptoms:
            st.warning("⚠️ Multimodal context incomplete. Please provide both image and description.")
        else:
            with st.spinner("Analyzing data streams..."):
                try:
                    image = Image.open(img_file)
                    
                    prompt = f"""
                    SYSTEM: You are a professional medical triage assistant prototype. 
                    USER SYMPTOMS: {symptoms}
                    
                    TASK:
                    1. Analyze the visual characteristics in the provided image.
                    2. Correlate visual findings with the user's verbal description.
                    3. Provide a structured triage report including:
                       - Visual Morphology (color, shape, borders)
                       - Correlated Assessment
                       - Triage Category (Emergent, Urgent, or Non-Urgent)
                       - Suggested next steps for the user.
                    
                    DISCLAIMER: Start with a bold disclaimer that this is a research prototype and not a diagnosis.
                    """

                    # Using the version of flash you specified in your working code
                    response = client.models.generate_content(
                        model="gemini-2.0-flash", 
                        contents=[prompt, image]
                    )

                    st.markdown(f'<div class="status-card">{response.text}</div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    else:
        st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; color: #94A3B8; border: 2px dashed #E2E8F0; border-radius: 15px;">
                <p>Awaiting multimodal input capture...</p>
            </div>
            """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()
st.markdown("""
    <div style="text-align: center; color: #64748B;">
        <p><strong>⚠️ RESEARCH PROTOTYPE ONLY</strong></p>
        <p style="font-size: 0.85rem;">This application demonstrates AI-assisted triage capabilities. It does not provide medical diagnoses. <br> 
        In case of emergency, contact local emergency services immediately.</p>
    </div>
    """, unsafe_allow_html=True)
