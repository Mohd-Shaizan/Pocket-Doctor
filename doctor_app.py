import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Doctor in Your Pocket | Research Prototype",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for a professional "Clinical" interface
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .status-card {
        background-color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #007bff;
    }
    .header-text { color: #1e3a8a; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

# --- API INITIALIZATION ---
# Accessing the masked API Key from Streamlit Secrets
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
except Exception:
    st.error("API Key not found. Please configure GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

# --- APP UI ---
st.markdown("<h1 class='header-text'>🩺 Doctor in Your Pocket</h1>", unsafe_allow_html=True)
st.write("### Hybrid Multimodal Triage Assistant")
st.info("🧬 **Serious Angle:** This demo shows how AI correlates verbal symptoms with visual markers for faster clinical triage.")

# Create the Split-Screen Layout
col_input, col_output = st.columns([1, 1], gap="medium")

with col_input:
    st.subheader("📥 Input Modalities")
    
    # Visual Input
    with st.expander("📷 Camera / Image Input", expanded=True):
        img_file = st.camera_input("Capture visible symptoms (e.g., skin changes, throat, swelling)")
    
    # Voice/Text Input
    with st.expander("💬 Symptom Description", expanded=True):
        symptoms = st.text_area(
            "Describe how you feel:",
            placeholder="e.g., 'I have a burning sensation on my forearm that started 3 hours ago. It's itchy and spreading.'",
            height=150
        )

with col_output:
    st.subheader("📋 AI Triage Summary")
    
    if st.button("Analyze Hybrid Context", type="primary", use_container_width=True):
        if not img_file or not symptoms:
            st.warning("Please provide both an image and a description for accurate triage.")
        else:
            with st.spinner("Processing multimodal data..."):
                try:
                    # Convert uploaded file to PIL Image
                    image = Image.open(img_file)
                    
                    # Construct the multimodal prompt
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

                    # Gemini 2.0 Flash is optimized for fast multimodal reasoning
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=[prompt, image]
                    )

                    st.markdown(f'<div class="status-card">{response.text}</div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    else:
        st.write("Submit data on the left to generate the triage summary.")

# --- FOOTER ---
st.divider()
st.caption("⚠️ **Research Prototype Only:** This system is for demonstration purposes. It does not provide medical advice. In an emergency, call 911 or your local emergency number.")
