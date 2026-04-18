import streamlit as st
import os
from stt import transcribe_and_embed
from process_incoming import run_rag_query
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

# Page Setup
st.set_page_config(
    page_title="AI Teaching Assistant",
    page_icon="🎓",
    layout="wide"
)

# Custom Styling for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Info
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3429/3429153.png", width=100)
    st.title("Project Overview")
    st.info("""
    **Tech Stack:**
    - **Frontend:** Streamlit
    - **STT:** OpenAI Whisper
    - **LLM:** Groq (Llama 3.3)
    - **Vector DB:** Local Embeddings (Sentence-Transformers)
    """)
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("🎥 RAG-Based AI Teaching Assistant")
st.write("Transform your lecture audio into an interactive knowledge base.")

# Layout Columns
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.subheader("📁 Step 1: Upload Lecture")
    uploaded_file = st.file_uploader("Upload an MP3 audio file", type=["mp3"])
    
    if uploaded_file:
        # Create directory if it doesn't exist
        os.makedirs("audios", exist_ok=True)
        file_path = os.path.join("audios", uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio(uploaded_file)

        if st.button("🚀 Process & Index Lecture"):
            with st.spinner("Transcribing and generating embeddings..."):
                try:
                    full_text = transcribe_and_embed(file_path)
                    st.success("Lecture Indexed Successfully!")
                    with st.expander("Preview Transcript"):
                        st.write(full_text)
                except Exception as e:
                    st.error(f"Error processing audio: {e}")

with col2:
    st.subheader("💬 Step 2: Ask the Assistant")
    
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask about the lecture content..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Searching transcripts..."):
                response = run_rag_query(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.caption("Developed by Anvesha Sharma | Full-Stack & ML Portfolio Project")