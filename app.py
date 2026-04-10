import os
import streamlit as st
from langchain_community.chat_models import ChatOllama

# Import our custom modules
from modules.audio import transcribe_audio
from modules.intent import detect_intent
from modules.tools import execute_action

# SAFETY CONSTRAINT: Ensure output directory exists
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache the LLM so it doesn't reload into RAM on every Streamlit interaction
@st.cache_resource
def load_llm():
    print("Loading Llama 3.2:1B for tool execution...")
    return ChatOllama(model="llama3.2:1b", temperature=0)

llm = load_llm()

st.set_page_config(page_title="Local AI Voice Agent", layout="centered")
st.title("🎙️ Local Voice-Controlled AI Agent")

st.subheader("1. Provide Audio Input")
input_method = st.radio("Choose input method:", ["Microphone", "Upload Audio File"])

audio_bytes = None

# Audio Input Logic
if input_method == "Microphone":
    audio_value = st.audio_input("Record a voice command")
    if audio_value:
        audio_bytes = audio_value.read()
        
elif input_method == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload .wav or .mp3", type=["wav", "mp3"])
    if uploaded_file:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format="audio/wav")

# --- Processing Pipeline ---
if audio_bytes:
    st.success("Audio captured! Ready to process.")
    
    if st.button("Process Command"):
        st.subheader("Pipeline Results")
        
        # 1. Save audio to a temporary file in the output folder
        temp_audio_path = os.path.join(OUTPUT_DIR, "temp_input.wav")
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)
            
        # 2. Run Transcription
        with st.spinner("Transcribing audio via Faster-Whisper..."):
            transcription = transcribe_audio(temp_audio_path)
            
        # 3. Detect Intent
        with st.spinner("Analyzing intent via Llama 3.2:1B..."):
            detected_intent = detect_intent(transcription)
            
        # 4. Execute Action via Tools Engine
        with st.spinner(f"Executing tool for intent: '{detected_intent}'..."):
            # This triggers the master router we built in Step 1
            execution_result = execute_action(detected_intent, transcription, llm)
            
            # Extract the dictionary values returned by tools.py
            action_taken = execution_result.get("action", "Unknown Action")
            final_output = execution_result.get("result", "No output generated.")
        
        # Display the results directly mapped to the requirements
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Transcribed Text:**")
            st.info(transcription) 
            
            st.markdown("**Detected Intent:**")
            st.info(detected_intent)
            
        with col2:
            st.markdown("**Action Taken:**")
            st.warning(action_taken)
            
            st.markdown("**Final Output:**")
            st.success(final_output)