from modules.audio import transcribe_audio
import streamlit as st
import os

# SAFETY CONSTRAINT: Ensure output directory exists
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        with st.spinner("Transcribing audio..."):
            transcription = transcribe_audio(temp_audio_path)
            
        # Placeholders for upcoming modules
        detected_intent = "Pending LLM..."
        action_taken = "Pending Execution..."
        final_output = "Pending..."
        
        # Display the results
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Transcribed Text:**")
            st.info(transcription)  # <--- Now displaying real transcription!
            
            st.markdown("**Detected Intent:**")
            st.info(detected_intent)
            
        with col2:
            st.markdown("**Action Taken:**")
            st.warning(action_taken)
            
            st.markdown("**Final Output:**")
            st.success(final_output)