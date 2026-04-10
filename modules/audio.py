from faster_whisper import WhisperModel

# We load the model globally so it only loads once when the app starts.
# "base" is small enough for your 8GB RAM. 
# device="auto" will try to use your RTX 2050, compute_type="int8" prevents it from maxing out VRAM.
print("Loading Faster-Whisper model... (This might take a minute the first time to download)")
model = WhisperModel("base", device="auto", compute_type="int8")

def transcribe_audio(file_path):
    """
    Takes an audio file path, runs it through Whisper, and returns the transcribed text.
    """
    try:
        # Generate transcription
        segments, info = model.transcribe(file_path, beam_size=5)
        
        # Combine all segments into a single string
        transcription = " ".join([segment.text for segment in segments])
        return transcription.strip()
    
    except Exception as e:
        return f"Error during transcription: {str(e)}"