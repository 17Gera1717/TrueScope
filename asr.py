from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE

# Load your trained model
asr_model = EncDecCTCModelBPE.restore_from("russian_asr_model.nemo")

def transcribe_audio(audio_path) -> str:
    """
    Transcribe audio using the ASR model.
    :param audio_path: Path to the audio file
    :return: Transcribed text
    """
    
    transcription = asr_model.transcribe([audio_path], return_hypotheses=False)
    return transcription[0].text

if __name__ == "__main__":
    # Example usage
    audio_path = "recording.wav"  # Replace with your audio file path
    text = transcribe_audio(audio_path)
    print(f"Transcription: {text}")