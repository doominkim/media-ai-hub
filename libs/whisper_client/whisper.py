# pip install openai-whisper 필요
import whisper

def transcribe(audio_path: str) -> str:
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"] 