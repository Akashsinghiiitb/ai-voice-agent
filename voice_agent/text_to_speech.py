import os

try:
    from gtts import gTTS
except ImportError:
    gTTS = None
    print("Warning: gTTS is not installed.")

class TextToSpeech:
    """
    Text to Speech module utilizing gTTS (Google Text-to-Speech).
    Converts synthesized text responses into MP3 audio streams.
    """
    def __init__(self, lang: str = "en"):
        self.lang = lang

    def synthesize(self, text: str, output_path: str) -> str:
        """
        Converts text string into an MP3 file at output_path.
        """
        # Ensure directories exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        if not gTTS:
            raise ImportError("gTTS is not installed or available on this system.")
            
        try:
            tts = gTTS(text=text, lang=self.lang, slow=False)
            tts.save(output_path)
            print(f"Speech audio successfully written to {output_path}")
            return output_path
        except Exception as e:
            raise RuntimeError(f"gTTS synthesis failed: {e}")
