import os

try:
    from faster_whisper import WhisperModel

    import_type = "faster-whisper"
except ImportError:
    try:
        import whisper

        import_type = "whisper"
    except ImportError:
        import_type = None


class SpeechToText:
    """
    Speech to Text module using local Whisper ASR models.
    Supports both faster-whisper and openai-whisper with real transcription.
    """

    def __init__(self, model_name: str = "tiny"):
        self.model_name = model_name
        self.import_type = import_type
        self.model = None

        if self.import_type == "faster-whisper":
            try:
                # Load faster-whisper model on CPU using 8-bit integer quantization to save RAM
                self.model = WhisperModel(
                    self.model_name, device="cpu", compute_type="int8"
                )
                print(f"Loaded faster-whisper model '{self.model_name}' on CPU.")
            except Exception as e:
                print(
                    f"Failed to load faster-whisper: {e}. Falling back to openai-whisper if available."
                )
                self.import_type = "whisper"

        if self.import_type == "whisper" and not self.model:
            try:
                import whisper

                self.model = whisper.load_model(self.model_name)
                print(f"Loaded openai-whisper model '{self.model_name}' on CPU.")
            except Exception as e:
                print(f"Failed to load openai-whisper: {e}")
                self.model = None

        if not self.model:
            raise ImportError(
                "Neither faster-whisper nor openai-whisper could be loaded. "
                "Please verify your virtual environment installations and CPU compatibility."
            )

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribes the uploaded audio path into clean plain text.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        try:
            if self.import_type == "faster-whisper":
                segments, info = self.model.transcribe(audio_path, beam_size=1)
                text = " ".join([segment.text for segment in segments])
                return text.strip()
            elif self.import_type == "whisper":
                result = self.model.transcribe(audio_path)
                return result.get("text", "").strip()
        except Exception as e:
            raise RuntimeError(f"ASR transcription runtime error: {e}")
