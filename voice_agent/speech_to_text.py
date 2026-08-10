import os


class SpeechToText:
    """
    Speech to Text module using local Whisper ASR models via faster-whisper.
    """

    def __init__(self, model_name: str = "tiny"):
        self.model_name = model_name
        self.model = None

        try:
            import psutil
        except ImportError:
            psutil = None

        try:
            from faster_whisper import WhisperModel

            if psutil:
                proc = psutil.Process()
                before = proc.memory_info().rss / (1024 * 1024)
                print(f"Memory before Whisper load: {before:.1f} MB")

            # Load faster-whisper model on CPU using 8-bit integer quantization to save RAM
            self.model = WhisperModel(
                self.model_name, device="cpu", compute_type="int8"
            )

            if psutil:
                after = proc.memory_info().rss / (1024 * 1024)
                print(f"Memory after Whisper load: {after:.1f} MB")

            print(f"Loaded faster-whisper model '{self.model_name}' on CPU.")
        except Exception as e:
            raise ImportError(
                f"Failed to load faster-whisper model '{self.model_name}': {e}. "
                "Please verify your virtual environment installations and CPU compatibility."
            )

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribes the uploaded audio path into clean plain text.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        try:
            segments, info = self.model.transcribe(audio_path, beam_size=1)
            text = " ".join([segment.text for segment in segments])
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"ASR transcription runtime error: {e}")

