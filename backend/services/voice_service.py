"""
Voice service for speech-to-text and text-to-speech
Optional components - application will work without these
"""

import os
import base64
import tempfile

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


class VoiceService:
    def __init__(self):
        """Initialize voice service with optional components."""
        if WHISPER_AVAILABLE:
            print("Loading Whisper model...")
            self.stt_model = whisper.load_model("base")
            print("Whisper model loaded.")
        else:
            print("[WARNING] Whisper not available - speech-to-text disabled")
            self.stt_model = None

    def transcribe(self, audio_path):
        """
        Converts Audio file -> Text + Language Detection
        Returns: (text, language) or ("", "en") if unavailable
        """
        if not WHISPER_AVAILABLE or self.stt_model is None:
            print("[WARNING] Transcription not available (Whisper not installed)")
            return "", "en"
            
        try:
            result = self.stt_model.transcribe(audio_path)
            text = result["text"].strip()
            language = result["language"]
            
            print(f"Transcribed ({language}): {text}")
            return text, language
        except Exception as e:
            print(f"Error in transcription: {e}")
            return "", "en"

    def text_to_speech(self, text, language='en', output_path=None):
        """
        Converts Text -> Audio file
        Returns: path to audio file or None if unavailable
        """
        if not GTTS_AVAILABLE:
            print("[WARNING] Text-to-speech not available (gTTS not installed)")
            return None
            
        try:
            tts = gTTS(text=text, lang=language, slow=False)
            
            if output_path is None:
                # Create temp file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                output_path = temp_file.name
                temp_file.close()
            
            tts.save(output_path)
            print(f"Audio saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            return None


    def speak(self, text, lang='en'):
        """
        Converts Text -> Audio (Base64 string) using Google TTS
        """
        try:
            # Map Whisper language codes to gTTS language codes if necessary
            # gTTS supports 'ta' (Tamil), 'hi' (Hindi), 'en' (English) natively
            if lang not in ['en', 'hi', 'ta']:
                lang = 'en' # Fallback to English

            # Generate audio in memory
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # Save to a temporary file first (gTTS limitation)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                tts.save(temp_audio.name)
                temp_filename = temp_audio.name

            # Read file and convert to Base64
            with open(temp_filename, "rb") as audio_file:
                audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

            # Cleanup
            os.unlink(temp_filename)
            
            return audio_base64
        except Exception as e:
            print(f"Error in speech generation: {e}")
            return None