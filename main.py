from faster_whisper import WhisperModel
import os
import ollama
import re

audio_path = "audio/"
audio_file_suffix = ".mp3"
transcript_path = "transcripts/"

class Transcriber:
    def __init__(self):
        model_type = os.getenv('WHISPER_MODEL', 'large-v3').replace("/", "_")
        base_model_cache = os.getenv('MODEL_CACHE_DIR', './models')
        model_cache_dir = os.path.join(base_model_cache, model_type)
        
        print(f"Loading faster-whisper model: {model_type}")
        
        # Optimized for M2 Max, configure yourself for best performance
        # Check if the model is already downloaded in the custom directory
        if os.path.exists(os.path.join(model_cache_dir, "model.bin")):
             print(f"Loading model from local path: {model_cache_dir}")
             self.model = WhisperModel(model_cache_dir, device="cpu", compute_type="int8", cpu_threads=8)
        else:
            try:
                print(f"Model not found in {model_cache_dir}, attempting download...")
                self.model = WhisperModel(model_type, device="cpu", download_root=model_cache_dir, compute_type="int8", cpu_threads=8)
            except Exception as e:
                print(f"Error loading model: {e}")
                raise

    def transcribe(self, audio_file_name: str) -> str:
        """Transcribe the audio file and return the transcript as a string"""
        full_path = f"{audio_path}{audio_file_name}{audio_file_suffix}"
        print(f"Transcribing: {full_path}")
        
        # use vad_filter to remove silence from the audio
        segments, info = self.model.transcribe(
            full_path, 
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        last_step = 0
        transcript_text = []
        for segment in segments:
            if info.duration > 0:
                current_step = int((segment.end / info.duration) * 100)
                if current_step > last_step:
                    print(f"Transcription Progress: {current_step}%")
                    last_step = current_step
            transcript_text.append(segment.text)

        return " ".join(transcript_text).strip()

    def create_transcript(self, audio_file_name: str) -> str:
        """Create a transcript file for the audio file and return the path to the transcript file"""
        os.makedirs(transcript_path, exist_ok=True)
        
        transcript_file_path = f"{transcript_path}{audio_file_name}.txt"
        transcript_content = self.transcribe(audio_file_name)

        with open(transcript_file_path, "w", encoding="utf-8") as f:
            f.write(transcript_content)
            
        print(f"Transcript saved to: {transcript_file_path}")
        return transcript_file_path


class Translator:
    def __init__(self):
        print(f"Loading ollama model {os.getenv('OLLAMA_MODEL')}")
        self.model = ollama.Client(model=os.getenv("OLLAMA_MODEL"))
        self.max_chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "1000"))

    def chunk_text(self, transcript_path: str) -> list[str]:
        # Read the transcript file
        with open(transcript_path, "r") as f:
            transcript = f.read()

        # Split the transcript into sentences
        sentences = re.split(r'(?<=[.!?])\s+', transcript)
        chunks = []
        current_chunk = []
        current_length = 0

        # Chunk the sentences into chunks with max lenght as defined
        for sentence in sentences:
            # If the sentence fits into the current chunk, append it
            if current_length + len(sentence) < self.max_chunk_size:
                current_chunk.append(sentence)
                current_length += len(sentence)
            # If the sentence doesn't fit into the current chunk, start a new chunk with the sentence that didn't fit
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        print(f"Chunked {len(chunks)} chunks")
        return chunks

if __name__ == "__main__":
    transcriber = Transcriber()
    transcript = transcriber.create_transcript("#016 Joh 4,13-14 Nur Jesus gibt lebendiges Wasser (Samuel Driesner)")
    