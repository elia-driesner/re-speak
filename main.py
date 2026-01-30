from faster_whisper import WhisperModel
import os
import shutil
import ollama
import re
import torch
import glob
from pydub import AudioSegment
from TTS.api import TTS

from dotenv import load_dotenv
load_dotenv()

audio_path = "./audio/"
model_cache = './.models'
tmp_path = "./.tmp/"
translated_audio_path = "./translated_audio/"
transcript_path = "./transcripts/"

os.makedirs(translated_audio_path, exist_ok=True)
os.makedirs(transcript_path, exist_ok=True)
os.makedirs(tmp_path, exist_ok=True)

target_language = os.getenv("TARGET_LANGUAGE", "en")

class Transcriber:
    def __init__(self):
        model_type = 'large-v3'
        model_cache_dir = os.path.join(model_cache, model_type)
        
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

    def _transcribe(self, audio_file_name: str) -> str:
        """Transcribe the audio file and return the transcript as a string"""
        full_path = f"{audio_path}{audio_file_name}.mp3"
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
        transcript_file_path = f"{transcript_path}{audio_file_name}.txt"
        transcript_content = self._transcribe(audio_file_name)

        with open(transcript_file_path, "w", encoding="utf-8") as f:
            f.write(transcript_content)
            
        print(f"Transcript saved to: {transcript_file_path}")
        return transcript_file_path


class Translator:
    def __init__(self):
        self.model_name = "qwen2.5:14b"
        self.client = ollama.Client()
        self.max_chunk_size = 1000
        print(f"Selected ollama model: {self.model_name} and max chunk size: {self.max_chunk_size}")

        self.system_instruction = (
            f"You are a professional dubbing translator. Translate the text into {target_language}. "
            "Do not explain your translation. Do not add notes. Do not translate the context. "
            "Output ONLY the translated text for the target section."
        )

    def _chunk_text(self, transcript_filename: str) -> list[str]:
        full_path = f"{transcript_path}{transcript_filename}.txt"
        with open(full_path, "r") as f:
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

    def _get_prompt(self, previous_chunk: str, chunk: str) -> list[dict]:
        message = [{"role": "system", "content": self.system_instruction}]
        if previous_chunk:
            message.append({"role": "system", "content": ("### CONTEXT (Previous sentence - DO NOT TRANSLATE):\n" f"{previous_chunk}\n\n")})
        message.append({"role": "user", "content": ("### TARGET (Translate this):\n" f"{chunk}\n\n")})
        return message

    def translate_transcript(self, transcript_filename: str):
        chunks = self._chunk_text(transcript_filename)
        translated_chunks = []
        for i, chunk in enumerate(chunks):
            previous_chunk = chunks[i-1] if i > 0 else ""
            translated_chunk = self.client.chat(model=self.model_name, 
                messages=self._get_prompt(previous_chunk, chunk)
            )
            translated_chunks.append(translated_chunk.message.content)

        with open(f"{transcript_path}{transcript_filename}_{target_language}.txt", "w") as f:
            f.write(" ".join(translated_chunks))


class Speaker:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.max_chunk_size = 250

        self.model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

    def _extract_reference_clip(self, input_audio_path: str, clip_duration_sec=20, start_sec=10) -> str:
        """
        Extracts a clean short clip from the beginning of the input audio to use as a voice reference.
        Saves it as a temp file and returns the path.
        """
        print(f"Extracting {clip_duration_sec}s reference clip from {input_audio_path}...")
        audio = AudioSegment.from_file(input_audio_path)

        start_ms = start_sec * 1000
        # Fallback to 0 if file is short
        if len(audio) < (start_sec + clip_duration_sec) * 1000:
            start_ms = 0
            
        clip = audio[start_ms:start_ms + (clip_duration_sec * 1000)]
        
        temp_ref_path = f"{tmp_path}temp_voice_ref.wav"
        clip.export(temp_ref_path, format="wav")
        return temp_ref_path

    def _recursive_split(self, text, max_chars):
        """
        Recursively splits text. 
        Priority:
        1. Sentences (. ! ?)
        2. Clauses (; : ,)
        3. Words (space)
        4. Hard Cut
        """
        if len(text) <= max_chars:
            return [text]

        separators = [r'(?<=[.!?])\s+', r'(?<=[;:,])\s+', r'\s+']
        
        for separator in separators:
            parts = re.split(separator, text)
            
            if len(parts) == 1:
                continue
                
            final_chunks = []
            current_chunk = ""
            
            for part in parts:
                if len(current_chunk) + len(part) < max_chars:
                    current_chunk += part + " "
                else:
                    if current_chunk:
                        final_chunks.append(current_chunk.strip())
                    if len(part) > max_chars:
                        sub_chunks = self._recursive_split(part, max_chars)
                        final_chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = part + " "
            
            if current_chunk:
                final_chunks.append(current_chunk.strip())
            
            return final_chunks

    def _chunk_transcript(self, transcript_filename: str):
        with open(f"{transcript_path}{transcript_filename}.txt", "r") as f:
            transcript = f.read()
        chunks = self._recursive_split(transcript, self.max_chunk_size)
        return chunks
    
    def _combine_wavs(
        self,
        filename: str,
    ):
        pattern = f"{tmp_path}{filename}/{filename}_{target_language}_*.wav"
        wav_files = sorted(glob.glob(pattern))

        if not wav_files:
            raise RuntimeError("No WAV chunks found")

        print(f"Combining {len(wav_files)} chunks...")

        combined = AudioSegment.from_wav(wav_files[0])

        for wav in wav_files[1:]:
            combined += AudioSegment.from_wav(wav)

        combined = (
            combined
            .set_frame_rate(24000)
            .set_channels(1)
            .set_sample_width(2)
        )

        out_mp3 = f"{translated_audio_path}{filename}_{target_language}.mp3"
        combined.export(out_mp3, format="mp3", bitrate="192k")

        print(f"Final MP3 written to: {out_mp3}")

    def _cleanup(self, filename: str):
        shutil.rmtree(f"{tmp_path}{filename}")
        os.remove(f"{tmp_path}temp_voice_ref.wav")

    def translate_transcript(self, filename: str):
        chunks = self._chunk_transcript(f"{filename}_{target_language}")
        refrenceClip = self._extract_reference_clip(f"{audio_path}{filename}.mp3")
        os.makedirs(f"{tmp_path}{filename}", exist_ok=True)

        for i, chunk in enumerate(chunks):
            print(f"Generating chunk {i+1}/{len(chunks)}...")
            self.model.tts_to_file(text=chunk, language=target_language, speaker_wav=refrenceClip, file_path=f"{tmp_path}{filename}/{filename}_{target_language}_{i}.wav")

        self._combine_wavs(filename)
        self._cleanup(filename)

if __name__ == "__main__":
    filename = "#016 Joh 4,13-14 Nur Jesus gibt lebendiges Wasser (Samuel Driesner)"
    transcriber = Transcriber()
    transcriber.create_transcript(filename)
    translator = Translator()
    translator.translate_transcript(filename)
    speaker = Speaker()
    speaker.translate_transcript(filename)