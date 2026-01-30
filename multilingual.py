from transformers import AutoTokenizer, VitsModel
import glob
import os
import re
import shutil
import soundfile as sf
import numpy as np
import torch
from pydub import AudioSegment
from dotenv import load_dotenv

from constants import *
load_dotenv()

target_language = os.getenv("TARGET_LANGUAGE", "en")

"""
Speaker class supporting more languages than the default one
"""
class SpeakerMultilingual:
    def __init__(self):
        self.device = "cpu"
        self.max_chunk_size = 250
        self.sample_rate = 16000

        print("Loading facebook/mms-tts-ron...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "facebook/mms-tts-ron",
            cache_dir=model_cache
        )

        self.model = VitsModel.from_pretrained(
            "facebook/mms-tts-ron",
            cache_dir=model_cache
        ).to(self.device)

        self.model.eval()

    # --- same splitting logic as before ---
    def _recursive_split(self, text, max_chars):
        if len(text) <= max_chars:
            return [text]

        separators = [r'(?<=[.!?])\s+', r'(?<=[;:,])\s+', r'\s+']

        for separator in separators:
            parts = re.split(separator, text)
            if len(parts) == 1:
                continue

            chunks, current = [], ""
            for part in parts:
                if len(current) + len(part) < max_chars:
                    current += part + " "
                else:
                    chunks.append(current.strip())
                    current = part + " "
            if current:
                chunks.append(current.strip())
            return chunks

        return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

    def _chunk_transcript(self, transcript_filename: str):
        with open(f"{transcript_path}{transcript_filename}.txt", "r", encoding="utf-8") as f:
            text = f.read()
        return self._recursive_split(text, self.max_chunk_size)

    def _tts_to_wav(self, text: str, out_path: str):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            waveform = self.model(**inputs).waveform

        audio = waveform.squeeze().cpu().numpy().astype(np.float32)

        sf.write(out_path, audio, self.sample_rate)

    def _combine_wavs(self, filename: str):
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
            .set_frame_rate(self.sample_rate)
            .set_channels(1)
            .set_sample_width(2)
        )

        out_mp3 = f"{translated_audio_path}{filename}_{target_language}.mp3"
        combined.export(out_mp3, format="mp3", bitrate="192k")

        print(f"Final MP3 written to: {out_mp3}")

    def _cleanup(self, filename: str):
        shutil.rmtree(f"{tmp_path}{filename}")

    def translate_transcript(self, filename: str):
        chunks = self._chunk_transcript(f"{filename}_{target_language}")
        os.makedirs(f"{tmp_path}{filename}", exist_ok=True)

        for i, chunk in enumerate(chunks):
            print(f"Generating chunk {i+1}/{len(chunks)}...")
            out_wav = f"{tmp_path}{filename}/{filename}_{target_language}_{i}.wav"
            self._tts_to_wav(chunk, out_wav)

        self._combine_wavs(filename)
        self._cleanup(filename)
