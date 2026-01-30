# ReSpeak

Translate spoken audio into another language. Optionally use voice cloning so the translated speech mimics the original speaker’s voice.

**Pipeline:** Download (or use local) audio → Transcribe (Whisper) → Translate (Ollama) → Speak (TTS). Output is an MP3 in the target language.

---

## What you need locally

- **Python 3.11+**
- **Ollama** installed and running, with a model pulled (e.g. `ollama pull qwen2.5:14b`)
- **ffmpeg** (for audio handling; often already installed)
- **yt-dlp** (only if you use the YouTube downloader): run `brew install yt-dlp` before using it

---

## Setup

1. **Clone and enter the project**
   ```bash
   cd re-speak
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Optional: keep models in the project folder**
   ```bash
   export TTS_HOME=$PWD/.models
   ```

4. **Create a `.env` file** in the project root:
   ```env
   TARGET_LANGUAGE=en
   VOICE_CLONING=true
   ```
   - `TARGET_LANGUAGE`: target language code (e.g. `en`, `ru`, `de`). Used for translation and TTS.
   - `VOICE_CLONING`: set to `true` to use XTTS voice cloning; `false` uses the multilingual MMS TTS (no cloning).

---

## How to run

Edit `main.py` at the bottom to match your case:

- **From a YouTube URL:**  
  Keep or set `download_youtube_audio(...)` with your URL and a `filename`. The script will download the audio and then run the full pipeline.

- **From an existing file:**  
  Put your MP3 in `./audio/` as `<filename>.mp3`, comment out the `download_youtube_audio(...)` line, and set `filename = "<your_filename>"`.

Then:

```bash
python main.py
```
---

## What is downloaded automatically

| Step        | Tool / model              
|------------|---------------------------
| Transcribe | faster-whisper `large-v3` 
| Translate  | Ollama model              
| Speak      | Coqui XTTS v2 (if cloning) or MMS TTS

First run will download these; later runs use the cache.

---

## Performance

Trasncription, translation and generating the mp3 took about **1.6× the duration of the audio** in testing (e.g. 10 minutes of audio → ~16 minutes to translate)


