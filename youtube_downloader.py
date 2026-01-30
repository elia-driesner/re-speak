import subprocess
from pathlib import Path

def download_youtube_audio(
    url: str,
    output_dir: str,
    filename: str,
    format: str = "mp3"
) -> Path:
    """
    Downloads YouTube audio and converts it to MP3 or WAV.
    Returns path to the audio file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{filename}.%(ext)s"

    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "--extract-audio",
        "--audio-format", format,
        "--audio-quality", "0",
        "-o", str(output_path),
        url,
    ]

    subprocess.run(cmd, check=True)
