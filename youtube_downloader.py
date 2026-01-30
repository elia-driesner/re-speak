import subprocess
from pathlib import Path

def download_youtube_audio(
    url: str,
    output_dir: str,
    format: str = "mp3"
) -> str:
    """
    Downloads YouTube audio and converts it to MP3 or WAV.
    Uses the video title as the filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the filename yt-dlp will use (video title, sanitized)
    result = subprocess.run(
        ["yt-dlp", "--print", "filename", "-o", "%(title)s.%(ext)s", url],
        capture_output=True,
        text=True,
        check=True,
    )
    output_filename = result.stdout.strip()

    output_template = str(output_dir / "%(title)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "--extract-audio",
        "--audio-format", format,
        "--audio-quality", "0",
        "-o", output_template,
        url,
    ]

    subprocess.run(cmd, check=True)
    return output_filename
