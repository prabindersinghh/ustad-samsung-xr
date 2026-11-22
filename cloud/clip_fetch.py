"""Fetch two EPIC-100 val videos on Modal and keep small 540p copies, for the demo film only.

    modal run cloud/clip_fetch.py
    modal volume get ustad-t4 clips ./data/clips
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .pip_install("requests", "pyyaml")
    .run_commands("git clone --depth 1 https://github.com/epic-kitchens/epic-kitchens-download-scripts /opt/epicdl")
)
vol = modal.Volume.from_name("ustad-t4", create_if_missing=True)
app = modal.App("ustad-t4-clips", image=image)


@app.function(cpu=4, timeout=3600, volumes={"/vol": vol})
def fetch(video_id: str) -> str:
    import glob, os, subprocess, sys
    work = f"/tmp/{video_id}"
    subprocess.run([sys.executable, "epic_downloader.py", "--videos", "--specific-videos", video_id, "--output-path", work],
                   cwd="/opt/epicdl", check=True, capture_output=True)
    src = (glob.glob(f"{work}/**/{video_id}.MP4", recursive=True) + glob.glob(f"{work}/**/{video_id}.mp4", recursive=True))[0]
    os.makedirs("/vol/clips", exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", "scale=-2:540", "-r", "30", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "24", "-an", f"/vol/clips/{video_id}.mp4"], check=True)
    vol.commit()
    return f"{video_id}: {os.path.getsize(f'/vol/clips/{video_id}.mp4') / 1e6:.1f} MB"


@app.local_entrypoint()
def main():
    for r in fetch.map(["P04_28", "P01_12"]):
        print(r)
