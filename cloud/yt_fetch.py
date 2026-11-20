"""Retry YouTube downloads from Modal (outside India) for region-blocked FPVSum videos.

    modal run cloud/yt_fetch.py --ids Scuba_KD74Ywuy9ng,Scuba_Um4kqB8bQBg
    modal volume get ustad-t4 fpvsum/ data/fpvsum/
"""
import modal

image = modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg").pip_install("yt-dlp")
vol = modal.Volume.from_name("ustad-t4", create_if_missing=True)
app = modal.App("ustad-t4-ytfetch", image=image)


@app.function(timeout=900, volumes={"/vol": vol})
def fetch(video_id: str) -> str:
    import os
    import subprocess

    yt = video_id.split("_", 1)[1][:11]
    os.makedirs("/vol/fpvsum", exist_ok=True)
    dest = f"/vol/fpvsum/{video_id}.mp4"
    r = subprocess.run(
        ["yt-dlp", "-q", "--no-warnings", "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]/b",
         "--merge-output-format", "mp4", "-o", dest, f"https://www.youtube.com/watch?v={yt}"],
        capture_output=True, text=True,
    )
    vol.commit()
    if r.returncode != 0 or not os.path.exists(dest):
        return f"{video_id}: failed: {r.stderr.strip()[-200:]}"
    return f"{video_id}: ok {os.path.getsize(dest) // 1_000_000} MB"


@app.local_entrypoint()
def main(ids: str):
    for res in fetch.map(ids.split(",")):
        print(res)
