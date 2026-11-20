"""Check this Modal account can run CPU functions and reach the EPIC data host.

    modal run cloud/cpu_probe.py
"""
import modal

app = modal.App("ustad-t4-cpu-probe")


@app.function(cpu=4, timeout=180, image=modal.Image.debian_slim().pip_install("requests"))
def probe() -> str:
    import os
    import time

    import requests

    t = time.time()
    r = requests.head("https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/test/P01/P01_11.MP4",
                      allow_redirects=True, timeout=30)
    return f"cpus={os.cpu_count()} epic_host_status={r.status_code} size={r.headers.get('content-length')} in {time.time()-t:.1f}s"


@app.local_entrypoint()
def main():
    print(probe.remote())
