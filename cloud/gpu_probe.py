"""Smallest possible check that this Modal account can run a GPU function.

    modal run cloud/gpu_probe.py
"""
import modal

app = modal.App("ustad-t4-probe")


@app.function(gpu="T4", timeout=120, image=modal.Image.debian_slim().pip_install("torch==2.4.1"))
def probe() -> str:
    import torch
    return f"cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}"


@app.local_entrypoint()
def main():
    print(probe.remote())
