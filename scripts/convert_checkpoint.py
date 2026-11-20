"""Turn the LaViLa training checkpoint into a plain, portable state_dict.

The released file pickles training args (a PosixPath, pandas objects, ...), so it
needs weights_only=False and extra packages, and fails on Windows. This reads it
with an unpickler that replaces every non-tensor class with an inert stub, keeps
only the tensors the classifier uses, and writes a file that loads safely with
weights_only=True anywhere.

    python scripts/convert_checkpoint.py <in.pth> <out.pth>
"""
import pickle
import sys
import types

import torch

SAFE_PREFIXES = ("torch", "collections", "numpy", "builtins", "_codecs")


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(SAFE_PREFIXES):
            return super().find_class(module, name)
        return _Stub


stub_pickle = types.ModuleType("stub_pickle")
stub_pickle.Unpickler = _Unpickler
stub_pickle.load = pickle.load
stub_pickle.__name__ = "pickle"

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    ckpt = torch.load(src, map_location="cpu", weights_only=False, pickle_module=stub_pickle)
    state = {k.replace("module.", "", 1): v for k, v in ckpt["state_dict"].items()}
    keep = {k: v for k, v in state.items() if k.startswith("visual.") or k.startswith("fc_cls.")}
    torch.save({"state_dict": keep}, dst)
    print(f"kept {len(keep)}/{len(state)} tensors -> {dst}")
