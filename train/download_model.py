"""
download_model.py
Downloads microsoft/bitnet-b1.58-2B-4T-bf16 from HuggingFace.
No token required — model is fully public (MIT license).
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config

def download():
    local_dir = Path(config.MODEL_LOCAL_DIR)

    # Check if weights already downloaded (not just config)
    weight_files = list(local_dir.glob("*.safetensors")) + list(local_dir.glob("pytorch_model*.bin"))
    if weight_files:
        print(f"[OK] Model already at {local_dir} -- skipping download.")
        print(f"     Found {len(weight_files)} weight file(s).")
        return str(local_dir)

    print(f"[>>] Downloading {config.BASE_MODEL_ID}")
    print(f"    -> Saving to: {local_dir}")
    print(f"    -> Size: ~4 GB (BF16 master weights for fine-tuning)")
    print()

    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id    = config.BASE_MODEL_ID,
        local_dir  = str(local_dir),
        local_dir_use_symlinks = False,
        ignore_patterns = ["*.msgpack", "*.h5", "flax_model*"],
    )

    print()
    print(f"[OK] Model downloaded to {local_dir}")
    return str(local_dir)


if __name__ == "__main__":
    download()
