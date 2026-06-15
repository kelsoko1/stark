# Cloud Training Guide — Q-BitNet + LingBot-World v3

## Option 1: Google Colab (T4 GPU — Free)
**Speed**: ~10–30x faster than your CPU (~3–5 sec/step vs 90 sec)

### Steps
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **Runtime → Change runtime type → T4 GPU → Save**
3. Upload your `stark/` folder to Google Drive
4. In a new cell, run:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/stark

# Install dependencies
!pip install -q transformers>=4.45 datasets sentencepiece huggingface_hub

# Run cloud training
!python colab/colab_train.py
```

**Free limits**: ~12 hrs/session, sessions disconnect after inactivity.
**Tip**: Save checkpoints to Google Drive so you don't lose progress on disconnect.

---

## Option 2: Kaggle Notebooks (T4/P100 GPU — Free 30hr/week)
**Speed**: Same as Colab T4 (~3–5 sec/step)
**Advantage**: 30 hrs/week quota, sessions persist longer

### Steps
1. Go to [kaggle.com/code](https://kaggle.com/code) → **+ New Notebook**
2. **Settings (right panel) → Accelerator → GPU T4×2**
3. Upload `stark/` as a Kaggle Dataset (one-time):
   - Kaggle → Datasets → New Dataset → Upload stark folder → Create
4. In the notebook:

```python
# Add your dataset first: + Add Data → Your Datasets → stark
import subprocess
subprocess.run(["pip", "install", "-q", "transformers", "datasets",
                "sentencepiece", "huggingface_hub"])

import os
os.chdir("/kaggle/input/stark")

exec(open("colab/colab_train.py").read())
```

**Kaggle TPU** (v3-8, 20hr/week): More complex — needs `torch_xla`. See Option 4 below.

---

## Option 3: Google Colab Pro (A100 — Paid ~$10/month)
**Speed**: ~100x faster than CPU (~1 sec/step)

- Same steps as Option 1 but select **A100** runtime
- With A100 (40GB VRAM) you can unfreeze the base model:
  - Set `FREEZE_BASE_MODEL = False` in `CloudConfig`
  - Set `BATCH_SIZE = 8`, `MAX_SEQ_LEN = 512`
  - Full model training — much better results

---

## Option 4: Kaggle TPU v3-8 (Free 20hr/week) — Advanced
**Speed**: ~200x faster but requires PyTorch/XLA

### Extra steps needed
```python
!pip install -q cloud-tpu-client torch-xla==2.1 torchvision

import torch_xla.core.xla_model as xm
device = xm.xla_device()

# Then run training — the colab_train.py script auto-detects TPU
exec(open("colab/colab_train.py").read())
```

**Note**: TPU training requires all tensors to stay on the TPU device.
The current code has `is_tpu` detection built in.

---

## GPU Config vs CPU Config

| Setting | CPU (local) | Cloud GPU | Cloud A100 |
|---|---|---|---|
| `BATCH_SIZE` | 1 | 4 | 8 |
| `MAX_SEQ_LEN` | 128 | 256 | 512 |
| `GRAD_ACCUM` | 4 | 2 | 1 |
| `FREEZE_BASE_MODEL` | True | True | **False** |
| sec/step (est.) | ~90 | ~4 | ~1 |
| steps/hour | ~40 | ~900 | ~3600 |

---

## Expected Training Speed

| Platform | sec/step | steps/hr | Phi growth/hr |
|---|---|---|---|
| Your CPU (current) | 90 | 40 | +0.06 |
| Colab T4 | 4–5 | 720–900 | +1.1 |
| Kaggle T4 | 4–5 | 720–900 | +1.1 |
| Colab A100 | 0.8–1.2 | 3000–4500 | +5.5 |
| Kaggle TPU v3-8 | 0.3–0.5 | 7200+ | +11.0 |

---

## After Training on Cloud

Download your checkpoints back to your PC:
```python
# In Colab, zip and download
import shutil
shutil.make_archive('/content/checkpoints', 'zip', './checkpoints')
from google.colab import files
files.download('/content/checkpoints.zip')
```

Then extract to `stark/checkpoints/` on your local machine
and resume with the local trainer.
