# Cloud Training Guide — Q-BitNet + LingBot-World v3

## Continuous Training on Google Colab

The training script now supports **continuous training** — it automatically
saves checkpoints to Google Drive and resumes from where it left off after
disconnects, crashes, or session timeouts.

### Quick Start

1. Upload your `stark/` folder to Google Drive
2. Open `colab/colab_train.ipynb` in Google Colab
   - Or go to [colab.research.google.com](https://colab.research.google.com)
     → File → Upload notebook → select `colab/colab_train.ipynb`
3. **Runtime → Change runtime type → T4 GPU → Save**
4. Run all cells in order

That's it. The notebook handles everything else:
- Mounts Google Drive
- Installs dependencies
- Downloads the base model (first run only, ~5 min)
- Checks for existing checkpoints and resumes if found
- Saves checkpoints every 200 steps + every epoch to Drive
- Keep-alive thread prevents idle disconnects
- Auto-retries on CUDA OOM or runtime errors

### After a Disconnect

Colab free tier disconnects after ~12 hours or on inactivity.
When this happens:

1. Reconnect to the runtime (or start a new one)
2. **Runtime → Run all** (or run cells top to bottom)
3. The script finds your latest checkpoint on Drive and resumes automatically

You can repeat this as many times as you want — training continues
indefinitely across sessions.

### Manual Setup (without the notebook)

```python
# Cell 1
from google.colab import drive
drive.mount('/content/drive')

# Cell 2
import os
os.chdir('/content/drive/MyDrive/stark')

# Cell 3
!pip install -q transformers>=4.45 datasets torch sentencepiece huggingface_hub accelerate

# Cell 4
!python colab/colab_train.py
```

---

## How Checkpointing Works

| Feature | Details |
|---|---|
| Save location | `/content/drive/MyDrive/stark_checkpoints/` |
| Save frequency | Every 200 steps + end of each epoch |
| What's saved | Model + optimizer + brain modules + epoch + step + Phi |
| Old checkpoints | Auto-deleted (keeps last 5) |
| Resume | Automatic — finds latest checkpoint via `LATEST` pointer file |

### Checkpoint files

```
stark_checkpoints/
  LATEST                    # points to most recent checkpoint
  ckpt_step_00002000.pt     # checkpoint at step 2000
  ckpt_step_00002200.pt     # checkpoint at step 2200
  ...
```

---

## Configuration

Edit `CloudConfig` in `colab/colab_train.py` to change settings:

| Setting | Default | Description |
|---|---|---|
| `MAX_EPOCHS` | 100 | High number for continuous training |
| `CHECKPOINT_EVERY` | 200 | Save every N steps |
| `KEEP_MAX_CHECKPOINTS` | 5 | Number of checkpoints to keep |
| `BATCH_SIZE` | 4 | Increase on A100 (8) |
| `MAX_SEQ_LEN` | 256 | Increase on A100 (512) |
| `LOG_EVERY` | 10 | Print every N steps |
| `RETRY_DELAY` | 30 | Seconds to wait before retry after crash |
| `MAX_RETRIES` | 10 | Max consecutive retries before giving up |

### For Colab Pro (A100)

```python
FREEZE_BASE_MODEL = False   # unfreeze for full fine-tuning
BATCH_SIZE = 8
MAX_SEQ_LEN = 512
```

---

## GPU Config vs CPU Config

| Setting | CPU (local) | Colab T4 | Colab A100 |
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
| Colab A100 | 0.8–1.2 | 3000–4500 | +5.5 |

---

## Monitor Training Progress

Use Cell 6 in the notebook to check checkpoint status:

```python
import torch
from pathlib import Path

ckpt_dir = Path('/content/drive/MyDrive/stark_checkpoints')
ckpts = sorted(ckpt_dir.glob('ckpt_step_*.pt'))
if ckpts:
    state = torch.load(ckpts[-1], map_location='cpu', weights_only=False)
    print(f"Epoch: {state['epoch']}, Step: {state['global_step']}, Phi: {state.get('phi', 0):.4f}")
```

---

## Download Checkpoints to Local Machine

Use Cell 7 in the notebook:

```python
import shutil
from google.colab import files

shutil.make_archive('/content/stark_checkpoints', 'zip',
                    '/content/drive/MyDrive/stark_checkpoints')
files.download('/content/stark_checkpoints.zip')
```

Then extract to `stark/checkpoints/` on your local machine.

---

## Start Fresh (Delete All Checkpoints)

If you want to restart training from scratch, use Cell 8 or run:

```python
import shutil
shutil.rmtree('/content/drive/MyDrive/stark_checkpoints', ignore_errors=True)
```

---

## Kaggle (Alternative)

Kaggle also works with the same script, but checkpoints won't auto-save
to Google Drive. You'll need to save to the Kaggle output directory instead.

```python
import subprocess
subprocess.run(["pip", "install", "-q", "transformers", "datasets",
                "sentencepiece", "huggingface_hub", "accelerate"])

import os
os.chdir("/kaggle/input/stark")

# Set checkpoint dir to Kaggle output (persists for the session)
# Edit CloudConfig.CHECKPOINT_DIR in colab_train.py to "/kaggle/working/checkpoints"

exec(open("colab/colab_train.py").read())
```
