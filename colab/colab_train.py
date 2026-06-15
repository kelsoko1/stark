# ============================================================
#  Q-BitNet + LingBot-World v3  —  Cloud Training Script
#  Works on: Google Colab (T4/A100), Kaggle (T4/P100/TPU)
#
#  USAGE:
#    1. Open Google Colab → Runtime → Change runtime → T4 GPU
#    2. Upload this file (or paste cell-by-cell into a notebook)
#    3. Run all cells in order
# ============================================================

# ── CELL 1: Install requirements ────────────────────────────
# !pip install -q transformers>=4.45 datasets torch sentencepiece

# ── CELL 2: Clone project from GitHub (or upload manually) ──
# If you've pushed stark/ to GitHub:
# !git clone https://github.com/YOUR_USERNAME/stark.git
# %cd stark
#
# OR upload your stark/ folder to Google Drive and mount it:
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/stark

# ── CELL 3: All training code (self-contained) ──────────────
import sys, os, time, json, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

# ── Detect environment ──────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
is_tpu = "COLAB_TPU_ADDR" in os.environ

if is_tpu:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    print(f"[>>] TPU detected: {device}")
elif torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    bf16_ok  = torch.cuda.is_bf16_supported()
    print(f"[>>] GPU: {gpu_name} ({gpu_mem:.1f} GB) | bf16={'yes' if bf16_ok else 'no (using fp16)'}")
else:
    bf16_ok  = False
    print("[>>] CPU only — consider enabling GPU in runtime settings")

# ── Cloud-optimised config ───────────────────────────────────
class CloudConfig:
    # Model — download directly from HuggingFace (no local path needed)
    MODEL_HF_ID         = "microsoft/BitNet-b1.58-2B-4T"
    MODEL_LOCAL_DIR     = "./models/bitnet-2b4t-bf16"   # cached after first run

    # Dataset
    DATASET_NAME        = "roneneldan/TinyStories"

    # GPU hyperparams (10-50x better than CPU settings)
    BATCH_SIZE          = 4      # was 1 on CPU
    MAX_SEQ_LEN         = 256    # was 128 on CPU
    GRAD_ACCUM          = 2      # effective batch = 4×2 = 8

    # Training
    MAX_EPOCHS          = 10
    LEARNING_RATE       = 2e-4
    MOMENTUM            = 0.95
    LOG_EVERY           = 1
    CHECKPOINT_DIR      = "./checkpoints"
    CHECKPOINT_EVERY    = 1       # save every epoch

    # Q-BitNet mods
    FREEZE_BASE_MODEL   = True
    USE_SUPERPOSITION   = True
    USE_ENTANGLEMENT    = True
    USE_PHASE_STREAM    = True
    USE_HADAMARD        = False   # disabled (FWHT shape bug)
    USE_TUNNELING       = True
    ENTANGLE_TOP_K      = 6

    # Brain modules
    USE_FREE_ENERGY     = True
    USE_INTEROCEPTION   = True
    USE_HIPPOCAMPUS     = True
    HIPPOCAMPUS_BUFFER  = 512

    # Optimizer
    TUNNEL_P0           = 0.02
    TUNNEL_DECAY        = 5000
    TUNNEL_SCALE        = 8.0

cfg = CloudConfig()

# ── Auto-download model if not cached ───────────────────────
def ensure_model():
    local = Path(cfg.MODEL_LOCAL_DIR)
    if local.exists() and any(local.iterdir()):
        print(f"[OK] Using cached model at {local}")
        return str(local)
    print(f"[>>] Downloading {cfg.MODEL_HF_ID} from HuggingFace ...")
    print("     This takes ~5 min on a fast connection (~4.8 GB)")
    local.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id        = cfg.MODEL_HF_ID,
        local_dir      = str(local),
        ignore_patterns= ["*.gguf", "*.bin"],   # prefer .safetensors
    )
    print("[OK] Model downloaded.")
    return str(local)

# ── Import train modules from project ───────────────────────
# (assumes you cloned stark/ and are in stark/)
sys.path.insert(0, ".")
from train.model.qbitnet   import QBitNetWrapper
from train.model.lingbot   import (InteroceptionLayer, HomeostaticController,
                                    ArtificialHippocampus, SyntheticAmygdala,
                                    FreeEnergyMinimiser)
from train.optimizer        import QuantumTunnelingMuon

# ── Dataset ──────────────────────────────────────────────────
class TokenisedDataset(IterableDataset):
    def __init__(self, tokenizer, max_len=256):
        self.tok = tokenizer
        self.max_len = max_len
        self.ds = load_dataset(cfg.DATASET_NAME, split="train", streaming=True)

    def __iter__(self):
        buf = []
        for ex in self.ds:
            ids = self.tok(ex["text"], truncation=False)["input_ids"]
            buf.extend(ids)
            while len(buf) >= self.max_len:
                chunk = buf[:self.max_len]
                buf   = buf[self.max_len:]
                t = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": t, "labels": t.clone()}

# ── Mixed-precision scaler ───────────────────────────────────
use_amp   = device == "cuda"
# Use bfloat16 on Ampere+ (A100, H100), float16 on Turing (T4)
amp_dtype = (torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported())
             else torch.float16 if use_amp
             else torch.float32)
scaler    = torch.amp.GradScaler("cuda") if use_amp else None

# ── Training loop ────────────────────────────────────────────
def train():
    model_path = ensure_model()

    tokenizer  = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype      = amp_dtype,          # float16 on T4, bfloat16 on A100
        device_map = "auto" if device == "cuda" else None,
    )
    if device != "cuda":
        base_model = base_model.to(device)

    if cfg.FREEZE_BASE_MODEL:
        for p in base_model.parameters():
            p.requires_grad = False
    if device == "cuda":
        base_model.gradient_checkpointing_enable()

    d = base_model.config.hidden_size
    model = QBitNetWrapper(base_model, cfg).to(device)

    # Brain modules
    intr  = InteroceptionLayer(d).to(device)
    hctrl = HomeostaticController()
    hipp  = ArtificialHippocampus(d, cfg.HIPPOCAMPUS_BUFFER).to(device)
    amyg  = SyntheticAmygdala(d).to(device)
    fe    = FreeEnergyMinimiser(d).to(device)

    brain_params = (list(intr.parameters()) + list(amyg.parameters()) +
                    list(fe.parameters())   + list(hipp.parameters()))

    optimizer = QuantumTunnelingMuon(
        list(model.parameters()),
        lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM,
        tunnel_p0=cfg.TUNNEL_P0, tunnel_decay=cfg.TUNNEL_DECAY,
        tunnel_scale=cfg.TUNNEL_SCALE,
    )
    brain_optimizer = torch.optim.AdamW(brain_params, lr=3e-3, weight_decay=1e-4)

    dataset = TokenisedDataset(tokenizer, cfg.MAX_SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=cfg.BATCH_SIZE)

    Path(cfg.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        model.train(); intr.train(); amyg.train(); fe.train()
        epoch_loss = 0.0; n = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                outputs  = model(input_ids=input_ids, labels=labels)
                lm_loss  = outputs.loss
                phi_loss = model.phi_loss(weight=1e-3)

                hidden = outputs.hidden_states[-1] if outputs.hidden_states else None
                fe_loss = amyg_loss = torch.tensor(0., device=device)

                if hidden is not None:
                    if cfg.USE_FREE_ENERGY:
                        fe_loss = fe(hidden.detach().float()) * 0.1
                    if cfg.USE_INTEROCEPTION:
                        valence, arousal = amyg(hidden.detach().float().mean(dim=1))
                        val_target = torch.clamp(
                            torch.exp(-lm_loss.detach()) * torch.ones_like(valence), 0., 1.)
                        amyg_loss = F.mse_loss(valence, val_target) * 0.02

                total_loss = lm_loss + phi_loss + fe_loss + amyg_loss

            (total_loss / cfg.GRAD_ACCUM).backward() if not use_amp else \
            scaler.scale(total_loss / cfg.GRAD_ACCUM).backward()

            global_step += 1
            epoch_loss  += lm_loss.item(); n += 1

            if global_step % cfg.GRAD_ACCUM == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(brain_params, 1.0)
                if use_amp:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                brain_optimizer.step(); brain_optimizer.zero_grad()

            if global_step % cfg.LOG_EVERY == 0:
                phi = model.phi_value()
                fe_val = fe_loss.item()
                val_v  = valence.item() if hidden is not None else 0.0
                print(f"  step {global_step:>6} | loss {lm_loss.item():.4f} | "
                      f"Phi {phi:.3f} | FE {fe_val:.4f} | val {val_v:.3f}")

        avg = epoch_loss / max(n, 1)
        print(f"\n[Epoch {epoch}/{cfg.MAX_EPOCHS}] avg loss = {avg:.4f}")
        path = Path(cfg.CHECKPOINT_DIR) / f"gen_{epoch}"
        path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path / "model.pt")
        print(f"[OK] Checkpoint saved -> {path}\n")

if __name__ == "__main__":
    train()
