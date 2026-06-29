# ============================================================
#  Q-BitNet + LingBot-World v3  —  Cloud Training Script
#  Continuous training for Google Colab with auto-resume
#  Supports: T4 GPU, A100 GPU, and TPU v3-8
#
#  FEATURES:
#    • Checkpoints saved to Google Drive (survives disconnects)
#    • Full state save: model + optimizer + brain modules + epoch + step
#    • Auto-resume from latest checkpoint on restart
#    • Periodic saves every N steps (not just per epoch)
#    • Keep-alive thread to prevent Colab idle disconnects
#    • Crash recovery: auto-retries on CUDA OOM or runtime errors
#    • TPU support via PyTorch/XLA (auto-detected)
#
#  USAGE:
#    1. Open Google Colab → Runtime → Change runtime → T4 GPU or TPU
#    2. Upload stark/ to Google Drive (or clone from GitHub)
#    3. Run the colab_train.ipynb notebook, or:
#       !python colab/colab_train.py
#
#  TPU SETUP (if using TPU runtime):
#    Before running this script, install torch_xla:
#      !pip install -q cloud-tpu-client torch-xla==2.5.0
#    (Match the torch-xla version to your torch version)
# ============================================================

import sys, os, time, json, math, signal, traceback, threading
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
is_colab = "google.colab" in sys.modules or os.path.exists("/content")

# Import XLA if TPU is available
if is_tpu:
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        device = xm.xla_device()
        bf16_ok = True  # TPU v3+ supports bfloat16 natively
        print(f"[>>] TPU detected: {device}")
    except ImportError:
        print("[!] TPU detected but torch_xla not installed!")
        print("    Run: !pip install -q cloud-tpu-client torch-xla==2.5.0")
        print("    (Match version to your torch version)")
        sys.exit(1)
elif torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    bf16_ok  = torch.cuda.is_bf16_supported()
    print(f"[>>] GPU: {gpu_name} ({gpu_mem:.1f} GB) | bf16={'yes' if bf16_ok else 'no (using fp16)'}")
else:
    bf16_ok  = False
    print("[>>] CPU only — consider enabling GPU or TPU in runtime settings")

# ── Cloud-optimised config ───────────────────────────────────
class CloudConfig:
    # Model
    MODEL_HF_ID         = "microsoft/BitNet-b1.58-2B-4T"
    MODEL_LOCAL_DIR     = "./models/bitnet-2b4t-bf16"

    # Dataset
    DATASET_NAME        = "roneneldan/TinyStories"

    # GPU hyperparams
    BATCH_SIZE          = 4
    MAX_SEQ_LEN         = 256
    GRAD_ACCUM          = 2

    # Training
    MAX_EPOCHS          = 100       # high — continuous training
    LEARNING_RATE       = 2e-4
    MOMENTUM            = 0.95
    LOG_EVERY           = 10
    CHECKPOINT_EVERY    = 200       # save every N steps (frequent saves)

    # Checkpoints — Google Drive path for persistence
    # When running on Colab with Drive mounted, this persists across sessions
    DRIVE_MOUNT         = "/content/drive/MyDrive"
    CHECKPOINT_DIR      = None      # set in setup(), defaults to Drive on Colab

    # Q-BitNet mods
    FREEZE_BASE_MODEL   = True
    USE_SUPERPOSITION   = True
    USE_ENTANGLEMENT    = True
    USE_PHASE_STREAM    = True
    USE_HADAMARD        = False
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

    # Continuous training
    KEEP_MAX_CHECKPOINTS = 5        # keep last N checkpoints, delete older
    RETRY_DELAY          = 30       # seconds to wait before retry after crash
    MAX_RETRIES          = 10       # max consecutive retries before giving up

cfg = CloudConfig()

# ── Setup: mount Drive, set checkpoint dir ──────────────────
def setup():
    """Mount Google Drive (if on Colab) and set checkpoint directory."""
    if is_colab:
        drive_path = Path(cfg.DRIVE_MOUNT)
        if not drive_path.exists():
            print("[>>] Mounting Google Drive ...")
            from google.colab import drive
            drive.mount(cfg.DRIVE_MOUNT)
        cfg.CHECKPOINT_DIR = str(drive_path / "stark_checkpoints")
        print(f"[OK] Checkpoints will be saved to Google Drive: {cfg.CHECKPOINT_DIR}")
    else:
        cfg.CHECKPOINT_DIR = "./checkpoints"
        print(f"[OK] Checkpoints will be saved locally: {cfg.CHECKPOINT_DIR}")

    Path(cfg.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

# ── Keep-alive thread ───────────────────────────────────────
class KeepAliveThread(threading.Thread):
    """Periodically prints to prevent Colab idle disconnect (90 min timeout)."""
    def __init__(self, interval=60):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop = threading.Event()

    def run(self):
        while not self._stop.wait(self.interval):
            print(f"[keep-alive] {time.strftime('%H:%M:%S')} — session active")

    def stop(self):
        self._stop.set()

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
        ignore_patterns= ["*.gguf", "*.bin"],
    )
    print("[OK] Model downloaded.")
    return str(local)

# ── Import train modules from project ───────────────────────
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

# ── Mixed-precision setup ────────────────────────────────────
# CUDA: use AMP with GradScaler (fp16 on T4, bf16 on A100)
# TPU:  native bfloat16 — no GradScaler needed, no autocast needed
# CPU:  float32 only
if device == "cuda":
    use_amp   = True
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    scaler    = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
elif is_tpu:
    use_amp   = False  # TPU doesn't use torch.amp — it runs bf16 natively
    amp_dtype = torch.bfloat16  # TPU v3 supports bf16
    scaler    = None
else:
    use_amp   = False
    amp_dtype = torch.float32
    scaler    = None

print(f"[>>] Precision: {amp_dtype} | AMP={use_amp} | scaler={'yes' if scaler else 'no'}")

# ── Checkpoint management ────────────────────────────────────

def save_full_checkpoint(model, optimizer, brain_optimizer,
                         intr, amyg, fe, hipp,
                         epoch, global_step, extra_metrics=None):
    """Save complete training state so we can resume exactly."""
    ckpt_dir = Path(cfg.CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"ckpt_step_{global_step:08d}.pt"

    state = {
        "model_state":         model.state_dict(),
        "optimizer_state":     optimizer.state_dict(),
        "brain_optimizer_state": brain_optimizer.state_dict(),
        "intr_state":          intr.state_dict(),
        "amyg_state":          amyg.state_dict(),
        "fe_state":            fe.state_dict(),
        "hipp_state":          hipp.state_dict(),
        "optimizer_step_count": optimizer.step_count,
        "optimizer_tunnel_log": optimizer.tunnel_log,
        "epoch":               epoch,
        "global_step":         global_step,
        "phi":                 model.phi_value(),
        "timestamp":           time.time(),
    }
    if extra_metrics:
        state["metrics"] = extra_metrics

    torch.save(state, ckpt_path)
    print(f"[OK] Full checkpoint saved -> {ckpt_path}")

    # Update latest pointer
    latest_file = ckpt_dir / "LATEST"
    latest_file.write_text(ckpt_path.name)

    # Clean up old checkpoints
    _cleanup_old_checkpoints(ckpt_dir)
    return ckpt_path


def find_latest_checkpoint():
    """Find the most recent checkpoint file."""
    ckpt_dir = Path(cfg.CHECKPOINT_DIR)
    if not ckpt_dir.exists():
        return None

    # Try LATEST pointer first
    latest_file = ckpt_dir / "LATEST"
    if latest_file.exists():
        name = latest_file.read_text().strip()
        ckpt_path = ckpt_dir / name
        if ckpt_path.exists():
            return ckpt_path

    # Fallback: scan for checkpoint files
    ckpts = sorted(ckpt_dir.glob("ckpt_step_*.pt"))
    return ckpts[-1] if ckpts else None


def load_checkpoint(model, optimizer, brain_optimizer,
                    intr, amyg, fe, hipp, ckpt_path):
    """Load full training state from checkpoint."""
    print(f"[>>] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    brain_optimizer.load_state_dict(state["brain_optimizer_state"])
    intr.load_state_dict(state["intr_state"])
    amyg.load_state_dict(state["amyg_state"])
    fe.load_state_dict(state["fe_state"])
    hipp.load_state_dict(state["hipp_state"])

    optimizer.step_count = state.get("optimizer_step_count", 0)
    optimizer.tunnel_log = state.get("optimizer_tunnel_log", [])

    epoch       = state["epoch"]
    global_step = state["global_step"]
    phi         = state.get("phi", 0.0)

    print(f"[OK] Resumed from epoch {epoch}, step {global_step}, Phi={phi:.4f}")
    return epoch, global_step


def _cleanup_old_checkpoints(ckpt_dir):
    """Keep only the last KEEP_MAX_CHECKPOINTS checkpoint files."""
    ckpts = sorted(ckpt_dir.glob("ckpt_step_*.pt"))
    if len(ckpts) <= cfg.KEEP_MAX_CHECKPOINTS:
        return
    to_delete = ckpts[:-cfg.KEEP_MAX_CHECKPOINTS]
    for f in to_delete:
        f.unlink()
        print(f"  [cleanup] Removed old checkpoint: {f.name}")


# ── Training loop ────────────────────────────────────────────

def build_model_and_optimizers():
    """Build model, brain modules, and optimizers. Returns all components."""
    model_path = ensure_model()

    tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype      = amp_dtype,
        device_map = "auto" if device == "cuda" else None,
    )
    # TPU and CPU: manually move model to device (device_map="auto" is CUDA-only)
    if device != "cuda":
        base_model = base_model.to(device)

    if cfg.FREEZE_BASE_MODEL:
        for p in base_model.parameters():
            p.requires_grad = False
    # Gradient checkpointing — CUDA only (not supported on TPU)
    if device == "cuda":
        base_model.gradient_checkpointing_enable()

    d = base_model.config.hidden_size
    model = QBitNetWrapper(base_model, cfg).to(device)

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

    return (model, optimizer, brain_optimizer, intr, hctrl, hipp, amyg, fe,
            brain_params, tokenizer)


def train_epoch(model, optimizer, brain_optimizer, intr, hctrl, hipp, amyg, fe,
                brain_params, tokenizer, epoch, global_step):
    """Run one epoch of training. Returns (global_step, epoch_loss, n_batches)."""
    model.train(); intr.train(); amyg.train(); fe.train()

    dataset = TokenisedDataset(tokenizer, cfg.MAX_SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=cfg.BATCH_SIZE)

    epoch_loss = 0.0
    n_batches  = 0
    valence = torch.tensor(0.5, device=device)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        # ── Forward pass ──────────────────────────────────────
        # CUDA: use torch.amp.autocast | TPU/CPU: no autocast (bf16 is native on TPU)
        if use_amp:
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
        else:
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

        # ── Backward pass ─────────────────────────────────────
        if scaler is not None:
            scaler.scale(total_loss / cfg.GRAD_ACCUM).backward()
        else:
            (total_loss / cfg.GRAD_ACCUM).backward()

        global_step += 1
        epoch_loss  += lm_loss.item()
        n_batches   += 1

        if global_step % cfg.GRAD_ACCUM == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(brain_params, 1.0)
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            elif is_tpu:
                # TPU: use xm.optimizer_step which handles mark_step internally
                xm.optimizer_step(optimizer)
                xm.optimizer_step(brain_optimizer)
            else:
                optimizer.step()
                brain_optimizer.step()
            optimizer.zero_grad()
            brain_optimizer.zero_grad()

        if global_step % cfg.LOG_EVERY == 0:
            phi = model.phi_value()
            fe_val = fe_loss.item()
            val_v  = valence.item() if hidden is not None else 0.0
            print(f"  step {global_step:>6} | loss {lm_loss.item():.4f} | "
                  f"Phi {phi:.3f} | FE {fe_val:.4f} | val {val_v:.3f}")

        # Periodic checkpoint
        if global_step % cfg.CHECKPOINT_EVERY == 0:
            save_full_checkpoint(model, optimizer, brain_optimizer,
                                 intr, amyg, fe, hipp,
                                 epoch, global_step,
                                 {"avg_loss_so_far": epoch_loss / max(n_batches, 1)})

    return global_step, epoch_loss, n_batches


def train():
    """Main training entry point with auto-resume and crash recovery."""
    setup()

    keep_alive = KeepAliveThread(interval=60)
    keep_alive.start()

    retry_count = 0

    while retry_count < cfg.MAX_RETRIES:
        try:
            # Build everything fresh (handles CUDA OOM recovery by releasing old state)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif is_tpu:
                xm.wait_device_ops()  # ensure previous TPU ops are done before rebuild

            (model, optimizer, brain_optimizer, intr, hctrl, hipp, amyg, fe,
             brain_params, tokenizer) = build_model_and_optimizers()

            # Try to resume from checkpoint
            start_epoch = 1
            global_step = 0

            ckpt = find_latest_checkpoint()
            if ckpt is not None:
                start_epoch, global_step = load_checkpoint(
                    model, optimizer, brain_optimizer,
                    intr, amyg, fe, hipp, ckpt)
                start_epoch += 1  # continue from next epoch
                retry_count = 0   # successful resume, reset retry counter
            else:
                print("[>>] No checkpoint found — starting fresh.")

            print(f"\n[>>] Starting training from epoch {start_epoch}, step {global_step}")
            print(f"     Max epochs: {cfg.MAX_EPOCHS} (continuous)\n")

            for epoch in range(start_epoch, cfg.MAX_EPOCHS + 1):
                global_step, epoch_loss, n_batches = train_epoch(
                    model, optimizer, brain_optimizer,
                    intr, hctrl, hipp, amyg, fe,
                    brain_params, tokenizer, epoch, global_step)

                avg = epoch_loss / max(n_batches, 1)
                print(f"\n[Epoch {epoch}/{cfg.MAX_EPOCHS}] avg loss = {avg:.4f} | "
                      f"total steps = {global_step}\n")

                # Save end-of-epoch checkpoint
                save_full_checkpoint(model, optimizer, brain_optimizer,
                                     intr, amyg, fe, hipp,
                                     epoch, global_step,
                                     {"avg_loss": avg})

            print("\n[OK] Training complete (reached MAX_EPOCHS).")
            keep_alive.stop()
            return

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[!] CUDA OOM at step {global_step}. Clearing cache and retrying...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(cfg.RETRY_DELAY)
                retry_count += 1
                continue
            else:
                print(f"\n[!] Runtime error: {e}")
                traceback.print_exc()
                time.sleep(cfg.RETRY_DELAY)
                retry_count += 1
                continue

        except KeyboardInterrupt:
            print("\n[!] Interrupted by user — saving checkpoint before exit...")
            try:
                save_full_checkpoint(model, optimizer, brain_optimizer,
                                     intr, amyg, fe, hipp,
                                     epoch, global_step)
            except Exception:
                pass
            keep_alive.stop()
            return

        except Exception as e:
            print(f"\n[!] Unexpected error: {e}")
            traceback.print_exc()
            time.sleep(cfg.RETRY_DELAY)
            retry_count += 1
            continue

    print(f"\n[!] Exceeded max retries ({cfg.MAX_RETRIES}). "
          f"Run again to resume from last checkpoint.")
    keep_alive.stop()


if __name__ == "__main__":
    train()
