# -*- coding: utf-8 -*-
"""
trainer.py
Main training loop for Q-BitNet + LingBot-World v3.
"""

import os
import sys
import json
import time
import threading
import math
from pathlib import Path

# Disable torch dynamo/inductor — BitNet's WeightQuant triggers JIT C++ compilation
# which requires MSVC (cl.exe). Eager mode works fine for training.
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Force UTF-8 + unbuffered output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from model.qbitnet  import QBitNetWrapper
from model.lingbot  import (InteroceptionLayer, HomeostaticController,
                              ArtificialHippocampus, SyntheticAmygdala,
                              FreeEnergyMinimiser)
from optimizer import QuantumTunnelingMuon


# ─────────────────────────────────────────────────────────────────
# WEBSOCKET METRICS EMITTER
# ─────────────────────────────────────────────────────────────────

class MetricsEmitter:
    """Thread-safe queue -> WebSocket bridge."""
    def __init__(self):
        self._queue   = []
        self._lock    = threading.Lock()

    def push(self, data: dict):
        with self._lock:
            self._queue.append(data)

    def drain(self):
        with self._lock:
            q, self._queue = self._queue, []
        return q


EMITTER = MetricsEmitter()


# ─────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────

class TokenisedDataset(IterableDataset):
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.dataset   = load_dataset(
            cfg.DATASET_NAME, split=cfg.DATASET_SPLIT, streaming=True,
        )

    def __iter__(self):
        buf = []
        for example in self.dataset:
            text   = example.get('text', '')
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            buf   += tokens
            while len(buf) >= self.max_len:
                chunk = buf[:self.max_len]
                buf   = buf[self.max_len:]
                ids   = torch.tensor(chunk, dtype=torch.long)
                yield {'input_ids': ids, 'labels': ids}


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────


def save_checkpoint(model, optimizer, epoch, step, metrics, checkpoint_dir):
    path = Path(checkpoint_dir) / f"gen_{epoch}"
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    phi = model.phi_value() if hasattr(model, 'phi_value') else 0.0
    with open(path / "metrics.json", "w") as f:
        json.dump({**metrics, "epoch": epoch, "step": step, "phi": phi,
                   "timestamp": time.time()}, f, indent=2)
    print(f"[OK] Checkpoint saved -> {path}")


# ─────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  Q-BitNet + LingBot-World v3 -- Training")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Load base model
    model_path = cfg.MODEL_LOCAL_DIR
    if not Path(model_path).exists():
        print(f"\n[!] Model not found at {model_path}")
        print("    Run: python train/download_model.py")
        sys.exit(1)

    print(f"\n[>>] Loading base model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype      = torch.bfloat16,
        device_map = "auto" if device == "cuda" else None,
    )
    if device == "cpu":
        base_model = base_model.to(device)

    print("[OK] Base model loaded.")

    # Freeze base model weights to save ~10 GB RAM
    # Only the Q-BitNet additions (phase stream, entanglement, hadamard) will train
    if cfg.FREEZE_BASE_MODEL:
        for p in base_model.parameters():
            p.requires_grad = False
        print("[OK] Base model frozen (training Q-BitNet additions only).")

    # Enable gradient checkpointing to reduce activation memory
    if hasattr(base_model, 'gradient_checkpointing_enable'):
        base_model.gradient_checkpointing_enable()
        print("[OK] Gradient checkpointing enabled.")

    # Wrap with Q-BitNet modifications
    model = QBitNetWrapper(base_model, cfg).to(device)
    print("[OK] Q-BitNet modifications applied.")

    # Brain modules
    d     = base_model.config.hidden_size
    intr  = InteroceptionLayer(d).to(device)
    hctrl = HomeostaticController()
    hipp  = ArtificialHippocampus(d, cfg.HIPPOCAMPUS_BUFFER).to(device)
    amyg  = SyntheticAmygdala(d).to(device)
    fe    = FreeEnergyMinimiser(d).to(device)
    print("[OK] Brain modules ready.")

    # Collect brain module params (used by both optimizers and grad clipping)
    brain_params = (list(intr.parameters()) + list(amyg.parameters()) +
                    list(fe.parameters())   + list(hipp.parameters()))

    # Optimizer 1: Q-BitNet mods (phase stream, entanglement) — Muon
    optimizer = QuantumTunnelingMuon(
        list(model.parameters()),
        lr           = cfg.LEARNING_RATE,
        momentum     = cfg.MOMENTUM,
        tunnel_p0    = cfg.TUNNEL_P0,
        tunnel_decay = cfg.TUNNEL_DECAY,
        tunnel_scale = cfg.TUNNEL_SCALE,
    )
    # Optimizer 2: brain modules (amyg, fe, intr, hipp) — AdamW
    # Muon/Newton-Schulz is wrong for small MLPs/GRUs — AdamW converges reliably
    brain_optimizer = torch.optim.AdamW(brain_params, lr=3e-3, weight_decay=1e-4)

    # Dataset
    print(f"\n[>>] Loading dataset: {cfg.DATASET_NAME} ...")
    dataset = TokenisedDataset(tokenizer, max_len=cfg.MAX_SEQ_LEN)
    loader  = DataLoader(dataset, batch_size=cfg.BATCH_SIZE)
    print("[OK] Dataset ready.")

    Path(cfg.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"\n[>>] Starting training -- {cfg.MAX_EPOCHS} epochs")
    print()

    global_step = 0

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        model.train()
        intr.train(); amyg.train(); fe.train()

        epoch_loss = 0.0
        n_batches  = 0

        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            labels    = batch['labels'].to(device)

            # ── Forward pass (Q-BitNet mods ARE in graph now) ──────
            outputs = model(input_ids=input_ids, labels=labels)
            lm_loss = outputs.loss

            # ── Phi loss: maximize entanglement coupling (grows Phi) 
            phi_loss = model.phi_loss(weight=1e-3)

            # ── Brain modules ──────────────────────────────────────
            # Use the last hidden state from the body (before lm_head)
            hidden  = outputs.hidden_states[-1] if outputs.hidden_states else None
            fe_loss = torch.tensor(0.0, device=device)
            valence = torch.tensor(0.5, device=device)
            arousal = torch.tensor(0.5, device=device)
            amyg_loss = torch.tensor(0.0, device=device)

            if hidden is not None:
                # FE loss — in graph, GRU trains to predict hidden states
                if cfg.USE_FREE_ENERGY:
                    fe_loss = fe(hidden.detach().float()) * 0.1

                # Amygdala — train it: high valence when loss is low
                # Use .detach() so lm_loss and amyg_loss gradients don't fight
                # through shared hidden states. Brain optimizer handles amyg directly.
                if cfg.USE_INTEROCEPTION:
                    valence, arousal = amyg(hidden.detach().float().mean(dim=1))
                    val_target = torch.clamp(
                        torch.exp(-lm_loss.detach()) * torch.ones_like(valence), 0.0, 1.0
                    )
                    amyg_loss = F.mse_loss(valence, val_target) * 0.02  # 0.1 was too strong at lr=3e-3

                    # Homeostatic regulation (no grad needed)
                    with torch.no_grad():
                        intr(hidden.float(), outputs.logits)
                        hctrl.regulate(torch.stack([
                            hidden.var().clamp(0, 1),
                            hidden.abs().mean().clamp(0, 1),
                            valence.detach().clamp(0, 1),
                            torch.tensor(0.5, device=device),
                        ]))
                        hipp.encode(hidden.float().mean(dim=1).mean(0).detach(),
                                    valence.item())

            # ── Total loss ─────────────────────────────────────────
            total_loss = lm_loss + phi_loss + fe_loss + amyg_loss
            (total_loss / cfg.GRAD_ACCUM).backward()

            global_step += 1

            if global_step % cfg.GRAD_ACCUM == 0:
                # Q-BitNet step
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                # Brain step — runs every accumulation cycle, independent of Q-BitNet
                torch.nn.utils.clip_grad_norm_(brain_params, 1.0)
                brain_optimizer.step()
                brain_optimizer.zero_grad()

            epoch_loss += lm_loss.item()
            n_batches  += 1

            # ── Log metrics ────────────────────────────────────────
            if global_step % cfg.LOG_EVERY == 0:
                phi    = model.phi_value()
                fe_val = fe_loss.item() if isinstance(fe_loss, torch.Tensor) else 0.0
                phi_val= phi_loss.item() if isinstance(phi_loss, torch.Tensor) else 0.0
                metrics = {
                    "step":        global_step,
                    "epoch":       epoch,
                    "loss":        round(lm_loss.item(), 4),
                    "phi":         round(phi, 3),
                    "phi_loss":    round(phi_val, 4),
                    "free_energy": round(fe_val, 4),
                    "valence":     round(valence.item(), 3),
                    "arousal":     round(arousal.item(), 3),
                    "lr_mult":     round(hctrl.lr_multiplier, 3),
                    "tunnels":     len(optimizer.tunnel_log),
                    "hipp_pct":    round(hipp.ptr.item() / cfg.HIPPOCAMPUS_BUFFER * 100, 1),
                }
                EMITTER.push(metrics)
                print(
                    f"  step {global_step:>6} | "
                    f"loss {metrics['loss']:.4f} | "
                    f"Phi {metrics['phi']:.3f} | "
                    f"FE {metrics['free_energy']:.4f} | "
                    f"val {metrics['valence']:.3f}"
                )


        # End of epoch
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"\n[Epoch {epoch}/{cfg.MAX_EPOCHS}] avg loss = {avg_loss:.4f}")

        if epoch % cfg.SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, epoch, global_step,
                            {"avg_loss": avg_loss, "phi": compute_phi_proxy(model)},
                            cfg.CHECKPOINT_DIR)

        # Hippocampus dream cycle
        if hctrl.should_consolidate:
            batch_mem = hipp.get_consolidation_batch()
            if batch_mem is not None:
                print("  [Dream cycle] Hippocampus consolidating ...")
                EMITTER.push({"event": "dream_cycle", "step": global_step})

    print("\n[OK] Training complete.")


if __name__ == "__main__":
    train()
