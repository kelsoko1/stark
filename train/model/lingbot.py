"""
model/lingbot.py
LingBot-World v3 — AI brain modules:
  InteroceptionLayer, HomeostaticController,
  ArtificialHippocampus, SyntheticAmygdala, FreeEnergyMinimiser
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# INTEROCEPTION LAYER — reads own activations
# ─────────────────────────────────────────────────────────────────

class InteroceptionLayer(nn.Module):
    """
    Samples the model's own internal activations as synthetic body signals.
    No external sensor. Outputs a 4-D internal state vector:
      [arousal, load, valence, novelty]
    """
    def __init__(self, d_model=2048, n_layers=32):
        super().__init__()
        self.register_buffer('running_mean', torch.zeros(d_model))
        self.register_buffer('running_var',  torch.ones(d_model))
        self.to_state = nn.Sequential(
            nn.Linear(d_model, 128), nn.SiLU(),
            nn.Linear(128, 4),      nn.Sigmoid()
        )

    @torch.no_grad()
    def forward(self, hidden_states, logits=None):
        """
        hidden_states: [B, T, D]  — final layer hidden states
        logits:        [B, T, V]  — optional, used for valence/confidence
        Returns: state_vec [4]  (arousal, load, valence, novelty)
        """
        h = hidden_states.float()

        # Arousal: variance of hidden activations (high = active/stressed)
        arousal_raw = h.var(dim=-1).mean()

        # Load: entropy of mean activation magnitudes
        h_mean = h.abs().mean(dim=[0, 1])

        # Novelty: KL from running mean
        novelty_raw = ((h_mean - self.running_mean) ** 2 / (self.running_var + 1e-6)).mean()

        # Update running stats (exponential moving average)
        self.running_mean = 0.99 * self.running_mean + 0.01 * h_mean.detach()
        self.running_var  = 0.99 * self.running_var  + 0.01 * (h_mean - self.running_mean).pow(2).detach()

        # Valence from logit confidence (if available)
        if logits is not None:
            probs   = F.softmax(logits.float(), dim=-1)
            valence_raw = probs.max(dim=-1).values.mean()
        else:
            valence_raw = torch.tensor(0.5)

        state_vec = torch.stack([
            arousal_raw.clamp(0, 1),
            h_mean.mean().clamp(0, 1),
            valence_raw.clamp(0, 1),
            novelty_raw.clamp(0, 1),
        ])
        return state_vec


# ─────────────────────────────────────────────────────────────────
# HOMEOSTATIC CONTROLLER — self-regulation
# ─────────────────────────────────────────────────────────────────

class HomeostaticController:
    """
    Rule-based homeostatic controller.
    Reads the 4-D interoception state and adjusts training hyperparameters:
      - temperature / top_p
      - learning rate multiplier
      - consolidation trigger
    """
    def __init__(self):
        self.temp_multiplier = 1.0
        self.lr_multiplier   = 1.0
        self.should_consolidate = False

    def regulate(self, state_vec):
        """
        state_vec: [arousal, load, valence, novelty]
        Returns dict of adjustments.
        """
        arousal, load, valence, novelty = state_vec.tolist()

        # High arousal + high load → slow down
        if arousal > 0.8 and load > 0.7:
            self.lr_multiplier   = max(0.5, self.lr_multiplier * 0.95)
            self.temp_multiplier = max(0.5, self.temp_multiplier * 0.9)

        # Low valence + low novelty → trigger consolidation (dream cycle)
        elif valence < 0.3 and novelty < 0.2:
            self.should_consolidate = True
            self.lr_multiplier = min(1.5, self.lr_multiplier * 1.05)

        # High novelty → boost exploration
        elif novelty > 0.7:
            self.temp_multiplier = min(2.0, self.temp_multiplier * 1.1)
            self.should_consolidate = False

        else:
            # Return toward baseline slowly
            self.temp_multiplier = 0.95 * self.temp_multiplier + 0.05 * 1.0
            self.lr_multiplier   = 0.95 * self.lr_multiplier   + 0.05 * 1.0
            self.should_consolidate = False

        return {
            'temp_multiplier': self.temp_multiplier,
            'lr_multiplier':   self.lr_multiplier,
            'consolidate':     self.should_consolidate,
        }


# ─────────────────────────────────────────────────────────────────
# ARTIFICIAL HIPPOCAMPUS — episodic memory
# ─────────────────────────────────────────────────────────────────

class ArtificialHippocampus(nn.Module):
    """
    Circular episodic buffer. Episodes encoded as mean hidden states.
    Retrieval via cosine similarity (phase-matched).
    Consolidation: top-K by valence replayed as additional loss.
    """
    def __init__(self, d_model=2048, buffer_size=2048):
        super().__init__()
        self.buffer_size = buffer_size
        self.register_buffer('keys',    torch.zeros(buffer_size, d_model))
        self.register_buffer('vals',    torch.zeros(buffer_size, d_model))
        self.register_buffer('valence', torch.zeros(buffer_size))
        self.register_buffer('ptr',     torch.tensor(0, dtype=torch.long))
        self.register_buffer('full',    torch.tensor(False))

    @torch.no_grad()
    def encode(self, hidden_state, valence_score: float):
        """Store one episode."""
        i = self.ptr.item()
        self.keys[i]    = hidden_state.mean(dim=0) if hidden_state.dim() > 1 else hidden_state
        self.vals[i]    = self.keys[i]
        self.valence[i] = valence_score
        self.ptr        = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0:
            self.full = torch.tensor(True)

    @torch.no_grad()
    def retrieve(self, query, top_k=8):
        """Retrieve top-K episodes by cosine similarity to query."""
        n       = self.buffer_size if self.full else self.ptr.item()
        if n == 0:
            return None, None
        keys    = self.keys[:n]
        q_norm  = F.normalize(query.mean(dim=0, keepdim=True), dim=-1)
        k_norm  = F.normalize(keys, dim=-1)
        scores  = (k_norm @ q_norm.T).squeeze(-1)
        top_k   = min(top_k, n)
        idxs    = scores.topk(top_k).indices
        return self.vals[idxs], self.valence[idxs]

    @torch.no_grad()
    def get_consolidation_batch(self, top_k=128):
        """Return highest-valence episodes for replay."""
        n = self.buffer_size if self.full else self.ptr.item()
        if n == 0:
            return None
        top_k = min(top_k, n)
        idxs  = self.valence[:n].topk(top_k).indices
        return self.vals[idxs]


# ─────────────────────────────────────────────────────────────────
# SYNTHETIC AMYGDALA — valence/arousal tagging
# ─────────────────────────────────────────────────────────────────

class SyntheticAmygdala(nn.Module):
    """
    Tags each internal state with valence and arousal scores.
    Provides the intrinsic reward signal for self-improvement.
    Trained implicitly: high valence = high Phi = good.
    """
    def __init__(self, d_model=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256), nn.SiLU(),
            nn.Linear(256, 64),      nn.SiLU(),
            nn.Linear(64, 2),        nn.Sigmoid()   # [valence, arousal]
        )

    def forward(self, hidden_state):
        """hidden_state: [D] or [B, D]"""
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)
        va = self.net(hidden_state)
        valence, arousal = va[:, 0].mean(), va[:, 1].mean()
        return valence, arousal


# ─────────────────────────────────────────────────────────────────
# FREE ENERGY MINIMISER
# ─────────────────────────────────────────────────────────────────

class FreeEnergyMinimiser(nn.Module):
    """
    Approximates Karl Friston's free energy as:
      FE = prediction_error + KL(current_state | expected_state)
    Used as an auxiliary training loss term.
    """
    def __init__(self, d_model=2048):
        super().__init__()
        self.predictor = nn.GRU(d_model, d_model, batch_first=True)
        self.register_buffer('prev_state', None)

    def forward(self, hidden_states):
        """
        hidden_states: [B, T, D]
        Returns scalar free energy estimate.
        """
        B, T, D = hidden_states.shape

        if self.prev_state is None or self.prev_state.shape[0] != B:
            self.prev_state = torch.zeros(1, B, D, device=hidden_states.device)

        predicted, new_state = self.predictor(hidden_states, self.prev_state)
        self.prev_state      = new_state.detach()

        prediction_error = F.mse_loss(predicted, hidden_states.detach())
        mu               = hidden_states.mean(dim=1)
        kl               = 0.5 * (mu.pow(2) + hidden_states.var(dim=1) - 1).mean()
        free_energy      = prediction_error + 0.1 * kl.clamp(min=0)
        return free_energy
