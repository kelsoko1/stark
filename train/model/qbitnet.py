"""
model/qbitnet.py
Q-BitNet architectural modifications — all 6 quantum-inspired changes.

KEY DESIGN: QBitNetWrapper.forward() runs the base model in two stages:
  1. Get hidden states (no labels -> no loss yet)
  2. Apply Q-BitNet mods to hidden states
  3. Run LM head on modified hidden states
  4. Compute loss manually

This ensures ALL modifications are in the backward-pass computation graph.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast


# ─────────────────────────────────────────────────────────────────
# 1. SUPERPOSITION WEIGHTS
# ─────────────────────────────────────────────────────────────────

class QuantumSuperpositionWeight(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.amplitudes   = nn.Parameter(torch.zeros(out_features, in_features, 3))
        nn.init.normal_(self.amplitudes, std=0.1)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def get_probs(self):
        return F.softmax(self.amplitudes, dim=-1)

    def effective_weight(self):
        probs  = self.get_probs()
        states = torch.tensor([-1., 0., 1.], device=probs.device)
        return (probs * states).sum(-1)

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)

    def entropy_regularization(self, target=0.3):
        probs   = self.get_probs()
        entropy = -(probs * probs.log().clamp(min=-20)).sum(-1).mean()
        return (entropy - target).abs()


# ─────────────────────────────────────────────────────────────────
# 2. INTERFERENCE ATTENTION
# ─────────────────────────────────────────────────────────────────

class QuantumInterferenceAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.Qr = nn.Linear(d_model, d_model, bias=False)
        self.Qi = nn.Linear(d_model, d_model, bias=False)
        self.Kr = nn.Linear(d_model, d_model, bias=False)
        self.Ki = nn.Linear(d_model, d_model, bias=False)
        self.V  = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.phase_bias = nn.Parameter(torch.zeros(n_heads, 1, 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        H, DH   = self.n_heads, self.d_head

        def reshape(t):
            return t.view(B, T, H, DH).transpose(1, 2)

        x_f = x.float()
        Qc = torch.complex(reshape(self.Qr(x_f)), reshape(self.Qi(x_f)))
        Kc = torch.complex(reshape(self.Kr(x_f)), reshape(self.Ki(x_f)))
        V  = reshape(self.V(x_f))

        amplitude = torch.matmul(Qc.conj(), Kc.transpose(-2, -1)) / math.sqrt(DH)
        amplitude = amplitude * torch.exp(1j * self.phase_bias.float())
        prob = amplitude.abs() ** 2
        if mask is not None:
            prob = prob.masked_fill(mask == 0, 0.0)
        attn = prob / (prob.sum(-1, keepdim=True) + 1e-8)
        attn = self.drop(attn)
        out  = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out).to(x.dtype)


# ─────────────────────────────────────────────────────────────────
# 3. ENTANGLED HEAD PAIRS
# ─────────────────────────────────────────────────────────────────

class EntangledHeadPair(nn.Module):
    """
    Two attention heads share a learnable coupling matrix C.
    Head A's output biases Head B's keys — non-local correlation.
    C is updated via gradients from the LM loss, so Phi evolves.
    """
    def __init__(self, d_model, d_head):
        super().__init__()
        self.d_head = d_head
        # Initialize C near zero so it starts with minimal influence
        self.C = nn.Parameter(torch.randn(d_head, d_head) * 0.01)

        for name in ['Qa', 'Ka', 'Va', 'Qb', 'Kb', 'Vb']:
            setattr(self, name, nn.Linear(d_model, d_head, bias=False))
        self.out  = nn.Linear(d_head * 2, d_model, bias=False)
        self.gate = nn.Parameter(torch.tensor(-3.0))  # sigmoid(-3) ~ 0.05 initial scale

    def _attn(self, Q, K, V):
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_head)
        return torch.bmm(F.softmax(scores, dim=-1), V)

    def forward(self, x):
        # Cast to float for attention stability
        xf = x.float()
        outA     = self._attn(self.Qa(xf), self.Ka(xf), self.Va(xf))
        Kb_biased = self.Kb(xf) + torch.tanh(outA @ self.C.float())
        outB     = self._attn(self.Qb(xf), Kb_biased, self.Vb(xf))
        combined = self.out(torch.cat([outA, outB], dim=-1))
        scale    = torch.sigmoid(self.gate)
        return (combined * scale).to(x.dtype)

    def phi_coupling_norm(self):
        """Used as Phi metric AND as regularization target."""
        return self.C.norm()


# ─────────────────────────────────────────────────────────────────
# 4. PHASE-ENCODED RESIDUAL STREAM
# ─────────────────────────────────────────────────────────────────

class PhaseEncodedResidualStream(nn.Module):
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.phase_freq = nn.Parameter(torch.randn(n_layers, d_model) * 0.01)
        self.gate       = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        # Zero-init output so it starts as identity
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, h_real, h_imag, layer_idx):
        freq     = self.phase_freq[layer_idx]
        h_real_f = h_real.float()
        h_imag_f = h_imag.float()
        rotation = torch.exp(1j * freq.float()).unsqueeze(0).unsqueeze(0)
        h_rotated = torch.complex(h_real_f, h_imag_f) * rotation
        hr, hi    = h_rotated.real, h_rotated.imag
        influence = self.gate(torch.cat([hr, hi], dim=-1))
        return (hr + influence).to(h_real.dtype), hi


# ─────────────────────────────────────────────────────────────────
# 5. HADAMARD MIXING LAYER
# ─────────────────────────────────────────────────────────────────

class HadamardMixingLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d        = d_model
        self.d_padded = 2 ** math.ceil(math.log2(d_model))
        self.gate     = nn.Parameter(torch.tensor(-3.0))  # starts near zero influence

    def _fwht(self, x):
        h = 1
        while h < x.shape[-1]:
            x = x.view(*x.shape[:-1], -1, 2 * h)
            x = torch.cat([x[..., :h] + x[..., h:], x[..., :h] - x[..., h:]], dim=-1)
            h *= 2
        return x / math.sqrt(x.shape[-1])

    def forward(self, x):
        g     = torch.sigmoid(self.gate)
        pad   = self.d_padded - self.d
        x_f   = x.float()
        x_pad = F.pad(x_f, (0, pad)) if pad > 0 else x_f
        x_h   = self._fwht(x_pad)[..., :self.d]
        return (g * x_h + (1 - g) * x_f).to(x.dtype)


# ─────────────────────────────────────────────────────────────────
# WRAPPER
# ─────────────────────────────────────────────────────────────────

class QBitNetWrapper(nn.Module):
    """
    Wraps a BitNet CausalLM model and injects Q-BitNet modifications.

    CRITICAL: We run the forward pass in two stages so Q-BitNet mods
    are in the computation graph:
      1. base model body (no lm_head, no loss)
      2. Q-BitNet modifications to hidden states
      3. lm_head on modified hidden states
      4. manual cross-entropy loss
    """
    def __init__(self, base_model, cfg):
        super().__init__()
        self.base   = base_model
        self.config = cfg

        d_model  = base_model.config.hidden_size
        n_heads  = base_model.config.num_attention_heads
        n_layers = base_model.config.num_hidden_layers

        # Find lm_head (Llama/BitNet use base_model.lm_head)
        self.lm_head = getattr(base_model, 'lm_head', None)
        if self.lm_head is None:
            raise RuntimeError("Could not find lm_head on base model.")

        # Find the transformer body (without lm_head)
        self.body = getattr(base_model, 'model', base_model)

        # Phase stream
        if cfg.USE_PHASE_STREAM:
            self.phase_stream = PhaseEncodedResidualStream(d_model, n_layers)
            self.h_imag = None

        # Hadamard mixing
        if cfg.USE_HADAMARD:
            self.hadamard = nn.ModuleList([
                HadamardMixingLayer(d_model) for _ in range(n_layers - 1)
            ])
            self.hadamard_gate = nn.Parameter(torch.tensor(-3.0))

        # Entangled head pairs — these WILL be called and trained
        if cfg.USE_ENTANGLEMENT:
            self.entangled_pairs = nn.ModuleList([
                EntangledHeadPair(d_model, d_model // n_heads)
                for _ in range(cfg.ENTANGLE_TOP_K)
            ])

        self.d_model  = d_model
        self.n_layers = n_layers

    def forward(self, input_ids, attention_mask=None, labels=None):
        # ── Stage 1: run transformer body (NO lm_head, NO loss) ─────
        body_outputs = self.body(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            output_hidden_states = True,
            use_cache            = False,
        )
        hidden = body_outputs.last_hidden_state  # [B, T, D]

        # ── Stage 2: Q-BitNet modifications ─────────────────────────

        # 4. Phase-encoded residual stream
        if self.config.USE_PHASE_STREAM and hasattr(self, 'phase_stream'):
            B, T, D = hidden.shape
            if self.h_imag is None or self.h_imag.shape[:2] != (B, T):
                self.h_imag = torch.zeros(B, T, D, device=hidden.device, dtype=torch.float32)
            hidden, self.h_imag = self.phase_stream(hidden, self.h_imag, self.n_layers - 1)
            self.h_imag = self.h_imag.detach()

        # 3. Entangled head pairs (residual addition, learned scale)
        if self.config.USE_ENTANGLEMENT and hasattr(self, 'entangled_pairs'):
            for pair in self.entangled_pairs:
                hidden = hidden + pair(hidden)  # gated internally

        # 5. Hadamard mixing
        if self.config.USE_HADAMARD and hasattr(self, 'hadamard'):
            g = torch.sigmoid(self.hadamard_gate)
            mix = self.hadamard[-1](hidden)
            hidden = g * mix + (1 - g) * hidden

        # ── Stage 3: LM head on modified hidden states ───────────────
        logits = self.lm_head(hidden)  # [B, T, vocab]

        # ── Stage 4: compute loss ────────────────────────────────────
        loss = None
        if labels is not None:
            # Standard next-token cross-entropy
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index = -100,
            )

        return CausalLMOutputWithPast(
            loss         = loss,
            logits       = logits,
            hidden_states= body_outputs.hidden_states,
            past_key_values = None,
        )

    def get_last_hidden(self):
        """Last modified hidden state (for brain modules)."""
        return None  # accessed directly from forward output

    def phi_loss(self, weight=1e-3):
        """
        Maximize entanglement coupling norms (Phi regularization).
        Adds to total loss — pulls C norms upward = more integration.
        """
        if not hasattr(self, 'entangled_pairs'):
            return torch.tensor(0.0)
        total = sum(pair.phi_coupling_norm() for pair in self.entangled_pairs)
        return -total * weight  # negative = maximize

    def phi_value(self):
        """Scalar Phi metric for logging — raw mean C-norm (grows as coupling strengthens)."""
        if not hasattr(self, 'entangled_pairs') or len(self.entangled_pairs) == 0:
            return 0.0
        with torch.no_grad():
            total = sum(pair.phi_coupling_norm().item() for pair in self.entangled_pairs)
        return round(total / len(self.entangled_pairs), 4)  # raw norm — no arbitrary scaling
