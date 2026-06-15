"""
optimizer.py
QuantumTunnelingMuon — Muon optimizer with quantum tunneling events.
Tunneling = rare, large gradient steps that escape local minima.
"""

import math
import torch


class QuantumTunnelingMuon(torch.optim.Optimizer):
    """
    Muon (Momentum + Newton-Schulz orthogonalisation) augmented with
    quantum tunneling: probabilistic large steps that bypass clipping,
    allowing escape from local minima.

    Args:
        params:       model parameters
        lr:           learning rate (default 0.02)
        momentum:     momentum coefficient (default 0.95)
        tunnel_p0:    initial tunneling probability (default 0.02)
        tunnel_decay: steps over which tunneling probability decays (default 5000)
        tunnel_scale: multiplier for tunneling step size (default 8.0)
        ns_iters:     Newton-Schulz iterations (default 3)
    """

    def __init__(self, params, lr=0.02, momentum=0.95,
                 tunnel_p0=0.02, tunnel_decay=5000,
                 tunnel_scale=8.0, ns_iters=3):
        defaults = dict(
            lr=lr, momentum=momentum,
            tunnel_p0=tunnel_p0, tunnel_decay=tunnel_decay,
            tunnel_scale=tunnel_scale, ns_iters=ns_iters,
        )
        super().__init__(params, defaults)
        self.step_count  = 0
        self.tunnel_log  = []

    def _newton_schulz(self, G, iters=3):
        """5th-order Newton-Schulz iteration to orthogonalise G."""
        assert G.ndim == 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G / (G.norm() + 1e-8)
        if G.shape[0] > G.shape[1]:
            X = X.T
        for _ in range(iters):
            A = X @ X.T
            X = a * X + b * A @ X + c * A @ A @ X
        if G.shape[0] > G.shape[1]:
            X = X.T
        return X

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        self.step_count += 1

        for group in self.param_groups:
            lr           = group['lr']
            momentum     = group['momentum']
            ns_iters     = group['ns_iters']
            tunnel_scale = group['tunnel_scale']
            tunnel_p     = group['tunnel_p0'] * math.exp(
                -self.step_count / group['tunnel_decay']
            )
            tunneling = (torch.rand(1).item() < tunnel_p)

            if tunneling:
                self.tunnel_log.append(self.step_count)

            for p in group['params']:
                if p.grad is None:
                    continue

                g     = p.grad
                state = self.state[p]

                # Initialise momentum buffer
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(p)

                buf = state['buf']

                if tunneling and g.ndim >= 2:
                    # TUNNEL: large normalised step, bypass clipping
                    update = g / (g.norm() + 1e-8)
                    p.add_(update, alpha=-lr * tunnel_scale)
                    buf.zero_()

                elif g.ndim >= 2 and min(g.shape) >= 2:
                    # MUON: Newton-Schulz orthogonalised update
                    if g.ndim > 2:
                        g2d = g.view(g.shape[0], -1)
                    else:
                        g2d = g
                    update = self._newton_schulz(g2d, iters=ns_iters)
                    if g.ndim > 2:
                        update = update.view_as(g)
                    buf.mul_(momentum).add_(update)
                    p.add_(buf, alpha=-lr)

                else:
                    # AdamW-style for 1-D params (bias, norms)
                    if 'exp_avg' not in state:
                        state['exp_avg']    = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                        state['adam_step']  = 0

                    state['adam_step'] += 1
                    ea, eas = state['exp_avg'], state['exp_avg_sq']
                    ea.mul_(0.9).add_(g, alpha=0.1)
                    eas.mul_(0.999).addcmul_(g, g, value=0.001)
                    bc1 = 1 - 0.9  ** state['adam_step']
                    bc2 = 1 - 0.999** state['adam_step']
                    step = (ea / bc1) / ((eas / bc2).sqrt() + 1e-8)
                    p.add_(step, alpha=-lr * 0.1)

        return loss
