import torch


@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # Initialize momentum buffer if first time
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                # Update momentum buffer: buf = momentum * buf + (1-momentum) * grad
                buf.lerp_(g, 1 - group["momentum"])
                # Apply Nesterov momentum if enabled, otherwise use standard momentum
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                # Apply zero-power normalization via Newton-Schulz iterations (make it close to orthonormal)
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                # Update parameters with adaptive scaling based on parameter shape
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
                # Updates parameters with an adaptive learning rate that scales based on the parameter tensor's aspect ratio (height/width). For matrices where height > width, it increases the effective learning rate by √(height/width)