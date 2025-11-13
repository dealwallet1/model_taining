import torch
from ..moe import MoE

def test_moe_forward_dims_and_grad():
    print("*************inside the moe layer**************")
    B,T,C = 2, 8, 32
    moe = MoE(dim=C, n_expert=4, k=1)
    print(f"*******THIS IS MOE*******************{moe}**********************")
    x = torch.randn(B,T,C, requires_grad=True)
    y, aux = moe(x)
    print("this aux loss**********************",aux)
    assert y.shape == x.shape
    loss = (y**2).mean() + 0.01*aux
    loss.backward()
    # some gradient must flow to gate and experts
    grads = [p.grad is not None for p in moe.parameters()]
    assert any(grads)

