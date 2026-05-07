import sys, os, math
sys.path.append(os.path.relpath("../QPyTorch"))
sys.path.append(os.path.relpath("../FixedTorch"))
import torch
from FixedTensor import FixedTensor
from Parameter import Parameter
import nn as qnn

Parameter.setfl(24); Parameter.setwl(64); Parameter.settrunc_type(1)
torch.manual_seed(0)
x_orig = torch.randn(4, 10)
target = torch.tensor([0, 1, 2, 3])

for sm in [0, 1, 2]:
    Parameter.setsoftmax_type(sm)
    x = FixedTensor(x_orig.clone()).round()
    x.requires_grad_(True)
    ce = qnn.CrossEntropyLoss(reduction='mean')
    loss = ce(x, target)
    loss.backward()
    print("sm={} loss={:.6f} grad_norm={:.6f} grad[0]={}".format(
        sm, float(loss), x.grad.norm().item(), [round(v, 5) for v in x.grad[0].tolist()]))
    print("  has_nan:", torch.isnan(x.grad).any().item())

# Reference
x2 = x_orig.clone().requires_grad_(True)
ref = torch.nn.CrossEntropyLoss(reduction='mean')
loss2 = ref(x2, target)
loss2.backward()
print("torch ref loss={:.6f} grad_norm={:.6f} grad[0]={}".format(
    float(loss2), x2.grad.norm().item(), [round(v, 5) for v in x2.grad[0].tolist()]))
