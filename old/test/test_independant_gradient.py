import torch
from torch import nn
from torch import autograd

module = nn.Linear(2, 3)
parameters = list(module.parameters())

x = torch.rand(10, 2)
grad = autograd.functional.jacobian(module, x)
print(grad.sum())
