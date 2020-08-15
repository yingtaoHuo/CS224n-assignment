import torch

a = torch.arange(10).reshape(5,2)
torch.split(a,1)