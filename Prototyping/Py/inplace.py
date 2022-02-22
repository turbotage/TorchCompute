
import torch

Hs = torch.tensor([[[0.9640, 0.9467],[0.9467, 0.9300]]])
print(Hs)
print(Hs.shape)
decomp, pivot, info = torch.lu(Hs, pivot=True,get_infos=True)

gs = torch.tensor([[[-306.4979],[-308.8952]]])
print(gs)
print(gs.shape)

x = torch.lu_solve(-gs, decomp, pivot)
print(x)

torch.bmm(Hs, x, out=x)

print(x)