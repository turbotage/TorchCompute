
import torch

a = torch.rand(10,3)
b = torch.rand(10,3)
c = torch.rand(2,2,1)

torch.divide(a,b,out=c)

print(c)