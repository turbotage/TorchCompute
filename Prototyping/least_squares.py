
import torch

import matplotlib.pyplot as plt
import numpy as np

nProbs = 100
nPoints = 20
nParams = 1

x = torch.rand(nProbs, nPoints, nParams)
y = (2*torch.rand(nProbs,1,1)-1)*torch.rand(nProbs, nPoints, 1) + torch.rand(nProbs,1,1)
# b contains the parameters

A = torch.cat((torch.ones(nProbs,nPoints,1),x),dim=2)

Q,R = torch.qr(A)

ATy = torch.bmm(A.transpose(1,2), y)

z,_ = torch.triangular_solve(ATy, R.transpose(1,2), False) # Rb

b,_ = torch.triangular_solve(z, R)

b1 = b[0]

x0 = x[0]
y0 = y[0]

xs = np.linspace(0,1,100)

print(b1[1])
print(b1[0])
print(y0)
print(x0)



for i in range(0,20):
    bi = b[i]

    xi = x[i]
    yi = y[i]
    
    regs = lambda x: bi[1].item()*x + bi[0].item()

    plt.figure()
    plt.plot(xi,yi, "x")
    plt.plot(xs, regs(xs), "-")

    plt.show()

print('Hello')


