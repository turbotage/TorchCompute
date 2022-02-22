
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

Q,R = torch.linalg.qr(A)
U, S, VT = torch.linalg.svd(A, full_matrices=False)


# QR linres
ATy = torch.bmm(A.transpose(1,2), y)

bq,_ = torch.triangular_solve(ATy, R.transpose(1,2), False) # Rb

bq,_ = torch.triangular_solve(bq, R)

# SVD linres

bs = torch.bmm(VT.transpose(1,2), torch.diag_embed(1 / S))
bs = torch.bmm(bs, U.transpose(1,2))
bs = torch.bmm(bs, y)



xs = np.linspace(0,1,100)

for i in range(0,20):
    xi = x[i]
    yi = y[i]
    
    # QR
    biq = bq[i]
    
    regsq = lambda x: biq[1].item()*x + biq[0].item()


    # SVD
    bis = bs[i]

    regss = lambda x: bis[1].item()*x + bis[0].item()



    plt.figure(1)
    plt.plot(xi,yi, "x")
    plt.plot(xs, regsq(xs), "-b")
    plt.plot(xs, regss(xs), "-r")
    plt.show()





