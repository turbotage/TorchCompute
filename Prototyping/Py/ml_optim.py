import torch
from torch import nn
from torch.optim import optimizer
from torch.functional import F

from fast_pytorch_kmeans import KMeans

nProblems = 40
nParams = 2
nData = 5
device_str = 'cpu'

params = torch.zeros(nProblems, nParams, device=device_str)
params[:,0] = torch.rand(nProblems, device=device_str)
params[:,1] = 0.01*torch.rand(nProblems, device=device_str)

deps = torch.zeros(nProblems, nData, 1, device=device_str)
deps[:,0,0] = 200
deps[:,1,0] = 300
deps[:,2,0] = 400
deps[:,3,0] = 600
deps[:,4,0] = 900
	
def model(deps, params):
	n_probs = params.size(0)
	return params[:,0].view(n_probs, 1) * torch.exp(-deps[:,:,0]*params[:,1].view(n_probs, 1))


data = model(deps, params)

guess = torch.rand(nProblems, nParams, requires_grad=True, device=device_str)
#guess[:,0] = 100000*torch.rand(nProblems, device=device_str)
#guess[:,1] = 100*torch.rand(nProblems, device=device_str)
#guess = guess.requires_grad_(True)

#optimizer = torch.optim.LBFGS([guess], max_iter=20, lr=1)
#optimizer = torch.optim.SGD([guess], lr=1)


def closure():
	optimizer.zero_grad()
	output = model(deps, guess)
	loss = F.mse_loss(output, data)
	loss.backward()
	return loss


kmeans = KMeans(n_clusters=10, mode='euclidean')
labels = kmeans.fit_predict(data)
print(labels)

for label_iter in range(10):
	idx = labels == label_iter
	label_params = guess[idx]
	label_deps = deps[idx]
	label_data = data[idx]                            
	
	label_params = GD(label_params, label_data, label_deps, model)

	guess[idx] = label_params


print(params)
print(guess)