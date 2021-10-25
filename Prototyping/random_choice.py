import torch

nProblems = 10000
nDims = 2

def model(deps, params):
	n_probs = params.size(0)
	return params[:,0].view(n_probs, 1) * torch.exp(-deps[:,:,0] * params[:,1].view(n_probs, 1))

products = []
prod = torch.arange(-10,10)
for i in range(nDims):
	products.append(prod)

permuts = torch.cartesian_prod(*products)
products = 0
prod = 0

print(permuts.numel())
print(permuts.shape)

deps = torch.rand(1,5,1)
params = torch.ones(1,2)
params[0,0] = 8000
params[0,1] = 100

data = model(deps, params)

perms = 10**(permuts)
permuts = 0

min_diff = float("inf")
min_params = torch.tensor([0,0])

for i in range(1000):
	test_params = (-torch.rand(400,1)+0.5)*perms

	test_data = model(deps,test_params)
	test_data[test_data.isnan()] = float("inf")
	diff = data - test_data
	diffnorm = torch.norm(diff, dim=1)
	(min, min_id) = torch.min(diffnorm, dim=0)

	if min < min_diff:
		min_diff = min
		min_params = test_params[min_id,:]
		print(min_params)

# This works meh


