import torch

#nProblems = 100000
nDims = 3

def model(deps, params):
	n_probs = params.size(0)
	return torch.sin(params[:,0]).view(n_probs,1)*(1-2*torch.exp(-deps[:,:,0] * params[:,1].view(n_probs, 1)))*torch.exp(-deps[:,:,1] * params[:,2].view(n_probs, 1))

products = []
prod = torch.arange(-3,3)
for i in range(nDims):
	products.append(prod)

permuts = torch.cartesian_prod(*products)
products = 0
prod = 0

print(permuts.numel())
print(permuts.shape)

deps = 100*torch.rand(1,5,2)
params = torch.ones(1,4)
params[0,0] = 1.2
params[0,1] = 0.01
params[0,2] = 0.1

data = model(deps, params)

perms = 10**(permuts)
permuts = 0

min_diff = float("inf")
min_params = torch.tensor([0,0])

for i in range(1000):
	test_params = (-torch.rand(perms.size(0),nDims)+0.5)*perms

	test_data = model(deps,test_params)
	test_data[test_data.isnan()] = float("inf")
	diff = data - test_data
	diffnorm = torch.norm(diff, dim=1)
	(min, min_id) = torch.min(diffnorm, dim=0)

	if min < min_diff:
		min_diff = min
		min_params = test_params[min_id,:]
		print("min_params: ", min_params)
		print("min_diff: ", min_diff)


# This works meh


