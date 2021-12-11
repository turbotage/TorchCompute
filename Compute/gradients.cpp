#include "gradients.hpp"




torch::Tensor tc::compute::jacobian(torch::Tensor y, torch::Tensor x)
{
	assert(y.sizes()[0] == x.sizes()[0]);

	int batch_size = y.size(0);
	int output_dim = y.size(1);
	int input_dim = x.size(1);

	auto dops = x.options().requires_grad(false); // maybe remove requires grad

	auto J = torch::empty({batch_size, output_dim, input_dim}, dops);

	for (int i = 0; i < output_dim; i++)
	{
		auto grad_output = torch::ones({ 1 }, dops);
		grad_output = grad_output.expand({ batch_size, 1 });
		//print(grad_output);
		//std::cout << "grad_output size: " << grad_output.sizes() << std::endl;

		auto gradient = torch::autograd::grad({ y.slice(/*dim=*/1, /*start=*/i,
			/*end=*/i + 1) },
			{ x },
			/*grad_outputs=*/{ grad_output },
			/*retain_graph=*/true,
			/*create_graph=*/true);
		auto grad = gradient[0].unsqueeze(/*dim=*/1);
		//std::cout << "grad size: " << grad.sizes() << std::endl;
		J.slice(1, i, i + 1) = grad;
	}

	return J;
}


// I suspect a bug in this jacobian version
torch::Tensor tc::compute::jacobian2(torch::Tensor y, torch::Tensor x)
{
	std::vector<torch::Tensor> grad_array;
	auto input_arr = std::vector<torch::Tensor>(1, x);
	torch::Tensor out_flat = y.reshape(-1);
	for (int i = 0; i < out_flat.numel(); ++i) {
		auto o = std::vector<torch::Tensor>(1, out_flat[i]);
		grad_array.push_back(torch::autograd::grad(o, input_arr, {}, true, false, false)[0]);
	}
	return torch::stack(grad_array, 0).unflatten(0, x.sizes());
}

