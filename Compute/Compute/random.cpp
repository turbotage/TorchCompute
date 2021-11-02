#include "random.hpp"

torch::Tensor compute::random::random_choice(int start, int end, int nsamples, torch::TensorOptions tops)
{
	return torch::multinomial(torch::arange(start, end, tops), nsamples, false);
}
