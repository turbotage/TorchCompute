#pragma once

#include "../pch.hpp"

namespace compute {

	namespace random {

		torch::Tensor random_choice(int start, int end, int nsamples, torch::TensorOptions tops);

	}

}
