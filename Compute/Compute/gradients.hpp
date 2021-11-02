#pragma once

#include "../pch.hpp"

namespace compute {

    torch::Tensor jacobian(torch::Tensor y, torch::Tensor x);

    torch::Tensor jacobian2(torch::Tensor y, torch::Tensor x);

}
