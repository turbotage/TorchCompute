#pragma once

#include "../pch.hpp"

namespace compute {

    torch::Tensor lstq_qr(torch::Tensor x, torch::Tensor y);

    torch::Tensor lstq_svd(torch::Tensor x, torch::Tensor y);

}