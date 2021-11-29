#pragma once

#include "../pch.hpp"
#include "../Optim/model.hpp"

namespace models {

    // S = S_0*exp(-b*ADC), varying b values

    torch::Tensor adc_func(std::vector<torch::Tensor> staticvars, torch::Tensor dependents, torch::Tensor parameters);

    torch::Tensor simple_adc_model_linear(torch::Tensor bvals, torch::Tensor data);

    torch::Tensor simple_adc_model_nonlinear(torch::Tensor bvals, torch::Tensor data, torch::Tensor parameter_guess);

    

    // S = S_0 * sin(FA) * (1 - exp(-TR/T1))/ (1 - exp(-TR/T1)cos(FA)), varying flip angles (FA)
    torch::Tensor vfa_func(std::vector<torch::Tensor> staticvars, torch::Tensor dependents, torch::Tensor parameters);

    torch::Tensor simple_vfa_model_linear(torch::Tensor flip_angles, torch::Tensor data, torch::Tensor TR);

    torch::Tensor simple_vfa_model_nonlinear(torch::Tensor flip_angles, torch::Tensor data, torch::Tensor TR, torch::Tensor parameter_guess);

}
