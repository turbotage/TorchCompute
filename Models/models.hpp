#pragma once

#include "../pch.hpp"
#include "../Optim/model.hpp"

namespace models {

    // S = S_0*exp(-b*ADC), varying b values

    torch::Tensor adc_func(std::vector<torch::Tensor> staticvars, torch::Tensor per_problem_inputs, torch::Tensor parameters);

    void adc_jacobian_setter(std::unique_ptr<optim::Model>& pModel, torch::Tensor& jacobian);

    torch::Tensor simple_adc_model_linear(torch::Tensor bvals, torch::Tensor data);


    

    // S = S_0 * sin(FA) * (1 - exp(-TR/T1))/ (1 - exp(-TR/T1)cos(FA)), varying flip angles (FA)
    torch::Tensor vfa_func(std::vector<torch::Tensor> staticvars, torch::Tensor per_problem_inputs, torch::Tensor parameters);

    void vfa_jacobian_setter(std::unique_ptr<optim::Model>& pModel, torch::Tensor& jacobian);
    
    torch::Tensor simple_vfa_model_linear(torch::Tensor flip_angles, torch::Tensor data, torch::Tensor TR);

}
