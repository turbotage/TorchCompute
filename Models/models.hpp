#pragma once

#include "../pch.hpp"
#include "../Optim/model.hpp"

namespace models {

	// S = S_0*exp(-b*ADC), varying b values
	void adc_eval_and_diff(std::vector<torch::Tensor>& constants, torch::Tensor& per_problem_inputs, torch::Tensor& parameters,
		OutRef<torch::Tensor> values, OptOutRef<torch::Tensor> jacobian, OptOutRef<const torch::Tensor> data);

	torch::Tensor simple_adc_model_linear(torch::Tensor bvals, torch::Tensor data);


	// S = S_0 * sin(FA) * (1 - exp(-TR/T1)) / (1 - exp(-TR/T1)cos(FA)), varying flip angles (FA)
	void vfa_eval_and_diff(std::vector<torch::Tensor>& constants, torch::Tensor& per_problem_inputs, torch::Tensor& parameters,
		OutRef<torch::Tensor> values, OptOutRef<torch::Tensor> jacobian, OptOutRef<const torch::Tensor> data);

	torch::Tensor simple_vfa_model_linear(torch::Tensor flip_angles, torch::Tensor data, torch::Tensor TR);

}
