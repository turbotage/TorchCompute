#include "models.hpp"
#include "../Compute/lstq.hpp"


void models::adc_eval_and_diff(std::vector<torch::Tensor>& constants, torch::Tensor& per_problem_inputs, torch::Tensor& parameters,
	OutRef<torch::Tensor> values, OptOutRef<torch::Tensor> jacobian)
{
	using namespace torch::indexing;

	torch::Tensor& ppi = per_problem_inputs;
	torch::Tensor& par = parameters;

	ui32 numProbs = par.size(0);
	ui32 numParams = par.size(1);
	ui32 numData;

	torch::Tensor S0 = par.index({ Slice(), 0 }).view({ par.size(0), 1 });
	torch::Tensor ADC = par.index({ Slice(), 1 }).view({ par.size(0), 1 });

	torch::Tensor expterm;
	if (per_problem_inputs.defined()) { // There were different b-values for all pixels
		numData = ppi.size(1);
		expterm = torch::exp(- ADC * ppi.index({ Slice(), Slice(), 0 }).view({ ppi.size(0), ppi.size(1) }));
	}
	else {
		numData = constants[0].size(0);
		expterm = torch::exp(-constants[0] * ADC);
	}

	values = S0 * expterm;

	if (jacobian.has_value()) {
		jacobian.index_put_({Slice(), 0}, expterm)
	}


}


torch::Tensor models::simple_adc_model_linear(torch::Tensor bvals, torch::Tensor data)
{
	using namespace torch::indexing;

	torch::Tensor deps;
	if (bvals.size(0) == 1) {
		deps = bvals.repeat({ data.size(0), 1, 1 });
	}
	else {
		deps = bvals;
	}

	// deps = parameters, reuse of var
	deps = compute::lstq_qr(deps, torch::log(data));

	deps.index_put_({ Slice(),0 }, torch::exp(deps.index({ Slice(),0 })));
	deps.index_put_({ Slice(),1 }, -1.0 * deps.index({ Slice(), 1 }));

	return deps;
}



torch::Tensor models::vfa_func(
	std::vector<torch::Tensor> staticvars,
	torch::Tensor per_problem_inputs, torch::Tensor parameters)
{
	using namespace torch::indexing;

	torch::Tensor temp = parameters.index({ Slice(), 0 }).view({ parameters.size(0), 1 }); // s0
	temp *= torch::sin(per_problem_inputs.index({ Slice(), Slice(), 0 }).view({ per_problem_inputs.size(0), per_problem_inputs.size(1) })); // s0 * sin(FA)
	torch::Tensor expterm = torch::exp(-staticvars[0] / parameters.index({ Slice(), 1 }).view({ parameters.size(0), 1 })); // exp(-TR/T1)
	temp *= (1 - expterm); // s0 * sin(FA) * (1-expterm)
	temp /= (1 - expterm * torch::cos(per_problem_inputs.index({ Slice(), Slice(), 0 }).view({ per_problem_inputs.size(0), per_problem_inputs.size(1) }))); // Full expression

	return temp;
}

torch::Tensor models::simple_vfa_model_linear(torch::Tensor flip_angles, torch::Tensor data, torch::Tensor TR)
{
	using namespace torch::indexing;

	torch::Tensor temp = data.view({ data.size(0), data.size(1), 1 }) / torch::sin(flip_angles); // s_n/sin(FA)
	{
		torch::Tensor rdatatan = data.view({ data.size(0), data.size(1), 1 }) / torch::tan(flip_angles); // s_n/tan(FA)
		temp = compute::lstq_qr(rdatatan, temp);
	}

	// Set S0
	temp.index_put_({ Slice(), Slice(), 0 },
		temp.index({ Slice(), Slice(), 0 }) / (1 - temp.index({ Slice(), Slice(), 1 })));
	// Set T1
	temp.index_put_({ Slice(), Slice(), 0 }, -TR / torch::log(temp.index({ Slice(), Slice(), 1 })));

	return temp;
}