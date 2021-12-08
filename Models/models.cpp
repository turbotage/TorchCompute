#include "models.hpp"
#include "../Compute/lstq.hpp"


void models::adc_eval_and_diff(std::vector<torch::Tensor>& constants, torch::Tensor& per_problem_inputs, torch::Tensor& parameters,
	OutRef<torch::Tensor> values, OptOutRef<torch::Tensor> jacobian, OptOutRef<const torch::Tensor> data)
{
	using namespace torch::indexing;

	torch::Tensor& ppi = per_problem_inputs;
	torch::Tensor& par = parameters;

	ui32 numProbs = par.size(0);
	ui32 numParams = par.size(1);
	ui32 numData;

	torch::Tensor S0 = par.index({ Slice(), 0 }).view({ par.size(0), 1 });
	torch::Tensor ADC = par.index({ Slice(), 1 }).view({ par.size(0), 1 });

	torch::Tensor b;
	torch::Tensor expterm;
	if (per_problem_inputs.defined()) { // There were different b-values for all problems
		numData = ppi.size(1);
		b = ppi.index({ Slice(), Slice(), 0 }).view({ ppi.size(0), ppi.size(1) });
	}
	else {
		numData = constants[0].size(0);
		b = constants[0];
	}
	expterm = torch::exp(-b * ADC);

	values = S0 * expterm;

	if (jacobian.has_value()) {
		torch::Tensor& J = jacobian.value().get();
		J.index_put_({ Slice(), Slice(), 0 }, expterm);
		J.index_put_({ Slice(), Slice(), 1 }, -b * values);
	}

	if (data.has_value()) {
		values = values - data.value().get();
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



void models::vfa_eval_and_diff(std::vector<torch::Tensor>& constants, torch::Tensor& per_problem_inputs, torch::Tensor& parameters,
	OutRef<torch::Tensor> values, OptOutRef<torch::Tensor> jacobian, OptOutRef<const torch::Tensor> data)
{
	using namespace torch::indexing;

	torch::Tensor& ppi = per_problem_inputs;
	torch::Tensor& par = parameters;

	ui32 numProbs = par.size(0);
	ui32 numParams = par.size(1);
	ui32 numData;

	torch::Tensor S0 = par.index({ Slice(), 0 }).view({ par.size(0), 1 });
	torch::Tensor T1 = par.index({ Slice(), 1 }).view({ par.size(0), 1 });
	torch::Tensor TR = constants[0];

	torch::Tensor FA;
	if (per_problem_inputs.defined()) { // There were different FA-values for all problems
		numData = ppi.size(1);
		FA = ppi.index({ Slice(), Slice(), 0 }).view({ ppi.size(0), ppi.size(1) });
	}
	else {
		numData = constants[1].size(0);
		FA = constants[1];
	}
	
	torch::Tensor expterm = torch::exp(-TR / T1);
	torch::Tensor denom = (1 - expterm * torch::cos(FA));

	values = torch::sin(FA);

	if (jacobian.has_value()) {
		torch::Tensor& J = jacobian.value().get();
		J.index_put_({ Slice(), Slice(), 0 }, values * (1 - expterm) / denom );
		J.index_put_({ Slice(), Slice(), 1 }, S0 * values * ((torch::cos(FA)-1) / torch::square(denom)) * expterm * TR / torch::square(T1));
	}

	if (jacobian.has_value()) {
		torch::Tensor& J = jacobian.value().get();
		values = J.index({ Slice(), Slice(),0 }) * S0;
	}
	else {
		values = S0 * values * (1 - expterm) / denom;
	}

	if (data.has_value()) {
		values = values - data.value().get();
	}
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

