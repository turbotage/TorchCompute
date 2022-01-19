#include "models.hpp"
#include "models.hpp"
#include "../pch.hpp"

#include "models.hpp"
#include "models.hpp"
#include "../Compute/lstq.hpp"


void tc::models::adc_eval_and_diff(std::vector<torch::Tensor>& constants, torch::Tensor& per_problem_inputs, torch::Tensor& parameters,
	tc::OutRef<torch::Tensor> values, tc::OptOutRef<torch::Tensor> jacobian, tc::OptOutRef<const torch::Tensor> data)
{
	using namespace torch::indexing;

	torch::Tensor& ppi = per_problem_inputs;
	torch::Tensor& par = parameters;

	tc::ui32 numProbs = par.size(0);
	tc::ui32 numParams = par.size(1);
	tc::ui32 numData;

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


torch::Tensor tc::models::simple_adc_model_linear(torch::Tensor bvals, torch::Tensor data)
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

	deps.index_put_({ Slice(),0 }, torch::exp(deps.index({ Slice(), 0 })));
	deps.index_put_({ Slice(),1 }, -1.0f * deps.index({ Slice(), 1 }));

	return deps.view({data.size(0), 2});
}



void tc::models::vfa_eval_and_diff(std::vector<torch::Tensor>& constants, torch::Tensor& per_problem_inputs, torch::Tensor& parameters,
	tc::OutRef<torch::Tensor> values, tc::OptOutRef<torch::Tensor> jacobian, tc::OptOutRef<const torch::Tensor> data)
{
	using namespace torch::indexing;

	torch::Tensor& ppi = per_problem_inputs;
	torch::Tensor& par = parameters;

	tc::ui32 numProbs = par.size(0);
	tc::ui32 numParams = par.size(1);
	tc::ui32 numData;

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
	torch::Tensor denom = (1.0f - expterm * torch::cos(FA));

	values = torch::sin(FA);

	if (jacobian.has_value()) {
		torch::Tensor& J = jacobian.value().get();
		J.index_put_({ Slice(), Slice(), 0 }, values * (1.0f - expterm) / denom );
		J.index_put_({ Slice(), Slice(), 1 }, S0 * values * ((torch::cos(FA)-1.0f) / torch::square(denom)) * expterm * TR / torch::square(T1));
	}

	if (jacobian.has_value()) {
		torch::Tensor& J = jacobian.value().get();
		values = J.index({ Slice(), Slice(), 0}) * S0;
	}
	else {
		values = S0 * values * (1.0f - expterm) / denom;
	}

	if (data.has_value()) {
		values = values - data.value().get();
	}
}

torch::Tensor tc::models::simple_vfa_model_linear(torch::Tensor flip_angles, torch::Tensor data, torch::Tensor TR)
{
	using namespace torch::indexing;

	torch::Tensor deps;
	if (flip_angles.size(0) == 1) {
		deps = flip_angles.repeat({ data.size(0), 1, 1 });
	}
	else {
		deps = flip_angles;
	}

	torch::Tensor temp = data / torch::sin(deps.squeeze()); // s_n/sin(FA)
	{
		torch::Tensor rdatatan = data / torch::tan(deps.squeeze()); // s_n/tan(FA)
		temp = compute::lstq_qr(rdatatan.view({ rdatatan.size(0),rdatatan.size(1), 1 }), temp);
	}

	// Set S0
	temp.index_put_({ Slice(), 0 },
		temp.index({ Slice(), 0 }) / (1.0f - temp.index({Slice(), 1 })));
	// Set T1
	temp.index_put_({ Slice(), 1 }, -TR / torch::log(temp.index({ Slice(), 1 })));

	return temp.view({data.size(0), 2});
}



void tc::models::ir_eval_and_diff(std::vector<torch::Tensor>& constants, torch::Tensor& per_problem_inputs, torch::Tensor& parameters, 
	tc::OutRef<torch::Tensor> values, tc::OptOutRef<torch::Tensor> jacobian, tc::OptOutRef<const torch::Tensor> data)
{
	using namespace torch::indexing;

	torch::Tensor& ppi = per_problem_inputs;
	torch::Tensor& par = parameters;

	tc::ui32 numProbs = par.size(0);
	tc::ui32 numParams = par.size(1);
	tc::ui32 numData;

	torch::Tensor S0 = par.index({ Slice(), 0 }).view({ par.size(0), 1 });
	torch::Tensor T1 = par.index({ Slice(), 1 }).view({ par.size(0), 1 });

	torch::Tensor TR = constants[0];
	torch::Tensor TI = constants[1];

	torch::Tensor FA_term = (torch::cos(constants[2]) - 1);

	torch::Tensor expterm1 = FA_term*torch::exp(-TI/T1);
	torch::Tensor expterm2 = torch::exp(-TR/T1);

	values = (1 + expterm1 - expterm2);

	torch::Tensor derivsign = values / torch::abs(values);

	if (jacobian.has_value()) {
		torch::Tensor& J = jacobian.value().get();
		J.index_put_({ Slice(), Slice(), 0 }, derivsign * values);
		J.index_put_({ Slice(), Slice(), 1 }, derivsign * S0 * (1 + (expterm1 * TI / torch::square(T1)) - (expterm2 * TR / torch::square(T1))));
	}

	values = torch::abs(S0 * values);

	if (data.has_value()) {
		values = values - data.value().get();
	}
}


void tc::models::ir_varfa_eval_and_diff(std::vector<torch::Tensor>& constants, torch::Tensor& per_problem_inputs, torch::Tensor& parameters, tc::OutRef<torch::Tensor> values, tc::OptOutRef<torch::Tensor> jacobian, tc::OptOutRef<const torch::Tensor> data)
{
	using namespace torch::indexing;

	torch::Tensor& ppi = per_problem_inputs;
	torch::Tensor& par = parameters;

	tc::ui32 numProbs = par.size(0);
	tc::ui32 numParams = par.size(1);
	tc::ui32 numData;

	torch::Tensor S0 = par.index({ Slice(), 0 }).view({ par.size(0), 1 });
	torch::Tensor T1 = par.index({ Slice(), 1 }).view({ par.size(0), 1 });
	torch::Tensor FA = par.index({ Slice(), 2 }).view({ par.size(0), 1 });

	torch::Tensor TR = constants[0];
	torch::Tensor TI = constants[1];
	torch::Tensor FA_term = torch::cos(FA);

	torch::Tensor expterm1 = torch::exp(-TI / T1);
	torch::Tensor totexp = (FA_term - 1) * expterm1;

	torch::Tensor expterm2 = torch::exp(-TR / T1);

	values = (1 + totexp - expterm2);

	torch::Tensor derivsign = values / torch::abs(values);

	if (jacobian.has_value()) {
		torch::Tensor& J = jacobian.value().get();
		J.index_put_({ Slice(), Slice(), 0 }, derivsign * values);
		J.index_put_({ Slice(), Slice(), 1 }, derivsign * S0 * (1 + (totexp * TI / torch::square(T1)) - (expterm2 * TR / torch::square(T1))));
		J.index_put_({ Slice(), Slice(), 2 }, derivsign * S0 * FA_term * expterm1);
	}

	values = torch::abs(S0 * values);

	if (data.has_value()) {
		values = values - data.value().get();
	}
}


void tc::models::t2_eval_and_diff(std::vector<torch::Tensor>& constants, torch::Tensor& per_problem_inputs, torch::Tensor& parameters, tc::OutRef<torch::Tensor> values, tc::OptOutRef<torch::Tensor> jacboian, tc::OptOutRef<const torch::Tensor> data)
{
	using namespace torch::indexing;

	torch::Tensor& ppi = per_problem_inputs;
	torch::Tensor& par = parameters;

	tc::ui32 numProbs = par.size(0);
	tc::ui32 numParams = par.size(1);
	tc::ui32 numData;

	torch::Tensor S0 = par.index({ Slice(), 0 }).view({ par.size(0), 1 });
	torch::Tensor T2 = par.index({ Slice(), 1 }).view({ par.size(0), 1 });

	torch::Tensor expterm;
	torch::Tensor TE;
	if (per_problem_inputs.defined()) {
		numData = ppi.size(1);
		TE = ppi.index({ Slice(), Slice(), 0 }).view({ ppi.size(0), ppi.size(1) });
	}
	else {
		numData = constants[0].size(0);
		TE = constants[0];
	}
	expterm = torch::exp(-TE / T2);

	values = S0 * expterm;

	if (jacboian.has_value()) {
		torch::Tensor& J = jacboian.value().get();
		J.index_put_({ Slice(), Slice(), 0 }, expterm);
		J.index_put_({ Slice(), Slice(), 1 }, values * TE / torch::square(T2));
	}

	if (data.has_value()) {
		values = values - data.value().get();
	}

}




