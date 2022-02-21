#include "../pch.hpp"

#include "tr.hpp"

tc::optim::TRP::TRP(TRPSettings& settings)
	: m_Mu(settings.mu), m_Eta(settings.eta),
	res(settings.start_residuals), J(settings.start_jacobian), delta(settings.start_deltas), Optimizer(settings)
{
	// Will be used by trailing point residuals, need to be preallocated
	reslike1 = torch::empty_like(res);

	scale_matrix = torch::diag_embed(settings.scaling);
	inv_scale_matrix = torch::diag_embed(torch::reciprocal(settings.scaling));
}

tc::optim::TRPResult tc::optim::TRP::eval()
{
	torch::InferenceMode im_guard;

	Optimizer::on_eval();

	setup_solve();

	solve();

	TRPResult ret;
	ret.finalParameters = m_pModel->getParameters();
	ret.finalDeltas = delta;
	ret.pFinalModel = std::move(m_pModel);

	torch::Tensor& gain = luinfo;
	gain.fill_(eGainType::ACCEPTABLE_GAIN);

	// after step, a poor gain mask will be filled in stepmask1 and a good gain mask will be set in stepmask2
	torch::Tensor& poor_gain = stepmask1;
	torch::Tensor& good_gain = stepmask2;

	gain.masked_fill_(poor_gain, eGainType::POOR_GAIN);
	gain.masked_fill_(good_gain, eGainType::GOOD_GAIN);

	// The last step we tool will be saved in p
	ret.gain = gain;

	torch::Tensor& p = plike4;
	ret.lastP = p;
	ret.lastJ = J;
	ret.lastR = res;

	return ret;
}

std::unique_ptr<tc::optim::OptimResult> tc::optim::TRP::base_eval()
{
	torch::InferenceMode im_guard;

	Optimizer::on_eval();

	setup_solve();

	solve();

	std::unique_ptr<TRPResult> ret = std::make_unique<TRPResult>();
	ret->finalParameters = m_pModel->getParameters();
	ret->finalDeltas = delta;
	ret->pFinalModel = std::move(m_pModel);

	torch::Tensor& gain = luinfo;
	gain.fill_(eGainType::ACCEPTABLE_GAIN);

	// after step, a poor gain mask will be filled in stepmask1 and a good gain mask will be set in stepmask2
	torch::Tensor& poor_gain = stepmask1;
	torch::Tensor& good_gain = stepmask2;

	gain.masked_fill_(poor_gain, eGainType::POOR_GAIN);
	gain.masked_fill_(good_gain, eGainType::GOOD_GAIN);

	// The last step we tool will be saved in p
	ret->gain = gain;

	torch::Tensor& p = plike4;
	ret->lastP = p;
	ret->lastJ = J;
	ret->lastR = res;

	return ret;
}

torch::Tensor tc::optim::TRP::default_delta_setup(torch::Tensor& parameters, float multiplier)
{
	torch::InferenceMode im_guard;
	return multiplier * torch::sqrt(torch::square(parameters).sum(1));
}

torch::Tensor tc::optim::TRP::default_scaling_setup(torch::Tensor& J)
{
	torch::InferenceMode im_guard;
	//return torch::sqrt(torch::square(J).sum(1));
	//return torch::ones({ J.size(0), J.size(2) }, J.options());

	return torch::sqrt(torch::diagonal(torch::bmm(J.transpose(1, 2), J), 0, -2, -1));
}

std::pair<torch::Tensor, torch::Tensor> tc::optim::TRP::default_res_J_setup(optim::Model& model, torch::Tensor data)
{
	torch::InferenceMode im_guard;

	torch::Tensor& pars = model.getParameters();

	torch::Tensor J = torch::empty({ pars.size(0), data.size(1), pars.size(1) }, pars.options());
	torch::Tensor res = torch::empty_like(data);

	model.res_diff(res, J, data);

	return std::make_pair(res, J);
}

void tc::optim::TRP::dogleg()
{
}

void tc::optim::TRP::step()
{
}

void tc::optim::TRP::setup_solve()
{
}

void tc::optim::TRP::solve()
{
}

void tc::optim::TRP::debug_print(bool sizes, bool types, bool values)
{
}
