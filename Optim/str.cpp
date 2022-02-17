#include "../pch.hpp"

#include "str.hpp"

constexpr int SUCCESSFULL_LU_DECOMP = 0;

tc::optim::STRPSettings::STRPSettings()
{
}

tc::optim::STRP::STRP(STRPSettings& settings)
	: m_Mu(settings.mu), m_Eta(settings.eta), 
	res(settings.start_residuals), J(settings.start_jacobian), delta(settings.start_deltas), Optimizer(settings)
{
	// Will be used by trailing point residuals, need to be preallocated
	reslike1 = torch::empty_like(res);

	scale_matrix = torch::diag_embed(settings.scaling);
	inv_scale_matrix = torch::diag_embed(torch::reciprocal(settings.scaling));

}

tc::optim::STRPResult tc::optim::STRP::eval()
{
	torch::InferenceMode im_guard;

	Optimizer::on_eval();

	solve();

	STRPResult ret;
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

	
	ret.lastP = p;
	ret.lastJ = J;
	ret.lastR = res;

	return ret;
}

std::unique_ptr<tc::optim::OptimResult> tc::optim::STRP::base_eval()
{
	torch::InferenceMode im_guard;

	Optimizer::on_eval();

	solve();

	std::unique_ptr<STRPResult> ret = std::make_unique<STRPResult>();
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


	ret->lastP = p;
	ret->lastJ = J;
	ret->lastR = res;

	return ret;
}

torch::Tensor tc::optim::STRP::default_delta_setup(torch::Tensor& parameters, float multiplier)
{
	torch::InferenceMode im_guard;
	return multiplier * torch::sqrt(torch::square(parameters).sum(1));
}

torch::Tensor tc::optim::STRP::default_scaling_setup(torch::Tensor& J)
{
	torch::InferenceMode im_guard;
	//return torch::rsqrt(torch::square(J).sum(1));
	return torch::ones({ J.size(0), J.size(2) }, J.options());
}

std::pair<torch::Tensor, torch::Tensor> tc::optim::STRP::default_res_J_setup(optim::Model& model, torch::Tensor data)
{
	torch::InferenceMode im_guard;

	torch::Tensor& pars = model.getParameters();

	torch::Tensor J = torch::empty({ pars.size(0), data.size(1), pars.size(1) }, pars.options());
	torch::Tensor res = torch::empty_like(data);

	model.res_diff(res, J, data);

	return std::make_pair(res, J);
}


void tc::optim::STRP::dogleg()
{
	torch::InferenceMode im_guard;

	// Create scaling matrix and scaled hessian
	//torch::Tensor& Jn = square1;
	torch::Tensor Jn = torch::sqrt(torch::square(J).sum(1));

	// Scaling matrix
	//torch::Tensor& invD = square2;
	torch::Tensor invD = torch::diag_embed(Jn);

	//torch::Tensor& D = square1; // Jn won't be used after step below, reuse it's memory
	torch::Tensor D = torch::diag_embed(torch::reciprocal(Jn));
	// Scaled Jacobian
	//torch::Tensor& Js = Jlike1;
	torch::Tensor Js = torch::bmm(J, D);

	// Scaled gradient
	//torch::Tensor& gs = plike1;
	torch::Tensor gs = torch::bmm(Js.transpose(1, 2), res.unsqueeze(-1)).squeeze(-1);

	// Scaled Hessian
	//torch::Tensor& Hs = square3;
	torch::Tensor Hs = torch::bmm(Js.transpose(1, 2), Js);


	// CALCULATE NEWTON-STEP
	//torch::Tensor& decomp = square4;
	torch::Tensor decomp;
	std::tie(decomp, pivots, luinfo) = at::_lu_with_info(Hs, true, false);

	std::cout << "luinfo: " << luinfo << std::endl;

	//torch::Tensor& pGN = p;
	torch::Tensor pGN = torch::bmm(D, torch::lu_solve(gs.unsqueeze(-1), decomp, pivots)).squeeze(-1);

	// Scale gauss newton step
	pGN = torch::bmm(scale_matrix, pGN.unsqueeze(-1)).squeeze(-1);

	std::cout << "pGN: " << torch::bmm(inv_scale_matrix, pGN.unsqueeze(-1)) << std::endl;

	//torch::Tensor& scaled_gn_norm = deltalike1;
	torch::Tensor scaled_gn_norm = torch::sqrt(torch::square(pGN).sum(1)).squeeze(-1);

	//torch::Tensor& gnstep = stepmask1;
	torch::Tensor gnstep = torch::logical_and(scaled_gn_norm <= delta, luinfo == SUCCESSFULL_LU_DECOMP);

	


	// CALCULATE CAUCHY-STEP
	//torch::Tensor& g = plike1; // gs won't be used after statement below, reuse memory
	torch::Tensor g = torch::bmm(invD, gs.unsqueeze(-1)).squeeze(-1);
	
	//torch::Tensor& invDg = plike2;
	torch::Tensor invDg = torch::bmm(invD, g.unsqueeze(-1)).squeeze(-1);

	//torch::Tensor& pCP = plike2; // invDg win't be used after statement below, reuse memory
	torch::Tensor pCP =  (-g.unsqueeze(-1) * (torch::bmm(g.unsqueeze(-1).transpose(1, 2), g.unsqueeze(-1)) /
		torch::bmm(invDg.unsqueeze(-1).transpose(1, 2), torch::bmm(Hs, invDg.unsqueeze(-1))))).squeeze(-1);

	// Scale cauchy step
	pCP = torch::bmm(scale_matrix, pCP.unsqueeze(-1)).squeeze(-1);

	//torch::Tensor& scaled_cp_norm = deltalike1;
	torch::Tensor scaled_cp_norm = torch::sqrt(torch::square(pCP).sum(1));

	// All problems with cauchy point outside trust region and problems with singular hessian should take a steepest descent step
	//torch::Tensor& cpstep = stepmask2;
	torch::Tensor cpstep = torch::logical_or(torch::logical_and(scaled_cp_norm > delta, torch::logical_not(gnstep)), luinfo != SUCCESSFULL_LU_DECOMP);

	pCP *= (cpstep * delta / scaled_cp_norm) + (1.0f * torch::logical_not(cpstep)); // cp steps should be scaled to trust region

	std::cout << "pCP: " << torch::bmm(inv_scale_matrix, pCP.unsqueeze(-1)) << std::endl;

	// CALCULATE INTERMEDIATE-STEP
	//torch::Tensor& GN_CP = deltalike1;
	torch::Tensor GN_CP = pGN - pCP;

	//torch::Tensor& A = deltalike2;
	torch::Tensor A = torch::square(GN_CP).sum(1);

	//torch::Tensor& B = deltalike3;
	torch::Tensor B = 2.0f * (pCP * GN_CP).sum(1);

	//torch::Tensor& C = deltalike4;
	torch::Tensor C = torch::square(pCP).sum(1) - torch::square(delta);

	//torch::Tensor& k = deltalike2; // A won't be used after step below, reuse it's memory
	torch::Tensor k = 0.5f * (-B + torch::sqrt(torch::square(B) - 4.0f * A * C)) / A;

	//torch::Tensor& pIP = plike1; // g isn't used anymore, reuse it's memory;
	torch::Tensor pIP = pCP + k * GN_CP;

	std::cout << "pIP: " << torch::bmm(inv_scale_matrix, pIP.unsqueeze(-1)) << std::endl;

	// All problems not taking steepest descent steps or full gn steps should take an interpolated step
	//torch::Tensor& ipstep = stepmask3;
	torch::Tensor ipstep = torch::logical_or(torch::logical_not(gnstep), torch::logical_not(cpstep));


	// Calculate final step
	p = torch::bmm(inv_scale_matrix, ((gnstep * pGN) + (cpstep * pCP) + (ipstep * pIP)).unsqueeze(-1)).squeeze(-1);

	std::cout << "p: " << p << std::endl;

}

void tc::optim::STRP::step()
{
	torch::InferenceMode im_guard;

	m_pModel->res_diff(res, J, m_Data);

	dogleg();

	//torch::Tensor& x_last = plike1;
	torch::Tensor x_last = m_pModel->getParameters();

	//torch::Tensor& ep = deltalike1;
	//torch::Tensor& et = deltalike2;

	torch::Tensor ep = 0.5f * torch::square(res).sum(1);

	m_pModel->setParameters(x_last + p);

	// residuals at trailing point
	torch::Tensor& res_tp = reslike1;

	m_pModel->res(res_tp, m_Data);

	torch::Tensor et = 0.5f * torch::square(res_tp).sum(1);

	//torch::Tensor& actual = deltalike1; // ep won't be used after step below, reuse it's memory
	torch::Tensor actual = ep - et;

	//torch::Tensor& Jp = plike2;
	torch::Tensor Jp = torch::bmm(J, p.unsqueeze(-1)).squeeze(-1);

	//torch::Tensor& predicted = deltalike2; // et isn't used anymore, reuse it's memory
	torch::Tensor predicted = -torch::bmm(res.unsqueeze(-1).transpose(1, 2), Jp.unsqueeze(-1)).squeeze(-1).squeeze(-1) - 0.5f * torch::square(Jp).sum(1);

	//torch::Tensor& rho = deltalike1; // predicted won't be used after step below, reuse it's memory
	torch::Tensor rho = actual / predicted;

	//std::cout << "rho: " << rho << std::endl;

	//torch::Tensor& poor_gain = stepmask1;
	torch::Tensor poor_gain = rho <= m_Mu;

	//torch::Tensor& good_gain = stepmask2;
	torch::Tensor good_gain = rho >= m_Eta;

	stepmask1 = poor_gain;
	stepmask2 = good_gain;

	//std::cout << "delta: " << delta << std::endl;
	//std::cout << "good_gain: " << good_gain << std::endl;
	//std::cout << "poor_gain: " << poor_gain << std::endl;

	delta *= 2.0f * good_gain + 0.5f * poor_gain + 1.0f*torch::logical_not(torch::logical_or(good_gain, poor_gain));

	// Step for all problems which have non poor gain-ratio
	p = p * torch::logical_not(poor_gain);

	// Update the model with our new parameters
	m_pModel->setParameters(x_last + p);
}

void tc::optim::STRP::solve()
{
	for (tc::ui32 iter = 0; iter < m_MaxIter + 1; ++iter) {
		step();

		if (Optimizer::should_stop())
			break;
		Optimizer::set_iter_info(iter, m_Data.size(0));
	}
}

