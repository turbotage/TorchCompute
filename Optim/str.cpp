#include "../pch.hpp"

#include <ATen/native/LinearAlgebra.h>

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

	setup_solve();

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

	torch::Tensor& p = plike4;
	ret.lastP = p;
	ret.lastJ = J;
	ret.lastR = res;

	return ret;
}

std::unique_ptr<tc::optim::OptimResult> tc::optim::STRP::base_eval()
{
	torch::InferenceMode im_guard;

	Optimizer::on_eval();

	setup_solve();

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

	torch::Tensor& p = plike4;
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
	//return torch::sqrt(torch::square(J).sum(1));
	//return torch::ones({ J.size(0), J.size(2) }, J.options());

	return torch::sqrt(torch::diagonal(torch::bmm(J.transpose(1, 2), J), 0, -2, -1));


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

	torch::Tensor& D = square1;
	torch::Tensor& invD = square2;
	torch::Tensor& Hs = square3;
	torch::Tensor& gs = plike4;
	{
		// Create scaling matrix and scaled hessian
		torch::Tensor& Jn = gs; // shadow gs as it isn't yet used
		torch::frobenius_norm_out(Jn.squeeze_(-1), J, 1);

		// Scaling matrix
		invD.set_(torch::diag_embed(Jn));

		D.set_(torch::diag_embed(Jn.reciprocal_()));
		// Scaled Jacobian
		torch::Tensor& Js = Jlike1;
		torch::bmm_out(Js, J, D);

		// Scaled gradient
		torch::bmm_out(gs.unsqueeze_(-1), Js.transpose(1, 2), res.unsqueeze(-1));

		// Scaled Hessian
		torch::bmm_out(Hs, Js.transpose(1, 2), Js);
	}

	// occupied - square1, square2, square3, plike4

	// CALCULATE NEWTON-STEP
	torch::Tensor& pGN = plike1;
	torch::Tensor& scaled_gn_norm = deltalike1;
	torch::Tensor& gnstep = stepmask1;
	{
		torch::Tensor& decomp = square4;
		std::tie(decomp, pivots, luinfo) = at::_lu_with_info(Hs, true, false);

		torch::bmm_out(pGN, D, torch::lu_solve_out(pGN, gs.neg(), decomp, pivots));

		// Scale gauss newton step
		torch::bmm_out(pGN, scale_matrix, pGN);

		torch::frobenius_norm_out(scaled_gn_norm.unsqueeze_(-1), pGN, 1).squeeze_(-1);

		torch::logical_and_out(gnstep, scaled_gn_norm <= delta, luinfo == SUCCESSFULL_LU_DECOMP);
	}

	// occupied - square1, square2, square3, plike4, plike1, deltalike1, stepmask1

	// CALCULATE CAUCHY-STEP
	torch::Tensor& pCP = plike2;
	torch::Tensor& cpstep = stepmask2;
	{
		torch::Tensor& g = gs; // shadow gs since it won't be used after statement below
		torch::bmm_out(g, invD, gs);

		torch::Tensor& invDg = pCP; // shadow pCP since it won't be used yet
		torch::bmm_out(invDg, invD, g);

		torch::Tensor& scaled_cp_norm = deltalike2;

		torch::Tensor& lambdaStar = scaled_cp_norm; // shadow scaled_cp_norm since we won't use it yet
		{
			torch::Tensor& lambdaStar1 = deltalike2;
			torch::Tensor& lambdaStar2 = deltalike3;

			torch::Tensor& plike_temp = plike3;
			torch::sum_out(lambdaStar1.unsqueeze_(-1), torch::square_out(plike_temp, g), 1).squeeze_(-1);
			torch::bmm_out(lambdaStar2.unsqueeze_(-1).unsqueeze_(-1), invDg.transpose(1, 2), torch::bmm_out(plike_temp, Hs, invDg)).squeeze_(-1).squeeze_(-1);

			torch::div_out(lambdaStar, lambdaStar1, lambdaStar2);
		}

		torch::neg_out(pCP, g).mul_(lambdaStar.unsqueeze(-1));

		// Scale cauchy step
		torch::bmm_out(pCP, scale_matrix, pCP);

		torch::frobenius_norm_out(scaled_cp_norm.unsqueeze_(-1), pCP, 1).squeeze_(-1);

		// All problems with cauchy point outside trust region and problems with singular hessian should take a steepest descent step (sets cpstep)
		{
			torch::Tensor& cpstep_temp = stepmask3;
			torch::logical_and_out(cpstep_temp, scaled_cp_norm > delta, torch::logical_not_out(cpstep_temp, gnstep));
			torch::ne_out(cpstep, luinfo, SUCCESSFULL_LU_DECOMP);
			torch::logical_or_out(cpstep, cpstep, cpstep_temp);
		}

		// All problems that should take a cpstep should be scaled to trust region
		{
			torch::Tensor& temp_multiplier = deltalike3;
			torch::mul_out(temp_multiplier, delta, cpstep);
			temp_multiplier.div_(scaled_cp_norm).add_(torch::logical_not(cpstep));
			pCP.mul_(temp_multiplier);
		}
	}

	// occupied - plike1, deltalike1, stepmask1, plike2, stepmask2


	torch::Tensor& pIP = plike3; // g isn't used anymore, reuse it's memory;
	torch::Tensor& ipstep = stepmask3;
	// CALCULATE INTERMEDIATE-STEP
	{
		torch::Tensor& GN_CP = pIP; // shadow pIP since it isn't used yet
		torch::sub_out(GN_CP, pGN, pCP);

		torch::Tensor& GN_CP2 = plike4;
		torch::square_out(GN_CP2, GN_CP);

		torch::Tensor& A = deltalike2;
		torch::sum_out(A.unsqueeze_(-1), GN_CP2, 1).squeeze_(-1);

		torch::Tensor& B = deltalike3;
		{
			torch::Tensor& Bvec = GN_CP2; // shadow GN_CP2 since it isn't used anymore
			torch::mul_out(Bvec, pCP, GN_CP).mul_(2.0f);
			torch::sum_out(B.unsqueeze_(-1), Bvec, 1).squeeze_(-1);
		}

		torch::Tensor& C = deltalike4;
		{
			torch::Tensor& C1 = C; // shadow C since it isn't used yet
			torch::Tensor& C2 = deltalike5;
			torch::sub_out(C, torch::square_out(C1, scaled_cp_norm), torch::square_out(C2, delta));
		}


		torch::Tensor& k = deltalike5;
		{
			torch::Tensor& k1 = k; // shadow k since it isn't used yet
			torch::square_out(k1, B); // k1 = B^2
			torch::addcmul_out(k1, k1, A, C, -4.0f); // k1 = B^2 - 4.0f * A * C
			k1.sqrt_(); // k1 = sqrt(B^2 - 4.0f * A * C)

			torch::sub_out(k, k1, B); // k = -B + sqrt(B^2 - 4.0f * A * C)
			torch::div_out(k, k, A); // k = -B + sqrt(B^2 - 4.0f * A * C) / A
			k.mul_(0.5f); // 0.5*(-B + sqrt(B^2 - 4.0f * A * C) / A)
		}

		torch::add_out(pIP, pCP, torch::mul_out(pCP, k.unsqueeze(-1), GN_CP));
		
		

		// All problems not taking steepest descent steps or full gn steps should take an interpolated step
		ipstep = torch::logical_not(torch::logical_or(gnstep, cpstep));
	}

	// occupied - plike1, deltalike1, stepmask1, plike2, stepmask2, plike3, stepmask3

	// Calculate final step
	torch::Tensor& p = plike4;
	pGN.mul_(gnstep);
	pCP.mul_(cpstep);
	pIP.mul_(ipstep);
	torch::add_out(p, pGN, torch::add_out(p, pCP, pIP));
	torch::bmm_out(p, inv_scale_matrix, p);

}

void tc::optim::STRP::step()
{
	torch::InferenceMode im_guard;

	// Make sure everything is back to shape
	


	m_pModel->res_diff(res, J, m_Data);

	dogleg();

	/*
		pGN				= plike1
		pCP				= plike2
		pIP				= plike3
		p				= plike4

		gnstep			= stepmask1
		cpstep			= stepmask2
		ipstep			= stepmask3

		scaled_cp_norm	= deltalike1
	*/
	torch::Tensor& pGN = plike1;
	torch::Tensor& pCP = plike2;
	torch::Tensor& pIP = plike3;
	torch::Tensor& p = plike4;
	torch::Tensor& gnstep = stepmask1;
	torch::Tensor& cpstep = stepmask2;
	torch::Tensor& ipstep = stepmask3;
	torch::Tensor& scaled_gn_norm = deltalike1;


	torch::Tensor& x_last = plike1;
	x_last = m_pModel->getParameters();

	torch::Tensor& ep = deltalike2;
	ep = 0.5f * torch::square(res).sum(1);

	m_pModel->getParameters().add_(p.squeeze_(-1));

	// residuals at trailing point
	torch::Tensor& res_tp = reslike1;

	m_pModel->res(res_tp, m_Data);

	torch::Tensor& et = deltalike3;
	et = 0.5f * torch::square(res_tp).sum(1);

	torch::Tensor& actual = ep.sub_(et); // shadow ep since it won't be used after step below

	torch::Tensor& Jp = res_tp; // shadow res_tp isn't used anymore
	//Jp = torch::bmm(J, p.unsqueeze(-1)).squeeze(-1);
	torch::bmm_out(Jp.unsqueeze_(-1), J, p.unsqueeze(-1));

	torch::Tensor& predicted = et; // et isn't used anymore, reuse it's memory
	{
		torch::Tensor& pred1 = predicted;
		torch::Tensor& pred2 = deltalike4; 
		//predicted = -torch::bmm(res.unsqueeze(-1).transpose(1, 2), Jp).squeeze(-1).squeeze(-1) - 0.5f * torch::square(Jp).sum(1);
		torch::bmm_out(pred1.unsqueeze_(-1).unsqueeze_(-1), res.unsqueeze(-1).transpose(1, 2), Jp).squeeze_(-1).squeeze_(-1);
		torch::frobenius_norm_out(pred2.unsqueeze_(-1), Jp, 1).squeeze_(-1).mul_(0.5f);
		torch::add_out(predicted, pred1, pred2).neg_();
	}


	torch::Tensor& rho = actual.div_(predicted); // shadow predicted since it won't be used after step below

	torch::Tensor& poor_gain = stepmask1;
	poor_gain = rho <= m_Mu;

	torch::Tensor& good_gain = stepmask2;
	good_gain = rho >= m_Eta;

	delta *= 2.0f * good_gain + 0.5f * poor_gain + 1.0f * torch::logical_not(torch::logical_or(good_gain, poor_gain));

	// Step for all problems which have non poor gain-ratio
	p = p * torch::logical_not(poor_gain).unsqueeze(-1);
	p.nan_to_num_(0.0, 0.0, 0.0);

	// Update the model with our new parameters
	torch::add_out(m_pModel->getParameters(), x_last, p);

	plike1.unsqueeze_(-1);
	plike2.unsqueeze_(-1);
	plike4.unsqueeze_(-1);

	std::cout << "plike1: " << plike1.sizes() << std::endl;
	std::cout << "plike2: " << plike2.sizes() << std::endl;
	std::cout << "plike3: " << plike3.sizes() << std::endl;
	std::cout << "plike4: " << plike4.sizes() << std::endl;

	std::cout << "deltalike1: " << deltalike1.sizes() << std::endl;
	std::cout << "deltalike2: " << deltalike2.sizes() << std::endl;
	std::cout << "deltalike3: " << deltalike3.sizes() << std::endl;
	std::cout << "deltalike4: " << deltalike4.sizes() << std::endl;
	std::cout << "deltalike5: " << deltalike5.sizes() << std::endl;


}

void tc::optim::STRP::setup_solve() {

	auto dops = m_pModel->getParameters().options();
	numProbs = m_Data.size(0);
	numParam = m_pModel->getParameters().size(1);
	numData = m_Data.size(1);

	reslike1 = torch::empty_like(res);

	deltalike1 = torch::empty_like(delta);
	deltalike2 = torch::empty_like(delta);
	deltalike3 = torch::empty_like(delta);
	deltalike4 = torch::empty_like(delta);
	deltalike5 = torch::empty_like(delta);

	Jlike1 = torch::empty_like(J);

	plike1 = torch::empty({ numProbs, numParam, 1 }, dops);
	plike2 = torch::empty({ numProbs, numParam, 1 }, dops);
	plike3 = torch::empty({ numProbs, numParam, 1 }, dops);
	plike4 = torch::empty({ numProbs, numParam, 1 }, dops);

	square1 = torch::empty({ numProbs, numParam, numParam }, dops);
	square2 = torch::empty_like(square1);
	square3 = torch::empty_like(square1);
	square4 = torch::empty_like(square1);

	pivots = torch::empty({ numProbs, numParam }, dops.dtype(torch::ScalarType::Int));
	luinfo = torch::empty({ numProbs }, dops.dtype(torch::ScalarType::Int));

	stepmask1 = torch::empty({ numProbs }, dops.dtype(torch::ScalarType::Bool));
	stepmask2 = torch::empty_like(stepmask1);
	stepmask3 = torch::empty_like(stepmask1);

}

void tc::optim::STRP::solve()
{
	torch::InferenceMode im_guard;

	for (tc::ui32 iter = 0; iter < m_MaxIter + 1; ++iter) {
		step();

		if (Optimizer::should_stop())
			break;
		Optimizer::set_iter_info(iter, m_Data.size(0));
	}
}

