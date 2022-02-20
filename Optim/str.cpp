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
		{
			torch::Tensor& Jntemp = Jlike1;
			torch::square_out(Jntemp, J);
			torch::sum_out(Jn.squeeze_(-1), J, 1);
		}

		// Scaling matrix
		invD = torch::diag_embed(Jn);

		D = torch::diag_embed(Jn.reciprocal_());

		// Scaled Jacobian
		torch::Tensor& Js = Jlike1;
		torch::bmm_out(Js, J, D);

		// Scaled gradient
		torch::bmm_out(gs.unsqueeze_(-1), Js.transpose(1, 2), res.unsqueeze(-1));

		// Scaled Hessian
		torch::bmm_out(Hs, Js.transpose(1, 2), Js);

	}

	//debug_print(false, false, true);

	// occupied - square1, square2, square3, plike4

	// CALCULATE NEWTON-STEP
	torch::Tensor& pGN = plike1;
	torch::Tensor& scaled_gn_norm = deltalike1;
	torch::Tensor& gnstep = stepmask1;
	{
		torch::Tensor& decomp = square4;
		// Make LU decomposition
		std::tie(decomp, pivots, luinfo) = at::_lu_with_info(Hs, true, false);
		// Solve conditioned gn normal equations
		torch::lu_solve_out(pGN, gs.neg(), decomp, pivots);
		// Unscale condition matrix
		torch::bmm_out(plike2, D, pGN);
		// Scale gauss newton step
		torch::bmm_out(pGN, scale_matrix, plike2);

		torch::frobenius_norm_out(scaled_gn_norm.unsqueeze_(-1), pGN, 1).squeeze_(-1);

		torch::le_out(stepmask2, scaled_gn_norm, delta);
		torch::eq_out(stepmask3, luinfo, SUCCESSFULL_LU_DECOMP);
		torch::logical_and_out(gnstep, stepmask2, stepmask3);
	}

	//debug_print(false, false, true);

	/*std::cout << "pGN: " << pGN << std::endl;
	std::cout << "gnstep: " << gnstep << std::endl;*/

	// occupied - square1, square2, square3, plike4, plike1, deltalike1, stepmask1

	// CALCULATE CAUCHY-STEP
	torch::Tensor& pCP = plike2;
	torch::Tensor& scaled_cp_norm = deltalike2;
	torch::Tensor& cpstep = stepmask2;
	{

		torch::Tensor& g = pCP; // shadow pCP since it won't be used yet
		torch::bmm_out(g, invD, gs);

		torch::Tensor& invDg = gs; // shadow gs since it isn't used anymore
		torch::bmm_out(invDg, invD, g);

		torch::Tensor& lambdaStar = scaled_cp_norm; // shadow scaled_cp_norm since we won't use it yet
		{
			torch::Tensor& lambdaStar1 = deltalike3;
			torch::Tensor& lambdaStar2 = deltalike4;

			torch::square_out(plike3, g);
			torch::sum_out(lambdaStar1.unsqueeze_(-1), plike3, 1).squeeze_(-1);
			torch::bmm_out(plike3, Hs, invDg);
			torch::bmm_out(lambdaStar2.unsqueeze_(-1).unsqueeze_(-1), invDg.transpose(1, 2), plike3).squeeze_(-1).squeeze_(-1);

			torch::div_out(lambdaStar, lambdaStar1, lambdaStar2);
		}
		
		torch::neg_out(plike3, g);
		plike3.mul_(lambdaStar.unsqueeze(-1).unsqueeze(-1));

		// Scale cauchy step
		torch::bmm_out(pCP, scale_matrix, plike3);

		torch::frobenius_norm_out(scaled_cp_norm.unsqueeze_(-1), pCP, 1).squeeze_(-1);

		// All problems with cauchy point outside trust region and problems with singular hessian should take a steepest descent step (sets cpstep)
		{ // cpstep = (!gnstep AND |pCP| > delta) OR luinfo != SUCCESSFULL
			torch::logical_not_out(cpstep, gnstep);
			torch::greater_out(stepmask3, scaled_cp_norm, delta);
			torch::logical_and_out(stepmask4, stepmask3, cpstep);
			torch::ne_out(stepmask3, luinfo, SUCCESSFULL_LU_DECOMP);
			torch::logical_or_out(cpstep, stepmask3, stepmask4);
		}

		// All problems that should take a cpstep should be scaled to trust region
		{ // pCP *= cpstep * delta / |pCP| + !cpstep
			torch::mul_out(deltalike3, delta, cpstep);
			torch::logical_not_out(stepmask3, cpstep);
			deltalike3.div_(scaled_cp_norm).add_(stepmask3);
			pCP.mul_(deltalike3.unsqueeze(-1).unsqueeze(-1));
		}
	}

	//debug_print(false, false, true);

	/*std::cout << "pCP: " << pCP << std::endl;
	std::cout << "cpstep: " << cpstep << std::endl;
	std::cout << "pCP Norm: " << scaled_cp_norm << std::endl;*/

	// occupied - plike1, deltalike1, stepmask1, plike2, deltalike2, stepmask2


	torch::Tensor& pIP = plike3; // g isn't used anymore, reuse it's memory;
	torch::Tensor& ipstep = stepmask3;
	// CALCULATE INTERMEDIATE-STEP
	{
		torch::Tensor& GN_CP = pIP; // shadow pIP since it isn't used yet
		torch::sub_out(GN_CP, pGN, pCP);

		torch::Tensor& C = deltalike3;
		{
			torch::square_out(deltalike4, scaled_cp_norm);
			torch::square_out(deltalike5, delta);
			torch::sub_out(C, deltalike4, deltalike5);
		}

		torch::Tensor& B = deltalike4;
		{
			torch::mul_out(plike4, pCP, GN_CP);
			plike4.mul_(2.0f);
			torch::sum_out(B.unsqueeze_(-1), plike4, 1).squeeze_(-1);
		}

		torch::Tensor& A = deltalike5;
		torch::square_out(plike4, GN_CP);
		torch::sum_out(A.unsqueeze_(-1), plike4, 1).squeeze_(-1);

		torch::Tensor& k = scaled_cp_norm; // shadow scaled_cp_norm since it isn't used anymore
		{
			torch::square_out(k, B); // k = B^2
			C.mul_(A).mul_(-4.0f); // C = -4.0f * A * C
			k.add_(C).sqrt_(); //k = sqrt(B ^ 2 - 4.0f * A * C)
			k.sub_(B).div_(A).mul_(0.5f);  // k = 0.5*(-B + sqrt(B^2 - 4.0f * A * C)) / A
		}

		torch::mul_out(plike4, k.unsqueeze(-1).unsqueeze(-1), GN_CP);
		torch::add_out(pIP, pCP, plike4);
		
		

		// All problems not taking steepest descent steps or full gn steps should take an interpolated step
		ipstep = torch::logical_not(torch::logical_or(gnstep, cpstep));
	}

	//debug_print(false, false, true);

	/*std::cout << "pIP: " << pIP << std::endl;
	std::cout << "ipstep: " << ipstep << std::endl;
	std::cout << "delta: " << delta << std::endl;*/

	// occupied - plike1, deltalike1, stepmask1, plike2, stepmask2, plike3, stepmask3

	/*
	std::cout << "pGN: " << pGN << std::endl;
	std::cout << "gnstep: " << gnstep << std::endl;

	std::cout << "pCP: " << pGN << std::endl;
	std::cout << "cpstep: " << cpstep << std::endl;

	std::cout << "pIP: " << pIP << std::endl;
	std::cout << "ipstep: " << ipstep << std::endl;
	*/

	// Calculate final step
	torch::Tensor& p = plike4;
	pGN.mul_(gnstep.unsqueeze(-1).unsqueeze(-1)).nan_to_num_(0.0, 0.0, 0.0);
	pCP.mul_(cpstep.unsqueeze(-1).unsqueeze(-1)).nan_to_num_(0.0, 0.0, 0.0);
	pIP.mul_(ipstep.unsqueeze(-1).unsqueeze(-1)).nan_to_num_(0.0, 0.0, 0.0);
	torch::add_out(p, pCP, pIP);
	torch::add_out(pCP, p, pGN);
	torch::bmm_out(p, inv_scale_matrix, pCP);
}

void tc::optim::STRP::step()
{
	torch::InferenceMode im_guard;
	//debug_print(true, false);

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
	torch::Tensor& p = plike4;
	torch::Tensor& gnstep = stepmask1;
	torch::Tensor& cpstep = stepmask2;
	torch::Tensor& ipstep = stepmask3;
	torch::Tensor& scaled_gn_norm = deltalike1;

	//std::cout << "p: " << p << std::endl;

	torch::Tensor& x_last = plike3;
	x_last.copy_(m_pModel->getParameters().unsqueeze(-1));

	torch::Tensor& ep = deltalike2;
	{
		torch::square_out(reslike1, res);
		torch::sum_out(ep, reslike1, 1).mul_(0.5f);
	}

	m_pModel->getParameters().add_(p.squeeze(-1));
	
	

	// residuals at trailing point
	torch::Tensor& res_tp = reslike1;

	m_pModel->res(res_tp, m_Data);

	torch::Tensor& et = deltalike3;
	{
		res_tp.square_();
		torch::sum_out(et, res_tp, 1).mul_(0.5f);
	}

	ep.sub_(et); // now holds actual
	torch::Tensor& actual = ep; // shadow ep since it won't be used after step below

	torch::Tensor& Jp = res_tp; // shadow res_tp isn't used anymore
	//Jp = torch::bmm(J, p.unsqueeze(-1)).squeeze(-1);
	torch::bmm_out(Jp.unsqueeze_(-1), J, p);

	torch::Tensor& predicted = et; // et isn't used anymore, reuse it's memory
	{
		//predicted = -torch::bmm(res.unsqueeze(-1).transpose(1, 2), Jp).squeeze(-1).squeeze(-1) - 0.5f * torch::square(Jp).sum(1);
		torch::bmm_out(deltalike4.unsqueeze_(-1).unsqueeze_(-1), res.unsqueeze(-1).transpose(1, 2), Jp).squeeze_(-1).squeeze_(-1);
		torch::frobenius_norm_out(deltalike5.unsqueeze_(-1), Jp, 1).squeeze_(-1).mul_(0.5f);
		torch::add_out(predicted, deltalike4, deltalike5).neg_();
	}

	actual.div_(predicted);
	torch::Tensor& rho = actual; // shadow predicted since it won't be used after step below

	//std::cout << "rho: " << rho << std::endl;

	torch::Tensor& poor_gain = stepmask1;
	torch::le_out(poor_gain, rho, m_Mu);

	//std::cout << "poor_gain: " << poor_gain << std::endl;

	torch::Tensor& good_gain = stepmask2;
	torch::ge_out(good_gain, rho, m_Eta);

	//std::cout << "good_gain: " << good_gain << std::endl;

	torch::Tensor& multiplier = rho; // shadow rho since it isn't used anymore
	multiplier.zero_();
	multiplier.add_(good_gain);
	multiplier.mul_(2.0f); // multiplier holds good gain

	deltalike4.zero_();
	deltalike4.add_(poor_gain);
	deltalike4.mul_(0.5f);
	multiplier.add_(deltalike4); // adds poor gain
	
	good_gain.logical_or_(poor_gain); // all problems with good or poor gain
	good_gain.logical_not_(); // all probelms with neutral gain

	multiplier.add_(good_gain);
	delta.mul_(multiplier); // multiply 

	// If delta is bigger than norm of gauss newton step we should decrease it below GN step
	torch::lt_out(stepmask3, scaled_gn_norm, delta);
	torch::logical_and_out(stepmask4, stepmask3, poor_gain);
	
	torch::div_out(deltalike4, scaled_gn_norm, delta);
	deltalike4.mul_(0.5f);
	deltalike4.mul_(stepmask4); // multiplier for all problems with |pGN| < delta and poor gain, multiplier = 0.5 * |pGN|

	// multiplier for all problems with |pGN| > delta or non poor gain ratio we should not decrease delta i.e multiplier = 1
	torch::logical_not_out(stepmask3, stepmask4);
	deltalike4.add_(stepmask3);

	// Now we can multiply delta with multiplier
	delta.mul_(deltalike4);

	// Step for all problems which have non poor gain-ratio
	torch::logical_not_out(stepmask3, poor_gain);
	p.mul_(stepmask3.unsqueeze(-1).unsqueeze(-1));
	// We don't step for Inf, -Inf, NaN
	p.nan_to_num_(0.0, 0.0, 0.0);

	//std::cout << "p: " << p << std::endl;

	// Update the model with our new parameters
	torch::add_out(m_pModel->getParameters(), x_last.squeeze(-1), p.squeeze(-1));

	//std::cout << "param: " << m_pModel->getParameters() << std::endl;

	// make sure all vars are back to shape for next iteration
	//plike4.unsqueeze_(-1);
	reslike1.squeeze_(-1);

	//debug_print(true, false);

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
	stepmask4 = torch::empty_like(stepmask1);

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




void tc::optim::STRP::debug_print(bool sizes, bool types, bool values) {

	std::cout << "<=================== BEGIN DEBUG-PRINT =====================>\n";

	if (sizes) {
		std::cout << "res: " << res.sizes() << std::endl;
		std::cout << "reslike1: " << reslike1.sizes() << std::endl;

		std::cout << "delta: " << delta.sizes() << std::endl;
		std::cout << "deltalike1: " << deltalike1.sizes() << std::endl;
		std::cout << "deltalike2: " << deltalike2.sizes() << std::endl;
		std::cout << "deltalike3: " << deltalike3.sizes() << std::endl;
		std::cout << "deltalike4: " << deltalike4.sizes() << std::endl;
		std::cout << "deltalike5: " << deltalike5.sizes() << std::endl;

		std::cout << "J: " << J.sizes() << std::endl;
		std::cout << "Jlike1: " << Jlike1.sizes() << std::endl;

		std::cout << "params: " << m_pModel->getParameters().sizes() << std::endl;
		std::cout << "plike1: " << plike1.sizes() << std::endl;
		std::cout << "plike2: " << plike2.sizes() << std::endl;
		std::cout << "plike3: " << plike3.sizes() << std::endl;
		std::cout << "plike4: " << plike4.sizes() << std::endl;

		std::cout << "square1: " << square1.sizes() << std::endl;
		std::cout << "square2: " << square2.sizes() << std::endl;
		std::cout << "square3: " << square3.sizes() << std::endl;
		std::cout << "square4: " << square4.sizes() << std::endl;

		std::cout << "pivots: " << pivots.sizes() << std::endl;
		std::cout << "luinfo: " << luinfo.sizes() << std::endl;

		std::cout << "scale_matrix: " << scale_matrix.sizes() << std::endl;
		std::cout << "inv_scale_matrix: " << inv_scale_matrix.sizes() << std::endl;

		std::cout << "stepmask1: " << stepmask1.sizes() << std::endl;
		std::cout << "stepmask2: " << stepmask2.sizes() << std::endl;
		std::cout << "stepmask3: " << stepmask3.sizes() << std::endl;
	}

	if (types) {
		std::cout << "res: " << res.dtype() << std::endl;
		std::cout << "reslike1: " << reslike1.dtype() << std::endl;

		std::cout << "delta: " << delta.sizes() << std::endl;
		std::cout << "deltalike1: " << deltalike1.dtype() << std::endl;
		std::cout << "deltalike2: " << deltalike2.dtype() << std::endl;
		std::cout << "deltalike3: " << deltalike3.dtype() << std::endl;
		std::cout << "deltalike4: " << deltalike4.dtype() << std::endl;
		std::cout << "deltalike5: " << deltalike5.dtype() << std::endl;

		std::cout << "J: " << J.dtype() << std::endl;
		std::cout << "Jlike1: " << Jlike1.dtype() << std::endl;

		std::cout << "params: " << m_pModel->getParameters().dtype() << std::endl;
		std::cout << "plike1: " << plike1.dtype() << std::endl;
		std::cout << "plike2: " << plike2.dtype() << std::endl;
		std::cout << "plike3: " << plike3.dtype() << std::endl;
		std::cout << "plike4: " << plike4.dtype() << std::endl;

		std::cout << "square1: " << square1.dtype() << std::endl;
		std::cout << "square2: " << square2.dtype() << std::endl;
		std::cout << "square3: " << square3.dtype() << std::endl;
		std::cout << "square4: " << square4.dtype() << std::endl;

		std::cout << "pivots: " << pivots.dtype() << std::endl;
		std::cout << "luinfo: " << luinfo.dtype() << std::endl;

		std::cout << "scale_matrix: " << scale_matrix.dtype() << std::endl;
		std::cout << "inv_scale_matrix: " << inv_scale_matrix.dtype() << std::endl;

		std::cout << "stepmask1: " << stepmask1.dtype() << std::endl;
		std::cout << "stepmask2: " << stepmask2.dtype() << std::endl;
		std::cout << "stepmask3: " << stepmask3.dtype() << std::endl;
	}

	if (values) {
		std::cout << "res: " << res << std::endl;
		std::cout << "reslike1: " << reslike1 << std::endl;

		std::cout << "delta: " << delta << std::endl;
		std::cout << "deltalike1: " << deltalike1 << std::endl;
		std::cout << "deltalike2: " << deltalike2 << std::endl;
		std::cout << "deltalike3: " << deltalike3 << std::endl;
		std::cout << "deltalike4: " << deltalike4 << std::endl;
		std::cout << "deltalike5: " << deltalike5 << std::endl;

		std::cout << "J: " << J << std::endl;
		std::cout << "Jlike1: " << Jlike1 << std::endl;

		std::cout << "params: " << m_pModel->getParameters() << std::endl;
		std::cout << "plike1: " << plike1 << std::endl;
		std::cout << "plike2: " << plike2 << std::endl;
		std::cout << "plike3: " << plike3 << std::endl;
		std::cout << "plike4: " << plike4 << std::endl;

		std::cout << "square1: " << square1 << std::endl;
		std::cout << "square2: " << square2 << std::endl;
		std::cout << "square3: " << square3 << std::endl;
		std::cout << "square4: " << square4 << std::endl;

		std::cout << "pivots: " << pivots << std::endl;
		std::cout << "luinfo: " << luinfo << std::endl;

		std::cout << "scale_matrix: " << scale_matrix << std::endl;
		std::cout << "inv_scale_matrix: " << inv_scale_matrix << std::endl;

		std::cout << "stepmask1: " << stepmask1 << std::endl;
		std::cout << "stepmask2: " << stepmask2 << std::endl;
		std::cout << "stepmask3: " << stepmask3 << std::endl;
	}

	std::cout << "<=================== END DEBUG-PRINT =====================>\n";

}

