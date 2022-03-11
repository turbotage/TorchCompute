#include "../../pch.hpp"

#include "mp_strp.hpp"

constexpr int SUCCESSFULL_LU_DECOMP = 0;


tc::optim::MP_STRPSettings::MP_STRPSettings(MP_STRPSettings&& settings)
	: MP_OptimizerSettings(std::move(settings)), 
	start_residuals(std::move(settings.start_residuals)), 
	start_jacobian(std::move(settings.start_jacobian)),
	start_deltas(std::move(settings.start_deltas)),
	scaling(std::move(settings.scaling)),
	mu(settings.mu), eta(settings.eta)
{
}

tc::optim::MP_STRPSettings::MP_STRPSettings(MP_OptimizerSettings&& optimsettings, const torch::Tensor& start_residuals, const torch::Tensor& start_jacobian,
	const torch::Tensor& start_deltas, const torch::Tensor& scaling, float mu, float eta)
	: MP_OptimizerSettings(std::move(optimsettings)),
	start_residuals(start_residuals),
	start_jacobian(start_jacobian),
	start_deltas(start_deltas),
	scaling(scaling),
	mu(mu), eta(eta)
{
}



std::unique_ptr<tc::optim::MP_STRPVars> tc::optim::MP_STRPVars::make(std::unique_ptr<optim::MP_Model>& pModel, const torch::Tensor& data,
	torch::Tensor& residuals, torch::Tensor& jacobian, torch::Tensor& delta, torch::Tensor& scaling,
	float mu, float eta)
{
	return std::unique_ptr<MP_STRPVars>(new MP_STRPVars(pModel, data, residuals, jacobian, delta, scaling, mu, eta));
}

void tc::optim::MP_STRPVars::to_device(const torch::Device& device)
{
	torch::InferenceMode im_guard;

	res.to(device);
	reslike1.to(device);

	delta.to(device);
	deltalike1.to(device);
	deltalike2.to(device);
	deltalike3.to(device);
	deltalike4.to(device);
	deltalike5.to(device);

	J.to(device);
	Jlike1.to(device);

	plike1.to(device);
	plike2.to(device);
	plike3.to(device);
	plike4.to(device);

	square1.to(device);
	square2.to(device);
	square3.to(device);
	square4.to(device);

	pivots.to(device);

	luinfo.to(device);

	scale_matrix.to(device);
	inv_scale_matrix.to(device);

	stepmask1.to(device);
	stepmask2.to(device);
	stepmask3.to(device);
	stepmask4.to(device);

}

void tc::optim::MP_STRPVars::to_float32()
{
	torch::InferenceMode im_guard;

	res.to(torch::ScalarType::Float);
	reslike1.to(torch::ScalarType::Float);

	delta.to(torch::ScalarType::Float);
	deltalike1.to(torch::ScalarType::Float);
	deltalike2.to(torch::ScalarType::Float);
	deltalike3.to(torch::ScalarType::Float);
	deltalike4.to(torch::ScalarType::Float);
	deltalike5.to(torch::ScalarType::Float);

	J.to(torch::ScalarType::Float);
	Jlike1.to(torch::ScalarType::Float);

	plike1.to(torch::ScalarType::Float);
	plike2.to(torch::ScalarType::Float);
	plike3.to(torch::ScalarType::Float);
	plike4.to(torch::ScalarType::Float);

	square1.to(torch::ScalarType::Float);
	square2.to(torch::ScalarType::Float);
	square3.to(torch::ScalarType::Float);
	square4.to(torch::ScalarType::Float);

	scale_matrix.to(torch::ScalarType::Float);
	inv_scale_matrix.to(torch::ScalarType::Float);
}

void tc::optim::MP_STRPVars::to_float64()
{
	torch::InferenceMode im_guard;

	res.to(torch::ScalarType::Double);
	reslike1.to(torch::ScalarType::Double);

	delta.to(torch::ScalarType::Double);
	deltalike1.to(torch::ScalarType::Double);
	deltalike2.to(torch::ScalarType::Double);
	deltalike3.to(torch::ScalarType::Double);
	deltalike4.to(torch::ScalarType::Double);
	deltalike5.to(torch::ScalarType::Double);

	J.to(torch::ScalarType::Double);
	Jlike1.to(torch::ScalarType::Double);

	plike1.to(torch::ScalarType::Double);
	plike2.to(torch::ScalarType::Double);
	plike3.to(torch::ScalarType::Double);
	plike4.to(torch::ScalarType::Double);

	square1.to(torch::ScalarType::Double);
	square2.to(torch::ScalarType::Double);
	square3.to(torch::ScalarType::Double);
	square4.to(torch::ScalarType::Double);

	scale_matrix.to(torch::ScalarType::Double);
	inv_scale_matrix.to(torch::ScalarType::Double);
}

void tc::optim::MP_STRPVars::debug_print(bool sizes, bool types, bool values) {
	torch::InferenceMode im_guard;

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


tc::optim::MP_STRPVars::MP_STRPVars(const std::unique_ptr<optim::MP_Model>& pModel, const torch::Tensor& data,
	torch::Tensor& residuals, torch::Tensor& jacobian, torch::Tensor& delta, torch::Tensor& scaling,
	float mu, float eta)
{
	torch::InferenceMode im_guard;

	auto dops = pModel->parameters().options();

	this->mu = mu;
	this->eta = eta;

	numProbs = data.size(0);
	numParam = pModel->parameters().size(1);
	numData = data.size(1);

	this->res = residuals;
	reslike1 = torch::empty_like(res);

	this->delta = delta;
	deltalike1 = torch::empty_like(delta);
	deltalike2 = torch::empty_like(delta);
	deltalike3 = torch::empty_like(delta);
	deltalike4 = torch::empty_like(delta);
	deltalike5 = torch::empty_like(delta);

	this->J = jacobian;
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

	scale_matrix = torch::diag_embed(scaling);
	inv_scale_matrix = torch::diag_embed(torch::reciprocal(scaling));

	stepmask1 = torch::empty({ numProbs }, dops.dtype(torch::ScalarType::Bool));
	stepmask2 = torch::empty_like(stepmask1);
	stepmask3 = torch::empty_like(stepmask1);
	stepmask4 = torch::empty_like(stepmask1);
}




std::unique_ptr<tc::optim::MP_STRP> tc::optim::MP_STRP::make(MP_STRPSettings&& settings)
{
	auto pVars = MP_STRPVars::make(settings.pModel, settings.data, settings.start_residuals,
		settings.start_jacobian, settings.start_deltas, settings.scaling, settings.mu, settings.eta);

	return std::make_unique<MP_STRP>(std::move(settings), std::move(pVars));
}

tc::optim::MP_STRP::MP_STRP(MP_OptimizerSettings&& optsettings, std::unique_ptr<MP_STRPVars> strpvars)
	: MP_Optimizer(std::move(optsettings)), m_pVars(std::move(strpvars))
{

}




torch::Tensor tc::optim::MP_STRP::last_parameters()
{
	if (!pModel)
		throw std::runtime_error("Tried to get parameters on optimizer where OptimResult had been acquired");

	return pModel->parameters();
}

torch::Tensor tc::optim::MP_STRP::last_step()
{
	if (!m_pVars)
		throw std::runtime_error("Tried to get last_step on optimizer where vars had been aquired");

	// After solve has been run, the last step is stored in plike4
	return m_pVars->plike4;
}

torch::Tensor tc::optim::MP_STRP::last_jacobian()
{
	if (!m_pVars)
		throw std::runtime_error("Tried to get last_jacobian on optimizer where vars had been aquired");

	return m_pVars->J;
}

torch::Tensor tc::optim::MP_STRP::last_residuals()
{
	if (!m_pVars)
		throw std::runtime_error("Tried to get last_residuals on optimizer where vars had been aquired");

	return m_pVars->res;
}

torch::Tensor tc::optim::MP_STRP::last_deltas()
{
	if (!m_pVars)
		throw std::runtime_error("Tried to get last_deltas on optimizer where vars had been aquired");

	return m_pVars->delta;
}

torch::Tensor tc::optim::MP_STRP::last_multiplier()
{
	if (!m_pVars)
		throw std::runtime_error("Tried to get last_multiplier on optimizer where vars had been aquired");

	// After solve has been run, the last multipliers are stored in deltalike2
	return m_pVars->deltalike2;
}




std::unique_ptr<tc::optim::MP_STRPVars> tc::optim::MP_STRP::acquire_vars()
{
	return std::move(m_pVars);
}



torch::Tensor tc::optim::MP_STRP::default_delta_setup(torch::Tensor& parameters, float multiplier)
{
	torch::InferenceMode im_guard;
	return multiplier * torch::sqrt(torch::square(parameters).sum(1));
}

torch::Tensor tc::optim::MP_STRP::default_scaling_setup(torch::Tensor& J)
{
	torch::InferenceMode im_guard;
	//return torch::sqrt(torch::square(J).sum(1));
	//return torch::ones({ J.size(0), J.size(2) }, J.options());

	return torch::sqrt(torch::diagonal(torch::bmm(J.transpose(1, 2), J), 0, -2, -1));
}

std::pair<torch::Tensor, torch::Tensor> tc::optim::MP_STRP::default_res_J_setup(optim::MP_Model& model, torch::Tensor data)
{
	torch::InferenceMode im_guard;

	torch::Tensor& pars = model.parameters();

	torch::Tensor J = torch::empty({ pars.size(0), data.size(1), pars.size(1) }, pars.options());
	torch::Tensor res = torch::empty_like(data);

	model.res_jac(res, J, data);

	return std::make_pair(res, J);
}

void tc::optim::MP_STRP::on_run()
{
	if (!m_pVars)
		throw std::runtime_error("Tried to run() on STRPOptimizer where vars had been acquired");

	solve();
}

tc::optim::MP_OptimResult tc::optim::MP_STRP::on_acquire_result()
{
	return MP_OptimResult(std::move(pModel));
}

void tc::optim::MP_STRP::on_abort()
{
}

void tc::optim::MP_STRP::dogleg()
{
	torch::InferenceMode im_guard;

	torch::Tensor& D = m_pVars->square1;
	torch::Tensor& invD = m_pVars->square2;
	torch::Tensor& Hs = m_pVars->square3;
	torch::Tensor& gs = m_pVars->plike4;
	{
		// Create scaling matrix and scaled hessian
		torch::Tensor& Jn = gs; // shadow gs as it isn't yet used
		{
			torch::Tensor& Jntemp = m_pVars->Jlike1;
			torch::square_out(Jntemp, m_pVars->J);
			torch::sum_out(Jn.squeeze_(-1), m_pVars->J, 1);
		}

		// Scaling matrix
		invD = torch::diag_embed(Jn);

		D = torch::diag_embed(Jn.reciprocal_());

		// Scaled Jacobian
		torch::Tensor& Js = m_pVars->Jlike1;
		torch::bmm_out(Js, m_pVars->J, D);

		// Scaled gradient
		torch::bmm_out(gs.unsqueeze_(-1), Js.transpose(1, 2), m_pVars->res.unsqueeze(-1));

		// Scaled Hessian
		torch::bmm_out(Hs, Js.transpose(1, 2), Js);

	}

	//debug_print(false, false, true);

	// occupied - square1, square2, square3, plike4

	// CALCULATE NEWTON-STEP
	torch::Tensor& pGN = m_pVars->plike1;
	torch::Tensor& scaled_gn_norm = m_pVars->deltalike1;
	torch::Tensor& gnstep = m_pVars->stepmask1;
	{
		torch::Tensor& decomp = m_pVars->square4;
		// Make LU decomposition
		std::tie(decomp, m_pVars->pivots, m_pVars->luinfo) = at::_lu_with_info(Hs, true, false);
		// Solve conditioned gn normal equations
		torch::lu_solve_out(pGN, gs.neg(), decomp, m_pVars->pivots);
		// Unscale condition matrix
		torch::bmm_out(m_pVars->plike2, D, pGN);
		// Scale gauss newton step
		torch::bmm_out(pGN, m_pVars->scale_matrix, m_pVars->plike2);

		torch::frobenius_norm_out(scaled_gn_norm.unsqueeze_(-1), pGN, 1).squeeze_(-1);

		torch::le_out(m_pVars->stepmask2, scaled_gn_norm, m_pVars->delta);
		torch::eq_out(m_pVars->stepmask3, m_pVars->luinfo, SUCCESSFULL_LU_DECOMP);
		torch::logical_and_out(gnstep, m_pVars->stepmask2, m_pVars->stepmask3);
	}

	//debug_print(false, false, true);

	//std::cout << "pGN: " << pGN << std::endl;
	//std::cout << "unscaled pGN: " << torch::bmm(inv_scale_matrix, pGN) << std::endl;
	//std::cout << "gnstep: " << gnstep << std::endl;

	// occupied - square1, square2, square3, plike4, plike1, deltalike1, stepmask1

	// CALCULATE CAUCHY-STEP
	torch::Tensor& pCP = m_pVars->plike2;
	torch::Tensor& scaled_cp_norm = m_pVars->deltalike2;
	torch::Tensor& cpstep = m_pVars->stepmask2;
	{

		torch::Tensor& g = pCP; // shadow pCP since it won't be used yet
		torch::bmm_out(g, invD, gs);

		torch::Tensor& invDg = gs; // shadow gs since it isn't used anymore
		torch::bmm_out(invDg, invD, g);

		torch::Tensor& lambdaStar = scaled_cp_norm; // shadow scaled_cp_norm since we won't use it yet
		{
			torch::Tensor& lambdaStar1 = m_pVars->deltalike3;
			torch::Tensor& lambdaStar2 = m_pVars->deltalike4;

			torch::square_out(m_pVars->plike3, g);
			torch::sum_out(lambdaStar1.unsqueeze_(-1), m_pVars->plike3, 1).squeeze_(-1);
			torch::bmm_out(m_pVars->plike3, Hs, invDg);
			torch::bmm_out(lambdaStar2.unsqueeze_(-1).unsqueeze_(-1), invDg.transpose(1, 2), m_pVars->plike3).squeeze_(-1).squeeze_(-1);

			torch::div_out(lambdaStar, lambdaStar1, lambdaStar2);
		}
		
		torch::neg_out(m_pVars->plike3, g);
		m_pVars->plike3.mul_(lambdaStar.unsqueeze(-1).unsqueeze(-1));

		// Scale cauchy step
		torch::bmm_out(pCP, m_pVars->scale_matrix, m_pVars->plike3);

		torch::frobenius_norm_out(scaled_cp_norm.unsqueeze_(-1), pCP, 1).squeeze_(-1);

		// All problems with cauchy point outside trust region and problems with singular hessian should take a steepest descent step (sets cpstep)
		{ // cpstep = (!gnstep AND |pCP| > delta) OR luinfo != SUCCESSFULL
			torch::logical_not_out(cpstep, gnstep);
			torch::greater_out(m_pVars->stepmask3, scaled_cp_norm, m_pVars->delta);
			torch::logical_and_out(m_pVars->stepmask4, m_pVars->stepmask3, cpstep);
			torch::ne_out(m_pVars->stepmask3, m_pVars->luinfo, SUCCESSFULL_LU_DECOMP);
			torch::logical_or_out(cpstep, m_pVars->stepmask3, m_pVars->stepmask4);
		}

		// All problems that should take a cpstep should be scaled to trust region
		{ // pCP *= cpstep * delta / |pCP| + !cpstep
			torch::mul_out(m_pVars->deltalike3, m_pVars->delta, cpstep);
			torch::logical_not_out(m_pVars->stepmask3, cpstep);
			m_pVars->deltalike3.div_(scaled_cp_norm).add_(m_pVars->stepmask3);
			pCP.mul_(m_pVars->deltalike3.unsqueeze(-1).unsqueeze(-1));
		}
	}

	//debug_print(false, false, true);

	/*std::cout << "pCP: " << pCP << std::endl;
	std::cout << "cpstep: " << cpstep << std::endl;
	std::cout << "pCP Norm: " << scaled_cp_norm << std::endl;*/

	// occupied - plike1, deltalike1, stepmask1, plike2, deltalike2, stepmask2


	torch::Tensor& pIP = m_pVars->plike3; // g isn't used anymore, reuse it's memory;
	torch::Tensor& ipstep = m_pVars->stepmask3;
	// CALCULATE INTERMEDIATE-STEP
	{
		torch::Tensor& GN_CP = pIP; // shadow pIP since it isn't used yet
		torch::sub_out(GN_CP, pGN, pCP);

		torch::Tensor& C = m_pVars->deltalike3;
		{
			torch::square_out(m_pVars->deltalike4, scaled_cp_norm);
			torch::square_out(m_pVars->deltalike5, m_pVars->delta);
			torch::sub_out(C, m_pVars->deltalike4, m_pVars->deltalike5);
		}

		torch::Tensor& B = m_pVars->deltalike4;
		{
			torch::mul_out(m_pVars->plike4, pCP, GN_CP);
			m_pVars->plike4.mul_(2.0f);
			torch::sum_out(B.unsqueeze_(-1), m_pVars->plike4, 1).squeeze_(-1);
		}

		torch::Tensor& A = m_pVars->deltalike5;
		torch::square_out(m_pVars->plike4, GN_CP);
		torch::sum_out(A.unsqueeze_(-1), m_pVars->plike4, 1).squeeze_(-1);

		torch::Tensor& k = scaled_cp_norm; // shadow scaled_cp_norm since it isn't used anymore
		{
			torch::square_out(k, B); // k = B^2
			C.mul_(A).mul_(-4.0f); // C = -4.0f * A * C
			k.add_(C).sqrt_(); //k = sqrt(B ^ 2 - 4.0f * A * C)
			k.sub_(B).div_(A).mul_(0.5f);  // k = 0.5*(-B + sqrt(B^2 - 4.0f * A * C)) / A
		}

		torch::mul_out(m_pVars->plike4, k.unsqueeze(-1).unsqueeze(-1), GN_CP);
		torch::add_out(pIP, pCP, m_pVars->plike4);

		// All problems not taking steepest descent steps or full gn steps should take an interpolated step
		torch::logical_not_out(ipstep, torch::logical_or(gnstep, cpstep));

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
	torch::Tensor& p = m_pVars->plike4;
	pGN.mul_(gnstep.unsqueeze(-1).unsqueeze(-1)).nan_to_num_(0.0, 0.0, 0.0);
	pCP.mul_(cpstep.unsqueeze(-1).unsqueeze(-1)).nan_to_num_(0.0, 0.0, 0.0);
	pIP.mul_(ipstep.unsqueeze(-1).unsqueeze(-1)).nan_to_num_(0.0, 0.0, 0.0);
	torch::add_out(p, pCP, pIP);
	torch::add_out(pCP, p, pGN);
	torch::bmm_out(p, m_pVars->inv_scale_matrix, pCP);

}

void tc::optim::MP_STRP::step()
{
	torch::InferenceMode im_guard;
	//debug_print(true, false);

	pModel->res_jac(m_pVars->res, m_pVars->J, data);

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
	torch::Tensor& p = m_pVars->plike4;
	torch::Tensor& gnstep = m_pVars->stepmask1;
	torch::Tensor& cpstep = m_pVars->stepmask2;
	torch::Tensor& ipstep = m_pVars->stepmask3;
	torch::Tensor& scaled_gn_norm = m_pVars->deltalike1;

	//std::cout << "p: " << p << std::endl;

	torch::Tensor& x_last = m_pVars->plike3;
	x_last.copy_(pModel->parameters().unsqueeze(-1));

	torch::Tensor& ep = m_pVars->deltalike2;
	{
		torch::square_out(m_pVars->reslike1, m_pVars->res);
		torch::sum_out(ep, m_pVars->reslike1, 1).mul_(0.5f);
	}

	pModel->parameters().add_(p.squeeze(-1));
	
	

	// residuals at trailing point
	torch::Tensor& res_tp = m_pVars->reslike1;

	pModel->res(res_tp, data);

	torch::Tensor& et = m_pVars->deltalike3;
	{
		res_tp.square_();
		torch::sum_out(et, res_tp, 1).mul_(0.5f);
	}

	ep.sub_(et); // now holds actual
	torch::Tensor& actual = ep; // shadow ep since it won't be used after step below

	torch::Tensor& Jp = res_tp; // shadow res_tp isn't used anymore
	//Jp = torch::bmm(J, p.unsqueeze(-1)).squeeze(-1);
	torch::bmm_out(Jp.unsqueeze_(-1), m_pVars->J, p);

	torch::Tensor& predicted = et; // et isn't used anymore, reuse it's memory
	{
		//predicted = -torch::bmm(res.unsqueeze(-1).transpose(1, 2), Jp).squeeze(-1).squeeze(-1) - 0.5f * torch::square(Jp).sum(1);
		torch::bmm_out(m_pVars->deltalike4.unsqueeze_(-1).unsqueeze_(-1), m_pVars->res.unsqueeze(-1).transpose(1, 2), Jp).squeeze_(-1).squeeze_(-1);
		torch::frobenius_norm_out(m_pVars->deltalike5.unsqueeze_(-1), Jp, 1).squeeze_(-1).mul_(0.5f);
		torch::add_out(predicted, m_pVars->deltalike4, m_pVars->deltalike5).neg_();
	}

	actual.div_(predicted);
	torch::Tensor& rho = actual; // shadow predicted since it won't be used after step below

	//std::cout << "rho: " << rho << std::endl;

	torch::Tensor& poor_gain = m_pVars->stepmask1;
	torch::le_out(poor_gain, rho, m_pVars->mu);

	//std::cout << "poor_gain: " << poor_gain << std::endl;

	torch::Tensor& good_gain = m_pVars->stepmask2;
	torch::ge_out(good_gain, rho, m_pVars->eta);

	//std::cout << "good_gain: " << good_gain << std::endl;

	torch::Tensor& multiplier = rho; // shadow rho since it isn't used anymore
	multiplier.zero_();
	multiplier.add_(good_gain);
	multiplier.mul_(2.0f); // multiplier holds good gain

	m_pVars->deltalike4.zero_();
	m_pVars->deltalike4.add_(poor_gain);
	m_pVars->deltalike4.mul_(0.5f);
	multiplier.add_(m_pVars->deltalike4); // adds poor gain
	
	good_gain.logical_or_(poor_gain); // all problems with good or poor gain
	good_gain.logical_not_(); // all probelms with neutral gain

	multiplier.add_(good_gain);
	m_pVars->delta.mul_(multiplier); // multiply 

	// If delta is bigger than norm of gauss newton step we should decrease it below GN step
	torch::lt_out(m_pVars->stepmask3, scaled_gn_norm, m_pVars->delta);
	torch::logical_and_out(m_pVars->stepmask4, m_pVars->stepmask3, poor_gain);
	
	torch::div_out(m_pVars->deltalike4, scaled_gn_norm, m_pVars->delta);
	m_pVars->deltalike4.mul_(0.5f);
	m_pVars->deltalike4.mul_(m_pVars->stepmask4); // multiplier for all problems with |pGN| < delta and poor gain, multiplier = 0.5 * |pGN|

	// multiplier for all problems with |pGN| > delta or non poor gain ratio we should not decrease delta i.e multiplier = 1
	torch::logical_not_out(m_pVars->stepmask3, m_pVars->stepmask4);
	m_pVars->deltalike4.add_(m_pVars->stepmask3);

	// Now we can multiply delta with multiplier
	m_pVars->delta.mul_(m_pVars->deltalike4);

	// Step for all problems which have non poor gain-ratio
	torch::logical_not_out(m_pVars->stepmask3, poor_gain);
	p.mul_(m_pVars->stepmask3.unsqueeze(-1).unsqueeze(-1));
	// We don't step for Inf, -Inf, NaN
	p.nan_to_num_(0.0, 0.0, 0.0);

	//std::cout << "p: " << p << std::endl;

	// Update the model with our new parameters
	torch::add_out(pModel->parameters(), x_last.squeeze(-1), p.squeeze(-1));

	//std::cout << "param: " << m_pModel->getParameters() << std::endl;

	// make sure all vars are back to shape for next iteration
	//plike4.unsqueeze_(-1);
	m_pVars->reslike1.squeeze_(-1);

	//debug_print(true, false);

	// multiplier stored in				deltalike
	// last step stored in				plike4
	// last jacobian stored in			J
	// last residuals stored in			res

}


void tc::optim::MP_STRP::solve()
{
	torch::InferenceMode im_guard;

	for (tc::ui32 iter = 0; iter < maxiter + 1; ++iter) {
		step();

		if (MP_Optimizer::should_stop())
			break;
		MP_Optimizer::set_n_iter(iter);
	}
}
