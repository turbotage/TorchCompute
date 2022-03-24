#include "../../pch.hpp"

#include "mp_slm.hpp"


tc::optim::MP_SLMSettings::MP_SLMSettings(MP_SLMSettings&& settings)
	: MP_OptimizerSettings(std::move(settings)),
	start_residuals(settings.start_residuals),
	start_jacobian(settings.start_jacobian),
	start_lambdas(settings.start_lambdas),
	scaling(settings.scaling),
	mu(settings.mu), eta(settings.eta),
	upmul(settings.upmul), downmul(settings.downmul)

{
}

tc::optim::MP_SLMSettings::MP_SLMSettings(MP_OptimizerSettings&& optimsettings, const torch::Tensor& start_residuals, const torch::Tensor& start_jacobian, 
	const torch::Tensor& start_lambdas, const torch::Tensor& scaling, float mu, float eta, float upmul, float downmul)
	: MP_OptimizerSettings(std::move(optimsettings)),
	start_residuals(start_residuals),
	start_jacobian(start_jacobian),
	start_lambdas(start_lambdas),
	scaling(scaling),
	mu(mu), eta(eta),
	upmul(upmul),
	downmul(downmul)
{
}

std::unique_ptr<tc::optim::MP_SLMVars> tc::optim::MP_SLMVars::make(std::unique_ptr<optim::MP_Model>& pModel, torch::Tensor& data, torch::Tensor& residuals, 
	torch::Tensor& jacobian, torch::Tensor& lambda, torch::Tensor& scaling, float mu, float eta, float upmul, float downmul)
{
	return std::unique_ptr<MP_SLMVars>(new MP_SLMVars(pModel, data, residuals, jacobian, lambda, scaling, mu, eta, upmul, downmul));
}

void tc::optim::MP_SLMVars::to_device(const torch::Device& device)
{
	torch::InferenceMode im_guard;

	res.to(device);
	reslike1.to(device);

	lambda.to(device);
	lambdalike1.to(device);
	lambdalike2.to(device);
	lambdalike3.to(device);

	J.to(device);

	plike1.to(device);
	plike2.to(device);

	square1.to(device);
	square2.to(device);
	square3.to(device);

	info.to(device);
	pivots.to(device);

	scaling.to(device);

	stepmask1.to(device);
	stepmask2.to(device);
	stepmask3.to(device);
}

void tc::optim::MP_SLMVars::to_float32()
{
	torch::InferenceMode im_guard;

	res.to(torch::ScalarType::Float);
	reslike1.to(torch::ScalarType::Float);

	lambda.to(torch::ScalarType::Float);
	lambdalike1.to(torch::ScalarType::Float);
	lambdalike2.to(torch::ScalarType::Float);
	lambdalike3.to(torch::ScalarType::Float);

	J.to(torch::ScalarType::Float);

	plike1.to(torch::ScalarType::Float);
	plike2.to(torch::ScalarType::Float);

	square1.to(torch::ScalarType::Float);
	square2.to(torch::ScalarType::Float);
	square3.to(torch::ScalarType::Float);

	scaling.to(torch::ScalarType::Float);
}

void tc::optim::MP_SLMVars::to_float64()
{
	torch::InferenceMode im_guard;

	res.to(torch::ScalarType::Double);
	reslike1.to(torch::ScalarType::Double);

	lambda.to(torch::ScalarType::Double);
	lambdalike1.to(torch::ScalarType::Double);
	lambdalike2.to(torch::ScalarType::Double);
	lambdalike3.to(torch::ScalarType::Double);

	J.to(torch::ScalarType::Double);

	plike1.to(torch::ScalarType::Double);
	plike2.to(torch::ScalarType::Double);

	square1.to(torch::ScalarType::Double);
	square2.to(torch::ScalarType::Double);
	square3.to(torch::ScalarType::Double);

	scaling.to(torch::ScalarType::Double);
}

void tc::optim::MP_SLMVars::debug_print(bool sizes, bool types, bool values)
{
	torch::InferenceMode im_guard;

	std::cout << "<=================== BEGIN DEBUG-PRINT =====================>\n";

	if (sizes) {
		std::cout << "res: " << res.sizes() << std::endl;
		std::cout << "reslike1: " << reslike1.sizes() << std::endl;

		std::cout << "lambda: " << lambda.sizes() << std::endl;
		std::cout << "lambdalike1: " << lambdalike1.sizes() << std::endl;
		std::cout << "lambdalike2: " << lambdalike2.sizes() << std::endl;
		std::cout << "lambdalike3: " << lambdalike3.sizes() << std::endl;

		std::cout << "J: " << J.sizes() << std::endl;

		std::cout << "plike1: " << plike1.sizes() << std::endl;
		std::cout << "plike2: " << plike2.sizes() << std::endl;

		std::cout << "square1: " << square1.sizes() << std::endl;
		std::cout << "square2: " << square2.sizes() << std::endl;
		std::cout << "square3: " << square3.sizes() << std::endl;

		std::cout << "info: " << info.sizes() << std::endl;
		std::cout << "pivots: " << pivots.sizes() << std::endl;

		std::cout << "scaling: " << scaling.sizes() << std::endl;

		std::cout << "stepmask1: " << stepmask1.sizes() << std::endl;
		std::cout << "stepmask2: " << stepmask2.sizes() << std::endl;
		std::cout << "stepmask3: " << stepmask3.sizes() << std::endl;
	}

	if (types) {
		std::cout << "res: " << res.dtype() << std::endl;
		std::cout << "reslike1: " << reslike1.dtype() << std::endl;

		std::cout << "lambda: " << lambda.dtype() << std::endl;
		std::cout << "lambdalike1: " << lambdalike1.dtype() << std::endl;
		std::cout << "lambdalike2: " << lambdalike2.dtype() << std::endl;
		std::cout << "lambdalike3: " << lambdalike3.dtype() << std::endl;

		std::cout << "J: " << J.dtype() << std::endl;

		std::cout << "plike1: " << plike1.dtype() << std::endl;
		std::cout << "plike2: " << plike2.dtype() << std::endl;

		std::cout << "square1: " << square1.dtype() << std::endl;
		std::cout << "square2: " << square2.dtype() << std::endl;
		std::cout << "square3: " << square3.dtype() << std::endl;

		std::cout << "info: " << info.dtype() << std::endl;
		std::cout << "pivots: " << pivots.dtype() << std::endl;

		std::cout << "scaling: " << scaling.dtype() << std::endl;

		std::cout << "stepmask1: " << stepmask1.dtype() << std::endl;
		std::cout << "stepmask2: " << stepmask2.dtype() << std::endl;
		std::cout << "stepmask3: " << stepmask3.dtype() << std::endl;
	}

	if (values) {
		std::cout << "res: " << res << std::endl;
		std::cout << "reslike1: " << reslike1 << std::endl;

		std::cout << "lambda: " << lambda << std::endl;
		std::cout << "lambdalike1: " << lambdalike1 << std::endl;
		std::cout << "lambdalike2: " << lambdalike2 << std::endl;
		std::cout << "lambdalike3: " << lambdalike3 << std::endl;

		std::cout << "J: " << J << std::endl;

		std::cout << "plike1: " << plike1 << std::endl;
		std::cout << "plike2: " << plike2 << std::endl;

		std::cout << "square1: " << square1 << std::endl;
		std::cout << "square2: " << square2 << std::endl;
		std::cout << "square3: " << square3 << std::endl;

		std::cout << "info: " << info << std::endl;
		std::cout << "info: " << info.dtype() << std::endl;

		std::cout << "scaling: " << scaling << std::endl;

		std::cout << "stepmask1: " << stepmask1 << std::endl;
		std::cout << "stepmask2: " << stepmask2 << std::endl;
		std::cout << "stepmask3: " << stepmask3 << std::endl;
	}

	std::cout << "<=================== END DEBUG-PRINT =====================>\n";
}

tc::optim::MP_SLMVars::MP_SLMVars(const std::unique_ptr<optim::MP_Model>& pModel, const torch::Tensor& data, torch::Tensor& residuals, 
	torch::Tensor& jacobian, torch::Tensor& lambda, torch::Tensor& scaling, float mu, float eta, float upmul, float downmul)
{
	torch::InferenceMode im_guard;

	auto dops = pModel->parameters().options();

	this->mu = mu;
	this->eta = eta;
	this->upmul = upmul;
	this->downmul = downmul;

	numProbs = data.size(0);
	numParam = pModel->parameters().size(1);
	numData = data.size(1);

	this->res = residuals;
	reslike1 = torch::empty_like(res);

	this->lambda = lambda;
	lambdalike1 = torch::empty_like(lambda);
	lambdalike2 = torch::empty_like(lambda);
	lambdalike3 = torch::empty_like(lambda);
	
	this->J = jacobian;

	plike1 = torch::empty({ numProbs, numParam, 1 }, dops);
	plike2 = torch::empty({ numProbs, numParam, 1 }, dops);

	square1 = torch::empty({ numProbs, numParam, numParam }, dops);
	square2 = torch::empty_like(square1);
	square3 = torch::empty_like(square1);

	info = torch::empty({ numProbs }, dops.dtype(torch::ScalarType::Int));
	pivots = torch::empty({ numProbs, numParam }, dops.dtype(torch::ScalarType::Int));

	this->scaling = scaling;

	stepmask1 = torch::empty({ numProbs }, dops.dtype(torch::ScalarType::Bool));
	stepmask2 = torch::empty_like(stepmask1);
	stepmask3 = torch::empty_like(stepmask1);
}




std::unique_ptr<tc::optim::MP_SLM> tc::optim::MP_SLM::make(MP_SLMSettings&& settings)
{
	auto pVars = MP_SLMVars::make(settings.pModel, settings.data, settings.start_residuals,
		settings.start_jacobian, settings.start_lambdas, settings.scaling, settings.mu, settings.eta);

	return std::make_unique<MP_SLM>(std::move(settings), std::move(pVars));
}

tc::optim::MP_SLM::MP_SLM(MP_OptimizerSettings&& optsettings, std::unique_ptr<MP_SLMVars> strpvars)
	: MP_Optimizer(std::move(optsettings)), m_pVars(std::move(strpvars))
{
}

torch::Tensor tc::optim::MP_SLM::last_parameters()
{
	if (!pModel)
		throw std::runtime_error("Tried to get parameters on optimizer where OptimResult had been acquired");

	return pModel->parameters();
}

torch::Tensor tc::optim::MP_SLM::last_step()
{
	if (!m_pVars)
		throw std::runtime_error("Tried to get last_step on optimizer where vars had been aquired");

	return m_pVars->plike2;
}

torch::Tensor tc::optim::MP_SLM::last_jacobian()
{
	if (!m_pVars)
		throw std::runtime_error("Tried to get last_jacobian on optimizer where vars had been aquired");

	return m_pVars->J;
}

torch::Tensor tc::optim::MP_SLM::last_residuals()
{
	if (!m_pVars)
		throw std::runtime_error("Tried to get last_residuals on optimizer where vars had been aquired");

	return m_pVars->res;
}

torch::Tensor tc::optim::MP_SLM::last_lambdas()
{
	if (!m_pVars)
		throw std::runtime_error("Tried to get last_deltas on optimizer where vars had been aquired");

	return m_pVars->lambda;
}

torch::Tensor tc::optim::MP_SLM::last_multiplier()
{
	if (!m_pVars)
		throw std::runtime_error("Tried to get last_multiplier on optimizer where vars had been aquired");

	// After solve has been run, the last multipliers are stored in deltalike2
	return m_pVars->lambdalike2;
}

torch::Tensor tc::optim::MP_SLM::last_scaling()
{
	if (!m_pVars)
		throw std::runtime_error("Tried to get last_scaling on optimizer where vars had been aquired");

	return m_pVars->scaling;
}

std::unique_ptr<tc::optim::MP_SLMVars> tc::optim::MP_SLM::acquire_vars()
{
	return std::move(m_pVars);
}

torch::Tensor tc::optim::MP_SLM::default_lambda_setup(const torch::Tensor& parameters, float multiplier)
{
	return multiplier * torch::ones({ parameters.size(0) }, parameters.options());
}

torch::Tensor tc::optim::MP_SLM::default_scaling_setup(const torch::Tensor& J, float minimum_scale)
{
	torch::InferenceMode im_guard;
	auto diag = torch::diagonal(torch::bmm(J.transpose(1, 2), J), 0, -2, -1);
	return torch::clamp(diag, minimum_scale);
}

std::pair<torch::Tensor, torch::Tensor> tc::optim::MP_SLM::default_res_J_setup(optim::MP_Model& model, const torch::Tensor& data)
{
	torch::InferenceMode im_guard;

	torch::Tensor& pars = model.parameters();

	torch::Tensor J = torch::empty({ pars.size(0), data.size(1), pars.size(1) }, pars.options());
	torch::Tensor res = torch::empty_like(data);

	model.res_jac(res, J, data);

	return std::make_pair(res, J);
}

void tc::optim::MP_SLM::on_run(tc::ui32 iter)
{
	if (!pModel)
		throw std::runtime_error("Tried to run() on STRPOptimizer where model had been acquired");

	if (!m_pVars)
		throw std::runtime_error("Tried to run() on STRPOptimizer where vars had been acquired");

	solve(iter);
}

void tc::optim::MP_SLM::on_acquire_model()
{
}

void tc::optim::MP_SLM::on_abort()
{
}

void tc::optim::MP_SLM::step()
{
	torch::InferenceMode im_guard;

	//m_pVars->debug_print(true, false, false);

	pModel->res_jac(m_pVars->res, m_pVars->J, data);

	m_pVars->J.nan_to_num_(0.0f, 0.0f, 0.0f);

	torch::Tensor& H = m_pVars->square1;
	torch::bmm_out(H, m_pVars->J.transpose(1, 2), m_pVars->J);

	torch::max_out(m_pVars->scaling, m_pVars->scaling, torch::diagonal(H, 0, -2, -1));

	torch::mul_out(m_pVars->square2, m_pVars->lambda.unsqueeze(-1).unsqueeze(-1), torch::diag_embed(m_pVars->scaling));
	// TODO: consider adding minimum damping term -> reducing risk of parameter evaporation
	
	torch::add_out(m_pVars->square3, m_pVars->square1, m_pVars->square2);

	torch::Tensor& g = m_pVars->plike1;
	torch::bmm_out(g, m_pVars->J.transpose(1, 2), m_pVars->res.unsqueeze(-1));

	/*
	if (m_pVars->numParam < 5) {
		std::tie(m_pVars->square2, m_pVars->pivots, m_pVars->info) = at::_lu_with_info(H, data.device().is_cpu() ? true : false, false);
		torch::lu_solve_out(m_pVars->plike2, g.neg(), m_pVars->square2, m_pVars->pivots);
	}
	else {
		torch::linalg_cholesky_ex_out(m_pVars->square2, m_pVars->info, m_pVars->square3);
		torch::cholesky_solve_out(m_pVars->plike2, g.neg(), m_pVars->square2);
	}
	*/
	
	std::tie(m_pVars->square2, m_pVars->pivots, m_pVars->info) = at::_lu_with_info(m_pVars->square3, data.device().is_cpu() ? true : false, false);
	torch::lu_solve_out(m_pVars->plike2, g.neg(), m_pVars->square2, m_pVars->pivots);

	//torch::linalg_cholesky_ex_out(m_pVars->square2, m_pVars->info, m_pVars->square3);
	//torch::cholesky_solve_out(m_pVars->plike2, g.neg(), m_pVars->square2);

	torch::Tensor& ep = m_pVars->lambdalike1;
	torch::square_out(m_pVars->reslike1, m_pVars->res);
	torch::sum_out(ep, m_pVars->reslike1, 1);
	ep.mul_(0.5f);

	m_pVars->plike1.copy_(pModel->parameters().unsqueeze(-1));
	torch::add_out(pModel->parameters(), m_pVars->plike1.squeeze(-1), m_pVars->plike2.squeeze(-1));
	pModel->res(m_pVars->reslike1, data);

	torch::Tensor& et = m_pVars->lambdalike2;
	m_pVars->reslike1.square_();
	torch::sum_out(et, m_pVars->reslike1, 1);
	et.mul_(0.5f);

	torch::Tensor& should_step = m_pVars->stepmask1;
	torch::le_out(should_step, et, ep);

	torch::Tensor& actual = m_pVars->lambdalike1; // trail soon unused lambdalike1
	actual.sub_(et);

	torch::Tensor& JpD = m_pVars->reslike1;
	torch::bmm_out(JpD.unsqueeze(-1), m_pVars->J, m_pVars->plike2);

	torch::Tensor& predicted = m_pVars->lambdalike2; // trail soon unused lambdalike2
	torch::sum_out(predicted, m_pVars->res * JpD, 1).neg_();
	predicted.sub_(torch::square(JpD.squeeze(-1)).sum(1).mul(0.5f));

	actual.div_(predicted);
	torch::Tensor& rho = actual;

	torch::Tensor& multiplier = m_pVars->lambdalike2; // predicted isn't used anymore
	multiplier.zero_();

	torch::Tensor& poor_gain = m_pVars->stepmask2;
	torch::le_out(poor_gain, rho, m_pVars->mu);

	multiplier.add_(poor_gain);
	multiplier.mul_(m_pVars->upmul);

	torch::Tensor& good_gain = m_pVars->stepmask3;
	torch::ge_out(good_gain, rho, m_pVars->eta);

	m_pVars->lambdalike3.zero_();
	m_pVars->lambdalike3.add_(good_gain);
	m_pVars->lambdalike3.mul_(m_pVars->downmul);

	multiplier.add_(m_pVars->lambdalike3);

	multiplier.add_(poor_gain.logical_or_(good_gain).logical_not_());

	// We take a step if we have good gain or if we have objective reduction
	should_step.logical_or_(good_gain);

	//m_pVars->debug_print(true, false, false);

	//std::cout << "rho: " << rho << std::endl;
	//std::cout << "should_step: " << should_step << std::endl;

	m_pVars->plike2.mul_(should_step.unsqueeze(-1).unsqueeze(-1));
	m_pVars->plike2.nan_to_num_(0.0f, 0.0f, 0.0f);
	torch::add_out(pModel->parameters(), m_pVars->plike1.squeeze(-1), m_pVars->plike2.squeeze(-1));

	//std::cout << "param: " << pModel->parameters() << std::endl;
	//std::cout << "lambda: " << m_pVars->lambda << std::endl;

	m_pVars->lambda.mul_(multiplier);

	//m_pVars->debug_print(true, false, false);
}

void tc::optim::MP_SLM::solve(tc::ui32 maxiter)
{
	torch::InferenceMode im_guard;

	for (tc::ui32 iter = 0; iter < maxiter; ++iter) {
		step();

		if (MP_Optimizer::should_stop())
			break;
		MP_Optimizer::set_n_iter(iter);
	}
}
