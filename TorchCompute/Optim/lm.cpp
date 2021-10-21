#include "lm.hpp"

#include "../Compute/gradients.hpp"
#include <tuple>

using namespace torch::indexing;



optim::LMP::LMP(model::Model& model)
	: model(model)
{
}

void optim::LMP::setParameterGuess(torch::Tensor& parameters)
{
	this->params = parameters;
}

void optim::LMP::setDependents(torch::Tensor& dependents)
{
	this->deps = dependents;
}

void optim::LMP::setData(torch::Tensor& data)
{
	this->data = data;
}

void optim::LMP::setDefaultTensorOptions(torch::TensorOptions dops)
{
	this->dops = dops;

}

void optim::LMP::setOnIterationCallback(std::function<void()> iterationCallback)
{
	onIterationCallback = iterationCallback;
}

void optim::LMP::setOnSwitchCallback(std::function<void()> switchCallback)
{
	onSwitchCallback = switchCallback;
}

void optim::LMP::setSwitching(int switchPercentage, torch::Device& device)
{
	onSwitchPercentage = switchPercentage;
	switchDevice = device;
}

void optim::LMP::setCopyConvergingEveryN(int n)
{
	copyConvergingEveryN = n;
}

void optim::LMP::setMu(float mu)
{
	this->mu = mu;
}

void optim::LMP::setEta(float eta)
{
	this->eta = eta;
}

void optim::LMP::setTolerance(float tol)
{
	this->tol = tol;
}

void optim::LMP::setMaxIteration(int iter)
{
	this->max_iter = iter;
}


void optim::LMP::run()
{
	setup_solve();
	solve();
}

torch::Tensor optim::LMP::getParameters() {
	return params;
}

torch::Tensor optim::LMP::getNonConvergingParameters() {
	return params_slice;
}


void optim::LMP::print_all()
{
	std::cout <<
		"res: " << res << std::endl <<
		"J: " << J << std::endl <<
		"p: " << p << std::endl <<
		"pGN: " << pGN << std::endl <<
		"t: " << t << std::endl <<
		"ep: " << ep << std::endl <<
		"gs: " << gs << std::endl <<
		"q: " << q << std::endl <<
		"CP: " << CP << std::endl <<
		std::endl;
}

void optim::LMP::print_devices()
{
	std::stringstream stream;

	try {
		stream << "res: " << res.device() << std::endl;
	}
	catch (c10::Error e) {}

	try {
		stream << "J: " << J.device() << std::endl;
	}
	catch (c10::Error e) {}

	try {
		stream << "p: " << p.device() << std::endl;
	}
	catch (c10::Error e) {}

	try {
		stream << "pGN: " << pGN.device() << std::endl;
	}
	catch (c10::Error e) {}

	try {
		stream << "t: " << t.device() << std::endl;
	}
	catch (c10::Error e) {}

	try {
		stream << "ep: " << ep.device() << std::endl;
	}
	catch (c10::Error e) {}

	try {
		stream << "gs: " << gs.device() << std::endl;
	}
	catch (c10::Error e) {}

	try {
		stream << "q: " << q.device() << std::endl;
	}
	catch (c10::Error e) {}

	try {
		stream << "CP: " << CP.device() << std::endl;
	}
	catch (c10::Error e) {}

	try {
		stream << "params: " << params.device() << std::endl;
	}
	catch (c10::Error e) {}

	try {
		stream << "deps: " << deps.device() << std::endl;
	}
	catch (c10::Error e) {}

	try {
		stream << "data: " << data.device() << std::endl;
	}
	catch (c10::Error e) {}

	try {
		stream << "params_slice: " << params_slice.device() << std::endl;
	}
	catch (c10::Error e) {}

	try {
		stream << "deps_slice: " << deps_slice.device() << std::endl;
	}
	catch (c10::Error e) {}

	try {
		stream << "data_slice: " << data_slice.device() << std::endl;
	}
	catch (c10::Error e) {}

	try {
		stream << "y: " << y.device() << std::endl;
	}
	catch (c10::Error e) {}

	std::cout << stream.str();
}

void optim::LMP::switch_dops()
{
	auto dev = switchDevice.value();

	dops = dops.device(dev);

	params = params.to(dev);
	params_slice = params_slice.to(dev);

	deps = deps.to(dev);
	deps_slice = deps_slice.to(dev);

	data = data.to(dev);
	data_slice = data_slice.to(dev);
	
	nci = nci.to(dev);

	//y = y.to(dops);
	
	//res = res.to(dops);
	//rt = rt.to(dops);
	//J = J.to(dops);
	//p = p.to(dev);

	//pGN = pGN.to(dev);
	pGN_Norm = pGN_Norm.to(dev);

	mask = mask.to(dev);
	mask2 = mask2.to(dev);

	t = t.to(dev);

	ep = ep.to(dev);
	et = et.to(dev);
	Jp = Jp.to(dev);

	cFactor = cFactor.to(dev);
	cSuccess = cSuccess.to(dev);
	q = q.to(dev);
	delta = delta.to(dev);

	step.to(dev);
	
	model.setDependents(deps_slice);
	model.setParameters(params_slice);

	model.to(dev);

	onSwitchCallback();
}



torch::Tensor optim::LMP::plane_convergence()
{
	return (torch::sqrt((Jp * Jp).sum(1)) <= tol * torch::sqrt((res * res).sum(1))).view({ nci_size });
}

// Requires J, res to be set and nci_sum to reflect the correct size
void optim::LMP::dogleg()
{
	torch::NoGradGuard no_grad;

	// Calculate norms of Jacobians
	Jn2 = (J * J).sum(1);
	Jn = torch::sqrt(Jn2);

	// Scaling matrix
	D = torch::diag_embed(1 / Jn);

	// Scale variables
	Js = torch::bmm(J, D);

	// Calculate Hessian and gradient
	Hs = torch::bmm(Js.transpose(1, 2), Js);
	gs = torch::bmm(Js.transpose(1, 2), res);

	// Get Cholesky factor and find ill-conditioned problems
	std::tie(cFactor, cSuccess) = torch::linalg_cholesky_ex(Hs);
	cSuccess = (cSuccess == 0);

	// If problem has been resized or q is uninitialized
	if ((q.size(0) != nci_size) || (q.size(1) != nParams)) {
		q = torch::empty({ nci_size, nParams, 1 }, dops);
	}
	if ((p.size(0) != nci_size) || (p.size(1) != nParams)) {
		p = torch::empty({ nci_size, nParams, 1 }, dops);
	}

	// Solve well-conditioned problems
	q.index_put_({cSuccess, Slice()}, torch::cholesky_solve(
		-gs.index({ cSuccess, Slice() }), cFactor.index({ cSuccess, Slice() })));


	// Unscale solution
	pGN = torch::bmm(D, q);
	pGN_Norm = torch::sqrt((pGN * pGN).sum(1));

	// Gives well-conditioned problems with solutions within trust region
	mask = torch::logical_and(pGN_Norm.view({ nci_size }) <= delta, cSuccess);
	// Set dogleg direction to Gauss-Newton direction for well-conditioned problems
	// within trust region
	if (mask.sum().item<int64_t>() != 0) {
		p.index_put_({ mask, Slice() }, pGN.index({ mask, Slice() }));
		step.index_put_({ mask }, 1); // Indicate Gauss-Newton step
	}

	// Calculate Cauchy-Point
	invD = torch::diag_embed(Jn);
	invD2 = torch::diag_embed(Jn2);

	invDgs = torch::bmm(invD, gs);
	invD2gs = torch::bmm(invD2, gs);

	lambdaStar = torch::bmm(invDgs.transpose(1, 2), invDgs) /
		(torch::bmm(invD2gs.transpose(1, 2), torch::bmm(Hs, invD2gs)));
	
	CP = -lambdaStar * invDgs;

	// If the Cauchy Point is outside the trust region and GN step was not accepted, scale the negative gradient
	auto CP_Norm = torch::sqrt((CP * CP).sum(1));
	mask2 = torch::logical_or(torch::logical_and(CP_Norm.view({ nci_size }) > delta,
		torch::logical_not(mask)), torch::logical_not(cSuccess));

	int64_t mask2sum = mask2.sum().item<int64_t>();
	if (mask2sum != 0) {
		auto invDgs_slice = invDgs.index({mask2, Slice()});
		auto invDgsNorm = torch::sqrt((invDgs_slice * invDgs_slice).sum(1));

		p.index_put_({ mask2, Slice() }, -(invDgs_slice / invDgsNorm.view({mask2sum, 1, 1})) *
			delta.index({ mask2 }).view({ mask2sum, 1, 1 }));

		step.index_put_({mask2}, 3);
	}

	// If neither Gauss-Newton nor Cauchy step is accepted, find the intersection of line CP-pGN and circle with radius delta
	mask = torch::logical_not(torch::logical_or(mask, mask2));
	int64_t masksum = mask.sum().item<int64_t>();
	if (masksum != 0) {

		auto CP_mask = CP.index({ mask, Slice() });
		auto pGN_mask = pGN.index({ mask, Slice() });

		// Calc coefficients
		torch::Tensor A = (CP_mask - pGN_mask).pow(2).sum(1);
		torch::Tensor B = (2.0 * CP_mask * (pGN_mask - CP_mask)).sum(1);
		torch::Tensor C = (CP_mask * CP_mask).sum(1) - delta.index({ mask }).pow(2).view({masksum, 1});

		// Positive root
		torch::Tensor k = (-B + torch::sqrt(B * B - 4 * A * C))/(2.0*A);

		// Intersection
		p.index_put_({ mask, Slice() }, CP_mask + k.view({masksum, 1, 1}) * (pGN_mask - CP_mask));
		// Set interpolated step
		step.index_put_({mask}, 2);
	}

	//return;
}

int optim::LMP::handle_convergence()
{
	torch::NoGradGuard no_grad;

	// Copy param_slice into params
	params.index_put_({ nci, Slice() }, params_slice.detach());

	// Check convergence
	converges = plane_convergence();
	auto not_converges = torch::logical_not(converges);

	nc_sum = not_converges.sum().item<int64_t>();

	// If all pixels have converged we can return
	if (nc_sum == 0) return 1;

	nci = nci.index({ not_converges });
	nci_size = nci.size(0);

	// Reset slices to exclude already converging pixels
	deps_slice = deps.index({ nci, Slice() });
	params_slice = params.index({ nci, Slice() });
	data_slice = data.index({ nci, Slice() });

	std::cout << "Number of solving problems: " << params_slice.size(0) << std::endl;

	delta = delta.index({ not_converges });
	step = step.index({ not_converges });

	//Set new deps_slice as model dependents
	model.setDependents(deps_slice);

	return 0;
}

void optim::LMP::next_step_iter()
{
	torch::NoGradGuard no_grad;

	res = (y - data_slice).view({ nci_size, nDependents, 1 });
	ep = 0.5 * torch::bmm(res.transpose(1, 2), res);

	// Set dogleg search directioen and Gauss-Newton step, step types

	dogleg();

	Jp = torch::bmm(J, p);

	// Calculate trailing point, new residuals ...
	t = params_slice + p.view({ nci_size, nParams });
	model.setParameters(t);
	rt = (model() - data_slice).view({ nci_size, nDependents, 1 });
	et = 0.5 * torch::bmm(rt.transpose(1, 2), rt);

	//std::cout << "p:" << p << std::endl;
	//std::cout << "params" << params << std::endl;

	// Calculate gain ratio
	predicted = -torch::bmm(res.transpose(1, 2), Jp) - 0.5 * torch::bmm(Jp.transpose(1, 2), Jp);
	actual = ep - et;
	rho = (actual / predicted).view({ nci_size });
	//std::cout << "rho: " << rho << std::endl;

	// Evaluate gain ratio
	mask = rho <= mu;
	not_mask = torch::logical_not(mask);

	// For a poor gain ratio, decrease trust region and discard trail point
	if (mask.sum().item<int64_t>() != 0) {
		delta.index_put_({ mask }, delta.index({ mask }) * 0.5);
		step.index_put_({ mask }, 0);

		mask = torch::logical_and(mask, delta > pGN_Norm.view({ nci_size }));
		int64_t masksum = mask.sum().item<int64_t>();
		if (masksum != 0) {
			auto new_delta = torch::pow(2, torch::ceil(
				torch::log2(delta.index({ mask }) / pGN_Norm.index({ mask }).view({ masksum }))));

			delta.index_put_({ mask }, delta.index({ mask }) / new_delta);
		}
	}

	// If the gain ratio is sufficient high, save trial point as new guess
	if (not_mask.sum().item<int64_t>() != 0) {
		params_slice.index_put_({ not_mask }, t.index({ not_mask, Slice() }));
	}

	mask = rho >= eta;
	// If the gain ratio is sufficient high, increase the trust region
	if (mask.sum().item<int64_t>() != 0) {
		delta.index_put_({ mask }, delta.index({ mask }) * 2.0);
	}
}

void optim::LMP::setup_solve()
{
	nProblems = params.size(0);
	nParams = params.size(1);
	nDependents = deps.size(1);

	nci = torch::arange(0, nProblems, dops).to(c10::ScalarType::Long);
	nci_size = nProblems;
	nc_sum = nProblems;

	delta = torch::sqrt((params * params).sum(1));
	step = torch::empty(nProblems, dops);

	deps_slice = deps.index({ nci, Slice() });
	params_slice = params.index({ nci, Slice() });
	data_slice = data.index({ nci, Slice() });


	model.setDependents(deps_slice);
}

void optim::LMP::solve()
{

	int iteration = 0;
	do {
		if (onIterationCallback) {
			onIterationCallback();
		}

		// Copy converging pixels every n iterations
		if ((iteration % copyConvergingEveryN == 0) && iteration > 0 ) {
			if (handle_convergence() == 1) return;
		}

		// Check if we should swith to the final compute device
		if ((((float)nProblems - (float)nc_sum) / (float)nProblems) > (onSwitchPercentage*0.01)) {
			if (!hasSwitched) {
				switch_dops();
				hasSwitched = true;
			}
		}

		// Calculate Jacobian on current point
		params_slice.requires_grad_(true);

		model.setParameters(params_slice);
		y = model();
		J = compute::jacobian(y, params_slice).detach();

		params_slice.requires_grad_(false);
		y = y.detach();
		//===========================================

		// Does all the work
		next_step_iter();
		// ==========================================

		// Increase iteration counter
		++iteration;
	} while (iteration < max_iter);


	// Copy last params_slice to params
	{
		torch::NoGradGuard no_grad;
		params.index_put_({ nci, Slice() }, params_slice.detach());
	}

}
