#include "../pch.hpp"

#include "slma.hpp"

tc::optim::SLMASettings::SLMASettings()
{

}

tc::optim::SLMA::SLMA(SLMASettings& settings)
	: m_CurrentDevice(settings.startDevice), m_SwitchDevice(settings.switchDevice), 
	m_SwitchNumber(settings.switchAtN), Optimizer(settings)
{
	
}

tc::optim::SLMAResult tc::optim::SLMA::eval()
{
	Optimizer::on_eval();

	solve();

	SLMAResult res;
	res.finalParameters = m_Parameters;
	res.pFinalModel = std::move(m_pModel);
	res.nonConvergingIndices = nci;

	return res;
}

void tc::optim::SLMA::step()
{
	using namespace torch::indexing;

	// Compute J,res
	res = res.view({ numProbs, numInputs });
	m_pModel->res_diff(res, J, data_slice);
	res = res.view({ numProbs, numInputs, 1 });

	torch::InferenceMode im_guard;

	//Objective function value at current point
	ep = 0.5f * torch::square(res).sum(1).view({ numProbs });
	g_norm = torch::sqrt(torch::square(torch::bmm(J.transpose(1, 2), res)).sum(1)).view({numProbs});

	torch::Tensor Jn = torch::sqrt(torch::square(J).sum(1));

	torch::Tensor D = torch::diag_embed(torch::reciprocal(Jn));
	torch::Tensor Js = torch::bmm(J, D);

	// Scaled gradient
	torch::Tensor gs = torch::bmm(Js.transpose(1, 2), res);
	// Scaled Hessian
	torch::Tensor Hs = torch::bmm(Js.transpose(1, 2), Js);

	Hs = Hs + lambda.unsqueeze(1).unsqueeze(2) * torch::diag_embed(Hs.diagonal(0, -2, -1));

	torch::Tensor Hs_chol;
	std::tie(Hs_chol, step_mask) = torch::linalg_cholesky_ex(Hs);

	torch::Tensor chol_mask = step_mask == eMaskTypes::SUCCESSFUL_CHOLESKY;

	pD.index_put_({chol_mask, Slice()},
		torch::bmm(D.index({chol_mask, Slice() }), torch::cholesky_solve(-gs.index({chol_mask, Slice() }),
			Hs_chol.index({chol_mask, Slice() }))));

	chol_mask = torch::logical_not(chol_mask);

	pD.index_put_({chol_mask, Slice()},
		torch::bmm(torch::diag_embed(Jn.index({chol_mask})), -gs.index({chol_mask})) / 
		lambda.index({chol_mask}).unsqueeze(1).unsqueeze(2));

	step_mask.index_put_({ chol_mask }, eMaskTypes::UNSUCCESSFUL_CHOLESKY);

	JpD = torch::bmm(J, pD);

	torch::Tensor currentParams = m_pModel->getParameters();

	// Update the parameters
	m_pModel->setParameters(m_pModel->getParameters() + pD.view({ numProbs, numParams }));

	res_t = res_t.view({ numProbs, numInputs });
	m_pModel->res(res_t, data_slice);
	res_t = res_t.view({ numProbs, numInputs, 1 });

	//Objective function value at current point
	torch::Tensor et = 0.5f * torch::square(res_t).sum(1).view({ numProbs });

	torch::Tensor gain_mask = et < ep;

	lambda.index_put_({ gain_mask }, lambda.index({ gain_mask }) * m_Decrease);
	step_mask.index_put_({ gain_mask }, step_mask.index({ gain_mask }).bitwise_or(eMaskTypes::LAMBDA_DECREASED));

	gain_mask = torch::logical_not(gain_mask);
	lambda.index_put_({ gain_mask }, lambda.index({ gain_mask }) * m_Increase);
	step_mask.index_put_({ gain_mask }, step_mask.index({ gain_mask }).bitwise_or(eMaskTypes::LAMBDA_INCREASED));

	m_pModel->getParameters().index_put_({ gain_mask, Slice()}, currentParams.index({gain_mask, Slice()}));

	//std::cout << "pD:\n" << pD << std::endl;
	//std::cout << "lambda:\n" << lambda << std::endl;
	//std::cout << "step:\n" << step_mask << std::endl;

}

bool tc::optim::SLMA::handle_convergence()
{
	using namespace torch::indexing;

	torch::NoGradGuard no_grad_guard;

	// plane convergence
	torch::Tensor converges = torch::sqrt(torch::square(JpD).sum(1).view({ numProbs })) <=
		m_Tolerance * (1 + torch::sqrt(torch::square(res).sum(1).view({ numProbs })));

	converges = torch::logical_and(converges, g_norm <= m_Tolerance * 100.0f * (1.0f + ep));

	// We only converge when damping being turned off
	converges = torch::logical_and(converges, step_mask == eMaskTypes::LAMBDA_DECREASED);

	// Copy the converging problems back to the final parameter tensor
	m_Parameters.index_put_({ nci.index({converges}), Slice() },
		m_pModel->getParameters().index({ converges, Slice() }));

	// Recreate the index list for the pixels that don't converge
	torch::Tensor not_converges = torch::logical_not(converges);

	// Extract and set non-converging problems

	m_pModel->setParameters(m_pModel->getParameters().index({ not_converges, Slice() }));

	if (m_pModel->hasPerProblemInputs()) {
		if (m_pModel->getPerProblemInputs().numel() != 0) {
			m_pModel->setPerProblemInputs(m_pModel->getPerProblemInputs().index({ not_converges, Slice(), Slice() }));
		}
	}

	nci = nci.index({ not_converges });
	numProbs = nci.size(0); // The new number of problems

	data_slice = data_slice.index({ not_converges, Slice() });

	res = res.index({ not_converges, Slice(), Slice() });
	res_t = res_t.index({ not_converges, Slice(), Slice() });
	ep = ep.index({ not_converges });
	g_norm = g_norm.index({ not_converges });
	

	pD = pD.index({ not_converges, Slice(), Slice() });

	J = J.index({ not_converges, Slice(), Slice() });
	lambda = lambda.index({ not_converges });

	step_mask = step_mask.index({ not_converges });

	if (numProbs == 0) // if no non-converging pixels are left we can return
		return true;
	return false;

}

void tc::optim::SLMA::switch_device()
{
	if (!m_SwitchDevice.has_value())
		return;

	torch::NoGradGuard no_grad_guard;
	m_HasSwitched = true;

	torch::Device& dev = m_SwitchDevice.value();

	m_Parameters = m_Parameters.to(dev);
	m_pModel->to(dev);

	if (m_PerProblemInputs.has_value())
		m_PerProblemInputs = m_PerProblemInputs.value().to(dev);

	m_Data = m_Data.to(dev);


	// New parameters
	auto dops = m_Parameters.options();

	nci = nci.to(dev);
	data_slice = data_slice.to(dev);

	res = torch::empty({ numProbs, numInputs, 1 }, dops); //res.to(dev);
	res_t = torch::empty({ numProbs, numInputs, 1 }, dops);
	ep = torch::empty({ numProbs }, dops);
	g_norm = torch::empty({ numProbs }, dops);

	pD = torch::empty({ numProbs, numParams, 1 }, dops); //pD.to(dev);

	J = torch::empty({ numProbs, numInputs, numParams }, dops); //J.to(dev);
	JpD = torch::empty({ numProbs, numInputs, 1 }, dops); //JpD.to(dev);

	lambda = lambda.to(dev);
	step_mask = torch::empty({ numProbs }, dops.dtype(torch::kInt32));

	m_CurrentDevice = dev;
}

void tc::optim::SLMA::setup_solve()
{
	torch::NoGradGuard no_grad_guard;

	m_pModel->to(m_StartDevice);
	m_Data = m_Data.to(m_StartDevice);

	m_Parameters = m_pModel->getParameters();

	if (m_pModel->hasPerProblemInputs())
		m_PerProblemInputs = m_pModel->getPerProblemInputs();

	data_slice = m_Data;

	numProbs = m_pModel->getNumProblems();
	numParams = m_pModel->getNumParametersPerProblem();
	numInputs = m_Data.size(1);

	torch::TensorOptions nci_ops =
		torch::TensorOptions().dtype(c10::ScalarType::Long).device(m_StartDevice);

	nci = torch::arange(0, numProbs, nci_ops);

	// fp options
	auto fp_ops = m_Parameters.options();

	// pD needs to be created before we run dogleg
	pD = torch::empty({ numProbs, numParams, 1 }, fp_ops);

	// allocate J, res, res_t
	J = torch::empty({ numProbs, numInputs, numParams }, fp_ops);
	res = torch::empty({ numProbs, numInputs, 1 }, fp_ops);
	res_t = torch::empty({ numProbs, numInputs, 1 }, fp_ops);
	ep = torch::empty({ numProbs }, fp_ops);
	g_norm = torch::empty({ numProbs }, fp_ops);
	
	lambda = torch::ones({ numProbs }, fp_ops);
}

void tc::optim::SLMA::solve()
{
	setup_solve();

	for (tc::ui32 iter = 0; iter < m_MaxIter; ++iter) {

		step();

		if (handle_convergence())
			break;

		if (Optimizer::should_stop())
			break;
		Optimizer::set_iter_info(iter, numProbs);

		if ((numProbs < m_SwitchNumber) && !m_HasSwitched)
			switch_device();

	}

	finalize_solve();
}

void tc::optim::SLMA::finalize_solve()
{
	using namespace torch::indexing;
	// Copy the non-converging problems back to the final parameter tensor
	m_Parameters.index_put_({ nci, Slice() }, m_pModel->getParameters());

}
