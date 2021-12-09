#include "gn.hpp"


optim::GNSettings::GNSettings()
{

}

optim::GN::GN(GNSettings& settings) 
	: m_CurrentDevice(settings.startDevice), m_SwitchDevice(settings.switchDevice), 
	m_SwitchNumber(settings.switchAtN), Optimizer(settings)
{

}

optim::GNResult optim::GN::eval()
{
	Optimizer::on_eval();

	solve();

	GNResult res;
	res.finalParameters = m_Parameters;
	res.pFinalModel = std::move(m_pModel);
	res.nonConvergingIndices = nci;

	return std::move(res);
}

void optim::GN::step()
{
	using namespace torch::indexing;

	torch::InferenceMode im_guard;

	res = res.view({ numProbs, numInputs });
	m_pModel->res_diff(res, J, data_slice);
	res = res.view({ numProbs, numInputs, 1 });

	// Gradient
	torch::Tensor g = torch::bmm(J.transpose(1, 2), res);

	// Hessian
	torch::Tensor H = torch::bmm(J.transpose(1, 2), J);

	torch::Tensor H_chol;
	std::tie(H_chol, step_mask) = torch::linalg_cholesky_ex(H);

	torch::Tensor chol_success = step_mask == MaskTypes::SUCCESSFUL_CHOLESKY_GN;
	pD.index_put_({ chol_success, Slice() },
		torch::cholesky_solve(-g.index({ chol_success, Slice() }), H_chol.index({ chol_success, Slice() })));

	// For problems where cholesky failed we move in the steepest descent direction hoping we get back to a point where
	// The hessian will be positive definite, we use a quite small damping parameter so we don't diverge as we move towards
	// safe full Gauss-Newton regions
	pD.index_put_({ torch::logical_not(chol_success), Slice() }, -gd_damp*g.index({ torch::logical_not(chol_success), Slice() }));

	// This is used to determine convergence
	JpD = torch::bmm(J, pD);

	// Update the parameters
	m_pModel->setParameters(m_pModel->getParameters() + pD.view({numProbs, numParams}));
}

bool optim::GN::handle_convergence()
{
	using namespace torch::indexing;

	c10::InferenceMode im_guard;

	torch::Tensor converges = torch::sqrt(torch::square(JpD).sum(1).view({ numProbs })) <=
		m_Tolerance * (1 + torch::sqrt(torch::square(res).sum(1).view({ numProbs })));

	// Copy the converging problems back to the final parameter tensor
	m_Parameters.index_put_({ nci.index({converges}), Slice() },
		m_pModel->getParameters().index({ converges, Slice() }));

	// Recreate the index list for the pixels that don't converge
	torch::Tensor not_converges = torch::logical_not(converges);
	nci = nci.index({ not_converges });
	numProbs = nci.size(0); // The new number of problems

	// Extract and set non-converging problems

	{
		m_pModel->setParameters(m_pModel->getParameters().index({ not_converges, Slice() }));

		if (m_pModel->getPerProblemInputs().defined()) {
			if (m_pModel->getPerProblemInputs().numel() != 0) {
				m_pModel->setPerProblemInputs(m_pModel->getPerProblemInputs().index({ not_converges, Slice(), Slice() }));
			}
		}
	}

	data_slice = data_slice.index({ not_converges, Slice() });

	res = res.index({ not_converges, Slice(), Slice() });

	pD = pD.index({ not_converges, Slice(), Slice() });

	J = J.index({ not_converges, Slice(), Slice() });

	step_mask = step_mask.index({ not_converges });

	if (numProbs == 0) // if no non-converging pixels are left we can return
		return true;
	return false;

}

void optim::GN::switch_device()
{
	if (!m_SwitchDevice.has_value())
		return;
	m_HasSwitched = true;

	torch::Device& dev = m_SwitchDevice.value();

	m_Parameters = m_Parameters.to(dev);
	m_PerProblemInputs = m_PerProblemInputs.to(dev);
	m_Data = m_Data.to(dev);

	m_pModel->to(dev);

	// New parameters
	auto dops = m_Parameters.options();

	nci = nci.to(dev);
	data_slice = data_slice.to(dev);

	res = torch::empty({ numProbs, numInputs, 1 }, dops); //res.to(dev);
	pD = torch::empty({ numProbs, numParams, 1 }, dops); //pD.to(dev);

	J = torch::empty({ numProbs, numInputs, numParams }, dops); //J.to(dev);
	JpD = torch::empty({ numProbs, numInputs, 1 }, dops); //JpD.to(dev);

	step_mask = torch::empty({ numProbs }, dops.dtype(torch::kInt32));

	m_CurrentDevice = dev;
}

void optim::GN::setup_solve() {
	m_pModel->to(m_StartDevice);
	m_Data = m_Data.to(m_StartDevice);

	m_Parameters = m_pModel->getParameters();
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
}

void optim::GN::solve()
{
	setup_solve();

	for (ui32 iter = 0; iter < m_MaxIter; ++iter) {
		step();

		if (handle_convergence())
			break;

		if ((numProbs < m_SwitchNumber) && !m_HasSwitched)
			switch_device();
	}

	finalize_solve();
}

void optim::GN::finalize_solve()
{
	using namespace torch::indexing;
	// Copy the non-converging problems back to the final parameter tensor
	m_Parameters.index_put_({ nci, Slice() }, m_pModel->getParameters());
}