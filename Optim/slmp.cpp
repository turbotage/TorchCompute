#include "slmp.hpp"


optim::SLMPSettings::SLMPSettings() 
{

}



optim::SLMP::SLMP(SLMPSettings& settings)
	: m_Mu(settings.mu), m_Eta(settings.eta), m_CurrentDevice(settings.startDevice), 
	m_SwitchDevice(settings.switchDevice), m_SwitchNumber(settings.switchAtN),
	Optimizer(settings)
{
}

optim::SLMPResult optim::SLMP::eval()
{
	Optimizer::on_eval();

	solve();

	SLMPResult res;
	res.finalParameters = m_Parameters;
	res.pFinalModel = std::move(m_pModel);
	res.nonConvergingIndices = nci;
	res.finalDeltas = delta;
	
	return std::move(res);
}


void optim::SLMP::dogleg()
{
	using namespace torch::indexing;

	//std::cout << "params:\n" << m_pModel->getParameters() << std::endl;

	c10::InferenceMode im_guard;

	torch::Tensor pGN_Norm;
	torch::Tensor chol_success;

	torch::Tensor pCP;
	torch::Tensor pCP_Norm;

	// Computes and sets pD, pD_Norm and g;
	{
		// Create scaling matrix and scaled hessian
		torch::Tensor Jn = torch::sqrt(torch::square(J).sum(1));

		torch::Tensor Hs;
		torch::Tensor gs;
		
		{
			// Scaling matrix
			torch::Tensor D = torch::diag_embed(torch::reciprocal(Jn));

			// Scaled Jacobian
			torch::Tensor Js = torch::bmm(J, D);

			// Scaled gradient
			gs = torch::bmm(Js.transpose(1, 2), res);

			// Scaled Hessian
			Hs = torch::bmm(Js.transpose(1, 2), Js);

			torch::Tensor Hs_chol;
			std::tie(Hs_chol, step_mask) = torch::linalg_cholesky_ex(Hs);

			// Fill pD with the Gauss-Newton step
			// chol(-gs,Hs) gives the solution to the scaled Gauss-Newton subproblem)
			// We multiply with D from right to get back the unscaled solution
			chol_success = step_mask == MaskTypes::SUCCESSFUL_CHOLESKY;
			pD.index_put_({ chol_success, Slice() },
				torch::bmm(D.index({ chol_success, Slice() }), torch::cholesky_solve(-gs.index({ chol_success, Slice() }), Hs_chol.index({ chol_success, Slice() }))));
			// Set step_mask for all unsuccessfull solves
			step_mask.index_put_({ torch::logical_not(chol_success) }, MaskTypes::UNSUCCESSFUL_CHOLESKY);
		}

		// Calculate pGN_Norm
		pGN_Norm = torch::sqrt(torch::square(pD).sum(1)).view({numProbs});

		// We need the inverse of scaling matrix D to get back our gradient
		torch::Tensor invD = torch::diag_embed(Jn);

		// Get back unscaled gradient
		torch::Tensor g = torch::bmm(invD, gs);
		torch::Tensor invDg = torch::bmm(invD, g);

		// Calculate Cauchy-Point
		pCP = -g * (torch::bmm(g.transpose(1, 2), g) / torch::bmm(invDg.transpose(1, 2), torch::bmm(Hs, invDg)));

		pCP_Norm = torch::sqrt(torch::square(pCP).sum(1)).view({numProbs});
		
	}

	// Indicate that all problems with successful cholesky decomp and GN-step less than trust region radius are taking a full GN-step
	torch::Tensor full_gn = torch::logical_and(chol_success, pGN_Norm < delta);
	step_mask.index_put_({full_gn}, torch::bitwise_or(step_mask.index({full_gn}), MaskTypes::FULL_GAUSS_NEWTON));


	// Those who couldn't take a full GN-step and have CP-point outside trust region, scale the negative gradient as step
	torch::Tensor cp_step = torch::logical_and(pCP_Norm > delta, torch::logical_not(full_gn)); // Those who didn't take a full step and has CP outside trustreg
	cp_step = torch::logical_or(cp_step, step_mask == MaskTypes::UNSUCCESSFUL_CHOLESKY); // We wan't unsuccessfull choleskys to also move in the gradient

	pD.index_put_({cp_step, Slice(), Slice()}, -pCP.index({cp_step, Slice(), Slice()}) * 
		(delta.index({cp_step}) / pCP_Norm.index({cp_step})).unsqueeze(1).unsqueeze(2));
	step_mask.index_put_({cp_step}, torch::bitwise_or(step_mask.index({cp_step}), MaskTypes::SCALED_GRADIENT));

	// If neither Gauss-Newton nor Cauchy step is accepted, find the intersection of line CP-pGN and circle with radius delta
	// Those who had successfull chol and has no bit set in full_gn or cp_step will be those who should be interpolated
	torch::Tensor interpol_step = step_mask == MaskTypes::SUCCESSFUL_CHOLESKY;
	{
		i32 masksum = interpol_step.sum().item<int64_t>();

		torch::Tensor CP = pCP.index({interpol_step, Slice()});
		torch::Tensor GN = pD.index({interpol_step, Slice()});

		torch::Tensor GN_CP = GN - CP;
		torch::Tensor A = torch::square(GN_CP).sum(1);
		torch::Tensor B = 2.0 * (CP * GN_CP).sum(1);
		torch::Tensor C = torch::square(CP).sum(1) - torch::square(delta.index({interpol_step}).view({masksum, 1}));

		torch::Tensor k = 0.5 * (-B + torch::sqrt(torch::square(B) - 4 * A * C)) / A;

		pD.index_put_({interpol_step, Slice()}, CP + k.view({masksum,1,1})*(GN - CP));
		step_mask.index_put_({interpol_step}, MaskTypes::INTERPOLATED);

	}
}

void optim::SLMP::step()
{
	using namespace torch::indexing;

	torch::InferenceMode im_guard;

	// Compute J,res
	res = res.view({ numProbs, numInputs });
	m_pModel->res_diff(res, J, data_slice);
	res = res.view({numProbs, numInputs, 1});

	// Perform the dogleg calculations
	dogleg();

	// Parameters before we have accepted any new step
	torch::Tensor currentParams = m_pModel->getParameters();

	torch::Tensor rho;
	// Calculate rho (gain ratio), JpD
	{

		// Actual reduction
		torch::Tensor actual;
		// Calculate actual reduction
		{
			// Objective function value at current point
			torch::Tensor ep = 0.5 * torch::square(res).sum(1).view({ numProbs });

			// Trailing point
			m_pModel->setParameters(currentParams + pD.view({ numProbs, numParams }));

			// Objective function value at proposed new poin
			res_t = res_t.view({ numProbs, numInputs });
			m_pModel->res(res_t, data_slice);
			res_t = res_t.view({ numProbs, numInputs, 1 });

			torch::Tensor et = 0.5 * torch::square(res_t).sum(1).view({ numProbs });

			// actual decrease
			actual = ep - et;
		}

		JpD = torch::bmm(J, pD);

		// predicted decrease
		torch::Tensor predicted = -torch::bmm(res.transpose(1, 2), JpD).view({ numProbs }) -
			0.5 * torch::square(JpD).sum(1).view({ numProbs });

		// gain rato
		rho = actual / predicted;
	}

	torch::Tensor gain_mask;
	gain_mask = rho <= m_Mu;

	// For poor gain ratios, decrease the trust region
	delta.index_put_({gain_mask}, 0.5 * delta.index({gain_mask}));

	// This mask gives all problems which had poor gain ratio and also didn't have structural problems (cholesky didn't fail)
	// these problems should not take a step
	gain_mask = torch::logical_and(
						torch::logical_not(
							torch::bitwise_and(step_mask, MaskTypes::UNSUCCESSFUL_CHOLESKY) == MaskTypes::UNSUCCESSFUL_CHOLESKY), 
							gain_mask);

	m_pModel->getParameters().index_put_({gain_mask, Slice()}, currentParams.index({gain_mask, Slice()})); // Now all steps are set

	// We must set JpD back for these problems (failing problems)
	JpD.index_put_({gain_mask, Slice(), Slice()}, 
		torch::bmm(J.index({gain_mask, Slice(), Slice()}), currentParams.index({gain_mask, Slice()}).unsqueeze(2)));

	// Mask for good gain ratio
	gain_mask = rho >= m_Eta;

	// For a good gain ratio we increase the trust region
	delta.index_put_({gain_mask}, 2.0*delta.index({gain_mask}));

}

bool optim::SLMP::handle_convergence()
{
	using namespace torch::indexing;

	c10::InferenceMode im_guard;

	std::cout << "paramshape:\n" << m_pModel->getParameters().sizes() << std::endl;

	// plane convergence
	torch::Tensor converges = torch::sqrt(torch::square(JpD).sum(1).view({numProbs})) <=
		m_Tolerance * (1 + torch::sqrt(torch::square(res).sum(1).view({numProbs})));

	// Copy the converging problems back to the final parameter tensor
	m_Parameters.index_put_({ nci.index({converges}), Slice() },
		m_pModel->getParameters().index({converges, Slice()}));

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
	res_t = res_t.index({ not_converges, Slice(), Slice() });

	pD = pD.index({ not_converges, Slice(), Slice() });

	J = J.index({ not_converges, Slice(), Slice() });
	delta = delta.index({ not_converges });

	step_mask = step_mask.index({ not_converges });

	if (numProbs == 0) // if no non-converging pixels are left we can return
		return true;
	return false;
}

void optim::SLMP::switch_device() {
	if (!m_SwitchDevice.has_value())
		return;
	m_HasSwitched = true;

	torch::Device& dev = m_SwitchDevice.value();

	m_Parameters =			m_Parameters.to(dev);
	m_PerProblemInputs =	m_PerProblemInputs.to(dev);
	m_Data =				m_Data.to(dev);

	m_pModel->to(dev);

	// New parameters
	auto dops = m_Parameters.options();

	nci =					nci.to(dev);
	data_slice =			data_slice.to(dev);

	res =					torch::empty({ numProbs, numInputs, 1 }, dops); //res.to(dev);
	res_t =					torch::empty({ numProbs, numInputs, 1 }, dops); //res_t.to(dev);

	pD =					torch::empty({ numProbs, numParams, 1 }, dops); //pD.to(dev);

	J =						torch::empty({ numProbs, numInputs, numParams }, dops); //J.to(dev);
	JpD =					torch::empty({ numProbs, numInputs, 1 }, dops); //JpD.to(dev);

	delta =					delta.to(dev);
	step_mask =				torch::empty({numProbs}, dops.dtype(torch::kInt32));

	m_CurrentDevice = dev;
}

void optim::SLMP::setup_solve() {
	
	m_pModel->to(m_StartDevice);
	m_Data = m_Data.to(m_StartDevice);

	m_Parameters = m_pModel->getParameters();
	m_PerProblemInputs = m_pModel->getPerProblemInputs();

	data_slice = m_Data;

	numProbs = m_pModel->getNumProblems();
	numParams = m_pModel->getNumParametersPerProblem();
	numInputs = m_pModel->getNumInputsPerProblem();

	torch::TensorOptions nci_ops =
		torch::TensorOptions().dtype(c10::ScalarType::Long).device(m_StartDevice);

	nci = torch::arange(0, numProbs, nci_ops);

	delta = torch::sqrt(torch::square(m_Parameters).sum(1)).view({ numProbs });
	
	// fp options
	auto fp_ops = m_Parameters.options();

	// pD needs to be created before we run dogleg
	pD = torch::empty({ numProbs, numParams, 1 }, fp_ops);

	// allocate J, res, res_t
	J = torch::empty({ numProbs, numInputs, numParams }, fp_ops);
	res = torch::empty({ numProbs, numInputs, 1 }, fp_ops);
	res_t = torch::empty({ numProbs, numInputs, 1 }, fp_ops);

}

void optim::SLMP::solve()
{
	setup_solve();

	for (ui32 iter = 0; iter < m_MaxIter; ++iter) {

		std::cout << "iter: " << iter << "numProbs: " << numProbs << std::endl;

		step();

		if (handle_convergence())
			break;

		if ((numProbs < m_SwitchNumber) && !m_HasSwitched)
			switch_device();

	}

	finalize_solve();
}

void optim::SLMP::finalize_solve() 
{
	using namespace torch::indexing;
	// Copy the non-converging problems back to the final parameter tensor
	m_Parameters.index_put_({ nci, Slice() }, m_pModel->getParameters());

}