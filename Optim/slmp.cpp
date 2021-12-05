#include "slmp.hpp"


optim::SLMPSettings::SLMPSettings() 
{

}



optim::SLMP::SLMP(SLMPSettings settings)
	: m_Mu(settings.mu), m_Eta(settings.eta), Optimizer(std::move(settings)), m_CurrentDevice(settings.startDevice)
{
}

optim::OptimResult optim::SLMP::operator()()
{
	Optimizer::operator()();

	solve();

	SLMPResult res;
	res.finalParameters = m_Parameters;
	res.pFinalModel = std::move(m_pModel);
	res.nonConvergingIndices = nci;
	res.finalDeltas = delta;
	
	return res;
}


void optim::SLMP::dogleg()
{
	using namespace torch::indexing;

	c10::InferenceMode im_guard;

	torch::Tensor& pGN_Norm = pr_1;

	torch::Tensor& pCP = pr_pa_1_1;
	torch::Tensor& pCP_Norm = pr_2;
	
	torch::Tensor& chol_success = pr_0;

	// Computes and sets pD, pD_Norm and g;
	{
		// Create scaling matrix and scaled hessian
		torch::Tensor& Jn = pr_pa_1;
		Jn = torch::sqrt(torch::square(J)).sum(1);

		// Scaling matrix
		torch::Tensor& D = pr_pa_pa_1;
		D = torch::diag_embed(torch::reciprocal(Jn));
		
		// Scaled Jacobian
		torch::Tensor& Js = pr_in_pa_1;
		Js = torch::bmm(J,D);
		
		// Scaled gradient
		torch::Tensor& gs = pr_pa_1_1;
		gs = torch::bmm(Js.transpose(1,2), res);
		
		// Scaled Hessian
		// We can alias the same temp as Js since Js is no longer needded
		torch::Tensor& Hs = pr_pa_pa_2;
		Hs = torch::bmm(Js.transpose(1,2), Js);

		// Hs_chol can alias same temp as D as D won't be used again
		torch::Tensor& Hs_chol = pr_pa_pa_1; 
		std::tie(Hs_chol, step_mask) = torch::linalg_cholesky_ex(Hs);

		// Fill pD with the Gauss-Newton step
		// chol(-gs,Hs) gives the solution to the scaled Gauss-Newton subproblem)
		// We multiply with D from right to get back the unscaled solution
		chol_success = step_mask == MaskTypes::SUCCESSFUL_CHOLESKY;
		pD.index_put_({chol_success, Slice()}, 
			torch::bmm(D.index({chol_success, Slice()}) ,torch::cholesky_solve(-gs.index({chol_success, Slice()}), Hs_chol.index({chol_success, Slice()}) )));

		// Set step_mask for all unsuccessfull solves
		step_mask.index_put_({torch::logical_not(chol_success)}, MaskTypes::UNSUCCESSFUL_CHOLESKY);

		// Calculate pGN_Norm (we also calculate pGN_Norm for failed GN-subproblem problems now, unecessary?)
		pGN_Norm = torch::sqrt((pD * pD).sum(1)).view({numProbs});

		// We need the inverse of scaling matrix D to get back our gradient
		torch::Tensor& invD = pr_pa_pa_1; // Hs_chol no longer used
		invD = torch::diag_embed(Jn);

		// Get back unscaled gradient
		// We can use same temp for pCP as g since we will compute pCP from g and then won't use g again
		torch::Tensor& g = pr_pa_1_1;
		g = torch::bmm(invD, gs);
		
		// Calculate Cauchy-Point
		pCP = -g * (torch::bmm(g.transpose(1,2), g) / torch::bmm(
			torch::bmm(g.transpose(1,2), invD), torch::bmm(Hs, g)));

		pCP_Norm = torch::sqrt((pCP * pCP).sum(1));
		
	}

	// Indicate that all problems with successful cholesky decomp and GN-step less than trust region radius are taking a full GN-step
	torch::Tensor& full_gn = pr_0; // this takes over chol_success
	full_gn = torch::logical_and(chol_success, pGN_Norm < delta);
	step_mask.index_put_({full_gn}, torch::bitwise_or(step_mask.index({full_gn}), MaskTypes::FULL_GAUSS_NEWTON));

	// Those who couldn't take a full GN-step and have CP-point outside trust region, scale the negative gradient as step
	torch::Tensor& cp_step = pr_0;
	cp_step = torch::logical_and(pCP_Norm > delta, torch::logical_not(full_gn)); // Those who didn't take a full step and has CP outside trustreg
	cp_step = torch::logical_or(cp_step, step_mask == MaskTypes::UNSUCCESSFUL_CHOLESKY); // We wan't unsuccessfull choleskys to also move in the gradient
	pD.index_put_({cp_step, Slice()},  -pCP.index({cp_step, Slice()}) * delta.index({cp_step}) / pCP_Norm.index({cp_step}));
	step_mask.index_put_({cp_step}, torch::bitwise_or(step_mask.index({cp_step}), MaskTypes::SCALED_GRADIENT));

	// If neither Gauss-Newton nor Cauchy step is accepted, find the intersection of line CP-pGN and circle with radius delta
	torch::Tensor& interpol_step = pr_0;
	// Those who had successfull chol and has no bit set in full_gn or cp_step will be those who should be interpolated
	interpol_step = step_mask == MaskTypes::SUCCESSFUL_CHOLESKY;

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
	torch::InferenceMode im_guard;

	// Compute J
	m_JacobianSetter(m_pModel, J);
	// Compute res
	res = ((*m_pModel)() - data_slice).view({numProbs, numInputs, 1});

	// Perform the dogleg calculations
	dogleg();


	torch::Tensor& JpD = pr_in_1_1;
	JpD = torch::bmm(J,pD);
	
	// Objective function value at current point
	torch::Tensor& ep = pr_1;
	ep = 0.5 * torch::square(res).sum(1).view({numProbs});

	// Trailing point
	torch::Tensor& p = pr_pa_1;
	p = m_pModel->getParameters();
	m_pModel->setParameters(p + pD.view({numProbs, numParams}));

	// Objective function value at proposed new poin
	torch::Tensor& et = pr_2;
	et = 0.5 * torch::square(((*m_pModel)() - data_slice).view({numProbs, numInputs, 1})).sum(1).view({numProbs}); 
	
	// actual decrease
	torch::Tensor& actual = pr_1;
	actual = ep - et;

	// predicted decrease
	torch::Tensor& predicted = pr_2;
	predicted = -torch::bmm(res.transpose(1,2), JpD).view({numProbs});

	// gain rato
	torch::Tensor& rho = pr_1;
	rho = actual / predicted;

	torch::Tensor& gain_mask = pr_0;
	gain_mask = rho <= m_Mu;

	// For poor gain ratios, decrease the trust region
	delta.index_put_({gain_mask}, 0.5 * delta.index({gain_mask}));

	// This mask gives all problems which had poor gain ratio and also didn't have structural problems (cholesky didn't fail)
	// these problems should not take a step
	gain_mask = torch::logical_and(
						torch::logical_not(
							torch::bitwise_and(step_mask, MaskTypes::UNSUCCESSFUL_CHOLESKY) == MaskTypes::UNSUCCESSFUL_CHOLESKY), 
							gain_mask);

	m_pModel->getParameters().index_put_({gain_mask}, p.index({gain_mask})); // Now all steps are set

	// We must set JpD back for these problems (failing problems)
	JpD.index_put_({gain_mask}, torch::bmm(J.index({gain_mask}), p.index({gain_mask})));

	// Mask for good gain ratio
	gain_mask = rho >= m_Eta;

	// For a good gain ratio we increase the trust region
	delta.index_put_({gain_mask}, 2.0*delta.index({gain_mask}));

}

bool optim::SLMP::handle_convergence()
{
	using namespace torch::indexing;

	c10::InferenceMode im_guard;

	// Catch Jp set in pr_in_1_1 by step
	torch::Tensor& Jp = pr_in_1_1;

	torch::Tensor& converges = pr_0; // plane convergence
	converges = torch::sqrt(torch::square(Jp * Jp).sum(1)) <=
		m_Tolerance * (1 + torch::sqrt(torch::square(res).sum(1).view({numProbs})));

	// Copy the converging problems back to the final parameter tensor
	m_Parameters.index_put_({ nci.index({converges}), Slice() },
		m_pModel->getParameters().index({ converges, Slice() }));

	// Recreate the index list for the pixels that don't converge
	nci = nci.index({ converges });
	numProbs = nci.size(0); // The new number of problems

	// Extract parameters and inputs to be those problems which havn't converged
	{
		torch::Tensor& params_slice = m_pModel->getPerProblemInputs();
		m_pModel->setPerProblemInputs(params_slice.index({ nci, Slice() }));
		
		torch::Tensor& deps_slice = m_pModel->getParameters();
		if (deps_slice.defined()) {
			if (deps_slice.numel() != 0)
				m_pModel->setParameters(deps_slice.index({ nci, Slice() }));
		}
	}


	// Extract data which corresponds to non-converging problems
	data_slice = data_slice.index({ nci, Slice() });

	// Extract deltas corresponding to non-converging problems
	delta = delta.index({ nci });

	if (numProbs == 0) // if no non-converging pixels are left we can return
		return true;
	return false;
}

void optim::SLMP::switch_device() {
	if (!m_SwitchDevice.has_value())
		return;

	torch::Device& dev = m_SwitchDevice.value();

	m_Parameters =			m_Parameters.to(dev);
	m_PerProblemInputs =	m_PerProblemInputs.to(dev);

	data_slice =			data_slice.to(dev);

	nci =					nci.to(dev);
	
	res =					res.to(dev);
	pr_in_1_1 =				pr_in_1_1.to(dev);

	pD =					pD.to(dev);
	pr_pa_1_1 =				pr_pa_1_1.to(dev);

	pr_pa_1 =				pr_pa_1.to(dev);

	J =						J.to(dev);
	pr_in_pa_1 =			pr_in_pa_1.to(dev);

	pr_pa_pa_1 =			pr_pa_pa_1.to(dev);
	pr_pa_pa_2 =			pr_pa_pa_2.to(dev);

	delta =					delta.to(dev);
	step_mask =				step_mask.to(dev);

	pr_0 =					pr_0.to(dev);

	pr_1 =					pr_1.to(dev);
	pr_2 =					pr_2.to(dev);

	m_CurrentDevice = dev;
}

void optim::SLMP::setup_solve() {
	
	m_pModel->to(m_StartDevice);
	m_Data.to(m_StartDevice);

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
}

void optim::SLMP::solve()
{
	setup_solve();

	for (ui32 iter = 0; iter < m_MaxIter; ++iter) {

		step();

		if (handle_convergence())
			break;

		if (numProbs < m_SwitchNumber && !m_HasSwitched)
			switch_device();

	}

	finalize_solve();
}

void optim::SLMP::finalize_solve() 
{
	using namespace torch::indexing;
	// Copy the non-converging problems back to the final parameter tensor
	m_Parameters.index_put_({ nci, Slice() }, m_pModel->getParameters());

	// Move OptimResult outputs to the specified final device
	m_Parameters.to(m_StopDevice);
	m_pModel->to(m_StopDevice);
	delta = delta.to(m_StopDevice);

}