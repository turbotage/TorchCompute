#include "lmp.hpp"


optim::LMP::LMP(LMPSettings settings)
	: m_Mu(settings.mu), m_Eta(settings.eta), Optimizer(std::move(settings))
{

}

void optim::LMP::dogleg()
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

void optim::LMP::step()
{
	torch::InferenceMode im_guard;

	// Compute J
	m_JacobianSetter(m_pModel, J);
	// Compute res
	res = ((*m_pModel)() - m_DataSlice).view({numProbs, numInputs, 1});

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
	et = 0.5 * torch::square(((*m_pModel)() - m_DataSlice).view({numProbs, numInputs, 1})).sum(1).view({numProbs}); 
	
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

void optim::LMP::handle_convergence()
{
	c10::InferenceMode im_guard;

	

}


void optim::LMP::solve()
{



	for (ui32 iter = 0; iter < m_MaxIter; ++iter) {

		step();

		handle_convergence();

	}

}