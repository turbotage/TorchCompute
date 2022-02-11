#include "../pch.hpp"

#include "str.hpp"

tc::optim::STRPSettings::STRPSettings()
{
}

tc::optim::STRP::STRP(STRPSettings& settings)
	: m_Mu(settings.mu), m_Eta(settings.eta), Optimizer(settings)
{
}

tc::optim::STRPResult tc::optim::STRP::eval()
{
	Optimizer::on_eval();

	solve();

	STRPResult res;
	res.finalParameters = m_pModel->getParameters();
	res.finalDeltas = delta;
	res.pFinalModel = std::move(m_pModel);

	return res;
}

std::unique_ptr<tc::optim::OptimResult> tc::optim::STRP::base_eval()
{
	Optimizer::on_eval();

	solve();

	std::unique_ptr<STRPResult> ret = std::make_unique<STRPResult>();
	ret->finalParameters = m_pModel->getParameters();
	ret->finalDeltas = delta;
	ret->pFinalModel = std::move(m_pModel);

	return ret;
}

void tc::optim::STRP::dogleg()
{
	torch::InferenceMode im_guard;

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

			

		}

	}

}

void tc::optim::STRP::step()
{
}

void tc::optim::STRP::solve()
{
}

