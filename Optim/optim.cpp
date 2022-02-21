#include "optim.hpp"
#include "optim.hpp"
#include "optim.hpp"
#include "optim.hpp"
#include "optim.hpp"
#include "optim.hpp"
#include "optim.hpp"
#include "optim.hpp"
#include "optim.hpp"
#include "optim.hpp"
#include "../pch.hpp"

#include "optim.hpp"

#include "../Compute/gradients.hpp"


tc::optim::OptimizerSettings::OptimizerSettings()
{

}

tc::optim::Optimizer::Optimizer(OptimizerSettings& settings) 
 : m_pModel(std::move(settings.pModel)), m_Data(settings.data),
	m_Tolerance(settings.tolerance), m_MaxIter(settings.maxIter)
{
	assert(m_Data.defined() && "Tried to create optimizer with no data");
	assert(m_Data.numel() > 0 && "Tried to create optimizer with no data");
	assert(m_pModel != nullptr && "Tried to create optimizer with pModel=nullptr");
	assert(m_pModel->getParameters().defined() && "Tried to create optimizer with no parameters");
	assert(m_pModel->getParameters().numel() > 0 && "Tried to create optimizer with no parameters");

}

tc::optim::Optimizer::~Optimizer()
{
	assert(m_HasRun && "on_eval() was never run, incorrect implementation of Optimizer");
}

void tc::optim::Optimizer::abort()
{
	m_ShouldStop = true;
	on_abort();
}

tc::ui32 tc::optim::Optimizer::get_n_iter()
{
	return m_Iter;
}

void tc::optim::Optimizer::on_abort()
{
}

void tc::optim::Optimizer::set_n_iter(tc::ui32 iter)
{

}



bool tc::optim::Optimizer::should_stop()
{
	return m_ShouldStop;
}

void tc::optim::Optimizer::on_eval() {
	assert(!m_HasRun && "Tried to evaluate optimizer twice");
	m_HasRun = true;
}

torch::Tensor tc::optim::get_plane_converging_problems_combined(
	torch::Tensor& lastJ, torch::Tensor& lastP, torch::Tensor& lastR, float tolerance)
{
	torch::InferenceMode im_guard;

	return torch::sqrt(torch::square(torch::bmm(lastJ, lastP)).sum(1)).squeeze() <
		tolerance * (1 + torch::sqrt(torch::square(lastR).sum(1)).squeeze());
}

torch::Tensor tc::optim::get_plane_converging_problems( 
	torch::Tensor& lastJ, torch::Tensor& lastP, torch::Tensor& lastR, float tolerance)
{
	torch::InferenceMode im_guard;

	return torch::sqrt(torch::square(torch::bmm(lastJ, lastP)).sum(1)).squeeze() <
		tolerance * torch::sqrt(torch::square(lastR).sum(1)).squeeze();
}

torch::Tensor tc::optim::get_gradient_converging_problems_absolute(torch::Tensor& J, torch::Tensor& R, float tolerance)
{
	torch::InferenceMode im_guard;

	return torch::sqrt(torch::square(torch::bmm(J, R)).sum(1)).squeeze() < tolerance;
}

torch::Tensor tc::optim::get_gradient_converging_problems_relative(torch::Tensor& J, torch::Tensor& R, float tolerance)
{
	torch::InferenceMode im_guard;

	return torch::sqrt(torch::square(torch::bmm(J, R)).sum(1)).squeeze() <
		tolerance * torch::sqrt(torch::square(R).sum(1)).squeeze();
}

torch::Tensor tc::optim::get_gradient_converging_problems_combined(torch::Tensor& J, torch::Tensor& R, float tolerance)
{
	torch::InferenceMode im_guard;

	return torch::sqrt(torch::square(torch::bmm(J, R)).sum(1)).squeeze() <
		tolerance * (1 + torch::sqrt(torch::square(R).sum(1)).squeeze());
}
