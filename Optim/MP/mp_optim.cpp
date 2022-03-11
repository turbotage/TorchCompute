#include "../../pch.hpp"

#include "mp_optim.hpp"

#include "../../Compute/gradients.hpp"




tc::optim::MP_OptimizerSettings::MP_OptimizerSettings(std::unique_ptr<optim::MP_Model> pModel, 
	const torch::Tensor& data, tc::ui32 maxiter)
{
	if (pModel->parameters().size(0) != data.size(0))
		throw std::runtime_error("Number sizes in parameter and data did not match");

	if (!pModel->parameters().is_contiguous())
		throw std::runtime_error("Optimizer requires contigous parameters");

	if (!data.is_contiguous())
		throw std::runtime_error("Optimizer requires contigous data");

	this->pModel = std::move(pModel);
	this->data = data;
	this->maxiter = maxiter;
}

/*
tc::optim::MPOptimizerSettings::MPOptimizerSettings(MPOptimizerSettings&& other) noexcept
	: pModel(std::move(other.pModel)),
	data(std::move(other.data)),
	maxiter(other.maxiter)
{
}
*/

tc::optim::MP_OptimizerSettings::~MP_OptimizerSettings()
{
}



tc::optim::MP_Optimizer::MP_Optimizer(MP_OptimizerSettings&& settings) 
 : pModel(std::move(settings.pModel)), data(settings.data),
	maxiter(settings.maxiter)
{

}

void tc::optim::MP_Optimizer::run()
{
	on_run();
}

std::unique_ptr<tc::optim::MP_Model> tc::optim::MP_Optimizer::acquire_model()
{
	if (m_HasAcquiredModel)
		throw std::runtime_error("Tried to acquire results from optimizer twice");

	m_HasAcquiredModel = true;
	on_acquire_model();
	return std::move(pModel);
}

void tc::optim::MP_Optimizer::abort()
{
	m_ShouldStop = true;
	on_abort();
}

tc::ui32 tc::optim::MP_Optimizer::get_n_iter() const
{
	return m_Iter;
}

void tc::optim::MP_Optimizer::set_n_iter(tc::ui32 iter)
{
	m_Iter = iter;
}

bool tc::optim::MP_Optimizer::should_stop() const
{
	return m_ShouldStop;
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
