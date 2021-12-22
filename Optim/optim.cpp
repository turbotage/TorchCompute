#include "optim.hpp"
#include "optim.hpp"
#include "optim.hpp"
#include "optim.hpp"
#include "optim.hpp"
#include "../pch.hpp"

#include "optim.hpp"

#include "../Compute/gradients.hpp"


tc::optim::OptimizerSettings::OptimizerSettings()
	: startDevice(torch::Device("cpu"))
{

}

tc::optim::Optimizer::Optimizer(OptimizerSettings& settings) 
 : m_pModel(std::move(settings.pModel)), m_Data(settings.data),
	m_StartDevice(settings.startDevice),
	m_Tolerance(settings.tolerance), m_MaxIter(settings.maxIter)
{
	assert(m_Data.defined() && "Tried to create optimizer with no data");
	assert(m_Data.numel() > 0 && "Tried to create optimizer with no data");
	assert(m_pModel != nullptr && "Tried to create optimizer with pModel=nullptr");
	assert(m_pModel->getParameters().defined() && "Tried to create optimizer with no parameters");
	assert(m_pModel->getParameters().numel() > 0 && "Tried to create optimizer with no parameters");

	m_NonConvergingProblems = settings.data.size(0);
}

tc::optim::Optimizer::~Optimizer()
{
	assert(m_HasRun && "on_eval() was never run, incorrect implementation of Optimizer");
}

void tc::optim::Optimizer::abort()
{
	m_ShouldStop = true;
}

std::pair<tc::ui32, tc::ui32> tc::optim::Optimizer::getIterInfo()
{
	return std::make_pair(m_Iter.load(), m_NonConvergingProblems.load());
}

void tc::optim::Optimizer::on_abort()
{
}

void tc::optim::Optimizer::set_iter_info(tc::ui32 iter, tc::ui32 non_converging_probs)
{
	m_Iter = iter;
	m_NonConvergingProblems = non_converging_probs;
}

bool tc::optim::Optimizer::should_stop()
{
	return m_ShouldStop;
}

void tc::optim::Optimizer::on_eval() {
	assert(!m_HasRun && "Tried to evaluate optimizer twice");
	m_HasRun = true;
}