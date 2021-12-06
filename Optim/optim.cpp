#include "optim.hpp"

#include "../Compute/gradients.hpp"

optim::OptimizerSettings::OptimizerSettings()
	: startDevice(torch::Device("cpu")), stopDevice(torch::Device("cpu"))
{

}


optim::Optimizer::Optimizer(OptimizerSettings settings)
	:   m_pModel(std::move(settings.pModel)), m_Data(settings.data),
		m_StartDevice(settings.startDevice), m_StopDevice(settings.stopDevice),
		m_Tolerance(settings.tolerance), m_MaxIter(settings.maxIter)
{
	
	if constexpr (BUILD_MODE == eBuildMode::Debug) {
		if (!m_Data.defined()) {
			throw std::runtime_error("Tried to create optimizer with no data");
		}
		if (m_Data.numel() == 0) {
			throw std::runtime_error("Tried to create optimizer with no data");
		}
		if (m_pModel == nullptr) {
			throw std::runtime_error("Tried to create optimizer with pModel=nullptr");
		}
		if (!m_pModel->getParameters().defined()) {
			throw std::runtime_error("Tried to create optimizer with no parameters");
		}
		if (!m_pModel->getParameters().numel() == 0) {
			throw std::runtime_error("Tried to create optimizer with no parameters");
		}
	}

}

optim::OptimResult optim::Optimizer::operator()() {
	if (m_HasRun)
		throw std::runtime_error("Tried to run optimizer twice!");
	m_HasRun = true;
}