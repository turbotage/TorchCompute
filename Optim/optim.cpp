#include "optim.hpp"

#include "../Compute/gradients.hpp"

void optim::default_jacobian_setter(std::unique_ptr<optim::Model>& pModel, torch::Tensor& jacobian)
{
	i32 numProb = pModel->getNumProblems();
	i32 numInpPerProb = pModel->getNumInputsPerProblem();

	torch::Tensor y = (*pModel)();
	jacobian = compute::jacobian(y, pModel->getParameters().requires_grad_(true)).requires_grad_(false);
	pModel->getParameters().requires_grad_(false);
}



optim::Optimizer::Optimizer(OptimizerSettings settings)
	:   m_pModel(std::move(settings.pModel)), m_JacobianSetter(settings.jacobianSetter), 
		m_StartTensorOptions(settings.startTensorOptions), m_StopTensorOptions(settings.stopTensorOptions),
		m_Tolerance(settings.tolerance), m_MaxIter(settings.maxIter)
{

}
