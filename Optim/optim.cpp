#include "optim.hpp"

#include "../Compute/gradients.hpp"

void optim::default_optim_fetcher(std::unique_ptr<optim::Model>& pModel, torch::Tensor& data,
        OptOutRef<torch::Tensor> residuals, OptOutRef<torch::Tensor> jacobian)
{
    i32 numProb = pModel->getNumProblems();
    i32 numInpPerProb = pModel->getNumInputsPerProblem();

    if (jacobian.has_value()) {
        if (residuals.has_value()) { // if we are gonna calculate residuals later we can use it for temporary storage
            residuals.value().get() = (*pModel)();
            jacobian.value().get() = compute::jacobian(residuals.value().get(), pModel->getParameters().requires_grad_(true)).detach_();
            pModel->getParameters().requires_grad_(false);
            residuals.value().get().requires_grad_(false);

            c10::InferenceMode im_guard;
            residuals.value().get() = (residuals.value().get() - data).view({numProb, numInpPerProb, 1});
            return;
        }
        else {
            torch::Tensor y = (*pModel)();
            jacobian.value().get() = compute::jacobian(y, pModel->getParameters().requires_grad_(true)).detach_();
            pModel->getParameters().requires_grad_(false);
            return;
        }
    }

    if (residuals.has_value()) {
        residuals.value().get() = ((*pModel)() - data).view({numProb, numInpPerProb, 1});
        return;
    }

    throw std::runtime_error("Optim fetcher was called without any values to set");
}



optim::Optimizer::Optimizer(OptimizerSettings settings)
    :   m_pModel(settings.pModel), m_OptimFetcher(settings.optimFetcher), 
        m_StartTensorOptions(settings.startTensorOptions), m_StopTensorOptions(settings.stopTensorOptions),
        m_Tolerance(settings.tolerance), m_MaxIter(settings.maxIter)
{

}
