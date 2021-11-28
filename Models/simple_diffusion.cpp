#include "models.hpp"

#include "../Compute/lstq.hpp"
#include "../Optim/lm.hpp"

models::ADCModel::ADCModel() 
{
    optim::ModelFunc modfunc = adc_func;

    m_pModel = std::make_unique<optim::Model>(modfunc);
}

void models::ADCModel::setDependents(torch::Tensor bvals)
{
    m_pModel->setDependents(bvals);
}

void models::ADCModel::setData(torch::Tensor data)
{
    m_Data = data;
}


torch::Tensor models::ADCModel::solve()
{
    using namespace torch::indexing;

    torch::Tensor deps = m_pModel->getDependents();
    torch::Tensor parameterGuess = compute::lstq_qr(deps, torch::log(m_Data));

    parameterGuess.index_put_({Slice(),0}, torch::exp(parameterGuess.index({Slice(),0})));

    auto cpudev = torch::Device("cpu");

    optim::LMP lmp(*m_pModel);
    lmp.setParameterGuess(parameterGuess);
    lmp.setDependents(deps);
    lmp.setData(m_Data);
    lmp.setDefaultTensorOptions(deps.options());
    lmp.setSwitching(10000, cpudev);
    lmp.setCopyConvergingEveryN(2);

    lmp.run();

    return lmp.getParameters();
}

torch::Tensor models::ADCModel::adc_func(
    std::vector<torch::Tensor> staticvars, 
    torch::Tensor dependents, torch::Tensor parameters) 
{
    using namespace torch::indexing;

    return parameters.index({Slice(), Slice(), 0}).view({parameters.size(0), 1}) * 
        torch::exp(
            -dependents.index({Slice(), Slice(), 0}).view({dependents.size(0),dependents.size(1)}) * 
            parameters.index({Slice(), Slice(), 1}).view({parameters.size(0), 1})
        );
}