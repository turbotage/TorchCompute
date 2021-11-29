
#include "models.hpp"

#include "../Compute/lstq.hpp"
#include "../Optim/lm.hpp"


torch::Tensor models::vfa_func(
    std::vector<torch::Tensor> staticvars, 
    torch::Tensor dependents, torch::Tensor parameters)
{
    using namespace torch::indexing;

    torch::Tensor temp = parameters.index({Slice(), 0}).view({parameters.size(0), 1}); // s0
    temp *= torch::sin(dependents.index({Slice(), Slice(), 0}).view({dependents.size(0), dependents.size(1)})); // s0 * sin(FA)
    torch::Tensor expterm = torch::exp(-staticvars[0]/parameters.index({Slice(), 1}).view({parameters.size(0), 1})); // exp(-TR/T1)
    temp *= (1 - expterm); // s0 * sin(FA) * (1-expterm)
    temp /= (1- expterm*torch::cos(dependents.index({Slice(), Slice(), 0}).view({dependents.size(0), dependents.size(1)}))); // Full expression

    return temp;
}

torch::Tensor models::simple_vfa_model_linear(torch::Tensor flip_angles, torch::Tensor data, torch::Tensor TR)
{
    using namespace torch::indexing;

    torch::Tensor temp = data.view({data.size(0), data.size(1), 1}) / torch::sin(flip_angles); // s_n/sin(FA)
    {
        torch::Tensor rdatatan = data.view({data.size(0), data.size(1), 1}) / torch::tan(flip_angles); // s_n/tan(FA)
        temp = compute::lstq_qr(rdatatan, temp);
    }
    
    // Set S0
    temp.index_put_({Slice(), Slice(), 0}, 
        temp.index({Slice(), Slice(), 0}) / (1 - temp.index({Slice(), Slice(), 1})) );
    // Set T1
    temp.index_put_({Slice(), Slice(), 0}, -TR / torch::log(temp.index({Slice(), Slice(), 1})));

    return temp;
}

torch::Tensor models::simple_vfa_model_nonlinear(torch::Tensor flip_angles, torch::Tensor data, torch::Tensor TR, torch::Tensor parameter_guess)
{
    auto cpudev = torch::Device("cpu");

    optim::ModelFunc modfunc = vfa_func;
    optim::Model model(modfunc);

    optim::LMP lmp(model);
    lmp.setParameterGuess(parameter_guess);
    lmp.setDependents(flip_angles);
    lmp.setData(data);
    lmp.setDefaultTensorOptions(parameter_guess.options());
    lmp.setSwitching(10000, cpudev);
    lmp.setCopyConvergingEveryN(2);

    lmp.run();

    return lmp.getParameters();
}