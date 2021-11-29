#include "models.hpp"

#include "../Compute/lstq.hpp"
#include "../Optim/lm.hpp"



torch::Tensor models::adc_func(
    std::vector<torch::Tensor> staticvars, 
    torch::Tensor dependents, torch::Tensor parameters) 
{
    using namespace torch::indexing;

    return parameters.index({Slice(), 0}).view({parameters.size(0), 1}) * 
        torch::exp(
            -dependents.index({Slice(), Slice(), 0}).view({dependents.size(0),dependents.size(1)}) * 
            parameters.index({Slice(), 1}).view({parameters.size(0), 1})
        );
}

torch::Tensor models::simple_adc_model_linear(torch::Tensor bvals, torch::Tensor data)
{
    using namespace torch::indexing;

    torch::Tensor deps;
    if (bvals.size(0) == 1) {
        deps = bvals.repeat({data.size(0), 1, 1});
    }
    else {
        deps = bvals;
    }

    // deps = parameters, reuse of var
    deps = compute::lstq_qr(deps, torch::log(data));

    deps.index_put_({Slice(),0}, torch::exp(deps.index({Slice(),0})));
    deps.index_put_({Slice(),1}, -1.0 * deps.index({Slice(), 1}));

    return deps;
}

torch::Tensor models::simple_adc_model_nonlinear(torch::Tensor bvals, torch::Tensor data, torch::Tensor parameter_guess)
{
    torch::Device cpudev("cpu");

    optim::ModelFunc modfunc = adc_func;
    optim::Model model(modfunc);

    optim::LMP lmp(model);
    lmp.setParameterGuess(parameter_guess);
    lmp.setDependents(bvals);
    lmp.setData(data);
    lmp.setDefaultTensorOptions(parameter_guess.options());
    lmp.setSwitching(10000, cpudev);
    lmp.setCopyConvergingEveryN(2);

    lmp.run();

    return lmp.getParameters();
}