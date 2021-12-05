#include "models.hpp"

#include "../Compute/lstq.hpp"



torch::Tensor models::adc_func(
    std::vector<torch::Tensor> staticvars, 
    torch::Tensor per_problem_inputs, torch::Tensor parameters) 
{
    using namespace torch::indexing;

    return parameters.index({Slice(), 0}).view({parameters.size(0), 1}) * 
        torch::exp(
            -per_problem_inputs.index({Slice(), Slice(), 0}).view({per_problem_inputs.size(0),per_problem_inputs.size(1)}) * 
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
