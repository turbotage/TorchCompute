
#include "models.hpp"

#include "../Compute/lstq.hpp"


torch::Tensor models::vfa_func(
    std::vector<torch::Tensor> staticvars, 
    torch::Tensor per_problem_inputs, torch::Tensor parameters)
{
    using namespace torch::indexing;

    torch::Tensor temp = parameters.index({Slice(), 0}).view({parameters.size(0), 1}); // s0
    temp *= torch::sin(per_problem_inputs.index({Slice(), Slice(), 0}).view({per_problem_inputs.size(0), per_problem_inputs.size(1)})); // s0 * sin(FA)
    torch::Tensor expterm = torch::exp(-staticvars[0]/parameters.index({Slice(), 1}).view({parameters.size(0), 1})); // exp(-TR/T1)
    temp *= (1 - expterm); // s0 * sin(FA) * (1-expterm)
    temp /= (1- expterm*torch::cos(per_problem_inputs.index({Slice(), Slice(), 0}).view({per_problem_inputs.size(0), per_problem_inputs.size(1)}))); // Full expression

    return temp;
}

void models::vfa_jacobian_setter(std::unique_ptr<optim::Model>& pModel, torch::Tensor& jacobian)
{

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

