
#include "models.hpp"

models::VFAModel_1::VFAModel_1() 
{

}

void models::VFAModel_1::setDependents(torch::Tensor flip_angles)
{
    m_pModel->setDependents(flip_angles);
}

void models::VFAModel_1::setData(torch::Tensor data)
{
    m_Data = data;
}

torch::Tensor models::VFAModel_1::solve()
{

}

torch::Tensor models::VFAModel_1::vfa_func(
    std::vector<torch::Tensor> staticvars, 
    torch::Tensor dependents, torch::Tensor parameters)
{


}
'
