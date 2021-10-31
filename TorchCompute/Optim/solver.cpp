#include "solver.hpp"

optim::BatchedKMeansThenLMP::BatchedKMeansThenLMP(std::unique_ptr<model::Model> pModel,
	torch::Tensor dependents, torch::Tensor data, uint64_t batch_size) 
{
	m_Batchsize = batch_size;
	m_Dependents = dependents;
	m_Data = data;
	
	if (m_Data.options() != m_Dependents.options()) {

	}

	m_pModel = std::move(pModel);

}

void optim::BatchedKMeansThenLMP::makeUniformGuess(std::vector<float> parameters)
{

}

void optim::BatchedKMeansThenLMP::solve()
{
}

