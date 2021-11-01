#include "solver.hpp"

#include "../Compute/kmeans.hpp"
#include "lm.hpp"


optim::BatchedKMeansThenLMP::BatchedKMeansThenLMP(
	model::Model& model,
	GuessFetchFunc guessFetcher,
	torch::Tensor dependents, torch::Tensor data, 
	uint64_t batch_size)
	: m_Model(model)
{
	m_Batchsize = batch_size;
	m_Dependents = dependents;
	m_Data = data;
	
	m_GuessFetchFunc = guessFetcher;

	if (m_Dependents.size(0) != m_Data.size(0))
		throw std::runtime_error("Not equal amounts of problems in dependents and data");
	else if (m_Dependents.size(1) != m_Data.size(1))
		throw std::runtime_error("Not equal amounts of data points as dependents");

}

// BUG HERE
void optim::BatchedKMeansThenLMP::solve()
{
	using namespace torch::indexing;

	uint64_t nProblems = m_Dependents.size(0);
	uint16_t nDataPoints = m_Dependents.size(1);
	uint16_t nDeps = m_Dependents.size(2);
	uint16_t nParams = m_Model.getNParameters();

	torch::TensorOptions tops = m_Dependents.options();
	torch::Device cpudev("cpu");

	m_Parameters = torch::empty({ (int64_t)nProblems, nParams}, tops);
	
	
	uint32_t nBatches = std::ceil(nProblems / m_Batchsize);
	
	uint16_t nClusters = (uint16_t)m_Batchsize * 0.05;
	if (m_Batchsize > 5000) {
		nClusters = 256;
	}
	
	compute::KMeans kmeans(nClusters, 100, 0.01, compute::eKMeansMode::EUCLIDEAN);

	torch::Tensor batchData;
	torch::Tensor batchDeps;
	torch::Tensor batchParams = torch::empty({(int64_t)m_Batchsize, nParams}, tops);
	for (int i = 0; i < nBatches; ++i) {
		batchData = m_Data.index({ Slice(i * m_Batchsize, (i + 1) * m_Batchsize) }).view({ (int64_t)m_Batchsize, nDataPoints });
		batchDeps = m_Dependents.index({ Slice(i * m_Batchsize, (i + 1) * m_Batchsize) }).view({ (int64_t)m_Batchsize, nDataPoints, nDeps });

		torch::Tensor labels = kmeans.fit_predict(m_Data, std::nullopt);
		
		torch::Tensor kmeansData = torch::empty({ nClusters, nDataPoints }, tops.dtype());
		torch::Tensor kmeansDeps = torch::empty({ nClusters, nDataPoints }, tops.dtype());
		torch::Tensor kmeansParm = torch::empty({ nClusters, nDataPoints }, tops.dtype());

		std::vector<torch::Tensor> idx;

		for (int j = 0; j < nClusters; ++j) {
			idx.push_back(labels == j);
			kmeansData[j] = batchData.index({idx[j]}).mean(
				{0}).view({1, nDataPoints}).to(cpudev);
			kmeansDeps[j] = batchDeps.index({idx[j]}).mean(
				{0}).view({1, nDataPoints, nDeps}).to(cpudev);
		}

		kmeansParm = m_GuessFetchFunc(kmeansDeps, kmeansData);

		m_Model.setDependents(kmeansDeps);
		m_Model.setParameters(kmeansParm);

		// Solve on kmeans data
		{
			LMP lmp(m_Model);
			lmp.setParameterGuess(kmeansParm);
			lmp.setDependents(kmeansDeps);
			lmp.setData(kmeansData);
			lmp.setCopyConvergingEveryN(2);
			lmp.run();

			kmeansParm = lmp.getParameters();
		}

		// Set the kmeans solutions to the parameters
		for (int j = 0; j < nClusters; ++j) {
			batchParams.index_put_({ idx[j] }, kmeansParm[j].to(tops));
		}

		// Solve on all parameters
		{
			LMP lmp(m_Model);
			lmp.setParameterGuess(batchParams);
			lmp.setDependents(batchDeps);
			lmp.setData(batchData);
			lmp.setCopyConvergingEveryN(2);
			lmp.run();

			m_Parameters.index_put_({ Slice(i * m_Batchsize, (i + 1) * m_Batchsize) },
				lmp.getParameters());
		}

	}

}

torch::Tensor optim::BatchedKMeansThenLMP::getParameters()
{
	return m_Parameters;
}

