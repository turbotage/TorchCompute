#include "solver.hpp"

#include "../Compute/kmeans.hpp"
#include "lm.hpp"


optim::BatchedKMeansThenLMP::BatchedKMeansThenLMP(
	optim::Model& model,
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
	if (m_Batchsize > nProblems)
		m_Batchsize = nProblems;

	uint16_t nDataPoints = m_Dependents.size(1);
	uint16_t nDeps = m_Dependents.size(2);
	uint16_t nParams = m_Model.getNParameters();

	torch::TensorOptions tops = m_Dependents.options();
	torch::Device cpudev("cpu");

	m_Parameters = torch::empty({ (int64_t)nProblems, nParams}, tops);
	
	
	uint32_t nBatches = std::ceil((double)nProblems / (double)m_Batchsize);
	
	uint16_t nClusters = m_Batchsize > 20000 ? 2000 : (uint16_t)m_Batchsize * 0.1;
	
	compute::KMeans kmeans(nClusters, 100, 0.01, compute::eKMeansMode::EUCLIDEAN);

	torch::Tensor batchData;
	torch::Tensor batchDeps;
	torch::Tensor batchParams = torch::empty({(int64_t)m_Batchsize, nParams}, tops);
	for (int i = 0; i < nBatches; ++i) {
		int64_t startIndex = i*m_Batchsize;
		int64_t endIndex;
		int64_t batchCount;
		if ((i+1)*m_Batchsize < nProblems) {
			endIndex = (i+1)*m_Batchsize;
			batchCount = m_Batchsize;
		} else {
			endIndex = nProblems; 
			batchCount = nProblems - startIndex;
			batchParams = torch::empty({batchCount, nParams}, tops);
		}

		//std::cout << m_Data.index({ Slice(startIndex, endIndex), Slice() }).sizes() << std::endl;
		//std::cout << m_Data.index({ Slice(startIndex, endIndex), Slice() }).view({ batchCount, nDataPoints }).sizes();

		batchData = m_Data.index({ Slice(startIndex, endIndex), Slice() }).view({ batchCount, nDataPoints });
		batchDeps = m_Dependents.index({ Slice(startIndex, endIndex), Slice(), Slice() }).view({ batchCount, nDataPoints, nDeps });
	
		std::cout << "Batch: " << i << std::endl;

		auto start = std::chrono::system_clock::now();
		torch::Tensor labels = kmeans.fit_predict(batchData, std::nullopt);
		auto end = std::chrono::system_clock::now();
		std::cout << "KMeans took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

		torch::Tensor kmeansData = torch::empty({ nClusters, nDataPoints }, tops.dtype());
		torch::Tensor kmeansDeps = torch::empty({ nClusters, nDataPoints, nDeps }, tops.dtype());
		torch::Tensor kmeansParm = torch::empty({ nClusters, nDataPoints }, tops.dtype());

		std::vector<torch::Tensor> idx;

		for (int j = 0; j < nClusters; ++j) {
			idx.push_back(labels == j);

			kmeansData.index_put_({j, Slice()} ,batchData.index({idx[j], Slice()}).mean(
				{0}, true).to(cpudev));

			kmeansDeps.index_put_({j, Slice(), Slice()}, batchDeps.index({idx[j], Slice(), Slice()}).mean(
				{0}, true).to(cpudev));

		}

		kmeansParm = m_GuessFetchFunc(kmeansDeps, kmeansData);

		m_Model.setDependents(kmeansDeps);
		m_Model.setParameters(kmeansParm);
		m_Model.to(cpudev);

		// Solve on kmeans data
		{
			LMP lmp(m_Model);
			lmp.setParameterGuess(kmeansParm);
			lmp.setDependents(kmeansDeps);
			lmp.setData(kmeansData);
			lmp.setDefaultTensorOptions(kmeansParm.options());
			lmp.setCopyConvergingEveryN(2);
			start = std::chrono::system_clock::now();
			lmp.run();
			end = std::chrono::system_clock::now();
			std::cout << "KMeans solve took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;


			kmeansParm = lmp.getParameters();
		}

		// Set the kmeans solutions to the parameters
		for (int j = 0; j < nClusters; ++j) {
			batchParams.index_put_({ idx[j], Slice() }, kmeansParm.index({j, Slice()}).to(tops));
		}

		m_Model.setDependents(batchDeps);
		m_Model.setParameters(batchParams);
		m_Model.to(tops.device());


		// Solve on all parameters
		{
			LMP lmp(m_Model);
			lmp.setParameterGuess(batchParams);
			lmp.setDependents(batchDeps);
			lmp.setData(batchData);
			lmp.setDefaultTensorOptions(batchParams.options());
			lmp.setCopyConvergingEveryN(2);
			lmp.setSwitching(100-100*(10000.0/(double)batchCount), cpudev);

			start = std::chrono::system_clock::now();
			lmp.run();
			end = std::chrono::system_clock::now();
			std::cout << "Batch solve took: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;


			m_Parameters.index_put_({ Slice(startIndex, endIndex) },
				lmp.getParameters());
		}

	}

}

torch::Tensor optim::BatchedKMeansThenLMP::getParameters()
{
	return m_Parameters;
}

