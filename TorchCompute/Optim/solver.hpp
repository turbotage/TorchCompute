#pragma once

#include "../pch.hpp"
#include "model.hpp"

namespace optim {

	// -------------------------------------PARAM(DEPS,DATA)------------------
	using GuessFetchFunc = std::function<torch::Tensor(torch::Tensor, torch::Tensor)>;
	

	class BatchedKMeansThenLMP {
	public:

		BatchedKMeansThenLMP(
			model::Model& model,
			GuessFetchFunc guessFetcher,
			torch::Tensor dependents,
			torch::Tensor data,
			uint64_t batch_size = 100000);

		void solve();

		torch::Tensor getParameters();

	private:
		uint64_t m_Batchsize;
		std::string m_Expression;
		torch::Tensor m_Dependents;
		torch::Tensor m_Parameters;
		torch::Tensor m_Data;
		
		model::Model& m_Model;
		GuessFetchFunc m_GuessFetchFunc;

	};

}

