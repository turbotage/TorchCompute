#pragma once

#include "../pch.hpp"
#include "model.hpp"

namespace optim {

	class BatchedKMeansThenLMP {
	public:

		BatchedKMeansThenLMP(
			std::unique_ptr<model::Model> pModel,
			torch::Tensor dependents,
			torch::Tensor data,
			uint64_t batch_size = 100000);
	
		void makeUniformGuess(std::vector<float> parameters);

		void solve();

	private:
		uint64_t m_Batchsize;
		std::string m_Expression;
		torch::Tensor m_Dependents;
		torch::Tensor m_Parameters;
		torch::Tensor m_Data;

		std::unique_ptr<model::Model> m_pModel;

	};

}

