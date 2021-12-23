#pragma once

#include "../pch.hpp"

#include "model.hpp"
#include "optim.hpp"

#include <limits>

namespace tc {
	namespace optim {

		struct SLMASettings : public OptimizerSettings {

			SLMASettings();

			std::optional<torch::Device> switchDevice;
			tc::i32 switchAtN = -1; // Don't switch on default

			float lambdaIncrease = 5.0f;
			float lambdaDecrease = 1.0 / 10.0f;
			float lambdaMax = 1e7;
			float lambdaMin = 1e-7;

		};

		struct SLMAResult : public OptimResult {
			torch::Tensor finalLambdas;
		};

		class SLMA : public Optimizer {
		public:

			SLMA() = delete;
			SLMA(const SLMA&) = delete;
			SLMA& operator=(const SLMA&) = delete;

			SLMA(SLMASettings& settings);

			SLMAResult eval();

			std::unique_ptr<OptimResult> base_eval() override;

		private:

			// performs a step
			void step();

			// removes converging problems from the solver process 
			// (resizes model->params, model->inputs, data_slice and delta
			bool handle_convergence();

			void switch_device();

			void setup_solve();

			void solve();

			void finalize_solve();

		private:

			torch::Device m_CurrentDevice;

			std::optional<torch::Device> m_SwitchDevice;
			tc::i32 m_SwitchNumber = -1;
			bool m_HasSwitched = false;

			torch::Tensor m_Parameters;
			std::optional<torch::Tensor> m_PerProblemInputs;


		private:

			float m_Increase = 5.0f;
			float m_Decrease = 1.0f / 10.0f;
			float m_LambdaMax = 1e7;
			float m_LambdaMin = 0;

			enum eMaskTypes {
				SUCCESSFUL_CHOLESKY = 0,
				UNSUCCESSFUL_CHOLESKY = 1,
				LAMBDA_DECREASED = 2,
				LAMBDA_INCREASED = 4
			};

			torch::Tensor nci;

			tc::i32 numProbs;
			tc::i32 numInputs;
			tc::i32 numParams;

			torch::Tensor data_slice;

			// (nProbs, nPerProbInps, 1) (PRIN)
			torch::Tensor res;					// fp
			torch::Tensor res_t;

			torch::Tensor ep;
			torch::Tensor g_norm;

			// (nProbs, nParams, 1) (PRPA)
			torch::Tensor pD;					// fp

			// (nProbs, nParams)
			// (nProbs, nPerProbInps, nParams)
			torch::Tensor J;					// fp

			// (nProbs, nPerProbInps, 1)
			torch::Tensor JpD;

			// (nProbs, nParams, nParams)

			// (nProbs)
			torch::Tensor lambda;				// fp

			torch::Tensor step_mask;			// int32

		};

	}
}

