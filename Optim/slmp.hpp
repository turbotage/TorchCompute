#pragma once

#include "model.hpp"
#include "optim.hpp"

#include <limits>

namespace tc {
	namespace optim {

		struct SLMPSettings : public OptimizerSettings {

			SLMPSettings();

			float mu = 0.25;
			float eta = 0.75;
			std::optional<torch::Device> switchDevice;
			tc::i32 switchAtN = -1; // Don't switch on default
		};

		struct SLMPResult : public OptimResult {
			torch::Tensor finalDeltas;
		};

		// Scaled Levenberg-Marquardt with Powel Dogleg (trust region LM)
		class SLMP : public Optimizer {
		public:

			SLMP() = delete;
			SLMP(const SLMP&) = delete;
			SLMP& operator=(const SLMP&) = delete;

			SLMP(SLMPSettings& settings);

			SLMPResult eval();

			std::unique_ptr<OptimResult> base_eval() override;

		private:

			// sets pD and step_mask
			void dogleg();

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
			float m_Mu;
			float m_Eta;

			torch::Device m_CurrentDevice;

			std::optional<torch::Device> m_SwitchDevice;
			tc::i32 m_SwitchNumber = -1;
			bool m_HasSwitched = false;

			torch::Tensor m_Parameters;
			std::optional<torch::Tensor> m_PerProblemInputs;


		private:

			enum eMaskTypes {
				SUCCESSFUL_CHOLESKY = 0,
				UNSUCCESSFUL_CHOLESKY = 1,
				FULL_GAUSS_NEWTON = 2,
				SCALED_GRADIENT = 4,
				INTERPOLATED = 8,
			};

			torch::Tensor nci;

			tc::i32 numProbs;
			tc::i32 numInputs;
			tc::i32 numParams;

			torch::Tensor data_slice;

			// (nProbs, nPerProbInps, 1) (PRIN)
			torch::Tensor res;					// fp
			torch::Tensor res_t;				// fp

			// (nProbs, nParams, 1) (PRPA)
			torch::Tensor pD;					// fp

			// (nProbs, nParams)
			// (nProbs, nPerProbInps, nParams)
			torch::Tensor J;					// fp

			// (nProbs, nPerProbInps, 1)
			torch::Tensor JpD;

			// (nProbs, nParams, nParams)

			// (nProbs)
			torch::Tensor delta;				// fp

			torch::Tensor step_mask;			// int32



		};

	}
}