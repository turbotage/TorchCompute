#pragma once

#include "model.hpp"
#include "optim.hpp"

namespace tc {
	namespace optim {

		struct GNSettings : public OptimizerSettings {

			GNSettings();

			std::optional<torch::Device> switchDevice;
			tc::ui32 switchAtN = -1;

		};

		struct GNResult : public OptimResult {};

		class GN : public Optimizer {
		public:

			GN() = delete;
			GN(const GN&) = delete;
			GN& operator=(const GN&) = delete;

			GN(GNSettings& settings);

			GNResult eval();

		private:

			void step();

			bool handle_convergence();

			void switch_device();

			void setup_solve();

			void solve();

			void finalize_solve();

		private:

			torch::Device m_CurrentDevice;

			std::optional<torch::Device> m_SwitchDevice;
			tc::i32 m_SwitchNumber;
			bool m_HasSwitched = false;

			torch::Tensor m_Parameters;
			std::optional<torch::Tensor> m_PerProblemInputs;

		private:

			enum MaskTypes {
				SUCCESSFUL_CHOLESKY_GN = 0, // Successful cholesky means gauss newton
				// Unsuccessful cholesky means gradient descent
			};

			float gd_damp = 0.1;

			torch::Tensor nci;

			tc::i32 numProbs;
			tc::i32 numInputs;
			tc::i32 numParams;

			torch::Tensor data_slice;

			// (nProbs, nPerProbInps, 1) (PRIN)
			torch::Tensor res;					// fp

			// (nProbs, nParams, 1) (PRPA)
			torch::Tensor pD;					// fp

			// (nProbs, nPerProbInps, nParams)
			torch::Tensor J;					// fp

			// (nProbs, nPerProbInps, 1)
			torch::Tensor JpD;

			// (nProbs)
			torch::Tensor step_mask;			// int32

		};

	}
}