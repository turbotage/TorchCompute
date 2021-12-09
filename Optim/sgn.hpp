#pragma once

#include "model.hpp"
#include "optim.hpp"

namespace optim {

	struct SGNSettings : public OptimizerSettings {

		SGNSettings();

		std::optional<torch::Device> switchDevice;
		ui32 switchAtN = -1;

	};

	struct SGNResult : public OptimResult {};

	class SGN : public Optimizer<SGNResult> {
	public:

		SGN() = delete;
		SGN(const SGN&) = delete;
		SGN& operator=(const SGN&) = delete;

		SGN(SGNSettings& settings);

		SGNResult eval() override;

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
		i32 m_SwitchNumber;
		bool m_HasSwitched = false;

		torch::Tensor m_Parameters;
		torch::Tensor m_PerProblemInputs;

	private:

		enum MaskTypes {
			SUCCESSFUL_CHOLESKY_GN = 0, // Successful cholesky means gauss newton
			// Unsuccessful cholesky means gradient descent
		};

		float gd_damp = 0.1;

		torch::Tensor nci;

		i32 numProbs;
		i32 numInputs;
		i32 numParams;

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