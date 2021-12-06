#pragma once

#include "../pch.hpp"

#include "model.hpp"
#include "optim.hpp"

namespace optim {

	struct SLMPSettings : public OptimizerSettings {

		SLMPSettings();

		float mu = 0.25;
		float eta = 0.75;
		std::optional<torch::Device> switchDevice;
		ui32 switchAtN = 5000;
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

		SLMP(SLMPSettings settings);

		OptimResult operator()() override;

	private:
		

	private:

		// sets pD and step_mask
		void dogleg();

		// performs a step (stores JpD in pr_in_1_1)
		void step();

		// removes converging pixels from the solver process 
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
		ui32 m_SwitchNumber;
		bool m_HasSwitched = false;

		torch::Tensor m_Parameters;
		torch::Tensor m_PerProblemInputs;


	private:

		torch::Tensor data_slice;

		torch::Tensor nci;

		i32 numProbs;
		i32 numInputs;
		i32 numParams;

		enum MaskTypes {
			SUCCESSFUL_CHOLESKY = 0,
			UNSUCCESSFUL_CHOLESKY = 1,
			FULL_GAUSS_NEWTON = 2,
			SCALED_GRADIENT = 4,
			INTERPOLATED = 8,
		};

		// (nProbs, nPerProbInps, 1) (PRIN)
		torch::Tensor res;					// fp
		torch::Tensor pr_in_1_1;			// fp (Jp)

		// (nProbs, nParams, 1) (PRPA)
		torch::Tensor pD;					// fp
		torch::Tensor pr_pa_1_1;			// fp (pCP, g, )

		// (nProbs, nParams)
		torch::Tensor pr_pa_1;				// fp (Jn, t)


		// (nProbs, nPerProbInps, nParams) (PRINPA)
		torch::Tensor J;					// fp
		torch::Tensor pr_in_pa_1;			// fp (Js)

		// (nProbs, nParams, nParams) (PRPAPA)
		torch::Tensor pr_pa_pa_1;			// fp (D, Hs_chol)
		torch::Tensor pr_pa_pa_2;			// fp (Hs, )

		// (nProbs)
		torch::Tensor delta;				// fp

		torch::Tensor step_mask;			// int32

		torch::Tensor pr_0;					// bool (chol_success, full_gn, interpol_step, gain_mask)

		torch::Tensor pr_1;					// fp (pGN_Norm, ep)
		torch::Tensor pr_2;					// fp (pCP_Norm, et)


	};

}
