#pragma once

#include "model.hpp"
#include "optim.hpp"

#include <limits>

namespace tc {
	namespace optim {

		struct STRPSettings : public OptimizerSettings {
		
			STRPSettings();

			torch::Tensor start_residuals;
			torch::Tensor start_jacobian;
			torch::Tensor start_deltas;

			torch::Tensor scaling;


			float mu = 0.25;
			float eta = 0.75;
		};

		struct STRPResult : public OptimResult {
			torch::Tensor finalDeltas;
			torch::Tensor gain;
			
			torch::Tensor lastP; // Last step
			torch::Tensor lastJ; // What the residual Jacboian was before applying last step
			torch::Tensor lastR; // What residuals was before applying last step
		};

		enum eGainType {
			GOOD_GAIN,
			ACCEPTABLE_GAIN,
			POOR_GAIN,
		};

		enum eStepType {
			GAUSS_NEWTON,
			INTERPOLATED,
			CAUCHY, // (Steepest descent)
		};

		class STRP : public Optimizer {
		public:

			STRP() = delete;
			STRP(const STRP&) = delete;
			STRP& operator=(const STRP&) = delete;

			STRP(STRPSettings& settings);

			STRPResult eval();

			std::unique_ptr<OptimResult> base_eval() override;

			static torch::Tensor default_delta_setup(torch::Tensor& parameters, float multiplier = 0.5f);

			static torch::Tensor default_scaling_setup(torch::Tensor& J);

			//					res,			  J
			static std::pair<torch::Tensor, torch::Tensor> default_res_J_setup(optim::Model& model, torch::Tensor data);

		private:

			void dogleg();

			void step();

			void solve();

		private:

			float m_Mu;
			float m_Eta;

		private:

			// (nProblems, nData)
			torch::Tensor data;

			// (nProblems, nData)
			torch::Tensor res;
			torch::Tensor reslike1;

			// (nProblems)
			torch::Tensor delta;
			torch::Tensor deltalike1;
			torch::Tensor deltalike2;
			torch::Tensor deltalike3;
			torch::Tensor deltalike4;

			// (nProblems, nData, nParams)
			torch::Tensor J;
			torch::Tensor Jlike1;

			// (nProblems, nParams)
			torch::Tensor p;
			torch::Tensor plike1;
			torch::Tensor plike2;

			// (nProblems, nParams, nParams)
			torch::Tensor square1;
			torch::Tensor square2;
			torch::Tensor square3;
			torch::Tensor square4;

			// (nProblems, nParams) - int32 type
			torch::Tensor pivots;
			// (nProblems) - int32 type
			torch::Tensor luinfo;

			// (nProblems, nParams, nParams) - floating type
			torch::Tensor scale_matrix;
			torch::Tensor inv_scale_matrix;

			// (nProblems) - booltype
			torch::Tensor stepmask1;
			torch::Tensor stepmask2;
			torch::Tensor stepmask3;

		};



	}
}