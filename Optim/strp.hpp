#pragma once

#include "optim.hpp"

#include <limits>

namespace tc {
	namespace optim {

		class STRPSettings final : public OptimizerSettings {
		public:
			STRPSettings() = delete;
			STRPSettings(const STRPSettings&) = delete;
			STRPSettings& operator=(const STRPSettings&) = delete;

			STRPSettings(STRPSettings&& settings);

			STRPSettings(OptimizerSettings&& optimsettings, const torch::Tensor& start_residuals, const torch::Tensor& start_jacobian,
				const torch::Tensor& start_deltas, const torch::Tensor& scaling, float mu = 0.25, float eta = 0.74);

			torch::Tensor start_residuals;
			torch::Tensor start_jacobian;
			torch::Tensor start_deltas;

			torch::Tensor scaling;

			float mu = 0.25f;
			float eta = 0.75f;
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

		class STRPVars {
		public:

			STRPVars() = delete;
			STRPVars(const STRPVars&) = delete;
			STRPVars& operator=(const STRPVars&) = delete;

			static std::unique_ptr<STRPVars> make(std::unique_ptr<optim::Model>& pModel, torch::Tensor& data,
				torch::Tensor& residuals, torch::Tensor& jacobian, torch::Tensor& delta, torch::Tensor& scaling, 
				float mu = 0.25, float eta = 0.75);

			void to_device(const torch::Device& device);

			void to_float32();

			void to_float64();

			void debug_print(bool sizes = true, bool types = true, bool values = false);

		public:

			float mu;
			float eta;

			int64_t numProbs;
			int64_t numData;
			int64_t numParam;

		public:

			// (nProblems, nData)
			torch::Tensor res;
			torch::Tensor reslike1;

			// (nProblems)
			torch::Tensor delta;
			torch::Tensor deltalike1;
			torch::Tensor deltalike2;
			torch::Tensor deltalike3;
			torch::Tensor deltalike4;
			torch::Tensor deltalike5;

			// (nProblems, nData, nParams)
			torch::Tensor J;
			torch::Tensor Jlike1;

			// (nProblems, nParams)
			torch::Tensor plike1;
			torch::Tensor plike2;
			torch::Tensor plike3;
			torch::Tensor plike4;

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
			torch::Tensor stepmask4;

		private:

			STRPVars(const std::unique_ptr<optim::Model>& pModel, torch::Tensor& data,
				torch::Tensor& residuals, torch::Tensor& jacobian, torch::Tensor& delta, torch::Tensor& scaling,
				float mu = 0.25, float eta = 0.75);

		};

		class STRP final : public Optimizer {
		public:

			STRP() = delete;
			STRP(const STRP&) = delete;
			STRP& operator=(const STRP&) = delete;

			STRP(STRP&&) = default;

			static STRP make(STRPSettings&& settings);
			STRP(OptimizerSettings&& optsettings, std::unique_ptr<STRPVars> strpvars);

			torch::Tensor last_parameters();
			torch::Tensor last_step();
			torch::Tensor last_jacobian();
			torch::Tensor last_residuals();
			torch::Tensor last_deltas();
			torch::Tensor last_multiplier();

			std::unique_ptr<STRPVars> acquire_vars();

			static torch::Tensor default_delta_setup(torch::Tensor& parameters, float multiplier = 1.0f);

			static torch::Tensor default_scaling_setup(torch::Tensor& J);

			//					res,			  J
			static std::pair<torch::Tensor, torch::Tensor> default_res_J_setup(optim::Model& model, torch::Tensor data);

		private:

			void on_run() override;

			OptimResult on_acquire_result() override;

			void on_abort() override;

		private:

			void dogleg();

			void step();

			void solve();

		private:

			std::unique_ptr<STRPVars> m_pVars;

		};



	}
}