#pragma once

#include "mp_optim.hpp"

namespace tc {
	namespace optim {

		class MP_SLMSettings final : public MP_OptimizerSettings {
		public:
			MP_SLMSettings() = delete;
			MP_SLMSettings(const MP_SLMSettings&) = delete;
			MP_SLMSettings& operator=(const MP_SLMSettings&) = delete;

			MP_SLMSettings(MP_STRPSettings&& settings);

			MP_SLMSettings(MP_OptimizerSettings&& optimsettings, const torch::Tensor& start_residuals, const torch::Tensor& start_jacobian,
				const torch::Tensor& start_deltas, const torch::Tensor& scaling, float mu = 0.25, float eta = 0.75);

			torch::Tensor start_residuals;
			torch::Tensor start_jacobian;
			torch::Tensor start_lambdas;

			torch::Tensor scaling;

			float mu = 0.25f;
			float eta = 0.75f;

		}

		class MP_SLMVars {
		public:

			MP_SLMVars() = delete;
			MP_SLMVars(const MP_SLMVars&) = delete;
			MP_SLMVars& operator=(const MP_SLMVars&) = delete;

			static std::unique_ptr<MP_SLMVars> make(std::unique_ptr<optim::MP_Model>& pModel, torch::Tensor& data,
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

		};

		class MP_SLM final : public MP_Optimizer {
		public:

			MP_SLM() = delete;
			MP_SLM(const MP_SLM&) = delete;
			MP_SLM& operator=(const MP_SLM&) = delete;

			MP_SLM(MP_SLM&&) = default;

			static MP_SLM make(MP_SLMSettings&& settings);
			MP_SLM(MP_OptimizerSettings&& optsettings, std::unique_ptr<MP_SLMVars> strpvars);

			torch::Tensor last_parameters();
			torch::Tensor last_step();
			torch::Tensor last_jacobian();
			torch::Tensor last_residuals();
			torch::Tensor last_lambdas();
			torch::Tensor last_multiplier();

			std::unique_ptr<MP_STRPVars> acquire_vars();

			static torch::Tensor default_lambda_setup(torch::Tensor& parameters, float multiplier = 1.0f);

			static torch::Tensor default_scaling_setup(torch::Tensor& J);

			//					res,			  J
			static std::pair<torch::Tensor, torch::Tensor> default_res_J_setup(optim::MP_Model& model, torch::Tensor data);

		private:

			void on_run() override;

			MP_OptimResult on_acquire_result() override;

			void on_abort() override;

		private:

			void dogleg();

			void step();

			void solve();

		private:

			std::unique_ptr<MP_STRPVars> m_pVars;

		};

	}
}

