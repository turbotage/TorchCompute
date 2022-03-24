#pragma once

#include "mp_optim.hpp"

namespace tc {
	namespace optim {

		class MP_SLMSettings final : public MP_OptimizerSettings {
		public:
			MP_SLMSettings() = delete;
			MP_SLMSettings(const MP_SLMSettings&) = delete;
			MP_SLMSettings& operator=(const MP_SLMSettings&) = delete;

			MP_SLMSettings(MP_SLMSettings&& settings);

			MP_SLMSettings(MP_OptimizerSettings&& optimsettings, const torch::Tensor& start_residuals, const torch::Tensor& start_jacobian,
				const torch::Tensor& start_lambdas, const torch::Tensor& scaling, float mu = 0.0f, float eta = 0.5f, float upmul = 2.0f, float downmul = 1.0f / 3.0f);

			torch::Tensor start_residuals;
			torch::Tensor start_jacobian;
			torch::Tensor start_lambdas;

			torch::Tensor scaling;

			float mu = 0.25f;
			float eta = 0.75f;

			float upmul = 2.0f;
			float downmul = 1.0f / 3.0f;

		};

		class MP_SLMVars {
		public:

			MP_SLMVars() = delete;
			MP_SLMVars(const MP_SLMVars&) = delete;
			MP_SLMVars& operator=(const MP_SLMVars&) = delete;

			static std::unique_ptr<MP_SLMVars> make(std::unique_ptr<optim::MP_Model>& pModel, torch::Tensor& data,
				torch::Tensor& residuals, torch::Tensor& jacobian, torch::Tensor& lambda, torch::Tensor& scaling,
				float mu = 0.0f, float eta = 0.5f, float upmul = 2.0f, float downmul = 1.0f / 3.0f);

			void to_device(const torch::Device& device);

			void to_float32();

			void to_float64();

			void debug_print(bool sizes = true, bool types = true, bool values = false);

		public:

			float mu;
			float eta;

			float upmul;
			float downmul;

			int64_t numProbs;
			int64_t numData;
			int64_t numParam;

		public:

			// (nProblems, nData)
			torch::Tensor res;
			torch::Tensor reslike1;

			// (nProblems)
			torch::Tensor lambda;
			torch::Tensor lambdalike1;
			torch::Tensor lambdalike2;
			torch::Tensor lambdalike3;

			// (nProblems, nData, nParams)
			torch::Tensor J;

			// (nProblems, nParams)
			torch::Tensor plike1;
			torch::Tensor plike2;

			// (nProblems, nParams, nParams)
			torch::Tensor square1;
			torch::Tensor square2;
			torch::Tensor square3;

			// (nProblems) - int32 type
			torch::Tensor info; // used either for cholesky or LU
			// (nProblems, nParams) - int32 type
			torch::Tensor pivots;

			// (nProblems, nParams) - floating type
			torch::Tensor scaling;

			// (nProblems) - booltype
			torch::Tensor stepmask1;
			torch::Tensor stepmask2;
			torch::Tensor stepmask3;

		private:

			MP_SLMVars(const std::unique_ptr<optim::MP_Model>& pModel, const torch::Tensor& data,
				torch::Tensor& residuals, torch::Tensor& jacobian, torch::Tensor& lambda, torch::Tensor& scaling,
				float mu = 0.0f, float eta = 0.5f, float upmul = 2.0f, float downmul = 1.0f / 3.0f);

		};

		class MP_SLM final : public MP_Optimizer {
		public:

			MP_SLM() = delete;
			MP_SLM(const MP_SLM&) = delete;
			MP_SLM& operator=(const MP_SLM&) = delete;

			MP_SLM(MP_SLM&&) = default;

			static std::unique_ptr<MP_SLM> make(MP_SLMSettings&& settings);
			MP_SLM(MP_OptimizerSettings&& optsettings, std::unique_ptr<MP_SLMVars> strpvars);

			torch::Tensor last_parameters();
			torch::Tensor last_step();
			torch::Tensor last_jacobian();
			torch::Tensor last_residuals();
			torch::Tensor last_lambdas();
			torch::Tensor last_multiplier();
			torch::Tensor last_scaling();

			std::unique_ptr<MP_SLMVars> acquire_vars();

			static torch::Tensor default_lambda_setup(const torch::Tensor& parameters, float multiplier = 1.0f);

			static torch::Tensor default_scaling_setup(const torch::Tensor& J, float minimum_scale = 1e-6f);

			//					res,			  J
			static std::pair<torch::Tensor, torch::Tensor> default_res_J_setup(optim::MP_Model& model, const torch::Tensor& data);

		private:

			void on_run(tc::ui32 iter) override;

			void on_acquire_model() override;

			void on_abort() override;

		private:

			void step();

			void solve(tc::ui32 iter);

		private:

			std::unique_ptr<MP_SLMVars> m_pVars;

		};

	}
}

