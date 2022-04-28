#pragma once

#include "mp_model.hpp"
#include <atomic>

namespace tc {
	namespace optim {

		class MP_OptimizerSettings {
		public:
			MP_OptimizerSettings() = delete;
			MP_OptimizerSettings(const MP_OptimizerSettings&) = delete;
			MP_OptimizerSettings& operator=(const MP_OptimizerSettings&) = delete;

			MP_OptimizerSettings(std::unique_ptr<optim::MP_Model> pModel, const torch::Tensor& data);

			MP_OptimizerSettings(MP_OptimizerSettings&&) = default;

			std::unique_ptr<optim::MP_Model>		pModel;
			torch::Tensor							data;
		};

		class MP_Optimizer {
		public:

			MP_Optimizer() = delete;
			MP_Optimizer(const MP_Optimizer&) = delete;
			MP_Optimizer& operator=(const MP_Optimizer&) = delete;

			MP_Optimizer(MP_Optimizer&&) = default;

			MP_Optimizer(MP_OptimizerSettings&& settings);

			virtual ~MP_Optimizer() = 0;

			void run(tc::ui32 iter);

			std::unique_ptr<optim::MP_Model> acquire_model();

			void abort();

			tc::ui32 get_n_iter() const;

		protected:

			virtual void on_run(tc::ui32 iter) = 0;

			virtual void on_acquire_model() = 0;

			virtual void on_abort() = 0;

			void set_n_iter(tc::ui32 iter);

			bool should_stop() const;

		protected:

			std::unique_ptr<optim::MP_Model>		pModel;
			torch::Tensor							data;

		private:
			bool m_HasAcquiredModel = false;
			// Thread access
			std::atomic<tc::ui32> m_Iter = 0;
			std::atomic<bool> m_ShouldStop = false;
		};




		torch::Tensor get_plane_converging_problems_combined(const torch::Tensor& lastJ, 
			const torch::Tensor& lastP, const torch::Tensor& lastR, float tolerance = 1e-6);

		torch::Tensor get_plane_converging_problems(const torch::Tensor& lastJ,
			const torch::Tensor& lastP, const torch::Tensor& lastR, float tolerance = 1e-6);

		torch::Tensor get_gradient_converging_problems_absolute(const torch::Tensor& J,
			const torch::Tensor& R, float tolerance = 1e-6);

		torch::Tensor get_gradient_converging_problems_relative(const torch::Tensor& J,
			const torch::Tensor& R, float tolerance = 1e-6);

		torch::Tensor get_gradient_converging_problems_combined(const torch::Tensor& J,
			const torch::Tensor& R, float tolerance = 1e-6);

		

	}
}