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

			MP_OptimizerSettings(std::unique_ptr<optim::MP_Model> pModel, const torch::Tensor& data, tc::ui32 maxiter = 50);

			MP_OptimizerSettings(MP_OptimizerSettings&&) = default;

			virtual ~MP_OptimizerSettings();

			std::unique_ptr<optim::MP_Model>		pModel;
			torch::Tensor							data;
			tc::ui32								maxiter = 50;
		};

		class MP_Optimizer {
		public:

			MP_Optimizer() = delete;
			MP_Optimizer(const MP_Optimizer&) = delete;
			MP_Optimizer& operator=(const MP_Optimizer&) = delete;

			MP_Optimizer(MP_Optimizer&&) = default;

			MP_Optimizer(MP_OptimizerSettings&& settings);

			void run();

			std::unique_ptr<optim::MP_Model> acquire_model();

			void abort();

			tc::ui32 get_n_iter() const;

		protected:

			virtual void on_run() = 0;

			virtual void on_acquire_model() = 0;

			virtual void on_abort() = 0;

			void set_n_iter(tc::ui32 iter);

			bool should_stop() const;

		protected:

			std::unique_ptr<optim::MP_Model>		pModel;
			torch::Tensor							data;
			tc::ui32								maxiter = 50;

		private:
			bool m_HasAcquiredModel = false;
			// Thread access
			std::atomic<tc::ui32> m_Iter = 0;
			std::atomic<bool> m_ShouldStop = false;
		};







		torch::Tensor get_plane_converging_problems_combined(torch::Tensor& lastJ, 
			torch::Tensor& lastP, torch::Tensor& lastR, float tolerance = 1e-6);

		torch::Tensor get_plane_converging_problems(torch::Tensor& lastJ,
			torch::Tensor& lastP, torch::Tensor& lastR, float tolerance = 1e-6);

		torch::Tensor get_gradient_converging_problems_absolute(torch::Tensor& J,
			torch::Tensor& R, float tolerance = 1e-6);

		torch::Tensor get_gradient_converging_problems_relative(torch::Tensor& J,
			torch::Tensor& R, float tolerance = 1e-6);

		torch::Tensor get_gradient_converging_problems_combined(torch::Tensor& J,
			torch::Tensor& R, float tolerance = 1e-6);

		

	}
}