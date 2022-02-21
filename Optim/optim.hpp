#pragma once

#include "../pch.hpp"

#include "model.hpp"
#include <atomic>

namespace tc {
	namespace optim {

		struct OptimizerSettings {

			OptimizerSettings();

			std::unique_ptr<optim::Model>			pModel;
			torch::Tensor							data;
			float									tolerance = 1e-6;
			tc::ui32								maxIter = 50;
		};

		struct OptimResult {
			std::unique_ptr<optim::Model> pFinalModel;
		};

		class Optimizer {
		public:

			Optimizer(OptimizerSettings& settings);
			virtual ~Optimizer();

			virtual std::unique_ptr<OptimResult> base_eval() = 0;

			void abort();

			tc::ui32 get_n_iter();

		protected:

			virtual void on_abort();

			void set_n_iter(tc::ui32 iter);

			bool should_stop();

			// This should always be called at begining of eval in other Optimizers deriving from this
			void on_eval();


		protected:

			std::unique_ptr<optim::Model>			m_pModel;
			torch::Tensor							m_Data;
			float									m_Tolerance = 1e-4;
			tc::ui32								m_MaxIter = 50;

		private:
			bool m_HasRun = false;

		private:
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