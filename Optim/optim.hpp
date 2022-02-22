#pragma once

#include "../pch.hpp"

#include "model.hpp"
#include <atomic>

namespace tc {
	namespace optim {

		class OptimizerSettings {
		public:
			OptimizerSettings() = delete;
			OptimizerSettings(const OptimizerSettings&) = delete;
			OptimizerSettings& operator=(const OptimizerSettings&) = delete;

			OptimizerSettings(std::unique_ptr<optim::Model> pModel, const torch::Tensor& data, tc::ui32 maxiter = 50);

			OptimizerSettings(OptimizerSettings&&) = default;

			virtual ~OptimizerSettings();

			std::unique_ptr<optim::Model>			pModel;
			torch::Tensor							data;
			tc::ui32								maxiter = 50;
		};

		class OptimResult {
		public:

			OptimResult() = delete;
			OptimResult(const OptimResult&) = delete;
			OptimResult& operator=(const OptimResult&) = delete;

			OptimResult(OptimResult&&) = default;

			OptimResult(std::unique_ptr<optim::Model> pFinalModel);
			
			std::unique_ptr<optim::Model> pFinalModel;

		protected:
			friend class Optimizer;


		};

		class Optimizer {
		public:

			Optimizer() = delete;
			Optimizer(const Optimizer&) = delete;
			Optimizer& operator=(const Optimizer&) = delete;

			Optimizer(Optimizer&&) = default;

			Optimizer(OptimizerSettings&& settings);

			void run();

			OptimResult acquire_result();

			void abort();

			tc::ui32 get_n_iter() const;

		protected:

			virtual void on_run() = 0;

			virtual OptimResult on_acquire_result() = 0;

			virtual void on_abort() = 0;

			void set_n_iter(tc::ui32 iter);

			bool should_stop() const;

		protected:

			std::unique_ptr<optim::Model>			pModel;
			torch::Tensor							data;
			tc::ui32								maxiter = 50;

		private:
			bool m_HasAcquiredResult = false;
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