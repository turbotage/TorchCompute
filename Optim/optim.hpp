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
			torch::Device							startDevice; // Set to CPU by default constructor
			float									tolerance = 1e-6;
			tc::ui32								maxIter = 50;
		};

		struct OptimResult {
			torch::Tensor finalParameters;
			std::unique_ptr<optim::Model> pFinalModel;
			torch::Tensor nonConvergingIndices;
		};

		template<typename ReturnType>
		class Optimizer {
		public:

			Optimizer(OptimizerSettings& settings);

			// IMPORTANT! Any class implementing this virtual function should also call on_eval() at the begining 
			virtual ReturnType eval() = 0;

			void abort();

			std::pair<tc::ui32, tc::ui32> getIterInfo();

		protected:

			void set_iter_info(tc::ui32 iter, tc::ui32 non_converging_probs);

			bool should_stop();

			void on_eval();

			~Optimizer();

		protected:

			std::unique_ptr<optim::Model>			m_pModel;
			torch::Tensor							m_Data;
			torch::Device							m_StartDevice;
			float									m_Tolerance = 1e-4;
			tc::ui32								m_MaxIter = 50;

		private:
			bool m_HasRun = false;

		private:
			// Thread access
			std::atomic<tc::i32> m_NonConvergingProblems;
			std::atomic<tc::ui32> m_Iter = 0;
			std::atomic<bool> m_ShouldStop = false;


		};

		template<typename ReturnType>
		inline Optimizer<ReturnType>::Optimizer(OptimizerSettings& settings)
			: m_pModel(std::move(settings.pModel)), m_Data(settings.data),
			m_StartDevice(settings.startDevice),
			m_Tolerance(settings.tolerance), m_MaxIter(settings.maxIter)
		{
			assert(m_Data.defined() && "Tried to create optimizer with no data");
			assert(m_Data.numel() > 0 && "Tried to create optimizer with no data");
			assert(m_pModel != nullptr && "Tried to create optimizer with pModel=nullptr");
			assert(m_pModel->getParameters().defined() && "Tried to create optimizer with no parameters");
			assert(m_pModel->getParameters().numel() > 0 && "Tried to create optimizer with no parameters");

			m_NonConvergingProblems = settings.data.size(0);
		}

		template<typename ReturnType>
		inline void Optimizer<ReturnType>::abort()
		{
			m_ShouldStop = true;
		}

		template<typename ReturnType>
		inline std::pair<tc::ui32, tc::ui32> Optimizer<ReturnType>::getIterInfo()
		{
			return std::make_pair(m_Iter.load(), m_NonConvergingProblems.load());
		}

		template<typename ReturnType>
		inline void Optimizer<ReturnType>::set_iter_info(tc::ui32 iter, tc::ui32 non_converging_probs)
		{
			m_Iter = iter;
			m_NonConvergingProblems = non_converging_probs;
		}

		template<typename ReturnType>
		inline bool Optimizer<ReturnType>::should_stop()
		{
			return m_ShouldStop;
		}

		template<typename ReturnType>
		inline void Optimizer<ReturnType>::on_eval()
		{
			assert(!m_HasRun && "Tried to evaluate optimizer twice");
			m_HasRun = true;
		}

		template<typename ReturnType>
		inline Optimizer<ReturnType>::~Optimizer()
		{
			assert(m_HasRun && "on_eval() was never run, incorrect implementation of Optimizer");
		}
	}
}