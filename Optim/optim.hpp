#pragma once

#include "../pch.hpp"

#include "model.hpp"

namespace optim {



	struct OptimizerSettings {

		OptimizerSettings();

		std::unique_ptr<optim::Model>			pModel;
		torch::Tensor							data;
		torch::Device							startDevice; // Set to CPU by default constructor
		float									tolerance = 1e-4;
		ui32									maxIter = 50;
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

	protected:
		
		void on_eval();

		~Optimizer();

	protected:

		std::unique_ptr<optim::Model>			m_pModel;
		torch::Tensor							m_Data;
		torch::Device							m_StartDevice;
		float									m_Tolerance = 1e-4;
		ui32									m_MaxIter = 50;

	private:
		

		bool m_HasRun = false;

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