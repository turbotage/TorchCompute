#pragma once

#include "../pch.hpp"

#include "model.hpp"

namespace optim {

															// Model			// Jacobian
	using JacobianSetter = std::function<void(std::unique_ptr<optim::Model>&, torch::Tensor&)>;

	void default_jacobian_setter(std::unique_ptr<optim::Model>& pModel, torch::Tensor& jacobian);


	struct OptimizerSettings {

		OptimizerSettings();

		std::unique_ptr<optim::Model>			pModel;
		torch::Tensor							data;
		JacobianSetter							jacobianSetter = default_jacobian_setter;
		torch::Device							startDevice; // Set to CPU by default constructor
		torch::Device							stopDevice;  // Set to CPU by default constructor
		float									tolerance = 1e-4;
		ui32									maxIter = 50;
	};
			
	struct OptimResult {
		torch::Tensor finalParameters;
		std::unique_ptr<optim::Model> pFinalModel;
		torch::Tensor nonConvergingIndices;
	};

	class Optimizer {
	public:

		// IMPORTANT! Any class implementing this virtual function should also call this
		// function at the begining
		virtual OptimResult operator()() = 0;

	protected:
		
		Optimizer(OptimizerSettings settings);

	protected:

		std::unique_ptr<optim::Model>			m_pModel;
		torch::Tensor							m_Data;
		JacobianSetter							m_JacobianSetter;
		torch::Device							m_StartDevice;
		torch::Device							m_StopDevice;
		float									m_Tolerance = 1e-4;
		ui32									m_MaxIter = 50;

	private:

		bool m_HasRun = false;

	};

}