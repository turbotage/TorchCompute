#pragma once

#include "../pch.hpp"

#include "model.hpp"

namespace optim {

															// Model			// Jacobian
	using JacobianSetter = std::function<void(std::unique_ptr<optim::Model>&, torch::Tensor&)>;

	void default_jacobian_setter(std::unique_ptr<optim::Model>& pModel, torch::Tensor& jacobian);


	struct OptimizerSettings {
		std::unique_ptr<optim::Model>			pModel;
		JacobianSetter							jacobianSetter = default_jacobian_setter;
		torch::TensorOptions					startTensorOptions;
		torch::TensorOptions					stopTensorOptions;
		float									tolerance = 1e-4;
		ui32									maxIter = 50;
	};

	class Optimizer {
	public:

		Optimizer(OptimizerSettings settings);

	protected:

		std::unique_ptr<optim::Model>			m_pModel;
		JacobianSetter							m_JacobianSetter;
		torch::TensorOptions					m_StartTensorOptions;
		torch::TensorOptions					m_StopTensorOptions;
		float									m_Tolerance = 1e-4;
		ui32									m_MaxIter = 50;

	private:


	};

}