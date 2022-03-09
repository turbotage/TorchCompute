#pragma once

#include "../pch.hpp"


namespace tc {

	namespace models {

		// S = S_0*exp(-b*ADC), varying b values
		void mp_adc_eval_jac_hess(
			// Constants									// Parameters
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,
			// Values										// Jacobian
			torch::Tensor& values,							tc::OptOutRef<torch::Tensor> jacobian,
			// Hessian										// Data
			tc::OptOutRef<torch::Tensor> hessian,			tc::OptRef<const torch::Tensor>data);

		
		// S = S_0 * sin(FA) * (1 - exp(-TR/T1)) / (1 - exp(-TR/T1)cos(FA)), varying flip angles (FA)
		void mp_vfa_eval_jac_hess(
			// Constants									// Parameters
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,
			// Values										// Jacobian
			torch::Tensor& values,							tc::OptOutRef<torch::Tensor> jacobian,
			// Hessian										// Data
			tc::OptOutRef<torch::Tensor> hessian,			tc::OptRef<const torch::Tensor>data);


		// S = S_0 * (1 + (cos(FA) - 1)*exp(-TI/T1) + exp(-TR/T1))
		void mp_psir_eval_jac_hess(
			// Constants									// Parameters
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,
			// Values										// Jacobian
			torch::Tensor& values,							tc::OptOutRef<torch::Tensor> jacobian,
			// Hessian										// Data
			tc::OptOutRef<torch::Tensor> hessian,			tc::OptRef<const torch::Tensor>data);


		// S = |S_0 * (1 + (cos(FA) - 1)*exp(-TI/T1) + exp(-TR/T1))|
		void mp_irmag_eval_jac_hess(
			// Constants									// Parameters
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,
			// Values										// Jacobian
			torch::Tensor& values,							tc::OptOutRef<torch::Tensor> jacobian,
			// Hessian										// Data
			tc::OptOutRef<torch::Tensor> hessian,			tc::OptRef<const torch::Tensor>data);


		// S = S_0 * exp(-TE/T2)
		void mp_t2_eval_jac_hess(
			// Constants									// Parameters
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,
			// Values										// Jacobian
			torch::Tensor& values,							tc::OptOutRef<torch::Tensor> jacobian,
			// Hessian										// Data
			tc::OptOutRef<torch::Tensor> hessian,			tc::OptRef<const torch::Tensor>data);


		void t2_eval_and_diff(std::vector<torch::Tensor>& constants, torch::Tensor& per_problem_inputs, torch::Tensor& parameters,
			tc::OutRef<torch::Tensor> values, tc::OptOutRef<torch::Tensor> jacboian, tc::OptOutRef<const torch::Tensor> data);


	}

}
