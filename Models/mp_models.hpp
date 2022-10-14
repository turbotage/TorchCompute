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
			tc::OptOutRef<torch::Tensor> hessian,			tc::OptRef<const torch::Tensor> data);

		void mp_adc_diff(
			// Constants									// Parameters						// Variable index
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,	int32_t index,
			// Derivative
			torch::Tensor& diff);

		void mp_adc_diff2(
			// Constants									// Parameters						// Variable indices
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,	const std::pair<int32_t, int32_t>& indices,
			// Derivative
			torch::Tensor& diff2);

		
		// S = S_0 * sin(FA) * (1 - exp(-TR/T1)) / (1 - exp(-TR/T1)cos(FA)), varying flip angles (FA)
		void mp_vfa_eval_jac_hess(
			// Constants									// Parameters
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,
			// Values										// Jacobian
			torch::Tensor& values,							tc::OptOutRef<torch::Tensor> jacobian,
			// Hessian										// Data
			tc::OptOutRef<torch::Tensor> hessian,			tc::OptRef<const torch::Tensor> data);

		void mp_vfa_diff(
			// Constants									// Parameters						// Variable index
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,	int32_t index,
			// Derivative
			torch::Tensor& diff);

		void mp_vfa_diff2(
			// Constants									// Parameters						// Variable indices
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,	const std::pair<int32_t, int32_t>& indices,
			// Derivative
			torch::Tensor& diff2);



		// S = S_0 * (1 + (cos(FA) - 1)*exp(-TI/T1) + exp(-TR/T1))
		void mp_psir_eval_jac_hess(
			// Constants									// Parameters
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,
			// Values										// Jacobian
			torch::Tensor& values,							tc::OptOutRef<torch::Tensor> jacobian,
			// Hessian										// Data
			tc::OptOutRef<torch::Tensor> hessian,			tc::OptRef<const torch::Tensor> data);

		void mp_psir_diff(
			// Constants									// Parameters						// Variable index
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,	int32_t index,
			// Derivative
			torch::Tensor& diff);

		void mp_psir_diff2(
			// Constants									// Parameters						// Variable indices
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,	const std::pair<int32_t, int32_t>& indices,
			// Derivative
			torch::Tensor& diff2);



		// S = S_0 * (1 + (cos(FA) - 1)*exp(-TI/T1) + exp(-TR/T1)) estimate FA too
		void mp_psirfa_eval_jac_hess(
			// Constants									// Parameters
			const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters,
			// Values										// Jacobian
			torch::Tensor& values, tc::OptOutRef<torch::Tensor> jacobian,
			// Hessian										// Data
			tc::OptOutRef<torch::Tensor> hessian, tc::OptRef<const torch::Tensor>data);

		void mp_psirfa_diff(
			// Constants									// Parameters						// Variable index
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,	int32_t index,
			// Derivative
			torch::Tensor& diff);

		void mp_psirfa_diff2(
			// Constants									// Parameters						// Variable indices
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,	const std::pair<int32_t, int32_t>& indices,
			// Derivative
			torch::Tensor& diff2);



		// S = |S_0 * (1 + (cos(FA) - 1)*exp(-TI/T1) + exp(-TR/T1))|
		void mp_irmag_eval_jac_hess(
			// Constants									// Parameters
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,
			// Values										// Jacobian
			torch::Tensor& values,							tc::OptOutRef<torch::Tensor> jacobian,
			// Hessian										// Data
			tc::OptOutRef<torch::Tensor> hessian,			tc::OptRef<const torch::Tensor> data);

		void mp_irmag_diff(
			// Constants									// Parameters						// Variable index
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,	int32_t index,
			// Derivative
			torch::Tensor& diff);

		void mp_irmag_diff2(
			// Constants									// Parameters						// Variable indices
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,	const std::pair<int32_t, int32_t>& indices,
			// Derivative
			torch::Tensor& diff2);


		// S = |S_0 * (1 + (cos(FA) - 1)*exp(-TI/T1) + exp(-TR/T1))| estimate FA too
		void mp_irmagfa_eval_jac_hess(
			// Constants									// Parameters
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,
			// Values										// Jacobian
			torch::Tensor& values,							tc::OptOutRef<torch::Tensor> jacobian,
			// Hessian										// Data
			tc::OptOutRef<torch::Tensor> hessian,			tc::OptRef<const torch::Tensor> data);

		void mp_irmagfa_diff(
			// Constants									// Parameters						// Variable index
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,	int32_t index,
			// Derivative
			torch::Tensor& diff);

		void mp_irmagfa_diff2(
			// Constants									// Parameters						// Variable indices
			const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, const std::pair<int32_t, int32_t>& indices,
			// Derivative
			torch::Tensor& diff2);




		// S = S_0 * exp(-TE/T2)
		void mp_t2_eval_jac_hess(
			// Constants									// Parameters
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,
			// Values										// Jacobian
			torch::Tensor& values,							tc::OptOutRef<torch::Tensor> jacobian,
			// Hessian										// Data
			tc::OptOutRef<torch::Tensor> hessian,			tc::OptRef<const torch::Tensor> data);

		void mp_t2_diff(
			// Constants									// Parameters						// Variable index
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,	int32_t index,
			// Derivative
			torch::Tensor& diff);

		void mp_t2_diff2(
			// Constants									// Parameters						// Variable indices
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,	const std::pair<int32_t, int32_t>& indices,
			// Derivative
			torch::Tensor& diff2);




		// S = S_0*(f*exp(-b*D1) + (1-f)*exp(-b*D2)), varying b values
		void mp_ivim_eval_jac_hess(
			// Constants									// Parameters
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,
			// Values										// Jacobian
			torch::Tensor& values,							tc::OptOutRef<torch::Tensor> jacobian,
			// Hessian										// Data
			tc::OptOutRef<torch::Tensor> hessian,			tc::OptRef<const torch::Tensor> data);

		void mp_ivim_diff(
			// Constants									// Parameters						// Variable index
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters, int32_t index,
			// Derivative
			torch::Tensor& diff);

		void mp_ivim_diff2(
			// Constants									// Parameters						// Variable indices
			const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters, const std::pair<int32_t, int32_t>& indices,
			// Derivative
			torch::Tensor& diff2);


	}

}
