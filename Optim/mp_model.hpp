#pragma once

#include "../pch.hpp"

#include <optional>
#include "../Expression/expression.hpp"
#include "mp_expr.hpp"

namespace tc {
	namespace optim {
	
		using MPEvalDiffHessFunc = std::function<void(
			// Constants					// PerProblemInputs				// Parameters
			std::vector<torch::Tensor>&, torch::Tensor&,					torch::Tensor&,
			// Values						// Jacobian						// Hessian						// Data,
			tc::OptOutRef<torch::Tensor>,	tc::OptOutRef<torch::Tensor>,	tc::OptOutRef<torch::Tensor>,	tc::OptRef<torch::Tensor>)>;


		class MPModel {
		public:

			MPModel() = delete;
		
			MPModel(MPEvalDiffHessFunc func);

			MPModel(MPExpr&& mpexpr);
			
			void to(torch::Device device);
			

			std::vector<torch::Tensor>& constants();
		
			torch::Tensor& parameters();


			void eval(torch::Tensor& value);

			void res(torch::Tensor& residual, const torch::Tensor& data);

			void eval_diff(torch::Tensor& value, torch::Tensor& jacobian);

			void res_diff(torch::Tensor& residual, torch::Tensor& jacobian, const torch::Tensor& data);
			
			void eval_diff_hess(torch::Tensor& value, torch::Tensor& jacobian, torch::Tensor& hessian, const torch::Tensor& data);

			void res_diff_hess(torch::Tensor& residual, torch::Tensor& jacobian, torch::Tensor& hessian, const torch::Tensor& data);



		private:

			MPEvalDiffHessFunc m_Func;

			std::unordered_map<std::string, int> m_ConstantMap;
			std::vector<torch::Tensor> m_Constants;

			std::unordered_map<std::string, int> m_ParameterMap;
			torch::Tensor m_Parameters;
		};

	}

}
