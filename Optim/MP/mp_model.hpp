#pragma once

#include "../../pch.hpp"

#include <optional>
#include "../../Expression/expression.hpp"
#include "mp_expr.hpp"

namespace tc {
	namespace optim {
	
		using MP_EvalDiffHessFunc = std::function<void(
			// Constants						// Parameters
			const std::vector<torch::Tensor>&,	const torch::Tensor&,
			// Values							// Jacobian						// Hessian						// Data,
			torch::Tensor&,						tc::OptOutRef<torch::Tensor>,	tc::OptOutRef<torch::Tensor>,	tc::OptRef<const torch::Tensor>)>;

		using MP_FirstDiff = std::function<void(
			// Constants						// Parameters			// Variable index
			const std::vector<torch::Tensor>&,	const torch::Tensor&,	int32_t,
			// Derivative
			torch::Tensor&)>;

		using MP_SecondDiff = std::function<void(
			// Constants						// Parameters			// Variable indices
			const std::vector<torch::Tensor>&,	const torch::Tensor&,	const std::pair<int32_t, int32_t>&,
			// Second Derivative
			torch::Tensor&)>;

		class MP_Model {
		public:

			MP_Model() = delete;
			MP_Model(const MP_Model&) = delete;
		
			MP_Model(const MP_EvalDiffHessFunc& func, const MP_FirstDiff& firstdiff, const MP_SecondDiff& seconddiff);

			MP_Model(const std::string& expression,
				const std::vector<std::string>& parameters,
				tc::OptRef<const std::vector<std::string>> constants);
			
			MP_Model(const std::string& expression,
				const std::vector<std::string>& diffexpressions,
				const std::vector<std::string>& parameters,
				tc::OptRef<const std::vector<std::string>> constants);

			MP_Model(const std::string& expression,
				const std::vector<std::string>& diffexpressions,
				const std::vector<std::string>& seconddiffexpressions,
				const std::vector<std::string>& parameters,
				tc::OptRef<const std::vector<std::string>> constants);

			void to(torch::Device device);
			

			std::vector<torch::Tensor>& constants();
		
			torch::Tensor& parameters();


			void eval(torch::Tensor& value);

			void res(torch::Tensor& residual, const torch::Tensor& data);

			void eval_jac(torch::Tensor& value, torch::Tensor& jacobian);

			void res_jac(torch::Tensor& residual, torch::Tensor& jacobian, const torch::Tensor& data);
			
			void res_jac_hess(torch::Tensor& residual, torch::Tensor& jacobian, torch::Tensor& hessian, const torch::Tensor& data);

			void diff(torch::Tensor& value, int32_t index);

			void second_diff(torch::Tensor& value, const std::pair<int32_t, int32_t>& indices);

		private:
			
			void build_funcs_from_expr();

		private:

			MP_EvalDiffHessFunc m_Func;
			MP_FirstDiff m_FirstDiff;
			MP_SecondDiff m_SecondDiff;

			std::unique_ptr<tc::expression::FetcherMap> m_pFetcherMap;
			std::unique_ptr<MP_Expr> m_pExpr;

			torch::Tensor m_Parameters;
			std::vector<torch::Tensor> m_Constants;
		};

	}

}
