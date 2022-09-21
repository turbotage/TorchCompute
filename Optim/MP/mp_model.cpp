#include "../../pch.hpp"

#include "mp_model.hpp"

#include "../../Compute/gradients.hpp"

tc::optim::MP_Model::MP_Model(const MP_EvalDiffHessFunc& func, const MP_FirstDiff& firstdiff, const MP_SecondDiff& seconddiff)
	: m_Func(func), m_FirstDiff(firstdiff), m_SecondDiff(seconddiff)
{

}

tc::optim::MP_Model::MP_Model(const std::string& expression, const std::vector<std::string>& parameters, tc::OptRef<const std::vector<std::string>> constants)
{
	m_pFetcherMap = std::make_unique<tc::expression::FetcherMap>();
	int32_t size = constants.has_value() ? parameters.size() + constants.value().get().size() : parameters.size();
	m_pFetcherMap->reserve(size);

	// Constants
	if (constants.has_value()) {
		auto& consts = constants.value().get();
		for (int i = 0; i < consts.size(); ++i) {
			tc::expression::FetcherFunc constant_fetcher = [this, i]() {
				return m_Constants[i];
			};

			m_pFetcherMap->emplace(consts[i], constant_fetcher);
			
		}
	}

	// Parameters
	for (int i = 0; i < parameters.size(); ++i) {
		tc::expression::FetcherFunc par_fetcher = [this, i]() {
			return m_Parameters.select(1, i).unsqueeze(-1);
		};

		m_pFetcherMap->emplace(parameters[i], std::move(par_fetcher));
	}

	m_pExpr = std::make_unique<MP_Expr>(std::move(MP_Expr(expression, *m_pFetcherMap, parameters, constants)));

	build_funcs_from_expr();

}

tc::optim::MP_Model::MP_Model(const std::string& expression, const std::vector<std::string>& diffexpressions, const std::vector<std::string>& parameters, tc::OptRef<const std::vector<std::string>> constants)
{
	m_pFetcherMap = std::make_unique<tc::expression::FetcherMap>();
	int32_t size = constants.has_value() ? parameters.size() + constants.value().get().size() : parameters.size();
	m_pFetcherMap->reserve(size);

	// Constants
	if (constants.has_value()) {
		auto& consts = constants.value().get();
		for (int i = 0; i < consts.size(); ++i) {
			tc::expression::FetcherFunc constant_fetcher = [this, i]() {
				return m_Constants[i];
			};

			m_pFetcherMap->emplace(consts[i], constant_fetcher);

		}
	}

	// Parameters
	for (int i = 0; i < parameters.size(); ++i) {
		tc::expression::FetcherFunc par_fetcher = [this, i]() {
			return m_Parameters.select(1, i).unsqueeze(-1);
		};

		m_pFetcherMap->emplace(parameters[i], std::move(par_fetcher));
	}

	m_pExpr = std::make_unique<MP_Expr>(std::move(MP_Expr(expression, *m_pFetcherMap, parameters, constants)));

	build_funcs_from_expr();
}

tc::optim::MP_Model::MP_Model(const std::string& expression, const std::vector<std::string>& diffexpressions, const std::vector<std::string>& seconddiffexpressions, const std::vector<std::string>& parameters, tc::OptRef<const std::vector<std::string>> constants)
{
	m_pFetcherMap = std::make_unique<tc::expression::FetcherMap>();
	int32_t size = constants.has_value() ? parameters.size() + constants.value().get().size() : parameters.size();
	m_pFetcherMap->reserve(size);

	// Constants
	if (constants.has_value()) {
		auto& consts = constants.value().get();
		for (int i = 0; i < consts.size(); ++i) {
			tc::expression::FetcherFunc constant_fetcher = [this, i]() {
				return m_Constants[i];
			};

			m_pFetcherMap->emplace(consts[i], constant_fetcher);

		}
	}

	// Parameters
	for (int i = 0; i < parameters.size(); ++i) {
		tc::expression::FetcherFunc par_fetcher = [this, i]() {
			return m_Parameters.select(1, i).unsqueeze(-1);
		};

		m_pFetcherMap->emplace(parameters[i], std::move(par_fetcher));
	}

	m_pExpr = std::make_unique<MP_Expr>(std::move(MP_Expr(expression, *m_pFetcherMap, parameters, constants)));

	build_funcs_from_expr();
}

void tc::optim::MP_Model::to(torch::Device device)
{
	// Move parameters
	if (m_Parameters.device() != device) {
		m_Parameters.to(device);
	}

	// Move constants
	for (auto& c : m_Constants) {
		if (c.device() != device) {
			c.to(device);
		}
	}

}

std::vector<torch::Tensor>& tc::optim::MP_Model::constants()
{
	return m_Constants;
}

torch::Tensor& tc::optim::MP_Model::parameters()
{
	return m_Parameters;
}

void tc::optim::MP_Model::eval(torch::Tensor& value)
{
	return m_Func(m_Constants, m_Parameters, value, std::nullopt, std::nullopt, std::nullopt);
}

void tc::optim::MP_Model::res(torch::Tensor& residual, const torch::Tensor& data)
{
	return m_Func(m_Constants, m_Parameters, residual, std::nullopt, std::nullopt, data);
}

void tc::optim::MP_Model::eval_jac(torch::Tensor& value, torch::Tensor& jacobian)
{
	return m_Func(m_Constants, m_Parameters, value, jacobian, std::nullopt, std::nullopt);
}

void tc::optim::MP_Model::res_jac(torch::Tensor& residual, torch::Tensor& jacobian, const torch::Tensor& data)
{
	return m_Func(m_Constants, m_Parameters, residual, jacobian, std::nullopt, data);
}

void tc::optim::MP_Model::res_jac_hess(torch::Tensor& residual, torch::Tensor& jacobian, torch::Tensor& hessian, const torch::Tensor& data)
{
	return m_Func(m_Constants, m_Parameters, residual, jacobian, hessian, data);
}

void tc::optim::MP_Model::diff(torch::Tensor& value, int32_t index)
{
	return m_FirstDiff(m_Constants, m_Parameters, index, value);
}

void tc::optim::MP_Model::second_diff(torch::Tensor& value, const std::pair<int32_t, int32_t>& indices)
{
	return m_SecondDiff(m_Constants, m_Parameters, indices, value);
}






void tc::optim::MP_Model::build_funcs_from_expr()
{
	
	m_Func = [this](
		// Constants									// Parameters
		const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,
		// Values								// Jacobian								// Hessian								// Data,
		tc::OptOutRef<torch::Tensor> values,	tc::OptOutRef<torch::Tensor> jacobian,	tc::OptOutRef<torch::Tensor> hessian,	tc::OptRef<const torch::Tensor> data)
	{

		if (values.has_value()) {
			values.value().get() = tc::expression::tensor_from_tentok(m_pExpr->eval->eval(), m_Parameters.device());
		}

		if (data.has_value()) {
			values.value().get() -= data.value();
		}

		if (hessian.has_value()) {
			// H = J^T @ J + r @ del2 r

			if (!jacobian.has_value()) {
				throw std::runtime_error("jacobian OptOutRef must be filled if hessian shall be evaluated");
			}

			if (!data.has_value()) {
				throw std::runtime_error("data OptRef must be filled if hessian shall be evaluated");
			}

			int64_t npar = jacobian.value().get().size(2);

			// r @ del2 r
			{
				auto it = m_pExpr->seconddiff.begin();
				for (int i = 0; i < npar; ++i) {
					for (int j = 0; j < i + 1; ++j) {
						hessian.value().get().select(1, i).select(1, j) = torch::sum(torch::mul(values.value().get(), tc::expression::tensor_from_tentok((*it)->eval(), m_Parameters.device())), 1);
						++it;
					}
				}

				// copy symmetric arguments
				for (int i = 0; i < npar; ++i) {
					for (int j = i + 1; j < npar; ++j) {
						hessian.value().get().select(1, i).select(1, j) = hessian.value().get().select(1, j).select(1, i);
					}
				}
			}


			// Jacobian
			for (int i = 0; i < m_pExpr->diff.size(); ++i) {
				jacobian.value().get().select(2, i) = tc::expression::tensor_from_tentok(m_pExpr->diff[i]->eval(), m_Parameters.device());
			}


			hessian.value().get() += torch::bmm(jacobian.value().get().transpose(1, 2), jacobian.value().get());
		}

		if (jacobian.has_value()) {
			// If we wanted hessian the jacobian is already calculated
			if (!hessian.has_value()) {
				for (int i = 0; i < m_pExpr->diff.size(); ++i) {
					jacobian.value().get().select(2, i) = tc::expression::tensor_from_tentok(m_pExpr->diff[i]->eval(), m_Parameters.device());
				}
			}
		}

	};

	m_FirstDiff = [this](
		// Constants									// Parameters						// Variable string
		const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,	int32_t index,
		// Derivative
		torch::Tensor& derivative)
	{
		derivative = tc::expression::tensor_from_tentok(m_pExpr->diff[index]->eval(), m_Parameters.device());
	};

	m_SecondDiff = [this](
		// Constants									// Parameters						// Variable string
		const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,	const std::pair<int32_t, int32_t>& indices,
		// Second Derivative
		torch::Tensor& secondderivative)
	{
		int index;
		if (indices.second > indices.first) {
			index = (indices.second * (indices.second + 1) / 2) + indices.first;
		}
		else {
			index = (indices.first * (indices.first + 1) / 2) + indices.second;
		}

		secondderivative = tc::expression::tensor_from_tentok(m_pExpr->seconddiff[index]->eval(), m_Parameters.device());
	};

}
