#include "model.hpp"
#include "model.hpp"

#include "../Expression/shunter.hpp"
#include "../Expression/parsing.hpp"

#include "../Compute/gradients.hpp"

tc::optim::Model::Model(const std::string& expression,
	std::optional<std::unordered_map<std::string, int>> parameter_map,
	std::optional<std::unordered_map<std::string, int>> per_problem_input_map,
	std::optional<std::unordered_map<std::string, int>> constant_map)
{
	using namespace torch::indexing;

	if (per_problem_input_map.has_value())
		m_PerProblemInputMap = per_problem_input_map.value();
	if (parameter_map.has_value())
		m_ParameterMap = parameter_map.value();
	if (constant_map.has_value())
		m_ConstantMap = constant_map.value();
	
	
	m_pEvalSyntaxTree = std::make_unique<expression::ExpressionGraph>(expression);

	// Constants
	if (constant_map.has_value()) {
		for (auto& c : m_ConstantMap) {
			int stv_index = c.second;

			std::function<torch::Tensor()> constant_fetcher = [this, stv_index]() {
				return m_Constants[stv_index];
			};

			m_pEvalSyntaxTree.value()->setVariableFetcher(c.first, constant_fetcher);
		}
	}

	// Per Problem Input Fetchers
	if (per_problem_input_map.has_value()) {
		for (auto& ppi : m_PerProblemInputMap) {
			int dep_index = ppi.second;

			std::function<torch::Tensor()> ppi_fetcher = [this, dep_index]() {
				return m_PerProblemInputs.index({ Slice(), Slice(), dep_index }).view({ m_PerProblemInputs.size(0), m_PerProblemInputs.size(1) });
			};

			m_pEvalSyntaxTree.value()->setVariableFetcher(ppi.first, ppi_fetcher);
		}
	}

	// Parameter fetchers
	if (parameter_map.has_value()) {
		for (auto& p : m_ParameterMap) {
			int par_index = p.second;

			std::function<torch::Tensor()> par_fetcher = [this, par_index]() {
				return m_Parameters.index({ Slice(), par_index }).view({ m_Parameters.size(0), 1 });
			};

			m_pEvalSyntaxTree.value()->setVariableFetcher(p.first, par_fetcher);
		}
	}

	auto eval_func = m_pEvalSyntaxTree.value()->getFunc();

	m_EvalAndDiffFunc = [eval_func, this](std::vector<torch::Tensor>& c, torch::Tensor& pi,
		torch::Tensor& pa, torch::Tensor& e, tc::OptOutRef<torch::Tensor> j, tc::OptOutRef<const torch::Tensor> d)
	{
		if (j.has_value()) {
			m_Parameters.requires_grad_(true);
			e = eval_func();
			j.value().get() = compute::jacobian(e, m_Parameters).detach_();
			m_Parameters.requires_grad_(false);
			e.detach_();
		}
		else {
			e = eval_func();
		}

		if (d.has_value()) {
			e = e - d.value().get();
		}

	};
}



tc::optim::Model::Model(const std::string& expression, const std::vector<std::string>& diffexpressions, 
	std::optional<std::unordered_map<std::string, int>> parameter_map, 
	std::optional<std::unordered_map<std::string, int>> per_problem_input_map,
	std::optional<std::unordered_map<std::string, int>> constant_map)
{
	using namespace torch::indexing;

	if (per_problem_input_map.has_value())
		m_PerProblemInputMap = per_problem_input_map.value();
	if (parameter_map.has_value())
		m_ParameterMap = parameter_map.value();
	if (constant_map.has_value())
		m_ConstantMap = constant_map.value();


	m_pEvalSyntaxTree = std::make_unique<expression::ExpressionGraph>(expression);
	m_pDiffSyntaxTrees.emplace();
	for (tc::ui32 i = 0; i < diffexpressions.size(); ++i) {
		m_pDiffSyntaxTrees.value().push_back(std::make_unique<expression::ExpressionGraph>(diffexpressions[i]));
	}


	// Constants
	if (constant_map.has_value()) {
		for (auto& c : m_ConstantMap) {
			int stv_index = c.second;

			std::function<torch::Tensor()> constant_fetcher = [this, stv_index]() {
				return m_Constants[stv_index];
			};

			m_pEvalSyntaxTree.value()->setVariableFetcher(c.first, constant_fetcher);
			for (auto& difftree : m_pDiffSyntaxTrees.value()) {
				difftree->setVariableFetcher(c.first, constant_fetcher);
			}

		}
	}

	// Per Problem Input Fetchers
	if (per_problem_input_map.has_value()) {
		for (auto& ppi : m_PerProblemInputMap) {
			int dep_index = ppi.second;

			std::function<torch::Tensor()> ppi_fetcher = [this, dep_index]() {
				return m_PerProblemInputs.index({ Slice(), Slice(), dep_index }).view({ m_PerProblemInputs.size(0), m_PerProblemInputs.size(1) });
			};

			m_pEvalSyntaxTree.value()->setVariableFetcher(ppi.first, ppi_fetcher);
			for (auto& difftree : m_pDiffSyntaxTrees.value()) {
				difftree->setVariableFetcher(ppi.first, ppi_fetcher);
			}
		}
	}

	// Parameter fetchers
	if (parameter_map.has_value()) {
		for (auto& p : m_ParameterMap) {
			int par_index = p.second;

			std::function<torch::Tensor()> par_fetcher = [this, par_index]() {
				return m_Parameters.index({ Slice(), par_index }).view({ m_Parameters.size(0), 1 });
			};

			m_pEvalSyntaxTree.value()->setVariableFetcher(p.first, par_fetcher);
			for (auto& difftree : m_pDiffSyntaxTrees.value()) {
				difftree->setVariableFetcher(p.first, par_fetcher);
			}
		}
	}

	auto eval_func = m_pEvalSyntaxTree.value()->getFunc();
	std::vector<std::function<torch::Tensor()>> diff_funcs;
	for (auto& difftree : m_pDiffSyntaxTrees.value()) {
		diff_funcs.push_back(difftree->getFunc());
	}

	m_EvalAndDiffFunc = [eval_func, diff_funcs, this](std::vector<torch::Tensor>& c, torch::Tensor& pi,
		torch::Tensor& pa, torch::Tensor& e, tc::OptOutRef<torch::Tensor> j, tc::OptOutRef<const torch::Tensor> d)
	{
		e = eval_func();

		if (j.has_value()) {
			tc::i32 i = 0;
			torch::Tensor& jref = j.value().get();
			for (auto& diff_func : diff_funcs) {
				jref.index_put_({Slice(), Slice(), i}, diff_func());
				++i;
			}
		}

		if (d.has_value()) {
			e = e - d.value().get();
		}

	};
}

tc::optim::Model::Model(EvalFunc func)
{
	m_EvalAndDiffFunc = [func](std::vector<torch::Tensor>& c, torch::Tensor& pi,
		torch::Tensor& pa, torch::Tensor& e, tc::OptOutRef<torch::Tensor> j, tc::OptOutRef<const torch::Tensor> d) 
	{
		if (j.has_value()) {
			pa.requires_grad_(true);
			e = func(c, pi, pa);
			j.value().get() = compute::jacobian(e, pa).detach_();
			pa.requires_grad_(false);
			e.detach_();
		}
		else {
			e = func(c, pi, pa);
		}

		if (d.has_value()) {
			e = e - d.value().get();
		}

	};
}

tc::optim::Model::Model(EvalAndDiffFunc func)
{
	m_EvalAndDiffFunc = func;
}

std::string tc::optim::Model::getReadableExpressionTree()
{
	assert(m_pEvalSyntaxTree.has_value() && "Run getReadableExpressionTree() on a model not built by expression");

	return m_pEvalSyntaxTree.value()->getReadableTree();
}



void tc::optim::Model::to(torch::Device device)
{
	// Static Vars
	for (int i = 0; i < m_Constants.size(); ++i) {
		if (m_Constants[i].device() != device) {
			m_Constants[i] = m_Constants[i].to(device);
		}
	}

	// Per Problem Inputs
	if (m_PerProblemInputs.defined()) {
		if (m_PerProblemInputs.device() != device) {
			m_PerProblemInputs = m_PerProblemInputs.to(device);
		}
	}

	// Parameters
	if (m_Parameters.device() != device) {
		m_Parameters = m_Parameters.to(device);
	}

	// We must move the numbers and constants in the expression to the correct device
	if (m_pEvalSyntaxTree.has_value()) {
		m_pEvalSyntaxTree.value()->to(device);
	}

	// We must move the numbers and constants in the differential expression to the correct device
	if (m_pDiffSyntaxTrees.has_value()) {
		for (auto& difftree : m_pDiffSyntaxTrees.value()) {
			difftree->to(device);
		}
	}

}

void tc::optim::Model::setConstants(const std::vector<torch::Tensor>& constants)
{
	if (m_pEvalSyntaxTree.has_value()) {
		if (constants.size() != m_ConstantMap.size())
			throw std::runtime_error("Tried to set more Static Variables than were available in Static Variables Map");
	}

	m_Constants = constants;
}

void tc::optim::Model::setPerProblemInputs(torch::Tensor per_problem_inputs)
{
	if (m_pEvalSyntaxTree.has_value()) {
		if (per_problem_inputs.size(2) != m_PerProblemInputMap.size())
			throw std::runtime_error("Tried to set more Dependents than were available in Dependents Map");
	}

	m_PerProblemInputs = std::move(per_problem_inputs);
}

void tc::optim::Model::setParameters(torch::Tensor parameters)
{
	if (m_pEvalSyntaxTree.has_value()) {
		if (parameters.size(1) != m_ParameterMap.size())
			throw std::runtime_error("Tried to set more Parameters than were available in Parameters Map");
	}

	m_Parameters = std::move(parameters);
}

tc::ui32 tc::optim::Model::getNumProblems()
{
	return m_Parameters.size(0);
}

tc::ui32 tc::optim::Model::getNumConstants()
{
	return m_Constants.size();
}

tc::ui32 tc::optim::Model::getNumParametersPerProblem()
{
	return m_Parameters.size(1);
}

tc::ui32 tc::optim::Model::getNumInputsPerProblem()
{
	return m_PerProblemInputs.size(1);
}

std::vector<torch::Tensor>& tc::optim::Model::getConstants()
{
	return m_Constants;
}

torch::Tensor& tc::optim::Model::getPerProblemInputs()
{
	return m_PerProblemInputs;
}

torch::Tensor& tc::optim::Model::getParameters()
{
	return m_Parameters;
}

void tc::optim::Model::eval(torch::Tensor& value)
{
	m_EvalAndDiffFunc(this->m_Constants, this->m_PerProblemInputs, this->m_Parameters, value, std::nullopt, std::nullopt);
}

void tc::optim::Model::res(torch::Tensor& value, const torch::Tensor& data) {
	m_EvalAndDiffFunc(this->m_Constants, this->m_PerProblemInputs, this->m_Parameters, value, std::nullopt, std::cref(data));
}

void tc::optim::Model::eval_diff(torch::Tensor& value, torch::Tensor& jacobian)
{
	m_EvalAndDiffFunc(this->m_Constants, this->m_PerProblemInputs, this->m_Parameters, value, std::ref(jacobian), std::nullopt);
}

void tc::optim::Model::res_diff(torch::Tensor& value, torch::Tensor& jacobian, const torch::Tensor& data) {
	m_EvalAndDiffFunc(this->m_Constants, this->m_PerProblemInputs, this->m_Parameters, value, std::ref(jacobian), std::cref(data));
}