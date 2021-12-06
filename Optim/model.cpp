#include "model.hpp"

#include "../Expression/shunter.hpp"
#include "../Expression/parsing.hpp"

#include "../Compute/gradients.hpp"

optim::Model::Model(std::string expression, torch::TensorOptions opts,
	std::optional<std::unordered_map<std::string, int>> per_problem_input_map,
	std::optional<std::unordered_map<std::string, int>> parameter_map,
	std::optional<std::unordered_map<std::string, int>> constant_map)
{
	using namespace torch::indexing;

	m_TensorOptions = opts;

	if (per_problem_input_map.has_value())
		m_PerProblemInputMap = per_problem_input_map.value();
	if (parameter_map.has_value())
		m_ParameterMap = parameter_map.value();
	if (constant_map.has_value())
		m_ConstantMap = constant_map.value();
	
	
	m_pSyntaxTree = std::make_unique<expression::ExpressionGraph>(expression);

	// Constants
	if (constant_map.has_value()) {
		for (auto& c : m_ConstantMap) {
			int stv_index = c.second;

			std::function<torch::Tensor()> constant_fetcher = [this, stv_index]() {
				return m_Constants[stv_index];
			};

			m_pSyntaxTree.value()->setVariableFetcher(c.first, constant_fetcher);
		}
	}

	// Per Problem Input Fetchers
	if (per_problem_input_map.has_value()) {
		for (auto& ppi : m_PerProblemInputMap) {
			int dep_index = ppi.second;

			std::function<torch::Tensor()> ppi_fetcher = [this, dep_index]() {
				return m_PerProblemInputs.index({ Slice(), Slice(), dep_index }).view({ m_PerProblemInputs.size(0), m_PerProblemInputs.size(1) });
			};

			m_pSyntaxTree.value()->setVariableFetcher(ppi.first, ppi_fetcher);
		}
	}

	// Parameter fetchers
	if (parameter_map.has_value()) {
		for (auto& p : m_ParameterMap) {
			int par_index = p.second;

			std::function<torch::Tensor()> par_fetcher = [this, par_index]() {
				return m_Parameters.index({ Slice(), par_index }).view({ m_Parameters.size(0), 1 });
			};

			m_pSyntaxTree.value()->setVariableFetcher(p.first, par_fetcher);
		}
	}

	auto eval_func = m_pSyntaxTree.value()->getFunc();

	m_EvalAndDiffFunc = [eval_func, this](std::vector<torch::Tensor>& c, torch::Tensor& pi,
		torch::Tensor& pa, OptOutRef<torch::Tensor> e, OptOutRef<torch::Tensor> j)
	{
		if (e.has_value() && j.has_value()) {

			m_Parameters.requires_grad_(true);

			e.value().get() = eval_func();

			j.value().get() = compute::jacobian(e.value().get(), m_Parameters).requires_grad_(false);

			m_Parameters.requires_grad_(false);
		}
		else if (e.has_value()) {
			e.value().get() = eval_func();
		}
		else if (j.has_value()) {
			m_Parameters.requires_grad_(true);
			j.value().get() = compute::jacobian(eval_func(), m_Parameters).requires_grad_(false);
			m_Parameters.requires_grad_(false);
		}
		else {
			throw std::runtime_error("Called avaluation functions with no output references to evaluate tensor or jacobian tensor");
		}

	};
}

optim::Model::Model(EvalFunc func)
{
	m_EvalAndDiffFunc = [this, func](std::vector<torch::Tensor>& c, torch::Tensor& pi,
		torch::Tensor& pa, OptOutRef<torch::Tensor> e, OptOutRef<torch::Tensor> j) 
	{
		if (e.has_value() && j.has_value()) {

			m_Parameters.requires_grad_(true);

			e.value().get() = func(this->m_Constants, this->m_PerProblemInputs, this->m_Parameters);

			j.value().get() = compute::jacobian(e.value().get(), m_Parameters).requires_grad_(false);

			m_Parameters.requires_grad_(false);
		}
		else if (e.has_value()) {
			e.value().get() = func(this->m_Constants, this->m_PerProblemInputs, this->m_Parameters);
		}
		else if (j.has_value()) {
			m_Parameters.requires_grad_(true);
			j.value().get() = compute::jacobian(func(this->m_Constants, this->m_PerProblemInputs, this->m_Parameters), 
				m_Parameters).requires_grad_(false);
			m_Parameters.requires_grad_(false);
		}
		else {
			throw std::runtime_error("Called avaluation functions with no output references to evaluate tensor or jacobian tensor");
		}

	};
}

optim::Model::Model(EvalAndDiffFunc func)
{
	m_EvalAndDiffFunc = func;
}



void optim::Model::to(torch::Device device)
{
	// Static Vars
	for (int i = 0; i < m_Constants.size(); ++i) {
		if (m_Constants[i].device() != device) {
			m_Constants[i] = m_Constants[i].to(device);
		}
	}

	// Per Problem Inputs
	if (m_PerProblemInputs.device() != device) {
		m_PerProblemInputs = m_PerProblemInputs.to(device);
	}

	// Parameters
	if (m_Parameters.device() != device) {
		m_Parameters = m_Parameters.to(device);
	}

	// We must move the numbers and constants in the expression to the correct device
	if (m_pSyntaxTree.has_value()) {
		m_pSyntaxTree.value()->to(device);
	}

	m_TensorOptions = m_TensorOptions.device(device);
}

void optim::Model::setConstants(std::vector<torch::Tensor> constants)
{
	if (m_pSyntaxTree.has_value()) {
		if (constants.size() != m_ConstantMap.size())
			throw std::runtime_error("Tried to set more Static Variables than were available in Static Variables Map");
	}

	m_Constants = std::move(constants);
}

void optim::Model::setPerProblemInputs(torch::Tensor per_problem_inputs)
{
	if (m_pSyntaxTree.has_value()) {
		if (per_problem_inputs.size(2) != m_PerProblemInputMap.size())
			throw std::runtime_error("Tried to set more Dependents than were available in Dependents Map");
	}

	m_PerProblemInputs = std::move(per_problem_inputs);
}

void optim::Model::setParameters(torch::Tensor parameters)
{
	if (m_pSyntaxTree.has_value()) {
		if (parameters.size(1) != m_ParameterMap.size())
			throw std::runtime_error("Tried to set more Parameters than were available in Parameters Map");
	}

	m_Parameters = std::move(parameters);
}

ui32 optim::Model::getNumProblems()
{
	return m_Parameters.size(0);
}

ui32 optim::Model::getNumConstants()
{
	return m_ConstantMap.size();
}

ui32 optim::Model::getNumParametersPerProblem()
{
	return m_ParameterMap.size();
}

ui32 optim::Model::getNumInputsPerProblem()
{
	return m_PerProblemInputs.size(0);
}

std::vector<torch::Tensor>& optim::Model::getConstants()
{
	return m_Constants;
}

torch::Tensor& optim::Model::getPerProblemInputs()
{
	return m_PerProblemInputs;
}

torch::Tensor& optim::Model::getParameters()
{
	return m_Parameters;
}

void optim::Model::eval(torch::Tensor& value)
{
	m_EvalAndDiffFunc(this->m_Constants, this->m_PerProblemInputs, this->m_Parameters, value, std::nullopt);
}

void optim::Model::eval_diff(torch::Tensor& value, torch::Tensor& jacobian)
{
	m_EvalAndDiffFunc(this->m_Constants, this->m_PerProblemInputs, this->m_Parameters, value, std::ref(jacobian));
}