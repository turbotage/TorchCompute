#include "model.hpp"

#include "../Expression/shunter.hpp"
#include "../Expression/parsing.hpp"

model::Model::Model(std::string expression, torch::TensorOptions opts,
	std::optional<std::unordered_map<std::string, int>> dependent_map,
	std::optional<std::unordered_map<std::string, int>> parameter_map,
	std::optional<std::unordered_map<std::string, int>> staticvar_map)
{
	using namespace torch::indexing;

	m_TensorOptions = opts;

	if (dependent_map.has_value())
		m_DependentMap = dependent_map.value();
	if (parameter_map.has_value())
		m_ParameterMap = parameter_map.value();
	if (staticvar_map.has_value())
		m_StaticVarMap = staticvar_map.value();

	expression::Shunter shunter(expression);
	std::stack<expression::Token> token_stack(shunter());

	expression::NumberResolver numberResolver = [this](const std::string& str)
	{
		return expression::defaultNumberResolver(str, m_TensorOptions);
	};
	

	m_pSyntaxTree = std::make_unique<expression::ExpressionGraph<torch::Tensor>>(token_stack, numberResolver);

	// Static Var fetchers
	if (staticvar_map.has_value()) {
		for (auto& v : m_StaticVarMap) {
			int stv_index = v.second;

			std::function<torch::Tensor()> statvar_fetcher = [this, stv_index]() {
				return m_StaticVars[stv_index];
			};

			m_pSyntaxTree.value()->setVariableFetcher(v.first, statvar_fetcher);
		}
	}

	// Dependent fetchers
	if (dependent_map.has_value()) {
		for (auto& d : m_DependentMap) {
			int dep_index = d.second;

			std::function<torch::Tensor()> dep_fetcher = [this, dep_index]() {
				return m_Dependents.index({ Slice(), Slice(), dep_index }).view({ m_Dependents.size(0), m_Dependents.size(1) });
			};

			m_pSyntaxTree.value()->setVariableFetcher(d.first, dep_fetcher);
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

	m_Runner = m_pSyntaxTree.value()->getFunc();
}

model::Model::Model(ModelFunc func)
{
	m_Runner = [this, func]() {
		return func(this->m_StaticVars, this->m_Dependents, this->m_Parameters);
	};
}



void model::Model::to(torch::Device device)
{
	// Static Vars
	for (int i = 0; i < m_StaticVars.size(); ++i) {
		if (m_StaticVars[i].device() != device) {
			m_StaticVars[i] = m_StaticVars[i].to(device);
		}
	}

	// Dependents
	if (m_Dependents.device() != device) {
		m_Dependents = m_Dependents.to(device);
	}

	// Parameters
	if (m_Parameters.device() != device) {
		m_Parameters = m_Parameters.to(device);
	}

	m_TensorOptions = m_TensorOptions.device(device);
}

void model::Model::setStaticVariables(std::vector<torch::Tensor>& staticvars)
{
	if (staticvars.size() != m_StaticVarMap.size())
		throw std::runtime_error("Tried to set more Static Variables than were available in Static Variables Map");

	m_StaticVars = staticvars;
}

void model::Model::setDependents(torch::Tensor dependents)
{
	if (dependents.size(2) != m_DependentMap.size())
		throw std::runtime_error("Tried to set more Dependents than were available in Dependents Map");

	m_Dependents = dependents;
}

void model::Model::setParameters(torch::Tensor parameters)
{
	if (parameters.size(1) != m_ParameterMap.size())
		throw std::runtime_error("Tried to set more Parameters than were available in Parameters Map");

	m_Parameters = parameters;
}

uint32_t model::Model::getNParameters()
{
	return m_ParameterMap.size();
}

uint32_t model::Model::getNDeps()
{
	return m_DependentMap.size();
}


torch::Tensor model::Model::operator()()
{
	return m_Runner();
}
