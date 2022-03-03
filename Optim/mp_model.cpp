#include "../pch.hpp"

#include "mp_model.hpp"

#include "../Compute/gradients.hpp"

tc::optim::MPModel::MPModel(MPEvalDiffHessFunc func)
	: m_Func(func)
{
}

tc::optim::MPModel::MPModel(const std::string& expression, std::optional<std::unordered_map<std::string, int>> parameter_map, std::optional<std::unordered_map<std::string, int>> per_problem_input_map, std::optional<std::unordered_map<std::string, int>> constant_map)
{
	if (per_problem_input_map.has_value())
		m_PerProblemInputMap = per_problem_input_map.value();
	if (parameter_map.has_value())
		m_ParameterMap = parameter_map.value();
	if (constant_map.has_value())
		m_ConstantMap = constant_map.value();



}
