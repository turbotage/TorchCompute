#include "mp_expr.hpp"
#include "../pch.hpp"

#include "mp_expr.hpp"
#include "../Expression/Parser/lexer.hpp"
#include "../Expression/Parser/shunter.hpp"

tc::optim::MPExpr::MPExpr(const std::string& expression, const tc::expression::FetcherMap& fetcher_map,
	const std::unordered_map<std::string, int>& parameter_map,
	tc::OptRef<const std::unordered_map<std::string, int>> per_problem_input_map,
	tc::OptRef<const std::unordered_map<std::string, int>> constant_map)
	: fetcher_map(fetcher_map)
{
	this->parameter_map = parameter_map;
	if (per_problem_input_map.has_value())
		this->per_problem_input_map = per_problem_input_map;
	if (constant_map.has_value())
		this->constant_map = constant_map;

	tc::expression::LexContext context;
	for (auto& var : this->parameter_map) {
		context.variables.emplace_back(var.first);
	}
	if (per_problem_input_map.has_value()) {
		for (auto& var : this->per_problem_input_map.value()) {
			context.variables.emplace_back(var.first);
		}
	}
	if (constant_map.has_value()) {
		for (auto& var : this->constant_map.value()) {
			context.variables.emplace_back(var.first);
		}
	}

	tc::expression::Lexer lexer(std::move(context));

	auto toks = lexer.lex(expression);

	tc::expression::Shunter shunter;
	auto shunter_toks = shunter.shunt(std::move(toks));

	eval = std::make_unique<tc::expression::Expression>(shunter_toks,
		tc::expression::Expression::default_expression_creation_map(), this->fetcher_map);

	// Create Jacobian expressions
	for (auto& p : this->parameter_map) {
		diff.emplace(p.second, eval->exprdiffnode(tc::expression::VariableToken(p.first)));
	}

	// Create Hessian expressions
	for (auto& d : diff) {
		for (auto& p : this->parameter_map) {
			seconddiff.emplace(std::make_pair(d.first, p.second), d.second.exprdiffnode(tc::expression::VariableToken(p.first)));
		}
	}

}

tc::optim::MPExpr::MPExpr(const std::string& expression, const std::vector<std::string>& diffexpressions, 
	const tc::expression::FetcherMap& fetcher_map, 
	const std::unordered_map<std::string, int>& parameter_map, 
	tc::OptRef<const std::unordered_map<std::string, int>> per_problem_input_map, 
	tc::OptRef<const std::unordered_map<std::string, int>> constant_map)
	: fetcher_map(fetcher_map)
{
	this->parameter_map = parameter_map;
	if (per_problem_input_map.has_value())
		this->per_problem_input_map = per_problem_input_map;
	if (constant_map.has_value())
		this->constant_map = constant_map;

	tc::expression::LexContext context;
	for (auto& var : this->parameter_map) {
		context.variables.emplace_back(var.first);
	}
	if (per_problem_input_map.has_value()) {
		for (auto& var : this->per_problem_input_map.value()) {
			context.variables.emplace_back(var.first);
		}
	}
	if (constant_map.has_value()) {
		for (auto& var : this->constant_map.value()) {
			context.variables.emplace_back(var.first);
		}
	}

	// Eval
	{
		tc::expression::Lexer lexer(std::move(context));

		auto toks = lexer.lex(expression);

		tc::expression::Shunter shunter;
		auto shunter_toks = shunter.shunt(std::move(toks));

		eval = std::make_unique<tc::expression::Expression>(shunter_toks,
			tc::expression::Expression::default_expression_creation_map(), this->fetcher_map);
	}

	// Diff
	

}
