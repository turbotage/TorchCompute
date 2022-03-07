#include "mp_expr.hpp"
#include "../pch.hpp"

#include "mp_expr.hpp"
#include "../Expression/Parser/lexer.hpp"
#include "../Expression/Parser/shunter.hpp"



tc::optim::MPExpr::MPExpr(const std::string& expression,
	const tc::expression::FetcherMap& fetcher_map,
	const std::vector<std::string>& parameters,
	tc::OptRef<const std::vector<std::string>> constants)
	: fetcher_map(fetcher_map)
{

	tc::expression::LexContext context;
	this->parameters = parameters;
	for (auto& var : this->parameters) {
		context.variables.emplace_back(var);
	}

	this->expression = expression;
	if (constants.has_value())
		this->constants = constants;
	if (constants.has_value()) {
		for (auto& var : this->constants.value()) {
			context.variables.emplace_back(var);
		}
	}

	tc::expression::Lexer lexer(std::move(context));

	auto toks = lexer.lex(expression);

	tc::expression::Shunter shunter;
	auto shunter_toks = shunter.shunt(std::move(toks));

	eval = std::make_unique<tc::expression::Expression>(shunter_toks,
		tc::expression::Expression::default_expression_creation_map(), this->fetcher_map);

	// Create diff expressions
	for (auto& p : this->parameters) {
		diff.emplace_back(eval->exprdiffnode(tc::expression::VariableToken(p)));
	}
	// Get back diff expressions // TODO


	// Create seconddiff expressions
	for (int i = 0; i < diff.size(); ++i) {
		for (int j = 0; j < i + 1; ++j) {
			seconddiff.emplace_back(diff[i].exprdiffnode(tc::expression::VariableToken(parameters[j])));
		}
	}
	// Get back seconddiff expressions // TODO

}

tc::optim::MPExpr::MPExpr(const std::string& expression,
	const std::vector<std::string>& diffexpressions,
	const tc::expression::FetcherMap& fetcher_map,
	const std::vector<std::string>& parameters,
	tc::OptRef<const std::vector<std::string>> constants)
	: fetcher_map(fetcher_map)
{

	this->expression = expression;
	this->parameters = parameters;
	if (constants.has_value())
		this->constants = constants;

	this->diffexpressions = diffexpressions;
	if (diffexpressions.size() != parameters.size())
		throw std::runtime_error("number of diffexpressions was not equal to number of parameters");

	tc::expression::LexContext basecontext;
	for (auto& var : this->parameters) {
		basecontext.variables.emplace_back(var);
	}
	if (constants.has_value()) {
		for (auto& var : this->constants.value()) {
			basecontext.variables.emplace_back(var);
		}
	}

	// Eval
	{
		auto context = basecontext;
		tc::expression::Lexer lexer(std::move(context));

		auto toks = lexer.lex(expression);

		tc::expression::Shunter shunter;
		auto shunter_toks = shunter.shunt(std::move(toks));

		eval = std::make_unique<tc::expression::Expression>(shunter_toks,
			tc::expression::Expression::default_expression_creation_map(), this->fetcher_map);
	}

	// Diff
	for (auto& diffexpr : diffexpressions) {
		auto context = basecontext;
		tc::expression::Lexer lexer(std::move(context));

		auto toks = lexer.lex(diffexpr);
		
		tc::expression::Shunter shunter;
		auto shunter_toks = shunter.shunt(std::move(toks));

		diff.emplace_back(shunter_toks, tc::expression::Expression::default_expression_creation_map(), this->fetcher_map);
	}

	// Create seconddiff expressions
	for (int i = 0; i < diff.size(); ++i) {
		for (int j = 0; j < i + 1; ++j) {
			seconddiff.emplace_back(diff[i].exprdiffnode(tc::expression::VariableToken(parameters[j])));
		}
	}

	// Get back hessian expressions here
	// TODO

}

tc::optim::MPExpr::MPExpr(const std::string& expression,
	const std::vector<std::string>& diffexpressions,
	const std::vector<std::string>& seconddiffexpressions,
	const tc::expression::FetcherMap& fetcher_map,
	const std::vector<std::string>& parameters,
	tc::OptRef<const std::vector<std::string>> constants)
{
	this->parameters = parameters;
	if (constants.has_value())
		this->constants = constants;

	int32_t npars = parameters.size();
	if (diffexpressions.size() != npars)
		throw std::runtime_error("number of diffexpressions was not equal to number of parameters");

	int32_t nhessentries = npars * (npars + 1) / 2;
	if (seconddiffexpressions.size() != nhessentries) {
		throw std::runtime_error("number of unique hessian entries should be " +
			nhessentries + std::string("when number of parameters is ") + std::to_string(npars));
	}

	tc::expression::LexContext basecontext;
	for (auto& var : this->parameters) {
		basecontext.variables.emplace_back(var);
	}
	if (constants.has_value()) {
		for (auto& var : this->constants.value()) {
			basecontext.variables.emplace_back(var);
		}
	}

	// Eval
	{
		auto context = basecontext;
		tc::expression::Lexer lexer(std::move(context));

		auto toks = lexer.lex(expression);

		tc::expression::Shunter shunter;
		auto shunter_toks = shunter.shunt(std::move(toks));

		eval = std::make_unique<tc::expression::Expression>(shunter_toks,
			tc::expression::Expression::default_expression_creation_map(), this->fetcher_map);
	}

	// Diff
	for (auto& d : diffexpressions) {
		auto context = basecontext;
		tc::expression::Lexer lexer(std::move(context));

		auto toks = lexer.lex(d);

		tc::expression::Shunter shunter;
		auto shunter_toks = shunter.shunt(std::move(toks));

		diff.emplace_back(shunter_toks, tc::expression::Expression::default_expression_creation_map(), this->fetcher_map);
	}

	// Second diff
	for (auto& h : seconddiffexpressions) {
		auto context = basecontext;
		tc::expression::Lexer lexer(std::move(context));

		auto toks = lexer.lex(h);

		tc::expression::Shunter shunter;
		auto shunter_toks = shunter.shunt(std::move(toks));

		seconddiff.emplace_back(shunter_toks, tc::expression::Expression::default_expression_creation_map(), this->fetcher_map);
	}

}
