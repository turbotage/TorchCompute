#include "../pch.hpp"

#include "lexer.hpp"

tc::expression::LexContext::LexContext()
{
	// 0
	unary_operators.emplace_back(
		std::string(1, DefaultOperatorChars::NEG),
		DefaultOperatorPrecedence::NEG,
		false,
		std::vector<tc::refw<Token>>{no_token, left_paren});



	// 0
	binary_operators.emplace_back(
		std::string(1, DefaultOperatorChars::POW),
		DefaultOperatorPrecedence::POW,
		false,
		false,
		false);
	// 1
	binary_operators.emplace_back(
		std::string(1, DefaultOperatorChars::MUL),
		DefaultOperatorPrecedence::MUL,
		true,
		true,
		false);
	// 2
	binary_operators.emplace_back(
		std::string(1, DefaultOperatorChars::DIV),
		DefaultOperatorPrecedence::DIV,
		true,
		false,
		false);
	// 3
	binary_operators.emplace_back(
		std::string(1, DefaultOperatorChars::ADD),
		DefaultOperatorPrecedence::ADD,
		true,
		true,
		false);
	// 4
	binary_operators.emplace_back(
		std::string(1, DefaultOperatorChars::SUB),
		DefaultOperatorPrecedence::SUB,
		true,
		false,
		true);


	// 0
	functions.emplace_back("sin", 1);
	// 1
	functions.emplace_back("cos", 1);
	// 2
	functions.emplace_back("tan", 1);
	// 3
	functions.emplace_back("exp", 1);
	// 4
	functions.emplace_back("log", 1);

}