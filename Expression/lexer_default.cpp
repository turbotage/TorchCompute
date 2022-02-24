#include "../pch.hpp"

#include "lexer.hpp"

tc::expression::LexContext::LexContext()
{
	// 0
	unary_operators.emplace_back(
		DefaultOperatorIDs::NEG,
		DefaultOperatorPrecedence::NEG,
		false,
		std::vector<tc::refw<Token>>{no_token, left_paren});



	// 0
	binary_operators.emplace_back(
		DefaultOperatorIDs::POW,
		DefaultOperatorPrecedence::POW,
		false,
		false,
		false);
	// 1
	binary_operators.emplace_back(
		DefaultOperatorIDs::MUL,
		DefaultOperatorPrecedence::MUL,
		true,
		true,
		false);
	// 2
	binary_operators.emplace_back(
		DefaultOperatorIDs::DIV,
		DefaultOperatorPrecedence::DIV,
		true,
		false,
		false);
	// 3
	binary_operators.emplace_back(
		DefaultOperatorIDs::ADD,
		DefaultOperatorPrecedence::ADD,
		true,
		true,
		false);
	// 4
	binary_operators.emplace_back(
		DefaultOperatorIDs::SUB,
		DefaultOperatorPrecedence::SUB,
		true,
		false,
		true);


	// 0
	functions.emplace_back(DefaultFunctionIDs::SIN, 1);
	// 1
	functions.emplace_back(DefaultFunctionIDs::COS, 1);
	// 2
	functions.emplace_back(DefaultFunctionIDs::TAN, 1);
	// 3
	functions.emplace_back(DefaultFunctionIDs::EXP, 1);
	// 4
	functions.emplace_back(DefaultFunctionIDs::LOG, 1);
	// 5
	functions.emplace_back(DefaultFunctionIDs::POW, 2);

	operator_id_name_map = DEFAULT_OPERATOR_MAPS;
	function_id_name_map = DEFAULT_FUNCTION_MAPS;

}