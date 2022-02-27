#include "../../pch.hpp"

#include "lexer.hpp"

tc::expression::LexContext::LexContext()
{
	// 0
	unary_operators.emplace_back(
		DefaultOperatorIDs::NEG_ID,												// id
		DefaultOperatorPrecedence::NEG_PRECEDENCE,								// precedence
		false,																	// is_left_associative
		std::vector<tc::refw<Token>>{no_token, left_paren, comma});				// allowed_tokens



	// 0
	binary_operators.emplace_back(
		DefaultOperatorIDs::POW_ID,												// id
		DefaultOperatorPrecedence::POW_PRECEDENCE,								// precedence
		false,																	// is_left_associative
		false,																	// commutative
		false,																	// anti_commutative
		std::vector<tc::refw<Token>>{no_token, left_paren, comma});				// disallowed_tokens
	// 1
	binary_operators.emplace_back(
		DefaultOperatorIDs::MUL_ID,												// id
		DefaultOperatorPrecedence::MUL_PRECEDENCE,								// precedence
		true,																	// is_left_associative
		true,																	// commutative
		false,																	// anti_commutative
		std::vector<tc::refw<Token>>{no_token, left_paren, comma});				// disallowed_tokens
	// 2
	binary_operators.emplace_back(
		DefaultOperatorIDs::DIV_ID,												// id
		DefaultOperatorPrecedence::DIV_PRECEDENCE,								// precedence
		true,																	// is_left_associative
		false,																	// commutative
		false,																	// anti_commutative
		std::vector<tc::refw<Token>>{no_token, left_paren, comma});				// disallowed_tokens
	// 3
	binary_operators.emplace_back(
		DefaultOperatorIDs::ADD_ID,												// id
		DefaultOperatorPrecedence::ADD_PRECEDENCE,								// precedence
		true,																	// is_left_associative
		true,																	// commutative
		false,																	// anti_commutative
		std::vector<tc::refw<Token>>{no_token, left_paren, comma});				// disallowed_tokens
	// 4
	binary_operators.emplace_back(
		DefaultOperatorIDs::SUB_ID,												// id
		DefaultOperatorPrecedence::SUB_PRECEDENCE,								// precedence
		true,																	// is_left_associative
		false,																	// commutative
		true,																	// anti_commutative
		std::vector<tc::refw<Token>>{no_token, left_paren, comma});				// disallowed_tokens


	functions.emplace_back(DefaultFunctionIDs::POW_ID, 2);

	functions.emplace_back(DefaultFunctionIDs::ABS_ID, 1);
	functions.emplace_back(DefaultFunctionIDs::SQRT_ID, 1);
	functions.emplace_back(DefaultFunctionIDs::SQUARE_ID, 1);
	functions.emplace_back(DefaultFunctionIDs::EXP_ID, 1);
	functions.emplace_back(DefaultFunctionIDs::LOG_ID, 1);

	functions.emplace_back(DefaultFunctionIDs::SIN_ID, 1);
	functions.emplace_back(DefaultFunctionIDs::COS_ID, 1);
	functions.emplace_back(DefaultFunctionIDs::TAN_ID, 1);
	functions.emplace_back(DefaultFunctionIDs::ASIN_ID, 1);
	functions.emplace_back(DefaultFunctionIDs::ACOS_ID, 1);
	functions.emplace_back(DefaultFunctionIDs::ATAN_ID, 1);
	functions.emplace_back(DefaultFunctionIDs::SINH_ID, 1);
	functions.emplace_back(DefaultFunctionIDs::COSH_ID, 1);
	functions.emplace_back(DefaultFunctionIDs::TANH_ID, 1);
	functions.emplace_back(DefaultFunctionIDs::ASINH_ID, 1);
	functions.emplace_back(DefaultFunctionIDs::ACOSH_ID, 1);
	functions.emplace_back(DefaultFunctionIDs::ATANH_ID, 1);

	operator_id_name_map = DEFAULT_OPERATOR_MAPS;
	function_id_name_map = DEFAULT_FUNCTION_MAPS;

}