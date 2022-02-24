#include "../../pch.hpp"

#include "lexer.hpp"

tc::expression::LexContext::LexContext()
{
	// 0
	unary_operators.emplace_back(
		DefaultOperatorIDs::NEG,												// id
		DefaultOperatorPrecedence::NEG,											// precedence
		false,																	// is_left_associative
		std::vector<tc::refw<Token>>{no_token, left_paren, comma});				// allowed_tokens



	// 0
	binary_operators.emplace_back(
		DefaultOperatorIDs::POW,												// id
		DefaultOperatorPrecedence::POW,											// precedence
		false,																	// is_left_associative
		false,																	// commutative
		false,																	// anti_commutative
		std::vector<tc::refw<Token>>{no_token, left_paren, comma});				// disallowed_tokens
	// 1
	binary_operators.emplace_back(
		DefaultOperatorIDs::MUL,												// id
		DefaultOperatorPrecedence::MUL,											// precedence
		true,																	// is_left_associative
		true,																	// commutative
		false,																	// anti_commutative
		std::vector<tc::refw<Token>>{no_token, left_paren, comma});				// disallowed_tokens
	// 2
	binary_operators.emplace_back(
		DefaultOperatorIDs::DIV,												// id
		DefaultOperatorPrecedence::DIV,											// precedence
		true,																	// is_left_associative
		false,																	// commutative
		false,																	// anti_commutative
		std::vector<tc::refw<Token>>{no_token, left_paren, comma});				// disallowed_tokens
	// 3
	binary_operators.emplace_back(
		DefaultOperatorIDs::ADD,												// id
		DefaultOperatorPrecedence::ADD,											// precedence
		true,																	// is_left_associative
		true,																	// commutative
		false,																	// anti_commutative
		std::vector<tc::refw<Token>>{no_token, left_paren, comma});				// disallowed_tokens
	// 4
	binary_operators.emplace_back(
		DefaultOperatorIDs::SUB,												// id
		DefaultOperatorPrecedence::SUB,											// precedence
		true,																	// is_left_associative
		false,																	// commutative
		true,																	// anti_commutative
		std::vector<tc::refw<Token>>{no_token, left_paren, comma});				// disallowed_tokens


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