#include "../pch.hpp"

#include "token.hpp"

const std::string& tc::expression::NoToken::get_id() const
{
	return "no_token";
}

std::int32_t tc::expression::NoToken::get_token_type() const
{
	return eTokenType::NO_TOKEN;
}


tc::expression::Operator::Operator(const std::string& id, std::int32_t precedence, bool is_left_associative)
	: id(id), precedence(precedence), is_left_associative(is_left_associative)
{
}

const std::string& tc::expression::Operator::get_id() const {
	return id;
}

std::int32_t tc::expression::Operator::get_token_type() const {
	return eTokenType::OPERATOR;
}

tc::expression::UnaryOperator::UnaryOperator(const std::string& id, std::int32_t precedence, bool is_left_associative,
	const std::vector<const std::reference_wrapper<expression::Token>>& allowed_left_tokens)
	: Operator(id, precedence, is_left_associative), allowed_left_tokens(allowed_left_tokens)
{
}

std::int32_t tc::expression::UnaryOperator::get_operator_type() const
{
	return eTokenType::UNARY_OPERATOR;
}

tc::expression::BinaryOperator::BinaryOperator(const std::string& id, std::int32_t precedence, bool is_left_associative, bool commutative, bool anti_commutative)
	: Operator(id, precedence, is_left_associative), commutative(commutative), anti_commutative(anti_commutative)
{
}

std::int32_t tc::expression::BinaryOperator::get_operator_type() const
{
	return eTokenType::BINARY_OPERATOR;
}

const std::string& tc::expression::Function::get_id() const
{
	return id;
}

std::int32_t tc::expression::Function::get_token_type() const
{
	return eTokenType::FUNCTION;
}

const std::string& tc::expression::Variable::get_id() const
{
	return id;
}

std::int32_t tc::expression::Variable::get_token_type() const
{
	return eTokenType::VARIABLE;
}

const std::string& tc::expression::Number::get_id() const
{
	return id;
}

std::int32_t tc::expression::Number::get_token_type() const
{
	return eTokenType::NUMBER;
}

const std::string& tc::expression::Zero::get_id() const
{
	return "0";
}

std::int32_t tc::expression::Zero::get_token_type() const
{
	return eTokenType::ZERO;
}

const std::string& tc::expression::Unity::get_id() const
{
	return "1.00";
}

std::int32_t tc::expression::Unity::get_token_type() const
{
	return eTokenType::UNITY;
}

const std::string& tc::expression::LeftParen::get_id() const
{
	return "(";
}

std::int32_t tc::expression::LeftParen::get_token_type() const
{
	return eTokenType::LEFT_PAREN;
}

const std::string& tc::expression::RightParen::get_id() const
{
	return ")";
}

std::int32_t tc::expression::RightParen::get_token_type() const
{
	return eTokenType::RIGHT_PAREN;
}

const std::string& tc::expression::Comma::get_id() const
{
	return ",";
}

std::int32_t tc::expression::Comma::get_token_type() const
{
	return eTokenType::COMMA;
}
