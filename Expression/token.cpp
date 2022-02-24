#include "token.hpp"
#include "token.hpp"
#include "token.hpp"
#include "../pch.hpp"

#include "token.hpp"
#include "lexer.hpp"

std::int32_t tc::expression::NoToken::get_id() const
{
	return FixedIDs::NO_TOKEN;
}

std::int32_t tc::expression::NoToken::get_token_type() const
{
	return TokenType::NO_TOKEN;
}


tc::expression::Operator::Operator(const Operator& other)
	: id(other.id), precedence(other.precedence), is_left_associative(other.is_left_associative)
{
}

tc::expression::Operator::Operator(std::int32_t id, std::int32_t precedence, bool is_left_associative)
	: id(id), precedence(precedence), is_left_associative(is_left_associative)
{
}

std::int32_t tc::expression::Operator::get_id() const {
	return id;
}

std::int32_t tc::expression::Operator::get_token_type() const {
	return TokenType::OPERATOR;
}

tc::expression::UnaryOperator::UnaryOperator(const UnaryOperator& other)
	: Operator(other), allowed_left_tokens(other.allowed_left_tokens)
{
}

tc::expression::UnaryOperator::UnaryOperator(std::int32_t id, std::int32_t precedence, bool is_left_associative,
	const std::vector<tc::refw<expression::Token>>& allowed_left_tokens)
	: Operator(id, precedence, is_left_associative), allowed_left_tokens(allowed_left_tokens)
{
}

std::int32_t tc::expression::UnaryOperator::get_operator_type() const
{
	return TokenType::UNARY_OPERATOR;
}

tc::expression::BinaryOperator::BinaryOperator(const BinaryOperator& other)
	: Operator(other), commutative(other.commutative), anti_commutative(other.anti_commutative)
{
}

tc::expression::BinaryOperator::BinaryOperator(std::int32_t id, std::int32_t precedence, bool is_left_associative, bool commutative, bool anti_commutative)
	: Operator(id, precedence, is_left_associative), commutative(commutative), anti_commutative(anti_commutative)
{
}

std::int32_t tc::expression::BinaryOperator::get_operator_type() const
{
	return TokenType::BINARY_OPERATOR;
}

tc::expression::Function::Function(const Function& other)
	: id(other.id), n_inputs(other.n_inputs), commutative(other.commutative),
	commutative_inputs(other.commutative_inputs), anti_commutative_inputs(other.anti_commutative_inputs)
{
}

tc::expression::Function::Function(std::int32_t id, std::int32_t n_inputs, bool commutative)
	: id(id), n_inputs(n_inputs), commutative(commutative)
{

}

tc::expression::Function::Function(std::int32_t id, std::int32_t n_inputs, bool commutative,
	const std::vector<std::vector<int>>& commutative_inputs,
	const std::vector<std::pair<int, int>>& anti_commutative_inputs)
	: id(id), n_inputs(n_inputs), commutative(commutative),
	commutative_inputs(commutative_inputs), anti_commutative_inputs(anti_commutative_inputs)
{

}

std::int32_t tc::expression::Function::get_id() const
{
	return id;
}

std::int32_t tc::expression::Function::get_token_type() const
{
	return TokenType::FUNCTION;
}

tc::expression::Variable::Variable()
	: name("DEFAULT_VARIABLE")
{
}

tc::expression::Variable::Variable(const std::string& name)
	: name(name)
{
}

std::int32_t tc::expression::Variable::get_id() const
{
	return FixedIDs::VARIABLE;
}

std::int32_t tc::expression::Variable::get_token_type() const
{
	return TokenType::VARIABLE;
}

tc::expression::Number::Number()
	: name("DEFAULT_NUMBER"), is_imaginary(false), num(0.0f, 0.0f)
{
}

tc::expression::Number::Number(const std::string& numberstr, bool is_imaginary)
	: name(numberstr), is_imaginary(is_imaginary), 
	num(is_imaginary ? c10::complex<float>(0.0f, std::atof(name.c_str())) : c10::complex<float>(std::atof(name.c_str()), 0.0f))
{
}

std::int32_t tc::expression::Number::get_id() const
{
	return FixedIDs::NUMBER;
}

std::int32_t tc::expression::Number::get_token_type() const
{
	return TokenType::NUMBER;
}

std::string tc::expression::Number::get_full_name() const
{
	return name + ((is_imaginary) ? "i" : "");
}

std::int32_t tc::expression::Zero::get_id() const
{
	return FixedIDs::ZERO;
}

std::int32_t tc::expression::Zero::get_token_type() const
{
	return TokenType::ZERO;
}

std::int32_t tc::expression::Unity::get_id() const
{
	return FixedIDs::UNITY;
}

std::int32_t tc::expression::Unity::get_token_type() const
{
	return TokenType::UNITY;
}

std::int32_t tc::expression::LeftParen::get_id() const
{
	return FixedIDs::LEFT_PAREN;
}

std::int32_t tc::expression::LeftParen::get_token_type() const
{
	return TokenType::LEFT_PAREN;
}

std::int32_t tc::expression::RightParen::get_id() const
{
	return FixedIDs::RIGHT_PAREN;
}

std::int32_t tc::expression::RightParen::get_token_type() const
{
	return TokenType::RIGHT_PAREN;
}

std::int32_t tc::expression::Comma::get_id() const
{
	return FixedIDs::COMMA;
}

std::int32_t tc::expression::Comma::get_token_type() const
{
	return TokenType::COMMA;
}
