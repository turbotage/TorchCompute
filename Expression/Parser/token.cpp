#include "../../pch.hpp"

#include "token.hpp"
#include "lexer.hpp"

// <=========================== NO_TOKEN ============================>

std::int32_t tc::expression::NoToken::get_id() const
{
	return FixedIDs::NO_TOKEN;
}

std::int32_t tc::expression::NoToken::get_token_type() const
{
	return TokenType::NO_TOKEN;
}

// <=========================== LEFT_PAREN ============================>

std::int32_t tc::expression::LeftParenToken::get_id() const
{
	return FixedIDs::LEFT_PAREN;
}

std::int32_t tc::expression::LeftParenToken::get_token_type() const
{
	return TokenType::LEFT_PAREN;
}

// <=========================== RIGHT_PAREN ============================>

std::int32_t tc::expression::RightParenToken::get_id() const
{
	return FixedIDs::RIGHT_PAREN;
}

std::int32_t tc::expression::RightParenToken::get_token_type() const
{
	return TokenType::RIGHT_PAREN;
}

// <=========================== COMMA ============================>

std::int32_t tc::expression::CommaToken::get_id() const
{
	return FixedIDs::COMMA;
}

std::int32_t tc::expression::CommaToken::get_token_type() const
{
	return TokenType::COMMA;
}

// <=========================== NUMBER ============================>

tc::expression::NumberToken::NumberToken()
	: name("DEFAULT_NUMBER"), is_imaginary(false), num(0.0f, 0.0f), sizes({ 1 })
{
}

tc::expression::NumberToken::NumberToken(const std::string& numberstr, bool is_imaginary)
	: name(numberstr), is_imaginary(is_imaginary),
	num(is_imaginary ? std::complex<float>(0.0f, std::atof(name.c_str())) : std::complex<float>(std::atof(name.c_str()), 0.0f)), sizes({ 1 })
{
}

tc::expression::NumberToken::NumberToken(const std::string& numberstr, bool is_imaginary, const std::vector<int64_t>& sizes)
	: name(numberstr), is_imaginary(is_imaginary),
	num(is_imaginary ? std::complex<float>(0.0f, std::atof(name.c_str())) : std::complex<float>(std::atof(name.c_str()), 0.0f)), sizes(sizes)
{
}

tc::expression::NumberToken::NumberToken(float number, bool is_imaginary)
	: name(is_imaginary ? std::to_string(number) + "i" : std::to_string(number)), is_imaginary(is_imaginary),
	num(is_imaginary ? std::complex<float>(0.0f, number) : std::complex<float>(number, 0.0f)), sizes({ 1 })
{
}

tc::expression::NumberToken::NumberToken(float number, bool is_imaginary, const std::vector<int64_t>& sizes)
	: name(is_imaginary ? std::to_string(number) + "i" : std::to_string(number)), is_imaginary(is_imaginary),
	num(is_imaginary ? std::complex<float>(0.0f, number) : std::complex<float>(number, 0.0f)), sizes(sizes)
{
}

tc::expression::NumberToken::NumberToken(std::complex<float> num, bool is_imaginary)
	: name(is_imaginary ? std::to_string(num.imag()) + "i" : std::to_string(num.real())), is_imaginary(is_imaginary), num(num), sizes({ 1 })
{
}

tc::expression::NumberToken::NumberToken(std::complex<float> num, bool is_imaginary, const std::vector<int64_t>& sizes)
	: name(is_imaginary ? std::to_string(num.imag()) + "i" : std::to_string(num.real())), is_imaginary(is_imaginary), num(num), sizes(sizes)
{
}

std::int32_t tc::expression::NumberToken::get_id() const
{
	return FixedIDs::NUMBER;
}

std::int32_t tc::expression::NumberToken::get_token_type() const
{
	return TokenType::NUMBER;
}

std::string tc::expression::NumberToken::get_full_name() const
{
	return name + ((is_imaginary) ? "i" : "");
}

// <=========================== VARIABLE ============================>

tc::expression::VariableToken::VariableToken()
	: name("DEFAULT_VARIABLE")
{
}

tc::expression::VariableToken::VariableToken(const std::string& name)
	: name(name)
{
}

std::int32_t tc::expression::VariableToken::get_id() const
{
	return FixedIDs::VARIABLE;
}

std::int32_t tc::expression::VariableToken::get_token_type() const
{
	return TokenType::VARIABLE;
}

// <=========================== ZERO ============================>

tc::expression::ZeroToken::ZeroToken()
	: sizes({ 1 })
{
}

tc::expression::ZeroToken::ZeroToken(const std::vector<int64_t>& sizes)
	: sizes(sizes)
{
}

std::int32_t tc::expression::ZeroToken::get_id() const
{
	return FixedIDs::ZERO;
}

std::int32_t tc::expression::ZeroToken::get_token_type() const
{
	return TokenType::ZERO;
}

// <=========================== UNITY ============================>

tc::expression::UnityToken::UnityToken()
	: sizes({ 1 })
{
}

tc::expression::UnityToken::UnityToken(const std::vector<int64_t>& sizes)
	: sizes(sizes)
{
}

std::int32_t tc::expression::UnityToken::get_id() const
{
	return FixedIDs::UNITY;
}

std::int32_t tc::expression::UnityToken::get_token_type() const
{
	return TokenType::UNITY;
}

// <=========================== NEG_UNITY ============================>

tc::expression::NegUnityToken::NegUnityToken()
	: sizes({ 1 })
{
}

tc::expression::NegUnityToken::NegUnityToken(const std::vector<int64_t>& sizes)
	: sizes(sizes)
{
}

std::int32_t tc::expression::NegUnityToken::get_id() const
{
	return FixedIDs::NEG_UNITY;
}

std::int32_t tc::expression::NegUnityToken::get_token_type() const
{
	return TokenType::NEG_UNITY;
}

// <=========================== OPERATOR ============================>

tc::expression::OperatorToken::OperatorToken(const OperatorToken& other)
	: id(other.id), precedence(other.precedence), is_left_associative(other.is_left_associative)
{
}

tc::expression::OperatorToken::OperatorToken(std::int32_t id, std::int32_t precedence, bool is_left_associative)
	: id(id), precedence(precedence), is_left_associative(is_left_associative)
{
}

std::int32_t tc::expression::OperatorToken::get_id() const {
	return id;
}

std::int32_t tc::expression::OperatorToken::get_token_type() const {
	return TokenType::OPERATOR;
}

// <=========================== UNARY_OPERATOR ============================>

tc::expression::UnaryOperatorToken::UnaryOperatorToken(const UnaryOperatorToken& other)
	: OperatorToken(other), allowed_left_tokens(other.allowed_left_tokens)
{
}

tc::expression::UnaryOperatorToken::UnaryOperatorToken(std::int32_t id, std::int32_t precedence, bool is_left_associative,
	const std::vector<tc::refw<expression::Token>>& allowed_left_tokens)
	: OperatorToken(id, precedence, is_left_associative), allowed_left_tokens(allowed_left_tokens)
{
}

std::int32_t tc::expression::UnaryOperatorToken::get_operator_type() const
{
	return TokenType::UNARY_OPERATOR;
}

// <=========================== UNARY_OPERATOR ============================>

tc::expression::BinaryOperatorToken::BinaryOperatorToken(const BinaryOperatorToken& other)
	: OperatorToken(other), commutative(other.commutative), anti_commutative(other.anti_commutative), disallowed_left_tokens(other.disallowed_left_tokens)
{
}

tc::expression::BinaryOperatorToken::BinaryOperatorToken(std::int32_t id, std::int32_t precedence, bool is_left_associative, bool commutative, bool anti_commutative)
	: OperatorToken(id, precedence, is_left_associative), commutative(commutative), anti_commutative(anti_commutative)
{
}

tc::expression::BinaryOperatorToken::BinaryOperatorToken(std::int32_t id, std::int32_t precedence, bool is_left_associative, bool commutative, bool anti_commutative, const std::vector<tc::refw<expression::Token>>& disallowed_left_tokens)
	: OperatorToken(id, precedence, is_left_associative), commutative(commutative), anti_commutative(anti_commutative), disallowed_left_tokens(disallowed_left_tokens)
{	
}

std::int32_t tc::expression::BinaryOperatorToken::get_operator_type() const
{
	return TokenType::BINARY_OPERATOR;
}

// <=========================== FUNCTION ============================>

tc::expression::FunctionToken::FunctionToken(const FunctionToken& other)
	: id(other.id), n_inputs(other.n_inputs), commutative(other.commutative),
	commutative_inputs(other.commutative_inputs), anti_commutative_inputs(other.anti_commutative_inputs)
{
}

tc::expression::FunctionToken::FunctionToken(std::int32_t id, std::int32_t n_inputs, bool commutative)
	: id(id), n_inputs(n_inputs), commutative(commutative)
{

}

tc::expression::FunctionToken::FunctionToken(std::int32_t id, std::int32_t n_inputs, bool commutative,
	const std::vector<std::vector<int>>& commutative_inputs,
	const std::vector<std::pair<int, int>>& anti_commutative_inputs)
	: id(id), n_inputs(n_inputs), commutative(commutative),
	commutative_inputs(commutative_inputs), anti_commutative_inputs(anti_commutative_inputs)
{

}

std::int32_t tc::expression::FunctionToken::get_id() const
{
	return id;
}

std::int32_t tc::expression::FunctionToken::get_token_type() const
{
	return TokenType::FUNCTION;
}
