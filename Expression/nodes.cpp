#include "../pch.hpp"

#include "nodes.hpp"

// <================================== NUMBER ===================================>

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::NumberNode::eval()
{
	return std::make_pair(std::nullopt, std::cref(m_NumToken));
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::NumberNode::diff(const VariableToken& var)
{
	return std::make_pair(std::nullopt, std::cref(m_ZeroToken)); // derivative of number is always zero
}

// <================================== VARIABLE ===================================>

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::VariableNode::eval()
{
	return std::make_pair(m_VariableFetcher(), std::nullopt);
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::VariableNode::diff(const VariableToken& var)
{
	auto sizes = m_VariableFetcher().sizes();
	std::vector<int64_t> vecsizes(sizes.data(), sizes.data() + sizes.size());

	if (var.name == m_VarToken.name) {
		m_UnityToken.sizes = vecsizes;
		return std::make_pair(std::nullopt, std::cref(m_UnityToken));
	}
	m_ZeroToken.sizes = vecsizes;
	return std::make_pair(std::nullopt, std::cref(m_ZeroToken));
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::NegNode::eval()
{
	auto eval_pair = m_Children[0]->eval();

	if (!eval_pair.first.has_value() && !eval_pair.second.has_value())
		throw std::runtime_error("Neither tensor nor token had value in NegNode");

	// Had tensor value, return negative tensor value
	if (eval_pair.first.has_value()) {
		return std::make_pair(-eval_pair.first.value(), std::nullopt);
	}

	// Is token valued, has returned NegUnity, Unity or Zero
	auto& tok = eval_pair.second.value().get();
	if (tok.get_token_type() == TokenType::ZERO) {
		const ZeroToken& ttok = static_cast<const ZeroToken&>(tok);
		m_ZeroToken.sizes = ttok.sizes;
		return std::make_pair(std::nullopt, std::cref(m_ZeroToken));
	}
	else if (tok.get_token_type() == TokenType::UNITY) {
		const UnityToken& ttok = static_cast<const UnityToken&>(tok);
		m_NegUnityToken.sizes = ttok.sizes;
		return std::make_pair(std::nullopt, std::cref(m_NegUnityToken));
	}
	else if (tok.get_token_type() == TokenType::NEG_UNITY) {
		const NegUnityToken& ttok = static_cast<const NegUnityToken&>(tok);
		m_UnityToken.sizes = ttok.sizes;
		return std::make_pair(std::nullopt, std::cref(m_UnityToken));
	}
	else if (tok.get_token_type() == TokenType::NUMBER) {
		const NumberToken& numtok = static_cast<const NumberToken&>(tok);
		m_NumToken.num = -numtok.num;
		m_NumToken.is_imaginary = numtok.is_imaginary;
		m_NumToken.name = "-" + numtok.name;
		m_NumToken.sizes = numtok.sizes;
		return std::make_pair(std::nullopt, std::cref(m_NumToken));
	}

	throw std::runtime_error("Node encountered unsupported token type in eval()");
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::NegNode::diff(const VariableToken& var)
{
	auto eval_pair = m_Children[0]->diff(var);

	if (!eval_pair.first.has_value() && !eval_pair.second.has_value())
		throw std::runtime_error("Neither tensor nor token had value in NegNode");

	// Had tensor value, return negative tensor value
	if (eval_pair.first.has_value()) {
		c10::IntArrayRef ref;
		return std::make_pair(-eval_pair.first.value(), std::nullopt);
	}

	// Is token valued, has returned NegUnity, Unity or Zero
	auto& tok = eval_pair.second.value().get();
	if (tok.get_token_type() == TokenType::ZERO) {
		return std::make_pair(std::nullopt, std::cref(m_ZeroToken));
	}
	else if (tok.get_token_type() == TokenType::UNITY) {
		return std::make_pair(std::nullopt, std::cref(m_NegUnityToken));
	}
	else if (tok.get_token_type() == TokenType::NEG_UNITY) {
		return std::make_pair(std::nullopt, std::cref(m_UnityToken));
	}
	else if (tok.get_token_type() == TokenType::NUMBER) {
		const NumberToken& numtok = dynamic_cast<const NumberToken&>(tok);
		m_pNumToken = std::make_unique<NumberToken>(-numtok.num, numtok.is_imaginary, numtok.sizes);
		return std::make_pair(std::nullopt, std::cref(*m_pNumToken));
	}

	throw std::runtime_error("Node encountered unsupported token type in diff()");
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::PowNode::eval()
{
	return std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>>();
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::PowNode::diff(const VariableToken& var)
{
	throw std::runtime_error("diff() for PowNode has not yet been implemented");
}

