#include "../pch.hpp"

#include "nodes.hpp"

std::pair<torch::Tensor, bool> tensor_from_eval_pair(std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>>& eval_pair, torch::Device& device)
{
	if (eval_pair.first.has_value())
		return std::make_pair(eval_pair.first.value(), true);


	auto& tok = eval_pair.second.value().get();
	if (tok.get_token_type() == tc::expression::TokenType::NUMBER_TYPE) {

	}
}


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

