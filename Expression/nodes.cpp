#include "../pch.hpp"

#include "nodes.hpp"

std::unique_ptr<tc::expression::Node> tc::expression::node_from_pair(const std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>>& pair)
{
	if (pair.first.has_value()) {
		return std::make_unique<TensorNode>(pair.first.value());
	}
	else if (pair.second.has_value()) {
		return std::make_unique<TokenNode>(pair.second.value());
	}
	throw std::runtime_error("node_from_pair() received a pair with only nullopt content");
}


// <================================== TOKEN ===================================>

tc::expression::TokenNode::TokenNode(const Token& tok)
{
	m_TokenType = tok.get_token_type();

	switch (m_TokenType) {
	case TokenType::ZERO_TYPE:
	{
		m_ZeroToken = static_cast<const ZeroToken&>(tok);
	}
	break;
	case TokenType::UNITY_TYPE:
	{
		m_UnityToken = static_cast<const UnityToken&>(tok);
		m_ZeroToken.sizes = m_UnityToken.sizes;
	}
	break;
	case TokenType::NEG_UNITY_TYPE:
	{
		m_NegUnityToken = static_cast<const NegUnityToken&>(tok);
		m_ZeroToken.sizes = m_NegUnityToken.sizes;
	}
	break;
	case TokenType::NAN_TYPE:
	{
		m_NanToken = static_cast<const NanToken&>(tok);
		m_ZeroToken.sizes = m_NanToken.sizes;
	}
	break;
	case TokenType::NUMBER_TYPE:
	{
		m_NumToken = static_cast<const NumberToken&>(tok);
		m_ZeroToken.sizes = m_NumToken.sizes;
	}
	break;
	default:
		throw std::runtime_error("Can't construct TokenNode from token type other than Zero,Unity,NegUnity,Nan,Number");
	}
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::TokenNode::eval()
{
	switch (m_TokenType) {
	case TokenType::ZERO_TYPE:
	{
		return std::make_pair(std::nullopt, std::cref(m_ZeroToken));
	}
	break;
	case TokenType::UNITY_TYPE:
	{
		return std::make_pair(std::nullopt, std::cref(m_UnityToken));
	}
	break;
	case TokenType::NEG_UNITY_TYPE:
	{
		return std::make_pair(std::nullopt, std::cref(m_NegUnityToken));
	}
	break;
	case TokenType::NAN_TYPE:
	{
		return std::make_pair(std::nullopt, std::cref(m_NanToken));
	}
	break;
	case TokenType::NUMBER_TYPE:
	{
		return std::make_pair(std::nullopt, std::cref(m_NumToken));
	}
	break;
	default:
		throw std::runtime_error("TokenNode was constructed with token type other than Zero,Unity,NegUnity,Nan,Number, this should not be possible");
	}
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::TokenNode::diff(const VariableToken& var)
{
	return std::make_pair(std::nullopt, std::cref(m_ZeroToken)); // derivative of number is always zero
}

// <================================== TOKEN ===================================>

tc::expression::TensorNode::TensorNode(const torch::Tensor& tensor)
	: m_Tensor(tensor)
{
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::TensorNode::eval()
{
	return std::make_pair(m_Tensor, std::nullopt);
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::TensorNode::diff(const VariableToken& var)
{
	auto sizes = m_Tensor.sizes();
	std::vector<int64_t> vecsizes(sizes.data(), sizes.data() + sizes.size());

	m_ZeroToken.sizes = vecsizes;
	return std::make_pair(std::nullopt, std::cref(m_ZeroToken));
}

// <================================== VARIABLE ===================================>

tc::expression::VariableNode::VariableNode(const VariableToken& token, const std::function<torch::Tensor()>& variable_fetcher)
	: m_VarToken(token), m_VariableFetcher(variable_fetcher)
{
}

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

// <================================== NEG ===================================>

tc::expression::NegNode::NegNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::NegNode::eval()
{
	auto ep = m_Children[0]->eval();

	if (ep.first.has_value()) {
		return std::make_pair(-ep.first.value(), std::nullopt);
	}
	else {
		m_pToken = std::move(-ep.second.value().get());
		return std::make_pair(std::nullopt, *m_pToken);
	}

}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::NegNode::diff(const VariableToken& var)
{
	auto ep = m_Children[0]->diff(var);

	if (ep.first.has_value()) {
		return std::make_pair(-ep.first.value(), std::nullopt);
	}
	else {
		m_pToken = std::move(-ep.second.value().get());
		return std::make_pair(std::nullopt, *m_pToken);
	}
}

// <================================== MUL ===================================>

tc::expression::MulNode::MulNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
{
	m_Children.resize(2);
	m_Children[0] = std::move(left_child);
	m_Children[1] = std::move(right_child);
}


std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::MulNode::eval()
{
	auto epl = m_Children[0]->eval();
	auto epr = m_Children[1]->eval();

	if (epl.first.has_value() && epr.first.has_value()) {
		return std::make_pair(epl.first.value() * epr.first.value(), std::nullopt);
	}
	else if (epl.first.has_value() && epr.second.has_value()) {
		return std::make_pair(epl.first.value() * epr.second.value(), std::nullopt);
	}
	else if (epl.second.has_value() && epr.first.has_value()) {
		return std::make_pair(epl.second.value() * epr.first.value(), std::nullopt);
	}
	else if (epl.second.has_value() && epr.second.has_value()) {
		m_pToken = std::move(epl.second.value() * epr.second.value());
		return std::make_pair(std::nullopt, *m_pToken);
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}

}

// Product rule
std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::MulNode::diff(const VariableToken& var)
{
	auto epl = node_from_pair(m_Children[0]->eval());
	auto epr = node_from_pair(m_Children[0]->eval());

	auto epldiff = node_from_pair(m_Children[0]->diff(var));
	auto eprdiff = node_from_pair(m_Children[0]->diff(var));

	auto l = node_from_pair(MulNode(std::move(epldiff), std::move(epr)).eval());
	auto r = node_from_pair(MulNode(std::move(eprdiff), std::move(epl)).eval());

	return AddNode(std::move(l), std::move(r)).eval();
}

// <================================== DIV ===================================>

tc::expression::DivNode::DivNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
{
	m_Children.resize(2);
	m_Children[0] = std::move(left_child);
	m_Children[1] = std::move(right_child);
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::DivNode::eval()
{
	auto epl = m_Children[0]->eval();
	auto epr = m_Children[1]->eval();

	if (epl.first.has_value() && epr.first.has_value()) {
		return std::make_pair(epl.first.value() / epr.first.value(), std::nullopt);
	}
	else if (epl.first.has_value() && epr.second.has_value()) {
		return std::make_pair(epl.first.value() / epr.second.value(), std::nullopt);
	}
	else if (epl.second.has_value() && epr.first.has_value()) {
		return std::make_pair(epl.second.value() / epr.first.value(), std::nullopt);
	}
	else if (epl.second.has_value() && epr.second.has_value()) {
		m_pToken = std::move(epl.second.value() / epr.second.value());
		return std::make_pair(std::nullopt, *m_pToken);
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::DivNode::diff(const VariableToken& var)
{
	// We use a mul node to get out the product rule in quotient rule
	MulNode temp_mul(std::move(m_Children[0]), std::move(m_Children[1]));
	auto product = node_from_pair(temp_mul.diff(var));

	// Get back left child
	m_Children[0] = std::move(temp_mul.m_Children[0]);

	SquareNode temp_square(std::move(temp_mul.m_Children[1]));
	auto square = node_from_pair(temp_square.eval());

	// Get back right child
	m_Children[1] = std::move(temp_square.m_Children[0]);

	DivNode temp_quotient(std::move(product), std::move(square));

	return temp_quotient.eval();
}

// <================================== ADD ===================================>

tc::expression::AddNode::AddNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
{
	m_Children.resize(2);
	m_Children[0] = std::move(left_child);
	m_Children[1] = std::move(right_child);
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::AddNode::eval()
{
	auto epl = m_Children[0]->eval();
	auto epr = m_Children[1]->eval();

	if (epl.first.has_value() && epr.first.has_value()) {
		return std::make_pair(epl.first.value() + epr.first.value(), std::nullopt);
	}
	else if (epl.first.has_value() && epr.second.has_value()) {
		return std::make_pair(epl.first.value() + epr.second.value(), std::nullopt);
	}
	else if (epl.second.has_value() && epr.first.has_value()) {
		return std::make_pair(epl.second.value() + epr.first.value(), std::nullopt);
	}
	else if (epl.second.has_value() && epr.second.has_value()) {
		m_pToken = std::move(epl.second.value() + epr.second.value());
		return std::make_pair(std::nullopt, *m_pToken);
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::AddNode::diff(const VariableToken& var)
{
	auto ldiff = node_from_pair(m_Children[0]->diff(var));
	auto rdiff = node_from_pair(m_Children[1]->diff(var));

	AddNode add(std::move(ldiff), std::move(rdiff));
	return add.eval();
}

// <================================== SUB ===================================>

tc::expression::SubNode::SubNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
{
	m_Children.resize(2);
	m_Children[0] = std::move(left_child);
	m_Children[1] = std::move(right_child);
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::SubNode::eval()
{
	auto epl = m_Children[0]->eval();
	auto epr = m_Children[1]->eval();

	if (epl.first.has_value() && epr.first.has_value()) {
		return std::make_pair(epl.first.value() - epr.first.value(), std::nullopt);
	}
	else if (epl.first.has_value() && epr.second.has_value()) {
		return std::make_pair(epl.first.value() - epr.second.value(), std::nullopt);
	}
	else if (epl.second.has_value() && epr.first.has_value()) {
		return std::make_pair(epl.second.value() - epr.first.value(), std::nullopt);
	}
	else if (epl.second.has_value() && epr.second.has_value()) {
		m_pToken = std::move(epl.second.value() - epr.second.value());
		return std::make_pair(std::nullopt, *m_pToken);
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::SubNode::diff(const VariableToken& var)
{
	auto ldiff = node_from_pair(m_Children[0]->diff(var));
	auto rdiff = node_from_pair(m_Children[1]->diff(var));

	SubNode sub(std::move(ldiff), std::move(rdiff));
	return sub.eval();
}

// <================================== POW ===================================>

tc::expression::PowNode::PowNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
{
	m_Children.resize(2);
	m_Children[0] = std::move(left_child);
	m_Children[1] = std::move(right_child);
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::PowNode::eval()
{
	auto epl = m_Children[0]->eval();
	auto epr = m_Children[1]->eval();

	if (epl.first.has_value() && epr.first.has_value()) {
		return std::make_pair(pow(epl.first.value(), epr.first.value()), std::nullopt);
	}
	else if (epl.first.has_value() && epr.second.has_value()) {
		return std::make_pair(pow(epl.first.value(), epr.second.value()), std::nullopt);
	}
	else if (epl.second.has_value() && epr.first.has_value()) {
		return std::make_pair(pow(epl.second.value(), epr.first.value()), std::nullopt);
	}
	else if (epl.second.has_value() && epr.second.has_value()) {
		m_pToken = std::move(pow(epl.second.value(), epr.second.value()));
		return std::make_pair(std::nullopt, *m_pToken);
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::PowNode::diff(const VariableToken& var)
{
	// h = f^g
	// dh = h*(dg*log(f)+g*df/f)

	std::unique_ptr<Node> f = node_from_pair(m_Children[0]->eval());
	std::unique_ptr<Node> df = node_from_pair(m_Children[0]->diff(var));

	std::unique_ptr<Node> g = node_from_pair(m_Children[1]->eval());
	std::unique_ptr<Node> dg = node_from_pair(m_Children[1]->diff(var));

	std::unique_ptr<Node> h;
	{
		PowNode temp(std::move(f), std::move(g));
		h = node_from_pair(temp.eval());

		f = std::move(temp.m_Children[0]);
		g = std::move(temp.m_Children[1]);
	}


	std::unique_ptr<Node> df_logf_g_df_g;
	{
		std::unique_ptr<Node> dg_logf;
		{
			std::unique_ptr<Node> logf;
			{
				LogNode temp(std::move(f));
				logf = node_from_pair(temp.eval());
				f = std::move(temp.m_Children[0]);
			}

			MulNode temp(std::move(dg), std::move(logf));
			dg_logf = node_from_pair(temp.eval());
			dg = std::move(temp.m_Children[0]);
		}

		std::unique_ptr<Node> g_df_f;
		{
			std::unique_ptr<Node> g_df;
			{
				MulNode temp(std::move(g), std::move(df));
				g_df = node_from_pair(temp.eval());

				g = std::move(temp.m_Children[0]);
				df = std::move(temp.m_Children[1]);
			}

			DivNode temp(std::move(g_df), std::move(f));
			g_df_f = node_from_pair(temp.eval());

			f = std::move(temp.m_Children[1]);
		}

		AddNode temp(std::move(dg_logf), std::move(g_df_f));
		df_logf_g_df_g = node_from_pair(temp.eval());
	}

	MulNode temp(std::move(h), std::move(df_logf_g_df_g));
	return temp.eval();
}

// <================================== POW ===================================>

tc::expression::SquareNode::SquareNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::SquareNode::eval()
{
	auto ep = m_Children[0]->eval();

	if (ep.first.has_value()) {
		return std::make_pair(torch::square(ep.first.value()), std::nullopt);
	}
	else {
		m_pToken = std::move(square(ep.second.value().get()));
		return std::make_pair(std::nullopt, *m_pToken);
	}
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::SquareNode::diff(const VariableToken& var)
{
	return std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>>();
}

// <================================== LOG ===================================>

tc::expression::LogNode::LogNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::LogNode::eval()
{
	auto ep = m_Children[0]->eval();

	if (ep.first.has_value()) {
		return std::make_pair(1.0f / ep.first.value(), std::nullopt);
	}
	else {
		m_pToken = std::move(log(ep.second.value().get()));
		return std::make_pair(std::nullopt, *m_pToken);
	}
}

std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>> tc::expression::LogNode::diff(const VariableToken& var)
{
	return std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>>();
}
