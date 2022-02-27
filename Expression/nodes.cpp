#include "../pch.hpp"

#include "nodes.hpp"

tc::expression::tentok tc::expression::from_number(float a)
{
	return std::make_pair(std::nullopt, std::make_unique<NumberToken>(a, false));
}

std::unique_ptr<tc::expression::Token> tc::expression::copy_token(const Token& tok)
{
	std::int32_t type = tok.get_token_type();

	switch (type) {
	case TokenType::ZERO_TYPE:
	{
		auto& t = static_cast<const ZeroToken&>(tok);
		return std::make_unique<ZeroToken>(t);
	}
	break;
	case TokenType::UNITY_TYPE:
	{
		auto& t = static_cast<const UnityToken&>(tok);
		return std::make_unique<UnityToken>(t);
	}
	break;
	case TokenType::NEG_UNITY_TYPE:
	{
		auto& t = static_cast<const NegUnityToken&>(tok);
		return std::make_unique<NegUnityToken>(t);
	}
	break;
	case TokenType::NAN_TYPE:
	{
		auto& t = static_cast<const NanToken&>(tok);
		return std::make_unique<NanToken>(t);
	}
	break;
	case TokenType::NUMBER_TYPE:
	{
		auto& t = static_cast<const NumberToken&>(tok);
		return std::make_unique<NumberToken>(t);
	}
	break;
	default:
		throw std::runtime_error("Can't construct TokenNode from token type other than Zero,Unity,NegUnity,Nan,Number");
	}
}

std::unique_ptr<tc::expression::Node> tc::expression::node_from_pair(const tentok& pair)
{
	if (pair.first.has_value()) {
		return std::make_unique<TensorNode>(pair.first.value());
	}
	else if (pair.second.has_value()) {
		return std::make_unique<TokenNode>(*pair.second.value());
	}
	throw std::runtime_error("node_from_pair() received a pair with only nullopt content");
}


// <================================== TOKEN ===================================>

tc::expression::TokenNode::TokenNode(const Token& tok)
{
	m_pToken = copy_token(tok);
}

tc::expression::tentok tc::expression::TokenNode::eval()
{
	return std::make_pair(std::nullopt, copy_token(*m_pToken));
}

tc::expression::tentok tc::expression::TokenNode::diff(const VariableToken& var)
{
	return std::make_pair(std::nullopt, std::make_unique<ZeroToken>(m_Sizes)); // derivative of number is always zero
}

// <================================== TOKEN ===================================>

tc::expression::TensorNode::TensorNode(const torch::Tensor& tensor)
	: m_Tensor(tensor)
{
}

tc::expression::tentok tc::expression::TensorNode::eval()
{
	return std::make_pair(m_Tensor, std::nullopt);
}

tc::expression::tentok tc::expression::TensorNode::diff(const VariableToken& var)
{
	auto sizes = m_Tensor.sizes();
	std::vector<int64_t> vecsizes(sizes.data(), sizes.data() + sizes.size());

	return std::make_pair(std::nullopt, std::make_unique<ZeroToken>(vecsizes));
}

// <================================== VARIABLE ===================================>

tc::expression::VariableNode::VariableNode(const VariableToken& token, const std::function<torch::Tensor()>& variable_fetcher)
	: m_VarToken(token), m_VariableFetcher(variable_fetcher)
{
}

tc::expression::tentok tc::expression::VariableNode::eval()
{
	return std::make_pair(m_VariableFetcher(), std::nullopt);
}

tc::expression::tentok tc::expression::VariableNode::diff(const VariableToken& var)
{
	auto sizes = m_VariableFetcher().sizes();
	std::vector<int64_t> vecsizes(sizes.data(), sizes.data() + sizes.size());

	if (var.name == m_VarToken.name) {
		return std::make_pair(std::nullopt, std::make_unique<UnityToken>(vecsizes));
	}
	return std::make_pair(std::nullopt, std::make_unique<ZeroToken>(vecsizes));
}

// <================================== NEG ===================================>

tc::expression::tentok tc::expression::operator-(const tentok& a)
{
	if (a.first.has_value()) {
		return std::make_pair(-a.first.value(), std::nullopt);
	}
	else if (a.second.has_value()) {
		return std::make_pair(std::nullopt, -(*a.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::NegNode::NegNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

tc::expression::tentok tc::expression::NegNode::eval()
{	
	return -m_Children[0]->eval();
}

tc::expression::tentok tc::expression::NegNode::diff(const VariableToken& var)
{
	return -m_Children[0]->diff(var);
}

// <================================== MUL ===================================>

tc::expression::tentok tc::expression::operator*(const tentok& a, const tentok& b)
{
	if (a.first.has_value() && b.first.has_value()) {
		return std::make_pair(a.first.value() * b.first.value(), std::nullopt);
	}
	else if (a.first.has_value() && b.second.has_value()) {
		return std::make_pair(a.first.value() * *b.second.value(), std::nullopt);
	}
	else if (a.second.has_value() && b.first.has_value()) {
		return std::make_pair(*a.second.value() * b.first.value(), std::nullopt);
	}
	else if (a.second.has_value() && b.second.has_value()) {
		return std::make_pair(std::nullopt, *a.second.value() * *b.second.value());
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::MulNode::MulNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
{
	m_Children.resize(2);
	m_Children[0] = std::move(left_child);
	m_Children[1] = std::move(right_child);
}

tc::expression::tentok tc::expression::MulNode::eval()
{
	auto l = m_Children[0]->eval();
	auto r = m_Children[1]->eval();

	return l * r;
}

tc::expression::tentok tc::expression::MulNode::diff(const VariableToken& var)
{
	auto l = m_Children[0]->eval();
	auto r = m_Children[1]->eval();

	auto dl = m_Children[0]->diff(var);
	auto dr = m_Children[1]->diff(var);

	return (dl * r + l * dr);
}

// <================================== DIV ===================================>

tc::expression::tentok tc::expression::operator/(const tentok& a, const tentok& b)
{
	if (a.first.has_value() && b.first.has_value()) {
		return std::make_pair(a.first.value() / b.first.value(), std::nullopt);
	}
	else if (a.first.has_value() && b.second.has_value()) {
		return std::make_pair(a.first.value() / *b.second.value(), std::nullopt);
	}
	else if (a.second.has_value() && b.first.has_value()) {
		return std::make_pair(*a.second.value() / b.first.value(), std::nullopt);
	}
	else if (a.second.has_value() && b.second.has_value()) {
		return std::make_pair(std::nullopt, *a.second.value() / *b.second.value());
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::DivNode::DivNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
{
	m_Children.resize(2);
	m_Children[0] = std::move(left_child);
	m_Children[1] = std::move(right_child);
}

tc::expression::tentok tc::expression::DivNode::eval()
{
	auto l = m_Children[0]->eval();
	auto r = m_Children[1]->eval();

	return l / r;
}

tc::expression::tentok tc::expression::DivNode::diff(const VariableToken& var)
{
	auto l = m_Children[0]->eval();
	auto dl = m_Children[0]->diff(var);

	auto r = m_Children[1]->eval();
	auto dr = m_Children[1]->diff(var);

	return (dl * r + l * dr) / square(r);
}

// <================================== ADD ===================================>

tc::expression::tentok tc::expression::operator+(const tentok& a, const tentok& b)
{
	if (a.first.has_value() && b.first.has_value()) {
		return std::make_pair(a.first.value() + b.first.value(), std::nullopt);
	}
	else if (a.first.has_value() && b.second.has_value()) {
		return std::make_pair(a.first.value() + *b.second.value(), std::nullopt);
	}
	else if (a.second.has_value() && b.first.has_value()) {
		return std::make_pair(*a.second.value() + b.first.value(), std::nullopt);
	}
	else if (a.second.has_value() && b.second.has_value()) {
		return std::make_pair(std::nullopt, *a.second.value() + *b.second.value());
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::AddNode::AddNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
{
	m_Children.resize(2);
	m_Children[0] = std::move(left_child);
	m_Children[1] = std::move(right_child);
}

tc::expression::tentok tc::expression::AddNode::eval()
{
	auto l = m_Children[0]->eval();
	auto r = m_Children[1]->eval();

	return l + r;
}

tc::expression::tentok tc::expression::AddNode::diff(const VariableToken& var)
{
	auto l = m_Children[0]->diff(var);
	auto r = m_Children[1]->diff(var);

	return l + r;
}

// <================================== SUB ===================================>

tc::expression::tentok tc::expression::operator-(const tentok& a, const tentok& b)
{
	if (a.first.has_value() && b.first.has_value()) {
		return std::make_pair(a.first.value() - b.first.value(), std::nullopt);
	}
	else if (a.first.has_value() && b.second.has_value()) {
		return std::make_pair(a.first.value() - *b.second.value(), std::nullopt);
	}
	else if (a.second.has_value() && b.first.has_value()) {
		return std::make_pair(*a.second.value() - b.first.value(), std::nullopt);
	}
	else if (a.second.has_value() && b.second.has_value()) {
		return std::make_pair(std::nullopt, *a.second.value() - *b.second.value());
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::SubNode::SubNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
{
	m_Children.resize(2);
	m_Children[0] = std::move(left_child);
	m_Children[1] = std::move(right_child);
}

tc::expression::tentok tc::expression::SubNode::eval()
{
	auto l = m_Children[0]->eval();
	auto r = m_Children[1]->eval();

	return l - r;
}

tc::expression::tentok tc::expression::SubNode::diff(const VariableToken& var)
{
	auto l = m_Children[0]->diff(var);
	auto r = m_Children[1]->diff(var);

	return l - r;
}

// <================================== POW ===================================>

tc::expression::tentok tc::expression::pow(const tentok& a, const tentok& b)
{
	if (a.first.has_value() && b.first.has_value()) {
		return std::make_pair(pow(a.first.value(), b.first.value()), std::nullopt);
	}
	else if (a.first.has_value() && b.second.has_value()) {
		return std::make_pair(pow(a.first.value(), *b.second.value()), std::nullopt);
	}
	else if (a.second.has_value() && b.first.has_value()) {
		return std::make_pair(pow(*a.second.value(), b.first.value()), std::nullopt);
	}
	else if (a.second.has_value() && b.second.has_value()) {
		return std::make_pair(std::nullopt, pow(*a.second.value(), *b.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::PowNode::PowNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child)
{
	m_Children.resize(2);
	m_Children[0] = std::move(left_child);
	m_Children[1] = std::move(right_child);
}

tc::expression::tentok tc::expression::PowNode::eval()
{
	auto l = m_Children[0]->eval();
	auto r = m_Children[1]->eval();

	return pow(l,r);
}

tc::expression::tentok tc::expression::PowNode::diff(const VariableToken& var)
{
	// h = l^r
	// dh = h*(dr*log(l)+r*dl/l)

	auto l = m_Children[0]->eval();
	auto dl = m_Children[0]->diff(var);

	auto r = m_Children[1]->eval();
	auto dr = m_Children[1]->diff(var);
	
	return pow(l, r) * (dr * log(l) + r * dl / l);
}

// <================================== ABS ===================================>

tc::expression::tentok tc::expression::abs(const tentok& a)
{
	if (a.first.has_value()) {
		return std::make_pair(torch::abs(a.first.value()), std::nullopt);
	}
	else if (a.second.has_value()) {
		return std::make_pair(std::nullopt, abs(*a.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::AbsNode::AbsNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

tc::expression::tentok tc::expression::AbsNode::eval()
{
	return abs(m_Children[0]->eval());
}

tc::expression::tentok tc::expression::AbsNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din * in / abs(in);
}

// <================================== SQRT ===================================>

tc::expression::tentok tc::expression::sqrt(const tentok& a)
{
	if (a.first.has_value()) {
		return std::make_pair(torch::sqrt(a.first.value()), std::nullopt);
	}
	else if (a.second.has_value()) {
		return std::make_pair(std::nullopt, sqrt(*a.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::SqrtNode::SqrtNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

tc::expression::tentok tc::expression::SqrtNode::eval()
{
	return sqrt(m_Children[0]->eval());
}

tc::expression::tentok tc::expression::SqrtNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return from_number(0.5f) * din / sqrt(in);
}

// <================================== SQUARE ===================================>

tc::expression::tentok tc::expression::square(const tentok& a)
{
	if (a.first.has_value()) {
		return std::make_pair(torch::square(a.first.value()), std::nullopt);
	}
	else if (a.second.has_value()) {
		return std::make_pair(std::nullopt, square(*a.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}


tc::expression::SquareNode::SquareNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

tc::expression::tentok tc::expression::SquareNode::eval()
{
	return square(m_Children[0]->eval());
}

tc::expression::tentok tc::expression::SquareNode::diff(const VariableToken& var)
{
	return from_number(2.0f) * m_Children[0]->eval() * m_Children[0]->diff(var);
}

// <================================== EXP ===================================>

tc::expression::tentok tc::expression::exp(const tentok& a)
{
	if (a.first.has_value()) {
		return std::make_pair(torch::exp(a.first.value()), std::nullopt);
	}
	else if (a.second.has_value()) {
		return std::make_pair(std::nullopt, exp(*a.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::ExpNode::ExpNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

tc::expression::tentok tc::expression::ExpNode::eval()
{
	return exp(m_Children[0]->eval());
}

tc::expression::tentok tc::expression::ExpNode::diff(const VariableToken& var)
{
	return exp(m_Children[0]->eval()) * m_Children[0]->diff(var);
}

// <================================== LOG ===================================>

tc::expression::tentok tc::expression::log(const tentok& a)
{
	if (a.first.has_value()) {
		return std::make_pair(torch::log(a.first.value()), std::nullopt);
	}
	else if (a.second.has_value()) {
		return std::make_pair(std::nullopt, log(*a.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::LogNode::LogNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

tc::expression::tentok tc::expression::LogNode::eval()
{
	return log(m_Children[0]->eval());
}

tc::expression::tentok tc::expression::LogNode::diff(const VariableToken& var)
{
	return m_Children[0]->diff(var) / m_Children[0]->eval();
}

// <================================== SIN ===================================>

tc::expression::tentok tc::expression::sin(const tentok& a)
{
	if (a.first.has_value()) {
		return std::make_pair(torch::sin(a.first.value()), std::nullopt);
	}
	else if (a.second.has_value()) {
		return std::make_pair(std::nullopt, sin(*a.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::SinNode::SinNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

tc::expression::tentok tc::expression::SinNode::eval()
{
	return sin(m_Children[0]->eval());
}

tc::expression::tentok tc::expression::SinNode::diff(const VariableToken& var)
{
	return m_Children[0]->diff(var) * cos(m_Children[0]->eval());
}

// <================================== COS ===================================>

tc::expression::tentok tc::expression::cos(const tentok& a)
{
	if (a.first.has_value()) {
		return std::make_pair(torch::cos(a.first.value()), std::nullopt);
	}
	else if (a.second.has_value()) {
		return std::make_pair(std::nullopt, cos(*a.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::CosNode::CosNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

tc::expression::tentok tc::expression::CosNode::eval()
{
	return cos(m_Children[0]->eval());
}

tc::expression::tentok tc::expression::CosNode::diff(const VariableToken& var)
{
	return -m_Children[0]->diff(var) * sin(m_Children[0]->eval());
}


