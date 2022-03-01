#include "../pch.hpp"

#include "nodes.hpp"

std::string tc::expression::tentok_to_string(const tentok& in)
{
	if (in.first.has_value()) {
		std::stringstream stream;
		stream << in.first.value();
		return stream.str();
	}
	else if (in.second.has_value()) {
		auto& tok = *in.second.value();
		
		switch (tok.get_token_type()) {
		case TokenType::ZERO_TYPE:
		{
			const ZeroToken& ttok = static_cast<const ZeroToken&>(tok);
			return "ZERO";
		}
		case TokenType::UNITY_TYPE:
		{
			const UnityToken& ttok = static_cast<const UnityToken&>(tok);
			return "UNITY";
		}
		case TokenType::NEG_UNITY_TYPE:
		{
			const NegUnityToken& ttok = static_cast<const NegUnityToken&>(tok);
			return "NEG_UNITY";
		}
		case TokenType::NAN_TYPE:
		{
			const NanToken& ttok = static_cast<const NanToken&>(tok);
			return "NAN";
		}
		case TokenType::NUMBER_TYPE:
		{
			const NumberToken& ttok = static_cast<const NumberToken&>(tok);
			return "NUM: " + std::to_string(ttok.num.real()) + "+" + std::to_string(ttok.num.imag()) + "i";
		}
		default:
			throw std::runtime_error("Expected Zero, Unity, NegUnity, Nan and Number");
		}
	}
	return "tentok had no value";
}

tc::expression::tentok tc::expression::tentok_from_number(float a)
{
	return std::make_pair(std::nullopt, std::make_unique<NumberToken>(a, false));
}

tc::expression::tentok tc::expression::tentok_from_zero()
{
	return std::make_pair(std::nullopt, std::make_unique<ZeroToken>());
}

tc::expression::tentok tc::expression::tentok_from_unity()
{
	return std::make_pair(std::nullopt, std::make_unique<UnityToken>());
}

tc::expression::tentok tc::expression::tentok_from_negunity()
{
	return std::make_pair(std::nullopt, std::make_unique<NegUnityToken>());
}

tc::expression::tentok tc::expression::tentok_from_nan()
{
	return std::make_pair(std::nullopt, std::make_unique<NanToken>());
}





std::unique_ptr<tc::expression::NumberBaseToken> tc::expression::copy_token(const Token& tok)
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

torch::Tensor tc::expression::tensor_from_tentok(const tentok& in, torch::Device& device)
{
	torch::TensorOptions tops;
	tops = tops.device(device);

	if (in.first.has_value()) {
		return in.first.value();
	}
	else if (in.second.has_value()) {
		auto& tok = in.second.value();
		switch (tok->get_token_type()) {
		case TokenType::NUMBER_TYPE:
		{
			const NumberToken& numtok = static_cast<const NumberToken&>(*tok);
			if (numtok.is_imaginary)
				return torch::full(numtok.sizes, c10::complex<float>(numtok.num), tops);
			return torch::full(numtok.sizes, numtok.num.real(), tops);
		}
		case TokenType::UNITY_TYPE:
		{
			return torch::ones(tok->sizes, tops);
		}
		case TokenType::NEG_UNITY_TYPE:
		{
			return -torch::ones(tok->sizes, tops);
		}
		case TokenType::ZERO_TYPE:
		{
			return -torch::zeros(tok->sizes, tops);
		}
		case TokenType::NAN_TYPE:
			return torch::full(tok->sizes, std::numeric_limits<float>::quiet_NaN(), tops);
		}
	}
	throw std::runtime_error("tentok contained two nullopt");
}

// <================================== NODE ===================================>

tc::expression::Node::Node(std::unique_ptr<NumberBaseToken> base_token)
	: m_pToken(std::move(base_token))
{
}


std::unique_ptr<tc::expression::Node> tc::expression::node_from_token(const Token& tok)
{
	return std::make_unique<TokenNode>(tok);
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
	: Node(copy_token(tok))
{
}

tc::expression::tentok tc::expression::TokenNode::eval()
{
	return std::make_pair(std::nullopt, copy_token(*m_pToken));
}

tc::expression::tentok tc::expression::TokenNode::diff(const VariableToken& var)
{
	return std::make_pair(std::nullopt, std::make_unique<ZeroToken>(m_pToken->sizes)); // derivative of number is always zero
}

// <================================== TENSOR-NODE ===================================>

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
	auto sizes = m_VariableFetcher().sizes().vec();

	if (var.name == m_VarToken.name) {
		return std::make_pair(std::nullopt, std::make_unique<UnityToken>(sizes));
	}
	return std::make_pair(std::nullopt, std::make_unique<ZeroToken>(sizes));
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

	return (dl * r - l * dr) / square(r);
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

	return tentok_from_number(0.5f) * din / sqrt(in);
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
	return tentok_from_number(2.0f) * m_Children[0]->eval() * m_Children[0]->diff(var);
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

// <================================== TAN ===================================>

tc::expression::tentok tc::expression::tan(const tentok& a)
{
	if (a.first.has_value()) {
		return std::make_pair(torch::tan(a.first.value()), std::nullopt);
	}
	else if (a.second.has_value()) {
		return std::make_pair(std::nullopt, tan(*a.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::TanNode::TanNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

tc::expression::tentok tc::expression::TanNode::eval()
{
	return tan(m_Children[0]->eval());
}

tc::expression::tentok tc::expression::TanNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din / square(cos(in));
}

// <================================== ASIN ===================================>

tc::expression::tentok tc::expression::asin(const tentok& a)
{
	if (a.first.has_value()) {
		return std::make_pair(torch::asin(a.first.value()), std::nullopt);
	}
	else if (a.second.has_value()) {
		return std::make_pair(std::nullopt, asin(*a.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::AsinNode::AsinNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

tc::expression::tentok tc::expression::AsinNode::eval()
{
	return asin(m_Children[0]->eval());
}

tc::expression::tentok tc::expression::AsinNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din / sqrt(tentok_from_unity() - square(in));
}

// <================================== ACOS ===================================>

tc::expression::tentok tc::expression::acos(const tentok& a)
{
	if (a.first.has_value()) {
		return std::make_pair(torch::acos(a.first.value()), std::nullopt);
	}
	else if (a.second.has_value()) {
		return std::make_pair(std::nullopt, acos(*a.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::AcosNode::AcosNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

tc::expression::tentok tc::expression::AcosNode::eval()
{
	return acos(m_Children[0]->eval());
}

tc::expression::tentok tc::expression::AcosNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return -din / sqrt(tentok_from_unity() - square(in));
}

// <================================== ATAN ===================================>

tc::expression::tentok tc::expression::atan(const tentok& a)
{
	if (a.first.has_value()) {
		return std::make_pair(torch::atan(a.first.value()), std::nullopt);
	}
	else if (a.second.has_value()) {
		return std::make_pair(std::nullopt, atan(*a.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::AtanNode::AtanNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

tc::expression::tentok tc::expression::AtanNode::eval()
{
	return atan(m_Children[0]->eval());
}

tc::expression::tentok tc::expression::AtanNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din / (tentok_from_unity() + square(in));
}

// <================================== SINH ===================================>

tc::expression::tentok tc::expression::sinh(const tentok& a)
{
	if (a.first.has_value()) {
		return std::make_pair(torch::sinh(a.first.value()), std::nullopt);
	}
	else if (a.second.has_value()) {
		return std::make_pair(std::nullopt, sinh(*a.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::SinhNode::SinhNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

tc::expression::tentok tc::expression::SinhNode::eval()
{
	return asin(m_Children[0]->eval());
}

tc::expression::tentok tc::expression::SinhNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din * cosh(in);
}

// <================================== COSH ===================================>

tc::expression::tentok tc::expression::cosh(const tentok& a)
{
	if (a.first.has_value()) {
		return std::make_pair(torch::cosh(a.first.value()), std::nullopt);
	}
	else if (a.second.has_value()) {
		return std::make_pair(std::nullopt, cosh(*a.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::CoshNode::CoshNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

tc::expression::tentok tc::expression::CoshNode::eval()
{
	return cosh(m_Children[0]->eval());
}

tc::expression::tentok tc::expression::CoshNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din * sinh(in);
}

// <================================== TANH ===================================>

tc::expression::tentok tc::expression::tanh(const tentok& a)
{
	if (a.first.has_value()) {
		return std::make_pair(torch::tanh(a.first.value()), std::nullopt);
	}
	else if (a.second.has_value()) {
		return std::make_pair(std::nullopt, tanh(*a.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::TanhNode::TanhNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

tc::expression::tentok tc::expression::TanhNode::eval()
{
	return tanh(m_Children[0]->eval());
}

tc::expression::tentok tc::expression::TanhNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din / square(cosh(in));
}

// <================================== ASINH ===================================>

tc::expression::tentok tc::expression::asinh(const tentok& a)
{
	if (a.first.has_value()) {
		return std::make_pair(torch::asinh(a.first.value()), std::nullopt);
	}
	else if (a.second.has_value()) {
		return std::make_pair(std::nullopt, asinh(*a.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::AsinhNode::AsinhNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

tc::expression::tentok tc::expression::AsinhNode::eval()
{
	return asinh(m_Children[0]->eval());
}

tc::expression::tentok tc::expression::AsinhNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din / sqrt(square(in)+tentok_from_unity());
}

// <================================== ACOSH ===================================>

tc::expression::tentok tc::expression::acosh(const tentok& a)
{
	if (a.first.has_value()) {
		return std::make_pair(torch::acosh(a.first.value()), std::nullopt);
	}
	else if (a.second.has_value()) {
		return std::make_pair(std::nullopt, acosh(*a.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::AcoshNode::AcoshNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

tc::expression::tentok tc::expression::AcoshNode::eval()
{
	return acosh(m_Children[0]->eval());
}

tc::expression::tentok tc::expression::AcoshNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din / sqrt(square(in) - tentok_from_unity());
}

// <================================== ATANH ===================================>

tc::expression::tentok tc::expression::atanh(const tentok& a)
{
	if (a.first.has_value()) {
		return std::make_pair(torch::atanh(a.first.value()), std::nullopt);
	}
	else if (a.second.has_value()) {
		return std::make_pair(std::nullopt, atanh(*a.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::AtanhNode::AtanhNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

tc::expression::tentok tc::expression::AtanhNode::eval()
{
	return atanh(m_Children[0]->eval());
}

tc::expression::tentok tc::expression::AtanhNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din / (tentok_from_unity() - square(in));
}
