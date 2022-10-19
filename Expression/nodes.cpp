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

std::unique_ptr<tc::expression::Node> tc::expression::TokenNode::evalnode()
{
	return std::make_unique<TokenNode>(*m_pToken);
}

tc::expression::tentok tc::expression::TokenNode::diff(const VariableToken& var)
{
	return std::make_pair(std::nullopt, std::make_unique<ZeroToken>(m_pToken->sizes)); // derivative of number is always zero
}

std::unique_ptr<tc::expression::Node> tc::expression::TokenNode::diffnode(const VariableToken& var)
{
	return std::make_unique<TokenNode>(ZeroToken(m_pToken->sizes));
}

// <================================== TOKEN-FETCHER ===================================>

tc::expression::TokenFetcherNode::TokenFetcherNode(const Token& tok, const FetcherFuncRef& fetcher)
	: m_VariableFetcher(fetcher), Node(copy_token(tok))
{
}

tc::expression::tentok tc::expression::TokenFetcherNode::eval()
{
	m_pToken->sizes = m_VariableFetcher().sizes().vec();
	return std::make_pair(std::nullopt, copy_token(*m_pToken));
}

std::unique_ptr<tc::expression::Node> tc::expression::TokenFetcherNode::evalnode()
{
	return std::make_unique<TokenFetcherNode>(*m_pToken, m_VariableFetcher);
}

tc::expression::tentok tc::expression::TokenFetcherNode::diff(const VariableToken& var)
{
	m_pToken->sizes = m_VariableFetcher().sizes().vec();
	return std::make_pair(std::nullopt, std::make_unique<ZeroToken>(m_pToken->sizes)); // derivative of number is always zero
}

std::unique_ptr<tc::expression::Node> tc::expression::TokenFetcherNode::diffnode(const VariableToken& var)
{
	return std::make_unique<TokenFetcherNode>(ZeroToken(), m_VariableFetcher);
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

std::unique_ptr<tc::expression::Node> tc::expression::TensorNode::evalnode()
{
	return std::make_unique<TensorNode>(m_Tensor);
}

tc::expression::tentok tc::expression::TensorNode::diff(const VariableToken& var)
{
	return std::make_pair(std::nullopt, std::make_unique<ZeroToken>(m_Tensor.sizes().vec()));
}

std::unique_ptr<tc::expression::Node> tc::expression::TensorNode::diffnode(const VariableToken& var)
{
	return std::make_unique<TokenNode>(ZeroToken(m_Tensor.sizes().vec()));
}

// <================================== VARIABLE ===================================>

tc::expression::VariableNode::VariableNode(const VariableToken& token, FetcherFuncRef variable_fetcher)
	: m_VarToken(token), m_VariableFetcher(variable_fetcher)
{
}

tc::expression::tentok tc::expression::VariableNode::eval()
{
	return std::make_pair(m_VariableFetcher(), std::nullopt);
}

std::unique_ptr<tc::expression::Node> tc::expression::VariableNode::evalnode()
{
	return std::make_unique<VariableNode>(m_VarToken, m_VariableFetcher);
}

tc::expression::tentok tc::expression::VariableNode::diff(const VariableToken& var)
{
	auto sizes = m_VariableFetcher.get()().sizes().vec();

	if (var.name == m_VarToken.name) {
		return std::make_pair(std::nullopt, std::make_unique<UnityToken>(sizes));
	}
	return std::make_pair(std::nullopt, std::make_unique<ZeroToken>(sizes));
}

std::unique_ptr<tc::expression::Node> tc::expression::VariableNode::diffnode(const VariableToken& var)
{
	if (var.name == m_VarToken.name) {
		return std::make_unique<TokenFetcherNode>(UnityToken(), m_VariableFetcher);
	}
	return std::make_unique<TokenFetcherNode>(ZeroToken(), m_VariableFetcher);
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

std::unique_ptr<tc::expression::Node> tc::expression::NegNode::evalnode()
{
	return std::make_unique<NegNode>(m_Children[0]->evalnode());
}

tc::expression::tentok tc::expression::NegNode::diff(const VariableToken& var)
{
	return -m_Children[0]->diff(var);
}

std::unique_ptr<tc::expression::Node> tc::expression::NegNode::diffnode(const VariableToken& var)
{
	return std::make_unique<NegNode>(m_Children[0]->diffnode(var));
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
	m_Children.at(0) = std::move(left_child);
	m_Children.at(1) = std::move(right_child);
}

tc::expression::tentok tc::expression::MulNode::eval()
{
	auto l = m_Children[0]->eval();
	auto r = m_Children[1]->eval();

	return l * r;
}

std::unique_ptr<tc::expression::Node> tc::expression::MulNode::evalnode()
{
	auto l = m_Children[0]->evalnode();
	auto r = m_Children[1]->evalnode();

	return std::make_unique<MulNode>(std::move(l), std::move(r));
}

tc::expression::tentok tc::expression::MulNode::diff(const VariableToken& var)
{
	auto l = m_Children[0]->eval();
	auto r = m_Children[1]->eval();

	auto dl = m_Children[0]->diff(var);
	auto dr = m_Children[1]->diff(var);

	return (dl * r + l * dr);
}

std::unique_ptr<tc::expression::Node> tc::expression::MulNode::diffnode(const VariableToken& var)
{
	auto l = m_Children[0]->evalnode();
	auto r = m_Children[1]->evalnode();

	auto dl = m_Children[0]->diffnode(var);
	auto dr = m_Children[1]->diffnode(var);

	auto dlr = std::make_unique<MulNode>(std::move(dl), std::move(r));
	auto ldr = std::make_unique<MulNode>(std::move(l), std::move(dr));

	return std::make_unique<AddNode>(std::move(dlr), std::move(ldr));
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

std::unique_ptr<tc::expression::Node> tc::expression::DivNode::evalnode()
{
	auto l = m_Children[0]->evalnode();
	auto r = m_Children[1]->evalnode();

	return std::make_unique<DivNode>(std::move(l), std::move(r));
}

tc::expression::tentok tc::expression::DivNode::diff(const VariableToken& var)
{
	auto l = m_Children[0]->eval();
	auto dl = m_Children[0]->diff(var);

	auto r = m_Children[1]->eval();
	auto dr = m_Children[1]->diff(var);

	return (dl * r - l * dr) / square(r);
}

std::unique_ptr<tc::expression::Node> tc::expression::DivNode::diffnode(const VariableToken& var)
{
	auto l = m_Children[0]->evalnode();
	auto dl = m_Children[0]->diffnode(var);

	auto r = m_Children[1]->evalnode();
	auto r2 = m_Children[1]->evalnode();
	auto dr = m_Children[1]->diffnode(var);

	auto dlr = std::make_unique<MulNode>(std::move(dl), std::move(r));
	auto ldr = std::make_unique<MulNode>(std::move(l), std::move(dr));
	auto dife = std::make_unique<SubNode>(std::move(dlr), std::move(ldr));

	return std::make_unique<DivNode>(std::move(dife), std::make_unique<SquareNode>(std::move(r2)));
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

std::unique_ptr<tc::expression::Node> tc::expression::AddNode::evalnode()
{
	return std::make_unique<AddNode>(m_Children[0]->evalnode(), m_Children[1]->evalnode());
}

tc::expression::tentok tc::expression::AddNode::diff(const VariableToken& var)
{
	auto l = m_Children[0]->diff(var);
	auto r = m_Children[1]->diff(var);

	return l + r;
}

std::unique_ptr<tc::expression::Node> tc::expression::AddNode::diffnode(const VariableToken& var)
{
	return std::make_unique<AddNode>(m_Children[0]->diffnode(var), m_Children[1]->diffnode(var));
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

std::unique_ptr<tc::expression::Node> tc::expression::SubNode::evalnode()
{
	return std::make_unique<SubNode>(m_Children[0]->evalnode(), m_Children[1]->evalnode());
}

tc::expression::tentok tc::expression::SubNode::diff(const VariableToken& var)
{
	auto l = m_Children[0]->diff(var);
	auto r = m_Children[1]->diff(var);

	return l - r;
}

std::unique_ptr<tc::expression::Node> tc::expression::SubNode::diffnode(const VariableToken& var)
{
	return std::make_unique<SubNode>(m_Children[0]->diffnode(var), m_Children[1]->diffnode(var));
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

std::unique_ptr<tc::expression::Node> tc::expression::PowNode::evalnode()
{
	return std::make_unique<PowNode>(m_Children[0]->evalnode(), m_Children[1]->evalnode());
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

std::unique_ptr<tc::expression::Node> tc::expression::PowNode::diffnode(const VariableToken& var)
{
	auto l1 = m_Children[0]->evalnode();
	auto l2 = m_Children[0]->evalnode();
	auto l3 = m_Children[0]->evalnode();

	auto dl = m_Children[0]->diffnode(var);

	auto r1 = m_Children[1]->evalnode();
	auto r2 = m_Children[1]->evalnode();
	auto dr = m_Children[1]->diffnode(var);

	auto pow = std::make_unique<PowNode>(std::move(l1), std::move(r1));
	auto left = std::make_unique<MulNode>(std::move(dr), std::make_unique<LogNode>(std::move(l2)));
	auto right = std::make_unique<MulNode>(std::move(r2), std::make_unique<DivNode>(std::move(dl), std::move(l3)));
	auto add = std::make_unique<AddNode>(std::move(left), std::move(right));
	return std::make_unique<MulNode>(std::move(pow), std::move(add));
}


// <================================== SIGN ===================================>

tc::expression::tentok tc::expression::sgn(const tentok& a)
{
	if (a.first.has_value()) {
		return std::make_pair(torch::sgn(a.first.value()), std::nullopt);
	}
	else if (a.second.has_value()) {
		return std::make_pair(std::nullopt, abs(*a.second.value()));
	}
	else {
		throw std::runtime_error("more than two eval() optionals was nullopt");
	}
}

tc::expression::SgnNode::SgnNode(std::unique_ptr<Node> child)
{
	m_Children.push_back(std::move(child));
}

tc::expression::tentok tc::expression::SgnNode::eval()
{
	return sgn(m_Children[0]->eval());
}

std::unique_ptr<tc::expression::Node> tc::expression::SgnNode::evalnode()
{
	return std::make_unique<SgnNode>(m_Children[0]->evalnode());
}

tc::expression::tentok tc::expression::SgnNode::diff(const VariableToken& var)
{
	return tentok_from_zero() / m_Children[0]->eval();
}

std::unique_ptr<tc::expression::Node> tc::expression::SgnNode::diffnode(const VariableToken& var)
{
	return std::make_unique<tc::expression::DivNode>(std::make_unique<TokenNode>(ZeroToken()), evalnode());
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

std::unique_ptr<tc::expression::Node> tc::expression::AbsNode::evalnode()
{
	return std::make_unique<AbsNode>(m_Children[0]->evalnode());
}

tc::expression::tentok tc::expression::AbsNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din * sgn(in);
}

std::unique_ptr<tc::expression::Node> tc::expression::AbsNode::diffnode(const VariableToken& var)
{
	return std::make_unique<MulNode>(m_Children[0]->diffnode(var),
		std::make_unique<SgnNode>(m_Children[0]->evalnode()));
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

std::unique_ptr<tc::expression::Node> tc::expression::SqrtNode::evalnode()
{
	return std::make_unique<SqrtNode>(m_Children[0]->evalnode());
}

tc::expression::tentok tc::expression::SqrtNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return tentok_from_number(0.5f) * din / sqrt(in);
}

std::unique_ptr<tc::expression::Node> tc::expression::SqrtNode::diffnode(const VariableToken& var)
{
	auto mul = std::make_unique<MulNode>(std::make_unique<TokenNode>(from_number(0.5f)), m_Children[0]->diffnode(var));
	return std::make_unique<DivNode>(std::move(mul), m_Children[0]->evalnode());
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

std::unique_ptr<tc::expression::Node> tc::expression::SquareNode::evalnode()
{
	return std::make_unique<SquareNode>(m_Children[0]->evalnode());
}

tc::expression::tentok tc::expression::SquareNode::diff(const VariableToken& var)
{
	return tentok_from_number(2.0f) * m_Children[0]->eval() * m_Children[0]->diff(var);
}

std::unique_ptr<tc::expression::Node> tc::expression::SquareNode::diffnode(const VariableToken& var)
{
	auto two = std::make_unique<TokenNode>(from_number(2.0f));
	auto two_child = std::make_unique<MulNode>(std::move(two), std::move(m_Children[0]->evalnode()));

	return std::make_unique<MulNode>(std::move(two_child), std::move(m_Children[0]->diffnode(var)));
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

std::unique_ptr<tc::expression::Node> tc::expression::ExpNode::evalnode()
{
	return std::make_unique<ExpNode>(m_Children[0]->evalnode());
}

tc::expression::tentok tc::expression::ExpNode::diff(const VariableToken& var)
{
	return exp(m_Children[0]->eval()) * m_Children[0]->diff(var);
}

std::unique_ptr<tc::expression::Node> tc::expression::ExpNode::diffnode(const VariableToken& var)
{
	auto exp = std::make_unique<ExpNode>(m_Children[0]->evalnode());
	return std::make_unique<MulNode>(std::move(exp), m_Children[0]->diffnode(var));
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

std::unique_ptr<tc::expression::Node> tc::expression::LogNode::evalnode()
{
	return std::make_unique<LogNode>(m_Children[0]->evalnode());
}

tc::expression::tentok tc::expression::LogNode::diff(const VariableToken& var)
{
	return m_Children[0]->diff(var) / m_Children[0]->eval();
}

std::unique_ptr<tc::expression::Node> tc::expression::LogNode::diffnode(const VariableToken& var)
{
	return std::make_unique<DivNode>(m_Children[0]->evalnode(), m_Children[0]->diffnode(var));
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

std::unique_ptr<tc::expression::Node> tc::expression::SinNode::evalnode()
{
	return std::make_unique<SinNode>(m_Children[0]->evalnode());
}

tc::expression::tentok tc::expression::SinNode::diff(const VariableToken& var)
{
	return m_Children[0]->diff(var) * cos(m_Children[0]->eval());
}

std::unique_ptr<tc::expression::Node> tc::expression::SinNode::diffnode(const VariableToken& var)
{
	return std::make_unique<MulNode>(m_Children[0]->diffnode(var), std::make_unique<CosNode>(m_Children[0]->evalnode()));
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

std::unique_ptr<tc::expression::Node> tc::expression::CosNode::evalnode()
{
	return std::make_unique<CosNode>(m_Children[0]->evalnode());
}

tc::expression::tentok tc::expression::CosNode::diff(const VariableToken& var)
{
	return -m_Children[0]->diff(var) * sin(m_Children[0]->eval());
}

std::unique_ptr<tc::expression::Node> tc::expression::CosNode::diffnode(const VariableToken& var)
{
	auto l = std::make_unique<NegNode>(m_Children[0]->diffnode(var));
	auto sinc = std::make_unique<SinNode>(m_Children[0]->evalnode());
	return std::make_unique<MulNode>(std::move(l), std::move(sinc));
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

std::unique_ptr<tc::expression::Node> tc::expression::TanNode::evalnode()
{
	return std::make_unique<TanNode>(m_Children[0]->evalnode());
}

tc::expression::tentok tc::expression::TanNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din / square(cos(in));
}

std::unique_ptr<tc::expression::Node> tc::expression::TanNode::diffnode(const VariableToken& var)
{
	auto cin = std::make_unique<CosNode>(m_Children[0]->evalnode());
	auto scin = std::make_unique<SquareNode>(std::move(cin));
	return std::make_unique<DivNode>(m_Children[0]->diffnode(var), std::move(scin));
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

std::unique_ptr<tc::expression::Node> tc::expression::AsinNode::evalnode()
{
	return std::make_unique<AsinNode>(m_Children[0]->evalnode());
}

tc::expression::tentok tc::expression::AsinNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din / sqrt(tentok_from_unity() - square(in));
}

std::unique_ptr<tc::expression::Node> tc::expression::AsinNode::diffnode(const VariableToken& var)
{
	auto unity = std::make_unique<TokenNode>(UnityToken());
	auto square = std::make_unique<SquareNode>(m_Children[0]->evalnode());
	auto dife = std::make_unique<SubNode>(std::move(unity), std::move(square));
	auto sqrt = std::make_unique<SqrtNode>(std::move(dife));

	return std::make_unique<DivNode>(m_Children[0]->diffnode(var), std::move(sqrt));
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

std::unique_ptr<tc::expression::Node> tc::expression::AcosNode::evalnode()
{
	return std::make_unique<AcosNode>(m_Children[0]->evalnode());
}

tc::expression::tentok tc::expression::AcosNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return -din / sqrt(tentok_from_unity() - square(in));
}

std::unique_ptr<tc::expression::Node> tc::expression::AcosNode::diffnode(const VariableToken& var)
{
	auto acos = evalnode();
	return std::make_unique<NegNode>(std::move(acos));
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

std::unique_ptr<tc::expression::Node> tc::expression::AtanNode::evalnode()
{
	return std::make_unique<AtanNode>(m_Children[0]->evalnode());
}

tc::expression::tentok tc::expression::AtanNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din / (tentok_from_unity() + square(in));
}

std::unique_ptr<tc::expression::Node> tc::expression::AtanNode::diffnode(const VariableToken& var)
{
	auto unity = std::make_unique<TokenNode>(UnityToken());
	auto square = std::make_unique<SquareNode>(m_Children[0]->evalnode());
	auto dife = std::make_unique<AddNode>(std::move(unity), std::move(square));

	return std::make_unique<DivNode>(std::move(m_Children[0]->diffnode(var)), std::move(dife));
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

std::unique_ptr<tc::expression::Node> tc::expression::SinhNode::evalnode()
{
	return std::make_unique<SinhNode>(std::move(m_Children[0]->evalnode()));
}

tc::expression::tentok tc::expression::SinhNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din * cosh(in);
}

std::unique_ptr<tc::expression::Node> tc::expression::SinhNode::diffnode(const VariableToken& var)
{
	return std::make_unique<MulNode>(std::move(m_Children[0]->diffnode(var)),
		std::make_unique<CoshNode>(std::move(m_Children[0]->evalnode())));
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

std::unique_ptr<tc::expression::Node> tc::expression::CoshNode::evalnode()
{
	return std::make_unique<CoshNode>(std::move(m_Children[0]->evalnode()));
}

tc::expression::tentok tc::expression::CoshNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din * sinh(in);
}

std::unique_ptr<tc::expression::Node> tc::expression::CoshNode::diffnode(const VariableToken& var)
{
	return std::make_unique<MulNode>(std::move(m_Children[0]->diffnode(var)),
		std::make_unique<SinhNode>(std::move(m_Children[0]->evalnode())));
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

std::unique_ptr<tc::expression::Node> tc::expression::TanhNode::evalnode()
{
	return std::make_unique<TanhNode>(std::move(m_Children[0]->evalnode()));
}

tc::expression::tentok tc::expression::TanhNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din / square(cosh(in));
}

std::unique_ptr<tc::expression::Node> tc::expression::TanhNode::diffnode(const VariableToken& var)
{
	auto square = std::make_unique<SquareNode>(std::make_unique<CoshNode>(std::move(m_Children[0]->evalnode())));
	return std::make_unique<DivNode>(std::move(m_Children[0]->diffnode(var)), std::move(square));
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

std::unique_ptr<tc::expression::Node> tc::expression::AsinhNode::evalnode()
{
	return std::make_unique<AsinhNode>(std::move(m_Children[0]->evalnode()));
}

tc::expression::tentok tc::expression::AsinhNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din / sqrt(square(in)+tentok_from_unity());
}

std::unique_ptr<tc::expression::Node> tc::expression::AsinhNode::diffnode(const VariableToken& var)
{
	auto square = std::make_unique<SquareNode>(m_Children[0]->evalnode());
	auto add = std::make_unique<AddNode>(std::move(square), std::make_unique<TokenNode>(UnityToken()));
	auto sqrt = std::make_unique<SqrtNode>(std::move(add));
	return std::make_unique<DivNode>(std::move(m_Children[0]->diffnode(var)), std::move(sqrt));
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

std::unique_ptr<tc::expression::Node> tc::expression::AcoshNode::evalnode()
{
	return std::make_unique<AcoshNode>(std::move(m_Children[0]->evalnode()));
}

tc::expression::tentok tc::expression::AcoshNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din / sqrt(square(in) - tentok_from_unity());
}

std::unique_ptr<tc::expression::Node> tc::expression::AcoshNode::diffnode(const VariableToken& var)
{
	auto square = std::make_unique<SquareNode>(m_Children[0]->evalnode());
	auto sub = std::make_unique<SubNode>(std::move(square), std::make_unique<TokenNode>(UnityToken()));
	auto sqrt = std::make_unique<SqrtNode>(std::move(sub));
	return std::make_unique<DivNode>(m_Children[0]->diffnode(var), std::move(sqrt));
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

std::unique_ptr<tc::expression::Node> tc::expression::AtanhNode::evalnode()
{
	return std::make_unique<AtanhNode>(m_Children[0]->evalnode());
}

tc::expression::tentok tc::expression::AtanhNode::diff(const VariableToken& var)
{
	auto in = m_Children[0]->eval();
	auto din = m_Children[0]->diff(var);

	return din / (tentok_from_unity() - square(in));
}

std::unique_ptr<tc::expression::Node> tc::expression::AtanhNode::diffnode(const VariableToken& var)
{
	auto square = std::make_unique<SquareNode>(m_Children[0]->evalnode());
	auto sub = std::make_unique<SubNode>(std::make_unique<TokenNode>(UnityToken()), std::move(square));
	return std::make_unique<DivNode>(m_Children[0]->diffnode(var), std::move(sub));
}
