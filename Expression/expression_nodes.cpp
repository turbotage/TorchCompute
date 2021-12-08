#include "expression_nodes.hpp"


std::string expression::Node::getNodeName()
{
	return m_NodeStr;
}

// Makes expression printable
std::string expression::readableNode(expression::Node* node, ui32 nTabs) {
	std::string ret;
	ret += node->getNodeName();
	VariableNode* vnode = dynamic_cast<VariableNode*>(node);
	if (vnode) {
		ret += " : " + vnode->getVarName();
	}
	NumberNode* nnode = dynamic_cast<NumberNode*>(node);
	if (nnode) {
		ret += ": " + nnode->getNumberName();
	}
	ret += "\n";
	for (auto& child : node->m_Children) {
		for (int i = 0; i < nTabs; ++i) {
			ret += "\t";
		}
		ret += readableNode(child.get(), nTabs + 1);
	}
	return ret;
}


expression::NumberNode::NumberNode(std::string& num_name, torch::Tensor& val)
	: m_NumberName(num_name), m_Value(val)
{
	this->m_NodeStr = "num";
}
std::function<torch::Tensor()> expression::NumberNode::runner()
{
	return [this]() { 
		return this->m_Value; 
	};
}

std::string expression::NumberNode::getNumberName()
{
	return m_NumberName;
}

expression::VariableNode::VariableNode(std::string& var_name, const std::function<torch::Tensor()>& var_fetcher)
	: m_VariableName(var_name), m_VariableFetcher(var_fetcher)
{
	this->m_NodeStr = "var";
}

std::function<torch::Tensor()> expression::VariableNode::runner()
{
	return [this]() { 
		return this->m_VariableFetcher();
	};
}

std::string expression::VariableNode::getVarName()
{
	return m_VariableName;
}

expression::AddNode::AddNode(std::unique_ptr<Node> left, std::unique_ptr<Node> right)
{
	this->m_NodeStr = "op:add";
	this->m_Children.push_back(std::move(left));
	this->m_Children.push_back(std::move(right));
}

std::function<torch::Tensor()> expression::AddNode::runner()
{
	m_Left = this->m_Children[0]->runner();
	m_Right = this->m_Children[1]->runner();

	return [this] {
		return this->m_Left() + this->m_Right();
	};
}

expression::SubNode::SubNode(std::unique_ptr<Node> left, std::unique_ptr<Node> right)
{
	this->m_NodeStr = "op:sub";
	this->m_Children.push_back(std::move(left));
	this->m_Children.push_back(std::move(right));
}

std::function<torch::Tensor()> expression::SubNode::runner()
{
	m_Left = this->m_Children[0]->runner();
	m_Right = this->m_Children[1]->runner();

	return [this] {
		return m_Left() - m_Right();
	};
}

expression::MulNode::MulNode(std::unique_ptr<Node> left, std::unique_ptr<Node> right)
{
	this->m_NodeStr = "op:mul";
	this->m_Children.push_back(std::move(left));
	this->m_Children.push_back(std::move(right));
}

std::function<torch::Tensor()> expression::MulNode::runner()
{
	m_Left = this->m_Children[0]->runner();
	m_Right = this->m_Children[1]->runner();

	return [this] {
		return m_Left() * m_Right();
	};
}

expression::DivNode::DivNode(std::unique_ptr<Node> left, std::unique_ptr<Node> right)
{
	this->m_NodeStr = "op:div";
	this->m_Children.push_back(std::move(left));
	this->m_Children.push_back(std::move(right));
}

std::function<torch::Tensor()> expression::DivNode::runner()
{
	m_Left = this->m_Children[0]->runner();
	m_Right = this->m_Children[1]->runner();

	return [this] {
		return m_Left() / m_Right();
	};
}

expression::SinNode::SinNode(std::unique_ptr<Node> input)
{
	this->m_NodeStr = "func:sin";
	this->m_Children.push_back(std::move(input));
}

std::function<torch::Tensor()> expression::SinNode::runner()
{
	m_Input = this->m_Children[0]->runner();

	return [this] {
		return torch::sin(m_Input());
	};
}

expression::CosNode::CosNode(std::unique_ptr<Node> input)
{
	this->m_NodeStr = "func:cos";
	this->m_Children.push_back(std::move(input));
}

std::function<torch::Tensor()> expression::CosNode::runner()
{
	m_Input = this->m_Children[0]->runner();

	return [this] {
		return torch::cos(m_Input());
	};
}

expression::TanNode::TanNode(std::unique_ptr<Node> input)
{
	this->m_NodeStr = "func:tan";
	this->m_Children.push_back(std::move(input));
}

std::function<torch::Tensor()> expression::TanNode::runner()
{
	m_Input = this->m_Children[0]->runner();

	return [this] {
		return torch::tan(m_Input());
	};
}

expression::SinhNode::SinhNode(std::unique_ptr<Node> input)
{
	this->m_NodeStr = "func:sinh";
	this->m_Children.push_back(std::move(input));
}

std::function<torch::Tensor()> expression::SinhNode::runner()
{
	m_Input = this->m_Children[0]->runner();

	return [this] {
		return torch::sinh(m_Input());
	};
}

expression::CoshNode::CoshNode(std::unique_ptr<Node> input)
{
	this->m_NodeStr = "func:cosh";
	this->m_Children.push_back(std::move(input));
}

std::function<torch::Tensor()> expression::CoshNode::runner()
{
	m_Input = this->m_Children[0]->runner();

	return [this] {
		return torch::cosh(m_Input());
	};
}

expression::TanhNode::TanhNode(std::unique_ptr<Node> input)
{
	this->m_NodeStr = "func:tanh";
	this->m_Children.push_back(std::move(input));
}

std::function<torch::Tensor()> expression::TanhNode::runner()
{
	m_Input = this->m_Children[0]->runner();

	return [this] {
		return torch::tanh(m_Input());
	};
}

expression::AsinNode::AsinNode(std::unique_ptr<Node> input)
{
	this->m_NodeStr = "func:asin";
	this->m_Children.push_back(std::move(input));
}

std::function<torch::Tensor()> expression::AsinNode::runner()
{
	m_Input = this->m_Children[0]->runner();

	return [this] {
		return torch::asin(m_Input());
	};
}

expression::AcosNode::AcosNode(std::unique_ptr<Node> input)
{
	this->m_NodeStr = "func:acos";
	this->m_Children.push_back(std::move(input));
}

std::function<torch::Tensor()> expression::AcosNode::runner()
{
	m_Input = this->m_Children[0]->runner();

	return [this] {
		return torch::acos(m_Input());
	};
}

expression::AtanNode::AtanNode(std::unique_ptr<Node> input)
{
	this->m_NodeStr = "func:atan";
	this->m_Children.push_back(std::move(input));
}

std::function<torch::Tensor()> expression::AtanNode::runner()
{
	m_Input = this->m_Children[0]->runner();

	return [this] {
		return torch::atan(m_Input());
	};
}

expression::Atan2Node::Atan2Node(std::unique_ptr<Node> input, std::unique_ptr<Node> other)
{
	this->m_NodeStr = "func:atan2";
	this->m_Children.push_back(std::move(input));
	this->m_Children.push_back(std::move(other));
}

std::function<torch::Tensor()> expression::Atan2Node::runner()
{
	m_Input = this->m_Children[0]->runner();
	m_Other = this->m_Children[1]->runner();

	return [this] {
		return torch::atan2(m_Input(), m_Other());
	};
}

expression::AsinhNode::AsinhNode(std::unique_ptr<Node> input)
{
	this->m_NodeStr = "func:asinh";
	this->m_Children.push_back(std::move(input));
}

std::function<torch::Tensor()> expression::AsinhNode::runner()
{
	m_Input = this->m_Children[0]->runner();

	return [this] {
		return torch::asinh(m_Input());
	};
}

expression::AcoshNode::AcoshNode(std::unique_ptr<Node> input)
{
	this->m_NodeStr = "func:acosh";
	this->m_Children.push_back(std::move(input));
}

std::function<torch::Tensor()> expression::AcoshNode::runner()
{
	m_Input = this->m_Children[0]->runner();

	return [this] {
		return torch::acosh(m_Input());
	};
}

expression::AtanhNode::AtanhNode(std::unique_ptr<Node> input)
{
	this->m_NodeStr = "func:atanh";
	this->m_Children.push_back(std::move(input));
}

std::function<torch::Tensor()> expression::AtanhNode::runner()
{
	m_Input = this->m_Children[0]->runner();

	return [this] {
		return torch::atanh(m_Input());
	};
}

expression::ExpNode::ExpNode(std::unique_ptr<Node> input)
{
	this->m_NodeStr = "func:exp";
	this->m_Children.push_back(std::move(input));
}

std::function<torch::Tensor()> expression::ExpNode::runner()
{
	m_Input = this->m_Children[0]->runner();

	return [this] {
		return torch::exp(m_Input());
	};
}

expression::PowNode::PowNode(std::unique_ptr<Node> base, std::unique_ptr<Node> exponent)
{
	this->m_NodeStr = "func:pow";
	this->m_Children.push_back(std::move(base));
	this->m_Children.push_back(std::move(exponent));
}

std::function<torch::Tensor()> expression::PowNode::runner()
{
	m_Base = this->m_Children[0]->runner();
	m_Exponent = this->m_Children[1]->runner();

	return [this] {
		return torch::pow(m_Base(), m_Exponent());
	};
}

expression::LogNode::LogNode(std::unique_ptr<Node> input)
{
	this->m_NodeStr = "func:log";
	this->m_Children.push_back(std::move(input));
}

std::function<torch::Tensor()> expression::LogNode::runner()
{
	m_Input = this->m_Children[0]->runner();

	return [this] {
		return torch::log(m_Input());
	};
}

expression::Log10Node::Log10Node(std::unique_ptr<Node> input)
{
	this->m_NodeStr = "func:log10";
	this->m_Children.push_back(std::move(input));
}

std::function<torch::Tensor()> expression::Log10Node::runner()
{
	m_Input = this->m_Children[0]->runner();

	return [this] {
		return torch::log10(m_Input());
	};
}
