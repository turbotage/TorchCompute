#pragma once
#include "../pch.hpp"

#include <cmath>
#include <functional>
#include <algorithm>
#include <stack>



namespace expression {

	// Base Node, Abstract
	class Node {
	public:

		Node() = default;

		virtual std::function<torch::Tensor()> runner() = 0;

	protected:

		std::string m_NodeStr;
		std::vector<std::unique_ptr<Node>> m_Children;

	};

	class NumberNode : public Node {
	public:

		NumberNode(std::string& num_name, torch::Tensor& val)
			: m_NumberName(num_name), m_Value(val)
		{
			this->m_NodeStr = "num";
		}
		
		std::function<torch::Tensor()> runner() override {
			return [this]() { return this->m_Value; };
		}

	private:
		std::string m_NumberName;
		torch::Tensor& m_Value;
	};

	// Holds leafs, fetches the variables at evaluation
	class VariableNode : public Node {
	public:

		VariableNode(std::string& var_name, const std::function<torch::Tensor()>& var_fetcher)
			: m_VariableName(var_name), m_VariableFetcher(var_fetcher)
		{
			this->m_NodeStr = "var";
		}

		std::function<torch::Tensor()> runner() override {
			return [this]() { return this->m_VariableFetcher(); };
		}

	private:
		
		std::string m_VariableName;
		const std::function<torch::Tensor()>& m_VariableFetcher;

	};

	// Basic operations
	class AddNode : public Node {
	public:

		AddNode(std::unique_ptr<Node> left, std::unique_ptr<Node> right) 
		{
			this->m_NodeStr = "op:add";
			this->m_Children.push_back(std::move(left));
			this->m_Children.push_back(std::move(right));
		}

		std::function<torch::Tensor()> runner() override {
			m_Left = this->m_Children[0]->runner();
			m_Right = this->m_Children[1]->runner();

			return [this] {
				return this->m_Left() + this->m_Right();
			};
		}

	private:
		std::function<torch::Tensor()> m_Left;
		std::function<torch::Tensor()> m_Right;
	};

	class SubNode : public Node {
	public:

		SubNode(std::unique_ptr<Node> left, std::unique_ptr<Node> right) {
			this->m_NodeStr = "op:sub";
			this->m_Children.push_back(std::move(left));
			this->m_Children.push_back(std::move(right));
		}

		std::function<torch::Tensor()> runner() override {
			m_Left = this->m_Children[0]->runner();
			m_Right = this->m_Children[1]->runner();

			return [this] {
				return m_Left() - m_Right();
			};
		}

	private:
		std::function<torch::Tensor()> m_Left;
		std::function<torch::Tensor()> m_Right;
	};

	class MulNode : public Node {
	public:

		MulNode(std::unique_ptr<Node> left, std::unique_ptr<Node> right) {
			this->m_NodeStr = "op:mul";
			this->m_Children.push_back(std::move(left));
			this->m_Children.push_back(std::move(right));
		}

		std::function<torch::Tensor()> runner() override {
			m_Left = this->m_Children[0]->runner();
			m_Right = this->m_Children[1]->runner();

			return [this] {
				return m_Left() * m_Right();
			};
		}

	private:
		std::function<torch::Tensor()> m_Left;
		std::function<torch::Tensor()> m_Right;
	};

	class DivNode : public Node {
	public:

		DivNode(std::unique_ptr<Node> left, std::unique_ptr<Node> right) {
			this->m_NodeStr = "op:div";
			this->m_Children.push_back(std::move(left));
			this->m_Children.push_back(std::move(right));
		}

		std::function<torch::Tensor()> runner() override {
			m_Left = this->m_Children[0]->runner();
			m_Right = this->m_Children[1]->runner();

			return [this] {
				return m_Left() / m_Right();
			};
		}

	private:
		std::function<torch::Tensor()> m_Left;
		std::function<torch::Tensor()> m_Right;
	};


	// Trigonometry
	class SinNode : public Node {
	public:

		SinNode(std::unique_ptr<Node> input) {
			this->m_NodeStr = "func:sin";
			this->m_Children.push_back(std::move(input));
		}

		std::function<torch::Tensor()> runner() override {
			m_Input = this->m_Children[0]->runner();

			return [this] {
				return torch::sin(m_Input());
			};
		}

	private:
		std::function<torch::Tensor()> m_Input;
	};
	
	class CosNode : public Node {
	public:

		CosNode(std::unique_ptr<Node> input) {
			this->m_NodeStr = "func:cos";
			this->m_Children.push_back(std::move(input));
		}

		std::function<torch::Tensor()> runner() override {
			m_Input = this->m_Children[0]->runner();

			return [this] {
				return torch::cos(m_Input());
			};
		}

	private:
		std::function<torch::Tensor()> m_Input;
	};

	class TanNode : public Node {
	public:

		TanNode(std::unique_ptr<Node> input) {
			this->m_NodeStr = "func:tan";
			this->m_Children.push_back(std::move(input));
		}

		std::function<torch::Tensor()> runner() override {
			m_Input = this->m_Children[0]->runner();

			return [this] {
				return torch::tan(m_Input());
			};
		}

	private:
		std::function<torch::Tensor()> m_Input;
	};

	class SinhNode : public Node {
	public:

		SinhNode(std::unique_ptr<Node> input) {
			this->m_NodeStr = "func:sinh";
			this->m_Children.push_back(std::move(input));
		}

		std::function<torch::Tensor()> runner() override {
			m_Input = this->m_Children[0]->runner();

			return [this] {
				return torch::sinh(m_Input());
			};
		}

	private:
		std::function<torch::Tensor()> m_Input;
	};

	class CoshNode : public Node {
	public:

		CoshNode(std::unique_ptr<Node> input) {
			this->m_NodeStr = "func:cosh";
			this->m_Children.push_back(std::move(input));
		}

		std::function<torch::Tensor()> runner() override {
			m_Input = this->m_Children[0]->runner();

			return [this] {
				return torch::cosh(m_Input());
			};
		}

	private:
		std::function<torch::Tensor()> m_Input;
	};

	class TanhNode : public Node {
	public:

		TanhNode(std::unique_ptr<Node> input) {
			this->m_NodeStr = "func:tanh";
			this->m_Children.push_back(std::move(input));
		}

		std::function<torch::Tensor()> runner() override {
			m_Input = this->m_Children[0]->runner();

			return [this] {
				return torch::tanh(m_Input());
			};
		}

	private:
		std::function<torch::Tensor()> m_Input;
	};

	
	class AsinNode : public Node {
	public:

		AsinNode(std::unique_ptr<Node> input) {
			this->m_NodeStr = "func:asin";
			this->m_Children.push_back(std::move(input));
		}

		std::function<torch::Tensor()> runner() override {
			m_Input = this->m_Children[0]->runner();

			return [this] {
				return torch::asin(m_Input());
			};
		}

	private:
		std::function<torch::Tensor()> m_Input;
	};

	
	class AcosNode : public Node {
	public:

		AcosNode(std::unique_ptr<Node> input) {
			this->m_NodeStr = "func:acos";
			this->m_Children.push_back(std::move(input));
		}

		std::function<torch::Tensor()> runner() override {
			m_Input = this->m_Children[0]->runner();

			return [this] {
				return torch::acos(m_Input());
			};
		}

	private:
		std::function<torch::Tensor()> m_Input;
	};

	
	class AtanNode : public Node {
	public:

		AtanNode(std::unique_ptr<Node> input) {
			this->m_NodeStr = "func:atan";
			this->m_Children.push_back(std::move(input));
		}

		std::function<torch::Tensor()> runner() override {
			m_Input = this->m_Children[0]->runner();

			return [this] {
				return torch::atan(m_Input());
			};
		}

	private:
		std::function<torch::Tensor()> m_Input;
	};

	
	class Atan2Node : public Node {
	public:

		Atan2Node(std::unique_ptr<Node> input, std::unique_ptr<Node> other) {
			this->m_NodeStr = "func:atan2";
			this->m_Children.push_back(std::move(input));
			this->m_Children.push_back(std::move(other));
		}

		std::function<torch::Tensor()> runner() override {
			m_Input = this->m_Children[0]->runner();
			m_Other = this->m_Children[1]->runner();

			return [this] {
				return torch::atan2(m_Input(), m_Other());
			};
		}

	private:
		std::function<torch::Tensor()> m_Input;
		std::function<torch::Tensor()> m_Other;
	};

	
	class AsinhNode : public Node {
	public:

		AsinhNode(std::unique_ptr<Node> input) {
			this->m_NodeStr = "func:asinh";
			this->m_Children.push_back(std::move(input));
		}

		std::function<torch::Tensor()> runner() override {
			m_Input = this->m_Children[0]->runner();

			return [this] {
				return torch::asinh(m_Input());
			};
	}

	private:
		std::function<torch::Tensor()> m_Input;
	};

	
	class AcoshNode : public Node {
	public:

		AcoshNode(std::unique_ptr<Node> input) {
			this->m_NodeStr = "func:acosh";
			this->m_Children.push_back(std::move(input));
		}

		std::function<torch::Tensor()> runner() override {
			m_Input = this->m_Children[0]->runner();

			return [this] {
				return torch::acosh(m_Input());
			};
		}

	private:
		std::function<torch::Tensor()> m_Input;
	};

	
	class AtanhNode : public Node {
	public:

		AtanhNode(std::unique_ptr<Node> input) {
			this->m_NodeStr = "func:atanh";
			this->m_Children.push_back(std::move(input));
		}

		std::function<torch::Tensor()> runner() override {
			m_Input = this->m_Children[0]->runner();

			return [this] {
				return torch::atanh(m_Input());
			};
		}

	private:
		std::function<torch::Tensor()> m_Input;
	};


	// Powers, Exp, Log
	
	class ExpNode : public Node {
	public:

		ExpNode(std::unique_ptr<Node> input) {
			this->m_NodeStr = "func:exp";
			this->m_Children.push_back(std::move(input));
		}

		std::function<torch::Tensor()> runner() override {
			m_Input = this->m_Children[0]->runner();

			return [this] {
				return torch::exp(m_Input());
			};
		}

	private:
		std::function<torch::Tensor()> m_Input;
	};

	
	class PowNode : public Node {
	public:

		PowNode(std::unique_ptr<Node> base, std::unique_ptr<Node> exponent) {
			this->m_NodeStr = "func:pow";
			this->m_Children.push_back(std::move(base));
			this->m_Children.push_back(std::move(exponent));
		}

		std::function<torch::Tensor()> runner() override {
			m_Base = this->m_Children[0]->runner();
			m_Exponent = this->m_Children[1]->runner();

			return [this] {
				return torch::pow(m_Base(), m_Exponent());
			};
		}

	private:
		std::function<torch::Tensor()> m_Base;
		std::function<torch::Tensor()> m_Exponent;
	};

	
	class LogNode : public Node {
	public:

		LogNode(std::unique_ptr<Node> input) {
			this->m_NodeStr = "func:log";
			this->m_Children.push_back(std::move(input));
		}

		std::function<torch::Tensor()> runner() override {
			m_Input = this->m_Children[0]->runner();

			return [this] {
				return torch::log(m_Input());
			};
	}

	private:
		std::function<torch::Tensor()> m_Input;
	};

	
	class Log10Node : public Node {
	public:

		Log10Node(std::unique_ptr<Node> input) {
			this->m_NodeStr = "func:log10";
			this->m_Children.push_back(std::move(input));
		}

		std::function<torch::Tensor()> runner() override {
			m_Input = this->m_Children[0]->runner();

			return [this] {
				return torch::log10(m_Input());
			};
		}

	private:
		std::function<torch::Tensor()> m_Input;
	};



}