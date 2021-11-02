#pragma once
#include "../pch.hpp"

#include <cmath>
#include <functional>
#include <algorithm>
#include <stack>



namespace expression {

	// Base Node, Abstract
	template<typename T>
	class Node {
	public:

		Node() = default;

		virtual std::function<T()> runner() = 0;

	protected:

		std::string m_NodeStr;
		std::vector<std::unique_ptr<Node>> m_Children;

	};

	// Holds the leafs, fetches the variables at evaluation
	template<typename T>
	class VariableNode : public Node<T> {
	public:

		VariableNode(std::string& var_name, std::function<T()>& var_fetcher)
			: m_VariableName(var_name), m_VariableFetcher(var_fetcher)
		{
			this->m_NodeStr = "var";
		}

		std::function<T()> runner() override {
			return m_VariableFetcher;
		}

	private:
		
		std::string m_VariableName;
		std::function<T()>& m_VariableFetcher;

	};

	// Basic operations
	template<typename T>
	class AddNode : public Node<T> {
	public:

		AddNode(std::unique_ptr<Node<T>> left, std::unique_ptr<Node<T>> right) 
		{
			this->m_NodeStr = "op:add";
			this->m_Children.push_back(std::move(left));
			this->m_Children.push_back(std::move(right));
		}

		std::function<T()> runner() override {
			m_Left = this->m_Children[0]->runner();
			m_Right = this->m_Children[1]->runner();

			return [this] {
				return m_Left() + m_Right();
			};
		}

	private:
		std::function<T()> m_Left;
		std::function<T()> m_Right;
	};

	template<typename T>
	class SubNode : public Node<T> {
	public:

		SubNode(std::unique_ptr<Node<T>> left, std::unique_ptr<Node<T>> right) {
			this->m_NodeStr = "op:sub";
			this->m_Children.push_back(std::move(left));
			this->m_Children.push_back(std::move(right));
		}

		std::function<T()> runner() override {
			m_Left = this->m_Children[0]->runner();
			m_Right = this->m_Children[1]->runner();

			return [this] {
				return m_Left() - m_Right();
			};
		}

	private:
		std::function<T()> m_Left;
		std::function<T()> m_Right;
	};

	template<typename T>
	class MulNode : public Node<T> {
	public:

		MulNode(std::unique_ptr<Node<T>> left, std::unique_ptr<Node<T>> right) {
			this->m_NodeStr = "op:mul";
			this->m_Children.push_back(std::move(left));
			this->m_Children.push_back(std::move(right));
		}

		std::function<T()> runner() override {
			m_Left = this->m_Children[0]->runner();
			m_Right = this->m_Children[1]->runner();

			return [this] {
				return m_Left() * m_Right();
			};
		}

	private:
		std::function<T()> m_Left;
		std::function<T()> m_Right;
	};

	template<typename T>
	class DivNode : public Node<T> {
	public:

		DivNode(std::unique_ptr<Node<T>> left, std::unique_ptr<Node<T>> right) {
			this->m_NodeStr = "op:div";
			this->m_Children.push_back(std::move(left));
			this->m_Children.push_back(std::move(right));
		}

		std::function<T()> runner() override {
			m_Left = this->m_Children[0]->runner();
			m_Right = this->m_Children[1]->runner();

			return [this] {
				return m_Left() / m_Right();
			};
		}

	private:
		std::function<T()> m_Left;
		std::function<T()> m_Right;
	};


	// Trigonometry
	template<typename T>
	class SinNode : public Node<T> {
	public:

		SinNode(std::unique_ptr<Node<T>> input) {
			this->m_NodeStr = "func:sin";
			this->m_Children.push_back(std::move(input));
		}

		std::function<T()> runner() override {
			m_Input = this->m_Children[0]->runner();

			if constexpr (std::is_same<T, torch::Tensor>::value) {
				return [this] {
					return torch::sin(m_Input());
				};
			}
			else {
				return [this] {
					return static_cast<T>(std::sin(m_Input()));
				};
			}
		}

	private:
		std::function<T()> m_Input;
	};
	
	template<typename T>
	class CosNode : public Node<T> {
	public:

		CosNode(std::unique_ptr<Node<T>> input) {
			this->m_NodeStr = "func:cos";
			this->m_Children.push_back(std::move(input));
		}

		std::function<T()> runner() override {
			m_Input = this->m_Children[0]->runner();

			if constexpr (std::is_same<T, torch::Tensor>::value) {
				return [this] {
					return torch::cos(m_Input());
				};
			}
			else {
				return [this] {
					return static_cast<T>(std::cos(m_Input()));
				};
			}
		}

	private:
		std::function<T()> m_Input;
	};

	template<typename T>
	class TanNode : public Node<T> {
	public:

		TanNode(std::unique_ptr<Node<T>> input) {
			this->m_NodeStr = "func:tan";
			this->m_Children.push_back(std::move(input));
		}

		std::function<T()> runner() override {
			m_Input = this->m_Children[0]->runner();

			if constexpr (std::is_same<T, torch::Tensor>::value) {
				return [this] {
					return torch::tan(m_Input());
				};
			}
			else {
				return [this] {
					return static_cast<T>(std::tan(m_Input()));
				};
			}
		}

	private:
		std::function<T()> m_Input;
	};

	template<typename T>
	class SinhNode : public Node<T> {
	public:

		SinhNode(std::unique_ptr<Node<T>> input) {
			this->m_NodeStr = "func:sinh";
			this->m_Children.push_back(std::move(input));
		}

		std::function<T()> runner() override {
			m_Input = this->m_Children[0]->runner();

			if constexpr (std::is_same<T, torch::Tensor>::value) {
				return [this] {
					return torch::sinh(m_Input());
				};
			}
			else {
				return [this] {
					return static_cast<T>(std::sinh(m_Input()));
				};
			}
		}

	private:
		std::function<T()> m_Input;
	};

	template<typename T>
	class CoshNode : public Node<T> {
	public:

		CoshNode(std::unique_ptr<Node<T>> input) {
			this->m_NodeStr = "func:cosh";
			this->m_Children.push_back(std::move(input));
		}

		std::function<T()> runner() override {
			m_Input = this->m_Children[0]->runner();

			if constexpr (std::is_same<T, torch::Tensor>::value) {
				return [this] {
					return torch::cosh(m_Input());
				};
			}
			else {
				return [this] {
					return static_cast<T>(std::cosh(m_Input()));
				};
			}
		}

	private:
		std::function<T()> m_Input;
	};

	template<typename T>
	class TanhNode : public Node<T> {
	public:

		TanhNode(std::unique_ptr<Node<T>> input) {
			this->m_NodeStr = "func:tanh";
			this->m_Children.push_back(std::move(input));
		}

		std::function<T()> runner() override {
			m_Input = this->m_Children[0]->runner();

			if constexpr (std::is_same<T, torch::Tensor>::value) {
				return [this] {
					return torch::tanh(m_Input());
				};
			}
			else {
				return [this] {
					return static_cast<T>(std::tanh(m_Input()));
				};
			}
		}

	private:
		std::function<T()> m_Input;
	};

	template<typename T>
	class AsinNode : public Node<T> {
	public:

		AsinNode(std::unique_ptr<Node<T>> input) {
			this->m_NodeStr = "func:asin";
			this->m_Children.push_back(std::move(input));
		}

		std::function<T()> runner() override {
			m_Input = this->m_Children[0]->runner();

			if constexpr (std::is_same<T, torch::Tensor>::value) {
				return [this] {
					return torch::asin(m_Input());
				};
			}
			else {
				return [this] {
					return static_cast<T>(std::asin(m_Input()));
				};
			}
		}

	private:
		std::function<T()> m_Input;
	};

	template<typename T>
	class AcosNode : public Node<T> {
	public:

		AcosNode(std::unique_ptr<Node<T>> input) {
			this->m_NodeStr = "func:acos";
			this->m_Children.push_back(std::move(input));
		}

		std::function<T()> runner() override {
			m_Input = this->m_Children[0]->runner();

			if constexpr (std::is_same<T, torch::Tensor>::value) {
				return [this] {
					return torch::acos(m_Input());
				};
			}
			else {
				return [this] {
					return static_cast<T>(std::acos(m_Input()));
				};
			}
		}

	private:
		std::function<T()> m_Input;
	};

	template<typename T>
	class AtanNode : public Node<T> {
	public:

		AtanNode(std::unique_ptr<Node<T>> input) {
			this->m_NodeStr = "func:atan";
			this->m_Children.push_back(std::move(input));
		}

		std::function<T()> runner() override {
			m_Input = this->m_Children[0]->runner();

			if constexpr (std::is_same<T, torch::Tensor>::value) {
				return [this] {
					return torch::atan(m_Input());
				};
			}
			else {
				return [this] {
					return static_cast<T>(std::atan(m_Input()));
				};
			}
		}

	private:
		std::function<T()> m_Input;
	};

	template<typename T>
	class Atan2Node : public Node<T> {
	public:

		Atan2Node(std::unique_ptr<Node<T>> input, std::unique_ptr<Node<T>> other) {
			this->m_NodeStr = "func:atan2";
			this->m_Children.push_back(std::move(input));
			this->m_Children.push_back(std::move(other));
		}

		std::function<T()> runner() override {
			m_Input = this->m_Children[0]->runner();
			m_Other = this->m_Children[1]->runner();

			if constexpr (std::is_same<T, torch::Tensor>::value) {
				return [this] {
					return torch::atan2(m_Input(), m_Other());
				};
			}
			else {
				return [this] {
					return static_cast<T>(std::atan2(m_Input(), m_Other()));
				};
			}
		}

	private:
		std::function<T()> m_Input;
		std::function<T()> m_Other;
	};

	template<typename T>
	class AsinhNode : public Node<T> {
	public:

		AsinhNode(std::unique_ptr<Node<T>> input) {
			this->m_NodeStr = "func:asinh";
			this->m_Children.push_back(std::move(input));
		}

		std::function<T()> runner() override {
			m_Input = this->m_Children[0]->runner();

			if constexpr (std::is_same<T, torch::Tensor>::value) {
				return [this] {
					return torch::asinh(m_Input());
				};
			}
			else {
				return [this] {
					return static_cast<T>(std::asinh(m_Input()));
				};
			}
		}

	private:
		std::function<T()> m_Input;
	};

	template<typename T>
	class AcoshNode : public Node<T> {
	public:

		AcoshNode(std::unique_ptr<Node<T>> input) {
			this->m_NodeStr = "func:acosh";
			this->m_Children.push_back(std::move(input));
		}

		std::function<T()> runner() override {
			m_Input = this->m_Children[0]->runner();

			if constexpr (std::is_same<T, torch::Tensor>::value) {
				return [this] {
					return torch::acosh(m_Input());
				};
			}
			else {
				return [this] {
					return static_cast<T>(std::acosh(m_Input()));
				};
			}
		}

	private:
		std::function<T()> m_Input;
	};

	template<typename T>
	class AtanhNode : public Node<T> {
	public:

		AtanhNode(std::unique_ptr<Node<T>> input) {
			this->m_NodeStr = "func:atanh";
			this->m_Children.push_back(std::move(input));
		}

		std::function<T()> runner() override {
			m_Input = this->m_Children[0]->runner();

			if constexpr (std::is_same<T, torch::Tensor>::value) {
				return [this] {
					return torch::atanh(m_Input());
				};
			}
			else {
				return [this] {
					return static_cast<T>(std::atanh(m_Input()));
				};
			}
		}

	private:
		std::function<T()> m_Input;
	};


	// Powers, Exp, Log
	template<typename T>
	class ExpNode : public Node<T> {
	public:

		ExpNode(std::unique_ptr<Node<T>> input) {
			this->m_NodeStr = "func:exp";
			this->m_Children.push_back(std::move(input));
		}

		std::function<T()> runner() override {
			m_Input = this->m_Children[0]->runner();

			if constexpr (std::is_same<T, torch::Tensor>::value) {
				return [this] {
					return torch::exp(m_Input());
				};
			}
			else {
				return [this] {
					return static_cast<T>(std::exp(m_Input()));
				};
			}
		}

	private:
		std::function<T()> m_Input;
	};

	template<typename T>
	class PowNode : public Node<T> {
	public:

		PowNode(std::unique_ptr<Node<T>> base, std::unique_ptr<Node<T>> exponent) {
			this->m_NodeStr = "func:pow";
			this->m_Children.push_back(std::move(base));
			this->m_Children.push_back(std::move(exponent));
		}

		std::function<T()> runner() override {
			m_Base = this->m_Children[0]->runner();
			m_Exponent = this->m_Children[1]->runner();

			if constexpr (std::is_same<T, torch::Tensor>::value) {
				return [this] {
					return torch::pow(m_Base(), m_Exponent());
				};
			}
			else {
				return [this] {
					return static_cast<T>(std::pow(m_Base(), m_Exponent()));
				};
			}
		}

	private:
		std::function<T()> m_Base;
		std::function<T()> m_Exponent;
	};

	template<typename T>
	class LogNode : public Node<T> {
	public:

		LogNode(std::unique_ptr<Node<T>> input) {
			this->m_NodeStr = "func:log";
			this->m_Children.push_back(std::move(input));
		}

		std::function<T()> runner() override {
			m_Input = this->m_Children[0]->runner();

			if constexpr (std::is_same<T, torch::Tensor>::value) {
				return [this] {
					return torch::log(m_Input());
				};
			}
			else {
				return [this] {
					return static_cast<T>(std::log(m_Input()));
				};
			}
		}

	private:
		std::function<T()> m_Input;
	};

	template<typename T>
	class Log10Node : public Node<T> {
	public:

		Log10Node(std::unique_ptr<Node<T>> input) {
			this->m_NodeStr = "func:log10";
			this->m_Children.push_back(std::move(input));
		}

		std::function<T()> runner() override {
			m_Input = this->m_Children[0]->runner();

			if constexpr (std::is_same<T, torch::Tensor>::value) {
				return [this] {
					return torch::log10(m_Input());
				};
			}
			else {
				return [this] {
					return static_cast<T>(std::log10(m_Input()));
				};
			}
		}

	private:
		std::function<T()> m_Input;
	};



}