#pragma once

#include "../pch.hpp"
#include "expression_nodes.hpp"

#include "shunter.hpp"

namespace expression {

	using NumberResolver = std::function<std::function<torch::Tensor()>(const std::string&)>;

	template<typename T>
	class ExpressionGraph {
	public:

		ExpressionGraph(std::stack<Token> token_stack, NumberResolver numberResolver);

		std::function<T()> getFunc();

		void setVariableFetcher(std::string var_name, std::function<T()>& variable_fetcher);

	private:

		static void handle_number_token(Token t, std::function<T()>& var_fetcher, std::stack<std::unique_ptr<Node<T>>>& node_stack);

		static void handle_variable_token(Token t, std::function<T()>& var_fetcher, std::stack<std::unique_ptr<Node<T>>>& node_stack);

		static void handle_function_token(Token t, std::stack<std::unique_ptr<Node<T>>>& node_stack);

		static void handle_operation_token(Token t, std::stack<std::unique_ptr<Node<T>>>& node_stack);

	private:

		std::map<std::string, std::function<T()>> m_Variables;
		std::unique_ptr<Node<T>> m_pRootNode;

		std::function<std::function<T()>(const std::string&)> m_NumberResolver;

	};

	// Public

	template<typename T>
	inline ExpressionGraph<T>::ExpressionGraph(std::stack<Token> token_stack, NumberResolver numberResolver)
	{
		m_NumberResolver = numberResolver;
		
		std::stack<std::unique_ptr<Node<T>>> node_stack;

		int num_counter = 0;
		while (!token_stack.empty()) {
			Token top = token_stack.top(); token_stack.pop();
			if (top.token_type == eTokenType::NUMBER) {
				auto num_name = "#" + std::to_string(num_counter) + ":" + top.token_str;
				++num_counter;
				m_Variables[num_name] = m_NumberResolver(top.token_str);
				handle_number_token(top, m_Variables[num_name], node_stack);
			}
			else if (top.token_type == eTokenType::VARIABLE) {
				handle_variable_token(top, m_Variables[top.token_str], node_stack);
			}
			else if (top.token_type == eTokenType::FUNCTION) {
				handle_function_token(top, node_stack);
			}
			else if (top.token_type == eTokenType::OPERATOR) {
				handle_operation_token(top, node_stack);
			}
			else {
				throw std::runtime_error("Not implemented token type in expression graph constructed");
			}
		}

		if (node_stack.size() != 1)
			throw std::runtime_error("Top nod not unique");

		m_pRootNode = std::move(node_stack.top());
		node_stack.pop();
	}

	template<typename T>
	inline std::function<T()> ExpressionGraph<T>::getFunc()
	{
		return m_pRootNode->runner();
	}

	template<typename T>
	inline void ExpressionGraph<T>::setVariableFetcher(std::string var_name, std::function<T()>& variable_fetcher)
	{
		m_Variables[var_name] = variable_fetcher;
	}


	// Private
	template<typename T>
	inline void ExpressionGraph<T>::handle_number_token(Token t, std::function<T()>& var_fetcher, std::stack<std::unique_ptr<Node<T>>>& node_stack)
	{
		std::unique_ptr<Node<T>> ptr = std::make_unique<VariableNode<T>>(t.token_str, var_fetcher);
		node_stack.push(std::move(ptr));
	}

	template<typename T>
	inline void ExpressionGraph<T>::handle_variable_token(Token t, std::function<T()>& var_fetcher, std::stack<std::unique_ptr<Node<T>>>& node_stack)
	{
		std::unique_ptr<Node<T>> ptr = std::make_unique<VariableNode<T>>(t.token_str, var_fetcher);
		node_stack.push(std::move(ptr));
	}

	template<typename T>
	inline void ExpressionGraph<T>::handle_function_token(Token t, std::stack<std::unique_ptr<Node<T>>>& node_stack)
	{
		std::unique_ptr<Node<T>> ptr;
		if (t.token_str == "sin") { // Trigonometry
			ptr = std::make_unique<SinNode<T>>(std::move(node_stack.top()));
			node_stack.pop();
		}
		else if (t.token_str == "cos") {
			ptr = std::make_unique<CosNode<T>>(std::move(node_stack.top()));
			node_stack.pop();
		}
		else if (t.token_str == "tan") {
			ptr = std::make_unique<TanNode<T>>(std::move(node_stack.top()));
			node_stack.pop();
		}
		else if (t.token_str == "sinh") {
			ptr = std::make_unique<SinhNode<T>>(std::move(node_stack.top()));
			node_stack.pop();
		}
		else if (t.token_str == "cosh") {
			ptr = std::make_unique<CoshNode<T>>(std::move(node_stack.top()));
			node_stack.pop();
		}
		else if (t.token_str == "tanh") {
			ptr = std::make_unique<TanhNode<T>>(std::move(node_stack.top()));
			node_stack.pop();
		}
		else if (t.token_str == "asin") {
			ptr = std::make_unique<AsinNode<T>>(std::move(node_stack.top()));
			node_stack.pop();
		}
		else if (t.token_str == "acos") {
			ptr = std::make_unique<AcosNode<T>>(std::move(node_stack.top()));
			node_stack.pop();
		}
		else if (t.token_str == "atan") {
			ptr = std::make_unique<AtanNode<T>>(std::move(node_stack.top()));
			node_stack.pop();
		}
		else if (t.token_str == "atan2") {
			auto x = std::move(node_stack.top()); node_stack.pop();
			ptr = std::make_unique<Atan2Node<T>>(std::move(x), std::move(node_stack.top()));
			node_stack.pop();
		}
		else if (t.token_str == "asinh") {
			ptr = std::make_unique<AsinhNode<T>>(std::move(node_stack.top()));
			node_stack.pop();
		}
		else if (t.token_str == "acosh") {
			ptr = std::make_unique<AcoshNode<T>>(std::move(node_stack.top()));
			node_stack.pop();
		}
		else if (t.token_str == "atanh") {
			ptr = std::make_unique<AtanhNode<T>>(std::move(node_stack.top()));
			node_stack.pop();
		}
		else if (t.token_str == "exp") { // Powers, Exp, Log
			ptr = std::make_unique<ExpNode<T>>(std::move(node_stack.top()));
			node_stack.pop();
		}
		else if (t.token_str == "pow") {
			auto base = std::move(node_stack.top()); node_stack.pop();
			ptr = std::make_unique<PowNode<T>>(std::move(base), std::move(node_stack.top()));
			node_stack.pop();
		}
		else if (t.token_str == "log") {
			ptr = std::make_unique<LogNode<T>>(std::move(node_stack.top()));
			node_stack.pop();
		}
		else if (t.token_str == "log10") {
			ptr = std::make_unique<Log10Node<T>>(std::move(node_stack.top()));
			node_stack.pop();
		}
		else {
			throw std::runtime_error("Not implemented function");
		}

		node_stack.push(std::move(ptr));
	}

	template<typename T>
	inline void ExpressionGraph<T>::handle_operation_token(Token t, std::stack<std::unique_ptr<Node<T>>>& node_stack)
	{
		std::unique_ptr<Node<T>> ptr;
		if (t.token_str == "+") {
			int i;
			auto right = std::move(node_stack.top()); node_stack.pop();
			auto left = std::move(node_stack.top()); node_stack.pop();
			ptr = std::make_unique<AddNode<T>>(std::move(left), std::move(right));
		}
		else if (t.token_str == "-") {
			auto right = std::move(node_stack.top()); node_stack.pop();
			auto left = std::move(node_stack.top()); node_stack.pop();
			ptr = std::make_unique<SubNode<T>>(std::move(left), std::move(right));
		}
		else if (t.token_str == "*") {
			auto right = std::move(node_stack.top()); node_stack.pop();
			auto left = std::move(node_stack.top()); node_stack.pop();
			ptr = std::make_unique<MulNode<T>>(std::move(left), std::move(right));
		}
		else if (t.token_str == "/") {
			auto right = std::move(node_stack.top()); node_stack.pop();
			auto left = std::move(node_stack.top()); node_stack.pop();
			ptr = std::make_unique<DivNode<T>>(std::move(left), std::move(right));
		}
		else if (t.token_str == "^") {
			auto right = std::move(node_stack.top()); node_stack.pop();
			auto left = std::move(node_stack.top()); node_stack.pop();
			ptr = std::make_unique<PowNode<T>>(std::move(left), std::move(right));
		}
		else {
			throw std::runtime_error("Not implemented operator");
		}

		node_stack.push(std::move(ptr));
	}

}