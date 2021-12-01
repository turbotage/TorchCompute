#pragma once

#include "../pch.hpp"
#include "expression_nodes.hpp"

namespace expression {

	class ExpressionGraph {
	public:

		ExpressionGraph(std::string expression);

		std::function<torch::Tensor()> getFunc();

		void setVariableFetcher(std::string var_name, const std::function<torch::Tensor()>& var_fetcher);

	private:

		static std::function<torch::Tensor()> fetch_and_negate(
			const std::function<torch::Tensor(torch::Tensor)>& negate,
			const std::function<torch::Tensor()>& fetch);

		static void handle_number(std::string numid, torch::Tensor& var, std::stack<std::unique_ptr<Node>>& node_stack);

		static void handle_variable(std::string varid, 
			const std::function<torch::Tensor()>& var_fetcher, std::stack<std::unique_ptr<Node>>& node_stack);

		static void handle_function(std::string funcid, std::stack<std::unique_ptr<Node>>& node_stack);

		static void handle_operation(std::string opid, std::stack<std::unique_ptr<Node>>& node_stack);


	private:

		std::string m_Expression;
		std::string m_LexedExpression;

		std::map<std::string, torch::Tensor> m_Numbers;
		std::map<std::string, std::function<torch::Tensor()>> m_VariableFetchers;
		
		std::map<std::string, std::function<torch::Tensor(torch::Tensor)>> m_VariableNegateMaps;

		std::unique_ptr<Node> m_pRootNode;
	};


	

}