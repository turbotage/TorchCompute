#pragma once

#include "../pch.hpp"

#include "expression_nodes.hpp"

/*
The Lexer will create internal representations of variable names and numbers, how these look will determine
the structure the shunter creates and finally how the expression tree is built. It is therefore extremly important
that the following rules are followed by expressions

GENERAL		:	expressions may not contain white-spaces (might be changed later)
			:	expressions should only contain valid operators, functions and valid variable names
			:	

VARIABLES	:	variable names starts with '$'  (eg 2*sin($X0))
				variable names can only contain alphanumerical characters
				variable names may not contain the substrings {"NEG","NUMVAR"}

*/


namespace expression {

	class ExpressionGraph {
	public:

		ExpressionGraph(std::string expression);

		std::function<torch::Tensor()> getFunc();

		void setVariableFetcher(std::string var_name, const std::function<torch::Tensor()>& var_fetcher);

		void to(torch::Device device);

		std::string getReadableTree();

	private:

		static std::pair<std::string, bool> check_variable_sign(const std::string& varname);

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
		std::list<std::function<torch::Tensor()>> m_VariableNegates;

		std::unique_ptr<Node> m_pRootNode;
	};


	

}