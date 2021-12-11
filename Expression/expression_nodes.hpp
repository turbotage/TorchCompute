#pragma once
#include "../pch.hpp"

#include <cmath>
#include <functional>
#include <algorithm>
#include <stack>


namespace tc {
	namespace expression {

		// Base Node, Abstract
		class Node {
		public:

			Node() = default;

			virtual std::function<torch::Tensor()> runner() = 0;

			std::string getNodeName();

			friend std::string readableNode(Node* node, tc::ui32 nTabs);

		protected:

			std::string m_NodeStr;
			std::vector<std::unique_ptr<Node>> m_Children;

		};

		class NumberNode : public Node {
		public:

			NumberNode(const std::string& num_name, torch::Tensor& val);
		
			std::function<torch::Tensor()> runner() override;

			std::string getNumberName();

		private:
			std::string m_NumberName;
			torch::Tensor& m_Value;
		};

		// Holds leafs, fetches the variables at evaluation
		class VariableNode : public Node {
		public:

			VariableNode(const std::string& var_name, const std::function<torch::Tensor()>& var_fetcher);

			std::function<torch::Tensor()> runner() override;

			std::string getVarName();

		private:
		
			std::string m_VariableName;
			const std::function<torch::Tensor()>& m_VariableFetcher;

		};

		// Basic operations
		class AddNode : public Node {
		public:

			AddNode(std::unique_ptr<Node> left, std::unique_ptr<Node> right);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Left;
			std::function<torch::Tensor()> m_Right;
		};

		class SubNode : public Node {
		public:

			SubNode(std::unique_ptr<Node> left, std::unique_ptr<Node> right);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Left;
			std::function<torch::Tensor()> m_Right;
		};

		class MulNode : public Node {
		public:

			MulNode(std::unique_ptr<Node> left, std::unique_ptr<Node> right);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Left;
			std::function<torch::Tensor()> m_Right;
		};

		class DivNode : public Node {
		public:

			DivNode(std::unique_ptr<Node> left, std::unique_ptr<Node> right);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Left;
			std::function<torch::Tensor()> m_Right;
		};


		// Trigonometry
		class SinNode : public Node {
		public:

			SinNode(std::unique_ptr<Node> input);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Input;
		};
	
		class CosNode : public Node {
		public:

			CosNode(std::unique_ptr<Node> input);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Input;
		};

		class TanNode : public Node {
		public:

			TanNode(std::unique_ptr<Node> input);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Input;
		};

		class SinhNode : public Node {
		public:

			SinhNode(std::unique_ptr<Node> input);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Input;
		};

		class CoshNode : public Node {
		public:

			CoshNode(std::unique_ptr<Node> input);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Input;
		};

		class TanhNode : public Node {
		public:

			TanhNode(std::unique_ptr<Node> input);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Input;
		};

	
		class AsinNode : public Node {
		public:

			AsinNode(std::unique_ptr<Node> input);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Input;
		};

	
		class AcosNode : public Node {
		public:

			AcosNode(std::unique_ptr<Node> input);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Input;
		};

	
		class AtanNode : public Node {
		public:

			AtanNode(std::unique_ptr<Node> input);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Input;
		};

	
		class Atan2Node : public Node {
		public:

			Atan2Node(std::unique_ptr<Node> input, std::unique_ptr<Node> other);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Input;
			std::function<torch::Tensor()> m_Other;
		};

	
		class AsinhNode : public Node {
		public:

			AsinhNode(std::unique_ptr<Node> input);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Input;
		};

	
		class AcoshNode : public Node {
		public:

			AcoshNode(std::unique_ptr<Node> input);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Input;
		};

	
		class AtanhNode : public Node {
		public:

			AtanhNode(std::unique_ptr<Node> input);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Input;
		};


		// Powers, Exp, Log
	
		class ExpNode : public Node {
		public:

			ExpNode(std::unique_ptr<Node> input);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Input;
		};

		class SqrtNode : public Node {
		public:

			SqrtNode(std::unique_ptr<Node> input);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Input;
		};
	
		class SquareNode : public Node {
		public:

			SquareNode(std::unique_ptr<Node> input);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Input;
		};

		class PowNode : public Node {
		public:

			PowNode(std::unique_ptr<Node> base, std::unique_ptr<Node> exponent);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Base;
			std::function<torch::Tensor()> m_Exponent;
		};
	
		class LogNode : public Node {
		public:

			LogNode(std::unique_ptr<Node> input);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Input;
		};

	
		class Log10Node : public Node {
		public:

			Log10Node(std::unique_ptr<Node> input);

			std::function<torch::Tensor()> runner() override;

		private:
			std::function<torch::Tensor()> m_Input;
		};



	}
}