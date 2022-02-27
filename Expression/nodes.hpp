#pragma once

#include "../pch.hpp"
#include "TokenAlgebra/token_algebra.hpp"

namespace tc {
	namespace expression {

		using tentok = std::pair<std::optional<torch::Tensor>, tc::OptUPtr<Token>>;

		tentok from_number(float a);

		std::unique_ptr<Token> copy_token(const Token& tok);

		class Node {
		public:

			Node() = default;

			virtual tentok eval() = 0;

			virtual tentok diff(const VariableToken& var) = 0;

		public:
			std::vector<std::unique_ptr<Node>> m_Children;

			std::unique_ptr<Token> m_pToken;
		};

		std::unique_ptr<Node> node_from_pair(const tentok& pair);

		class TokenNode : public Node {
		public:

			TokenNode(const Token& tok);

			tentok eval() override;

			tentok diff(const VariableToken& var) override;

		private:
			std::uint32_t m_TokenType;
			std::vector<int64_t> m_Sizes;
		};

		class TensorNode : public Node {
		public:

			TensorNode(const torch::Tensor& tensor);

			tentok eval() override;

			tentok diff(const VariableToken& var) override;

		private:
			torch::Tensor m_Tensor;
		};

		class VariableNode : public Node {
		public:

			VariableNode(const VariableToken& token, const std::function<torch::Tensor()>& variable_fetcher);

			tentok eval() override;

			tentok diff(const VariableToken& var) override;

		private:
			const VariableToken& m_VarToken;
			const std::function<torch::Tensor()>& m_VariableFetcher; // fetches the tensor
		};

		// Operators

		tentok operator-(const tentok& a);

		class NegNode : public Node {
		public:

			NegNode(std::unique_ptr<Node> child);

			tentok eval() override;

			tentok diff(const VariableToken& var) override;

		};

		tentok operator*(const tentok& a, const tentok& b);

		class MulNode : public Node {
		public:

			MulNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			tentok eval() override;

			tentok diff(const VariableToken& var) override;

		};

		tentok operator/(const tentok& a, const tentok& b);

		class DivNode : public Node {
		public:

			DivNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			tentok eval() override;

			tentok diff(const VariableToken& var) override;
		};

		tentok operator+(const tentok& a, const tentok& b);

		class AddNode : public Node {
		public:

			AddNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			tentok eval() override;

			tentok diff(const VariableToken& var) override;
		};

		tentok operator-(const tentok& a, const tentok& b);

		class SubNode : public Node {
		public:

			SubNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			tentok eval() override;

			tentok diff(const VariableToken& var) override;
		};

		tentok pow(const tentok& a, const tentok& b);

		class PowNode : public Node {
		public:

			PowNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			tentok eval() override;

			tentok diff(const VariableToken& var) override;

		};
		

		// Unary

		tentok abs(const tentok& a);

		class AbsNode : public Node {
		public:

			AbsNode(std::unique_ptr<Node> child);

			tentok eval() override;

			tentok diff(const VariableToken& var) override;

		};

		tentok sqrt(const tentok& a);

		class SqrtNode : public Node {
		public:

			SqrtNode(std::unique_ptr<Node> child);

			tentok eval() override;

			tentok diff(const VariableToken& var) override;

		};

		tentok square(const tentok& a);

		class SquareNode : public Node {
		public:

			SquareNode(std::unique_ptr<Node> child);

			tentok eval() override;

			tentok diff(const VariableToken& var) override;

		};

		tentok exp(const tentok& a);

		class ExpNode : public Node {
		public:

			ExpNode(std::unique_ptr<Node> child);

			tentok eval() override;

			tentok diff(const VariableToken& var) override;

		};

		tentok log(const tentok& a);

		class LogNode : public Node {
		public:

			LogNode(std::unique_ptr<Node> child);

			tentok eval() override;

			tentok diff(const VariableToken& var) override;

		};

		// Trig

		tentok sin(const tentok& a);

		class SinNode : public Node {
		public:

			SinNode(std::unique_ptr<Node> child);

			tentok eval() override;

			tentok diff(const VariableToken& var) override;

		};
		
		tentok cos(const tentok& a);

		class CosNode : public Node {
		public:

			CosNode(std::unique_ptr<Node> child);

			tentok eval() override;

			tentok diff(const VariableToken& var) override;

		};


	}
}