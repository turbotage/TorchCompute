#pragma once

#include "../pch.hpp"
#include "TokenAlgebra/token_algebra.hpp"

namespace tc {
	namespace expression {


		class Node {
		public:

			Node() = default;

			virtual std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> eval() = 0;

			virtual std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> diff(const VariableToken& var) = 0;

		public:
			std::vector<std::unique_ptr<Node>> m_Children;

			ZeroToken m_ZeroToken;
			UnityToken m_UnityToken;
			NegUnityToken m_NegUnityToken;
			NanToken m_NanToken;
			NumberToken m_NumToken;
			std::unique_ptr<Token> m_pToken;
		};

		std::unique_ptr<Node> node_from_pair(const std::pair<std::optional<torch::Tensor>, tc::OptRef<const tc::expression::Token>>& pair);

		class TokenNode : public Node {
		public:

			TokenNode(const Token& tok);

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> eval() override;

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> diff(const VariableToken& var) override;

		private:
			std::uint32_t m_TokenType;
		};

		class TensorNode : public Node {
		public:

			TensorNode(const torch::Tensor& tensor);

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> eval() override;

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> diff(const VariableToken& var) override;

		private:
			torch::Tensor m_Tensor;
		};

		class VariableNode : public Node {
		public:

			VariableNode(const VariableToken& token, const std::function<torch::Tensor()>& variable_fetcher);

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> eval() override;

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> diff(const VariableToken& var) override;

		private:
			const VariableToken& m_VarToken;
			const std::function<torch::Tensor()>& m_VariableFetcher; // fetches the tensor
		};

		class NegNode : public Node {
		public:

			NegNode(std::unique_ptr<Node> child);

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> eval() override;

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> diff(const VariableToken& var) override;

		};

		class MulNode : public Node {
		public:

			MulNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> eval() override;

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> diff(const VariableToken& var) override;

		};

		class DivNode : public Node {
		public:

			DivNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> eval() override;

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> diff(const VariableToken& var) override;
		};

		class AddNode : public Node {
		public:

			AddNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> eval() override;

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> diff(const VariableToken& var) override;
		};

		class SubNode : public Node {
		public:

			SubNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> eval() override;

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> diff(const VariableToken& var) override;
		};

		class PowNode : public Node {
		public:

			PowNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> eval() override;

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> diff(const VariableToken& var) override;

		};

		class SquareNode : public Node {
		public:

			SquareNode(std::unique_ptr<Node> child);

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> eval() override;

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> diff(const VariableToken& var) override;

		};

		class LogNode : public Node {
		public:

			LogNode(std::unique_ptr<Node> child);

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> eval() override;

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> diff(const VariableToken& var) override;

		};

	}
}