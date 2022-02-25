#pragma once

#include "../pch.hpp"
#include "Parser/lexer.hpp"

namespace tc {
	namespace expression {

		class Node {
		public:

			Node() = default;

			virtual std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> eval() = 0;

			virtual std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> diff(const VariableToken& var) = 0;

		protected:
			std::vector<std::unique_ptr<Node>> m_Children;

			ZeroToken m_ZeroToken;
			UnityToken m_UnityToken;
			NegUnityToken m_NegUnityToken;
			NumberToken m_NumToken;
		};

		class NumberNode : public Node {
		public:

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> eval() override;

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> diff(const VariableToken& var) override;

		};

		class VariableNode : public Node {
		public:

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> eval() override;

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> diff(const VariableToken& var) override;

		private:
			VariableToken m_VarToken;
			const std::function<torch::Tensor()>& m_VariableFetcher; // fetches the tensor
		};

		class NegNode : public Node {
		public:

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> eval() override;

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> diff(const VariableToken& var) override;

		};

		class MulNode : public Node {
		public:

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> eval() override;

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> diff(const VariableToken& var) override;

		};

		class PowNode : public Node {
		public:

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> eval() override;

			std::pair<std::optional<torch::Tensor>, tc::OptRef<const Token>> diff(const VariableToken& var) override;

		};

	}
}