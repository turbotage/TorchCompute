#pragma once

#include "../pch.hpp"
#include "TokenAlgebra/token_algebra.hpp"

namespace tc {
	namespace expression {


		using FetcherFunc = std::function<torch::Tensor()>;
		using FetcherFuncRef = tc::refw<const std::function<torch::Tensor()>>;

		using tentok = std::pair<std::optional<torch::Tensor>, tc::OptUPtr<NumberBaseToken>>;

		std::string tentok_to_string(const tentok& in);

		tentok tentok_from_number(float a);
		tentok tentok_from_zero();
		tentok tentok_from_unity();
		tentok tentok_from_negunity();
		tentok tentok_from_nan();

		std::unique_ptr<NumberBaseToken> copy_token(const Token& tok);

		torch::Tensor tensor_from_tentok(const tentok& in, torch::Device& device);

		class Node {
		public:

			Node() = default;

			Node(std::unique_ptr<NumberBaseToken> base_token);

			virtual tentok eval() = 0;

			virtual std::unique_ptr<Node> evalnode() = 0;

			virtual tentok diff(const VariableToken& var) = 0;

			virtual std::unique_ptr<Node> diffnode(const VariableToken& var) = 0;

		public:
			std::vector<std::unique_ptr<Node>> m_Children;

			std::unique_ptr<NumberBaseToken> m_pToken;
		};

		std::unique_ptr<Node> node_from_token(const Token& tok);

		std::unique_ptr<Node> node_from_pair(const tentok& pair);

		class TokenNode : public Node {
		public:

			TokenNode(const Token& tok);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		class TensorNode : public Node {
		public:

			TensorNode(const torch::Tensor& tensor);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		private:
			torch::Tensor m_Tensor;
		};

		class VariableNode : public Node {
		public:

			VariableNode(const VariableToken& token, FetcherFuncRef variable_fetcher);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		private:
			const VariableToken& m_VarToken;
			FetcherFuncRef m_VariableFetcher; // fetches the tensor
		};

		// Operators

		tentok operator-(const tentok& a);

		class NegNode : public Node {
		public:

			NegNode(std::unique_ptr<Node> child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		tentok operator*(const tentok& a, const tentok& b);

		class MulNode : public Node {
		public:

			MulNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		tentok operator/(const tentok& a, const tentok& b);

		class DivNode : public Node {
		public:

			DivNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;
		};

		tentok operator+(const tentok& a, const tentok& b);

		class AddNode : public Node {
		public:

			AddNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		tentok operator-(const tentok& a, const tentok& b);

		class SubNode : public Node {
		public:

			SubNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		tentok pow(const tentok& a, const tentok& b);

		class PowNode : public Node {
		public:

			PowNode(std::unique_ptr<Node> left_child, std::unique_ptr<Node> right_child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};
		

		// Unary

		tentok abs(const tentok& a);

		class AbsNode : public Node {
		public:

			AbsNode(std::unique_ptr<Node> child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		tentok sqrt(const tentok& a);

		class SqrtNode : public Node {
		public:

			SqrtNode(std::unique_ptr<Node> child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		tentok square(const tentok& a);

		class SquareNode : public Node {
		public:

			SquareNode(std::unique_ptr<Node> child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		tentok exp(const tentok& a);

		class ExpNode : public Node {
		public:

			ExpNode(std::unique_ptr<Node> child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		tentok log(const tentok& a);

		class LogNode : public Node {
		public:

			LogNode(std::unique_ptr<Node> child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		// Trig

		tentok sin(const tentok& a);

		class SinNode : public Node {
		public:

			SinNode(std::unique_ptr<Node> child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};
		
		tentok cos(const tentok& a);

		class CosNode : public Node {
		public:

			CosNode(std::unique_ptr<Node> child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		tentok tan(const tentok& a);

		class TanNode : public Node {
		public:

			TanNode(std::unique_ptr<Node> child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		tentok asin(const tentok& a);

		class AsinNode : public Node {
		public:

			AsinNode(std::unique_ptr<Node> child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		tentok acos(const tentok& a);

		class AcosNode : public Node {
		public:

			AcosNode(std::unique_ptr<Node> child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		tentok atan(const tentok& a);

		class AtanNode : public Node {
		public:

			AtanNode(std::unique_ptr<Node> child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		tentok sinh(const tentok& a);

		class SinhNode : public Node {
		public:

			SinhNode(std::unique_ptr<Node> child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		tentok cosh(const tentok& a);

		class CoshNode : public Node {
		public:

			CoshNode(std::unique_ptr<Node> child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		tentok tanh(const tentok& a);

		class TanhNode : public Node {
		public:

			TanhNode(std::unique_ptr<Node> child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		tentok asinh(const tentok& a);

		class AsinhNode : public Node {
		public:

			AsinhNode(std::unique_ptr<Node> child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		tentok acosh(const tentok& a);

		class AcoshNode : public Node {
		public:

			AcoshNode(std::unique_ptr<Node> child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

		tentok atanh(const tentok& a);

		class AtanhNode : public Node {
		public:

			AtanhNode(std::unique_ptr<Node> child);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

		};

	}
}