#pragma once

#include "token.hpp"
#include "nodes.hpp"

namespace tc {
	namespace expression {
		
		using FetcherMap = std::unordered_map<std::string, FetcherFunc>;

		using ExpressionCreationMap = std::unordered_map<int32_t, std::function<void(
			const Token&, const FetcherMap&, std::vector<std::unique_ptr<Node>>&)>>;


		class Expression : public Node {
		public:

			Expression(std::unique_ptr<Node> root_child, const FetcherMap& fetchers);

			Expression(const std::deque<std::unique_ptr<Token>>& tokens, const ExpressionCreationMap& creation_map,
				const FetcherMap& fetchers);

			tentok eval() override;

			std::unique_ptr<Node> evalnode() override;

			std::unique_ptr<Expression> exprevalnode();

			tentok diff(const VariableToken& var) override;

			std::unique_ptr<Node> diffnode(const VariableToken& var) override;

			std::unique_ptr<Expression> exprdiffnode(const VariableToken& var);

			static ExpressionCreationMap default_expression_creation_map();

		private:

			const FetcherMap& m_VariableFetchers;

		};

	}
}


