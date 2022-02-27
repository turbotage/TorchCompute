#pragma once

#include "token.hpp"
#include "nodes.hpp"

namespace tc {
	namespace expression {
		
		using FetcherMap = std::unordered_map<std::string, std::function<torch::Tensor()>>;

		using ExpressionCreationMap = std::unordered_map<int32_t, std::function<void(
			const Token&, const FetcherMap&, std::vector<std::unique_ptr<Node>>&)>>;


		class Expression : protected Node {
		public:

			Expression(const std::deque<std::unique_ptr<Token>>& tokens, const ExpressionCreationMap& creation_map,
				const FetcherMap& fetchers);

			static ExpressionCreationMap default_expression_creation_map();

			virtual tentok eval();

			virtual tentok diff(const VariableToken& var);

		private:

			std::unordered_map<std::string, std::function<torch::Tensor()>> m_VariableFetchers;

		};

	}
}


