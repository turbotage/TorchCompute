#pragma once

#include "token.hpp"

namespace tc {
	namespace expression {

		struct LexContext {

			LexContext() = default;

			LexContext(LexContext&) = delete;
			LexContext& operator=(LexContext&) = delete;

			LexContext(LexContext&&) = default;

			std::vector<UnaryOperator>	unary_operators;
			std::vector<BinaryOperator> binary_operators;
			std::vector<Function>		functions;
			std::vector<std::string>	variables;

		};

		class Lexer {
		public:

			Lexer(LexContext&& lex_context);

		private:

			std::pair<std::string_view, tc::OptRef<UnaryOperator>> begins_with_unary_operator(std::string_view expr);

			std::pair<std::string_view, tc::OptRef<BinaryOperator>> begins_with_binary_operator(std::string_view expr);

			std::pair<std::string_view, tc::OptRef<Function>> begins_with_function(std::string_view expr);

		private:

			LexContext m_LexContext;

			std::vector<tc::refw<Token>> m_LexedTokens;

		};

	}
}