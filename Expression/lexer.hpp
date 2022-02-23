#pragma once

#include "token.hpp"

namespace tc {
	namespace expression {

		struct LexContext {

			LexContext();

			LexContext(LexContext&) = delete;
			LexContext& operator=(LexContext&) = delete;

			LexContext(LexContext&&) = default;

			NoToken						no_token;
			LeftParen					left_paren;
			RightParen					right_paren;

			std::vector<UnaryOperator>	unary_operators;
			std::vector<BinaryOperator> binary_operators;
			std::vector<Function>		functions;
			std::vector<Variable>		variables;

		};

		class Lexer {
		public:

			Lexer(LexContext&& lex_context);

			std::vector<std::unique_ptr<Token>> lex(std::string expression) const;

		private:
			
			
			
			std::pair<std::string_view, std::unique_ptr<tc::expression::Token>> lex_token(std::string_view expr,
				const std::vector<std::unique_ptr<Token>>& lexed_tokens) const;
			


			std::pair<std::string_view, std::optional<LeftParen>> begins_with_left_paren(std::string_view expr) const;

			std::pair<std::string_view, std::optional<RightParen>> begins_with_right_paren(std::string_view expr) const;

			std::pair<std::string_view, std::optional<Comma>> begins_with_comma(std::string_view expr) const;

			std::pair<std::string_view, std::optional<UnaryOperator>> begins_with_unary_operator(std::string_view expr, 
				const std::vector<std::unique_ptr<Token>>& lexed_tokens) const;

			std::pair<std::string_view, std::optional<BinaryOperator>> begins_with_binary_operator(std::string_view expr) const;

			std::pair<std::string_view, std::optional<Function>> begins_with_function(std::string_view expr) const;

			std::pair<std::string_view, std::optional<Variable>> begins_with_variable(std::string_view expr) const;

			std::pair<std::string_view, std::optional<Number>> begins_with_numberstr(std::string_view expr) const;

			std::optional<Zero> begins_with_zero(const Number& num) const;

			std::optional<Unity> begins_with_unity(const Number& num) const;

		private:

			LexContext m_LexContext;

		};
		
		struct DefaultOperatorPrecedence {
			enum {
				NEG = 10,
				POW = 10,
				MUL = 5,
				DIV = 5,
				ADD = 3,
				SUB = 3,
			};
		};

		struct DefaultOperatorChars {
			enum {
				NEG = (int)'-',
				POW = (int)'^',
				MUL = (int)'*',
				DIV = (int)'/',
				ADD = (int)'+',
				SUB = (int)'-',
			};
		};

		struct eDefaultFunctions {
			enum {
				// Trigonometric
				SIN,
				COS,
				//
				EXP,
				LOG
			};
		};

	}
}