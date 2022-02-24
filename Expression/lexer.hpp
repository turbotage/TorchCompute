#pragma once

#include "token.hpp"

namespace tc {
	namespace expression {

		struct LexContext {

			LexContext();

			LexContext(LexContext&) = default;
			LexContext& operator=(LexContext&) = default;

			LexContext(LexContext&&) = default;

			// Fixed tokens
			NoToken						no_token;
			LeftParen					left_paren;
			RightParen					right_paren;
			Comma						comma;
			Unity						unity;
			Zero						zero;
			Number						number;
			Variable					variable;


			std::vector<UnaryOperator>	unary_operators;
			std::vector<BinaryOperator> binary_operators;
			std::vector<Function>		functions;
			std::vector<Variable>		variables;

			std::unordered_map<int32_t, std::string> operator_id_name_map;
			std::unordered_map<int32_t, std::string> function_id_name_map;



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
		
		struct FixedIDs {
			enum {
				// Fixed Tokens
				NO_TOKEN,
				LEFT_PAREN,
				RIGHT_PAREN,
				COMMA,
				UNITY,
				ZERO,
				NUMBER,
				VARIABLE
			};
		};

		static std::unordered_map<int32_t, std::string> FIXED_ID_MAPS = {
			// Fixed Tokens
			{FixedIDs::NO_TOKEN, "NO_TOKEN"},
			{FixedIDs::LEFT_PAREN, "("},
			{FixedIDs::RIGHT_PAREN, ")"},
			{FixedIDs::COMMA, ","},
			{FixedIDs::UNITY, "UNITY"},
			{FixedIDs::ZERO, "ZERO"},
			{FixedIDs::NUMBER, "NUMBER"},
			{FixedIDs::VARIABLE, "VARIABLE"},
		};

		struct DefaultOperatorIDs {
			enum {
				// Operators
				NEG = FixedIDs::VARIABLE + 1,
				POW,
				MUL,
				DIV,
				ADD,
				SUB,
			};
		};

		static std::unordered_map<int32_t, std::string> DEFAULT_OPERATOR_MAPS = {
			// Operators
			{DefaultOperatorIDs::NEG, "-"},
			{DefaultOperatorIDs::POW, "^"},
			{DefaultOperatorIDs::MUL, "*"},
			{DefaultOperatorIDs::DIV, "/"},
			{DefaultOperatorIDs::ADD, "+"},
			{DefaultOperatorIDs::SUB, "-"},
		};

		struct DefaultFunctionIDs {
			enum {
				// Functions
				SIN = DefaultOperatorIDs::SUB + 1,
				COS,
				TAN,
				EXP,
				LOG,
				POW,
			};
		};

		static std::unordered_map<int32_t, std::string> DEFAULT_FUNCTION_MAPS = {
			// Functions
			{DefaultFunctionIDs::SIN, "sin"},
			{DefaultFunctionIDs::COS, "cos"},
			{DefaultFunctionIDs::TAN, "tan"},
			{DefaultFunctionIDs::EXP, "exp"},
			{DefaultFunctionIDs::LOG, "log"},
			{DefaultFunctionIDs::POW, "pow"},
		};

	}
}