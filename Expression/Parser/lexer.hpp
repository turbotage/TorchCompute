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
			LeftParenToken				left_paren;
			RightParenToken				right_paren;
			CommaToken					comma;
			UnityToken					unity;
			NegUnityToken				neg_unity_token;
			ZeroToken					zero;
			NumberToken					number;
			VariableToken				variable;


			std::vector<UnaryOperatorToken>	unary_operators;
			std::vector<BinaryOperatorToken> binary_operators;
			std::vector<FunctionToken>		functions;
			std::vector<VariableToken>		variables;

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
			


			std::pair<std::string_view, std::optional<LeftParenToken>> begins_with_left_paren(std::string_view expr) const;

			std::pair<std::string_view, std::optional<RightParenToken>> begins_with_right_paren(std::string_view expr) const;

			std::pair<std::string_view, std::optional<CommaToken>> begins_with_comma(std::string_view expr) const;

			std::pair<std::string_view, std::optional<UnaryOperatorToken>> begins_with_unary_operator(std::string_view expr, 
				const std::vector<std::unique_ptr<Token>>& lexed_tokens) const;

			std::pair<std::string_view, std::optional<BinaryOperatorToken>> begins_with_binary_operator(std::string_view expr,
				const std::vector<std::unique_ptr<Token>>& lexed_tokens) const;

			std::pair<std::string_view, std::optional<FunctionToken>> begins_with_function(std::string_view expr) const;

			std::pair<std::string_view, std::optional<VariableToken>> begins_with_variable(std::string_view expr) const;

			std::pair<std::string_view, std::optional<NumberToken>> begins_with_numberstr(std::string_view expr) const;

			std::optional<ZeroToken> begins_with_zero(const NumberToken& num) const;

			std::optional<UnityToken> begins_with_unity(const NumberToken& num) const;

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
				NEG_UNITY,
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
			{FixedIDs::NEG_UNITY, "NEG_UNITY"},
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
				// FunctionTokens
				SIN = DefaultOperatorIDs::SUB + 1,
				COS,
				TAN,
				EXP,
				LOG,
				POW,
			};
		};

		static std::unordered_map<int32_t, std::string> DEFAULT_FUNCTION_MAPS = {
			// FunctionTokens
			{DefaultFunctionIDs::SIN, "sin"},
			{DefaultFunctionIDs::COS, "cos"},
			{DefaultFunctionIDs::TAN, "tan"},
			{DefaultFunctionIDs::EXP, "exp"},
			{DefaultFunctionIDs::LOG, "log"},
			{DefaultFunctionIDs::POW, "pow"},
		};

	}
}