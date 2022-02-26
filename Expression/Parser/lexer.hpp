#pragma once

#include "../token.hpp"

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
			NanToken					nan_token;
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

		/*
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
		*/
		
		struct FixedIDs {
			enum {
				// Fixed Tokens
				NO_TOKEN_ID,
				LEFT_PAREN_ID,
				RIGHT_PAREN_ID,
				COMMA_ID,
				UNITY_ID,
				NEG_UNITY_ID,
				ZERO_ID,
				NAN_ID,
				NUMBER_ID,
				VARIABLE_ID
			};
		};

		static std::unordered_map<int32_t, std::string> FIXED_ID_MAPS = {
			// Fixed Tokens
			{FixedIDs::NO_TOKEN_ID, "NO_TOKEN"},
			{FixedIDs::LEFT_PAREN_ID, "("},
			{FixedIDs::RIGHT_PAREN_ID, ")"},
			{FixedIDs::COMMA_ID, ","},
			{FixedIDs::UNITY_ID, "UNITY"},
			{FixedIDs::NEG_UNITY_ID, "NEG_UNITY"},
			{FixedIDs::ZERO_ID, "ZERO"},
			{FixedIDs::NAN_ID, "NAN"},
			{FixedIDs::NUMBER_ID, "NUMBER"},
			{FixedIDs::VARIABLE_ID, "VARIABLE"},
		};

		struct DefaultOperatorIDs {
			enum {
				// Operators
				NEG_ID = FixedIDs::VARIABLE_ID + 1,
				POW_ID,
				MUL_ID,
				DIV_ID,
				ADD_ID,
				SUB_ID,
			};
		};

		static std::unordered_map<int32_t, std::string> DEFAULT_OPERATOR_MAPS = {
			// Operators
			{DefaultOperatorIDs::NEG_ID, "-"},
			{DefaultOperatorIDs::POW_ID, "^"},
			{DefaultOperatorIDs::MUL_ID, "*"},
			{DefaultOperatorIDs::DIV_ID, "/"},
			{DefaultOperatorIDs::ADD_ID, "+"},
			{DefaultOperatorIDs::SUB_ID, "-"},
		};

		struct DefaultFunctionIDs {
			enum {
				// FunctionTokens
				SIN_ID = DefaultOperatorIDs::SUB_ID + 1,
				COS_ID,
				TAN_ID,
				EXP_ID,
				LOG_ID,
				POW_ID,
			};
		};

		static std::unordered_map<int32_t, std::string> DEFAULT_FUNCTION_MAPS = {
			// FunctionTokens
			{DefaultFunctionIDs::SIN_ID, "sin"},
			{DefaultFunctionIDs::COS_ID, "cos"},
			{DefaultFunctionIDs::TAN_ID, "tan"},
			{DefaultFunctionIDs::EXP_ID, "exp"},
			{DefaultFunctionIDs::LOG_ID, "log"},
			{DefaultFunctionIDs::POW_ID, "pow"},
		};

	}
}