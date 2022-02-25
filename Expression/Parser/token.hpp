#pragma once

#include <complex>

namespace tc {
	namespace expression {

		struct FixedTokens {
			enum {
				COMMA_CHAR = (int)',',
				RIGHT_PAREN_CHAR = (int)')',
				LEFT_PAREN_CHAR = (int)'('
			};
		};

		struct TokenType {
			enum {
				NO_TOKEN_TYPE,
				OPERATOR_TYPE,
				UNARY_OPERATOR_TYPE,
				BINARY_OPERATOR_TYPE,
				FUNCTION_TYPE,
				VARIABLE_TYPE,
				NUMBER_TYPE,
				ZERO_TYPE,
				UNITY_TYPE,
				NEG_UNITY_TYPE,
				NAN_TYPE,
				LEFT_PAREN_TYPE,
				RIGHT_PAREN_TYPE,
				COMMA_TYPE,
			};
		};

		class Token {
		public:

			Token() = default;
			Token(const Token&) = default;

			virtual std::int32_t get_id() const = 0;

			virtual std::int32_t get_token_type() const = 0;

		};

		class NoToken : public Token {
		public:
			NoToken() = default;

			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;
		};

		class LeftParenToken : public Token {
		public:

			LeftParenToken() = default;

			LeftParenToken(const LeftParenToken&) = default;

			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;
		};

		class RightParenToken : public Token {
		public:

			RightParenToken() = default;

			RightParenToken(const RightParenToken&) = default;

			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;
		};

		class CommaToken : public Token {
		public:

			CommaToken() = default;

			CommaToken(const CommaToken&) = default;

			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;
		};

		class NumberToken : public Token {
		public:

			NumberToken();

			NumberToken(const NumberToken& other);

			NumberToken(const std::string& realnumberstr, bool is_imaginary);

			NumberToken(const std::string& realnumberstr, bool is_imaginary, const std::vector<int64_t>& sizes);

			NumberToken(float number, bool is_imaginary);

			NumberToken(float number, bool is_imaginary, const std::vector<int64_t>& sizes);

			NumberToken(std::complex<float> num, bool is_imaginary);

			NumberToken(std::complex<float> num, bool is_imaginary, const std::vector<int64_t>& sizes);

			NumberToken(const std::string& numberstr, std::complex<float> num, bool is_imaginary);

			NumberToken(const std::string& numberstr, std::complex<float> num, bool is_imaginary, const std::vector<int64_t>& sizes);

			std::string name;
			bool is_imaginary;
			std::complex<float> num;
			std::vector<int64_t> sizes;

			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;

			std::string get_full_name() const;
		};

		class VariableToken : public Token {
		public:

			VariableToken();

			VariableToken(const VariableToken&) = default;

			VariableToken(const std::string& name);

			const std::string name;

			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;

		};

		class ZeroToken : public Token {
		public:

			ZeroToken();

			ZeroToken(const std::vector<int64_t>& sizes);

			ZeroToken(const ZeroToken&) = default;

			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;

			std::vector<int64_t> sizes;

		};

		class UnityToken : public Token {
		public:

			UnityToken();

			UnityToken(const std::vector<int64_t>& sizes);

			UnityToken(const UnityToken&) = default;

			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;

			std::vector<int64_t> sizes;
		};

		class NegUnityToken : public Token {
		public:

			NegUnityToken();

			NegUnityToken(const std::vector<int64_t>& sizes);

			NegUnityToken(const NegUnityToken&) = default;

			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;

			std::vector<int64_t> sizes;
		};

		class NanToken : public Token {
		public:

			NanToken();

			NanToken(const std::vector<int64_t>& sizes);

			NanToken(const NanToken&) = default;

			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;

			std::vector<int64_t> sizes;

		};

		class OperatorToken : public Token {
		public:

			OperatorToken(const OperatorToken& other);
			OperatorToken(OperatorToken&&) = default;

			OperatorToken(std::int32_t id, std::int32_t precedence, bool is_left_associative);

			const std::int32_t id;
			const std::int32_t precedence;
			const bool is_left_associative;

			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;

			virtual std::int32_t get_operator_type() const = 0;

		};

		class UnaryOperatorToken : public OperatorToken {
		public:

			UnaryOperatorToken(const UnaryOperatorToken& other);
			UnaryOperatorToken(UnaryOperatorToken&&) = default;

			UnaryOperatorToken(std::int32_t id, std::int32_t precedence, bool is_left_associative,
				const std::vector<tc::refw<expression::Token>>& allowed_left_tokens);

			const std::vector<tc::refw<expression::Token>> allowed_left_tokens;

			std::int32_t get_operator_type() const override;

		};

		class BinaryOperatorToken : public OperatorToken {
		public:

			BinaryOperatorToken(const BinaryOperatorToken& other);
			BinaryOperatorToken(BinaryOperatorToken&&) = default;

			BinaryOperatorToken(std::int32_t id, std::int32_t precedence, bool is_left_associative,
				bool commutative = false, bool anti_commutative = false);

			BinaryOperatorToken(std::int32_t id, std::int32_t precedence, bool is_left_associative,
				bool commutative, bool anti_commutative,
				const std::vector<tc::refw<expression::Token>>& disallowed_left_tokens);

			const bool commutative = false;
			const bool anti_commutative = false;
			const std::vector<tc::refw<expression::Token>> disallowed_left_tokens;

			std::int32_t get_operator_type() const override;
		};

		class FunctionToken : public Token {
		public:

			FunctionToken(const FunctionToken& other);
			FunctionToken& operator=(const FunctionToken&) = default;

			FunctionToken(std::int32_t id, std::int32_t n_inputs, bool commutative = false);

			FunctionToken(std::int32_t id, std::int32_t n_inputs, bool commutative,
				const std::vector<std::vector<int>>& commutative_inputs,
				const std::vector<std::pair<int, int>>& anti_commutative_inputs);

			const std::int32_t id;
			const std::int32_t n_inputs;
			const bool commutative = false;
			const std::vector<std::vector<int>> commutative_inputs;
			const std::vector<std::pair<int, int>> anti_commutative_inputs;

			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;

		};

	}
}