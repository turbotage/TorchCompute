#pragma once

namespace tc {
	namespace expression {

		struct FixedTokens {
			enum {
				COMMA = (int)',',
				RIGHT_PAREN = (int)')',
				LEFT_PAREN = (int)'('
			};
		};

		struct TokenType {
			enum {
				NO_TOKEN,
				OPERATOR,
				UNARY_OPERATOR,
				BINARY_OPERATOR,
				FUNCTION,
				VARIABLE,
				NUMBER,
				ZERO,
				UNITY,
				LEFT_PAREN,
				RIGHT_PAREN,
				COMMA,
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

		class LeftParen : public Token {
		public:

			LeftParen() = default;

			LeftParen(const LeftParen&) = default;

			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;
		};

		class RightParen : public Token {
		public:

			RightParen() = default;

			RightParen(const RightParen&) = default;

			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;
		};

		class Comma : public Token {
		public:

			Comma() = default;

			Comma(const Comma&) = default;

			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;
		};

		class Number : public Token {
		public:

			Number();

			Number(const Number&) = default;

			Number(const std::string& numberstr, bool is_imaginary);

			const std::string name;
			const bool is_imaginary;
			const c10::complex<float> num;


			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;

			std::string get_full_name() const;
		};

		class Variable : public Token {
		public:

			Variable();

			Variable(const Variable&) = default;

			Variable(const std::string& name);

			const std::string name;

			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;

		};

		class Unity : public Token {
		public:

			Unity() = default;

			Unity(const Unity&) = default;

			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;
		};

		class Zero : public Token {
		public:

			Zero() = default;

			Zero(const Zero&) = default;

			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;

		};

		class Operator : public Token {
		public:

			Operator(const Operator& other);
			Operator(Operator&&) = default;

			Operator(std::int32_t id, std::int32_t precedence, bool is_left_associative);

			const std::int32_t id;
			const std::int32_t precedence;
			const bool is_left_associative;

			std::int32_t get_id() const override;

			std::int32_t get_token_type() const override;

			virtual std::int32_t get_operator_type() const = 0;

		};

		class UnaryOperator : public Operator {
		public:

			UnaryOperator(const UnaryOperator& other);
			UnaryOperator(UnaryOperator&&) = default;

			UnaryOperator(std::int32_t id, std::int32_t precedence, bool is_left_associative,
				const std::vector<tc::refw<expression::Token>>& allowed_left_tokens);

			const std::vector<tc::refw<expression::Token>> allowed_left_tokens;

			std::int32_t get_operator_type() const override;

		};

		class BinaryOperator : public Operator {
		public:

			BinaryOperator(const BinaryOperator& other);
			BinaryOperator(BinaryOperator&&) = default;

			BinaryOperator(std::int32_t id, std::int32_t precedence, bool is_left_associative,
				bool commutative = false, bool anti_commutative = false);

			const bool commutative = false;
			const bool anti_commutative = false;

			std::int32_t get_operator_type() const override;
		};

		class Function : public Token {
		public:

			Function(const Function& other);
			Function& operator=(const Function&) = default;

			Function(std::int32_t id, std::int32_t n_inputs, bool commutative = false);

			Function(std::int32_t id, std::int32_t n_inputs, bool commutative,
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