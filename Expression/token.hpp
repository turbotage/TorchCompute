#pragma once

namespace tc {
	namespace expression {

		enum eTokenType {
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

		class Token {
		public:

			virtual const std::string& get_id() const = 0;

			virtual std::int32_t get_token_type() const = 0;

		};

		class NoToken : public Token {
		public:
			const std::string& get_id() const override;

			std::int32_t get_token_type() const override;
		};

		class Operator : public Token {
		public:

			Operator(const std::string& id, std::int32_t precedence, bool is_left_associative);

			const std::string id;
			const std::int32_t precedence;
			const bool is_left_associative;

			const std::string& get_id() const override;

			std::int32_t get_token_type() const override;

			virtual std::int32_t get_operator_type() const = 0;

		};

		class UnaryOperator : public Operator {
		public:

			UnaryOperator(const std::string& id, std::int32_t precedence, bool is_left_associative,
				const std::vector<const tc::refw<expression::Token>>& allowed_left_tokens);

			const std::vector<const tc::refw<expression::Token>> allowed_left_tokens;

			std::int32_t get_operator_type() const override;

		};

		class BinaryOperator : public Operator {
		public:

			BinaryOperator(const std::string& id, std::int32_t precedence, bool is_left_associative,
				bool commutative = false, bool anti_commutative = false);

			const bool commutative = false;
			const bool anti_commutative = false;

			std::int32_t get_operator_type() const override;
		};

		class Function : public Token {
		public:
			const std::string id;
			const std::int32_t n_inputs;
			const bool commutative = false;
			std::vector<std::vector<int>> commutative_inputs;
			std::vector<std::pair<int, int>> anti_commutative_inputs;

			const std::string& get_id() const override;

			std::int32_t get_token_type() const override;

		};

		class Variable : public Token {
		public:
			const std::string id;

			const std::string& get_id() const override;

			std::int32_t get_token_type() const override;

		};

		class Number : public Token {
		public:
			const std::string id;

			const std::string& get_id() const override;

			std::int32_t get_token_type() const override;

		};

		class Zero : public Token {
		public:

			const std::string& get_id() const override;

			std::int32_t get_token_type() const override;

		};

		class Unity : public Token {
		public:

			const std::string& get_id() const override;

			std::int32_t get_token_type() const override;
		};

		class LeftParen : public Token {
		public:
			const std::string& get_id() const override;

			std::int32_t get_token_type() const override;
		};

		class RightParen : public Token {
		public:
			const std::string& get_id() const override;

			std::int32_t get_token_type() const override;
		};

		class Comma : public Token {
			const std::string& get_id() const override;

			std::int32_t get_token_type() const override;
		};

	}
}