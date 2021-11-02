#pragma once

#include "../pch.hpp"

namespace expression {

	// Used to convert expression to RPT or postfix
	enum eTokenType {
		INVALID,
		NUMBER,
		VARIABLE,
		FUNCTION,
		OPERATOR
	};
	typedef std::uint8_t TokenTypeFlags;

	enum eOperatorAssociativity {
		LEFT,
		RIGHT
	};
	typedef std::uint8_t OperatorAssociativityFlags;

	enum eOperatorPrecedence {
		BASIC,
		ADDITION,
		MULTIPLICATION,
		EXPONENTIATION
	};
	typedef std::uint8_t OperatorPrecedenceFlags;


	constexpr char VARIABLE_START_CHARACTER = '@';

	using OperatorTuple = std::tuple<char, OperatorPrecedenceFlags, OperatorAssociativityFlags>;
	const std::vector<OperatorTuple> DEFAULT_OPERATORS =
	{
		{'+', eOperatorPrecedence::ADDITION,		eOperatorAssociativity::LEFT},
		{'-', eOperatorPrecedence::ADDITION,		eOperatorAssociativity::LEFT},
		{'*', eOperatorPrecedence::MULTIPLICATION,	eOperatorAssociativity::LEFT},
		{'/', eOperatorPrecedence::MULTIPLICATION,	eOperatorAssociativity::LEFT},
		{'^', eOperatorPrecedence::EXPONENTIATION,	eOperatorAssociativity::RIGHT},
	};

	struct Token {

		Token()
			: token_str(""), token_type(eTokenType::INVALID) {}

		Token(std::string tstr, TokenTypeFlags ttype)
			: token_str(tstr), token_type(ttype) {}

		std::string token_str;
		TokenTypeFlags token_type;
	};

	typedef std::function<std::tuple<std::string, Token>(const std::string&)> Tokenizer;



	class Shunter {
	public:

		Shunter(std::string& expression);

		// Optional
		void setOperators(std::vector<OperatorTuple> operators);
		// Optional, the current default number tokenizer only handles integers
		void setNumberTokenizer(Tokenizer numberTokenizer);
		// Optional
		void setVariableTokenizer(Tokenizer variableTokenizer);
		// Optional
		void setFunctionTokenizer(Tokenizer functionTokenizer);
		// Optional
		void setOperatorTokenizer(Tokenizer operatorTokenizer);

		std::deque<Token> operator()();

	private:

		int getOpPrecedence(Token t);
		int getOpAssociativity(Token t);

		std::tuple<std::string, Token> getNextToken(const std::string& str);

		std::deque<Token> shunt();

	private:

		std::string m_Expression;

		std::vector<OperatorTuple> m_Operators;

		Tokenizer m_NumberTokenizer;
		Tokenizer m_VariableTokenizer;
		Tokenizer m_FunctionTokenizer;
		Tokenizer m_OperatorTokenizer;

	};

}