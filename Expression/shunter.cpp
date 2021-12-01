#include "shunter.hpp"

#include <stack>
#include <algorithm>

#include "parsing.hpp"


expression::Shunter::Shunter(std::string& expression)
	: m_Expression(expression)
{
	m_Operators = DEFAULT_OPERATORS;

	// Variable Tokenizer
	m_VariableTokenizer = [](const std::string& str) {
		std::string token;
		token += str[0];
		ui32 i = 1;
		if (str[0] == VARIABLE_START_CHARACTER) {
			if (str.size() > 1) {
				if (std::isalpha(str[1])) {
					for (; i < str.size() && (std::isdigit(str[i]) || std::isalpha(str[i])); ++i)
						token += str[i];
					return std::make_tuple(str.substr(i), Token(token, eTokenType::VARIABLE));
				}
			}
		}

		return std::make_tuple(str, Token());
	};

	// Function Tokenizer
	m_FunctionTokenizer = [](const std::string& str) {
		std::string token;
		token += str[0];
		ui32 i = 1;
		if (std::isalpha(str[0])) {
			for (; i < str.size() && (std::isdigit(str[i]) || std::isalpha(str[i])); ++i)
				token += str[i];
			return std::make_tuple(str.substr(i), Token(token, eTokenType::FUNCTION));
		}

		return std::make_tuple(str, Token());
	};

	// Operator Tokenizer
	m_OperatorTokenizer = [this](const std::string& str) {
		std::string token;

		ui32 longestOpString = 0;
		for (auto& op : m_Operators) {
			ui32 opLength = std::get<0>(op).length();
			if (opLength > longestOpString)
				longestOpString = opLength;
		}

		for (ui32 i = 0; i < longestOpString; ++i) {
			token += str[0];
			for (auto& op : m_Operators) {
				if (token == std::get<0>(op)) {
					return std::make_tuple(str.substr(1), Token(token, eTokenType::OPERATOR));
				}
			}
		}

		return std::make_tuple(str, Token());
	};

}

void expression::Shunter::setOperators(std::vector<OperatorTuple> operators)
{
	m_Operators = operators;
}

void expression::Shunter::setVariableTokenizer(Tokenizer variableTokenizer)
{
	m_VariableTokenizer = variableTokenizer;
}

void expression::Shunter::setFunctionTokenizer(Tokenizer functionTokenizer)
{
	m_FunctionTokenizer = functionTokenizer;
}

void expression::Shunter::setOperatorTokenizer(Tokenizer operatorTokenizer)
{
	m_OperatorTokenizer = operatorTokenizer;
}

std::deque<expression::Token> expression::Shunter::operator()()
{
	return shunt();
}

int expression::Shunter::getOpPrecedence(Token t)
{
	for (auto& op : m_Operators) {
		if (std::get<0>(op) == t.token_str) {
			return std::get<1>(op);
		}
	}
	throw std::runtime_error("Tried to get precedence of non-existing operator");
}

int expression::Shunter::getOpAssociativity(Token t)
{
	for (auto& op : m_Operators) {
		if (std::get<0>(op) == t.token_str) {
			return std::get<2>(op);
		}
	}
	throw std::runtime_error("Tried to get precedence of non-existing operator");
}

std::tuple<std::string, expression::Token> expression::Shunter::getNextToken(const std::string& str)
{
	if (str.empty())
		return std::make_tuple(str, Token());

	std::string restr = str;
	Token retok;

	static std::vector<char> elem_ops = { '(', ')', ',' };
	for (auto& op : elem_ops) {
		if (op == str[0]) {
			return std::make_tuple(str.substr(1), Token(std::string(1, str[0]), eTokenType::OPERATOR));
		}
	}


	std::tie(restr, retok) = m_VariableTokenizer(restr);
	if (retok.token_type != eTokenType::INVALID)
		return std::make_tuple(restr, retok);

	std::tie(restr, retok) = m_FunctionTokenizer(restr);
	if (retok.token_type != eTokenType::INVALID)
		return std::make_tuple(restr, retok);

	std::tie(restr, retok) = m_OperatorTokenizer(restr);
	if (retok.token_type != eTokenType::INVALID)
		return std::make_tuple(restr, retok);

	return std::make_tuple(str, Token());
}

std::deque <expression::Token> expression::Shunter::shunt()
{
	std::string expr = m_Expression;

	std::stack<Token> operator_stack;
	std::deque<Token> output;

	Token token;

	// While there are tokens to be read
	while (true) {
		std::tie(expr, token) = getNextToken(expr);

		if (token.token_type == eTokenType::INVALID && expr != "")
			throw std::runtime_error("Incorrect expression");
		if (token.token_str == "")
			break;

		// Push variables onto output
		if (token.token_type == eTokenType::VARIABLE) {
			output.push_back(token);
			continue;
		}

		// Push functions onto the operator stack
		if (token.token_type == eTokenType::FUNCTION) {
			operator_stack.push(token);
			continue;
		}

		// Left parenthesis
		if (token.token_str == "(") {
			operator_stack.push(token);
			continue;
		}

		// Right parenthesis
		if (token.token_str == ")") {
			if (operator_stack.empty())
				throw std::runtime_error("missmatched parenthesis");

			Token top = operator_stack.top();
			while (top.token_str != "(") {
				output.push_back(top);
				operator_stack.pop();

				if (operator_stack.empty())
					throw std::runtime_error("missmatched parenthesis");

				top = operator_stack.top();
			}
			operator_stack.pop();

			if (!operator_stack.empty()) {
				top = operator_stack.top();
				if (top.token_type == eTokenType::FUNCTION) {
					output.push_back(top);
					operator_stack.pop();
				}
			}
			continue;
		}

		// Ignore the comma operator
		if (token.token_str == ",")
			continue;

		// Operators
		if (token.token_type == eTokenType::OPERATOR) {
			while (!operator_stack.empty()) {
				Token top = operator_stack.top();
				if (top.token_str != "(") {
					if (getOpPrecedence(top) > getOpPrecedence(token)) {
						output.push_back(top);
						operator_stack.pop();
						continue;
					}
					else if (getOpPrecedence(top) == getOpPrecedence(token)) {
						if (getOpAssociativity(token) == eOperatorAssociativity::LEFT) {
							output.push_back(top);
							operator_stack.pop();
							continue;
						}
					}
				}
				break;
			}
			operator_stack.push(token);
			continue;
		}

	}

	// Push remaining operators on operator-stack to output
	while (!operator_stack.empty()) {
		Token top = operator_stack.top();
		if (top.token_str == "(")
			throw std::runtime_error("missmatched parenthesis");
		output.push_back(top);
		operator_stack.pop();
	}

	std::reverse(output.begin(), output.end());

	return output;
}

