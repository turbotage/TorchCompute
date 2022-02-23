#include "../pch.hpp"

#include "shunter.hpp"

std::deque<tc::expression::Token> tc::expression::Shunter::shunt(std::vector<std::unique_ptr<Token>>&& tokens)
{
	std::deque<std::unique_ptr<Token>> operator_stack;
	std::deque<std::unique_ptr<Token>> output;

	for (auto& tok : tokens) {

		switch (tok->get_token_type()) {
		case TokenType::NUMBER:
			output.emplace_back(std::move(tok));
			break;
		case TokenType::VARIABLE:
			output.emplace_back(std::move(tok));
			break;
		case TokenType::FUNCTION:
			operator_stack.emplace_back(std::move(tok));
			break;
		case TokenType::OPERATOR:

			break;
		}

	}
}
