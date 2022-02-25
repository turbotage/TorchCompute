#include "../../pch.hpp"

#include "shunter.hpp"

std::deque<std::unique_ptr<tc::expression::Token>> tc::expression::Shunter::shunt(std::vector<std::unique_ptr<Token>>&& tokens)
{
	auto toks = std::move(tokens);

	for (auto& tok : toks) {

		switch (tok->get_token_type()) {
		case TokenType::NUMBER_TYPE:
			m_Output.emplace_back(std::move(tok));
			break;
		case TokenType::VARIABLE_TYPE:
			m_Output.emplace_back(std::move(tok));
			break;
		case TokenType::FUNCTION_TYPE:
			m_OperatorStack.emplace_back(std::move(tok));
			break;
		case TokenType::OPERATOR_TYPE:
			handle_operator(dynamic_cast<OperatorToken&>(*tok));
			m_OperatorStack.emplace_back(std::move(tok));
			break;
		case TokenType::LEFT_PAREN_TYPE:
			m_OperatorStack.emplace_back(std::move(tok));
			break;
		case TokenType::RIGHT_PAREN_TYPE:
			handle_rparen();
			break;
		case TokenType::COMMA_TYPE:
			break;
		default:
			throw std::runtime_error("This token type should not be in list of lexed tokens to be shunted, type: " + 
				tok->get_token_type() + std::string("  id: ") + std::to_string(tok->get_id()));
		}
	}

	if (shift_until(LeftParenToken()))
		throw std::runtime_error("missmatched parenthesis");

	if (!m_OperatorStack.empty())
		throw std::runtime_error("operator stack should be empty after shunt");

	auto ret = std::move(m_Output);
	m_Output.clear();
	m_OperatorStack.clear();

	return ret;
}

void tc::expression::Shunter::handle_operator(const OperatorToken& op)
{
	while (!m_OperatorStack.empty()) {
		auto& ptok_back = m_OperatorStack.back();
		switch (ptok_back->get_token_type()) {
		case TokenType::LEFT_PAREN_TYPE:
			return;
			break;
		case TokenType::OPERATOR_TYPE:
			{
				OperatorToken& top_op = dynamic_cast<OperatorToken&>(*ptok_back);
				auto p = top_op.precedence;
				auto q = op.precedence;
				if ((p > q) || (p == q && op.is_left_associative)) {
					m_Output.emplace_back(std::move(ptok_back));
					m_OperatorStack.pop_back();
					continue;
				}
				return;
			}
			break;
		default:
			throw std::runtime_error(ptok_back->get_id() + " must not be on operator stack");
		}
	}
}

void tc::expression::Shunter::handle_rparen()
{
	if (!shift_until(LeftParenToken())) {
		throw std::runtime_error("missmatched parenthesis");
	}

	if (!m_OperatorStack.empty()) {
		auto& ptok_back = m_OperatorStack.back();
		if (ptok_back->get_token_type() == TokenType::FUNCTION_TYPE) {
			m_Output.emplace_back(std::move(ptok_back));
			m_OperatorStack.pop_back();
		}
	}
}

bool tc::expression::Shunter::shift_until(const Token& stop)
{
	int32_t stacksize = m_OperatorStack.size();
	for (int32_t i = 0; i < stacksize; ++i) {
		std::unique_ptr<Token> ptok = std::move(m_OperatorStack.back());
		m_OperatorStack.pop_back();

		if (ptok->get_id() == stop.get_id()) {
			return true;
		}

		m_Output.emplace_back(std::move(ptok));
	}
	return false;
}
