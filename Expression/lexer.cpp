#include "../pch.hpp"

#include "lexer.hpp"

tc::expression::Lexer::Lexer(LexContext&& lex_context)
	: m_LexContext(std::move(lex_context))
{

}

std::pair<std::string_view, tc::OptRef<tc::expression::UnaryOperator>> tc::expression::Lexer::begins_with_unary_operator(std::string_view expr)
{
	for (auto& uop : m_LexContext.unary_operators) {
		if (expr.rfind(uop.get_id(), 0) == 0) {
			const std::string& previous_token_id = m_LexedTokens.back().get().get_id();
			for (auto& allowed_op : uop.allowed_left_tokens) {
				if (allowed_op.get().get_id() == previous_token_id) {
					return std::make_pair(expr.substr(uop.get_id().length()), std::ref(uop));
				}
			}
			// The tokens id matched but previous token did not match any left allowed token, might for instance be another unary operator
			return std::make_pair(expr, std::nullopt);
		}
	}
	// No match
	return std::make_pair(expr, std::nullopt);
}

std::pair<std::string_view, tc::OptRef<tc::expression::BinaryOperator>> tc::expression::Lexer::begins_with_binary_operator(std::string_view expr)
{
	for (auto& bop : m_LexContext.binary_operators) {
		if (expr.rfind(bop.get_id(), 0) == 0) {
			return std::make_pair(expr.substr(bop.get_id().length()), std::ref(bop));
		}
	}
	// No match
	return std::make_pair(expr, std::nullopt);
}

std::pair<std::string_view, tc::OptRef<tc::expression::Function>> tc::expression::Lexer::begins_with_function(std::string_view expr)
{
	for (auto& func : m_LexContext.functions) {
		if (expr.rfind(func.get_id(), 0) == 0) {

		}
	}
}
