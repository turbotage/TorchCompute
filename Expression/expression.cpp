#include "../pch.hpp"

#include "expression.hpp"
#include "Parser/lexer.hpp"
#include "nodes.hpp"

tc::expression::Expression::Expression(const std::deque<std::unique_ptr<Token>>& tokens, const ExpressionCreationMap& creation_map,
	const FetcherMap& fetchers)
	: m_VariableFetchers(fetchers)
{
	std::vector<std::unique_ptr<Node>> nodes;

	for (auto& token : tokens) {
		auto creation_func = creation_map.at(token->get_id());
		creation_func(*token, m_VariableFetchers, nodes);
	}

	if (nodes.size() != 1)
		throw std::runtime_error("Expression construction failed, more than one node was left after creation_map usage");

	m_Children.push_back(std::move(nodes[0]));
}

tc::expression::tentok tc::expression::Expression::eval()
{
	return m_Children[0]->eval();
}

tc::expression::tentok tc::expression::Expression::diff(const VariableToken& var)
{
	return m_Children[0]->diff(var);
}

tc::expression::ExpressionCreationMap tc::expression::Expression::default_expression_creation_map()
{
	return ExpressionCreationMap{
		// Fixed Tokens
		{FixedIDs::UNITY_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				nodes.push_back(node_from_token(tok));
			}
		},
		{FixedIDs::NEG_UNITY_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				nodes.push_back(node_from_token(tok));
			}
		},
		{FixedIDs::ZERO_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				nodes.push_back(node_from_token(tok));
			}
		},
		{FixedIDs::NAN_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				nodes.push_back(node_from_token(tok));
			}
		},
		{FixedIDs::NUMBER_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				nodes.push_back(node_from_token(tok));
			}
		},
		{FixedIDs::VARIABLE_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				const VariableToken& vtok = static_cast<const VariableToken&>(tok);
				nodes.push_back(std::make_unique<VariableNode>(vtok, fetcher_map.at(vtok.name)));
			}
		},
		// Operators
		{DefaultOperatorIDs::NEG_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<NegNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{DefaultOperatorIDs::POW_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto rc = std::move(nodes.back());
				nodes.pop_back();
				auto lc = std::move(nodes.back());
				nodes.pop_back();

				nodes.push_back(std::make_unique<PowNode>(std::move(lc), std::move(rc)));
			}
		},
		{DefaultOperatorIDs::MUL_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto rc = std::move(nodes.back());
				nodes.pop_back();
				auto lc = std::move(nodes.back());
				nodes.pop_back();

				nodes.push_back(std::make_unique<MulNode>(std::move(lc), std::move(rc)));
			}
		},
		{DefaultOperatorIDs::DIV_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto rc = std::move(nodes.back());
				nodes.pop_back();
				auto lc = std::move(nodes.back());
				nodes.pop_back();

				nodes.push_back(std::make_unique<DivNode>(std::move(lc), std::move(rc)));
			}
		},
		{DefaultOperatorIDs::ADD_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto rc = std::move(nodes.back());
				nodes.pop_back();
				auto lc = std::move(nodes.back());
				nodes.pop_back();

				nodes.push_back(std::make_unique<AddNode>(std::move(lc), std::move(rc)));
			}
		},
		{DefaultOperatorIDs::SUB_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto rc = std::move(nodes.back());
				nodes.pop_back();
				auto lc = std::move(nodes.back());
				nodes.pop_back();

				nodes.push_back(std::make_unique<SubNode>(std::move(lc), std::move(rc)));
			}
		},
		// Functions
		// Binary
		{ DefaultFunctionIDs::POW_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto rc = std::move(nodes.back());
				nodes.pop_back();
				auto lc = std::move(nodes.back());
				nodes.pop_back();

				nodes.push_back(std::make_unique<PowNode>(std::move(lc), std::move(rc)));
			}
		},

		// Unary
		{ DefaultFunctionIDs::ABS_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<AbsNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::SQRT_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<SqrtNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::SQUARE_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<SquareNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::EXP_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<ExpNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::LOG_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<LogNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		// Trig
		{DefaultFunctionIDs::SIN_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<SinNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::COS_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<CosNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::TAN_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<TanNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::ASIN_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<AsinNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::ACOS_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<AcosNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::ATAN_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<AtanNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::SINH_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<SinhNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::COSH_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<CoshNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::TANH_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<TanhNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::ASINH_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<AsinhNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::ACOSH_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<AcoshNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},
		{ DefaultFunctionIDs::ATANH_ID,
		[](const Token& tok, const FetcherMap& fetcher_map, std::vector<std::unique_ptr<Node>>& nodes)
			{
				auto node = std::make_unique<AtanhNode>(std::move(nodes.back()));
				nodes.pop_back();
				nodes.push_back(std::move(node));
			}
		},

	};
}
