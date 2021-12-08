#include "expression.hpp"

#include "lexer.hpp"
#include "shunter.hpp"

expression::ExpressionGraph::ExpressionGraph(std::string expression)
{
	m_Expression = expression;

	expression::Lexer lex;

	// Lex expression, get variable map and variable setter map
	std::tie(m_Numbers, m_LexedExpression) = lex(expression);

	// Run shunting yard algorithm, get token stack in RPT
	expression::Shunter shunter(m_LexedExpression);
	std::stack<Token> token_stack(shunter());

	std::stack<std::unique_ptr<Node>> node_stack;

	// Build node stack from token stack
	int num_counter = 0;
	while (!token_stack.empty()) {
		Token top = token_stack.top(); token_stack.pop();
		const std::string& tstr = top.token_str;
		if (top.token_type == eTokenType::VARIABLE) {
			// The variable is a number
			auto numit = m_Numbers.find(tstr);
			if (numit != m_Numbers.end()) {
				handle_number(tstr, m_Numbers[tstr], node_stack);
				continue;
			}

			auto varpair = check_variable_sign(tstr);

			if (varpair.second) { // This was a negated variable
				std::string varname = varpair.first;
				auto& varfetcherref = m_VariableFetchers[varname];

				std::function<torch::Tensor()> vf = [&varfetcherref]() { return -varfetcherref(); };

				auto& varfetcher = m_VariableNegates.emplace_front(std::move(vf));

				handle_variable(
					"-" + varname,
					varfetcher,
					node_stack);
			}
			else {
				handle_variable(
					tstr,
					m_VariableFetchers[tstr],
					node_stack);
			}

			// The variable is a variable
			continue;
		}
		else if (top.token_type == eTokenType::FUNCTION) {
			handle_function(tstr, node_stack);
			continue;
		}
		else if (top.token_type == eTokenType::OPERATOR) {
			handle_operation(tstr, node_stack);
			continue;
		}
		else {
			throw std::runtime_error("Not implemented token type in expression graph constructed");
		}
	}

	if (node_stack.size() != 1)
		throw std::runtime_error("Top node not unique");

	m_pRootNode = std::move(node_stack.top());
	node_stack.pop();
}

std::function<torch::Tensor()> expression::ExpressionGraph::getFunc()
{
	return m_pRootNode->runner();
}

void expression::ExpressionGraph::setVariableFetcher(std::string var_name, const std::function<torch::Tensor()>& var_fetcher)
{
	m_VariableFetchers[var_name] = var_fetcher;
}

void expression::ExpressionGraph::to(torch::Device device)
{
	for (auto& t : m_Numbers) {
		t.second.to(device);
	}
}

std::string expression::ExpressionGraph::getReadableTree()
{
	return readableNode(m_pRootNode.get(), 1);
}

std::pair<std::string, bool> expression::ExpressionGraph::check_variable_sign(const std::string& varname)
{
	bool neg = false;
	std::string ret = varname;
	auto n = varname.find("NEG_");
	if (n != std::string::npos) {
		ret = ret.erase(n, 4);
		neg = true;
	}
	return std::make_pair(ret, neg);
}







// TO CHANGE BEHAVIOUR OF OPERATORS AND ADD FUNCTIONS ONLY CHANGE CODE BELOW THIS LINE






void expression::ExpressionGraph::handle_number(std::string numid, torch::Tensor& var, std::stack<std::unique_ptr<Node>>& node_stack) {
	std::unique_ptr<Node> ptr = std::make_unique<NumberNode>(numid, var);
	node_stack.push(std::move(ptr));
}

void expression::ExpressionGraph::handle_variable(std::string varid, const std::function<torch::Tensor()>& var_fetcher, std::stack<std::unique_ptr<Node>>& node_stack)
{
	std::unique_ptr<Node> ptr = std::make_unique<VariableNode>(varid, var_fetcher);
	node_stack.push(std::move(ptr));
}

void expression::ExpressionGraph::handle_function(std::string funcid, std::stack<std::unique_ptr<Node>>& node_stack)
{
	std::unique_ptr<Node> ptr;
	if (funcid == "sin") { // Trigonometry
		ptr = std::make_unique<SinNode>(std::move(node_stack.top()));
		node_stack.pop();
	}
	else if (funcid == "cos") {
		ptr = std::make_unique<CosNode>(std::move(node_stack.top()));
		node_stack.pop();
	}
	else if (funcid == "tan") {
		ptr = std::make_unique<TanNode>(std::move(node_stack.top()));
		node_stack.pop();
	}
	else if (funcid == "sinh") {
		ptr = std::make_unique<SinhNode>(std::move(node_stack.top()));
		node_stack.pop();
	}
	else if (funcid == "cosh") {
		ptr = std::make_unique<CoshNode>(std::move(node_stack.top()));
		node_stack.pop();
	}
	else if (funcid == "tanh") {
		ptr = std::make_unique<TanhNode>(std::move(node_stack.top()));
		node_stack.pop();
	}
	else if (funcid == "asin") {
		ptr = std::make_unique<AsinNode>(std::move(node_stack.top()));
		node_stack.pop();
	}
	else if (funcid == "acos") {
		ptr = std::make_unique<AcosNode>(std::move(node_stack.top()));
		node_stack.pop();
	}
	else if (funcid == "atan") {
		ptr = std::make_unique<AtanNode>(std::move(node_stack.top()));
		node_stack.pop();
	}
	else if (funcid == "atan2") {
		auto x = std::move(node_stack.top()); node_stack.pop();
		ptr = std::make_unique<Atan2Node>(std::move(x), std::move(node_stack.top()));
		node_stack.pop();
	}
	else if (funcid == "asinh") {
		ptr = std::make_unique<AsinhNode>(std::move(node_stack.top()));
		node_stack.pop();
	}
	else if (funcid == "acosh") {
		ptr = std::make_unique<AcoshNode>(std::move(node_stack.top()));
		node_stack.pop();
	}
	else if (funcid == "atanh") {
		ptr = std::make_unique<AtanhNode>(std::move(node_stack.top()));
		node_stack.pop();
	}
	else if (funcid == "exp") { // Powers, Exp, Log
		ptr = std::make_unique<ExpNode>(std::move(node_stack.top()));
		node_stack.pop();
	}
	else if (funcid == "pow") {
		auto base = std::move(node_stack.top()); node_stack.pop();
		ptr = std::make_unique<PowNode>(std::move(base), std::move(node_stack.top()));
		node_stack.pop();
	}
	else if (funcid == "log") {
		ptr = std::make_unique<LogNode>(std::move(node_stack.top()));
		node_stack.pop();
	}
	else if (funcid == "log10") {
		ptr = std::make_unique<Log10Node>(std::move(node_stack.top()));
		node_stack.pop();
	}
	else {
		throw std::runtime_error("Not implemented function");
	}

	node_stack.push(std::move(ptr));
}

void expression::ExpressionGraph::handle_operation(std::string opid, std::stack<std::unique_ptr<Node>>& node_stack)
{
	std::unique_ptr<Node> ptr;
	if (opid == "+") {
		int i;
		auto right = std::move(node_stack.top()); node_stack.pop();
		auto left = std::move(node_stack.top()); node_stack.pop();
		ptr = std::make_unique<AddNode>(std::move(left), std::move(right));
	}
	else if (opid == "-") {
		auto right = std::move(node_stack.top()); node_stack.pop();
		auto left = std::move(node_stack.top()); node_stack.pop();
		ptr = std::make_unique<SubNode>(std::move(left), std::move(right));
	}
	else if (opid == "*") {
		auto right = std::move(node_stack.top()); node_stack.pop();
		auto left = std::move(node_stack.top()); node_stack.pop();
		ptr = std::make_unique<MulNode>(std::move(left), std::move(right));
	}
	else if (opid == "/") {
		auto right = std::move(node_stack.top()); node_stack.pop();
		auto left = std::move(node_stack.top()); node_stack.pop();
		ptr = std::make_unique<DivNode>(std::move(left), std::move(right));
	}
	else if (opid == "^") {
		auto right = std::move(node_stack.top()); node_stack.pop();
		auto left = std::move(node_stack.top()); node_stack.pop();
		ptr = std::make_unique<PowNode>(std::move(left), std::move(right));
	}
	else {
		throw std::runtime_error("Not implemented operator");
	}

	node_stack.push(std::move(ptr));
}
