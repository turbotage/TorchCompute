#include "../compute.hpp"

int main() {
	using namespace tc::expression;
	
	torch::Tensor x = torch::rand({4,2});

	std::function<torch::Tensor()> xfetcher = [&x]() {
		return x;
	};

	FetcherMap map{ {"x", xfetcher} };

	VariableToken varx("x");

	LexContext context;
	context.variables.push_back(varx);
	LexContext context_copy = context;

	Lexer lexer(std::move(context));
	std::string expr = "1+x";

	std::vector<std::unique_ptr<Token>> toks = lexer.lex(expr);

	Shunter shunter;
	std::deque<std::unique_ptr<Token>> shunted_toks = shunter.shunt(std::move(toks));

	c10::IntArrayRef arr1{ 5,2 };
	c10::IntArrayRef arr2{ 1,1,2 };
	auto out1 = tc::tc_broadcast_shapes(arr1, arr2);
	std::cout << out1 << std::endl;

	std::vector<int64_t> arr3{ 5,2 };
	std::vector<int64_t> arr4{ 1,1,2 };
	auto out2 = tc::tc_broadcast_shapes(arr3, arr4);
	std::cout << out2 << std::endl;


	Expression expression(shunted_toks, Expression::default_expression_creation_map(), map);

	std::cout << "x: " << x << std::endl;
	std::cout << "eval: " << ": " << tentok_to_string(expression.eval()) << std::endl;
	std::cout << "diff: " << ": " << tentok_to_string(expression.diff(varx)) << std::endl;
	/*
	try {
		std::cout << expr << ": " << tentok_to_string(expression.eval()) << std::endl;
	}
	catch (c10::Error e) {
		std::cout << e.what() << std::endl;
	}
	*/

}
