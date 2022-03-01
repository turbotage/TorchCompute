#include "../compute.hpp"

void test_values() {
	using namespace tc::expression;

	int64_t nprob = 3;
	int64_t nparam = 2;
	int64_t ndata = 4;
	torch::Tensor x = torch::rand({ nprob,nparam });
	torch::Tensor b = torch::rand({ 1,ndata });

	std::function<torch::Tensor()> xfetcher = [&x]() {
		//std::cout << "xsize: " << x.select(1, 0).unsqueeze(-1).sizes() << std::endl;
		return x.select(1, 0).unsqueeze(-1);
	};

	std::function<torch::Tensor()> yfetcher = [&x]() {
		//std::cout << "ysize: " << x.select(1, 1).unsqueeze(-1).sizes() << std::endl;
		return x.select(1, 1).unsqueeze(-1);
	};

	std::function<torch::Tensor()> bfetcher = [&b]() {
		//std::cout << "bsize: " << b.sizes() << std::endl;
		return b;
	};

	FetcherMap map{ {"x", xfetcher}, {"y", yfetcher}, {"b", bfetcher} };

	VariableToken varx("x");
	VariableToken vary("y");
	VariableToken varb("b");

	LexContext context;
	context.variables.push_back(varx);
	context.variables.push_back(vary);
	context.variables.push_back(varb);
	LexContext context_copy = context;

	Lexer lexer(std::move(context));
	std::string expr = "1+x-x";

	std::vector<std::unique_ptr<Token>> toks = lexer.lex(expr);

	Shunter shunter;
	std::deque<std::unique_ptr<Token>> shunted_toks = shunter.shunt(std::move(toks));


	Expression expression(shunted_toks, Expression::default_expression_creation_map(), map);


	x.requires_grad_();
	torch::Tensor y;
	torch::Tensor j1;

	y = tensor_from_tentok(expression.eval(), x.device());
	std::cout << "ysizes: " << y.sizes() << std::endl;
	j1 = tc::compute::jacobian(y, x).detach_();
	std::cout << "j1sizes: " << j1.sizes() << std::endl;

	torch::Tensor j2_1;
	torch::Tensor j2_2;


	std::cout << "autograddiff: " << j1 << std::endl;

	j2_1 = tensor_from_tentok(expression.diff(varx), x.device());
	j2_2 = tensor_from_tentok(expression.diff(vary), x.device());

	std::cout << "j2_1sizes: " << j2_1.sizes() << std::endl;
	std::cout << "j2_2sizes: " << j2_2.sizes() << std::endl;

	torch::Tensor j2 = torch::empty({ nprob, 1, nparam });

	using namespace torch::indexing;

	j2.select(2, 0) = j2_1;
	j2.select(2, 1) = j2_2;

	//j2.index_put_({ Slice(), Slice(), 0 }, j2_1);
	//j2.index_put_({ Slice(), Slice(), 1 }, j2_2);

	std::cout << "diff: " << ": " << j2 << std::endl;


}

void test_times() {
	using namespace tc::expression;

	int64_t nprob = 4000;
	int64_t nparam = 2;
	int64_t ndata = 4;
	torch::Tensor x = torch::rand({ nprob,nparam });
	torch::Tensor b = torch::rand({ 1,ndata });

	std::function<torch::Tensor()> xfetcher = [&x]() {
		//std::cout << "xsize: " << x.select(1, 0).unsqueeze(-1).sizes() << std::endl;
		return x.select(1, 0).unsqueeze(-1);
	};

	std::function<torch::Tensor()> yfetcher = [&x]() {
		//std::cout << "ysize: " << x.select(1, 1).unsqueeze(-1).sizes() << std::endl;
		return x.select(1, 1).unsqueeze(-1);
	};

	std::function<torch::Tensor()> bfetcher = [&b]() {
		//std::cout << "bsize: " << b.sizes() << std::endl;
		return b;
	};

	FetcherMap map{ {"x", xfetcher}, {"y", yfetcher}, {"b", bfetcher} };

	VariableToken varx("x");
	VariableToken vary("y");
	VariableToken varb("b");

	LexContext context;
	context.variables.push_back(varx);
	context.variables.push_back(vary);
	context.variables.push_back(varb);
	LexContext context_copy = context;

	Lexer lexer(std::move(context));
	std::string expr = "x*exp(-b*y)+x-x";

	std::vector<std::unique_ptr<Token>> toks = lexer.lex(expr);

	Shunter shunter;
	std::deque<std::unique_ptr<Token>> shunted_toks = shunter.shunt(std::move(toks));


	Expression expression(shunted_toks, Expression::default_expression_creation_map(), map);


	x.requires_grad_();
	torch::Tensor y;
	torch::Tensor j1;

	y = tensor_from_tentok(expression.eval(), x.device());

	auto t1 = std::chrono::steady_clock::now();
	j1 = tc::compute::jacobian(y, x).detach_();
	auto t2 = std::chrono::steady_clock::now();

	std::cout << "autograddiff time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	torch::Tensor j2_1;
	torch::Tensor j2_2;
	
	torch::Tensor j2 = torch::empty({ nprob, ndata, nparam });
	using namespace torch::indexing;

	auto t3 = std::chrono::steady_clock::now();
	j2_1 = tensor_from_tentok(expression.diff(varx), x.device());
	j2_2 = tensor_from_tentok(expression.diff(vary), x.device());
	j2.select(2, 0) = j2_1;
	j2.select(2, 1) = j2_2;
	//j2.index_put_({ Slice(), Slice(), 0 }, j2_1);
	//j2.index_put_({ Slice(), Slice(), 1 }, j2_2);
	//j2.masked_scatter_({ Slice(), Slice(), 0 }, j2_1);
	auto t4 = std::chrono::steady_clock::now();

	std::cout << "diff time: " << ": " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << std::endl;

}

int main() {
	
	test_values();

	test_times();
	test_times();

}
