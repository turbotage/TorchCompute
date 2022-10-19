#include "../compute.hpp"


void test_values() {
	using namespace tc::expression;

	torch::InferenceMode im_guard;

	int64_t nprob = 2;
	int64_t nparam = 5;
	int64_t ndata = 10;
	torch::Tensor x = torch::rand({ nprob,nparam });
	torch::Tensor b = torch::rand({ 1, ndata });
	torch::Tensor g = torch::rand({ 1, ndata });

	std::function<torch::Tensor()> xfetcher = [&x]() {
		//std::cout << "xsize: " << x.select(1, 0).unsqueeze(-1).sizes() << std::endl;
		return x.select(1, 0).unsqueeze(-1);
	};

	std::function<torch::Tensor()> yfetcher = [&x]() {
		//std::cout << "ysize: " << x.select(1, 1).unsqueeze(-1).sizes() << std::endl;
		return x.select(1, 1).unsqueeze(-1);
	};

	std::function<torch::Tensor()> zfetcher = [&x]() {
		//std::cout << "ysize: " << x.select(1, 1).unsqueeze(-1).sizes() << std::endl;
		return x.select(1, 2).unsqueeze(-1);
	};

	std::function<torch::Tensor()> wfetcher = [&x]() {
		//std::cout << "ysize: " << x.select(1, 1).unsqueeze(-1).sizes() << std::endl;
		return x.select(1, 3).unsqueeze(-1);
	};

	std::function<torch::Tensor()> ufetcher = [&x]() {
		//std::cout << "ysize: " << x.select(1, 1).unsqueeze(-1).sizes() << std::endl;
		return x.select(1, 4).unsqueeze(-1);
	};

	std::function<torch::Tensor()> bfetcher = [&b]() {
		//std::cout << "bsize: " << b.sizes() << std::endl;
		return b;
	};

	std::function<torch::Tensor()> gfetcher = [&g]() {
		//std::cout << "bsize: " << b.sizes() << std::endl;
		return g;
	};

	FetcherMap map{ {"x1", xfetcher}, {"y0", yfetcher}, {"z", zfetcher}, {"w", wfetcher}, {"u", ufetcher}, {"b", bfetcher}, {"g", gfetcher} };

	VariableToken varx("x1");
	VariableToken vary("y0");
	VariableToken varz("z");
	VariableToken varw("w");
	VariableToken varu("u");
	VariableToken varb("b");
	VariableToken varg("g");

	LexContext context;
	context.variables.push_back(varx);
	context.variables.push_back(vary);
	context.variables.push_back(varz);
	context.variables.push_back(varw);
	context.variables.push_back(varu);
	context.variables.push_back(varb);
	context.variables.push_back(varg);
	LexContext context_copy = context;

	Lexer lexer(std::move(context));
	//std::string expr = "x*sin(b)*(1-exp(-g/y))/(1-cos(b)*exp(-g/y))+pow(x,2.0)";
	std::string expr = "pow(x1,2.0)+y0";

	std::vector<std::unique_ptr<Token>> toks = lexer.lex(expr);

	Shunter shunter;
	std::deque<std::unique_ptr<Token>> shunted_toks = shunter.shunt(std::move(toks));


	Expression expression(shunted_toks, Expression::default_expression_creation_map(), map);

	torch::Tensor y;
	torch::Tensor j1;


	std::chrono::steady_clock::time_point t1;
	torch::Tensor xtemp = x;
	{
		torch::InferenceMode im_guard2(false);
		bool is_inference = x.is_inference();
		if (is_inference) {
			x = x.clone();
		}
		x.requires_grad_(true);

		y = tensor_from_tentok(expression.eval(), x.device());

		t1 = std::chrono::steady_clock::now();
		j1 = tc::compute::jacobian(y, x).detach_();
		y.detach_();

		if (is_inference) {
			x = xtemp;
		}

	}
	auto t2 = std::chrono::steady_clock::now();

	std::cout << "autograddiff: " << j1 << std::endl;

	torch::Tensor j2 = torch::empty({ nprob, ndata, nparam });
	using namespace torch::indexing;

	auto t3 = std::chrono::steady_clock::now();
	y = tensor_from_tentok(expression.eval(), x.device());
	{
		j2.select(2, 0) = tensor_from_tentok(expression.diff(varx), x.device());
		j2.select(2, 1) = tensor_from_tentok(expression.diff(vary), x.device());
		j2.select(2, 2) = tensor_from_tentok(expression.diff(varz), x.device());
		j2.select(2, 3) = tensor_from_tentok(expression.diff(varw), x.device());
		j2.select(2, 4) = tensor_from_tentok(expression.diff(varu), x.device());
	}
	auto t4 = std::chrono::steady_clock::now();

	std::cout << "my autograddiff: " << j2 << std::endl;


}

void test_times(int64_t nprob) {
	using namespace tc::expression;

	torch::InferenceMode im_guard;

	int64_t nparam = 5;
	int64_t ndata = 10;
	torch::Tensor x = torch::rand({ nprob,nparam });
	torch::Tensor b = torch::rand({ 1, ndata });
	torch::Tensor g = torch::rand({ 1, ndata });

	std::function<torch::Tensor()> xfetcher = [&x]() {
		//std::cout << "xsize: " << x.select(1, 0).unsqueeze(-1).sizes() << std::endl;
		return x.select(1, 0).unsqueeze(-1);
	};

	std::function<torch::Tensor()> yfetcher = [&x]() {
		//std::cout << "ysize: " << x.select(1, 1).unsqueeze(-1).sizes() << std::endl;
		return x.select(1, 1).unsqueeze(-1);
	};

	std::function<torch::Tensor()> zfetcher = [&x]() {
		//std::cout << "ysize: " << x.select(1, 1).unsqueeze(-1).sizes() << std::endl;
		return x.select(1, 2).unsqueeze(-1);
	};

	std::function<torch::Tensor()> wfetcher = [&x]() {
		//std::cout << "ysize: " << x.select(1, 1).unsqueeze(-1).sizes() << std::endl;
		return x.select(1, 3).unsqueeze(-1);
	};

	std::function<torch::Tensor()> ufetcher = [&x]() {
		//std::cout << "ysize: " << x.select(1, 1).unsqueeze(-1).sizes() << std::endl;
		return x.select(1, 4).unsqueeze(-1);
	};

	std::function<torch::Tensor()> bfetcher = [&b]() {
		//std::cout << "bsize: " << b.sizes() << std::endl;
		return b;
	};

	std::function<torch::Tensor()> gfetcher = [&g]() {
		//std::cout << "bsize: " << b.sizes() << std::endl;
		return g;
	};

	FetcherMap map{ {"x", xfetcher}, {"y0", yfetcher}, {"z", zfetcher}, {"w", wfetcher}, {"u", ufetcher}, {"b", bfetcher}, {"g", gfetcher} };

	VariableToken varx("x");
	VariableToken vary("y0");
	VariableToken varz("z");
	VariableToken varw("w");
	VariableToken varu("u");
	VariableToken varb("b");
	VariableToken varg("g");

	LexContext context;
	context.variables.push_back(varx);
	context.variables.push_back(vary);
	context.variables.push_back(varz);
	context.variables.push_back(varw);
	context.variables.push_back(varu);
	context.variables.push_back(varb);
	context.variables.push_back(varg);
	LexContext context_copy = context;

	Lexer lexer(std::move(context));
	//std::string expr = "x*sin(b)*(1-exp(-g/y))/(1-cos(b)*exp(-g/y))";
	std::string expr = "pow(x,2.0)+y0";

	std::vector<std::unique_ptr<Token>> toks = lexer.lex(expr);

	Shunter shunter;
	std::deque<std::unique_ptr<Token>> shunted_toks = shunter.shunt(std::move(toks));


	Expression expression(shunted_toks, Expression::default_expression_creation_map(), map);

	torch::Tensor y;
	torch::Tensor j1;


	auto t1 = std::chrono::steady_clock::now();
	torch::Tensor xtemp = x;
	{
		torch::InferenceMode im_guard2(false);
		bool is_inference = x.is_inference();
		if (is_inference) {
			x = x.clone();
		}
		x.requires_grad_(true);

		y = tensor_from_tentok(expression.eval(), x.device());
		j1 = tc::compute::jacobian(y, x).detach_();
		y.detach_();

		if (is_inference) {
			x = xtemp;
		}

	}
	auto t2 = std::chrono::steady_clock::now();

	std::cout << "autograddiffiff time: " << ": " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	torch::Tensor j2 = torch::empty({ nprob, ndata, nparam });
	using namespace torch::indexing;

	auto t3 = std::chrono::steady_clock::now();
	{
		y = tensor_from_tentok(expression.eval(), x.device());
		j2.select(2, 0) = tensor_from_tentok(expression.diff(varx), x.device());
		j2.select(2, 1) = tensor_from_tentok(expression.diff(vary), x.device());
		j2.select(2, 2) = tensor_from_tentok(expression.diff(varz), x.device());
		j2.select(2, 3) = tensor_from_tentok(expression.diff(varw), x.device());
		j2.select(2, 4) = tensor_from_tentok(expression.diff(varu), x.device());
	}
	auto t4 = std::chrono::steady_clock::now();

	std::cout << "my autograddiffiff time: " << ": " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << std::endl;

}

int main() {
	
	test_values();

	test_times(10);
	test_times(10);
	test_times(10);
	test_times(10);

}
