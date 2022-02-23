
#include "../compute.hpp"

int main() {

	using namespace tc::expression;

	LexContext context;
	context.variables.push_back(Variable("X"));

	Lexer lexer(std::move(context));

	std::string expr = "log(2.24e-15i+X,X)";

	std::vector<std::unique_ptr<Token>> toks;
	try {
		toks = lexer.lex(expr);
	}
	catch (std::runtime_error e) {
		std::cout << e.what() << std::endl;
	}

	for (auto& tok : toks) {
		std::cout << "token-id: " << tok->get_id() << " token-type: " << tok->get_token_type() << std::endl;
	}

	return 0;
}