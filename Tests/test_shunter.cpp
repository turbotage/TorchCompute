#include "../compute.hpp"

#include <iterator>
#include <iostream>



int main() {

	using namespace tc::expression;

	LexContext context;
	context.variables.push_back(VariableToken("X"));
	context.variables.push_back(VariableToken("Y"));

	LexContext context_copy = context;

	Lexer lexer(std::move(context));

	std::string expr = "-log(2.24e-15i-24*X)-sin(X)^cos(Y)^2*pow(X,+Y)";

	std::vector<std::unique_ptr<Token>> toks;
	try {
		toks = lexer.lex(expr);
	}
	catch (std::runtime_error e) {
		std::cout << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	std::cout << "lexed tokens:\n";
	for (auto& tok : toks) {
		std::cout << "token-id: " << tok->get_id() << " token-type: " << tok->get_token_type() << std::endl;
	}
	std::cout << "\n\n";

	Shunter shunter;

	std::deque<std::unique_ptr<Token>> shunted_toks;
	try {
		shunted_toks = shunter.shunt(std::move(toks));
	}
	catch (std::runtime_error e) {
		std::cout << e.what() << std::endl;
	}

	std::cout << "shunted tokens:\n";

	std::unordered_map<int32_t, std::string> allmighty_map;

	allmighty_map.insert(FIXED_ID_MAPS.begin(), FIXED_ID_MAPS.end());
	allmighty_map.insert(DEFAULT_OPERATOR_MAPS.begin(), DEFAULT_OPERATOR_MAPS.end());
	allmighty_map.insert(DEFAULT_FUNCTION_MAPS.begin(), DEFAULT_FUNCTION_MAPS.end());

	for (auto& tok : shunted_toks) {
		std::string str = allmighty_map.at(tok->get_id());
		if (tok->get_token_type() == TokenType::NUMBER) {
			NumberToken& num = dynamic_cast<NumberToken&>(*tok);
			std::cout << "token: " << num.get_full_name() << " token-type: " << tok->get_token_type() << " token-id: " << tok->get_id() << std::endl;
		}
		else if (tok->get_token_type() == TokenType::VARIABLE) {
			VariableToken& var = dynamic_cast<VariableToken&>(*tok);
			std::cout << "token: " << var.name << " token-type: " << tok->get_token_type() << " token-id: " << tok->get_id() << std::endl;
		}
		else {
			std::cout << "token: " << allmighty_map.at(tok->get_id()) << " token-type: " << tok->get_token_type() << " token-id: " << tok->get_id() << std::endl;
		}
	}
	std::cout << "\n\n";

	return 0;
}