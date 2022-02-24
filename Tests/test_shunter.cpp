#include "../compute.hpp"

#include <iterator>

#include <iterator>



int main() {

	using namespace tc::expression;

	LexContext context;
	context.variables.push_back(Variable("X"));
	context.variables.push_back(Variable("Y"));

	LexContext context_copy = context;

	Lexer lexer(std::move(context));

	std::string expr = "-log(2.24e-15i+X)-sin(X)^cos(Y)^2";

	std::vector<std::unique_ptr<Token>> toks;
	try {
		toks = lexer.lex(expr);
	}
	catch (std::runtime_error e) {
		std::cout << e.what() << std::endl;
	}

	std::cout << "lexed tokens:\n";
	for (auto& tok : toks) {
		std::cout << "token-id: " << tok->get_id() << " token-type: " << tok->get_token_type() << std::endl;
	}
	std::cout << "\n\n";

	Shunter shunter;

	auto shunted_toks = shunter.shunt(std::move(toks));

	std::cout << "shunted tokens:\n";

	std::unordered_map<int32_t, std::string> allmighty_map;

	allmighty_map.insert(FIXED_ID_MAPS.begin(), FIXED_ID_MAPS.end());
	allmighty_map.insert(DEFAULT_OPERATOR_MAPS.begin(), DEFAULT_OPERATOR_MAPS.end());
	allmighty_map.insert(DEFAULT_FUNCTION_MAPS.begin(), DEFAULT_FUNCTION_MAPS.end());

	for (auto& tok : shunted_toks) {
		std::string str = allmighty_map.at(tok->get_id());
		if (tok->get_token_type() == TokenType::NUMBER) {
			Number& num = dynamic_cast<Number&>(*tok);
			std::cout << "token: " << num.name << " token-type: " << tok->get_token_type() << " token-id: " << tok->get_id() << std::endl;
		}
		else if (tok->get_token_type() == TokenType::VARIABLE) {
			Variable& var = dynamic_cast<Variable&>(*tok);
			std::cout << "token: " << var.name << " token-type: " << tok->get_token_type() << " token-id: " << tok->get_id() << std::endl;
		}
		else {
			std::cout << "token: " << allmighty_map.at(tok->get_id()) << " token-type: " << tok->get_token_type() << " token-id: " << tok->get_id() << std::endl;
		}
	}
	std::cout << "\n\n";

	return 0;
}