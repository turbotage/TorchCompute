#include "lexer.hpp"

#include <regex>





std::tuple<std::map<std::string, torch::Tensor>, std::string> expression::Lexer::operator()(const std::string& expression)
{
	std::string exp = expression;
	std::map<std::string, std::string> numberstringmap = lexfix(exp);
	std::map<std::string, torch::Tensor> numbermap;

	for (auto& num : numberstringmap) {
		if (num.second.back() == 'i') { // This was a imaginary number
			std::string str = num.second;
			str.pop_back();
			numbermap.insert({ num.first, torch::complex(torch::full({}, 0), torch::full({}, std::stof(str))) });
		}
		else {
			numbermap.insert({ num.first, torch::full({}, std::stof(num.second)) });
		}
	}
	
	return std::make_tuple(std::move(numbermap), exp);
}




bool is_negative(std::string& expression, int i) {
	if (i > 0) {
		if (expression[i - 1] == '-') {
			if (i > 1) {
				char c = expression[i - 2];
				if (c == ',' || c == '(') {
					return true;
				}
			}
			else {
				return true;
			}
		}
	}
	return false;
}

std::map<std::string, std::string> expression::Lexer::lexfix(std::string& expression)
{
	std::string regstr = "^\\d+(([.]\\d+))?([eE][+-]?\\d+)?([i]?)";
	std::regex r(regstr);
	std::smatch m;

	std::map<std::string, std::string> numbermap;

	int numbers = 0;
	for (int i = 0; i < expression.length(); ++i) {

		// This is a variable, read until it terminates
		if (expression[i] == '$') {
			bool is_neg = is_negative(expression, i);

			std::string varname;
			do {
				varname += expression[i];
				++i;
			} while (i < expression.length() && std::isalnum(expression[i]));

			if (is_neg) {
				expression.erase(i - varname.length() - 1, 1);
				expression.insert(i - varname.length(), "NEG_");
				i += 3;
			}
		}

		std::string substr = expression.substr(i);

		std::regex_search(substr, m, r);

		if (m[0].matched) {
			std::string match_str = m[0].str();

			std::string variable_name = "$NUMVAR" + std::to_string(numbers);

			bool is_neg = is_negative(expression, i);

			if (is_neg) {
				match_str = "-" + match_str;
				--i;
			}

			expression.erase(i, match_str.length());
			expression.insert(i, variable_name);

			numbermap.insert({ variable_name, match_str });

			i += variable_name.length();
			++numbers;
		}
	}

	return numbermap;
}



