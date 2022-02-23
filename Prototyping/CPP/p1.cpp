#include <regex>
#include <iostream>

static std::regex regex("^([\\d]+(.\\d+)?(?:e-?\\d+)?)?(i?)", std::regex_constants::ECMAScript | std::regex_constants::icase);



int main() {
	//std::string regexp = "^(?=[iI.\\d+-])([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?(?![iI.\\d]))?([+-]?(?:(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?)?[iI])?$";
	//std::string regexp = "^([\\d]+(.\\d+)?(?:e-?\\d+)?)?(i?)";

	std::string expression = "2.0e+3i";

	std::cmatch m;
	
	if (std::regex_search(expression.c_str(), m, regex, std::regex_constants::match_not_null)) {
		std::cout << "m is empty: " << m.empty() << std::endl;
		std::cout << "m[0]: " << m[0].str() << std::endl;
		std::cout << "m[1]: " << m[1].str() << std::endl;
		std::cout << "m[2]: " << m[2].str() << std::endl;
		std::cout << "m[3]: " << m[3].str() << std::endl;
		float value = std::atof(m[1].str().c_str());
		std::cout << "value: " << value << std::endl;
	}

	
	return 0;
}