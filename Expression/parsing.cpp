#include "../pch.hpp"

#include "parsing.hpp"

#include <regex>


std::tuple<std::string, std::string, tc::expression::NumberTypeBits> tc::expression::extractNumberString(const std::string& str)
{
	std::string r1 = "^\\d*(([.]\\d{3})+)?([.]\\d+)?([eE][+-]?\\d+)?";
	std::regex r(r1);
	std::smatch m;

	int matchit = 0;
	std::uint8_t flag = eNumberType::NO_NUMBER;
	std::string mstr;

	std::regex_search(str, m, r);


	// We matched a number
	if (!m.empty()) {
		if (m[0].str().size() > 0) {
			mstr = m[0];
			matchit = mstr.size();
			flag = eNumberType::REAL;
			// There are more characters in the string, could be a complex number
			if (str.size() > mstr.size()) {
				// It was an imaginary number
				if (str[mstr.size()] == 'i') {
					flag = eNumberType::COMPLEX;
					matchit += 1;
					mstr += "i";
				}
			}
		}
	}

	return std::make_tuple(str.substr(matchit), mstr, flag);
}

std::tuple<std::string, double, tc::expression::NumberTypeBits> tc::expression::getNumberFromString(const std::string& str)
{
	std::string r1 = "^\\d*(([.]\\d{3})+)?([.]\\d+)?([eE][+-]?\\d+)?";
	std::regex r(r1);
	std::smatch m;

	int matchit = 0;
	double ret = 0.0f;
	std::uint8_t flag = eNumberType::NO_NUMBER;

	std::regex_search(str, m, r);

	// We matched a number
	if (!m.empty()) {
		if (m[0].str().size() > 0) {
			std::string mstr = m[0];
			ret = stod(mstr);
			matchit = mstr.size();
			// There are more characters in the string, could be a complex number
			if (str.size() > mstr.size()) {
				// It was an imaginary number
				if (str[mstr.size()] == 'i') {
					flag = eNumberType::IMAGINARY;
					matchit += 1;
				}
				else {
					flag = eNumberType::REAL;
				}
			}
		}
	}

	return std::make_tuple(str.substr(matchit), ret, flag);
}

std::tuple<double, tc::expression::NumberTypeBits> tc::expression::getNumber(std::string str)
{
	if (str.empty())
		throw std::runtime_error("tried to getNumber from empty string");

	NumberTypeBits flag = eNumberType::REAL;
	if (str.back() == 'i') {
		str.pop_back();
		flag = eNumberType::IMAGINARY;
	}

	return std::make_tuple(std::stod(str), flag);
}

std::function<torch::Tensor()> tc::expression::defaultNumberResolver(std::string str, torch::TensorOptions& ops)
{
	double number;
	NumberTypeBits type;
	std::tie(number, type) = getNumber(str);

	if (type == eNumberType::IMAGINARY) {
		return [&ops, number] {
			torch::TensorOptions nops(ops.device_opt().value());
			switch (ops.dtype_opt().value().toScalarType())
			{
			case (torch::kHalf):
				nops = nops.dtype(torch::kComplexHalf);
				break;
			case (torch::kFloat):
				nops = nops.dtype(torch::kComplexFloat);
				break;
			case (torch::kDouble):
				nops = nops.dtype(torch::kComplexDouble);
				break;
			default:
				throw std::runtime_error("The given dtype is not available");
				break;
			}

			return torch::tensor(c10::complex<double>(0.0, number), nops);
		};
	}
	else if (type == eNumberType::REAL) { // type = eNumberType::REAL
		return [&ops, number] {
			return torch::tensor(number, ops);
		};
	}
	else {
		throw std::runtime_error("defaultNumberResolver received a number string which was neither real nor imaginary");
	}
}

/*

void tc::expression::special_print(std::string str) {
	for (int i = 0; i < str.size(); ++i) {
		std::cout << (int)str[i];
	}
	std::cout << std::endl;
}

void tc::expression::print_match(std::smatch m) {
	for (auto& a : m) {
		std::cout << a << " , ";
	}
	std::cout << "sizeof match: " << m.size() << std::endl;
	std::cout << std::endl;
}

std::tuple<std::string, std::complex<double>> tc::expression::get_next_number_match(const std::string& str) {
	// Long complex number regex matcher
	std::string r1 = "^(?=[iI.\\d+-])([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?(?![iI.\\d]))?([+-]?(?:(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?)?[iI])?$";

	std::regex r(r1);

	std::vector<std::string> matches;
	std::string substr;
	int matchit;
	for (int i = 0; i < str.size(); ++i) {
		substr += str[i];
		std::smatch m;
		if(std::regex_match(substr, m, r)) {
			matchit = i;
			matches.clear();
			for (auto a : m) {
				if (a.matched) {
					//std::cout << "a: " << a << " sizeof(a): " << a.str().size() << std::endl;
					matches.push_back(a.str());
				}
			}
		}
	}

	std::complex<double> out;

	if (matches.size() == 2) {
		std::string mstr = matches[1];
		if (mstr[mstr.size()-1] == 'i') {
			mstr = mstr.substr(0,mstr.size()-1);
			if (mstr.empty()) {
				mstr += "1";
			}
			std::istringstream is("(0," + mstr + ")");
			if (!(is >> out))
				throw std::runtime_error("A string matched pure imaginary regex but could not be parsed");
		}
		else {
			std::istringstream is(mstr);
			if(!(is >> out))
				throw std::runtime_error("A string matched pure real regex but could not be parsed");
		}
	}
	else if (matches.size() == 3) {
		std::string real = matches[1];
		std::string imag = matches[2];
		imag = imag.substr(0,imag.size()-1);
		if (imag.size() == 1) {
			imag += "1";
		}
		std::istringstream is ("(" + real + "," + imag + ")");
		if (!(is >> out))
			throw std::runtime_error("A string matched complex regex but could not be parsed");
	}
	else {
		return std::make_tuple(str, 0);
	}

	return std::make_tuple(str.substr(matchit+1,str.size()-1), out);
}

*/
