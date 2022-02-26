#include "../compute.hpp"


int main() {

	tc::expression::ZeroToken zt;
	tc::expression::UnityToken ut;
	tc::expression::NegUnityToken nut;
	tc::expression::NanToken nat;
	tc::expression::NumberToken numt;

	// zt
	std::cout << (zt + ut).get_token_type() << std::endl;
	std::cout << (zt + nut).get_token_type() << std::endl;
	std::cout << (zt + zt).get_token_type() << std::endl;
	std::cout << (zt + nat).get_token_type() << std::endl;
	std::cout << (zt + numt).get_token_type() << std::endl;

	std::cout << (nut + zt).get_token_type() << std::endl;
	std::cout << (nut + nut).get_token_type() << std::endl;
	std::cout << (nut + nut).get_token_type() << std::endl;
	std::cout << (nut + nut).get_token_type() << std::endl;
	std::cout << (nut + nut).get_token_type() << std::endl;

	std::cout << (nat + zt).get_token_type() << std::endl;
	std::cout << (nat + nat).get_token_type() << std::endl;

	std::cout << (numt + zt).get_token_type() << std::endl;
	std::cout << (numt + numt).get_token_type() << std::endl;

	// ut
	std::cout << (ut + ut).get_token_type() << std::endl;
	std::cout << (ut + zt).get_token_type() << std::endl;
	std::cout << (ut + ut).get_token_type() << std::endl;
	std::cout << (zt + zt).get_token_type() << std::endl;

	return 0;
}