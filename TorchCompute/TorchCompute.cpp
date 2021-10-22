// ComputeCMake.cpp : Defines the entry point for the application.
//

#include "tests.hpp"


int main()
{
	try {
		try {
			test::test_kmeans();
		}
		catch (c10::Error e1) {
			std::cout << e1.what() << std::endl;
		}
	}
	catch (std::runtime_error e2) {
		std::cout << e2.what() << std::endl;
	}
	
	return 0;
}
