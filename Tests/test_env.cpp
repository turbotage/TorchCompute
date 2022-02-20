
#include "../compute.hpp"

void test_bmm() {
	int n = 5000000;

	torch::Tensor x1 = torch::rand({ n,2,1 });
	torch::Tensor x2 = torch::rand({ n,2,1 });

	torch::Tensor& x1ref = x1;
	torch::Tensor& x2ref = x2;

	torch::Tensor D = torch::rand({ n,2,2 });
	torch::Tensor& Dref = D;

	long long tdiff1 = 0;
	long long tdiff2 = 0;

	auto t1 = std::chrono::steady_clock::now();
	torch::bmm_out(x1ref, Dref, x2ref);
	auto t2 = std::chrono::steady_clock::now();
	tdiff1 += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	tdiff1 = 0;

	std::cout << "bmm_out" << std::endl;
	for (int i = 0; i < 10; ++i) {
		t1 = std::chrono::steady_clock::now();
		torch::bmm_out(x1ref, Dref, x2ref);
		t2 = std::chrono::steady_clock::now();
		tdiff1 += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	}

	auto t3 = std::chrono::steady_clock::now();
	x1 = torch::bmm(Dref, x2ref);
	auto t4 = std::chrono::steady_clock::now();
	tdiff2 += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
	tdiff2 = 0;

	std::cout << "bmm" << std::endl;
	for (int i = 0; i < 10; ++i) {
		t3 = std::chrono::steady_clock::now();
		x1 = torch::bmm(Dref, x2ref);
		t4 = std::chrono::steady_clock::now();
		tdiff2 += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
	}

	std::cout << "T1: " << tdiff1 << std::endl;
	std::cout << "T2: " << tdiff2 << std::endl;
}

int main() {

	int n = 1;

	torch::Tensor x1 = torch::ones({ n,2,1 });
	torch::Tensor x2 = 3.0f * torch::ones({ n,2,1 });
	torch::Tensor x3 = torch::rand({ n,2,1 });

	torch::Tensor& x3ref = torch::sub_out(x3, x2, x1);
	std::cout << x3 << std::endl;
	std::cout << x3ref << std::endl;

	std::cout << torch::frobenius_norm(x1, 1) << std::endl;
	std::cout << torch::sqrt(torch::square(x1).sum(1)) << std::endl;

}