
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

void test_adc_values() {


	//auto params = torch::rand({ 10, 2 });
	//auto b = torch::rand({ 1,4 });

	int n = 1;

	torch::Tensor params = torch::empty({ n,2 });
	params.select(1, 0).fill_(1000.0f);
	params.select(1, 1).fill_(0.0080f);

	torch::Tensor b = torch::empty({ 1,4 });
	b.select(1, 0).fill_(0.0f);
	b.select(1, 1).fill_(200.0f);
	b.select(1, 2).fill_(400.0f);
	b.select(1, 3).fill_(800.0f);

	std::vector<torch::Tensor> constants({ b });

	torch::Tensor values = torch::empty({ n,4 });
	torch::Tensor jacobian = torch::empty({ n,4,2 });
	torch::Tensor hessian = torch::empty({ n,2,2 });

	tc::models::mp_adc_eval_jac_hess(constants, params, values, std::nullopt, std::nullopt, std::nullopt);

	torch::Tensor data = values + 0.01*torch::randn({ n,4 });

	tc::models::mp_adc_eval_jac_hess(constants, params, values, jacobian, hessian, data);

	std::cout << "data: " << data << std::endl;
	std::cout << "values: " << values << std::endl;
	std::cout << "jacobian: " << jacobian << std::endl;
	std::cout << "hessian: " << hessian << std::endl;

}

int main() {

	test_adc_values();

}