
#include "../compute.hpp"


void test_linear_adc(int n, bool print) {
	using namespace tc;

	std::cout << "ADC-model - cpu" << std::endl;
	{
		torch::Tensor bvals = torch::rand({ n,4,1 });
		torch::Tensor params = torch::rand({ n,2 });

		std::unique_ptr<optim::Model> pModel = std::make_unique<optim::Model>(models::adc_eval_and_diff);
		pModel->setParameters(params);
		pModel->setPerProblemInputs(bvals);

		torch::Tensor data = torch::empty({ n, 4});
		pModel->eval(data);

		if (print)
			std::cout << "True params:\n" << params << std::endl;

		params = models::simple_adc_model_linear(bvals, data);

		if (print)
			std::cout << "Found params:\n" << params << std::endl;

	}

	std::cout << "ADC-model - cuda" << std::endl;
	{
		torch::TensorOptions dops;
		dops = dops.dtype(torch::kFloat64).device(torch::Device("cuda:0"));

		torch::Tensor bvals = torch::rand({ n,4,1 }, dops);
		torch::Tensor params = torch::rand({ n,2 }, dops);

		std::unique_ptr<optim::Model> pModel = std::make_unique<optim::Model>(models::adc_eval_and_diff);
		pModel->setParameters(params);
		pModel->setPerProblemInputs(bvals);


		torch::Tensor data = torch::empty({ n, 4 }, dops);
		pModel->eval(data);

		if (print)
			std::cout << "True params:\n" << params << std::endl;

		auto start_time = std::chrono::high_resolution_clock::now();
		params = models::simple_adc_model_linear(bvals, data);
		torch::cuda::synchronize(0);
		auto end_time = std::chrono::high_resolution_clock::now();
		auto time = end_time - start_time;
		std::cout << "time: " << time / std::chrono::milliseconds(1) << std::endl;

		if (print)
			std::cout << "Found params:\n" << params << std::endl;

	}
}

void test_linear_vfa(int n, bool print) {
	using namespace tc;

	std::cout << "VFA-model - cpu" << std::endl;
	{
		torch::Tensor fa = torch::rand({ n,4,1 });
		torch::Tensor params = torch::rand({ n,2 });
		torch::Tensor TR = torch::full({ 1 }, 1);

		std::unique_ptr<optim::Model> pModel = std::make_unique<optim::Model>(models::vfa_eval_and_diff);
		pModel->setParameters(params);
		pModel->setPerProblemInputs(fa);
		pModel->setConstants(std::vector<torch::Tensor>{TR});

		torch::Tensor data = torch::empty({ n, 4 });
		pModel->eval(data);

		if (print)
			std::cout << "True params:\n" << params << std::endl;

		auto start_time = std::chrono::high_resolution_clock::now();
		params = models::simple_vfa_model_linear(fa, data, TR);
		auto end_time = std::chrono::high_resolution_clock::now();
		auto time = end_time - start_time;
		std::cout << "time: " << time / std::chrono::milliseconds(1) << std::endl;

		if (print)
			std::cout << "Found params:\n" << params << std::endl;

	}

	std::cout << "VFA-model - cuda" << std::endl;
	{
		torch::TensorOptions dops;
		dops = dops.dtype(torch::kFloat64).device(torch::Device("cuda:0"));

		torch::Tensor fa = torch::rand({ n,4,1 }, dops);
		torch::Tensor params = torch::rand({ n,2 }, dops);
		torch::Tensor TR = torch::full({ 1 }, 1, dops);

		std::unique_ptr<optim::Model> pModel = std::make_unique<optim::Model>(models::vfa_eval_and_diff);
		pModel->setParameters(params);
		pModel->setPerProblemInputs(fa);
		pModel->setConstants(std::vector<torch::Tensor>{TR});

		torch::Tensor data = torch::empty({ n, 4 }, dops);
		pModel->eval(data);

		if (print)
			std::cout << "True params:\n" << params << std::endl;

		auto start_time = std::chrono::high_resolution_clock::now();
		params = models::simple_vfa_model_linear(fa, data, TR);
		torch::cuda::synchronize(0);
		auto end_time = std::chrono::high_resolution_clock::now();
		auto time = end_time - start_time;
		std::cout << "time: " << time / std::chrono::milliseconds(1) << std::endl;

		if (print)
			std::cout << "Found params:\n" << params << std::endl;

	}
}


int main() {

	try {
		test_linear_adc(4, true);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}

	try {
		test_linear_vfa(4, true);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}

	try {
		test_linear_vfa(200000, false);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}


}