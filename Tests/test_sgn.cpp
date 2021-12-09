
#include "../compute.hpp"

void gn_cpu_adc_vfa_anal(int n, bool print) {
	std::cout << "ADC model" << std::endl;
	std::cout << "per problem b-vals : eval_and_diff" << std::endl;

	optim::SGNSettings settings;

	std::unique_ptr<optim::Model> pModel;
	{
		using namespace torch::indexing;

		pModel = std::make_unique<optim::Model>(models::adc_eval_and_diff);

		torch::TensorOptions dops;
		dops.dtype(torch::kFloat64);

		auto params = torch::rand({ n, 2 }, dops);

		auto ppi = torch::rand({ n, 3, 1 }, dops);

		pModel->setParameters(params);
		pModel->setPerProblemInputs(ppi);

		torch::Tensor data = torch::empty({ n, 3 }, dops);
		pModel->eval(data);


		auto guess = torch::empty({ n, 2 }, dops);
		guess.index_put_({ Slice(), 0 }, 0.5);
		guess.index_put_({ Slice(), 1 }, 0.5);
		pModel->setParameters(guess);

		settings.pModel = std::move(pModel);
		settings.data = data;
		settings.maxIter = 10;

		if (print) {
			std::cout << "true params: " << params << std::endl;
		}

		optim::SGNResult res = optim::SGN(settings).eval();

		if (print) {
			std::cout << "found params: " << res.finalParameters << std::endl;
		}

		std::cout << "No crash, Success!" << std::endl;
	}

	std::cout << "VFA model" << std::endl;
	std::cout << "per problem FA-vals : eval_and_diff" << std::endl;

	{
		using namespace torch::indexing;

		pModel = std::make_unique<optim::Model>(models::vfa_eval_and_diff);

		torch::TensorOptions dops;
		dops.dtype(torch::kFloat64);

		auto params = torch::rand({ n, 2 }, dops);

		auto ppi = torch::rand({ n, 4, 1 }, dops);

		auto TR = torch::full({ 1 }, 1, dops);

		pModel->setParameters(params);
		pModel->setPerProblemInputs(ppi);
		pModel->setConstants(std::vector<torch::Tensor>{ TR });

		torch::Tensor data = torch::empty({ n, 4 }, dops);
		pModel->eval(data);


		auto guess = torch::empty({ n, 2 }, dops);
		guess.index_put_({ Slice(), 0 }, 0.5);
		guess.index_put_({ Slice(), 1 }, 0.5);
		pModel->setParameters(guess);

		settings.pModel = std::move(pModel);
		settings.data = data;
		settings.maxIter = 10;

		if (print) {
			std::cout << "true params: " << params << std::endl;
		}

		optim::SGNResult res = optim::SGN(settings).eval();

		if (print) {
			std::cout << "found params: " << res.finalParameters << std::endl;
		}

		std::cout << "No crash, Success!" << std::endl;
	}

}

void slmp_cuda_adc_vfa_anal(int n, bool print) {
	std::cout << "ADC model" << std::endl;
	std::cout << "per problem b-vals : eval_and_diff : switch to cpu" << std::endl;

	optim::SGNSettings settings;

	std::unique_ptr<optim::Model> pModel;
	{
		using namespace torch::indexing;

		pModel = std::make_unique<optim::Model>(models::adc_eval_and_diff);

		torch::TensorOptions dops;
		dops = dops.dtype(torch::kFloat64).device(torch::Device("cuda:0"));

		auto params = torch::rand({ n, 2 }, dops);

		auto ppi = torch::rand({ n, 3, 1 }, dops);

		pModel->setParameters(params);
		pModel->setPerProblemInputs(ppi);

		torch::Tensor data = torch::empty({ n, 3 }, dops);
		pModel->eval(data);


		auto guess = torch::empty({ n, 2 }, dops);
		guess.index_put_({ Slice(), 0 }, 0.5);
		guess.index_put_({ Slice(), 1 }, 0.5);
		pModel->setParameters(guess);

		settings.pModel = std::move(pModel);
		settings.data = data;
		settings.maxIter = 30;
		settings.startDevice = torch::Device("cuda:0");
		settings.switchDevice = torch::Device("cpu");
		settings.switchAtN = 10000;

		if (print) {
			std::cout << "true params: " << params << std::endl;
		}

		optim::SGNResult res = optim::SGN(settings).eval();

		if (print) {
			std::cout << "found params: " << res.finalParameters << std::endl;
		}

		std::cout << "No crash, Success!" << std::endl;
	}

	std::cout << "VFA model" << std::endl;
	std::cout << "per problem FA-vals : eval_and_diff" << std::endl;
	{
		using namespace torch::indexing;

		pModel = std::make_unique<optim::Model>(models::vfa_eval_and_diff);

		torch::TensorOptions dops;
		dops = dops.dtype(torch::kFloat64).device(torch::Device("cuda:0"));

		auto params = torch::rand({ n, 2 }, dops);

		auto ppi = torch::rand({ n, 4, 1 }, dops);

		pModel->setParameters(params);
		pModel->setPerProblemInputs(ppi);
		auto TR = torch::full({ 1 }, 1, dops);
		pModel->setConstants(std::vector<torch::Tensor>{ TR });

		torch::Tensor data = torch::empty({ n, 4 }, dops);
		pModel->eval(data);


		auto guess = torch::empty({ n, 2 }, dops);
		guess.index_put_({ Slice(), 0 }, 0.5);
		guess.index_put_({ Slice(), 1 }, 0.5);
		pModel->setParameters(guess);

		settings.pModel = std::move(pModel);
		settings.data = data;
		settings.maxIter = 20;
		settings.startDevice = torch::Device("cuda:0");
		//settings.switchDevice = torch::Device("cpu");
		//settings.switchAtN = 10000;

		if (print) {
			std::cout << "true params: " << params << std::endl;
		}

		optim::SGNResult res = optim::SGN(settings).eval();

		if (print) {
			std::cout << "found params: " << res.finalParameters << std::endl;
		}

		std::cout << "No crash, Success!" << std::endl;
	}

}



int main() {

	try {
		gn_cpu_adc_vfa_anal(4, true);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}

	try {
		gn_cpu_adc_vfa_anal(40000, false);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}

	try {
		slmp_cuda_adc_vfa_anal(200000, false);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}

}
