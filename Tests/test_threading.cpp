
#include "../compute.hpp"
#include <future>
#include <thread>
#include <chrono>

void test_threading(int n) {
	using namespace tc;

	std::cout << "ADC model" << std::endl;
	std::cout << "per problem b-vals : eval_and_diff" << std::endl;

	optim::SLMPSettings settings;

	std::cout << "get iterinfo test" << std::endl;
	std::unique_ptr<optim::Model> pModel;
	{
		using namespace torch::indexing;
		using namespace std::chrono_literals;

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

		optim::SLMP slmp(settings);

		std::packaged_task<optim::SLMPResult()> run([&settings, &slmp]() { return slmp.eval(); });
		std::future<optim::SLMPResult> res = run.get_future();
		std::cout << "starting solver" << std::endl;
		std::thread run_thread(std::move(run));

		std::this_thread::sleep_for(500ms);
		std::cout << "starting mock threads" << std::endl;
		for (int i = 0; i < 25; ++i) {
			auto pair = slmp.getIterInfo();
			std::cout << "iter: " << pair.first << "  numProb: " << pair.second << std::endl;
			std::this_thread::sleep_for(500ms);
		}

		std::cout << "waiting for results" << std::endl;

		optim::SLMPResult result = res.get();

		std::cout << "got result" << std::endl;

		run_thread.join();

		std::cout << "No crash, Success!" << std::endl;
	}

	std::cout << "abort test" << std::endl;

	{
		using namespace torch::indexing;
		using namespace std::chrono_literals;

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

		optim::SLMP slmp(settings);

		std::packaged_task<optim::SLMPResult()> run([&settings, &slmp]() { return slmp.eval(); });
		std::future<optim::SLMPResult> res = run.get_future();
		std::cout << "starting solver" << std::endl;
		std::thread run_thread(std::move(run));

		std::this_thread::sleep_for(500ms);
		std::cout << "starting mock threads" << std::endl;
		for (int i = 0; i < 10; ++i) {
			auto pair = slmp.getIterInfo();
			std::cout << "iter: " << pair.first << "  numProb: " << pair.second << std::endl;
			std::this_thread::sleep_for(500ms);
		}

		std::cout << "aborting" << std::endl;
		slmp.abort();


		std::cout << "after aborting" << std::endl;
		auto pair = slmp.getIterInfo();
		std::cout << "iter: " << pair.first << "  numProb: " << pair.second << std::endl;

		std::cout << "waiting for results" << std::endl;

		optim::SLMPResult result = res.get();

		std::cout << "after waiting for results" << std::endl;
		pair = slmp.getIterInfo();
		std::cout << "iter: " << pair.first << "  numProb: " << pair.second << std::endl;

		std::cout << "got result" << std::endl;


		run_thread.join();

		std::cout << "No crash, Success!" << std::endl;
	}


}



int main() {
	try {
		test_threading(200000);
	}
	catch (c10::Error e) {
		std::cout << e.what() << std::endl;
	}
}
