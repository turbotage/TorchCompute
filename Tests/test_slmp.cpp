
#include "../compute.hpp"

void run_cpu_test_anal(int n, bool print) {

	std::cout << "ADC model" << std::endl;
	std::cout << "per problem b-vals : eval_and_diff" << std::endl;

	optim::SLMPSettings settings;
	
	std::unique_ptr<optim::Model> pModel;
	{
		using namespace torch::indexing;

		pModel = std::make_unique<optim::Model>(models::adc_eval_and_diff);

		torch::TensorOptions dops;
		dops.dtype(torch::kFloat64);

		auto params =	torch::rand({ n, 2 }, dops);

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

		optim::SLMPResult res = optim::SLMP(settings).eval();

		if (print) {
			std::cout << "found params: " << res.finalParameters << std::endl;
		}

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

		optim::SLMPResult res = optim::SLMP(settings).eval();

		if (print) {
			std::cout << "found params: " << res.finalParameters << std::endl;
		}

	}

}

void run_cuda_test_anal(int n, bool print) {
	std::cout << "ADC model" << std::endl;
	std::cout << "per problem b-vals : eval_and_diff : switch to cpu" << std::endl;

	optim::SLMPSettings settings;

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

		optim::SLMPResult res = optim::SLMP(settings).eval();

		if (print) {
			std::cout << "found params: " << res.finalParameters << std::endl;
		}

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

		optim::SLMPResult res = optim::SLMP(settings).eval();

		if (print) {
			std::cout << "found params: " << res.finalParameters << std::endl;
		}

	}

}

void compare_graph_vs_anal() {

	torch::TensorOptions dops;
	dops = dops.dtype(torch::kFloat64).device(torch::Device("cuda:0"));
	int n = 3;

	auto params = torch::rand({ n, 2 }, dops);
	auto ppi = torch::rand({ n, 4, 1 }, dops);

	std::cout << "VFA-model\n\n";

	{
		auto TR = torch::full({ 1 }, 1, dops);

		// Model 1
		std::unique_ptr<optim::Model> pModel1 = std::make_unique<optim::Model>(models::vfa_eval_and_diff);
		pModel1->setParameters(params);
		pModel1->setPerProblemInputs(ppi);
		pModel1->setConstants(std::vector<torch::Tensor>{ TR });
		pModel1->to(dops.device());

		// Model 2
		std::string expr = "$X0*sin($D0)*(1-exp(-$TR/$X1))/(1-exp(-$TR/$X1)*cos($D0))";

		std::unordered_map<std::string, int> ppimap;
		ppimap["$D0"] = 0;

		std::unordered_map<std::string, int> parmap;
		parmap["$X0"] = 0;
		parmap["$X1"] = 1;

		std::unordered_map<std::string, int> constsmap;
		constsmap["$TR"] = 0;

		std::unique_ptr<optim::Model> pModel2 = std::make_unique<optim::Model>(expr, ppimap, parmap, constsmap);
		pModel2->setParameters(params);
		pModel2->setPerProblemInputs(ppi);
		pModel2->setConstants(std::vector<torch::Tensor>{ TR });
		pModel2->to(dops.device());

		std::cout << pModel2->getReadableExpressionTree() << std::endl;

		
		torch::Tensor eval	= torch::empty({n, 2}, dops);
		torch::Tensor J		= torch::empty({n, 4, 2}, dops);

		pModel1->eval_diff(eval, J);

		std::cout << "analytical:" << std::endl;
		std::cout << "eval:\n" << eval << std::endl;
		std::cout << "J:\n" << J << std::endl;

		pModel2->eval_diff(eval, J);

		std::cout << "graph-calculated:" << std::endl;
		std::cout << "eval:\n" << eval << std::endl;
		std::cout << "J:\n" << J << std::endl;
		
	}

	std::cout << "\n ADC-model \n\n";

	{
		// Model 1
		std::unique_ptr<optim::Model> pModel1 = std::make_unique<optim::Model>(models::adc_eval_and_diff);
		pModel1->setParameters(params);
		pModel1->setPerProblemInputs(ppi);
		pModel1->to(dops.device());

		// Model 2
		std::string expr = "$X0*exp(-$D0*$X1)";

		std::unordered_map<std::string, int> ppimap;
		ppimap["$D0"] = 0;

		std::unordered_map<std::string, int> parmap;
		parmap["$X0"] = 0;
		parmap["$X1"] = 1;

		std::unique_ptr<optim::Model> pModel2 = std::make_unique<optim::Model>(expr, ppimap, parmap, std::nullopt);
		pModel2->setParameters(params);
		pModel2->setPerProblemInputs(ppi);
		pModel2->to(dops.device());

		std::cout << pModel2->getReadableExpressionTree() << std::endl;


		torch::Tensor eval = torch::empty({ n, 2 }, dops);
		torch::Tensor J = torch::empty({ n, 4, 2 }, dops);

		pModel1->eval_diff(eval, J);

		std::cout << "analytical:" << std::endl;
		std::cout << "eval:\n" << eval << std::endl;
		std::cout << "J:\n" << J << std::endl;

		pModel2->eval_diff(eval, J);

		std::cout << "graph-calculated:" << std::endl;
		std::cout << "eval:\n" << eval << std::endl;
		std::cout << "J:\n" << J << std::endl;

	}

}

int main() {

	/*try {
		run_cpu_test(4, true);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}*/

	/*try {
		run_cpu_test_anal(40000, false);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}*/

	/*try {
		run_cuda_test_anal(200000, false);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}*/

	/*try {
		compare_graph_vs_anal();
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}*/

}

