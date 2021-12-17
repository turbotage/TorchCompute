
#include "../compute.hpp"

void slmp_cpu_adc_diffexpr_specific(int n, bool print)
{
	using namespace tc;
	using namespace torch::indexing;

	std::cout << "ADC model" << std::endl;
	std::cout << "per problem b-vals : eval_and_diff" << std::endl;

	tc::optim::SLMPSettings settings;
	torch::TensorOptions dops;
	dops = dops.dtype(torch::kFloat64);
	auto params = torch::rand({ n, 2 }, dops);
	params.index_put_({ Slice(), 0 }, 1000.0f);
	params.index_put_({ Slice(), 1 }, 0.002f);

	auto b = torch::rand({ 1, 4 }, dops);
	b.index_put_({ 0, 0 }, 200.0f);
	b.index_put_({ 0, 1 }, 400.0f);
	b.index_put_({ 0, 2 }, 600.0f);
	b.index_put_({ 0, 3 }, 800.0f);

	{
		using namespace torch::indexing;

		std::string expr = "$S0*exp(-$b*$ADC)";
		std::vector<std::string> diffexpr;
		diffexpr.push_back("exp(-$b*$ADC)");
		diffexpr.push_back("-$b*$S0*exp(-$b*$ADC)");

		std::unordered_map<std::string, int> parmap;
		parmap["$S0"] = 0;
		parmap["$ADC"] = 1;

		std::unordered_map<std::string, int> constsmap;
		constsmap["$b"] = 0;

		std::unique_ptr<optim::Model> pModel = std::make_unique<optim::Model>(expr, diffexpr, parmap, std::nullopt, constsmap);
		pModel->setParameters(params);
		pModel->setConstants(std::vector<torch::Tensor>{b});

		torch::Tensor data = torch::empty({ n, 4 }, dops);
		pModel->eval(data);

		auto guess = torch::empty({ n, 2 }, dops);
		guess.index_put_({ Slice(), 0 }, 200.0f);
		guess.index_put_({ Slice(), 1 }, 0.005f);
		pModel->setParameters(guess);

		settings.pModel = std::move(pModel);
		settings.data = data;
		settings.maxIter = 40;

		if (print) {
			std::cout << "true params: " << params << std::endl;
		}

		optim::SLMP slmp(settings);
		optim::SLMPResult res = slmp.eval();
		auto par = slmp.getIterInfo();
		std::cout << "iter: " << par.first << std::endl;


		if (print) {
			std::cout << "found params: " << res.finalParameters << std::endl;
		}

		std::cout << "No crash, Success!" << std::endl;
	}
}

void slmp_cpu_adc_expr_specific(int n, bool print)
{
	using namespace tc;
	using namespace torch::indexing;

	std::cout << "ADC model" << std::endl;
	std::cout << "per problem b-vals : eval_and_diff" << std::endl;

	tc::optim::SLMPSettings settings;
	torch::TensorOptions dops;
	dops = dops.dtype(torch::kFloat64);
	auto params = torch::rand({ n, 2 }, dops);
	params.index_put_({ Slice(), 0 }, 1000.0f);
	params.index_put_({ Slice(), 1 }, 0.002f);

	auto b = torch::rand({ 1, 4 }, dops);
	b.index_put_({ 0, 0 }, 200.0f);
	b.index_put_({ 0, 1 }, 400.0f);
	b.index_put_({ 0, 2 }, 600.0f);
	b.index_put_({ 0, 3 }, 800.0f);

	{
		using namespace torch::indexing;

		std::string expr = "$S0*exp(-$b*$ADC)";

		std::unordered_map<std::string, int> parmap;
		parmap["$S0"] = 0;
		parmap["$ADC"] = 1;

		std::unordered_map<std::string, int> constsmap;
		constsmap["$b"] = 0;

		std::unique_ptr<optim::Model> pModel = std::make_unique<optim::Model>(expr, parmap, std::nullopt, constsmap);
		pModel->setParameters(params);
		pModel->setConstants(std::vector<torch::Tensor>{b});

		torch::Tensor data = torch::empty({ n, 4 }, dops);
		pModel->eval(data);

		auto guess = torch::empty({ n, 2 }, dops);
		guess.index_put_({ Slice(), 0 }, 200.0f);
		guess.index_put_({ Slice(), 1 }, 0.005f);
		pModel->setParameters(guess);

		settings.pModel = std::move(pModel);
		settings.data = data;
		settings.maxIter = 40;

		if (print) {
			std::cout << "true params: " << params << std::endl;
		}

		optim::SLMP slmp(settings);
		optim::SLMPResult res = slmp.eval();
		auto par = slmp.getIterInfo();
		std::cout << "iter: " << par.first << std::endl;


		if (print) {
			std::cout << "found params: " << res.finalParameters << std::endl;
		}

		std::cout << "No crash, Success!" << std::endl;
	}
}

void slmp_cpu_adc_expr(int n, bool print)
{
	using namespace tc;

	std::cout << "ADC model" << std::endl;
	std::cout << "per problem b-vals : eval_and_diff" << std::endl;

	tc::optim::SLMPSettings settings;
	torch::TensorOptions dops;
	dops = dops.dtype(torch::kFloat64);
	auto params = torch::rand({ n, 2 }, dops);
	auto b = torch::rand({ 1, 4 }, dops);

	{
		using namespace torch::indexing;

		std::string expr = "$S0*exp(-$b*$ADC)";
		std::vector<std::string> diffexpr;
		diffexpr.push_back("exp(-$b*$ADC)");
		diffexpr.push_back("-$b*exp(-$b*$ADC)");

		std::unordered_map<std::string, int> parmap;
		parmap["$S0"] = 0;
		parmap["$ADC"] = 1;

		std::unordered_map<std::string, int> constsmap;
		constsmap["$b"] = 0;

		std::unique_ptr<optim::Model> pModel = std::make_unique<optim::Model>(expr, diffexpr, parmap, std::nullopt, constsmap);
		pModel->setParameters(params);
		pModel->setConstants(std::vector<torch::Tensor>{b});

		torch::Tensor data = torch::empty({ n, 4 }, dops);
		pModel->eval(data);

		auto guess = torch::empty({ n, 2 }, dops);
		pModel->setParameters(guess);

		settings.pModel = std::move(pModel);
		settings.data = data;
		settings.maxIter = 40;

		if (print) {
			std::cout << "true params: " << params << std::endl;
		}

		optim::SLMP slmp(settings);
		optim::SLMPResult res = slmp.eval();
		auto par = slmp.getIterInfo();
		std::cout << "iter: " << par.first << std::endl;


		if (print) {
			std::cout << "found params: " << res.finalParameters << std::endl;
		}

		std::cout << "No crash, Success!" << std::endl;
	}
}

void slmp_cuda_adc_expr(int n, bool print)
{
	using namespace tc;

	std::cout << "ADC model" << std::endl;
	std::cout << "per problem b-vals : eval_and_diff" << std::endl;

	tc::optim::SLMPSettings settings;
	torch::TensorOptions dops;
	dops = dops.dtype(torch::kFloat64).device(torch::Device("cuda:0"));

	auto params = torch::rand({ n, 2 }, dops);
	auto b = torch::rand({ 1, 4 }, dops);

	{
		using namespace torch::indexing;

		std::string expr = "$S0*exp(-$b*$ADC)";
		std::vector<std::string> diffexpr;
		diffexpr.push_back("exp(-$b*$ADC)");
		diffexpr.push_back("-$b*exp(-$b*$ADC)");

		std::unordered_map<std::string, int> parmap;
		parmap["$S0"] = 0;
		parmap["$ADC"] = 1;

		std::unordered_map<std::string, int> constsmap;
		constsmap["$b"] = 0;

		std::unique_ptr<optim::Model> pModel = std::make_unique<optim::Model>(expr, diffexpr, parmap, std::nullopt, constsmap);
		pModel->setParameters(params);
		pModel->setConstants(std::vector<torch::Tensor>{b});
		pModel->to(torch::Device("cuda:0"));

		torch::Tensor data = torch::empty({ n, 4 }, dops);
		pModel->eval(data);

		auto guess = torch::empty({ n, 2 }, dops);
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

		optim::SLMP slmp(settings);
		optim::SLMPResult res = slmp.eval();
		auto par = slmp.getIterInfo();
		std::cout << "iter: " << par.first << std::endl;


		if (print) {
			std::cout << "found params: " << res.finalParameters << std::endl;
		}

		std::cout << "No crash, Success!" << std::endl;
	}
}


int main() {
	/*
	try {
		slmp_cpu_adc_diffexpr_specific(1, true);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}
	*/

	try {
		slmp_cpu_adc_expr_specific(1, true);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}

	try {
		slmp_cpu_adc_expr(10000, false);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}

	try {
		slmp_cuda_adc_expr(100000, false);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}
}
