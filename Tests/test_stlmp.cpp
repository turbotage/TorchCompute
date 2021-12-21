
#include "../compute.hpp"

void stlmp_cpu_adc_anal_specific(int n, bool print) {

	using namespace tc;

	std::cout << "ADC model" << std::endl;
	std::cout << "per problem b-vals : eval_and_diff" << std::endl;

	tc::optim::STLMPSettings settings;

	std::unique_ptr<tc::optim::Model> pModel;
	{
		using namespace torch::indexing;

		std::unique_ptr<optim::Model> pModel = std::make_unique<optim::Model>(models::adc_eval_and_diff);

		torch::TensorOptions dops;
		dops = dops.dtype(torch::kFloat64);

		auto params = torch::rand({ n, 2 }, dops);
		params.index_put_({ Slice(), 0 }, 1000.0f);
		params.index_put_({ Slice(), 1 }, 0.002f);

		auto ppi = torch::rand({ n, 4, 1 }, dops);
		ppi.index_put_({ Slice(), 0, 0 }, 200.0f);
		ppi.index_put_({ Slice(), 1, 0 }, 400.0f);
		ppi.index_put_({ Slice(), 2, 0 }, 600.0f);
		ppi.index_put_({ Slice(), 3, 0 }, 800.0f);

		pModel->setParameters(params);
		pModel->setPerProblemInputs(ppi);

		torch::Tensor data = torch::empty({ n, 4 }, dops);
		pModel->eval(data);

		std::cout << "data:\n" << data << std::endl;

		auto guess = torch::empty({ n, 2 }, dops);
		guess.index_put_({ Slice(), 0 }, 200.0f);
		guess.index_put_({ Slice(), 1 }, 0.05f);
		pModel->setParameters(guess);

		settings.pModel = std::move(pModel);
		settings.data = data;
		settings.maxIter = 200;
		settings.tolerance = 1e-5;

		if (print) {
			std::cout << "true params: " << params << std::endl;
		}

		optim::STLMP stlmp(settings);
		optim::STLMPResult res = stlmp.eval();
		auto par = stlmp.getIterInfo();
		std::cout << "iter: " << par.first << std::endl;


		if (print) {
			std::cout << "found params: " << res.finalParameters << std::endl;
		}

		std::cout << "No crash, Success!" << std::endl;
	}

}

void stlmp_cpu_vfa_anal_specific(int n, bool print) {

	using namespace tc;

	std::cout << "VFA model" << std::endl;
	std::cout << "per problem b-vals : eval_and_diff" << std::endl;

	tc::optim::STLMPSettings settings;

	std::unique_ptr<tc::optim::Model> pModel;
	{
		using namespace torch::indexing;

		std::string expr = "$S0*sin($FA)*(1-exp(-$TR/$T1))/(1-exp(-$TR/$T1)*cos($FA))";

		std::unordered_map<std::string, int> parmap;
		parmap["$S0"] = 0;
		parmap["$T1"] = 1;

		std::unordered_map<std::string, int> constsmap;
		constsmap["$TR"] = 0;
		constsmap["$FA"] = 1;

		std::unique_ptr<optim::Model> pModel = std::make_unique<optim::Model>(expr, parmap, std::nullopt, constsmap);

		torch::TensorOptions dops;
		dops = dops.dtype(torch::kFloat64);

		auto params = torch::rand({ n, 2 }, dops);
		params.index_put_({ Slice(), 0 }, 1000.0f);
		params.index_put_({ Slice(), 1 }, 0.5f);

		auto TR = torch::full({ 1 }, 1.0f, dops);

		auto fa = torch::rand({ 1, 4 }, dops);
		fa.index_put_({ 0, 0 }, 10.0f * 3.141592f / 180.0f);
		fa.index_put_({ 0, 1 }, 25.0f * 3.141592f / 180.0f);
		fa.index_put_({ 0, 2 }, 50.0f * 3.141592f / 180.0f);
		fa.index_put_({ 0, 3 }, 75.0f * 3.141592f / 180.0f);

		pModel->setParameters(params);
		pModel->setConstants(std::vector<torch::Tensor>{TR, fa});

		torch::Tensor data = torch::empty({ n, 4 }, dops);
		data += torch::rand({ n,4 }, dops);

		pModel->eval(data);

		std::cout << "data:\n" << data << std::endl;

		auto guess = torch::empty({ n, 2 }, dops);
		guess.index_put_({ Slice(), 0 }, 1e4f);
		guess.index_put_({ Slice(), 1 }, 0.05f);
		pModel->setParameters(guess);

		settings.pModel = std::move(pModel);
		settings.data = data;
		settings.maxIter = 200;
		settings.tolerance = 1e-5;

		if (print) {
			std::cout << "true params: " << params << std::endl;
		}

		optim::STLMP stlmp(settings);
		optim::STLMPResult res = stlmp.eval();
		auto par = stlmp.getIterInfo();
		std::cout << "iter: " << par.first << std::endl;


		if (print) {
			std::cout << "found params: " << res.finalParameters << std::endl;
		}

		std::cout << "No crash, Success!" << std::endl;
	}

}

int main() {


	
	try {
		stlmp_cpu_adc_anal_specific(1, true);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}
	catch (std::runtime_error e2) {
		std::cout << e2.what() << std::endl;
	}
	


	try {
		stlmp_cpu_vfa_anal_specific(1, true);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}
	catch (std::runtime_error e2) {
		std::cout << e2.what() << std::endl;
	}

}


