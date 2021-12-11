
#include "../compute.hpp"

void test_diff_adc(int n, bool print) {
	using namespace tc;


	torch::TensorOptions dops;
	dops = dops.dtype(torch::kFloat64).device(torch::Device("cuda:0"));
	auto params = torch::rand({ n, 2 }, dops);
	auto b = torch::rand({ 1, 4 }, dops);

	// Model 1
	std::unique_ptr<optim::Model> pModel1 = std::make_unique<optim::Model>(models::adc_eval_and_diff);
	pModel1->setParameters(params);
	pModel1->setConstants(std::vector<torch::Tensor>{ b });
	pModel1->to(dops.device());

	// Model 2
	std::string expr = "$S0*exp(-$b*$ADC)";
	std::vector<std::string> diffexpr;
	diffexpr.push_back("exp(-$b*$ADC)");
	diffexpr.push_back("-$b*exp(-$b*$ADC)");

	std::unordered_map<std::string, int> parmap;
	parmap["$S0"] = 0;
	parmap["$ADC"] = 1;

	std::unordered_map<std::string, int> constsmap;
	constsmap["$b"] = 0;

	std::unique_ptr<optim::Model> pModel2 = std::make_unique<optim::Model>(expr, diffexpr, parmap, std::nullopt, constsmap);
	pModel2->setParameters(params);
	pModel2->setConstants(std::vector<torch::Tensor>{b});
	pModel2->to(dops.device());

	torch::Tensor eval = torch::empty({ n, 2 }, dops);
	torch::Tensor J = torch::empty({ n, 4, 2 }, dops);

	pModel1->eval_diff(eval, J);

	std::cout << "analytical:" << std::endl;
	std::cout << "eval:\n" << eval << std::endl;
	std::cout << "J:\n" << J << std::endl;

	pModel2->eval_diff(eval, J);

	std::cout << "diff-expression" << std::endl;
	std::cout << "eval:\n" << eval << std::endl;
	std::cout << "J:\n" << J << std::endl;

}



void test_diff_vfa(int n, bool print) {
	using namespace tc;

	torch::TensorOptions dops;
	dops = dops.dtype(torch::kFloat64).device(torch::Device("cuda:0"));
	auto params = torch::rand({ n, 2 }, dops);
	auto fa = torch::rand({ 1, 3 }, dops);
	auto TR = torch::full({ 1 }, 1, dops);

	// Model 1
	std::unique_ptr<optim::Model> pModel1 = std::make_unique<optim::Model>(models::vfa_eval_and_diff);
	pModel1->setParameters(params);
	pModel1->setConstants(std::vector<torch::Tensor>{ TR, fa });
	pModel1->to(dops.device());

	// Model 2
	std::string expr = "$S0*sin($FA)*(1-exp(-$TR/$T1))/(1-exp(-$TR/$T1)*cos($FA))";
	std::vector<std::string> diffexpr;
	diffexpr.push_back("sin($FA)*(1-exp(-$TR/$T1))/(1-exp(-$TR/$T1)*cos($FA))");
	diffexpr.push_back("$S0*exp(-$TR/$T1)*$TR*sin($FA)*(cos($FA)-1)/(square(1-exp(-$TR/$T1)*cos($FA))*square($T1))");

	std::unordered_map<std::string, int> parmap;
	parmap["$S0"] = 0;
	parmap["$T1"] = 1;

	std::unordered_map<std::string, int> constsmap;
	constsmap["$TR"] = 0;
	constsmap["$FA"] = 1;

	std::unique_ptr<optim::Model> pModel2 = std::make_unique<optim::Model>(expr, diffexpr, parmap, std::nullopt, constsmap);
	pModel2->setParameters(params);
	pModel2->setConstants(std::vector<torch::Tensor>{ TR, fa });
	pModel2->to(dops.device());


	torch::Tensor eval = torch::empty({ n, 2 }, dops);
	torch::Tensor J = torch::empty({ n, 3, 2 }, dops);

	pModel1->eval_diff(eval, J);

	std::cout << "analytical:" << std::endl;
	std::cout << "eval:\n" << eval << std::endl;
	std::cout << "J:\n" << J << std::endl;

	pModel2->eval_diff(eval, J);

	std::cout << "diff-expression" << std::endl;
	std::cout << "eval:\n" << eval << std::endl;
	std::cout << "J:\n" << J << std::endl;

}


int main() {
	try {
		test_diff_adc(3, true);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}

	try {
		test_diff_vfa(4, true);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}
	catch (std::runtime_error e2) {
		std::cout << e2.what() << std::endl;
	}

}