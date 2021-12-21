
#include "../compute.hpp"


void compare_graph_vs_anal() {
	using namespace tc;
	using namespace torch::indexing;

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

		std::unique_ptr<optim::Model> pModel2 = std::make_unique<optim::Model>(expr, parmap, ppimap, constsmap);
		pModel2->setParameters(params);
		pModel2->setPerProblemInputs(ppi);
		pModel2->setConstants(std::vector<torch::Tensor>{ TR });
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

		std::unique_ptr<optim::Model> pModel2 = std::make_unique<optim::Model>(expr, parmap, ppimap, std::nullopt);
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

	try {
		compare_graph_vs_anal();
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}

}