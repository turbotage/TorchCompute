
#include "../compute.hpp"

void strp_cpu_ir_anal_specific(int n, int iter, bool print) {

	using namespace tc;

	torch::InferenceMode im_guard;

	std::cout << "ADC model" << std::endl;
	std::cout << "per problem b-vals : eval_and_diff" << std::endl;

	using namespace torch::indexing;

	std::unique_ptr<optim::Model> pModel = std::make_unique<optim::Model>(models::psir_eval_and_diff);

	torch::TensorOptions dops;
	dops = dops.dtype(torch::kFloat64);

	auto params = torch::rand({ n, 2 }, dops);
	params.index_put_({ Slice(), 0 }, 10000.0f);
	params.index_put_({ Slice(), 1 }, 800.0f);

	torch::Tensor TR = torch::full({ 1 }, 5000.0f, dops);
	torch::Tensor TI = torch::empty({ 1, 4 }, dops);
	TI.index_put_({ 0, 0 }, 200.0f);
	TI.index_put_({ 0, 1 }, 400.0f);
	TI.index_put_({ 0, 2 }, 1200.0f);
	TI.index_put_({ 0, 3 }, 2000.0f);
	torch::Tensor FA = torch::full({ 1 }, 180.0f * 3.141592 / 180.0f);
	FA = torch::cos(FA) - 1;
	std::vector<torch::Tensor> constants{ TR, TI, FA };


	pModel->setParameters(params);
	pModel->setConstants(constants);

	torch::Tensor data = torch::empty({ n, 4 }, dops);
	pModel->eval(data);
	if (print) {
		std::cout << "data:\n" << data << std::endl;
	}

	auto guess = torch::empty({ n, 2 }, dops);
	guess.index_put_({ Slice(), 0 }, 1000.0f);
	guess.index_put_({ Slice(), 1 }, 100.0f);
	pModel->setParameters(guess);
	
	auto resJ = tc::optim::STRP::default_res_J_setup(*pModel, data);
	auto delta = tc::optim::STRP::default_delta_setup(pModel->getParameters(), 1.0f);
	auto scaling = tc::optim::STRP::default_scaling_setup(resJ.second);

	if (print) {
		std::cout << "true params: " << params << std::endl;
		std::cout << "start residuals: " << resJ.first << std::endl;
		std::cout << "start jacobian: " << resJ.second << std::endl;
		std::cout << "start deltas: " << delta << std::endl;
		std::cout << "scaling: " << scaling << std::endl;
	}

	tc::optim::OptimizerSettings optsettings(std::move(pModel), data, iter);

	tc::optim::STRPSettings strpsettings(std::move(optsettings), resJ.first, resJ.second, delta, scaling);

	auto t1 = std::chrono::steady_clock::now();


	optim::STRP strp = optim::STRP::make(std::move(strpsettings));
	strp.run();

	auto t2 = std::chrono::steady_clock::now();
	std::cout << "time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	if (print) {
		std::cout << "found params: " << strp.last_parameters() << std::endl;
	}

	std::cout << "No crash, Success!" << std::endl;

}

void strp_cuda_ir_anal_specific(int n, int iter, bool print) {

	using namespace tc;

	torch::InferenceMode im_guard;

	std::cout << "ADC model" << std::endl;
	std::cout << "per problem b-vals : eval_and_diff" << std::endl;

	using namespace torch::indexing;

	std::unique_ptr<optim::Model> pModel = std::make_unique<optim::Model>(models::psir_eval_and_diff);

	torch::TensorOptions dops;
	dops = dops.dtype(torch::kFloat64).device(torch::Device("cuda:0"));

	auto params = torch::rand({ n, 2 }, dops);
	params.index_put_({ Slice(), 0 }, 10000.0f);
	params.index_put_({ Slice(), 1 }, 800.0f);

	torch::Tensor TR = torch::full({ 1 }, 5000.0f, dops);
	torch::Tensor TI = torch::empty({ 1, 4 }, dops);
	TI.index_put_({ 0, 0 }, 200.0f);
	TI.index_put_({ 0, 1 }, 400.0f);
	TI.index_put_({ 0, 2 }, 1200.0f);
	TI.index_put_({ 0, 3 }, 2000.0f);
	torch::Tensor FA = torch::full({ 1 }, 180.0f * 3.141592 / 180.0f);
	FA = torch::cos(FA) - 1;
	std::vector<torch::Tensor> constants{ TR, TI, FA };


	pModel->setParameters(params);
	pModel->setConstants(constants);

	torch::Tensor data = torch::empty({ n, 4 }, dops);
	pModel->eval(data);
	if (print) {
		std::cout << "data:\n" << data << std::endl;
	}

	auto guess = torch::empty({ n, 2 }, dops);
	guess.index_put_({ Slice(), 0 }, 1000.0f);
	guess.index_put_({ Slice(), 1 }, 100.0f);
	pModel->setParameters(guess);

	auto resJ = tc::optim::STRP::default_res_J_setup(*pModel, data);
	auto delta = tc::optim::STRP::default_delta_setup(pModel->getParameters());
	auto scaling = tc::optim::STRP::default_scaling_setup(resJ.second);

	if (print) {
		std::cout << "true params: " << params << std::endl;
		std::cout << "start residuals: " << resJ.first << std::endl;
		std::cout << "start jacobian: " << resJ.second << std::endl;
		std::cout << "start deltas: " << delta << std::endl;
		std::cout << "scaling: " << scaling << std::endl;
	}

	tc::optim::OptimizerSettings optsettings(std::move(pModel), data, iter);

	tc::optim::STRPSettings strpsettings(std::move(optsettings), resJ.first, resJ.second, delta, scaling);

	auto t1 = std::chrono::steady_clock::now();


	optim::STRP strp = optim::STRP::make(std::move(strpsettings));
	strp.run();

	auto t2 = std::chrono::steady_clock::now();
	std::cout << "time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	if (print) {
		std::cout << "found params: " << strp.last_parameters() << std::endl;
	}

	std::cout << "No crash, Success!" << std::endl;

}


int main() {


	
	try {
		strp_cpu_ir_anal_specific(2, 53, true);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}
	catch (std::runtime_error e2) {
		std::cout << e2.what() << std::endl;
	}
	

	


}