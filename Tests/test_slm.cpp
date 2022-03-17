#include "../compute.hpp"

void strp_cpu_ir_anal_specific(int32_t n, int32_t iter, bool print) {

	using namespace tc;

	torch::InferenceMode im_guard;

	auto mp_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_psir_eval_jac_hess, tc::models::mp_psir_diff, tc::models::mp_psir_diff2);

	torch::TensorOptions dops;
	dops = dops.dtype(torch::kFloat64);

	auto params = torch::rand({ n, 2 }, dops);
	params.select(1, 0).fill_(10000.0f);
	params.select(1, 1).fill_(800.0f);

	torch::Tensor TR = torch::full({ 1 }, 5000.0f, dops);
	torch::Tensor TI = torch::empty({ 1, 4 }, dops);
	TI.select(1, 0).fill_(200.0f);
	TI.select(1, 1).fill_(400.0f);
	TI.select(1, 2).fill_(1200.0f);
	TI.select(1, 3).fill_(2000.0f);

	torch::Tensor FA = torch::full({ 1 }, 180.0f * 3.141592 / 180.0f, dops);
	FA = torch::cos(FA) - 1;
	std::vector<torch::Tensor> consts{ TR, TI, FA };

	mp_model->parameters() = params;
	mp_model->constants() = consts;

	torch::Tensor data = torch::empty({ n, 4 }, dops);
	mp_model->eval(data);
	if (print) {
		std::cout << "data:\n" << data << std::endl;
	}

	auto guess = torch::empty({ n, 2 }, dops);
	guess.select(1, 0).fill_(100.0f);
	guess.select(1, 1).fill_(10.0f);

	mp_model->parameters() = guess;

	auto resJ = tc::optim::MP_SLM::default_res_J_setup(*mp_model, data);
	auto lambda = tc::optim::MP_SLM::default_lambda_setup(mp_model->parameters(), 1.0f);
	auto scaling = tc::optim::MP_SLM::default_scaling_setup(resJ.second);

	if (print) {
		std::cout << "true params: " << params << std::endl;
		std::cout << "start residuals: " << resJ.first << std::endl;
		std::cout << "start jacobian: " << resJ.second << std::endl;
		std::cout << "start deltas: " << lambda << std::endl;
		std::cout << "scaling: " << scaling << std::endl;
	}

	tc::optim::MP_OptimizerSettings optsettings(std::move(mp_model), data);

	tc::optim::MP_SLMSettings strpsettings(std::move(optsettings), resJ.first, resJ.second, lambda, scaling);

	auto t1 = std::chrono::steady_clock::now();


	auto strp = optim::MP_SLM::make(std::move(strpsettings));
	strp->run(iter);

	auto t2 = std::chrono::steady_clock::now();
	std::cout << "time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	if (print) {
		std::cout << "found params: " << strp->last_parameters() << std::endl;
	}

	std::cout << "No crash, Success!" << std::endl;

}

void strp_cuda_ir_anal_specific(int32_t n, int32_t iter, bool print) {

	using namespace tc;

	torch::InferenceMode im_guard;

	auto mp_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_psir_eval_jac_hess, tc::models::mp_psir_diff, tc::models::mp_psir_diff2);

	torch::TensorOptions dops;
	dops = dops.dtype(torch::kFloat64).device(torch::Device("cuda:0"));

	auto params = torch::rand({ n, 2 }, dops);
	params.select(1, 0).fill_(10000.0f);
	params.select(1, 1).fill_(800.0f);

	torch::Tensor TR = torch::full({ 1 }, 5000.0f, dops);
	torch::Tensor TI = torch::empty({ 1, 4 }, dops);
	TI.select(1, 0).fill_(200.0f);
	TI.select(1, 1).fill_(400.0f);
	TI.select(1, 2).fill_(1200.0f);
	TI.select(1, 3).fill_(2000.0f);

	torch::Tensor FA = torch::full({ 1 }, 180.0f * 3.141592 / 180.0f, dops);
	FA = torch::cos(FA) - 1;
	std::vector<torch::Tensor> consts{ TR, TI, FA };

	mp_model->parameters() = params;
	mp_model->constants() = consts;

	torch::Tensor data = torch::empty({ n, 4 }, dops);
	mp_model->eval(data);
	if (print) {
		std::cout << "data:\n" << data << std::endl;
	}

	auto guess = torch::empty({ n, 2 }, dops);
	guess.select(1, 0).fill_(100.0f);
	guess.select(1, 1).fill_(10.0f);

	mp_model->parameters() = guess;

	auto resJ = tc::optim::MP_SLM::default_res_J_setup(*mp_model, data);
	auto lambda = tc::optim::MP_SLM::default_lambda_setup(mp_model->parameters(), 1.0f);
	auto scaling = tc::optim::MP_SLM::default_scaling_setup(resJ.second);

	if (print) {
		std::cout << "true params: " << params << std::endl;
		std::cout << "start residuals: " << resJ.first << std::endl;
		std::cout << "start jacobian: " << resJ.second << std::endl;
		std::cout << "start deltas: " << lambda << std::endl;
		std::cout << "scaling: " << scaling << std::endl;
	}

	tc::optim::MP_OptimizerSettings optsettings(std::move(mp_model), data);

	tc::optim::MP_SLMSettings strpsettings(std::move(optsettings), resJ.first, resJ.second, lambda, scaling);

	auto t1 = std::chrono::steady_clock::now();


	auto strp = optim::MP_SLM::make(std::move(strpsettings));
	strp->run(iter);

	auto t2 = std::chrono::steady_clock::now();
	std::cout << "time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	if (print) {
		std::cout << "found params: " << strp->last_parameters() << std::endl;
	}

	std::cout << "No crash, Success!" << std::endl;

}


int main() {

	try {
		//strp_cpu_ir_anal_specific(2, 0, true);
		//strp_cpu_ir_anal_specific(2, 1, true);
		//strp_cpu_ir_anal_specific(2, 2, true);
		//strp_cpu_ir_anal_specific(2, 3, true);
		//strp_cpu_ir_anal_specific(2, 4, true);
		//strp_cpu_ir_anal_specific(2, 5, true);
		strp_cpu_ir_anal_specific(2, 18, true);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}
	catch (std::runtime_error e2) {
		std::cout << e2.what() << std::endl;
	}
	//strp_cpu_ir_anal_specific(5000, 2, false);
	//strp_cpu_ir_anal_specific(1000, 2, false);
	/*
	try {
		strp_cuda_ir_anal_specific(10000, 2, false);
		strp_cuda_ir_anal_specific(500, 2, false);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}
	catch (std::runtime_error e2) {
		std::cout << e2.what() << std::endl;
	}
	*/

	/*
	strp_cpu_ir_anal_specific(1, 25, true);
	strp_cpu_ir_anal_specific(1, 30, true);
	strp_cpu_ir_anal_specific(1, 35, true);
	strp_cpu_ir_anal_specific(1, 50, true);
	strp_cpu_ir_anal_specific(1, 51, true);
	strp_cpu_ir_anal_specific(1, 52, true);
	strp_cpu_ir_anal_specific(1, 53, true);
	*/

}