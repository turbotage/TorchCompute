#include "../compute.hpp"

void strp_cpu_psir_anal_specific(int32_t n, int32_t iter, bool print) {

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

void strp_cuda_psir_anal_specific(int32_t n, int32_t iter, bool print) {

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


void strp_cpu_irmag_anal_specific(int32_t n, int32_t iter, bool print) {

	using namespace tc;

	torch::InferenceMode im_guard;

	auto mp_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_irmag_eval_jac_hess, tc::models::mp_irmag_diff, tc::models::mp_irmag_diff2);

	torch::TensorOptions dops;
	dops = dops.dtype(torch::kFloat64);

	auto params = torch::rand({ n, 2 }, dops);
	params.select(1, 0).fill_(10000.0f);
	params.select(1, 1).fill_(800.0f);

	torch::Tensor TR = torch::full({ 1 }, 5000.0f, dops);
	torch::Tensor TI = torch::empty({ 1, 10 }, dops);
	TI.select(1, 0).fill_(200.0f);
	TI.select(1, 1).fill_(300.0f);
	TI.select(1, 2).fill_(500.0f);
	TI.select(1, 3).fill_(550.0f);
	TI.select(1, 4).fill_(600.0f);
	TI.select(1, 5).fill_(650.0f);
	TI.select(1, 6).fill_(700.0f);
	TI.select(1, 7).fill_(800.0f);
	TI.select(1, 8).fill_(1200.0f);
	TI.select(1, 9).fill_(2000.0f);

	torch::Tensor FA = torch::full({ 1 }, 180.0f * 3.141592 / 180.0f, dops);
	FA = torch::cos(FA) - 1;
	std::vector<torch::Tensor> consts{ TR, TI, FA };

	mp_model->parameters() = params;
	mp_model->constants() = consts;

	torch::Tensor data = torch::empty({ n, 10 }, dops);
	mp_model->eval(data);
	if (print) {
		std::cout << "data:\n" << data << std::endl;
	}

	auto guess = torch::empty({ n, 2 }, dops);
	guess.select(1, 0).fill_(15000.0f);
	guess.select(1, 1).fill_(2000.0f);

	mp_model->parameters() = guess;

	auto resJ = tc::optim::MP_SLM::default_res_J_setup(*mp_model, data);
	auto lambda = tc::optim::MP_SLM::default_lambda_setup(mp_model->parameters(), 4.0f);
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

void strp_cuda_irmag_anal_specific(int32_t n, int32_t iter, bool print) {

	using namespace tc;

	torch::InferenceMode im_guard;

	auto mp_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_irmag_eval_jac_hess, tc::models::mp_irmag_diff, tc::models::mp_irmag_diff2);

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

void test_ivim_hessian() {
	auto param = torch::rand({ 2,4 });
	std::vector<torch::Tensor> constants = { torch::rand({3}).unsqueeze(0) };

	std::cout << "param: " << param << std::endl;
	std::cout << "constants: " << constants[0] << std::endl;

	auto values = torch::empty({ 2, 3 });
	auto jacobian = torch::empty({ 2, 3, 4 });
	auto hessian = torch::empty({ 2, 4, 4 });

	auto data = torch::empty({ 2,3 });

	tc::models::mp_ivim_eval_jac_hess(constants, param, data, std::nullopt, std::nullopt, std::nullopt);

	std::cout << "data: " << data << std::endl;

	tc::models::mp_ivim_eval_jac_hess(constants, param, values, jacobian, hessian, data);


	std::string expr = "$S0*($f*exp(-$b*$D1)+(1-$f)*exp(-$b*$D2))";
	std::vector<std::string> param_names = { "$S0", "$f", "$D1", "$D2" };
	std::vector<std::string> const_names = { "$b" };
	std::unique_ptr<tc::optim::MP_Model> model = std::make_unique<tc::optim::MP_Model>(expr, param_names, std::make_optional(const_names));

	model->constants() = constants;
	model->parameters() = param;

	auto values2 = torch::empty({ 2, 3 });
	auto jacobian2 = torch::empty({ 2, 3, 4 });
	auto hessian2 = torch::empty({ 2, 4, 4 });

	model->res_jac_hess(values2, jacobian2, hessian2, data);

	std::cout << "values: " << values << std::endl;
	std::cout << "values2: " << values2 << std::endl;
	std::cout << "jacobian: " << jacobian << std::endl;
	std::cout << "jacobian2: " << jacobian2 << std::endl;
	std::cout << "hessian: " << hessian << std::endl;
	std::cout << "hessian2: " << hessian2 << std::endl;
}

void test_irmag_hessian() {
	auto param = torch::rand({ 2,2 });
	std::vector<torch::Tensor> constants = { 
		torch::rand({3}).unsqueeze(0), 
		torch::rand({3}).unsqueeze(0),
		torch::rand({1}).unsqueeze(0)
	};

	std::cout << "param: " << param << std::endl;
	std::cout << "constants: " << constants[0] << std::endl;

	auto values = torch::empty({ 2, 3 });
	auto jacobian = torch::empty({ 2, 3, 2 });
	auto hessian = torch::empty({ 2, 2, 2 });

	auto data = torch::empty({ 2,3 });

	tc::models::mp_irmag_eval_jac_hess(constants, param, data, std::nullopt, std::nullopt, std::nullopt);

	std::cout << "data: " << data << std::endl;

	tc::models::mp_irmag_eval_jac_hess(constants, param, values, jacobian, hessian, data);


	std::string expr = "abs($S0*(1.0+$FA*exp(-$TI/$T1)+exp(-$TR/$T1)))";
	std::vector<std::string> param_names = { "$S0", "$T1"};
	std::vector<std::string> const_names = { "$TR", "$TI", "$FA" };
	std::unique_ptr<tc::optim::MP_Model> model = std::make_unique<tc::optim::MP_Model>(expr, param_names, std::make_optional(const_names));
	

	model->constants() = constants;
	model->parameters() = param;

	auto values2 = torch::empty({ 2, 3 });
	auto jacobian2 = torch::empty({ 2, 3, 2 });
	auto hessian2 = torch::empty({ 2, 2, 2 });

	model->res_jac_hess(values2, jacobian2, hessian2, data);

	std::cout << "values: " << values << std::endl;
	std::cout << "values2: " << values2 << std::endl;
	std::cout << "jacobian: " << jacobian << std::endl;
	std::cout << "jacobian2: " << jacobian2 << std::endl;
	std::cout << "hessian: " << hessian << std::endl;
	std::cout << "hessian2: " << hessian2 << std::endl;
}

int main() {

	test_ivim_hessian();
	//strp_cpu_irmag_anal_specific(1, 50, true);
	/*
	strp_cpu_irmag_anal_specific(1, 1, true);
	strp_cpu_irmag_anal_specific(1, 2, true);
	strp_cpu_irmag_anal_specific(1, 5, true);
	strp_cpu_irmag_anal_specific(1, 10, true);
	strp_cpu_irmag_anal_specific(1, 50, true);
	strp_cpu_irmag_anal_specific(1, 100, true);
	*/
	
	
	

	//strp_cuda_ir_anal_specific(1, 10, true);

	//strp_cuda_ir_anal_specific(2, 20, true);

	//strp_cpu_ir_anal_specific(5000000, 10, false);
	//strp_cuda_ir_anal_specific(5000000, 10, false);

	/*
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
	*/
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