#include "../compute.hpp"

void slm_cpu_psir_anal_specific(int32_t n, int32_t iter, bool print) {

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

void slm_cuda_psir_anal_specific(int32_t n, int32_t iter, bool print) {

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


void slm_cpu_irmag_anal_specific(int32_t n, int32_t iter, bool print) {

	using namespace tc;

	torch::InferenceMode im_guard;

	auto mp_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_irmag_eval_jac_hess, tc::models::mp_irmag_diff, tc::models::mp_irmag_diff2);

	torch::TensorOptions dops;
	dops = dops.dtype(torch::kFloat64);

	auto params = torch::rand({ n, 2 }, dops);
	params.select(1, 0).fill_(653.4f);
	params.select(1, 1).fill_(495.3f);

	torch::Tensor TR = torch::full({ 1 }, 9820.0f, dops);
	torch::Tensor TI = torch::empty({ 1, 6 }, dops);
	TI.select(1, 0).fill_(100.0f);
	TI.select(1, 1).fill_(208.0f);
	TI.select(1, 2).fill_(400.0f);
	TI.select(1, 3).fill_(750.0f);
	TI.select(1, 4).fill_(1000.0f);
	TI.select(1, 5).fill_(2000.0f);

	torch::Tensor FA = torch::full({ 1 }, 180.0f * 3.141592 / 180.0f, dops);
	FA = torch::cos(FA) - 1;
	std::vector<torch::Tensor> consts{ TR, TI, FA };

	mp_model->parameters() = params;
	mp_model->constants() = consts;

	torch::Tensor data = torch::empty({ n, 6 }, dops);
	//mp_model->eval(data);
	data.select(1, 0).fill_(418.0f);
	data.select(1, 1).fill_(156.0f);
	data.select(1, 2).fill_(54.0f);
	data.select(1, 3).fill_(338.0f);
	data.select(1, 4).fill_(480.0f);
	data.select(1, 5).fill_(662.0f);

	if (print) {
		std::cout << "data:\n" << data << std::endl;
	}

	auto guess = torch::empty({ n, 2 }, dops);
	guess.select(1, 0).fill_(706.1);
	guess.select(1, 1).fill_(577.1);

	mp_model->parameters() = guess;

	auto resJ = tc::optim::MP_SLM::default_res_J_setup(*mp_model, data);
	auto lambda = tc::optim::MP_SLM::default_lambda_setup(mp_model->parameters(), 1.0f);
	auto scaling = tc::optim::MP_SLM::default_scaling_setup(resJ.second, 1.0e-4);

	if (print) {
		std::cout << "true params: " << params << std::endl;
		/*
		std::cout << "start residuals: " << resJ.first << std::endl;
		std::cout << "start jacobian: " << resJ.second << std::endl;
		std::cout << "start deltas: " << lambda << std::endl;
		std::cout << "scaling: " << scaling << std::endl;
		*/
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

void slm_cuda_irmag_anal_specific(int32_t n, int32_t iter, bool print) {

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
	torch::TensorOptions dops{ torch::ScalarType::Float };
	auto param = torch::tensor({ 700.0f, 0.2f, 0.1f, 0.001f }, dops);
	std::vector<torch::Tensor> constants = { 
		torch::tensor({
			0.0f, 10.0f, 20.0f, 30.0f, 40.0f, 60.0f, 80.0f, 100.0f, 120.0f, 140.0f, 160.0f, 180.0f, 200.0f,
			300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f, 900.0f, 1000.0f
		}, dops).unsqueeze(0)
	};
	param.unsqueeze_(0);

	std::cout << "param: " << param << std::endl;
	std::cout << "constants: " << constants[0] << std::endl;

	auto values = torch::empty({ 1, 21 });
	auto jacobian = torch::empty({ 1, 21, 4 });
	auto hessian = torch::empty({ 1, 4, 4 });

	auto data = torch::tensor({
		908.02686f, 905.39154f, 906.08997f, 700.7829f, 753.0848f, 859.9136f,
	   870.48846f, 755.96893f, 617.3499f, 566.2044f , 746.62067f, 662.47424f,
	   628.8806f, 459.7746f , 643.30554f, 318.58453f, 416.5493f, 348.34335f,
	   411.74026f, 284.17468f, 290.30487f }, dops);
	data.unsqueeze_(0);

	//tc::models::mp_ivim_eval_jac_hess(constants, param, data, std::nullopt, std::nullopt, std::nullopt);

	std::cout << "data: " << data << std::endl;

	tc::models::mp_ivim_eval_jac_hess(constants, param, values, jacobian, hessian, data);


	std::string expr = "$S0*($f*exp(-$b*$D1)+(1-$f)*exp(-$b*$D2))";
	std::vector<std::string> param_names = { "$S0", "$f", "$D1", "$D2" };
	std::vector<std::string> const_names = { "$b" };
	std::unique_ptr<tc::optim::MP_Model> model = std::make_unique<tc::optim::MP_Model>(expr, param_names, std::make_optional(const_names));

	model->constants() = constants;
	model->parameters() = param;

	auto values2 = torch::empty({ 1, 21 });
	auto jacobian2 = torch::empty({ 1, 21, 4 });
	auto hessian2 = torch::empty({ 1, 4, 4 });

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
	//test_ivim_hessian();

	slm_cuda_irmag_anal_specific(500000, 30, false);

}