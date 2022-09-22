#include "../compute.hpp"


/*
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
*/

void slm_cpu_ivim_anal_specific(int32_t n, int32_t iter, bool print) {

	using namespace tc;

	torch::InferenceMode im_guard;

	auto mp_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_ivim_eval_jac_hess, tc::models::mp_ivim_diff, tc::models::mp_ivim_diff2);

	torch::TensorOptions dops;
	dops = dops.dtype(torch::kFloat64);

	auto params = torch::rand({ n, 4 }, dops);
	params.select(1, 0).fill_(895.8240);
	params.select(1, 1).fill_(0.3061);
	params.select(1, 2).fill_(0.0058);
	params.select(1, 3).fill_(0.0008);

	torch::Tensor bvals = torch::empty({ 1, 21 }, dops);
	std::vector<float> bvalsVec = { 0,10,20,30,40,60,80,100,120,140,160,180,200,300,400,500,600,700,800,900,1000 };
	for (int i = 0; i < bvalsVec.size(); ++i) {
		bvals.select(1, i).fill_(bvalsVec[i]);
	}
	std::vector<torch::Tensor> consts{ bvals };

	if (print)
	{
		//std::cout << "bvals: " << bvals << std::endl;
	}

	mp_model->parameters() = params;
	mp_model->constants() = consts;

	torch::Tensor data = torch::empty({ n, 21 }, dops);
	std::vector<float> dataVec = { 
		908.0269,905.3915,906.0900,700.7829,753.0848,859.9136,870.4885,
		755.9689,617.3499,566.2044,746.6207,662.4742,628.8806,459.7746,
		643.3055,318.5845,416.5493,348.3433,411.7403,284.1747,290.3049 };
	for (int i = 0; i < dataVec.size(); ++i) {
		data.select(1, i).fill_(dataVec[i]);
	}

	torch::Tensor test_data = torch::empty({ n, 21 }, dops);
	mp_model->eval(test_data);
	//mp_model->eval(data);
	if (print) {
		//std::cout << "test_data: " << test_data << std::endl;
		//std::cout << "data:\n" << data << std::endl;
	}

	auto guess = torch::empty({ n, 4 }, dops);
	guess.select(1, 0).fill_(3000);
	guess.select(1, 1).fill_(0.9);
	guess.select(1, 2).fill_(0.02);
	guess.select(1, 3).fill_(0.0000001);

	if (print) {
		std::cout << "guess: " << guess << std::endl;
	}

	mp_model->parameters() = guess;

	auto resJ = tc::optim::MP_SLM::default_res_J_setup(*mp_model, data);
	auto lambda = tc::optim::MP_SLM::default_lambda_setup(mp_model->parameters(), 1.0f);
	auto scaling = tc::optim::MP_SLM::default_scaling_setup(resJ.second);

	if (print) {
		//std::cout << "true params: " << params << std::endl;
		//std::cout << "start residuals: " << resJ.first << std::endl;
		//std::cout << "start jacobian: " << resJ.second << std::endl;
		//std::cout << "start deltas: " << lambda << std::endl;
		//std::cout << "scaling: " << scaling << std::endl;
	}

	tc::optim::MP_OptimizerSettings optsettings(std::move(mp_model), data);

	tc::optim::MP_SLMSettings strpsettings(std::move(optsettings), resJ.first, resJ.second, lambda, scaling);

	auto t1 = std::chrono::steady_clock::now();


	auto strp = optim::MP_SLM::make(std::move(strpsettings));
	strp->run(iter);

	auto t2 = std::chrono::steady_clock::now();
	//std::cout << "time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	if (print) {
		std::cout << "found params: " << strp->last_parameters() << std::endl;
	}

	//std::cout << "No crash, Success!" << std::endl;

}


int main() {

	slm_cpu_ivim_anal_specific(1, 1, true);
	slm_cpu_ivim_anal_specific(1, 5, true);
	slm_cpu_ivim_anal_specific(1, 10, true);
	slm_cpu_ivim_anal_specific(1, 20, true);
	slm_cpu_ivim_anal_specific(1, 50, true);

}
