#include "../compute.hpp"


void test_adc_values() {
	int32_t nprob = 3;
	int32_t npar = 2;
	int32_t ndata = 4;

	std::vector<std::string> parameters{ "S0","ADC" };
	std::vector<std::string> constants{ "b" };

	std::string expression = "S0*exp(-b*ADC)";

	tc::optim::MP_Model mp_model(expression, parameters, constants);
	tc::optim::MP_Model mp_model2(tc::models::mp_adc_eval_jac_hess, tc::models::mp_adc_diff, tc::models::mp_adc_diff2);

	torch::Tensor params = torch::rand({ nprob, npar });
	torch::Tensor guess = torch::rand({ nprob, npar });
	std::vector<torch::Tensor> consts{ torch::rand({1,ndata}) };

	std::cout << "params: " << params << std::endl;
	std::cout << "consts: " << consts[0] << std::endl;

	mp_model.parameters() = params;
	mp_model.constants() = consts;
	mp_model2.parameters() = params;
	mp_model2.constants() = consts;

	torch::Tensor res = torch::empty({ nprob, ndata });
	torch::Tensor res2 = torch::empty({ nprob, ndata });
	torch::Tensor data = torch::empty({ nprob, ndata });
	torch::Tensor data2 = torch::empty({ nprob, ndata });

	mp_model.eval(data);
	mp_model2.eval(data2);

	std::cout << "data: " << data << std::endl;
	std::cout << "data2: " << data << std::endl;

	torch::Tensor jac = torch::empty({ nprob, ndata, npar });
	torch::Tensor jac2 = torch::empty({ nprob, ndata, npar });

	torch::Tensor hes = torch::empty({ nprob, npar, npar });
	torch::Tensor hes2 = torch::empty({ nprob, npar, npar });

	auto noised_params = params + 0.1 * torch::rand({ nprob, npar });
	mp_model.parameters() = noised_params;
	mp_model2.parameters() = noised_params;

	std::cout << "noised params1: " << mp_model.parameters() << std::endl;
	std::cout << "noised params2:" << mp_model2.parameters() << std::endl;

	//mp_model.res_jac(res, jac, data);

	mp_model.res_jac_hess(res, jac, hes, data);
	mp_model2.res_jac_hess(res2, jac2, hes2, data);

	std::cout << "res: " << res << std::endl;
	std::cout << "res2: " << res2 << std::endl;

	std::cout << "jac: " << jac << std::endl;
	std::cout << "jac2: " << jac2 << std::endl;

	std::cout << "hes: " << hes << std::endl;
	std::cout << "hes2: " << hes2 << std::endl;

	std::cout << "hess-approx: " << torch::bmm(jac.transpose(1, 2), jac) << std::endl;
	std::cout << "hess-approx2: " << torch::bmm(jac2.transpose(1, 2), jac2) << std::endl;
}

void test_adc_times(int32_t nprob, int32_t ndata) {
	int32_t npar = 2;

	std::vector<std::string> parameters{ "S0","ADC" };
	std::vector<std::string> constants{ "b" };

	std::string expression = "S0*exp(-b*ADC)";

	tc::optim::MP_Model mp_model(expression, parameters, constants);
	tc::optim::MP_Model mp_model2(tc::models::mp_adc_eval_jac_hess, tc::models::mp_adc_diff, tc::models::mp_adc_diff2);

	torch::Tensor params = torch::rand({ nprob, npar });
	torch::Tensor guess = torch::rand({ nprob, npar });
	std::vector<torch::Tensor> consts{ torch::rand({1,ndata}) };

	mp_model.parameters() = params;
	mp_model.constants() = consts;
	mp_model2.parameters() = params;
	mp_model2.constants() = consts;

	torch::Tensor res = torch::empty({ nprob, ndata });
	torch::Tensor res2 = torch::empty({ nprob, ndata });
	torch::Tensor data = torch::empty({ nprob, ndata });
	torch::Tensor data2 = torch::empty({ nprob, ndata });

	auto t1 = std::chrono::steady_clock::now();
	mp_model.eval(data);
	auto t2 = std::chrono::steady_clock::now();

	std::cout << "evaltime1: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model2.eval(data2);
	t2 = std::chrono::steady_clock::now();

	std::cout << "evaltime2: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	torch::Tensor jac = torch::empty({ nprob, ndata, npar });
	torch::Tensor jac2 = torch::empty({ nprob, ndata, npar });

	torch::Tensor hes = torch::empty({ nprob, npar, npar });
	torch::Tensor hes2 = torch::empty({ nprob, npar, npar });

	auto noised_params = params + 0.1 * torch::rand({ nprob, npar });
	mp_model.parameters() = noised_params;
	mp_model2.parameters() = noised_params;

	t1 = std::chrono::steady_clock::now();
	mp_model.res(res, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "restime1: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model2.res(res2, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "restime2: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model.res_jac(res, jac, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "resjactime1: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model2.res_jac(res2, jac2, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "resjactime2: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model.res_jac_hess(res, jac, hes, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "resjachestime1: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model2.res_jac_hess(res2, jac2, hes2, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "resjachestime2: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

}

void test_psir_values() {
	int32_t nprob = 3;
	int32_t npar = 2;
	int32_t ndata = 4;

	std::vector<std::string> parameters{ "S0","T1" };
	std::vector<std::string> constants{ "TR", "TI", "FA"};

	std::string expression = "S0*(1+FA*exp(-TI/T1)+exp(-TR/T1))";

	tc::optim::MP_Model mp_model(expression, parameters, constants);
	tc::optim::MP_Model mp_model2(tc::models::mp_psir_eval_jac_hess, tc::models::mp_psir_diff, tc::models::mp_psir_diff2);

	torch::Tensor params = torch::rand({ nprob, npar });
	torch::Tensor guess = torch::rand({ nprob, npar });
	std::vector<torch::Tensor> consts{ 3.0f * torch::ones({1}), torch::rand({1,ndata}), -2.0 * torch::ones({1}) };

	std::cout << "params: " << params << std::endl;
	std::cout << "consts: " << consts[0] << std::endl;

	mp_model.parameters() = params;
	mp_model.constants() = consts;
	mp_model2.parameters() = params;
	mp_model2.constants() = consts;

	torch::Tensor res = torch::empty({ nprob, ndata });
	torch::Tensor res2 = torch::empty({ nprob, ndata });
	torch::Tensor data = torch::empty({ nprob, ndata });
	torch::Tensor data2 = torch::empty({ nprob, ndata });

	mp_model.eval(data);
	mp_model2.eval(data2);

	std::cout << "data: " << data << std::endl;
	std::cout << "data2: " << data2 << std::endl;

	torch::Tensor jac = torch::empty({ nprob, ndata, npar });
	torch::Tensor jac2 = torch::empty({ nprob, ndata, npar });

	torch::Tensor hes = torch::empty({ nprob, npar, npar });
	torch::Tensor hes2 = torch::empty({ nprob, npar, npar });

	auto noised_params = params + 0.1 * torch::rand({ nprob, npar });
	mp_model.parameters() = noised_params;
	mp_model2.parameters() = noised_params;

	std::cout << "noised params1: " << mp_model.parameters() << std::endl;
	std::cout << "noised params2:" << mp_model2.parameters() << std::endl;

	//mp_model.res_jac(res, jac, data);

	mp_model.res_jac_hess(res, jac, hes, data);
	mp_model2.res_jac_hess(res2, jac2, hes2, data);

	std::cout << "res: " << res << std::endl;
	std::cout << "res2: " << res2 << std::endl;

	std::cout << "jac: " << jac << std::endl;
	std::cout << "jac2: " << jac2 << std::endl;

	std::cout << "hes: " << hes << std::endl;
	std::cout << "hes2: " << hes2 << std::endl;

	std::cout << "hess-approx: " << torch::bmm(jac.transpose(1, 2), jac) << std::endl;
	std::cout << "hess-approx2: " << torch::bmm(jac2.transpose(1, 2), jac2) << std::endl;
}

void test_psir_times(int32_t nprob, int32_t ndata) {
	int32_t npar = 2;

	torch::InferenceMode im_guard;

	std::vector<std::string> parameters{ "S0","T1" };
	std::vector<std::string> constants{ "TR", "TI", "FA" };

	std::string expression = "S0*(1+FA*exp(-TI/T1)+exp(-TR/T1))";

	tc::optim::MP_Model mp_model(expression, parameters, constants);
	tc::optim::MP_Model mp_model2(tc::models::mp_psir_eval_jac_hess, tc::models::mp_psir_diff, tc::models::mp_psir_diff2);

	torch::Tensor params = torch::rand({ nprob, npar });
	torch::Tensor guess = torch::rand({ nprob, npar });
	std::vector<torch::Tensor> consts{ 3.0f * torch::ones({1}), torch::rand({1,ndata}), -2.0*torch::ones({1})};

	mp_model.parameters() = params;
	mp_model.constants() = consts;
	mp_model2.parameters() = params;
	mp_model2.constants() = consts;

	torch::Tensor res = torch::empty({ nprob, ndata });
	torch::Tensor res2 = torch::empty({ nprob, ndata });
	torch::Tensor data = torch::empty({ nprob, ndata });
	torch::Tensor data2 = torch::empty({ nprob, ndata });

	auto t1 = std::chrono::steady_clock::now();
	mp_model.eval(data);
	auto t2 = std::chrono::steady_clock::now();

	std::cout << "evaltime1: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model2.eval(data2);
	t2 = std::chrono::steady_clock::now();

	std::cout << "evaltime2: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	torch::Tensor jac = torch::empty({ nprob, ndata, npar });
	torch::Tensor jac2 = torch::empty({ nprob, ndata, npar });

	torch::Tensor hes = torch::empty({ nprob, npar, npar });
	torch::Tensor hes2 = torch::empty({ nprob, npar, npar });

	auto noised_params = params + 0.1 * torch::rand({ nprob, npar });
	mp_model.parameters() = noised_params;
	mp_model2.parameters() = noised_params;

	t1 = std::chrono::steady_clock::now();
	mp_model.res(res, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "restime1: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model2.res(res2, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "restime2: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model.res_jac(res, jac, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "resjactime1: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model2.res_jac(res2, jac2, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "resjactime2: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model.res_jac_hess(res, jac, hes, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "resjachestime1: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model2.res_jac_hess(res2, jac2, hes2, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "resjachestime2: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

}

void test_irmag_values() {
	int32_t nprob = 3;
	int32_t npar = 2;
	int32_t ndata = 4;

	std::vector<std::string> parameters{ "S0","T1" };
	std::vector<std::string> constants{ "TR", "TI", "FA" };

	std::string expression = "S0*(1+FA*exp(-TI/T1)+exp(-TR/T1))";

	tc::optim::MP_Model mp_model(expression, parameters, constants);
	tc::optim::MP_Model mp_model2(tc::models::mp_psir_eval_jac_hess, tc::models::mp_psir_diff, tc::models::mp_psir_diff2);

	torch::Tensor params = torch::rand({ nprob, npar });
	torch::Tensor guess = torch::rand({ nprob, npar });
	std::vector<torch::Tensor> consts{ 3.0f * torch::ones({1}), torch::rand({1,ndata}), -2.0 * torch::ones({1}) };

	std::cout << "params: " << params << std::endl;
	std::cout << "consts: " << consts[0] << std::endl;

	mp_model.parameters() = params;
	mp_model.constants() = consts;
	mp_model2.parameters() = params;
	mp_model2.constants() = consts;

	torch::Tensor res = torch::empty({ nprob, ndata });
	torch::Tensor res2 = torch::empty({ nprob, ndata });
	torch::Tensor data = torch::empty({ nprob, ndata });
	torch::Tensor data2 = torch::empty({ nprob, ndata });

	mp_model.eval(data);
	mp_model2.eval(data2);

	std::cout << "data: " << data << std::endl;
	std::cout << "data2: " << data2 << std::endl;

	torch::Tensor jac = torch::empty({ nprob, ndata, npar });
	torch::Tensor jac2 = torch::empty({ nprob, ndata, npar });

	torch::Tensor hes = torch::empty({ nprob, npar, npar });
	torch::Tensor hes2 = torch::empty({ nprob, npar, npar });

	auto noised_params = params + 0.1 * torch::rand({ nprob, npar });
	mp_model.parameters() = noised_params;
	mp_model2.parameters() = noised_params;

	std::cout << "noised params1: " << mp_model.parameters() << std::endl;
	std::cout << "noised params2:" << mp_model2.parameters() << std::endl;

	//mp_model.res_jac(res, jac, data);

	mp_model.res_jac_hess(res, jac, hes, data);
	mp_model2.res_jac_hess(res2, jac2, hes2, data);

	std::cout << "res: " << res << std::endl;
	std::cout << "res2: " << res2 << std::endl;

	std::cout << "jac: " << jac << std::endl;
	std::cout << "jac2: " << jac2 << std::endl;

	std::cout << "hes: " << hes << std::endl;
	std::cout << "hes2: " << hes2 << std::endl;

	std::cout << "hess-approx: " << torch::bmm(jac.transpose(1, 2), jac) << std::endl;
	std::cout << "hess-approx2: " << torch::bmm(jac2.transpose(1, 2), jac2) << std::endl;
}

void test_irmag_times(int32_t nprob, int32_t ndata) {
	int32_t npar = 2;

	torch::InferenceMode im_guard;

	std::vector<std::string> parameters{ "S0","T1" };
	std::vector<std::string> constants{ "TR", "TI", "FA" };

	std::string expression = "abs(S0*(1+FA*exp(-TI/T1)+exp(-TR/T1)))";

	tc::optim::MP_Model mp_model(expression, parameters, constants);
	tc::optim::MP_Model mp_model2(tc::models::mp_irmag_eval_jac_hess, tc::models::mp_irmag_diff, tc::models::mp_irmag_diff2);

	torch::Tensor params = torch::rand({ nprob, npar });
	torch::Tensor guess = torch::rand({ nprob, npar });
	std::vector<torch::Tensor> consts{ 3.0f * torch::ones({1}), torch::rand({1,ndata}), -2.0 * torch::ones({1}) };

	mp_model.parameters() = params;
	mp_model.constants() = consts;
	mp_model2.parameters() = params;
	mp_model2.constants() = consts;

	torch::Tensor res = torch::empty({ nprob, ndata });
	torch::Tensor res2 = torch::empty({ nprob, ndata });
	torch::Tensor data = torch::empty({ nprob, ndata });
	torch::Tensor data2 = torch::empty({ nprob, ndata });

	auto t1 = std::chrono::steady_clock::now();
	mp_model.eval(data);
	auto t2 = std::chrono::steady_clock::now();

	std::cout << "evaltime1: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model2.eval(data2);
	t2 = std::chrono::steady_clock::now();

	std::cout << "evaltime2: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	torch::Tensor jac = torch::empty({ nprob, ndata, npar });
	torch::Tensor jac2 = torch::empty({ nprob, ndata, npar });

	torch::Tensor hes = torch::empty({ nprob, npar, npar });
	torch::Tensor hes2 = torch::empty({ nprob, npar, npar });

	auto noised_params = params + 0.1 * torch::rand({ nprob, npar });
	mp_model.parameters() = noised_params;
	mp_model2.parameters() = noised_params;

	t1 = std::chrono::steady_clock::now();
	mp_model.res(res, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "restime1: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model2.res(res2, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "restime2: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model.res_jac(res, jac, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "resjactime1: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model2.res_jac(res2, jac2, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "resjactime2: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model.res_jac_hess(res, jac, hes, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "resjachestime1: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model2.res_jac_hess(res2, jac2, hes2, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "resjachestime2: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

}


void test_t2_values() {
	int32_t nprob = 3;
	int32_t npar = 2;
	int32_t ndata = 4;

	std::vector<std::string> parameters{ "S0","T2" };
	std::vector<std::string> constants{ "TE" };

	std::string expression = "S0*exp(-TE/T2)";

	tc::optim::MP_Model mp_model(expression, parameters, constants);
	tc::optim::MP_Model mp_model2(tc::models::mp_t2_eval_jac_hess, tc::models::mp_t2_diff, tc::models::mp_t2_diff2);

	torch::Tensor params = torch::rand({ nprob, npar });
	torch::Tensor guess = torch::rand({ nprob, npar });
	std::vector<torch::Tensor> consts{ torch::rand({1,ndata}) };

	std::cout << "params: " << params << std::endl;
	std::cout << "consts: " << consts[0] << std::endl;

	mp_model.parameters() = params;
	mp_model.constants() = consts;
	mp_model2.parameters() = params;
	mp_model2.constants() = consts;

	torch::Tensor res = torch::empty({ nprob, ndata });
	torch::Tensor res2 = torch::empty({ nprob, ndata });
	torch::Tensor data = torch::empty({ nprob, ndata });
	torch::Tensor data2 = torch::empty({ nprob, ndata });

	mp_model.eval(data);
	mp_model2.eval(data2);

	std::cout << "data: " << data << std::endl;
	std::cout << "data2: " << data2 << std::endl;

	torch::Tensor jac = torch::empty({ nprob, ndata, npar });
	torch::Tensor jac2 = torch::empty({ nprob, ndata, npar });

	torch::Tensor hes = torch::empty({ nprob, npar, npar });
	torch::Tensor hes2 = torch::empty({ nprob, npar, npar });

	auto noised_params = params + 0.1 * torch::rand({ nprob, npar });
	mp_model.parameters() = noised_params;
	mp_model2.parameters() = noised_params;

	std::cout << "noised params1: " << mp_model.parameters() << std::endl;
	std::cout << "noised params2:" << mp_model2.parameters() << std::endl;

	//mp_model.res_jac(res, jac, data);

	mp_model.res_jac_hess(res, jac, hes, data);
	mp_model2.res_jac_hess(res2, jac2, hes2, data);

	std::cout << "res: " << res << std::endl;
	std::cout << "res2: " << res2 << std::endl;

	std::cout << "jac: " << jac << std::endl;
	std::cout << "jac2: " << jac2 << std::endl;

	std::cout << "hes: " << hes << std::endl;
	std::cout << "hes2: " << hes2 << std::endl;

	std::cout << "hess-approx: " << torch::bmm(jac.transpose(1, 2), jac) << std::endl;
	std::cout << "hess-approx2: " << torch::bmm(jac2.transpose(1, 2), jac2) << std::endl;
}

void test_t2_times(int32_t nprob, int32_t ndata) {
	int32_t npar = 2;

	torch::InferenceMode im_guard;

	std::vector<std::string> parameters{ "S0","T2" };
	std::vector<std::string> constants{ "TE" };

	std::string expression = "S0*exp(-TE/T2)";

	tc::optim::MP_Model mp_model(expression, parameters, constants);
	tc::optim::MP_Model mp_model2(tc::models::mp_t2_eval_jac_hess, tc::models::mp_t2_diff, tc::models::mp_t2_diff2);

	torch::Tensor params = torch::rand({ nprob, npar });
	torch::Tensor guess = torch::rand({ nprob, npar });
	std::vector<torch::Tensor> consts{ torch::rand({1,ndata}) };

	mp_model.parameters() = params;
	mp_model.constants() = consts;
	mp_model2.parameters() = params;
	mp_model2.constants() = consts;

	torch::Tensor res = torch::empty({ nprob, ndata });
	torch::Tensor res2 = torch::empty({ nprob, ndata });
	torch::Tensor data = torch::empty({ nprob, ndata });
	torch::Tensor data2 = torch::empty({ nprob, ndata });

	auto t1 = std::chrono::steady_clock::now();
	mp_model.eval(data);
	auto t2 = std::chrono::steady_clock::now();

	std::cout << "evaltime1: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model2.eval(data2);
	t2 = std::chrono::steady_clock::now();

	std::cout << "evaltime2: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	torch::Tensor jac = torch::empty({ nprob, ndata, npar });
	torch::Tensor jac2 = torch::empty({ nprob, ndata, npar });

	torch::Tensor hes = torch::empty({ nprob, npar, npar });
	torch::Tensor hes2 = torch::empty({ nprob, npar, npar });

	auto noised_params = params + 0.1 * torch::rand({ nprob, npar });
	mp_model.parameters() = noised_params;
	mp_model2.parameters() = noised_params;

	t1 = std::chrono::steady_clock::now();
	mp_model.res(res, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "restime1: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model2.res(res2, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "restime2: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model.res_jac(res, jac, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "resjactime1: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model2.res_jac(res2, jac2, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "resjactime2: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model.res_jac_hess(res, jac, hes, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "resjachestime1: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

	t1 = std::chrono::steady_clock::now();
	mp_model2.res_jac_hess(res2, jac2, hes2, data);
	t2 = std::chrono::steady_clock::now();
	std::cout << "resjachestime2: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

}


int main() {

	/*
	test_adc_values();

	test_adc_times(1000000, 5);
	test_adc_times(1000000, 5);
	*/

	/*
	test_psir_values();

	test_psir_times(1000000, 10);
	test_psir_times(1000000, 10);
	*/

	test_irmag_values();

	test_irmag_times(1000000, 10);
	test_irmag_times(1000000, 10);

	/*
	test_t2_values();
	test_t2_times(1000000, 10);
	test_t2_times(1000000, 10);
	*/
	
}