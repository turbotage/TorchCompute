#include "../compute.hpp"


void test_values() {
	int32_t nprob = 3;
	int32_t npar = 2;
	int32_t ndata = 4;

	std::vector<std::string> parameters{ "S0","ADC" };
	std::vector<std::string> constants{ "b" };

	std::string expression = "S0*exp(-b*ADC)";

	tc::optim::MP_Model mp_model(expression, parameters, constants);

	torch::Tensor params = torch::rand({ nprob, npar });
	torch::Tensor guess = torch::rand({ nprob, npar });
	std::vector<torch::Tensor> consts{ torch::rand({1,ndata}) };

	std::cout << "params: " << params << std::endl;
	std::cout << "consts: " << consts[0] << std::endl;

	mp_model.parameters() = params;
	mp_model.constants() = consts;

	torch::Tensor res = torch::empty({ nprob, ndata });
	torch::Tensor res2 = torch::empty({ nprob, ndata });
	torch::Tensor data = torch::empty({ nprob, ndata });

	mp_model.eval(data);

	std::cout << "data: " << data << std::endl;

	torch::Tensor jac = torch::empty({ nprob, ndata, npar });
	torch::Tensor jac2 = torch::empty({ nprob, ndata, npar });

	torch::Tensor hes = torch::empty({ nprob, npar, npar });
	torch::Tensor hes2 = torch::empty({ nprob, npar, npar });

	mp_model.res(res, data);

	std::cout << "res: " << res << std::endl;

	mp_model.parameters() += 0.1 * torch::rand({ nprob, npar });

	std::cout << "params: " << mp_model.parameters() << std::endl;

	//mp_model.res_jac(res, jac, data);

	mp_model.res_jac_hess(res, jac, hes, data);

	tc::optim::MP_EvalDiffHessFunc func = tc::models::mp_adc_eval_jac_hess;
	tc::optim::MP_FirstDiff fdiff = tc::models::mp_adc_diff;
	tc::optim::MP_SecondDiff sdiff = tc::models::mp_adc_diff2;

	tc::optim::MP_Model mp_model2(tc::models::mp_adc_eval_jac_hess, tc::models::mp_adc_diff, tc::models::mp_adc_diff2);

	mp_model2.parameters() = params;
	mp_model2.constants() = consts;

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

int main() {

	

}