#include "../compute.hpp"


int main() {

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

	torch::Tensor res = torch::empty({nprob, ndata});
	torch::Tensor data = torch::empty({ nprob, ndata });

	mp_model.eval(data);

	std::cout << "data: " << data << std::endl;

	torch::Tensor jac = torch::empty({ nprob, ndata, npar });
	torch::Tensor hes = torch::empty({ nprob, npar, npar });

	mp_model.res(res, data);

	std::cout << "res: " << res << std::endl;

	mp_model.parameters() += 0.1*torch::rand({ nprob, npar });

	std::cout << "params: " << mp_model.parameters() << std::endl;

	mp_model.res_jac(res, jac, data);

	std::cout << "res: " << res << std::endl;
	std::cout << "jac: " << jac << std::endl;

	mp_model.res_jac_hess(res, jac, hes, data);

	std::cout << "hes: " << hes << std::endl;

	std::cout << "hess-approx: " << torch::bmm(jac.transpose(1, 2), jac) << std::endl;

}