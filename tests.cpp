#include "tests.hpp"

#include <chrono>

#include "Optim/model.hpp"
#include "Optim/lm.hpp"
#include "Optim/solver.hpp"

#include "Compute/kmeans.hpp"


void test::test_model() {

	torch::Device cuda_device(torch::kCUDA);
	torch::Device cpu_device("cpu");

	torch::TensorOptions dops =
		torch::TensorOptions().device(cuda_device).dtype(torch::ScalarType::Float);
	torch::TensorOptions switch_dops =
		torch::TensorOptions().device(cpu_device).dtype(torch::ScalarType::Float);

	std::string expr = "@X0+@D0*sin(@X1)+3";

	std::unordered_map<std::string, int> dependents;
	std::unordered_map<std::string, int> parameters;

	dependents["@D0"] = 0;

	parameters["@X0"] = 0;
	parameters["@X1"] = 1;

	model::Model mod(expr, dops, dependents, parameters, std::nullopt);

	auto dep = torch::rand({ 3000000,5,1 }, dops) * 2;
	auto inp = torch::rand({ 3000000,2 }, dops) * 3.141592;

	mod.setDependents(dep);
	mod.setParameters(inp);


	using namespace std::chrono;
	auto start = high_resolution_clock::now();

	torch::Tensor ret1 = mod();

	auto stop = high_resolution_clock::now();

	//std::cout << ret1 << std::endl;
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << duration.count() << std::endl;

	mod.to(switch_dops.device());

	start = high_resolution_clock::now();

	torch::Tensor ret2 = mod();

	stop = high_resolution_clock::now();
	duration = duration_cast<microseconds>(stop - start);
	std::cout << duration.count();

	/*
	try {
		std::cout << mod() << std::endl;
	}
	catch (std::runtime_error e) {
		std::cout << e.what() << std::endl;
	}
	*/

}


void test::test_lmp() {
	using namespace torch::indexing;

	torch::Device cuda_device("cuda:0");
	torch::Device cpu_device("cpu");

	torch::TensorOptions dops =
		torch::TensorOptions().device(cuda_device).dtype(torch::ScalarType::Float);
	torch::TensorOptions switch_dops =
		torch::TensorOptions().device(cpu_device).dtype(torch::ScalarType::Float);

	torch::DeviceGuard guard(dops.device_opt().value());

	int nProblems = 2*1048576;
	int nParams = 2;
	int nData = 5;

	torch::Tensor params = torch::rand({ nProblems, nParams }, dops);
	params.index_put_({ Slice(), 0 }, torch::rand({ nProblems }, dops));
	params.index_put_({ Slice(), 1 }, 0.01 * torch::rand({ nProblems }, dops));

	torch::Tensor guess = torch::rand({ nProblems, nParams }, dops);
	guess.index_put_({ Slice(), 0 }, torch::rand({ nProblems }, dops));
	guess.index_put_({ Slice(), 1 }, 0.01 * torch::rand({ nProblems }, dops));

	torch::Tensor deps = torch::rand({ nProblems, nData, 1 }, dops);
	deps.index_put_({ Slice(), 0, 0 }, 10.0 * 3.1415 / 180.0);
	deps.index_put_({ Slice(), 1, 0 }, 30.0 * 3.1415 / 180.0);
	deps.index_put_({ Slice(), 2, 0 }, 50.0 * 3.1415 / 180.0);
	deps.index_put_({ Slice(), 3, 0 }, 70.0 * 3.1415 / 180.0);
	deps.index_put_({ Slice(), 3, 0 }, 90.0 * 3.1415 / 180.0);

	std::string expr = "@X0*sin(@D0)*(1-exp(@TR/@X1))/(1-exp(@TR/@X1)*cos(@D0))";

	std::unordered_map<std::string, int> dependents;
	dependents["@D0"] = 0;

	std::unordered_map<std::string, int> parameters;
	parameters["@X0"] = 0;
	parameters["@X1"] = 1;

	std::unordered_map<std::string, int> staticvars;
	staticvars["@TR"] = 0;
	std::vector<torch::Tensor> vars;
	vars.push_back(torch::tensor(-0.01, dops));

	model::Model mod(expr, dops, dependents, parameters, staticvars);

	torch::Tensor data;
	mod.setDependents(deps);
	mod.setParameters(params);
	mod.setStaticVariables(vars);
	data = mod();
	data += 0.01 * data * (1 - torch::rand({ nProblems, nData }, dops));


	optim::LMP lmp(mod);
	lmp.setParameterGuess(guess);
	lmp.setDependents(deps);
	lmp.setData(data);
	lmp.setDefaultTensorOptions(dops);
	lmp.setSwitching(99, cpu_device);
	lmp.setCopyConvergingEveryN(2);


	/*
	int it = 0;
	std::function<void()> iterationCallback = [&it]() {
		std::cout << it << std::endl;
		++it;
	};

	lmp.setOnIterationCallback(iterationCallback);
	
	std::function<void()> switchCallback = []() {
		std::cout << "Switched to cpu";
	};
	
	lmp.setOnSwitchCallback(switchCallback);
	*/

	auto start = std::chrono::system_clock::now();

	lmp.run();

	auto end = std::chrono::system_clock::now();

	torch::Tensor newParams = lmp.getParameters();

	std::cout << "elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

	//std::cout << "Parameters:\n " << newParams << std::endl;

}


void test::test_kmeans() {

	torch::Device cuda_device(torch::kCUDA);
	torch::Device cpu_device("cpu");

	torch::TensorOptions dops =
		torch::TensorOptions().device(cuda_device).dtype(torch::ScalarType::Float);
	torch::TensorOptions switch_dops =
		torch::TensorOptions().device(cpu_device).dtype(torch::ScalarType::Float);

	torch::Tensor points = torch::rand({ 100000, 10 }, switch_dops);

	auto start = std::chrono::system_clock::now();
	compute::KMeans kmeans(1024, 100, 0.001, compute::eKMeansMode::EUCLIDEAN);
	torch::Tensor out = kmeans.fit_predict(points, std::nullopt);
	auto end = std::chrono::system_clock::now();


	std::cout << "elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
}


void test::test_solver1() {

	using namespace torch::indexing;

	torch::Device cuda_device("cuda:0");
	torch::Device cpu_device("cpu");

	torch::TensorOptions dops =
		torch::TensorOptions().device(cuda_device).dtype(torch::ScalarType::Float);
	torch::TensorOptions switch_dops =
		torch::TensorOptions().device(cpu_device).dtype(torch::ScalarType::Float);

	torch::DeviceGuard guard(dops.device_opt().value());

	int nProblems = 2*1048576;
	int nParams = 2;
	int nData = 5;

	torch::Tensor params = torch::rand({ nProblems, nParams }, dops);
	params.index_put_({ Slice(), 0 }, torch::rand({ nProblems }, dops));
	params.index_put_({ Slice(), 1 }, 0.01 * torch::rand({ nProblems }, dops));

	torch::Tensor guess = torch::rand({ nProblems, nParams }, dops);
	guess.index_put_({ Slice(), 0 }, torch::rand({ nProblems }, dops));
	guess.index_put_({ Slice(), 1 }, 0.01 * torch::rand({ nProblems }, dops));

	torch::Tensor deps = torch::rand({ nProblems, nData, 1 }, dops);
	deps.index_put_({ Slice(), 0, 0 }, 10.0 * 3.1415 / 180.0);
	deps.index_put_({ Slice(), 1, 0 }, 30.0 * 3.1415 / 180.0);
	deps.index_put_({ Slice(), 2, 0 }, 50.0 * 3.1415 / 180.0);
	deps.index_put_({ Slice(), 3, 0 }, 70.0 * 3.1415 / 180.0);
	deps.index_put_({ Slice(), 3, 0 }, 80.0 * 3.1415 / 180.0);

	std::string expr = "@X0*sin(@D0)*(1-exp(@TR/@X1))/(1-exp(@TR/@X1)*cos(@D0))";

	std::unordered_map<std::string, int> dependents;
	dependents["@D0"] = 0;

	std::unordered_map<std::string, int> parameters;
	parameters["@X0"] = 0;
	parameters["@X1"] = 1;

	std::unordered_map<std::string, int> staticvars;
	staticvars["@TR"] = 0;
	std::vector<torch::Tensor> vars;

	vars.push_back(torch::tensor(-0.01, dops));

	model::Model mod(expr, dops, dependents, parameters, staticvars);

	torch::Tensor data;
	mod.setDependents(deps);
	mod.setParameters(params);
	mod.setStaticVariables(vars);
	data = mod();
	data += 0.01 * data * (1 - torch::rand({ nProblems, nData }, dops));
	
	
	optim::GuessFetchFunc fetchFunc = [nParams](torch::Tensor deps, torch::Tensor data) {
		int64_t nProbs = deps.size(0);
		torch::TensorOptions d_ops = deps.options();
		torch::Tensor p = torch::rand({ nProbs, nParams }, d_ops);
		p.index_put_({ Slice(), 0 }, torch::rand({ nProbs }, d_ops));
		p.index_put_({ Slice(), 1 }, 0.01 * torch::rand({ nProbs }, d_ops));
		return p;
	};

	optim::BatchedKMeansThenLMP bklmp(mod, fetchFunc, deps, data, 500000);

	auto start = std::chrono::system_clock::now();
	bklmp.solve();
	auto end = std::chrono::system_clock::now();

	torch::Tensor newParams = bklmp.getParameters();

	std::cout << "elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;


}


/*
int main()
{
	try {
		try {
			std::cout << "BKLMP" << std::endl;
			test::test_solver1();

			std::cout << "LMP" << std::endl;
			test::test_lmp();

		}
		catch (c10::Error e1) {
			std::cout << e1.what() << std::endl;
		}
	}
	catch (std::runtime_error e2) {
		std::cout << e2.what() << std::endl;
	}

	//std::cout << "LMP" << std::endl;
	
	return 0;
}
*/

