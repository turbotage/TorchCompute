
#include "../compute.hpp"

void strp_cpu_adc_anal_specific(int n, bool print) {

	using namespace tc;

	std::cout << "ADC model" << std::endl;
	std::cout << "per problem b-vals : eval_and_diff" << std::endl;

	tc::optim::STRPSettings settings;

	std::unique_ptr<tc::optim::Model> pModel;
	{
		using namespace torch::indexing;

		std::unique_ptr<optim::Model> pModel = std::make_unique<optim::Model>(models::adc_eval_and_diff);

		torch::TensorOptions dops;
		dops = dops.dtype(torch::kFloat64);

		auto params = torch::rand({ n, 2 }, dops);
		params.index_put_({ Slice(), 0 }, 1000.0f);
		params.index_put_({ Slice(), 1 }, 0.002f);

		auto ppi = torch::rand({ n, 4, 1 }, dops);
		ppi.index_put_({ Slice(), 0, 0 }, 200.0f);
		ppi.index_put_({ Slice(), 1, 0 }, 400.0f);
		ppi.index_put_({ Slice(), 2, 0 }, 600.0f);
		ppi.index_put_({ Slice(), 3, 0 }, 800.0f);

		pModel->setParameters(params);
		pModel->setPerProblemInputs(ppi);

		torch::Tensor data = torch::empty({ n, 4 }, dops);
		pModel->eval(data);
		if (print) {
			std::cout << "data:\n" << data << std::endl;
		}

		auto guess = torch::empty({ n, 2 }, dops);
		guess.index_put_({ Slice(), 0 }, 20000.0f);
		guess.index_put_({ Slice(), 1 }, 0.02f);
		pModel->setParameters(guess);

		settings.pModel = std::move(pModel);
		settings.data = data;
		settings.maxIter = 20;

		auto resJ = tc::optim::STRP::default_res_J_setup(*settings.pModel, data);

		settings.start_residuals = resJ.first;
		settings.start_jacobian = resJ.second;
		settings.start_deltas = tc::optim::STRP::default_delta_setup(settings.pModel->getParameters());
		settings.scaling = tc::optim::STRP::default_scaling_setup(resJ.second);

		if (print) {
			std::cout << "true params: " << params << std::endl;
			std::cout << "start residuals: " << settings.start_residuals << std::endl;
			std::cout << "start jacobian: " << settings.start_jacobian << std::endl;
			std::cout << "start deltas: " << settings.start_deltas << std::endl;
			std::cout << "scaling: " << settings.scaling << std::endl;
		}

		auto t1 = std::chrono::steady_clock::now();
		optim::STRP strp(settings);
		optim::STRPResult res = strp.eval();
		auto t2 = std::chrono::steady_clock::now();
		std::cout << "time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

		if (print) {
			std::cout << "found params: " << res.finalParameters << std::endl;
		}

		std::cout << "No crash, Success!" << std::endl;
	}

}


void strp_cuda_adc_anal_specific(int n, bool print) {

	using namespace tc;

	std::cout << "ADC model" << std::endl;
	std::cout << "per problem b-vals : eval_and_diff" << std::endl;

	tc::optim::STRPSettings settings;

	std::unique_ptr<tc::optim::Model> pModel;
	{
		using namespace torch::indexing;

		std::unique_ptr<optim::Model> pModel = std::make_unique<optim::Model>(models::adc_eval_and_diff);

		torch::TensorOptions dops;
		dops = dops.dtype(torch::kFloat64).device(torch::Device("cuda:0"));

		auto params = torch::rand({ n, 2 }, dops);
		params.index_put_({ Slice(), 0 }, 1000.0f);
		params.index_put_({ Slice(), 1 }, 0.002f);

		auto ppi = torch::rand({ n, 4, 1 }, dops);
		ppi.index_put_({ Slice(), 0, 0 }, 200.0f);
		ppi.index_put_({ Slice(), 1, 0 }, 400.0f);
		ppi.index_put_({ Slice(), 2, 0 }, 600.0f);
		ppi.index_put_({ Slice(), 3, 0 }, 800.0f);

		pModel->setParameters(params);
		pModel->setPerProblemInputs(ppi);

		torch::Tensor data = torch::empty({ n, 4 }, dops);
		pModel->eval(data);
		if (print) {
			std::cout << "data:\n" << data << std::endl;
		}

		auto guess = torch::empty({ n, 2 }, dops);
		guess.index_put_({ Slice(), 0 }, 20000.0f);
		guess.index_put_({ Slice(), 1 }, 0.02f);
		pModel->setParameters(guess);

		settings.pModel = std::move(pModel);
		settings.data = data;
		settings.maxIter = 20;

		auto resJ = tc::optim::STRP::default_res_J_setup(*settings.pModel, data);

		settings.start_residuals = resJ.first;
		settings.start_jacobian = resJ.second;
		settings.start_deltas = tc::optim::STRP::default_delta_setup(settings.pModel->getParameters());
		settings.scaling = tc::optim::STRP::default_scaling_setup(resJ.second);

		if (print) {
			std::cout << "true params: " << params << std::endl;
			std::cout << "start residuals: " << settings.start_residuals << std::endl;
			std::cout << "start jacobian: " << settings.start_jacobian << std::endl;
			std::cout << "start deltas: " << settings.start_deltas << std::endl;
			std::cout << "scaling: " << settings.scaling << std::endl;
		}

		auto t1 = std::chrono::steady_clock::now();
		optim::STRP strp(settings);
		optim::STRPResult res = strp.eval();
		auto t2 = std::chrono::steady_clock::now();
		std::cout << "time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

		if (print) {
			std::cout << "found params: " << res.finalParameters << std::endl;
		}

		std::cout << "No crash, Success!" << std::endl;
	}

}


void strp_cpu_ir_anal_specific(int n, bool print) {

	using namespace tc;

	torch::InferenceMode im_guard;

	std::cout << "ADC model" << std::endl;
	std::cout << "per problem b-vals : eval_and_diff" << std::endl;

	tc::optim::STRPSettings settings;

	std::unique_ptr<tc::optim::Model> pModel;
	{
		using namespace torch::indexing;

		std::unique_ptr<optim::Model> pModel = std::make_unique<optim::Model>(models::psir_eval_and_diff);

		torch::TensorOptions dops;
		dops = dops.dtype(torch::kFloat64);

		auto params = torch::rand({ n, 2 }, dops);
		params.index_put_({ Slice(), 0 }, 10000.0f);
		params.index_put_({ Slice(), 1 }, 800.0f);

		/*
		auto ppi = torch::rand({ n, 4, 1 }, dops);
		ppi.index_put_({ Slice(), 0, 0 }, 200.0f);
		ppi.index_put_({ Slice(), 1, 0 }, 400.0f);
		ppi.index_put_({ Slice(), 2, 0 }, 600.0f);
		ppi.index_put_({ Slice(), 3, 0 }, 800.0f);
		*/

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
		//pModel->setPerProblemInputs(ppi);
		pModel->setConstants(constants);

		torch::Tensor data = torch::empty({ n, 4 }, dops);
		pModel->eval(data);
		if (print) {
			std::cout << "data:\n" << data << std::endl;
		}

		auto guess = torch::empty({ n, 2 }, dops);
		guess.index_put_({ Slice(), 0 }, 100.0f);
		guess.index_put_({ Slice(), 1 }, 100.0f);
		pModel->setParameters(guess);

		settings.pModel = std::move(pModel);
		settings.data = data;
		settings.maxIter = 50;

		auto resJ = tc::optim::STRP::default_res_J_setup(*settings.pModel, data);

		settings.start_residuals = resJ.first;
		settings.start_jacobian = resJ.second;
		settings.start_deltas = tc::optim::STRP::default_delta_setup(settings.pModel->getParameters());
		settings.scaling = tc::optim::STRP::default_scaling_setup(resJ.second);

		if (print) {
			std::cout << "true params: " << params << std::endl;
			std::cout << "start residuals: " << settings.start_residuals << std::endl;
			std::cout << "start jacobian: " << settings.start_jacobian << std::endl;
			std::cout << "start deltas: " << settings.start_deltas << std::endl;
			std::cout << "scaling: " << settings.scaling << std::endl;
		}

		auto t1 = std::chrono::steady_clock::now();
		optim::STRP strp(settings);
		optim::STRPResult res = strp.eval();
		auto t2 = std::chrono::steady_clock::now();
		std::cout << "time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

		if (print) {
			std::cout << "found params: " << res.finalParameters << std::endl;
		}

		std::cout << "No crash, Success!" << std::endl;
	}

}

void strp_cuda_ir_anal_specific(int n, bool print) {

	using namespace tc;

	std::cout << "ADC model" << std::endl;
	std::cout << "per problem b-vals : eval_and_diff" << std::endl;

	tc::optim::STRPSettings settings;

	std::unique_ptr<tc::optim::Model> pModel;
	{
		using namespace torch::indexing;

		std::unique_ptr<optim::Model> pModel = std::make_unique<optim::Model>(models::psir_eval_and_diff);

		torch::TensorOptions dops;
		dops = dops.dtype(torch::kFloat64).device(torch::Device("cuda:0"));

		auto params = torch::rand({ n, 2 }, dops);
		params.index_put_({ Slice(), 0 }, 10000.0f);
		params.index_put_({ Slice(), 1 }, 800.0f);

		/*
		auto ppi = torch::rand({ n, 4, 1 }, dops);
		ppi.index_put_({ Slice(), 0, 0 }, 200.0f);
		ppi.index_put_({ Slice(), 1, 0 }, 400.0f);
		ppi.index_put_({ Slice(), 2, 0 }, 600.0f);
		ppi.index_put_({ Slice(), 3, 0 }, 800.0f);
		*/

		torch::Tensor TR = torch::full({ 1 }, 5000.0f, dops);
		torch::Tensor TI = torch::empty({ 1, 4 }, dops);
		TI.index_put_({ 0, 0 }, 200.0f);
		TI.index_put_({ 0, 1 }, 400.0f);
		TI.index_put_({ 0, 2 }, 1200.0f);
		TI.index_put_({ 0, 3 }, 2000.0f);
		torch::Tensor FA = torch::full({ 1 }, 180.0f * 3.141592 / 180.0f, dops);
		FA = torch::cos(FA) - 1;
		std::vector<torch::Tensor> constants{ TR, TI, FA };


		pModel->setParameters(params);
		//pModel->setPerProblemInputs(ppi);
		pModel->setConstants(constants);

		torch::Tensor data = torch::empty({ n, 4 }, dops);
		pModel->eval(data);
		if (print) {
			std::cout << "data:\n" << data << std::endl;
		}

		auto guess = torch::empty({ n, 2 }, dops);
		guess.index_put_({ Slice(), 0 }, 100.0f);
		guess.index_put_({ Slice(), 1 }, 100.0f);
		pModel->setParameters(guess);

		settings.pModel = std::move(pModel);
		settings.data = data;
		settings.maxIter = 20;

		auto resJ = tc::optim::STRP::default_res_J_setup(*settings.pModel, data);

		settings.start_residuals = resJ.first;
		settings.start_jacobian = resJ.second;
		settings.start_deltas = tc::optim::STRP::default_delta_setup(settings.pModel->getParameters());
		settings.scaling = tc::optim::STRP::default_scaling_setup(resJ.second);

		if (print) {
			std::cout << "true params: " << params << std::endl;
			std::cout << "start residuals: " << settings.start_residuals << std::endl;
			std::cout << "start jacobian: " << settings.start_jacobian << std::endl;
			std::cout << "start deltas: " << settings.start_deltas << std::endl;
			std::cout << "scaling: " << settings.scaling << std::endl;
		}

		auto t1 = std::chrono::steady_clock::now();
		optim::STRP strp(settings);
		optim::STRPResult res = strp.eval();
		auto t2 = std::chrono::steady_clock::now();
		std::cout << "time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;

		if (print) {
			std::cout << "found params: " << res.finalParameters << std::endl;
		}

		std::cout << "No crash, Success!" << std::endl;
	}

}


int main() {

	/*
	try {
		strp_cpu_adc_anal_specific(500000, false);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}
	catch (std::runtime_error e2) {
		std::cout << e2.what() << std::endl;
	}
	*/
	/*
	try {
		strp_cpu_adc_anal_specific(500000, false);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}
	catch (std::runtime_error e2) {
		std::cout << e2.what() << std::endl;
	}
	*/
	/*
	try {
		strp_cuda_adc_anal_specific(500000, false);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}
	catch (std::runtime_error e2) {
		std::cout << e2.what() << std::endl;
	}
	*/
	
	/*
	try {
		strp_cuda_adc_anal_specific(10000000, false);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}
	catch (std::runtime_error e2) {
		std::cout << e2.what() << std::endl;
	}
	*/
	/*
	try {
		strp_cpu_ir_anal_specific(256*256*64, false);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}
	catch (std::runtime_error e2) {
		std::cout << e2.what() << std::endl;
	}
	*/

	/*
	try {
		strp_cpu_ir_anal_specific(1000000, false);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}
	catch (std::runtime_error e2) {
		std::cout << e2.what() << std::endl;
	}
	*/

	/*
	try {
		strp_cpu_ir_anal_specific(1000000, false);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}
	catch (std::runtime_error e2) {
		std::cout << e2.what() << std::endl;
	}
	*/
	

	
	try {
		strp_cuda_ir_anal_specific(1, true);
	}
	catch (c10::Error e1) {
		std::cout << e1.what() << std::endl;
	}
	catch (std::runtime_error e2) {
		std::cout << e2.what() << std::endl;
	}

	


}