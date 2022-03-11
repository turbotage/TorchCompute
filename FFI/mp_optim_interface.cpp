#include "../pch.hpp"

#include "mp_optim_interface.h"

#include "../Models/mp_models.hpp"




void ffi::model_create_from_type(ffi::ModelHandle** model_handle, int32_t modeltype)
{
	switch (modeltype) {
	case eMP_ModelType::ADC:
		{
			auto mh = *model_handle;
			mh = new ffi::ModelHandle;
			mh->p_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_adc_eval_jac_hess, 
				tc::models::mp_adc_diff, tc::models::mp_adc_diff2);
		}
		break;
	case eMP_ModelType::VFA:
		{
			auto mh = *model_handle;
			mh = new ffi::ModelHandle;
			mh->p_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_vfa_eval_jac_hess,
				tc::models::mp_vfa_diff, tc::models::mp_vfa_diff2);
		}
		break;
	case eMP_ModelType::T2:
		{
			auto mh = *model_handle;
			mh = new ffi::ModelHandle;
			mh->p_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_t2_eval_jac_hess,
				tc::models::mp_t2_diff, tc::models::mp_t2_diff2);
		}
		break;
	case eMP_ModelType::PSIR:
		{
			auto mh = *model_handle;
			mh = new ffi::ModelHandle;
			mh->p_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_psir_eval_jac_hess,
				tc::models::mp_psir_diff, tc::models::mp_psir_diff2);
		}
		break;
	case eMP_ModelType::IR:
		{
			auto mh = *model_handle;
			mh = new ffi::ModelHandle;
			mh->p_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_irmag_eval_jac_hess,
				tc::models::mp_irmag_diff, tc::models::mp_irmag_diff2);
		}
		break;
	default:
		throw new std::runtime_error("Model ID is not implemented");
	}
}

void ffi::model_create_from_expr(ffi::ModelHandle** model_handle, const char* expression, const char** parameters, int num_parameters, const char** constants, int num_constants)
{
	auto mh = *model_handle;

	std::string expr(expression);
	std::vector<std::string> params(&parameters[0], &parameters[num_parameters]);
	
	mh = new ffi::ModelHandle;
	
	if (constants != nullptr) {
		std::vector<std::string> consts(&constants[0], &constants[num_constants]);
		mh->p_model = std::make_unique<tc::optim::MP_Model>(expr, params, consts);
	}
	else {
		mh->p_model = std::make_unique<tc::optim::MP_Model>(expr, params, std::nullopt);
	}

}

void ffi::model_free(ffi::ModelHandle* model_handle)
{
	delete model_handle;
}

void ffi::model_set_parameters(ffi::ModelHandle* model_handle, torch::Tensor* parameters)
{
	model_handle->p_model->parameters() = *parameters;
}

void ffi::model_set_constants(ffi::ModelHandle* model_handle, torch::Tensor** constants, int num_constants)
{
	std::vector<torch::Tensor> consts(constants[0], constants[num_constants]);
	model_handle->p_model->constants() = consts;
}

void ffi::model_eval(ffi::ModelHandle* model_handle, torch::Tensor* value)
{
	model_handle->p_model->eval(*value);
}

void ffi::model_res(ffi::ModelHandle* model_handle, torch::Tensor* value, const torch::Tensor* data)
{
	model_handle->p_model->res(*value, *data);
}

void ffi::model_eval_jac(ffi::ModelHandle* model_handle, torch::Tensor* value, torch::Tensor* jac)
{
	model_handle->p_model->eval_jac(*value, *jac);
}

void ffi::model_res_jac(ffi::ModelHandle* model_handle, torch::Tensor* value, torch::Tensor* jac, const torch::Tensor* data)
{
	model_handle->p_model->res_jac(*value, *jac, *data);
}

void ffi::model_res_jac_hess(ffi::ModelHandle* model_handle, torch::Tensor* value, torch::Tensor* jac, torch::Tensor* hes, const torch::Tensor* data)
{
	model_handle->p_model->res_jac_hess(*value, *jac, *hes, *data);
}









void ffi::optim_free(ffi::OptimHandle* optim_handle)
{
	delete optim_handle;
}

void ffi::optim_run(ffi::OptimRunHandle** model_run_handle, ffi::OptimHandle* optim_handle, uint32_t iter)
{
	auto mrh = *model_run_handle;
	mrh = new ffi::OptimRunHandle;
	mrh->task_future = std::async(std::launch::async, [optim_handle, iter]() {
			optim_handle->p_optim_handle->run(iter); });
}

void ffi::optim_wait(OptimRunHandle* optim_run_handle) {
	optim_run_handle->task_future.get();
}

void ffi::optim_run_free(ffi::OptimRunHandle* optim_run_handle)
{
	delete optim_run_handle;
}

void ffi::optim_get_model(ModelHandle* model_handle, OptimHandle* optim_handle)
{
	model_handle->p_model = std::move(optim_handle->p_optim_handle->acquire_model());
}

void ffi::optim_abort(ffi::OptimHandle* optim_handle, ffi::OptimRunHandle* optim_run_handle)
{
	optim_handle->p_optim_handle->abort();
	optim_run_handle->task_future.get();
}

void ffi::optim_get_info(ffi::OptimHandle* optim_handle, ffi::OptimRunHandle* optim_run_handle, uint32_t* iter)
{
	*iter = optim_handle->p_optim_handle->get_n_iter();
}





// Convergence

void ffi::get_plane_converging_problems_combined(torch::Tensor** cp, torch::Tensor* lastJ, torch::Tensor* lastP, torch::Tensor* lastR, float tolerance)
{
	auto c = *cp;
	c = new torch::Tensor(tc::optim::get_plane_converging_problems_combined(*lastJ, *lastP, *lastR, tolerance));
}

void ffi::get_plane_converging_problems(torch::Tensor** cp, torch::Tensor* lastJ, torch::Tensor* lastP, torch::Tensor* lastR, float tolerance)
{
	auto c = *cp;
	c = new torch::Tensor(tc::optim::get_plane_converging_problems(*lastJ, *lastP, *lastR, tolerance));
}

void ffi::get_gradient_converging_problems_absolute(torch::Tensor** cp, torch::Tensor* J, torch::Tensor* R, float tolerance)
{
	auto c = *cp;
	c = new torch::Tensor(tc::optim::get_gradient_converging_problems_absolute(*J, *R, tolerance));
}

void ffi::get_gradient_converging_problems_relative(torch::Tensor** cp, torch::Tensor* J, torch::Tensor* R, float tolerance)
{
	auto c = *cp;
	c = new torch::Tensor(tc::optim::get_gradient_converging_problems_relative(*J, *R, tolerance));
}

void ffi::get_gradient_converging_problems_combined(torch::Tensor** cp, torch::Tensor* J, torch::Tensor* R, float tolerance)
{
	auto c = *cp;
	c = new torch::Tensor(tc::optim::get_gradient_converging_problems_combined(*J, *R, tolerance));
}











void ffi::strp_create(ffi::OptimHandle** optim_handle, ffi::ModelHandle* model_handle, const torch::Tensor* data, float eta, float mu)
{
	auto& mod = model_handle->p_model;

	auto nprob = mod->parameters().size(0);
	auto ndata = data->size(1);
	auto npar = mod->parameters().size(1);
	auto dops = mod->parameters().options();

	
	auto rJ = tc::optim::MP_STRP::default_res_J_setup(*mod, *data);
	torch::Tensor delta = tc::optim::MP_STRP::default_delta_setup(mod->parameters());
	torch::Tensor scaling = tc::optim::MP_STRP::default_scaling_setup(rJ.second);

	tc::optim::MP_OptimizerSettings optsettings(std::move(mod), *data);
	tc::optim::MP_STRPSettings strpsettings(std::move(optsettings), rJ.first, rJ.second, delta, scaling, mu, eta);

	auto oh = *optim_handle;
	oh = new ffi::OptimHandle;

	auto mpp = tc::optim::MP_STRP::make(std::move(strpsettings));
	oh->p_optim_handle = std::move(mpp);

}

void ffi::strp_last_parameters(ffi::OptimHandle* optim_handle, torch::Tensor** last_parameters)
{
	tc::optim::MP_Optimizer& opt = *optim_handle->p_optim_handle;
	tc::optim::MP_STRP& strp = dynamic_cast<tc::optim::MP_STRP&>(opt);
	*last_parameters = new torch::Tensor(strp.last_parameters());
}

void ffi::strp_last_step(ffi::OptimHandle* optim_handle, torch::Tensor** last_step)
{
	tc::optim::MP_Optimizer& opt = *optim_handle->p_optim_handle;
	tc::optim::MP_STRP& strp = dynamic_cast<tc::optim::MP_STRP&>(opt);
	*last_step = new torch::Tensor(strp.last_step());
}

void ffi::strp_last_jacobian(ffi::OptimHandle* optim_handle, torch::Tensor** last_jacobian)
{
	tc::optim::MP_Optimizer& opt = *optim_handle->p_optim_handle;
	tc::optim::MP_STRP& strp = dynamic_cast<tc::optim::MP_STRP&>(opt);
	*last_jacobian = new torch::Tensor(strp.last_parameters());
}

void ffi::strp_last_residuals(ffi::OptimHandle* optim_handle, torch::Tensor** last_residuals)
{
	tc::optim::MP_Optimizer& opt = *optim_handle->p_optim_handle;
	tc::optim::MP_STRP& strp = dynamic_cast<tc::optim::MP_STRP&>(opt);
	*last_residuals = new torch::Tensor(strp.last_residuals());
}

void ffi::strp_last_deltas(ffi::OptimHandle* optim_handle, torch::Tensor** last_deltas)
{
	tc::optim::MP_Optimizer& opt = *optim_handle->p_optim_handle;
	tc::optim::MP_STRP& strp = dynamic_cast<tc::optim::MP_STRP&>(opt);
	*last_deltas = new torch::Tensor(strp.last_deltas());
}

void ffi::strp_last_multiplier(ffi::OptimHandle* optim_handle, torch::Tensor** last_multiplier)
{
	tc::optim::MP_Optimizer& opt = *optim_handle->p_optim_handle;
	tc::optim::MP_STRP& strp = dynamic_cast<tc::optim::MP_STRP&>(opt);
	*last_multiplier = new torch::Tensor(strp.last_multiplier());
}
