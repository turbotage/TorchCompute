#include "../pch.hpp"

#include "mp_optim_interface.h"

#include "../Models/mp_models.hpp"




void ffi::model_create_from_type(ffi::ModelHandle** model_handle, int32_t modeltype)
{
	switch (modeltype) {
	case eMP_ModelType::ADC:
		{
			auto& mh = *model_handle;
			mh = new ffi::ModelHandle;
			mh->p_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_adc_eval_jac_hess, 
				tc::models::mp_adc_diff, tc::models::mp_adc_diff2);
		}
		break;
	case eMP_ModelType::VFA:
		{
			auto& mh = *model_handle;
			mh = new ffi::ModelHandle;
			mh->p_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_vfa_eval_jac_hess,
				tc::models::mp_vfa_diff, tc::models::mp_vfa_diff2);
		}
		break;
	case eMP_ModelType::T2:
		{
			auto& mh = *model_handle;
			mh = new ffi::ModelHandle;
			mh->p_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_t2_eval_jac_hess,
				tc::models::mp_t2_diff, tc::models::mp_t2_diff2);
		}
		break;
	case eMP_ModelType::PSIR:
		{
			auto& mh = *model_handle;
			mh = new ffi::ModelHandle;
			mh->p_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_psir_eval_jac_hess,
				tc::models::mp_psir_diff, tc::models::mp_psir_diff2);
		}
		break;
	case eMP_ModelType::IR:
		{
			auto& mh = *model_handle;
			mh = new ffi::ModelHandle;
			mh->p_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_irmag_eval_jac_hess,
				tc::models::mp_irmag_diff, tc::models::mp_irmag_diff2);
		}
		break;
	case eMP_ModelType::IVIM_FULL:
		{
			auto& mh = *model_handle;
			mh = new ffi::ModelHandle;
			mh->p_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_ivim_eval_jac_hess,
				tc::models::mp_ivim_diff, tc::models::mp_ivim_diff2);
		}
		break;
	case eMP_ModelType::IVIM_PARTIAL:
	{
		auto& mh = *model_handle;
		mh = new ffi::ModelHandle;
		mh->p_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_ivim_partial_eval_jac_hess,
			tc::models::mp_ivim_partial_diff, tc::models::mp_ivim_partial_diff2);
	}
		break;
	case eMP_ModelType::PSIRFA:
	{
		auto& mh = *model_handle;
		mh = new ffi::ModelHandle;
		mh->p_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_psirfa_eval_jac_hess,
			tc::models::mp_psirfa_diff, tc::models::mp_psirfa_diff2);
	}
		break;
	case eMP_ModelType::IRFA:
	{
		auto& mh = *model_handle;
		mh = new ffi::ModelHandle;
		mh->p_model = std::make_unique<tc::optim::MP_Model>(tc::models::mp_irmagfa_eval_jac_hess,
			tc::models::mp_irmagfa_diff, tc::models::mp_irmagfa_diff2);
	}
		break;
	default:
		throw new std::runtime_error("Model ID is not implemented");
	}
}

void ffi::model_create_from_expr(ffi::ModelHandle** model_handle, const char* expression, const char** parameters, int num_parameters, const char** constants, int num_constants)
{
	auto& mh = *model_handle;

	std::vector<std::string> params;
	params.reserve(num_parameters);
	std::vector<std::string> consts;
	consts.reserve(num_constants);

	std::string expr(expression);
	if (parameters != nullptr && num_parameters > 0) {
		for (int i = 0; i < num_parameters; ++i) {
			params.emplace_back(parameters[i]);
		}
	}
	else {
		throw std::runtime_error("Cannot create a model without parameters");
	}
	
	mh = new ffi::ModelHandle;
	
	if (constants != nullptr && num_constants > 0) {
		for (int i = 0; i < num_constants; ++i) {
			consts.emplace_back(constants[i]);
		}
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
	std::vector<torch::Tensor> consts(num_constants);
	for (int i = 0; i < num_constants; ++i) {
		consts[i] = *constants[i];
	}

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

void ffi::model_diff(ModelHandle* model_handle, torch::Tensor* value, std::uint32_t index)
{
	model_handle->p_model->diff(*value, index);
}









void ffi::optim_free(ffi::OptimHandle* optim_handle)
{
	delete optim_handle;
}

void ffi::optim_run(ffi::OptimRunHandle** model_run_handle, ffi::OptimHandle* optim_handle, uint32_t iter)
{
	auto& mrh = *model_run_handle;
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

void ffi::get_plane_converging_problems_combined(torch::Tensor** cp, const torch::Tensor* lastJ, const torch::Tensor* lastP, const torch::Tensor* lastR, float tolerance)
{
	auto& c = *cp;
	c = new torch::Tensor(tc::optim::get_plane_converging_problems_combined(*lastJ, *lastP, *lastR, tolerance));
}

void ffi::get_plane_converging_problems(torch::Tensor** cp, const torch::Tensor* lastJ, const torch::Tensor* lastP, const torch::Tensor* lastR, float tolerance)
{
	auto& c = *cp;
	c = new torch::Tensor(tc::optim::get_plane_converging_problems(*lastJ, *lastP, *lastR, tolerance));
}

void ffi::get_gradient_converging_problems_absolute(torch::Tensor** cp, const torch::Tensor* J, const torch::Tensor* R, float tolerance)
{
	auto& c = *cp;
	c = new torch::Tensor(tc::optim::get_gradient_converging_problems_absolute(*J, *R, tolerance));
}

void ffi::get_gradient_converging_problems_relative(torch::Tensor** cp, const torch::Tensor* J, const torch::Tensor* R, float tolerance)
{
	auto& c = *cp;
	c = new torch::Tensor(tc::optim::get_gradient_converging_problems_relative(*J, *R, tolerance));
}

void ffi::get_gradient_converging_problems_combined(torch::Tensor** cp, const torch::Tensor* J, const torch::Tensor* R, float tolerance)
{
	auto& c = *cp;
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

	auto& oh = *optim_handle;
	oh = new ffi::OptimHandle;

	auto mpp = tc::optim::MP_STRP::make(std::move(strpsettings));
	oh->p_optim_handle = std::move(mpp);

}

void ffi::strp_last_parameters(const ffi::OptimHandle* optim_handle, torch::Tensor** last_parameters)
{
	tc::optim::MP_Optimizer& opt = *optim_handle->p_optim_handle;
	tc::optim::MP_STRP& strp = dynamic_cast<tc::optim::MP_STRP&>(opt);
	*last_parameters = new torch::Tensor(strp.last_parameters());
}

void ffi::strp_last_step(const ffi::OptimHandle* optim_handle, torch::Tensor** last_step)
{
	tc::optim::MP_Optimizer& opt = *optim_handle->p_optim_handle;
	tc::optim::MP_STRP& strp = dynamic_cast<tc::optim::MP_STRP&>(opt);
	*last_step = new torch::Tensor(strp.last_step());
}

void ffi::strp_last_jacobian(const ffi::OptimHandle* optim_handle, torch::Tensor** last_jacobian)
{
	tc::optim::MP_Optimizer& opt = *optim_handle->p_optim_handle;
	tc::optim::MP_STRP& strp = dynamic_cast<tc::optim::MP_STRP&>(opt);
	*last_jacobian = new torch::Tensor(strp.last_jacobian());
}

void ffi::strp_last_residuals(const ffi::OptimHandle* optim_handle, torch::Tensor** last_residuals)
{
	tc::optim::MP_Optimizer& opt = *optim_handle->p_optim_handle;
	tc::optim::MP_STRP& strp = dynamic_cast<tc::optim::MP_STRP&>(opt);
	*last_residuals = new torch::Tensor(strp.last_residuals());
}

void ffi::strp_last_deltas(const ffi::OptimHandle* optim_handle, torch::Tensor** last_deltas)
{
	tc::optim::MP_Optimizer& opt = *optim_handle->p_optim_handle;
	tc::optim::MP_STRP& strp = dynamic_cast<tc::optim::MP_STRP&>(opt);
	*last_deltas = new torch::Tensor(strp.last_deltas());
}

void ffi::strp_last_multiplier(const ffi::OptimHandle* optim_handle, torch::Tensor** last_multiplier)
{
	tc::optim::MP_Optimizer& opt = *optim_handle->p_optim_handle;
	tc::optim::MP_STRP& strp = dynamic_cast<tc::optim::MP_STRP&>(opt);
	*last_multiplier = new torch::Tensor(strp.last_multiplier());
}





void ffi::slm_default_lambda(torch::Tensor** lambda, torch::Tensor* parameters, float multiplier)
{
	*lambda = new torch::Tensor(tc::optim::MP_SLM::default_lambda_setup(*parameters, multiplier));
}

void ffi::slm_default_scaling(torch::Tensor** scaling, torch::Tensor* J, float minimum_scale)
{
	*scaling = new torch::Tensor(tc::optim::MP_SLM::default_scaling_setup(*J, minimum_scale));
}

void ffi::slm_default_res_J(torch::Tensor** res, torch::Tensor** J, const ModelHandle* model_handle, const torch::Tensor* data)
{
	auto p = tc::optim::MP_SLM::default_res_J_setup(*model_handle->p_model, *data);
	*res = new torch::Tensor(p.first);
	*J = new torch::Tensor(p.second);
}

void ffi::slm_create(OptimHandle** optim_handle, ModelHandle* model_handle, const torch::Tensor* data, torch::Tensor* residuals, torch::Tensor* jacobian, 
	torch::Tensor* lambda, torch::Tensor* scaling, float eta, float mu, float upmul, float downmul)
{
	auto& mod = model_handle->p_model;

	tc::optim::MP_OptimizerSettings optsettings(std::move(mod), *data);
	tc::optim::MP_SLMSettings strpsettings(std::move(optsettings),*residuals, *jacobian, *lambda, *scaling, mu, eta, upmul, downmul);

	auto& oh = *optim_handle;
	oh = new ffi::OptimHandle;

	auto mpp = tc::optim::MP_SLM::make(std::move(strpsettings));
	oh->p_optim_handle = std::move(mpp);
}

void ffi::slm_last_parameters(const OptimHandle* optim_handle, torch::Tensor** last_parameters)
{
	tc::optim::MP_Optimizer& opt = *optim_handle->p_optim_handle;
	tc::optim::MP_SLM& strp = dynamic_cast<tc::optim::MP_SLM&>(opt);
	*last_parameters = new torch::Tensor(strp.last_parameters());
}

void ffi::slm_last_step(const OptimHandle* optim_handle, torch::Tensor** last_step)
{
	tc::optim::MP_Optimizer& opt = *optim_handle->p_optim_handle;
	tc::optim::MP_SLM& strp = dynamic_cast<tc::optim::MP_SLM&>(opt);
	*last_step = new torch::Tensor(strp.last_step());
}

void ffi::slm_last_jacobian(const OptimHandle* optim_handle, torch::Tensor** last_jacobian)
{
	tc::optim::MP_Optimizer& opt = *optim_handle->p_optim_handle;
	tc::optim::MP_SLM& strp = dynamic_cast<tc::optim::MP_SLM&>(opt);
	*last_jacobian = new torch::Tensor(strp.last_jacobian());
}

void ffi::slm_last_residuals(const OptimHandle* optim_handle, torch::Tensor** last_residuals)
{
	tc::optim::MP_Optimizer& opt = *optim_handle->p_optim_handle;
	tc::optim::MP_SLM& strp = dynamic_cast<tc::optim::MP_SLM&>(opt);
	*last_residuals = new torch::Tensor(strp.last_residuals());
}

void ffi::slm_last_lambdas(const OptimHandle* optim_handle, torch::Tensor** last_deltas)
{
	tc::optim::MP_Optimizer& opt = *optim_handle->p_optim_handle;
	tc::optim::MP_SLM& strp = dynamic_cast<tc::optim::MP_SLM&>(opt);
	*last_deltas = new torch::Tensor(strp.last_lambdas());
}

void ffi::slm_last_multiplier(const OptimHandle* optim_handle, torch::Tensor** last_multiplier)
{
	tc::optim::MP_Optimizer& opt = *optim_handle->p_optim_handle;
	tc::optim::MP_SLM& strp = dynamic_cast<tc::optim::MP_SLM&>(opt);
	*last_multiplier = new torch::Tensor(strp.last_multiplier());
}

void ffi::slm_last_scaling(const OptimHandle* optim_handle, torch::Tensor** last_scaling)
{
	tc::optim::MP_Optimizer& opt = *optim_handle->p_optim_handle;
	tc::optim::MP_SLM& strp = dynamic_cast<tc::optim::MP_SLM&>(opt);
	*last_scaling = new torch::Tensor(strp.last_scaling());
}
