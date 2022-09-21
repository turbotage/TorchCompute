
#include "../Optim/MP/mp_model.hpp"
#include "../Optim/MP/mp_strp.hpp"
#include "../Optim/MP/mp_slm.hpp"

#include "ffi.h"

#include <future>

namespace ffi {

	enum eMP_ModelType {
		ADC = 1,
		VFA = 2,
		T2 = 3,
		PSIR = 4,
		IR = 5,
		IVIM_FULL = 6,
	};

	struct ModelHandle {
		std::unique_ptr<tc::optim::MP_Model> p_model;
	};

	struct OptimHandle {
		std::unique_ptr<tc::optim::MP_Optimizer> p_optim_handle;
	};

	struct OptimRunHandle {
		std::future<void> task_future;
	};



	void model_create_from_type(ModelHandle** model_handle, int32_t modeltype);

	void model_create_from_expr(ModelHandle** model_handle, const char* expression,
		const char** parameters, int num_parameters,
		const char** constants, int num_constants);

	void model_free(ModelHandle* model_handle);

	void model_set_parameters(ModelHandle* model_handle, torch::Tensor* parameters);

	void model_set_constants(ModelHandle* model_handle, torch::Tensor** constants, int num_constants);

	void model_eval(ModelHandle* model_handle, torch::Tensor* value);

	void model_res(ModelHandle* model_handle, torch::Tensor* value, const torch::Tensor* data);

	void model_eval_jac(ModelHandle* model_handle, torch::Tensor* value, torch::Tensor* jac);

	void model_res_jac(ModelHandle* model_handle, torch::Tensor* value, torch::Tensor* jac, const torch::Tensor* data);

	void model_res_jac_hess(ModelHandle* model_handle, torch::Tensor* value, torch::Tensor* jac, torch::Tensor* hes, const torch::Tensor* data);

	void model_diff(ModelHandle* model_handle, torch::Tensor* value, std::uint32_t index);


	void optim_free(OptimHandle* optim_handle);

	void optim_run(OptimRunHandle** model_run_handle, OptimHandle* optim_handle, uint32_t iter);

	void optim_wait(OptimRunHandle* optim_run_handle);
	
	void optim_run_free(OptimRunHandle* optim_run_handle);



	// OBS! This breaks the optimizer
	void optim_get_model(ModelHandle* model_handle, OptimHandle* optim_handle);

	void optim_abort(OptimHandle* optim_handle, OptimRunHandle* optim_run_handle);

	void optim_get_info(OptimHandle* optim_handle, OptimRunHandle* optim_run_handle, uint32_t* iter);



	void get_plane_converging_problems_combined(torch::Tensor** cp,
		const torch::Tensor* lastJ, const torch::Tensor* lastP, const torch::Tensor* lastR, float tolerance);

	void get_plane_converging_problems(torch::Tensor** cp, const torch::Tensor* lastJ,
		const torch::Tensor* lastP, const torch::Tensor* lastR, float tolerance);

	void get_gradient_converging_problems_absolute(torch::Tensor** cp, const torch::Tensor* J,
		const torch::Tensor* R, float tolerance);

	void get_gradient_converging_problems_relative(torch::Tensor** cp, const torch::Tensor* J,
		const torch::Tensor* R, float tolerance);

	void get_gradient_converging_problems_combined(torch::Tensor** cp, const torch::Tensor* J,
		const torch::Tensor* R, float tolerance);



	// STRP
	void strp_create(OptimHandle** optim_handle, ModelHandle* model_handle, const torch::Tensor* data,
		float eta, float mu);

	void strp_last_parameters(const OptimHandle* optim_handle, torch::Tensor** last_parameters);
	void strp_last_step(const OptimHandle* optim_handle, torch::Tensor** last_step);
	void strp_last_jacobian(const OptimHandle* optim_handle, torch::Tensor** last_jacobian);
	void strp_last_residuals(const OptimHandle* optim_handle, torch::Tensor** last_residuals);
	void strp_last_deltas(const OptimHandle* optim_handle, torch::Tensor** last_deltas);
	void strp_last_multiplier(const OptimHandle* optim_handle, torch::Tensor** last_multiplier);

	// SLM
	void slm_default_lambda(torch::Tensor** lambda, torch::Tensor* parameters, float multiplier);

	void slm_default_scaling(torch::Tensor** scaling, torch::Tensor* J, float minimum_scale);

	void slm_default_res_J(torch::Tensor** res, torch::Tensor** J, const ModelHandle* model_handle, const torch::Tensor* data);

	void slm_create(OptimHandle** optim_handle, ModelHandle* model_handle, const torch::Tensor* data,
		torch::Tensor* residuals, torch::Tensor* jacobian, torch::Tensor* lambda, torch::Tensor* scaling,
		float eta, float mu, float upmul, float downmul);

	void slm_last_parameters(const OptimHandle* optim_handle, torch::Tensor** last_parameters);
	void slm_last_step(const OptimHandle* optim_handle, torch::Tensor** last_step);
	void slm_last_jacobian(const OptimHandle* optim_handle, torch::Tensor** last_jacobian);
	void slm_last_residuals(const OptimHandle* optim_handle, torch::Tensor** last_residuals);
	void slm_last_lambdas(const OptimHandle* optim_handle, torch::Tensor** last_deltas);
	void slm_last_multiplier(const OptimHandle* optim_handle, torch::Tensor** last_multiplier);
	void slm_last_scaling(const OptimHandle* optim_handle, torch::Tensor** last_scaling);
}
