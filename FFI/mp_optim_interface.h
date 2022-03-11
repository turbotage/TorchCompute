
#include "../Optim/MP/mp_model.hpp"
#include "../Optim/MP/mp_strp.hpp"

#include "ffi.h"

#include <future>

namespace ffi {

	enum eMP_ModelType {
		ADC = 1,
		VFA = 2,
		T2 = 3,
		PSIR = 4,
		IR = 5,
	};

	struct ModelHandle;
	struct OptimHandle;
	struct OptimRunHandle;
	struct OptimEvalHandle;



	void model_create_from_type(ModelHandle** model_handle, int32_t modeltype);

	void model_create_from_expr(ModelHandle** model_handle, const char* expression,
		const char** parameters, int num_parameters,
		const char** constants, int num_constants);

	void tc_model_free(ModelHandle* model_handle);

	void model_set_parameters(ModelHandle* model_handle, torch::Tensor* parameters);

	void model_set_constants(ModelHandle* model_handle, torch::Tensor** constants, int num_constants);

	void model_eval(ModelHandle* model_handle, torch::Tensor* value);

	void model_res(ModelHandle* model_handle, torch::Tensor* value, const torch::Tensor* data);

	void model_eval_jac(ModelHandle* model_handle, torch::Tensor* value, torch::Tensor* jac);

	void model_res_jac(ModelHandle* model_handle, torch::Tensor* value, torch::Tensor* jac, const torch::Tensor* data);

	void model_res_jac_hess(ModelHandle* model_handle, torch::Tensor* value, torch::Tensor* jac, torch::Tensor* hes, const torch::Tensor* data);



	void optim_free(OptimHandle* optim_handle);

	void optim_run(OptimRunHandle** model_run_handle, OptimHandle* optim_handle);

	void optim_run_free(OptimRunHandle* optim_run_handle);

	void optim_eval(OptimEvalHandle** optim_eval_handle, OptimRunHandle* optim_run_handle, OptimHandle* optim_handle);

	void optim_eval_free(OptimEvalHandle* optim_eval_handle);

	void optim_get_param(torch::Tensor** params, OptimEvalHandle* optim_eval_handle);

	// OBS! This breaks the optimizer
	void optim_get_model(ModelHandle* model_handle, OptimEvalHandle* optim_eval_handle);

	void optim_abort(OptimHandle* optim_handle, OptimRunHandle* optim_run_handle);

	void optim_get_info(OptimHandle* optim_handle, OptimRunHandle* optim_run_handle, uint32_t* iter);



	void get_plane_converging_problems_combined(torch::Tensor** cp,
		torch::Tensor* lastJ, torch::Tensor* lastP, torch::Tensor* lastR, float tolerance);

	void get_plane_converging_problems(torch::Tensor** cp, torch::Tensor* lastJ,
		torch::Tensor* lastP, torch::Tensor* lastR, float tolerance);

	void get_gradient_converging_problems_absolute(torch::Tensor** cp, torch::Tensor* J,
		torch::Tensor* R, float tolerance);

	void get_gradient_converging_problems_relative(torch::Tensor** cp, torch::Tensor* J,
		torch::Tensor* R, float tolerance);

	void get_gradient_converging_problems_combined(torch::Tensor** cp, torch::Tensor* J,
		torch::Tensor* R, float tolerance);




	void strp_create(OptimHandle** optim_handle, ModelHandle* model_handle, const torch::Tensor* data,
		float eta, float mu, uint32_t max_iter);

	void strp_last_parameters(OptimHandle* optim_handle, torch::Tensor** last_parameters);
	void strp_last_step(OptimHandle* optim_handle, torch::Tensor** last_step);
	void strp_last_jacobian(OptimHandle* optim_handle, torch::Tensor** last_jacobian);
	void strp_last_residuals(OptimHandle* optim_handle, torch::Tensor** last_residuals);
	void strp_last_deltas(OptimHandle* optim_handle, torch::Tensor** last_deltas);
	void strp_last_multiplier(OptimHandle* optim_handle, torch::Tensor** last_multiplier);


}
