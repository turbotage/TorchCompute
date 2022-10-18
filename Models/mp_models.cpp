#include "mp_models.hpp"
#include "mp_models.hpp"
#include "mp_models.hpp"
#include "../pch.hpp"

#include "mp_models.hpp"

void tc::models::mp_adc_eval_jac_hess(
	// Constants									// Parameters
	const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,
	// Values										// Jacobian
	torch::Tensor& values,							tc::OptOutRef<torch::Tensor> jacobian,
	// Hessian										// Data
	tc::OptOutRef<torch::Tensor> hessian,			tc::OptRef<const torch::Tensor> data)
{

	torch::Tensor S0 = parameters.select(1, 0).unsqueeze(-1);
	torch::Tensor ADC = parameters.select(1, 1).unsqueeze(-1);
	torch::Tensor b = constants[0];

	torch::mul_out(values, b.neg(), ADC);
	values.exp_(); // values = exp(-b*ADC)

	if (!jacobian.has_value() && !hessian.has_value()) {
		values.mul_(S0);

		if (data.has_value())
			values.sub_(data.value());

		return;
	}

	// Sets jacobian
	torch::Tensor& J = jacobian.value().get();
	J.select(2, 0) = values;
	values.mul_(S0);
	torch::mul_out(J.select(2, 1), b.neg(), values);

	// Jacobian set and eval

	if (data.has_value())
		values.sub_(data.value());

	// Sets hessian
	if (hessian.has_value()) {
		if (!jacobian.has_value()) {
			throw std::runtime_error("jacobian OptOutRef must be filled if hessian shall be evaluated");
		}
		if (!data.has_value()) {
			throw std::runtime_error("data OptRef must be filled if hessian shall be evaluated");
		}

		// If we have come here jacobian is set and values = residuals
		torch::Tensor& H = hessian.value().get();

		H.select(1, 0).select(1, 0).zero_(); // S0,S0
		torch::sum_out(H.select(1, 1).select(1, 0), values * torch::div(J.select(2, 1), S0), 1); // ADC,S0
		H.select(1, 0).select(1, 1) = H.select(1, 1).select(1, 0); // S0,ADC
		
		torch::sum_out(H.select(1, 1).select(1, 1), values * torch::mul(J.select(2, 1), b.neg()), 1); // ADC,ADC

		H += torch::bmm(J.transpose(1, 2), J);
	}

}

void tc::models::mp_adc_diff(
	// Constants									// Parameters						// Variable index
	const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, int32_t index,
	// Derivative
	torch::Tensor& diff) 
{
	torch::Tensor S0 = parameters.select(1, 0).unsqueeze(-1);
	torch::Tensor ADC = parameters.select(1, 1).unsqueeze(-1);
	torch::Tensor b = constants[0];

	if (index == 0) {
		torch::exp_out(diff, b.neg() * ADC);
		return;
	}
	else if (index == 1) {
		torch::mul_out(diff, b.neg(), S0 * torch::exp(b.neg() * ADC));
		return;
	}

	throw std::runtime_error("Only allowed indices for mp_adc_diff is 1 or 0");
}

void tc::models::mp_adc_diff2(
	// Constants									// Parameters						// Variable indices
	const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, const std::pair<int32_t, int32_t>& indices,
	// Derivative
	torch::Tensor& diff2)
{
	throw std::runtime_error("Not implemented");

	torch::Tensor S0 = parameters.select(1, 0).unsqueeze(-1);
	torch::Tensor ADC = parameters.select(1, 1).unsqueeze(-1);
	torch::Tensor b = constants[0];

	if (indices.first == 0 && indices.second == 0) {
		diff2.zero_();
		return;
	}
	else if (( indices.first == 0 && indices.second == 1) || (indices.first == 1 && indices.second == 0)) {
		torch::mul_out(diff2, b.neg(), torch::exp(b.neg() * ADC));
		return;
	}
	else if (indices.first == 1 && indices.second == 1) {
		torch::mul_out(diff2, torch::square(b) * S0, torch::exp(b.neg() * ADC));
	}

	throw std::runtime_error("Only allowed indices for mp_adc_diff is 1 or 0");
}





void tc::models::mp_vfa_eval_jac_hess(
	// Constants									// Parameters
	const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,
	// Values										// Jacobian
	torch::Tensor& values,							tc::OptOutRef<torch::Tensor> jacobian,
	// Hessian										// Data
	tc::OptOutRef<torch::Tensor> hessian,			tc::OptRef<const torch::Tensor>data)
{
	torch::Tensor S0 = parameters.select(1, 0).unsqueeze(-1);
	torch::Tensor T1 = parameters.select(1, 1).unsqueeze(-1);
	torch::Tensor TR = constants[0];
	torch::Tensor FA = constants[1];

	torch::Tensor expterm = torch::exp(-TR / T1);
	torch::Tensor costerm = torch::cos(FA);
	torch::Tensor sinterm = torch::sin(FA);
	torch::Tensor denom = 1.0f - expterm * costerm;

	torch::mul_out(values, sinterm, 1.0f - expterm);
	values.div_(denom);

	if (!jacobian.has_value() && !hessian.has_value()) {
		values.mul_(S0);

		if (data.has_value())
			values.sub_(data.value());

		return;
	}

	// Sets jacobian
	torch::Tensor& J = jacobian.value().get();
	J.select(2, 0) = values;
	values.mul_(S0);
	
	torch::Tensor costerm2 = costerm - 1.0f;
	torch::Tensor expterm2 = torch::exp(TR / T1);
	torch::Tensor T1_2 = torch::square(T1);
	torch::sub_out(denom, expterm2, costerm);
	torch::Tensor denom2 = torch::square(denom);

	J.select(2, 1) = TR * S0 * costerm2 * sinterm * expterm2 / (T1_2 * denom2);

	// Jacobian set and eval

	if (data.has_value())
		values.sub_(data.value());

	// Sets hessian
	if (hessian.has_value()) {
		if (!jacobian.has_value()) {
			throw std::runtime_error("jacobian OptOutRef must be filled if hessian shall be evaluated");
		}
		if (!data.has_value()) {
			throw std::runtime_error("data OptRef must be filled if hessian shall be evaluated");
		}
		torch::Tensor& J = jacobian.value().get();

		// If we have come here jacobian is set and values = residuals
		torch::Tensor& H = hessian.value().get();

		H.select(1, 0).select(1, 0).zero_(); // S0,S0
		torch::sum_out(H.select(1, 1).select(1, 0), values * torch::div(J.select(2, 1), S0), 1);
		H.select(1, 0).select(1, 1) = H.select(1, 1).select(1, 0);

		denom = torch::square(T1_2) * denom * denom2;
		denom2 = TR * S0 * sinterm * costerm2 * expterm2 *
			(costerm * (TR + 2.0f * T1) + expterm2 * (TR - 2.0f * T1)) / denom;

		torch::sum_out(H.select(1, 1).select(1, 1), values * denom2, 1);

		H += torch::bmm(J.transpose(1, 2), J);

	}

}

void tc::models::mp_vfa_diff(
	// Constants									// Parameters						// Variable index
	const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, int32_t index,
	// Derivative
	torch::Tensor& diff)
{
	throw std::runtime_error("Not implemented");
}

void tc::models::mp_vfa_diff2(
	// Constants									// Parameters						// Variable indices
	const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, const std::pair<int32_t, int32_t>& indices,
	// Derivative
	torch::Tensor& diff2)
{
	throw std::runtime_error("Not implemented");
}





void tc::models::mp_psir_eval_jac_hess(
	// Constants									// Parameters
	const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,
	// Values										// Jacobian
	torch::Tensor& values,							tc::OptOutRef<torch::Tensor> jacobian,
	// Hessian										// Data
	tc::OptOutRef<torch::Tensor> hessian,			tc::OptRef<const torch::Tensor>data)
{
	torch::Tensor S0 = parameters.select(1, 0).unsqueeze(-1);
	torch::Tensor T1 = parameters.select(1, 1).unsqueeze(-1);
	torch::Tensor TR = constants[0];
	torch::Tensor TI = constants[1];
	torch::Tensor FAterm = constants[2];

	torch::Tensor expterm1 = torch::exp(-TI / T1);
	torch::Tensor expterm2 = torch::exp(-TR / T1);

	torch::Tensor FAexp1 = FAterm * expterm1;

	values = 1.0f + FAexp1 + expterm2;

	if (!jacobian.has_value() && !hessian.has_value()) {
		values.mul_(S0);

		if (data.has_value())
			values.sub_(data.value());

		return;
	}

	// Sets jacobian
	torch::Tensor& J = jacobian.value().get();
	J.select(2, 0) = values;
	values.mul_(S0);
	torch::Tensor T1_2 = torch::square(T1);
	torch::Tensor TI_FAexp1 = TI * FAexp1;
	torch::Tensor TR_exp2 = TR * expterm2;

	J.select(2, 1) = S0 * (TR_exp2 + TI_FAexp1) / T1_2;
	// Jacobian set and eval

	if (data.has_value())
		values.sub_(data.value());

	// Sets hessian
	if (hessian.has_value()) {
		if (!jacobian.has_value()) {
			throw std::runtime_error("jacobian OptOutRef must be filled if hessian shall be evaluated");
		}
		if (!data.has_value()) {
			throw std::runtime_error("data OptRef must be filled if hessian shall be evaluated");
		}

		// If we have come here jacobian is set and values = residuals
		torch::Tensor& H = hessian.value().get();

		H.select(1, 0).select(1, 0).zero_(); // S0,S0
		torch::sum_out(H.select(1, 1).select(1, 0), values * torch::div(J.select(2, 1), S0), 1);
		H.select(1, 0).select(1, 1) = H.select(1, 1).select(1, 0);
		
		torch::Tensor t1t1 = S0 * (TI_FAexp1 * (TI - 2.0f * T1) + TR_exp2 * (TR - 2.0f * T1)) / torch::square(T1_2);
		torch::sum_out(H.select(1, 1).select(1, 1), values * t1t1, 1);

		H += torch::bmm(J.transpose(1, 2), J);
	}

}

void tc::models::mp_psir_diff(
	// Constants									// Parameters						// Variable index
	const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, int32_t index,
	// Derivative
	torch::Tensor& diff)
{
	throw std::runtime_error("Not implemented");
}

void tc::models::mp_psir_diff2(
	// Constants									// Parameters						// Variable indices
	const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, const std::pair<int32_t, int32_t>& indices,
	// Derivative
	torch::Tensor& diff2)
{
	throw std::runtime_error("Not implemented");
}



void tc::models::mp_psirfa_eval_jac_hess(
	// Constants									// Parameters
	const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters,
	// Values										// Jacobian
	torch::Tensor& values, tc::OptOutRef<torch::Tensor> jacobian,
	// Hessian										// Data
	tc::OptOutRef<torch::Tensor> hessian, tc::OptRef<const torch::Tensor>data)
{
	torch::Tensor S0 = parameters.select(1, 0).unsqueeze(-1);
	torch::Tensor T1 = parameters.select(1, 1).unsqueeze(-1);
	torch::Tensor FA = parameters.select(1, 2).unsqueeze(-1);
	torch::Tensor TR = constants[0];
	torch::Tensor TI = constants[1];
	torch::Tensor FAterm = torch::cos(FA) - 1.0f;

	torch::Tensor expterm1 = torch::exp(-TI / T1);
	torch::Tensor expterm2 = torch::exp(-TR / T1);

	torch::Tensor FAexp1 = FAterm * expterm1;

	values = 1.0f + FAexp1 + expterm2;

	if (!jacobian.has_value() && !hessian.has_value()) {
		values.mul_(S0);

		if (data.has_value())
			values.sub_(data.value());

		return;
	}

	// Sets jacobian
	torch::Tensor& J = jacobian.value().get();
	J.select(2, 0) = values;
	values.mul_(S0);
	torch::Tensor T1_2 = torch::square(T1);
	torch::Tensor TI_FAexp1 = TI * FAexp1;
	torch::Tensor TR_exp2 = TR * expterm2;

	J.select(2, 1) = S0 * (TR_exp2 + TI_FAexp1) / T1_2;
	J.select(2, 2) = S0.neg() * torch::sin(FA) * expterm1;
	// Jacobian set and eval

	if (data.has_value())
		values.sub_(data.value());

	// Sets hessian
	if (hessian.has_value()) {

		if (!jacobian.has_value()) {
			throw std::runtime_error("jacobian OptOutRef must be filled if hessian shall be evaluated");
		}
		if (!data.has_value()) {
			throw std::runtime_error("data OptRef must be filled if hessian shall be evaluated");
		}

		// If we have come here jacobian is set and values = residuals
		torch::Tensor& H = hessian.value().get();

		// S0,S0
		H.select(1, 0).select(1, 0).zero_();
		// S0, T1
		torch::sum_out(H.select(1, 0).select(1, 1), values * torch::div(J.select(2, 1), S0), 1);
		H.select(1, 1).select(1, 0) = H.select(1, 0).select(1, 1);
		// S0, FA
		torch::sum_out(H.select(1, 0).select(1, 2), values * torch::sin(FA) * expterm1, 1);
		H.select(1, 2).select(1, 0) = H.select(1, 0).select(1, 2);
		// T1, T1
		torch::Tensor t1t1 = S0 * (TI_FAexp1 * (TI - 2.0f * T1) + TR_exp2 * (TR - 2.0f * T1)) / torch::square(T1_2);
		torch::sum_out(H.select(1, 1).select(1, 1), values * t1t1, 1);
		// T1, FA
		torch::sum_out(H.select(1, 1).select(1, 2), values * S0.neg() * TI * torch::sin(FA) * expterm1 / T1_2, 1);
		H.select(1, 2).select(1, 1) = H.select(1, 1).select(1, 2);
		// FA, FA
		torch::sum_out(H.select(1, 2).select(1, 2), values * S0.neg() * torch::cos(FA) * expterm1, 1);


		H += torch::bmm(J.transpose(1, 2), J);
	}

}

void tc::models::mp_psirfa_diff(
	// Constants									// Parameters						// Variable index
	const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, int32_t index,
	// Derivative
	torch::Tensor& diff)
{
	throw std::runtime_error("Not implemented");
}

void tc::models::mp_psirfa_diff2(
	// Constants									// Parameters						// Variable indices
	const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, const std::pair<int32_t, int32_t>& indices,
	// Derivative
	torch::Tensor& diff2)
{
	throw std::runtime_error("Not implemented");
}






void tc::models::mp_irmag_eval_jac_hess(
	// Constants									// Parameters
	const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,
	// Values										// Jacobian
	torch::Tensor& values,							tc::OptOutRef<torch::Tensor> jacobian,
	// Hessian										// Data
	tc::OptOutRef<torch::Tensor> hessian,			tc::OptRef<const torch::Tensor>data)
{
	torch::Tensor S0 = parameters.select(1, 0).unsqueeze(-1);
	torch::Tensor T1 = parameters.select(1, 1).unsqueeze(-1);
	torch::Tensor TR = constants[0];
	torch::Tensor TI = constants[1];
	torch::Tensor FAterm = constants[2];

	torch::Tensor expterm1 = torch::exp(-TI / T1);
	torch::Tensor expterm2 = torch::exp(-TR / T1);

	torch::Tensor FAexp1 = FAterm * expterm1;

	values = S0 * (1.0f + FAexp1 + expterm2);

	if (!jacobian.has_value() && !hessian.has_value()) {
		values.abs_();

		if (data.has_value())
			values.sub_(data.value());

		return;
	}

	torch::Tensor sig = torch::sign(values);

	// Sets jacobian
	torch::Tensor& J = jacobian.value().get();
	J.select(2, 0) = sig * values / S0;
	values.abs_();
	torch::Tensor T1_2 = torch::square(T1);
	torch::Tensor TI_FAexp1 = TI * FAexp1;
	torch::Tensor TR_exp2 = TR * expterm2;

	J.select(2, 1) = sig * S0 * (TR_exp2 + TI_FAexp1) / T1_2;
	// Jacobian set and eval

	if (data.has_value())
	{
		values.sub_(data.value());
	}

	// Sets hessian
	if (hessian.has_value()) {
		if (!jacobian.has_value()) {
			throw std::runtime_error("jacobian OptOutRef must be filled if hessian shall be evaluated");
		}
		if (!data.has_value()) {
			throw std::runtime_error("data OptRef must be filled if hessian shall be evaluated");
		}

		// If we have come here jacobian is set and values = residuals
		torch::Tensor& H = hessian.value().get();

		H.select(1, 0).select(1, 0).zero_(); // S0,S0
		torch::sum_out(H.select(1, 1).select(1, 0), values * torch::div(J.select(2, 1), S0), 1);
		H.select(1, 0).select(1, 1) = H.select(1, 1).select(1, 0);

		torch::Tensor t1t1 = sig * S0 * (TI_FAexp1 * (TI - 2.0f * T1) + TR_exp2 * (TR - 2.0f * T1)) / torch::square(T1_2);
		torch::sum_out(H.select(1, 1).select(1, 1), values * t1t1, 1);

		H += torch::bmm(J.transpose(1, 2), J);
	}

}

void tc::models::mp_irmag_diff(
	// Constants									// Parameters						// Variable index
	const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, int32_t index,
	// Derivative
	torch::Tensor& diff)
{
	throw std::runtime_error("Not implemented");
}

void tc::models::mp_irmag_diff2(
	// Constants									// Parameters						// Variable indices
	const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, const std::pair<int32_t, int32_t>& indices,
	// Derivative
	torch::Tensor& diff2)
{
	throw std::runtime_error("Not implemented");
}


void tc::models::mp_irmagfa_eval_jac_hess(
	// Constants									// Parameters
	const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters,
	// Values										// Jacobian
	torch::Tensor& values, tc::OptOutRef<torch::Tensor> jacobian,
	// Hessian										// Data
	tc::OptOutRef<torch::Tensor> hessian, tc::OptRef<const torch::Tensor> data)
{
	torch::Tensor S0 = parameters.select(1, 0).unsqueeze(-1);
	torch::Tensor T1 = parameters.select(1, 1).unsqueeze(-1);
	torch::Tensor FA = parameters.select(1, 2).unsqueeze(-1);
	torch::Tensor TR = constants[0];
	torch::Tensor TI = constants[1];

	torch::Tensor expterm1 = torch::exp(-TI / T1);
	torch::Tensor expterm2 = torch::exp(-TR / T1);

	torch::Tensor FAexp1 = (torch::cos(FA) - 1.0f) * expterm1;

	values = S0 * (1.0f + FAexp1 + expterm2);

	if (!jacobian.has_value() && !hessian.has_value()) {
		values.abs_();

		if (data.has_value())
			values.sub_(data.value());

		return;
	}

	torch::Tensor sig = torch::sign(values);

	// Sets jacobian
	torch::Tensor& J = jacobian.value().get();
	J.select(2, 0) = sig * values / S0;
	values.abs_();
	torch::Tensor T1_2 = torch::square(T1);
	torch::Tensor TI_FAexp1 = TI * FAexp1;
	torch::Tensor TR_exp2 = TR * expterm2;

	J.select(2, 1) = sig * S0 * (TR_exp2 + TI_FAexp1) / T1_2;
	// Jacobian set and eval
	J.select(2, 2) = sig.neg() * S0 * torch::sin(FA) * expterm1;

	if (data.has_value())
	{
		values.sub_(data.value());
	}

	// Sets hessian
	if (hessian.has_value()) {

		if (!jacobian.has_value()) {
			throw std::runtime_error("jacobian OptOutRef must be filled if hessian shall be evaluated");
		}
		if (!data.has_value()) {
			throw std::runtime_error("data OptRef must be filled if hessian shall be evaluated");
		}

		// If we have come here jacobian is set and values = residuals
		torch::Tensor& H = hessian.value().get();

		// S0,S0
		H.select(1, 0).select(1, 0).zero_();
		// S0, T1
		torch::sum_out(H.select(1, 0).select(1, 1), values * torch::div(J.select(2, 1), S0), 1);
		H.select(1, 1).select(1, 0) = H.select(1, 0).select(1, 1);
		// S0, FA
		torch::sum_out(H.select(1, 0).select(1, 2), values * sig * torch::sin(FA) * expterm1, 1);
		H.select(1, 2).select(1, 0) = H.select(1, 0).select(1, 2);
		// T1, T1
		torch::Tensor t1t1 = sig * S0 * (TI_FAexp1 * (TI - 2.0f * T1) + TR_exp2 * (TR - 2.0f * T1)) / torch::square(T1_2);
		torch::sum_out(H.select(1, 1).select(1, 1), values * t1t1, 1);
		// T1, FA
		torch::sum_out(H.select(1, 1).select(1, 2), values * sig * S0.neg() * TI * torch::sin(FA) * expterm1 / T1_2, 1);
		H.select(1, 2).select(1, 1) = H.select(1, 1).select(1, 2);
		// FA, FA
		torch::sum_out(H.select(1, 2).select(1, 2), values * sig * S0.neg() * torch::cos(FA) * expterm1, 1);

		H += torch::bmm(J.transpose(1, 2), J);
	}
}

void tc::models::mp_irmagfa_diff(
	// Constants									// Parameters						// Variable index
	const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, int32_t index,
	// Derivative
	torch::Tensor& diff)
{
	throw std::runtime_error("Not implemented");
}

void tc::models::mp_irmagfa_diff2(
	// Constants									// Parameters						// Variable indices
	const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, const std::pair<int32_t, int32_t>& indices,
	// Derivative
	torch::Tensor& diff2)
{
	throw std::runtime_error("Not implemented");
}



void tc::models::mp_t2_eval_jac_hess(
	// Constants									// Parameters
	const std::vector<torch::Tensor>& constants,	const torch::Tensor& parameters,
	// Values										// Jacobian
	torch::Tensor& values,							tc::OptOutRef<torch::Tensor> jacobian,
	// Hessian										// Data
	tc::OptOutRef<torch::Tensor> hessian,			tc::OptRef<const torch::Tensor>data)
{
	torch::Tensor S0 = parameters.select(1, 0).unsqueeze(-1);
	torch::Tensor T2 = parameters.select(1, 1).unsqueeze(-1);
	torch::Tensor TE = constants[0];

	torch::Tensor expterm1 = torch::exp(-TE / T2);

	values = expterm1;

	if (!jacobian.has_value() && !hessian.has_value()) {
		values.mul_(S0);

		if (data.has_value())
			values.sub_(data.value());

		return;
	}

	// Sets jacobian
	torch::Tensor& J = jacobian.value().get();
	J.select(2, 0) = values;
	values.mul_(S0);
	torch::Tensor T2_2 = torch::square(T2);
	torch::Tensor frac = TE / T2_2;

	J.select(2, 1) = values * frac;
	// Jacobian set and eval

	if (data.has_value())
		values.sub_(data.value());

	// Sets hessian
	if (hessian.has_value()) {
		if (!jacobian.has_value()) {
			throw std::runtime_error("jacobian OptOutRef must be filled if hessian shall be evaluated");
		}
		if (!data.has_value()) {
			throw std::runtime_error("data OptRef must be filled if hessian shall be evaluated");
		}
		torch::Tensor& J = jacobian.value().get();

		// If we have come here jacobian is set and values = residuals
		torch::Tensor& H = hessian.value().get();

		H.select(1, 0).select(1, 0).zero_(); // S0,S0
		torch::sum_out(H.select(1, 1).select(1, 0), values * torch::div(J.select(2, 1), S0), 1);
		H.select(1, 0).select(1, 1) = H.select(1, 1).select(1, 0);

		torch::sum_out(H.select(1, 1).select(1, 1), values * J.select(2, 1) * (TE - 2.0f * T2) / T2_2, 1);

		H += torch::bmm(J.transpose(1, 2), J);
	}

}

void tc::models::mp_t2_diff(
	// Constants									// Parameters						// Variable index
	const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, int32_t index,
	// Derivative
	torch::Tensor& diff)
{
	throw std::runtime_error("Not implemented");
}

void tc::models::mp_t2_diff2(
	// Constants									// Parameters						// Variable indices
	const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, const std::pair<int32_t, int32_t>& indices,
	// Derivative
	torch::Tensor& diff2)
{
	throw std::runtime_error("Not implemented");
}


void tc::models::mp_ivim_eval_jac_hess(
	const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, 
	torch::Tensor& values, tc::OptOutRef<torch::Tensor> jacobian, 
	tc::OptOutRef<torch::Tensor> hessian, tc::OptRef<const torch::Tensor> data)
{
	torch::Tensor S0 = parameters.select(1, 0).unsqueeze(-1);
	torch::Tensor f = parameters.select(1, 1).unsqueeze(-1);
	torch::Tensor D1 = parameters.select(1, 2).unsqueeze(-1);
	torch::Tensor D2 = parameters.select(1, 3).unsqueeze(-1);

	torch::Tensor b = constants[0];

	torch::Tensor expterm1 = torch::exp(b.neg() * D1);
	torch::Tensor expterm2 = torch::exp(b.neg() * D2);

	torch::Tensor term1 = f * expterm1;
	torch::Tensor term2 = (1 - f) * expterm2;

	values = S0 * (term1 + term2);

	if (!jacobian.has_value() && !hessian.has_value()) {
		if (data.has_value())
			values.sub_(data.value());
		return;
	}

	torch::Tensor& J = jacobian.value().get();
	J.select(2, 0) = values / S0;
	J.select(2, 1) = S0 * (expterm1 - expterm2);
	J.select(2, 2) = b.neg() * S0 * term1;
	J.select(2, 3) = b.neg() * S0 * term2;

	if (data.has_value())
		values.sub_(data.value());

	if (hessian.has_value()) {
		if (!jacobian.has_value()) {
			throw std::runtime_error("jacobian OptOutRef must be filled if hessian shall be evaluated");
		}
		if (!data.has_value()) {
			throw std::runtime_error("data OptRef must be filled if hessian shall be evaluated");
		}

		// If we have come here jacobian is set and values = residuals
		torch::Tensor& H = hessian.value().get();

		// S0, S0
		H.select(1, 0).select(1, 0).zero_();
		// S0, f
		torch::sum_out(H.select(1, 0).select(1, 1), values * torch::div(J.select(2, 1), S0), 1);
		H.select(1, 1).select(1, 0) = H.select(1, 0).select(1, 1);
		// S0, D1
		torch::sum_out(H.select(1, 0).select(1, 2), values * term1 * b.neg(), 1);
		H.select(1, 2).select(1, 0) = H.select(1, 0).select(1, 2);
		// S0, D2
		torch::sum_out(H.select(1, 0).select(1, 3), values * term2 * b.neg(), 1);
		H.select(1, 3).select(1, 0) = H.select(1, 0).select(1, 3);
		// f, f
		H.select(1, 1).select(1, 1).zero_();
		// f, D1
		torch::sum_out(H.select(1, 1).select(1, 2), values * J.select(2, 2) / f, 1);
		H.select(1, 2).select(1, 1) = H.select(1, 1).select(1, 2);
		// f, D2
		torch::sum_out(H.select(1, 1).select(1, 3), values * b * S0 * expterm2, 1);
		H.select(1, 3).select(1, 1) = H.select(1, 1).select(1, 3);
		// D1, D1
		torch::sum_out(H.select(1, 2).select(1, 2), values * b.neg() * J.select(2, 2), 1);
		// D1, D2
		H.select(1, 2).select(1, 3).zero_();
		H.select(1, 3).select(1, 2).zero_();
		// D2, D2
		torch::sum_out(H.select(1, 3).select(1, 3), values * b.neg() * J.select(2, 3), 1);

		H += torch::bmm(J.transpose(1, 2), J);

	}

}

void tc::models::mp_ivim_diff(const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, int32_t index, torch::Tensor& diff)
{
	throw std::runtime_error("Not implemented");
}

void tc::models::mp_ivim_diff2(const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, const std::pair<int32_t, int32_t>& indices, torch::Tensor& diff2)
{
	throw std::runtime_error("Not implemented");
}



void tc::models::mp_ivim_partial_eval_jac_hess(
	const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters,
	torch::Tensor& values, tc::OptOutRef<torch::Tensor> jacobian,
	tc::OptOutRef<torch::Tensor> hessian, tc::OptRef<const torch::Tensor> data)
{
	torch::Tensor f = parameters.select(1, 0).unsqueeze(-1);
	torch::Tensor D1 = parameters.select(1, 1).unsqueeze(-1);
	//torch::Tensor D1 = parameters.select(1, 2).unsqueeze(-1);
	//torch::Tensor D2 = parameters.select(1, 3).unsqueeze(-1);

	torch::Tensor b = constants[0];
	torch::Tensor S0 = constants[1];
	torch::Tensor D2 = constants[2];

	torch::Tensor expterm1 = torch::exp(b.neg() * D1);
	torch::Tensor expterm2 = torch::exp(b.neg() * D2);

	torch::Tensor term1 = f * expterm1;
	torch::Tensor term2 = (1 - f) * expterm2;

	values = S0 * (term1 + term2);

	if (!jacobian.has_value() && !hessian.has_value()) {
		if (data.has_value())
			values.sub_(data.value());
		return;
	}

	torch::Tensor& J = jacobian.value().get();
	J.select(2, 0) = S0 * (expterm1 - expterm2);
	J.select(2, 1) = b.neg() * S0 * term1;

	if (data.has_value())
		values.sub_(data.value());

	if (hessian.has_value()) {
		if (!jacobian.has_value()) {
			throw std::runtime_error("jacobian OptOutRef must be filled if hessian shall be evaluated");
		}
		if (!data.has_value()) {
			throw std::runtime_error("data OptRef must be filled if hessian shall be evaluated");
		}

		// If we have come here jacobian is set and values = residuals
		torch::Tensor& H = hessian.value().get();


		// f, f
		H.select(1, 1).select(1, 1).zero_();
		// f, D1
		torch::sum_out(H.select(1, 1).select(1, 2), values * J.select(2, 2) / f, 1);
		H.select(1, 2).select(1, 1) = H.select(1, 1).select(1, 2);
		// D1, D1
		torch::sum_out(H.select(1, 2).select(1, 2), values * b.neg() * J.select(2, 2), 1);

		H += torch::bmm(J.transpose(1, 2), J);

	}

}

void tc::models::mp_ivim_partial_diff(const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, int32_t index, torch::Tensor& diff)
{
	throw std::runtime_error("Not implemented");
}

void tc::models::mp_ivim_partial_diff2(const std::vector<torch::Tensor>& constants, const torch::Tensor& parameters, const std::pair<int32_t, int32_t>& indices, torch::Tensor& diff2)
{
	throw std::runtime_error("Not implemented");
}



