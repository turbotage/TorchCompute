#pragma once

#include "../pch.hpp"

#include <optional>

#include "../Expression/expression.hpp"


namespace optim {
	
	// Function used to evaluate a model
								
	using EvalFunc = std::function<torch::Tensor(
		// Constants					// PerProblemInputs		// Parameters
		std::vector<torch::Tensor>&, 	torch::Tensor&, 		torch::Tensor&)>;




	// Function used to get back the Jacobian for a model
									// Derivative (Jacobian)
	using DiffFunc = std::function<torch::Tensor(
		// Constants					// PerProblemInputs		// Parameters
		std::vector<torch::Tensor>&, 	torch::Tensor&, 		torch::Tensor&)>;




	// Function used to either evaluate the model and get back the jacobian
	using EvalAndDiffFunc = std::function<void(
		// Constants					// PerProblemInputs			// Parameters
		std::vector<torch::Tensor>&, 	torch::Tensor&, 			torch::Tensor&,
		// Evaluate						// Derivative (Jacobian)
		OutRef<torch::Tensor>,			OptOutRef<torch::Tensor> )>;





	class Model {
	public:

		Model() = delete;
		
		/// <summary>
		/// Build the model from an expression
		/// </summary>
		/// <param name="expression"></param>
		/// <param name="opts"></param>
		/// <param name="dependent_map"></param>
		/// <param name="parameter_map"></param>
		/// <param name="staticvar_map"></param>
		Model(std::string expression, torch::TensorOptions opts,
			std::optional<std::unordered_map<std::string, int>> dependent_map,
			std::optional<std::unordered_map<std::string, int>> parameter_map,
			std::optional<std::unordered_map<std::string, int>> staticvar_map);

		/// <summary>
		///	Create the model from a EvalFunc, Jacobian gets computed by default_jacobian_fetcher
		/// </summary>
		/// <param name="func"></param>
		Model(EvalFunc func);

		/// <summary>
		///	Create the model from a EvalAndDiffFunc
		/// </summary>
		/// <param name="func"></param>
		Model(EvalAndDiffFunc func);

		/// <summary>
		/// Send, parameters, dependents and staticvars to specified device if they
		/// don't already reside there.
		/// </summary>
		/// <param name="device"></param>
		void to(torch::Device device);
		
		/// <summary>
		/// Sets the values of the static vars, these are used inplace of f.i constants
		/// Slightly higher performance than stringed numbers if many function invokations shall be
		/// done, since no new tensor needs to be created.
		/// </summary>
		/// <param name="staticvars"></param>
		void setConstants(std::vector<torch::Tensor> constants);

		/// <summary>
		/// The known per problem inputs of the model, the x
		/// values in y = f(x,p)
		/// </summary>
		/// <param name="dependents"></param>
		void setPerProblemInputs(torch::Tensor per_problem_inputs);
		
		/// <summary>
		/// The parameters
		/// </summary>
		/// <param name="parameters"></param>
		void setParameters(torch::Tensor parameters);

		/// <summary>
		/// Get number of problems in the model
		/// </summary>
		/// <returns></returns>
		ui32 getNumProblems();

		/// <summary>
		/// Get number of constants in the model
		/// </summary>
		/// <returns></returns>
		ui32 getNumConstants();

		/// <summary>
		/// Get number of parameters in the model
		/// </summary>
		/// <returns></returns>
		ui32 getNumParametersPerProblem();

		/// <summary>
		/// Get number inputs per problem in the model
		/// </summary>
		/// <returns></returns>
		ui32 getNumInputsPerProblem();

		/// <summary>
		/// Gets the current constants of the model
		/// </summary>
		/// <return></returns>
		std::vector<torch::Tensor>& getConstants();

		/// <summary>
		/// Gets the current per problem inputs of the model
		/// </summary>
		/// <return></returns>
		torch::Tensor& getPerProblemInputs();

		/// <summary>
		/// Gets the current parameters of the model
		/// </summary>
		/// <return></returns>
		torch::Tensor& getParameters();

		/// <summary>
		/// Evaluates the model at last set variables
		/// </summary>
		/// <returns></returns>
		void eval(torch::Tensor& value);

		/// <summary>
		/// Evaluates and differentiates the model at last set variables (gives Jacobian)
		/// </summary>
		/// <returns></returns>
		void eval_diff(torch::Tensor& value, torch::Tensor& jacobian);

	private:

		using ExpGraphPtr = std::unique_ptr<expression::ExpressionGraph>;
		std::optional<ExpGraphPtr> m_pSyntaxTree;
		 
		EvalAndDiffFunc m_EvalAndDiffFunc;

		std::unordered_map<std::string, int> m_ConstantMap;
		std::vector<torch::Tensor> m_Constants;

		std::unordered_map<std::string, int> m_PerProblemInputMap;
		torch::Tensor m_PerProblemInputs;

		std::unordered_map<std::string, int> m_ParameterMap;
		torch::Tensor m_Parameters;

		torch::TensorOptions m_TensorOptions;
	};

}


