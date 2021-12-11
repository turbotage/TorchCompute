#pragma once

#include "../pch.hpp"

#include <optional>

#include "../Expression/expression.hpp"

namespace tc {
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




		// Function used to either evaluate the model and get back the jacobian (or get residuals)
		using EvalAndDiffFunc = std::function<void(
			// Constants					// PerProblemInputs			// Parameters
			std::vector<torch::Tensor>&, 	torch::Tensor&, 			torch::Tensor&,
			// Values						// Derivative (Jacobian)	// Data
			torch::Tensor&,					tc::OptOutRef<torch::Tensor>,	tc::OptOutRef<const torch::Tensor>)>;





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
			Model(const std::string& expression,
				std::optional<std::unordered_map<std::string, int>> parameter_map,
				std::optional<std::unordered_map<std::string, int>> per_problem_input_map,
				std::optional<std::unordered_map<std::string, int>> constant_map);

			/// <summary>
			/// Build the model and its Jacobian from a model expression and differential expressions
			/// </summary>
			/// <param name="expression">expression</param>
			/// <param name="diffexpressions">expressions to build the jacobian from</param>
			/// <param name="parameter_map"></param>
			/// <param name="dependent_map"></param>
			/// <param name="constant_map"></param>
			Model(const std::string& expression, const std::vector<std::string>& diffexpressions,
				std::optional<std::unordered_map<std::string, int>> parameter_map,
				std::optional<std::unordered_map<std::string, int>> per_problem_input_map,
				std::optional<std::unordered_map<std::string, int>> constant_map);

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
			///	Gets printable string to the expression this model runs (if run by expression)
			/// </summary>
			/// <param name="func"></param>
			std::string getReadableExpressionTree();

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
			void setConstants(const std::vector<torch::Tensor>& constants);

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
			tc::ui32 getNumProblems();

			/// <summary>
			/// Get number of constants in the model
			/// </summary>
			/// <returns></returns>
			tc::ui32 getNumConstants();

			/// <summary>
			/// Get number of parameters in the model
			/// </summary>
			/// <returns></returns>
			tc::ui32 getNumParametersPerProblem();

			/// <summary>
			/// Get number inputs per problem in the model
			/// </summary>
			/// <returns></returns>
			tc::ui32 getNumInputsPerProblem();

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
			/// Evaluates the model at last set variables
			/// Returns the residuals instread of model values
			/// </summary>
			/// <returns></returns>
			void res(torch::Tensor& value, const torch::Tensor& data);

			/// <summary>
			/// Evaluates and differentiates the model at last set variables (gives Jacobian)
			/// </summary>
			/// <returns></returns>
			void eval_diff(torch::Tensor& value, torch::Tensor& jacobian);

			/// <summary>
			/// Evaluates and differentiates the model at last set variables (gives Jacobian)
			/// gives back residuals instead of model value
			/// </summary>
			/// <returns></returns>
			void res_diff(torch::Tensor& value, torch::Tensor& jacobian, const torch::Tensor& data);


		private:

			using ExpGraphPtr = std::unique_ptr<expression::ExpressionGraph>;
			std::optional<ExpGraphPtr> m_pEvalSyntaxTree;
			std::optional<std::vector<ExpGraphPtr>> m_pDiffSyntaxTrees;
		
			EvalAndDiffFunc m_EvalAndDiffFunc;

			std::unordered_map<std::string, int> m_ConstantMap;
			std::vector<torch::Tensor> m_Constants;

			std::unordered_map<std::string, int> m_PerProblemInputMap;
			torch::Tensor m_PerProblemInputs;

			std::unordered_map<std::string, int> m_ParameterMap;
			torch::Tensor m_Parameters;
		};

	}

}
