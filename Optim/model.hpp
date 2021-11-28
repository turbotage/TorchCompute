#pragma once

#include "../pch.hpp"

#include <optional>

#include "../Expression/expression.hpp"


namespace optim {

	namespace {
		using ExpGraphPtr = std::unique_ptr<expression::ExpressionGraph<torch::Tensor>>;
	}

	using ModelFunc = std::function<torch::Tensor(std::vector<torch::Tensor>, torch::Tensor, torch::Tensor)>;

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
		///	func(static_vars, dependents, parameters)
		/// </summary>
		/// <param name="func"></param>
		Model(ModelFunc func);


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
		void setStaticVariables(std::vector<torch::Tensor>& staticvars);

		/// <summary>
		/// The known dependents of the model, the x
		/// values in y = f(x,p)
		/// </summary>
		/// <param name="dependents"></param>
		void setDependents(torch::Tensor dependents);
		
		/// <summary>
		/// The parameters
		/// </summary>
		/// <param name="parameters"></param>
		void setParameters(torch::Tensor parameters);

		/// <summary>
		/// Get number of parameters in the model
		/// </summary>
		/// <returns></returns>
		uint32_t getNParameters();

		/// <summary>
		/// Get number of dependents in the model
		/// </summary>
		/// <returns></returns>
		uint32_t getNDeps();

		/// <summary>
		/// Gets the current dependents of the model
		/// </summary>
		/// <return></returns>
		torch::Tensor getDependents();

		/// <summary>
		/// Gets the current parameters of the model
		/// </summary>
		/// <return></returns>
		torch::Tensor getParameters();

		/// <summary>
		/// Evaluates the model at last set static variables, dependents and static vars
		/// </summary>
		/// <returns></returns>
		torch::Tensor operator()();

	private:

		std::optional<ExpGraphPtr> m_pSyntaxTree;
		 
		std::function<torch::Tensor()> m_Runner;

		std::unordered_map<std::string, int> m_StaticVarMap;
		std::vector<torch::Tensor> m_StaticVars;

		std::unordered_map<std::string, int> m_DependentMap;
		torch::Tensor m_Dependents;

		std::unordered_map<std::string, int> m_ParameterMap;
		torch::Tensor m_Parameters;

		torch::TensorOptions m_TensorOptions;
	};

}


