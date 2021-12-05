#pragma once

#include "../pch.hpp"

#include "model.hpp"


namespace optim {

	class LMP {
	public:

		LMP(optim::Model& model);
		
		// Requirest set functions

		void setParameterGuess(torch::Tensor& parameters);
		void setDependents(torch::Tensor& dependents);
		void setData(torch::Tensor& data);

		void setDefaultTensorOptions(torch::TensorOptions dops);

		void setOnIterationCallback(std::function<void()> iterationCallback);
		void setOnSwitchCallback(std::function<void()> switchCallback);

		// Non required set functions

		void setSwitching(int64_t switchNumber, torch::Device& device);

		void setCopyConvergingEveryN(int n);

		void setMu(float mu);
		void setEta(float eta);
		void setTolerance(float tol);

		void setMaxIteration(int iter);
		

		// Run functions

		void run();
		torch::Tensor getParameters();
		torch::Tensor getNonConvergingParameters();

	private:

		void print_all();
		void print_devices();
		void switch_device(torch::Device device);

		torch::Tensor plane_convergence();

		void dogleg();

		int handle_convergence();
		void next_step_iter();


		void setup_solve();
		void solve();

	private:

		std::function<void()> onIterationCallback;
		std::function<void()> onSwitchCallback;

		torch::TensorOptions dops;

		int64_t nParams;
		int64_t nDependents; // = nData
		int64_t nProblems;

		optim::Model& model;

		torch::Tensor params; // (nProblems,nParams,1)
		torch::Tensor params_slice;

		torch::Tensor deps; // (nProblems,nDeps,1)
		torch::Tensor deps_slice;

		torch::Tensor data; // (nProblems,nDeps,1) nDeps = nData
		torch::Tensor data_slice;

		float mu = 0.25f;
		float eta = 0.75f;
		float tol = 0.0001;
		int max_iter = 100;

		int copyConvergingEveryN = 6;
		int64_t onSwitchNumber = -1; // We don't switch on default
		std::optional<torch::Device> switchDevice;
		bool hasSwitched = false;

		// Non-converging indices
		torch::Tensor nci;
		int64_t nci_size;
		int64_t nc_sum;
		
		// Function evaluation y = func(deps,params)
		torch::Tensor y;
		// Residuals, residuals on trailing point
		torch::Tensor res, rt;
		// Jacobian
		torch::Tensor J;
		// Dogleg search direction, Gauss-Newton search direction
		torch::Tensor p, pGN;
		// Gauss-Newton norm
		torch::Tensor pGN_Norm;
		// Trailpoint, step in dogleg direction
		torch::Tensor t;
		// Error att parameters, Error att trailing point
		torch::Tensor ep, et;
		// Gradient in dogleg search direction
		torch::Tensor Jp;
		// Step
		torch::Tensor step;
		// Gain Ratio
		// Predicted, Actual
		torch::Tensor predicted, actual;
		// Gain metric
		torch::Tensor rho;
		// Delta, trust region
		torch::Tensor delta;
		// Masks, determines type of search direction
		torch::Tensor mask, mask2;
		// Storage for logical not mask
		torch::Tensor not_mask;
		// Scaling variables
		torch::Tensor Jn, Jn2;
		torch::Tensor D;
		// Scaled Jacobian
		torch::Tensor Js;
		// Approximate Hessian
		torch::Tensor Hs;
		// Gradient
		torch::Tensor gs;
		// Cholesky factor, infos
		torch::Tensor cFactor, cSuccess;
		// Unscaled solution
		torch::Tensor q;

		// Cauchy-Point locals
		torch::Tensor invD, invD2;
		torch::Tensor invDgs, invD2gs;

		torch::Tensor lambdaStar;
		torch::Tensor CP;

		// Convergence
		torch::Tensor converges;
	};

}




