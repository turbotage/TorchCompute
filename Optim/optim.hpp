#pragma once

#include "../pch.hpp"

#include "model.hpp"

namespace optim {
    
    using OptimFetcher = std::function<
        void(
            // Model                        // Data
            std::unique_ptr<optim::Model>&, torch::Tensor&,
            // Residuals              // Jacobian
            OptOutRef<torch::Tensor>, OptOutRef<torch::Tensor>
        )>;

    void default_optim_fetcher(std::unique_ptr<optim::Model>& pModel, torch::Tensor& data,
        OptOutRef<torch::Tensor> residuals, OptOutRef<torch::Tensor> jacobian);

    struct OptimizerSettings {
        std::unique_ptr<optim::Model>           pModel;
        OptimFetcher                            optimFetcher;
        torch::TensorOptions                    startTensorOptions;
        torch::TensorOptions                    stopTensorOptions;
        float                                   tolerance = 1e-4;
        ui32                                    maxIter = 50;
    };

    class Optimizer {
    public:

        Optimizer(OptimizerSettings settings);

    protected:

        std::unique_ptr<optim::Model>           m_pModel;
        std::reference_wrapper<OptimFetcher>    m_OptimFetcher;
        torch::TensorOptions                    m_StartTensorOptions;
        torch::TensorOptions                    m_StopTensorOptions;
        float                                   m_Tolerance = 1e-4;
        ui32                                    m_MaxIter = 50;

    private:


    };

}