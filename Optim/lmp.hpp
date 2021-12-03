#pragma once

#include "../pch.hpp"

#include "model.hpp"
#include "optim.hpp"

namespace optim {

    struct LMPSettings : OptimizerSettings {
        float mu;
        float eta;

    }

    class LMP : public Optimizer{
    public:

        LMP(LMPSettings settings);

    private:

        void dogleg();

    private:

        torch::TensorOptions m_CurrentTensorOptions;
        float m_Mu;
        float m_Eta;

    private:


        i32 numProbs;
        i32 numInputs;
        i32 numParams;

        // (nProbs, nPerProbInps, 1) (PRIN)
        torch::Tensor res;
        //torch::Tensor pr_in_1_1;

        // (nProbs, nParams, 1) (PRPA)
        torch::Tensor pr_pa_1_1;

        // (nProbs, nPerProbInps, nParams) (PRINPA)
        torch::Tensor J;
        torch::Tensor pr_in_pa_1;
        torch::Tensor pr_in_pa_2;

        // (nProbs, nParams, nParams) (PRPAPA)
        torch::Tensor pr_pa_pa_1;
        torch::Tensor pr_pa_pa_2;

        torch::Tensor pD;
        torch::Tensor temp4;

        torch::Tensor step_mask;
        torch::Tensor delta_mask;

    }

}
