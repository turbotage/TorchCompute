#pragma once

#include "../pch.hpp"
#include "../Optim/model.hpp"

namespace models {

    // S = S_0*exp(-b*ADC), varying b values
    class ADCModel {
    public:

        ADCModel();

        // bvalues
        void setDependents(torch::Tensor bvals);

        void setData(torch::Tensor data);

        torch::Tensor solve();

    private:

        static torch::Tensor adc_func(std::vector<torch::Tensor> staticvars, torch::Tensor dependents, torch::Tensor parameters);

    private:

        std::unique_ptr<optim::Model> m_pModel;

        torch::Tensor m_Data;

    };
    

    // S = S_0 * sin(FA) * (1 - exp(-TR/T1))/ (1 - exp(-TR/T1)cos(FA)), varying flip angles (FA)
    class VFAModel_1 {
    public:

        VFAModel_1();

        // flip values
        void setDependents(torch::Tensor flip_angles);

        void setData(torch::Tensor data);

        torch::Tensor solve();

    private:

        static torch::Tensor vfa_func(std::vector<torch::Tensor> staticvars, torch::Tensor dependents, torch::Tensor parameters);

    private:

        std::unique_ptr<optim::Model> m_pModel;

        torch::Tensor m_Data;

    };

}
