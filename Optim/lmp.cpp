#include "lmp.hpp"


optim::LMP::LMP(LMPSettings settings)
    : Optimizer(settings), m_Mu(settings.mu), m_Eta(settings.eta)
{

}

void optim::LMP::dogleg()
{
    c10::InferenceMode im_guard;

    torch::Tensor& D = pr_pa_pa_1;
    torch::Tensor& gs = pr_pa_1_1; 
    torch::Tensor& Hs = pr_pa_pa_2;
    
    {
        torch::Tensor& Js = temp2; // alias an available temp variable
        D = torch::diag_embed(torch::rsqrt((J*J).sum(1)));      // Scaling matrix
        Js = torch::bmm(J,D);                                // Scale Jacobian
        gs = torch::bmm(Js.transpose(1,2), res);
        Hs = torch::bmm(Js.transpose(1,2), Js);
    }

    std::tie(Hs, mask) = torch::linalg_cholesky_ex(Hs);

    if (pD.size(0) != numProbs || pD.size(1) != numParams) {
        pD = torch::empty({numProbs, numParams, 1}, m_CurrentTensorOptions);
    }

    pD.index_put_({})



    

}