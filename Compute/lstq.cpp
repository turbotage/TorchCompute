
#include "lstq.hpp"

torch::Tensor compute::lstq_qr(torch::Tensor x, torch::Tensor y) {

    int64_t nProbs = x.size(0);
    int64_t nPoints = x.size(1);
    int64_t nParams = x.size(2);

    torch::Tensor A = torch::cat({torch::ones({nProbs,nPoints,1}), x}, 2);

    torch::Tensor Q, R;

    std::tie(Q,R) = torch::linalg::qr(A);

    torch::Tensor ATy = torch::bmm(A.transpose(1,2), y.view({y.size(0), y.size(1), 1}));

    torch::Tensor b;

    std::tie(b,std::ignore) = torch::triangular_solve(ATy, R.transpose(1,2), false);

    std::tie(b,std::ignore) = torch::triangular_solve(b, R);

    return b;
}

// Not implemented
torch::Tensor compute::lstq_svd(torch::Tensor x, torch::Tensor y) {

    int64_t nProbs = x.size(0);
    int64_t nPoints = x.size(1);
    int64_t nParams = x.size(2);

    torch::Tensor A = torch::cat({torch::ones({nProbs,nPoints,1}), x}, 2);

    torch::Tensor U,S,VT;

    std::tie(U,S,VT) = torch::linalg::svd(A, false);

    torch::Tensor b;

    b = torch::bmm(VT.transpose(1,2), torch::diag_embed(1 / S));
    b = torch::bmm(b, U.transpose(1,2));
    b = torch::bmm(b, y);

    return b;
}
