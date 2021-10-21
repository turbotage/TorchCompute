#include "kmeans.hpp"

#include "cuda.h"
#include "cuda_runtime_api.h"

torch::Tensor compute::KMeans::cosineSimilarity(torch::Tensor a, torch::Tensor b) {
    auto a_norm = a.norm(-1, true);
    auto b_norm = b.norm(-1, true);
    a = a / (a_norm + 1e-8);
    b = b / (b_norm + 1e-8);
    return torch::matmul(a, b.transpose(-2,-1));
}

torch::Tensor compute::KMeans::euclideanSimilarity(torch::Tensor a, torch::Tensor b) {
    using namespace torch::indexing;
    return torch::matmul(a, b.transpose(-2, -1)) - 
            torch::pow(a,2).sum(1).index({"...", Slice(), None}) -
            torch::pow(b,2).sum(1).index({"...", None, Slice()});
}

std::tuple<torch::Tensor, torch::Tensor> compute::KMeans::maxSimilarity(torch::Tensor a, torch::Tensor b) {
    auto device = a.device();
    auto batch_size = a.size(0);
    
    std::function<torch::Tensor(torch::Tensor,torch::Tensor)> sim_func;
    if (m_KMeansMode == eKMeansMode::COSINE) {
        sim_func = [this](torch::Tensor a, torch::Tensor b) { return this->cosineSimilarity(a,b); };
    }
    else if (m_KMeansMode == eKMeansMode::EUCLIDEAN) {
        sim_func = [this](torch::Tensor a, torch::Tensor b) { return this->euclideanSimilarity(a,b); };
    }

    if (device.is_cpu()) {
        auto sim = sim_func(a, b);
        return sim.max(-1);
    }

    // Cuda
    int expected;
    if (a.dtype() == torch::kFloat64)
        expected = a.size(0) * a.size(1) * b.size(0) * 8;
    else if (a.dtype() == torch::kFloat32)
        expected = a.size(0) * a.size(1) * b.size(0) * 4;
    else if (a.dtype() == torch::kFloat16)
        expected = a.size(0) * a.size(1) * b.size(0) * 2;
    else
        throw new std::runtime_error("Unsupported type for cuda device in KMeans routine");

    size_t free, total;
    cudaMemGetInfo(&free, &total);
    int allocated = total - free;

    auto ratio = (int)std::ceil(expected / allocated);
    auto subbatch_size = std::ceil(batch_size / ratio);
    std::vector<torch::Tensor> msv, msi;
    for (int i = 0; i < ratio; ++i) {
        if (i*subbatch_size >= batch_size)
            continue;

        int it1 = i*subbatch_size;
        int it2 = (i+1)*subbatch_size;
        auto sub_x = a.index({it1, it2});
        auto sub_sim = sim_func(sub_x, b);
        torch::Tensor sub_max_sim_v, sub_max_sim_i;
        std::tie(sub_max_sim_v, sub_max_sim_i) = sub_sim.max(-1);
        msv.push_back(sub_max_sim_v);
        msi.push_back(sub_max_sim_i);
    }

    torch::Tensor max_sim_v, max_sim_i;
    if (ratio == 1) {
        max_sim_v = msv[0];
        max_sim_i = msi[0];
    } else {
        max_sim_v = torch::cat(torch::ArrayRef(msv.data(), msv.size()), 0);
        max_sim_i = torch::cat(torch::ArrayRef(msi.data(), msi.size()), 0);
    }
    return std::make_tuple(max_sim_v, max_sim_i);
}

torch::Tensor fit_predict(torch::Tensor X, std::optional<torch::Tensor> centroids) {
    int batch_size = X.size(0);
    int emb_dim = X.size(1);

    auto device = X.device();

    
    return torch::tensor({1});
}
