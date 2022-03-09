#pragma once

#include "../pch.hpp"

#include <optional>

namespace tc {
    namespace compute {

        enum eKMeansMode {
            COSINE,
            EUCLIDEAN
        };
        typedef uint8_t KMeansModeBits;

        class KMeans {
        public:

            KMeans(int nclusters, int maxiter, float tol, KMeansModeBits mode);

            torch::Tensor cosineSimilarity(torch::Tensor a, torch::Tensor b);

            torch::Tensor euclideanSimilarity(torch::Tensor a, torch::Tensor b);

            std::tuple<torch::Tensor, torch::Tensor> maxSimilarity(torch::Tensor a, torch::Tensor b);

            torch::Tensor fit_predict(torch::Tensor X, std::optional<torch::Tensor> centroids);

        private:
            int m_nClusters;
            int m_MaxIter;
            float m_Tol;
            KMeansModeBits m_KMeansMode;


        };

    }
}

