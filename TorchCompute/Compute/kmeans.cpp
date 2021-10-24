#include "kmeans.hpp"
#include "random.hpp"


#include "cuda.h"
#include "cuda_runtime_api.h"

compute::KMeans::KMeans(int nclusters, int maxiter=50, float tol=0.001, KMeansModeBits mode=eKMeansMode::EUCLIDEAN) {
	m_nClusters = nclusters;
	m_MaxIter = maxiter;
	m_Tol = tol;
	m_KMeansMode = mode;
}

torch::Tensor compute::KMeans::cosineSimilarity(torch::Tensor a, torch::Tensor b) {
	auto a_norm = a.norm(c10::nullopt, -1, true);
	auto b_norm = b.norm(c10::nullopt, -1, true);
	a = a / (a_norm + torch::tensor(1e-8));
	b = b / (b_norm + torch::tensor(1e-8));
	return torch::matmul(a, b.transpose(-2, -1));
}

torch::Tensor compute::KMeans::euclideanSimilarity(torch::Tensor a, torch::Tensor b) {
	using namespace torch::indexing;
	return 2 * torch::matmul(a, b.transpose(-2, -1)) -
		torch::pow(a, 2).sum(1).index({ "...", Slice(), None }) -
		torch::pow(b, 2).sum(1).index({ "...", None, Slice() });
}

std::tuple<torch::Tensor, torch::Tensor> compute::KMeans::maxSimilarity(torch::Tensor a, torch::Tensor b) {
	using namespace torch::indexing;

	auto device = a.device();
	auto batch_size = a.size(0);

	std::function<torch::Tensor(torch::Tensor, torch::Tensor)> sim_func;
	if (m_KMeansMode == eKMeansMode::COSINE) {
		sim_func = [this](torch::Tensor a, torch::Tensor b) { return this->cosineSimilarity(a, b); };
	}
	else if (m_KMeansMode == eKMeansMode::EUCLIDEAN) {
		sim_func = [this](torch::Tensor a, torch::Tensor b) { return this->euclideanSimilarity(a, b); };
	}

	if (device.is_cpu()) {
		auto sim = sim_func(a, b);
		return sim.max(-1);
	}

	// Cuda
	uint64_t expected;
	if (a.dtype() == torch::kDouble)
		expected = a.size(0) * a.size(1) * b.size(0) * 8;
	else if (a.dtype() == torch::kFloat)
		expected = a.size(0) * a.size(1) * b.size(0) * 4;
	else if (a.dtype() == torch::kHalf)
		expected = a.size(0) * a.size(1) * b.size(0) * 2;
	else
		throw new std::runtime_error("Unsupported type for cuda device in KMeans routine");

	size_t free, total;
	cudaMemGetInfo(&free, &total);
	uint64_t allocated = total - free;

	uint64_t ratio = (uint64_t)std::ceil((double)expected / (double)allocated);
	uint64_t subbatch_size = std::ceil((double)batch_size / (double)ratio);
	std::vector<torch::Tensor> msv, msi;
	for (int i = 0; i < ratio; ++i) {
		if (i * subbatch_size >= batch_size)
			continue;

		uint64_t it1 = i * subbatch_size;
		uint64_t it2 = (i + 1) * subbatch_size;
		auto sub_x = a.index({ Slice(it1, it2) });
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
	}
	else {
		max_sim_v = torch::cat(torch::ArrayRef(msv.data(), msv.size()), 0);
		max_sim_i = torch::cat(torch::ArrayRef(msi.data(), msi.size()), 0);
	}
	return std::make_tuple(max_sim_v, max_sim_i);
}

torch::Tensor compute::KMeans::fit_predict(torch::Tensor X, std::optional<torch::Tensor> centroids) {
	using namespace torch::indexing;
	
	uint64_t batch_size = X.size(0);
	uint64_t emb_dim = X.size(1);

	auto device = X.device();
	auto ops = X.options();

	if (!centroids.has_value())
		centroids = X.index({ compute::random::random_choice(0, batch_size, m_nClusters, ops) });

	auto num_points_in_clusters = torch::ones(m_nClusters, ops);

	torch::Tensor closest;
	for (int i = 0; i < m_MaxIter; ++i) {
		closest = std::get<1>(maxSimilarity(X, centroids.value()));
		torch::Tensor matched_clusters, counts;
		std::tie(matched_clusters, std::ignore, counts) = torch::_unique2(closest, true, false, true);
		auto c_grad = torch::zeros_like(centroids.value());

		auto expanded_closest = closest.index({ None }).expand({ m_nClusters, -1 });
		auto mask = (expanded_closest == torch::arange(m_nClusters, ops).index({ Slice(), None })).to(torch::kFloat);
		c_grad = torch::matmul(mask, X ) / mask.sum(-1).index({ "...", Slice(), None });
		c_grad.index({ c_grad != c_grad }) = 0;

		auto error = (c_grad - centroids.value()).pow(2).sum();
		double lr = 1.0;
		num_points_in_clusters.index({ matched_clusters }) += counts;
		centroids = c_grad * lr;
		if ( error.item<double>() <= m_Tol )
			break;
	}

	return closest;
}
