#include "pch.hpp"

std::vector<int64_t> tc::tc_broadcast_shapes(const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2)
{
	if (shape1.size() == 0 || shape2.size() == 0)
		throw std::runtime_error("shapes must have atleast one dimension to be broadcastable");

	auto& small = (shape1.size() > shape2.size()) ? shape2 : shape1;
	auto& big = (shape1.size() > shape2.size()) ? shape1 : shape2;

	std::vector<int64_t> ret(big.size());

	auto retit = ret.rbegin();
	auto smallit = small.rbegin();
	for (auto bigit = big.rbegin(); bigit != big.rend(); ) {
		if (smallit != small.rend()) {
			if (*smallit == *bigit) {
				*retit = *bigit;
			}
			else if (*smallit > *bigit && *bigit == 1) {
				*retit = *smallit;
			}
			else if (*bigit > *smallit && *smallit == 1) {
				*retit = *bigit;
			}
			else {
				throw std::runtime_error("shapes where not broadcastable");
			}
			++smallit;
		}
		else {
			*retit = *bigit;
		}

		++bigit;
		++retit;
	}

	return ret;
}
