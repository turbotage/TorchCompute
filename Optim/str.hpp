#pragma once

#include "model.hpp"
#include "optim.hpp"

#include <limits>

namespace tc {
	namespace optim {

		struct STRPSettings : public OptimizerSettings {
		
			STRPSettings();

			float mu = 0.25;
			float eta = 0.75;
			float minimumTrustRadius = 1e-30;
		};

		struct STRPResult : public OptimResult {
			torch::Tensor finalDeltas;
		};

		class STRP : public Optimizer {
		public:

			STRP() = delete;
			STRP(const STRP&) = delete;
			STRP& operator=(const STRP&) = delete;

			STRP(STRPSettings& settings);

			STRPResult eval();

			std::unique_ptr<OptimResult> base_eval() override;

		private:

			void dogleg();

			void step();

			void solve();

		private:

			float m_Mu;
			float m_Eta;

			float m_MinimumTrustRadius;

		private:

			torch::Tensor data;

			torch::Tensor res;
			torch::Tensor res_tp;

			torch::Tensor p;

			torch::Tensor J;

			torch::Tensor JpD;

			torch::Tensor delta;

		};

	}
}