#pragma once

#include "../pch.hpp"

#include "../Expression/expression.hpp"
#include "../Expression/nodes.hpp"

namespace tc {
	namespace optim {

		class MPExpr {
		public:

			MPExpr(const std::string& expression, 
				const tc::expression::FetcherMap& fetcher_map,
				const std::unordered_map<std::string, int>& parameter_map,
				tc::OptRef<const std::unordered_map<std::string, int>> per_problem_input_map,
				tc::OptRef<const std::unordered_map<std::string, int>> constant_map);

			MPExpr(const std::string& expression, const std::vector<std::string>& diffexpressions, 
				const tc::expression::FetcherMap& fetcher_map,
				const std::unordered_map<std::string, int>& parameter_map,
				tc::OptRef<const std::unordered_map<std::string, int>> per_problem_input_map,
				tc::OptRef<const std::unordered_map<std::string, int>> constant_map);

		private:

			std::unique_ptr<tc::expression::Expression> eval;
			std::unordered_map<int, tc::expression::Expression> diff;
			std::unordered_map<std::pair<int,int>, tc::expression::Expression> seconddiff;

			std::unordered_map<std::string, int> parameter_map;
			std::optional<std::unordered_map<std::string, int>> per_problem_input_map;
			std::optional<std::unordered_map<std::string, int>> constant_map;

			tc::expression::FetcherMap fetcher_map;

		};

	}
}