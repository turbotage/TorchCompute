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
				const std::vector<std::string>& parameters,
				tc::OptRef<const std::vector<std::string>> constants);

			MPExpr(const std::string& expression, 
				const std::vector<std::string>& diffexpressions,
				const tc::expression::FetcherMap& fetcher_map,
				const std::vector<std::string>& parameters,
				tc::OptRef<const std::vector<std::string>> constants);

			MPExpr(const std::string& expression, 
				const std::vector<std::string>& diffexpressions,
				const std::vector<std::string>& seconddiffexpressions,
				const tc::expression::FetcherMap& fetcher_map,
				const std::vector<std::string>& parameters,
				tc::OptRef<const std::vector<std::string>> constants);

			std::string expression;
			std::unique_ptr<tc::expression::Expression> eval;

			std::vector<std::string> diffexpressions;
			std::vector<std::unique_ptr<tc::expression::Expression>> diff;

			std::vector<std::string> seconddiffexpressions;
			std::vector<std::unique_ptr<tc::expression::Expression>> seconddiff;

			std::vector<std::string> parameters;
			std::optional<std::vector<std::string>> constants;

			const tc::expression::FetcherMap& fetcher_map;

		};

	}
}