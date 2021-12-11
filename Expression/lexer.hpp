#pragma once

#include "../pch.hpp"

#include <tuple>
#include <map>
#include <functional>

namespace tc {
	namespace expression {

		struct LexResult {
			//			NumberName		// Number
			std::map<	std::string, torch::Tensor> numberMap;
			//			Newname					Original		Negate
			std::map<	std::string, std::string> variableNegateMaps;
		};

		class Lexer {
		public:

			Lexer() = default;

			std::tuple<std::map<std::string, torch::Tensor>, std::string> operator()(const std::string& expression);

			static std::map<std::string, std::string> lexfix(std::string& expression);

		};

	}
}