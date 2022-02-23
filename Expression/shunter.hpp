#pragma once

#include <deque>
#include "token.hpp"

namespace tc {
	namespace expression {

		class Shunter {
		public:

			std::deque<Token> shunt(std::vector<std::unique_ptr<Token>>&& tokens);



		};

	}
}