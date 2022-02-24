#pragma once

#include <deque>
#include "token.hpp"

namespace tc {
	namespace expression {

		class Shunter {
		public:

			std::deque<std::unique_ptr<Token>> shunt(std::vector<std::unique_ptr<Token>>&& tokens);

		private:

			void handle_operator(const OperatorToken& op);

			void handle_rparen();

			bool shift_until(const Token& stop);

		private:

			std::deque<std::unique_ptr<Token>> m_Output;
			std::deque<std::unique_ptr<Token>> m_OperatorStack;

		};

	}
}