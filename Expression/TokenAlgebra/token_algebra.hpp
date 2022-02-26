#pragma once

#include "../token.hpp"

#include "Unary/unary.hpp"
#include "Unary/trig.hpp"
#include "Unary/neg.hpp"

#include "Binary/sub.hpp"
#include "Binary/pow.hpp"
#include "Binary/mul.hpp"
#include "Binary/div.hpp"
#include "Binary/add.hpp"

namespace tc {
	namespace expression {

		std::unique_ptr<Token> to_ptr(const ZeroToken& tok);
		std::unique_ptr<Token> to_ptr(const UnityToken& tok);
		std::unique_ptr<Token> to_ptr(const NegUnityToken& tok);
		std::unique_ptr<Token> to_ptr(const NanToken& tok);
		std::unique_ptr<Token> to_ptr(const NumberToken& tok);

		NumberToken to_num(const Token& tok);
		NumberToken to_num(const ZeroToken& tok);
		NumberToken to_num(const UnityToken& tok);
		NumberToken to_num(const NegUnityToken& tok);
		NumberToken to_num(const NanToken& tok);
		NumberToken to_num(const NumberToken& tok);

	}
}