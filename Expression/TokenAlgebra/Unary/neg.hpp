#pragma once

#include "../../token.hpp"

namespace tc {
	namespace expression {

		// Neg

		std::unique_ptr<Token> operator-(const Token& other);

		ZeroToken operator-(const ZeroToken& other);
		UnityToken operator-(const NegUnityToken& other);
		NegUnityToken operator-(const UnityToken& other);
		NanToken operator-(const NanToken& other);
		NumberToken operator-(const NumberToken& other);

	}
}




