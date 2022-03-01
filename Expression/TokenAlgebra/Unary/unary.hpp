#pragma once

#include "../../token.hpp"

namespace tc {
	namespace expression {

		// Abs

		std::unique_ptr<NumberBaseToken> abs(const Token& in);
		ZeroToken abs(const ZeroToken& in);
		UnityToken abs(const NegUnityToken& in);
		UnityToken abs(const UnityToken& in);
		NanToken abs(const NanToken& in);
		NumberToken abs(const NumberToken& in);

		// Sqrt

		std::unique_ptr<NumberBaseToken> sqrt(const Token& in);
		ZeroToken sqrt(const ZeroToken& in);
		NumberToken sqrt(const NegUnityToken& in);
		UnityToken sqrt(const UnityToken& in);
		NanToken sqrt(const NanToken& in);
		NumberToken sqrt(const NumberToken& in);

		// Square

		std::unique_ptr<NumberBaseToken> square(const Token& in);
		ZeroToken square(const ZeroToken& in);
		UnityToken square(const NegUnityToken& in);
		UnityToken square(const UnityToken& in);
		NanToken square(const NanToken& in);
		NumberToken square(const NumberToken& in);

		// Exp

		std::unique_ptr<NumberBaseToken> exp(const Token& in);
		UnityToken exp(const ZeroToken& in);
		NumberToken exp(const NegUnityToken& in);
		NumberToken exp(const UnityToken& in);
		NanToken exp(const NanToken& in);
		NumberToken exp(const NumberToken& in);

		// Log

		std::unique_ptr<NumberBaseToken> log(const Token& in);
		NanToken log(const ZeroToken& in);
		NanToken log(const NegUnityToken& in);
		ZeroToken log(const UnityToken& in);
		NanToken log(const NanToken& in);
		NumberToken log(const NumberToken& in);

	}
}
