#pragma once

#include "../../token.hpp"

namespace tc {
	namespace expression {

		// Abs

		ZeroToken abs(const ZeroToken& in);
		UnityToken abs(const NegUnityToken& in);
		UnityToken abs(const UnityToken& in);
		NanToken abs(const NanToken& in);
		NumberToken abs(const NumberToken& in);

		// Sqrt

		ZeroToken sqrt(const ZeroToken& in);
		NumberToken sqrt(const NegUnityToken& in);
		UnityToken sqrt(const UnityToken& in);
		NanToken sqrt(const NanToken& in);
		NumberToken sqrt(const NumberToken& in);

		// Exp

		UnityToken exp(const ZeroToken& in);
		NumberToken exp(const NegUnityToken& in);
		NumberToken exp(const UnityToken& in);
		NanToken exp(const NanToken& in);
		NumberToken exp(const NumberToken& in);

		// Log

		NanToken log(const ZeroToken& in);
		NanToken log(const NegUnityToken& in);
		ZeroToken log(const UnityToken& in);
		NanToken log(const NanToken& in);
		NumberToken log(const NumberToken& in);

	}
}
