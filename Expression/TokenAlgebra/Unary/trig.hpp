#pragma once

#include "../../token.hpp"

namespace tc {
	namespace expression {

		
		// Sin

		std::unique_ptr<Token> sin(const Token& in);
		ZeroToken sin(const ZeroToken& in);
		NumberToken sin(const NegUnityToken& in);
		NumberToken sin(const UnityToken& in);
		NanToken sin(const NanToken& in);
		NumberToken sin(const NumberToken& in);

		// Cos

		std::unique_ptr<Token> cos(const Token& in);
		UnityToken cos(const ZeroToken& in);
		NumberToken cos(const NegUnityToken& in);
		NumberToken cos(const UnityToken& in);
		NanToken cos(const NanToken& in);
		NumberToken cos(const NumberToken& in);

		// Tan

		std::unique_ptr<Token> tan(const Token& in);
		UnityToken tan(const ZeroToken& in);
		NumberToken tan(const NegUnityToken& in);
		NumberToken tan(const UnityToken& in);
		NanToken tan(const NanToken& in);
		NumberToken tan(const NumberToken& in);

		// Asin

		std::unique_ptr<Token> asin(const Token& in);
		UnityToken asin(const ZeroToken& in);
		NumberToken asin(const NegUnityToken& in);
		NumberToken asin(const UnityToken& in);
		NanToken asin(const NanToken& in);
		NumberToken asin(const NumberToken& in);

		// Acos

		std::unique_ptr<Token> acos(const Token& in);
		NumberToken acos(const ZeroToken& in);
		NumberToken acos(const NegUnityToken& in);
		ZeroToken acos(const UnityToken& in);
		NanToken acos(const NanToken& in);
		NumberToken acos(const NumberToken& in);

		// Atan

		std::unique_ptr<Token> atan(const Token& in);
		ZeroToken atan(const ZeroToken& in);
		NumberToken atan(const NegUnityToken& in);
		NumberToken atan(const UnityToken& in);
		NanToken atan(const NanToken& in);
		NumberToken atan(const NumberToken& in);

		// Sinh

		std::unique_ptr<Token> sinh(const Token& in);
		ZeroToken sinh(const ZeroToken& in);
		NumberToken sinh(const NegUnityToken& in);
		NumberToken sinh(const UnityToken& in);
		NanToken sinh(const NanToken& in);
		NumberToken sinh(const NumberToken& in);

		// Cosh

		std::unique_ptr<Token> cosh(const Token& in);
		UnityToken cosh(const ZeroToken& in);
		NumberToken cosh(const NegUnityToken& in);
		NumberToken cosh(const UnityToken& in);
		NanToken cosh(const NanToken& in);
		NumberToken cosh(const NumberToken& in);

		// Tanh

		std::unique_ptr<Token> tanh(const Token& in);
		ZeroToken tanh(const ZeroToken& in);
		NumberToken tanh(const NegUnityToken& in);
		NumberToken tanh(const UnityToken& in);
		NanToken tanh(const NanToken& in);
		NumberToken tanh(const NumberToken& in);

		// Asinh

		std::unique_ptr<Token> asinh(const Token& in);
		ZeroToken asinh(const ZeroToken& in);
		NumberToken asinh(const NegUnityToken& in);
		NumberToken asinh(const UnityToken& in);
		NanToken asinh(const NanToken& in);
		NumberToken asinh(const NumberToken& in);

		// Acosh

		std::unique_ptr<Token> acosh(const Token& in);
		NanToken acosh(const ZeroToken& in);
		NanToken acosh(const NegUnityToken& in);
		ZeroToken acosh(const UnityToken& in);
		NanToken acosh(const NanToken& in);
		NumberToken acosh(const NumberToken& in);

		// Atanh

		std::unique_ptr<Token> atanh(const Token& in);
		ZeroToken atanh(const ZeroToken& in);
		NanToken atanh(const NegUnityToken& in);
		NanToken atanh(const UnityToken& in);
		NanToken atanh(const NanToken& in);
		NumberToken atanh(const NumberToken& in);


	}
}
