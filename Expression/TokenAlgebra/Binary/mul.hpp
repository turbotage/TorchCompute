#pragma once

#include "../../token.hpp"

namespace tc {
	namespace expression {

		// Mul

		torch::Tensor operator*(const torch::Tensor& a, const Token& b);
		torch::Tensor operator*(const Token& a, const torch::Tensor& b);

		std::unique_ptr<Token> operator*(const Token& a, const Token& b);

		std::unique_ptr<Token> operator*(const ZeroToken& a, const Token& b);
		std::unique_ptr<Token> operator*(const UnityToken& a, const Token& b);
		std::unique_ptr<Token> operator*(const NegUnityToken& a, const Token& b);
		std::unique_ptr<Token> operator*(const NanToken& a, const Token& b);
		std::unique_ptr<Token> operator*(const NumberToken& a, const Token& b);

		ZeroToken operator*(const ZeroToken& a, const ZeroToken& b);
		ZeroToken operator*(const ZeroToken& a, const NegUnityToken& b);
		ZeroToken operator*(const ZeroToken& a, const UnityToken& b);
		NanToken operator*(const ZeroToken& a, const NanToken& b);
		ZeroToken operator*(const ZeroToken& a, const NumberToken& b);

		ZeroToken operator*(const NegUnityToken& a, const ZeroToken& b);
		UnityToken operator*(const NegUnityToken& a, const NegUnityToken& b);
		NegUnityToken operator*(const NegUnityToken& a, const UnityToken& b);
		NanToken operator*(const NegUnityToken& a, const NanToken& b);
		NumberToken operator*(const NegUnityToken& a, const NumberToken& b);

		ZeroToken operator*(const UnityToken& a, const ZeroToken& b);
		NegUnityToken operator*(const UnityToken& a, const NegUnityToken& b);
		UnityToken operator*(const UnityToken& a, const UnityToken& b);
		NanToken operator*(const UnityToken& a, const NanToken& b);
		NumberToken operator*(const UnityToken& a, const NumberToken& b);

		NanToken operator*(const NanToken& a, const ZeroToken& b);
		NanToken operator*(const NanToken& a, const NegUnityToken& b);
		NanToken operator*(const NanToken& a, const UnityToken& b);
		NanToken operator*(const NanToken& a, const NanToken& b);
		NanToken operator*(const NanToken& a, const NumberToken& b);

		ZeroToken operator*(const NumberToken& a, const ZeroToken& b); // This might lead to bugs if number is nan for instance
		NumberToken operator*(const NumberToken& a, const NegUnityToken& b);
		NumberToken operator*(const NumberToken& a, const UnityToken& b);
		NanToken operator*(const NumberToken& a, const NanToken& b);
		NumberToken operator*(const NumberToken& a, const NumberToken& b);

	}
}