#pragma once

#include "../../token.hpp"

namespace tc {
	namespace expression {

		// Pow

		torch::Tensor pow(const torch::Tensor& a, const Token& b);
		torch::Tensor pow(const Token& a, const torch::Tensor& b);

		std::unique_ptr<Token> pow(const Token& a, const Token& b);

		std::unique_ptr<Token> pow(const ZeroToken& a, const Token& b);
		std::unique_ptr<Token> pow(const UnityToken& a, const Token& b);
		std::unique_ptr<Token> pow(const NegUnityToken& a, const Token& b);
		std::unique_ptr<Token> pow(const NanToken& a, const Token& b);
		std::unique_ptr<Token> pow(const NumberToken& a, const Token& b);

		UnityToken pow(const ZeroToken& a, const ZeroToken& b);
		NanToken pow(const ZeroToken& a, const NegUnityToken& b);
		ZeroToken pow(const ZeroToken& a, const UnityToken& b);
		NanToken pow(const ZeroToken& a, const NanToken& b);
		NumberToken pow(const ZeroToken& a, const NumberToken& b);

		UnityToken pow(const NegUnityToken& a, const ZeroToken& b);
		NegUnityToken pow(const NegUnityToken& a, const NegUnityToken& b);
		NegUnityToken pow(const NegUnityToken& a, const UnityToken& b);
		NanToken pow(const NegUnityToken& a, const NanToken& b);
		NumberToken pow(const NegUnityToken& a, const NumberToken& b);

		UnityToken pow(const UnityToken& a, const ZeroToken& b);
		UnityToken pow(const UnityToken& a, const NegUnityToken& b);
		UnityToken pow(const UnityToken& a, const UnityToken& b);
		NanToken pow(const UnityToken& a, const NanToken& b);
		NumberToken pow(const UnityToken& a, const NumberToken& b);

		UnityToken pow(const NanToken& a, const ZeroToken& b);
		NanToken pow(const NanToken& a, const NegUnityToken& b);
		NanToken pow(const NanToken& a, const UnityToken& b);
		NanToken pow(const NanToken& a, const NanToken& b);
		NanToken pow(const NanToken& a, const NumberToken& b);

		NumberToken pow(const NumberToken& a, const ZeroToken& b);
		NumberToken pow(const NumberToken& a, const NegUnityToken& b);
		NumberToken pow(const NumberToken& a, const UnityToken& b);
		NanToken pow(const NumberToken& a, const NanToken& b);
		NumberToken pow(const NumberToken& a, const NumberToken& b);

	}
}