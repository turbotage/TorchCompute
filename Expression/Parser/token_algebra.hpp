#pragma once

#include "token.hpp"

namespace tc {
	namespace expression {

		ZeroToken operator-(const ZeroToken& other);
		UnityToken operator-(const NegUnityToken& other);
		NegUnityToken operator-(const UnityToken& other);
		NanToken operator-(const NanToken& other);
		NumberToken operator-(const NumberToken& other);
		
		// Add

		std::unique_ptr<Token> operator+(const ZeroToken& a, const ZeroToken& b);
		std::unique_ptr<Token> operator+(const ZeroToken& a, const NegUnityToken& b);
		std::unique_ptr<Token> operator+(const ZeroToken& a, const UnityToken& b);
		std::unique_ptr<Token> operator+(const ZeroToken& a, const NanToken& b);
		std::unique_ptr<Token> operator+(const ZeroToken& a, const NumberToken& b);

		std::unique_ptr<Token> operator+(const NegUnityToken& a, const ZeroToken& b);
		std::unique_ptr<Token> operator+(const NegUnityToken& a, const NegUnityToken& b);
		std::unique_ptr<Token> operator+(const NegUnityToken& a, const UnityToken& b);
		std::unique_ptr<Token> operator+(const NegUnityToken& a, const NanToken& b);
		std::unique_ptr<Token> operator+(const NegUnityToken& a, const NumberToken& b);

		std::unique_ptr<Token> operator+(const UnityToken& a, const ZeroToken& b);
		std::unique_ptr<Token> operator+(const UnityToken& a, const NegUnityToken& b);
		std::unique_ptr<Token> operator+(const UnityToken& a, const UnityToken& b);
		std::unique_ptr<Token> operator+(const UnityToken& a, const NanToken& b);
		std::unique_ptr<Token> operator+(const UnityToken& a, const NumberToken& b);

		std::unique_ptr<Token> operator+(const NanToken& a, const ZeroToken& b);
		std::unique_ptr<Token> operator+(const NanToken& a, const NegUnityToken& b);
		std::unique_ptr<Token> operator+(const NanToken& a, const UnityToken& b);
		std::unique_ptr<Token> operator+(const NanToken& a, const NanToken& b);
		std::unique_ptr<Token> operator+(const NanToken& a, const NumberToken& b);

		std::unique_ptr<Token> operator+(const NumberToken& a, const ZeroToken& b);
		std::unique_ptr<Token> operator+(const NumberToken& a, const NegUnityToken& b);
		std::unique_ptr<Token> operator+(const NumberToken& a, const UnityToken& b);
		std::unique_ptr<Token> operator+(const NumberToken& a, const NanToken& b);
		std::unique_ptr<Token> operator+(const NumberToken& a, const NumberToken& b);



		// Sub

		std::unique_ptr<Token> operator-(const ZeroToken& a, const ZeroToken& b);
		std::unique_ptr<Token> operator-(const ZeroToken& a, const NegUnityToken& b);
		std::unique_ptr<Token> operator-(const ZeroToken& a, const UnityToken& b);
		std::unique_ptr<Token> operator-(const ZeroToken& a, const NanToken& b);
		std::unique_ptr<Token> operator-(const ZeroToken& a, const NumberToken& b);

		std::unique_ptr<Token> operator-(const NegUnityToken& a, const ZeroToken& b);
		std::unique_ptr<Token> operator-(const NegUnityToken& a, const NegUnityToken& b);
		std::unique_ptr<Token> operator-(const NegUnityToken& a, const UnityToken& b);
		std::unique_ptr<Token> operator-(const NegUnityToken& a, const NanToken& b);
		std::unique_ptr<Token> operator-(const NegUnityToken& a, const NumberToken& b);

		std::unique_ptr<Token> operator-(const UnityToken& a, const ZeroToken& b);
		std::unique_ptr<Token> operator-(const UnityToken& a, const NegUnityToken& b);
		std::unique_ptr<Token> operator-(const UnityToken& a, const UnityToken& b);
		std::unique_ptr<Token> operator-(const UnityToken& a, const NanToken& b);
		std::unique_ptr<Token> operator-(const UnityToken& a, const NumberToken& b);

		std::unique_ptr<Token> operator-(const NanToken& a, const ZeroToken& b);
		std::unique_ptr<Token> operator-(const NanToken& a, const NegUnityToken& b);
		std::unique_ptr<Token> operator-(const NanToken& a, const UnityToken& b);
		std::unique_ptr<Token> operator-(const NanToken& a, const NanToken& b);
		std::unique_ptr<Token> operator-(const NanToken& a, const NumberToken& b);

		std::unique_ptr<Token> operator-(const NumberToken& a, const ZeroToken& b);
		std::unique_ptr<Token> operator-(const NumberToken& a, const NegUnityToken& b);
		std::unique_ptr<Token> operator-(const NumberToken& a, const UnityToken& b);
		std::unique_ptr<Token> operator-(const NumberToken& a, const NanToken& b);
		std::unique_ptr<Token> operator-(const NumberToken& a, const NumberToken& b);


		// Mul

		std::unique_ptr<Token> operator*(const ZeroToken& a, const ZeroToken& b);
		std::unique_ptr<Token> operator*(const ZeroToken& a, const NegUnityToken& b);
		std::unique_ptr<Token> operator*(const ZeroToken& a, const UnityToken& b);
		std::unique_ptr<Token> operator*(const ZeroToken& a, const NanToken& b);
		std::unique_ptr<Token> operator*(const ZeroToken& a, const NumberToken& b);

		std::unique_ptr<Token> operator*(const NegUnityToken& a, const ZeroToken& b);
		std::unique_ptr<Token> operator*(const NegUnityToken& a, const NegUnityToken& b);
		std::unique_ptr<Token> operator*(const NegUnityToken& a, const UnityToken& b);
		std::unique_ptr<Token> operator*(const NegUnityToken& a, const NanToken& b);
		std::unique_ptr<Token> operator*(const NegUnityToken& a, const NumberToken& b);

		std::unique_ptr<Token> operator*(const UnityToken& a, const ZeroToken& b);
		std::unique_ptr<Token> operator*(const UnityToken& a, const NegUnityToken& b);
		std::unique_ptr<Token> operator*(const UnityToken& a, const UnityToken& b);
		std::unique_ptr<Token> operator*(const UnityToken& a, const NanToken& b);
		std::unique_ptr<Token> operator*(const UnityToken& a, const NumberToken& b);

		std::unique_ptr<Token> operator*(const NanToken& a, const ZeroToken& b);
		std::unique_ptr<Token> operator*(const NanToken& a, const NegUnityToken& b);
		std::unique_ptr<Token> operator*(const NanToken& a, const UnityToken& b);
		std::unique_ptr<Token> operator*(const NanToken& a, const NanToken& b);
		std::unique_ptr<Token> operator*(const NanToken& a, const NumberToken& b);

		std::unique_ptr<Token> operator*(const NumberToken& a, const ZeroToken& b);
		std::unique_ptr<Token> operator*(const NumberToken& a, const NegUnityToken& b);
		std::unique_ptr<Token> operator*(const NumberToken& a, const UnityToken& b);
		std::unique_ptr<Token> operator*(const NumberToken& a, const NanToken& b);
		std::unique_ptr<Token> operator*(const NumberToken& a, const NumberToken& b);


	}
}