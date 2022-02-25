#include "../../pch.hpp"

#include "token_algebra.hpp"

// <===================================== NEG =========================================>

tc::expression::ZeroToken tc::expression::operator-(const ZeroToken& other)
{
	return other;
}

tc::expression::UnityToken tc::expression::operator-(const NegUnityToken& other)
{
	return UnityToken(other.sizes);
}

tc::expression::NegUnityToken tc::expression::operator-(const UnityToken& other)
{
	return NegUnityToken(other.sizes);
}

tc::expression::NanToken tc::expression::operator-(const NanToken& other)
{
	return other;
}

tc::expression::NumberToken tc::expression::operator-(const NumberToken& other)
{
	return NumberToken(-other.num, other.is_imaginary, other.sizes);
}


// <====================================== ADD ============================================>
//1
std::unique_ptr<tc::expression::Token> tc::expression::operator+(const ZeroToken& a, const ZeroToken& b)
{
	return std::make_unique<ZeroToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const ZeroToken& a, const NegUnityToken& b)
{
	return std::make_unique<NegUnityToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const ZeroToken& a, const UnityToken& b)
{
	return std::make_unique<UnityToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const ZeroToken& a, const NanToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const ZeroToken& a, const NumberToken& b)
{
	return std::make_unique<NumberToken>(b.num, b.is_imaginary, tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

// 2
std::unique_ptr<tc::expression::Token> tc::expression::operator+(const NegUnityToken& a, const ZeroToken& b)
{
	return b + a;
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const NegUnityToken& a, const NegUnityToken& b)
{
	return std::make_unique<NumberToken>(-2.0f, false, tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const NegUnityToken& a, const UnityToken& b)
{
	return std::make_unique<ZeroToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const NegUnityToken& a, const NanToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const NegUnityToken& a, const NumberToken& b)
{
	return std::make_unique<NumberToken>(b.num - 1.0f, b.is_imaginary, tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

// 3
std::unique_ptr<tc::expression::Token> tc::expression::operator+(const UnityToken& a, const ZeroToken& b)
{
	return b + a;
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const UnityToken& a, const NegUnityToken& b)
{
	return b + a;
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const UnityToken& a, const UnityToken& b)
{
	return std::make_unique<NumberToken>(-2.0f, false, tc::tc_broadcast_shapes(a.sizes, b.sizes)); // could be done like -((-a)+(-a))
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const UnityToken& a, const NanToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const UnityToken& a, const NumberToken& b)
{
	return std::make_unique<NumberToken>(b.num + 1.0f, b.is_imaginary, tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

// 4
std::unique_ptr<tc::expression::Token> tc::expression::operator+(const NanToken& a, const ZeroToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const NanToken& a, const NegUnityToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const NanToken& a, const UnityToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const NanToken& a, const NanToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const NanToken& a, const NumberToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

// 5
std::unique_ptr<tc::expression::Token> tc::expression::operator+(const NumberToken& a, const ZeroToken& b)
{
	return b + a;
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const NumberToken& a, const NegUnityToken& b)
{
	return b + a;
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const NumberToken& a, const UnityToken& b)
{
	return b + a;
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const NumberToken& a, const NanToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator+(const NumberToken& a, const NumberToken& b)
{
	auto num = a.num + b.num;
	auto numstr = std::to_string(num.real()) + "+" + std::to_string(num.imag()) + "i";
	return std::make_unique<NumberToken>(numstr, num, a.is_imaginary || b.is_imaginary, tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

// <====================================== ADD ============================================>

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const ZeroToken& a, const ZeroToken& b)
{
	return a + (-b);
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const ZeroToken& a, const NegUnityToken& b)
{
	return a + (-b);
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const ZeroToken& a, const UnityToken& b)
{
	return a + (-b);
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const ZeroToken& a, const NanToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const ZeroToken& a, const NumberToken& b)
{
	return a + (-b);
}

// 2
std::unique_ptr<tc::expression::Token> tc::expression::operator-(const NegUnityToken& a, const ZeroToken& b)
{
	return a + (-b);
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const NegUnityToken& a, const NegUnityToken& b)
{
	return a + (-b);
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const NegUnityToken& a, const UnityToken& b)
{
	return a + (-b);
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const NegUnityToken& a, const NanToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const NegUnityToken& a, const NumberToken& b)
{
	return a + (-b);
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const UnityToken& a, const ZeroToken& b)
{
	return a + (-b);
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const UnityToken& a, const NegUnityToken& b)
{
	return a + (-b);
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const UnityToken& a, const UnityToken& b)
{
	return a + (-b);
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const UnityToken& a, const NanToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const UnityToken& a, const NumberToken& b)
{
	return a + (-b);
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const NanToken& a, const ZeroToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const NanToken& a, const NegUnityToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const NanToken& a, const UnityToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const NanToken& a, const NanToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const NanToken& a, const NumberToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const NumberToken& a, const ZeroToken& b)
{
	return a + (-b);
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const NumberToken& a, const NegUnityToken& b)
{
	return a + (-b);
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const NumberToken& a, const UnityToken& b)
{
	return a + (-b);
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const NumberToken& a, const NanToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator-(const NumberToken& a, const NumberToken& b)
{
	return a + (-b);
}

// <====================================== MUL ============================================>

//1
std::unique_ptr<tc::expression::Token> tc::expression::operator*(const ZeroToken& a, const ZeroToken& b)
{
	return std::make_unique<ZeroToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const ZeroToken& a, const NegUnityToken& b)
{
	return std::make_unique<ZeroToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const ZeroToken& a, const UnityToken& b)
{
	return std::make_unique<ZeroToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const ZeroToken& a, const NanToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const ZeroToken& a, const NumberToken& b)
{
	return std::make_unique<ZeroToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

// 2
std::unique_ptr<tc::expression::Token> tc::expression::operator*(const NegUnityToken& a, const ZeroToken& b)
{
	return std::make_unique<ZeroToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const NegUnityToken& a, const NegUnityToken& b)
{
	return std::make_unique<UnityToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const NegUnityToken& a, const UnityToken& b)
{
	return std::make_unique<NegUnityToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const NegUnityToken& a, const NanToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const NegUnityToken& a, const NumberToken& b)
{
	return std::make_unique<NumberToken>(-b.num, b.is_imaginary, tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

// 3
std::unique_ptr<tc::expression::Token> tc::expression::operator*(const UnityToken& a, const ZeroToken& b)
{
	return std::make_unique<ZeroToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const UnityToken& a, const NegUnityToken& b)
{
	return std::make_unique<NegUnityToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const UnityToken& a, const UnityToken& b)
{
	return std::make_unique<UnityToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const UnityToken& a, const NanToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const UnityToken& a, const NumberToken& b)
{
	return std::make_unique<NumberToken>(b.num, b.is_imaginary, tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

// 4
std::unique_ptr<tc::expression::Token> tc::expression::operator*(const NanToken& a, const ZeroToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const NanToken& a, const NegUnityToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const NanToken& a, const UnityToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const NanToken& a, const NanToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const NanToken& a, const NumberToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const NumberToken& a, const ZeroToken& b)
{
	return b * a;
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const NumberToken& a, const NegUnityToken& b)
{
	return b * a;
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const NumberToken& a, const UnityToken& b)
{
	return b * a;
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const NumberToken& a, const NanToken& b)
{
	return std::make_unique<NanToken>(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

std::unique_ptr<tc::expression::Token> tc::expression::operator*(const NumberToken& a, const NumberToken& b)
{
	return std::make_unique<NumberToken>(a.num * b.num, a.is_imaginary | b.is_imaginary, tc::tc_broadcast_shapes(a.sizes, b.sizes));
}






