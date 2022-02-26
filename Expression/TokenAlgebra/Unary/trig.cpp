#include "../../../pch.hpp"

#include "neg.hpp"
#include "../token_algebra.hpp"

namespace tc {
namespace expression {

// <====================================== SIN ============================================>

ZeroToken sin(const ZeroToken& in)
{
	return in;
}

NumberToken sin(const NegUnityToken& in)
{
	return NumberToken(std::sin(-1.0), false, in.sizes);
}

NumberToken sin(const UnityToken& in)
{
	return NumberToken(std::sin(1.0), false, in.sizes);
}

NanToken sin(const NanToken& in)
{
	return in;
}

NumberToken sin(const NumberToken& in)
{
	return NumberToken(std::sin(in.num), in.is_imaginary, in.sizes);
}

// <====================================== COS ============================================>

UnityToken cos(const ZeroToken& in)
{
	return UnityToken(in.sizes);
}

NumberToken cos(const NegUnityToken& in)
{
	return NumberToken(std::cos(-1.0), false, in.sizes);
}

NumberToken cos(const UnityToken& in)
{
	return NumberToken(std::cos(1.0), false, in.sizes);
}

NanToken cos(const NanToken& in)
{
	return in;
}

NumberToken cos(const NumberToken& in)
{
	return NumberToken(std::cos(in.num), in.is_imaginary, in.sizes);
}

// <====================================== TAN ============================================>

UnityToken tan(const ZeroToken& in)
{
	return UnityToken(in.sizes);
}

NumberToken tan(const NegUnityToken& in)
{
	return NumberToken(std::tan(-1.0), false, in.sizes);
}

NumberToken tan(const UnityToken& in)
{
	return NumberToken(std::tan(1.0), false, in.sizes);
}

NanToken tan(const NanToken& in)
{
	return in;
}

NumberToken tan(const NumberToken& in)
{
	return NumberToken(std::tan(in.num), in.is_imaginary, in.sizes);
}

// <====================================== ASIN ============================================>

UnityToken asin(const ZeroToken& in)
{
	return UnityToken(in.sizes);
}

NumberToken asin(const NegUnityToken& in)
{
	return NumberToken(std::asin(-1.0), false, in.sizes);
}

NumberToken asin(const UnityToken& in)
{
	return NumberToken(std::asin(-1.0), false, in.sizes);
}

NanToken asin(const NanToken& in)
{
	return in;
}

NumberToken asin(const NumberToken& in)
{
	return NumberToken(std::asin(in.num), in.is_imaginary, in.sizes);
}

// <====================================== ACOS ============================================>

NumberToken acos(const ZeroToken& in)
{
	return NumberToken(std::acos(0.0), false, in.sizes);
}

NumberToken acos(const NegUnityToken& in)
{
	return NumberToken(std::acos(-1.0), false, in.sizes);
}

ZeroToken acos(const UnityToken& in)
{
	return ZeroToken(in.sizes);
}

NanToken acos(const NanToken& in)
{
	return in;
}

NumberToken acos(const NumberToken& in)
{
	return NumberToken(std::acos(in.num), in.is_imaginary, in.sizes);
}

// <====================================== ATAN ============================================>

ZeroToken atan(const ZeroToken& in)
{
	return in;
}

NumberToken atan(const NegUnityToken& in)
{
	return NumberToken(std::atan(-1.0), false, in.sizes);
}

NumberToken atan(const UnityToken& in)
{
	return NumberToken(std::atan(1.0), false, in.sizes);
}

NanToken atan(const NanToken& in)
{
	return NanToken();
}

NumberToken atan(const NumberToken& in)
{
	return NumberToken(std::atan(in.num), in.is_imaginary, in.sizes);
}

// <====================================== SINH ============================================>

ZeroToken sinh(const ZeroToken& in)
{
	return in;
}

NumberToken sinh(const NegUnityToken& in)
{
	return NumberToken(std::sinh(-1.0), false, in.sizes);
}

NumberToken sinh(const UnityToken& in)
{
	return NumberToken(std::sinh(1.0), false, in.sizes);
}

NanToken sinh(const NanToken& in)
{
	return in;
}

NumberToken sinh(const NumberToken& in)
{
	return NumberToken(std::sinh(in.num), in.is_imaginary, in.sizes);
}

// <====================================== COSH ============================================>

UnityToken cosh(const ZeroToken& in)
{
	return UnityToken(in.sizes);
}

NumberToken cosh(const NegUnityToken& in)
{
	return NumberToken(std::cosh(-1.0), false, in.sizes);
}

NumberToken cosh(const UnityToken& in)
{
	return NumberToken(std::cosh(1.0), false, in.sizes);
}

NanToken cosh(const NanToken& in)
{
	return in;
}

NumberToken cosh(const NumberToken& in)
{
	return NumberToken(std::cosh(in.num), in.is_imaginary, in.sizes);
}

// <====================================== TANH ============================================>

ZeroToken tanh(const ZeroToken& in)
{
	return in;
}

NumberToken tanh(const NegUnityToken& in)
{
	return NumberToken(std::tanh(-1.0), false, in.sizes);
}

NumberToken tanh(const UnityToken& in)
{
	return NumberToken(std::tanh(1.0), false, in.sizes);
}

NanToken tanh(const NanToken& in)
{
	return in;
}

NumberToken tanh(const NumberToken& in)
{
	return NumberToken(std::tanh(in.num), in.is_imaginary, in.sizes);
}

// <====================================== ASINH ============================================>

ZeroToken asinh(const ZeroToken& in)
{
	return in;
}

NumberToken asinh(const NegUnityToken& in)
{
	return NumberToken(std::asinh(-1.0), false, in.sizes);
}

NumberToken asinh(const UnityToken& in)
{
	return NumberToken(std::asinh(1.0), false, in.sizes);
}

NanToken asinh(const NanToken& in)
{
	return in;
}

NumberToken asinh(const NumberToken& in)
{
	return NumberToken(std::asinh(in.num), in.is_imaginary, in.sizes);
}

// <====================================== ACOSH ============================================>

NanToken acosh(const ZeroToken& in)
{
	return NanToken(in.sizes);
}

NanToken acosh(const NegUnityToken& in)
{
	return NanToken(in.sizes);
}

ZeroToken acosh(const UnityToken& in)
{
	return ZeroToken(in.sizes);
}

NanToken acosh(const NanToken& in)
{
	return in;
}

NumberToken acosh(const NumberToken& in)
{
	return NumberToken(std::acosh(in.num), in.is_imaginary, in.sizes);
}

// <====================================== ATANH ============================================>

ZeroToken atanh(const ZeroToken& in)
{
	return in;
}

NanToken atanh(const NegUnityToken& in)
{
	return NanToken(in.sizes);
}

NanToken atanh(const UnityToken& in)
{
	return NanToken(in.sizes);
}

NanToken atanh(const NanToken& in)
{
	return in;
}

NumberToken atanh(const NumberToken& in)
{
	return NumberToken(std::atanh(in.num), in.is_imaginary, in.sizes);
}

}
}