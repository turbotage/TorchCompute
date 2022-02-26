#include "../../../pch.hpp"

#include "neg.hpp"
#include "../token_algebra.hpp"

namespace tc {
namespace expression {

// <====================================== ABS ============================================>

ZeroToken abs(const ZeroToken& in)
{
	return in;
}

UnityToken abs(const NegUnityToken& in)
{
	return UnityToken(in.sizes);
}

UnityToken abs(const UnityToken& in)
{
	return in;
}

NanToken abs(const NanToken& in)
{
	return in;
}

NumberToken abs(const NumberToken& in)
{
	return NumberToken(std::abs(in.num), false, in.sizes);
}


// <====================================== SQRT ============================================>

ZeroToken sqrt(const ZeroToken& in)
{
	return in;
}

NumberToken sqrt(const NegUnityToken& in)
{
	return NumberToken(1.0f, true, in.sizes);
}

UnityToken sqrt(const UnityToken& in)
{
	return in;
}

NanToken sqrt(const NanToken& in)
{
	return in;
}

NumberToken sqrt(const NumberToken& in)
{
	return NumberToken(std::sqrt(in.num), in.is_imaginary, in.sizes);
}

// <====================================== EXP ============================================>

UnityToken exp(const ZeroToken& in)
{
	return UnityToken(in.sizes);
}

NumberToken exp(const NegUnityToken& in)
{
	return NumberToken((float)std::exp(-1.0), true, in.sizes);
}

NumberToken exp(const UnityToken& in)
{
	return NumberToken((float)std::exp(1.0), true, in.sizes);
}

NanToken exp(const NanToken& in)
{
	return in;
}

NumberToken exp(const NumberToken& in)
{
	return NumberToken(std::exp(in.num), in.is_imaginary, in.sizes);
}

// <====================================== LOG ============================================>

NanToken log(const ZeroToken& in)
{
	return NanToken(in.sizes);
}

NanToken log(const NegUnityToken& in)
{
	return NanToken(in.sizes);
}

ZeroToken log(const UnityToken& in)
{
	return ZeroToken(in.sizes);
}

NanToken log(const NanToken& in)
{
	return in;
}

NumberToken log(const NumberToken& in)
{
	return NumberToken(std::log(in.num), in.is_imaginary, in.sizes);
}

}
}