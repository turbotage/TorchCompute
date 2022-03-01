#include "../../../pch.hpp"

#include "neg.hpp"
#include "../token_algebra.hpp"

namespace tc {
namespace expression {

// <====================================== ABS ============================================>

std::unique_ptr<NumberBaseToken> abs(const Token& in)
{
	switch (in.get_token_type()) {
	case TokenType::ZERO_TYPE:
	{
		const ZeroToken& atok = static_cast<const ZeroToken&>(in);
		return std::make_unique<ZeroToken>(abs(atok));
	}
	case TokenType::UNITY_TYPE:
	{
		const UnityToken& atok = static_cast<const UnityToken&>(in);
		return std::make_unique<UnityToken>(abs(atok));
	}
	case TokenType::NEG_UNITY_TYPE:
	{
		const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
		return std::make_unique<UnityToken>(abs(atok));
	}
	case TokenType::NAN_TYPE:
	{
		const NanToken& atok = static_cast<const NanToken&>(in);
		return std::make_unique<NanToken>(abs(atok));
	}
	case TokenType::NUMBER_TYPE:
	{
		const NumberToken& atok = static_cast<const NumberToken&>(in);
		return std::make_unique<NumberToken>(abs(atok));
	}
	default:
		throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
	}
}

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

std::unique_ptr<NumberBaseToken> sqrt(const Token& in)
{
	switch (in.get_token_type()) {
	case TokenType::ZERO_TYPE:
	{
		const ZeroToken& atok = static_cast<const ZeroToken&>(in);
		return std::make_unique<ZeroToken>(sqrt(atok));
	}
	case TokenType::UNITY_TYPE:
	{
		const UnityToken& atok = static_cast<const UnityToken&>(in);
		return std::make_unique<UnityToken>(sqrt(atok));
	}
	case TokenType::NEG_UNITY_TYPE:
	{
		const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
		return std::make_unique<NumberToken>(sqrt(atok));
	}
	case TokenType::NAN_TYPE:
	{
		const NanToken& atok = static_cast<const NanToken&>(in);
		return std::make_unique<NanToken>(sqrt(atok));
	}
	case TokenType::NUMBER_TYPE:
	{
		const NumberToken& atok = static_cast<const NumberToken&>(in);
		return std::make_unique<NumberToken>(sqrt(atok));
	}
	default:
		throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
	}
}

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

// <====================================== SQUARE ============================================>

std::unique_ptr<NumberBaseToken> square(const Token& in)
{
	switch (in.get_token_type()) {
	case TokenType::ZERO_TYPE:
	{
		const ZeroToken& atok = static_cast<const ZeroToken&>(in);
		return std::make_unique<ZeroToken>(square(atok));
	}
	case TokenType::UNITY_TYPE:
	{
		const UnityToken& atok = static_cast<const UnityToken&>(in);
		return std::make_unique<UnityToken>(square(atok));
	}
	case TokenType::NEG_UNITY_TYPE:
	{
		const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
		return std::make_unique<UnityToken>(square(atok));
	}
	case TokenType::NAN_TYPE:
	{
		const NanToken& atok = static_cast<const NanToken&>(in);
		return std::make_unique<NanToken>(square(atok));
	}
	case TokenType::NUMBER_TYPE:
	{
		const NumberToken& atok = static_cast<const NumberToken&>(in);
		return std::make_unique<NumberToken>(square(atok));
	}
	default:
		throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
	}
}

ZeroToken square(const ZeroToken& in)
{
	return in;
}

UnityToken square(const NegUnityToken& in)
{
	return UnityToken(in.sizes);
}

UnityToken square(const UnityToken& in)
{
	return in;
}

NanToken square(const NanToken& in)
{
	return in;
}

NumberToken square(const NumberToken& in)
{
	return NumberToken(in.num*in.num, in.is_imaginary, in.sizes);
}

// <====================================== EXP ============================================>

std::unique_ptr<NumberBaseToken> exp(const Token& in)
{
	switch (in.get_token_type()) {
	case TokenType::ZERO_TYPE:
	{
		const ZeroToken& atok = static_cast<const ZeroToken&>(in);
		return std::make_unique<UnityToken>(exp(atok));
	}
	case TokenType::UNITY_TYPE:
	{
		const UnityToken& atok = static_cast<const UnityToken&>(in);
		return std::make_unique<NumberToken>(exp(atok));
	}
	case TokenType::NEG_UNITY_TYPE:
	{
		const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
		return std::make_unique<NumberToken>(exp(atok));
	}
	case TokenType::NAN_TYPE:
	{
		const NanToken& atok = static_cast<const NanToken&>(in);
		return std::make_unique<NanToken>(exp(atok));
	}
	case TokenType::NUMBER_TYPE:
	{
		const NumberToken& atok = static_cast<const NumberToken&>(in);
		return std::make_unique<NumberToken>(exp(atok));
	}
	default:
		throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
	}
}

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

std::unique_ptr<NumberBaseToken> log(const Token& in)
{
	switch (in.get_token_type()) {
	case TokenType::ZERO_TYPE:
	{
		const ZeroToken& atok = static_cast<const ZeroToken&>(in);
		return std::make_unique<NanToken>(log(atok));
	}
	case TokenType::UNITY_TYPE:
	{
		const UnityToken& atok = static_cast<const UnityToken&>(in);
		return std::make_unique<ZeroToken>(log(atok));
	}
	case TokenType::NEG_UNITY_TYPE:
	{
		const NegUnityToken& atok = static_cast<const NegUnityToken&>(in);
		return std::make_unique<NanToken>(log(atok));
	}
	case TokenType::NAN_TYPE:
	{
		const NanToken& atok = static_cast<const NanToken&>(in);
		return std::make_unique<NanToken>(log(atok));
	}
	case TokenType::NUMBER_TYPE:
	{
		const NumberToken& atok = static_cast<const NumberToken&>(in);
		return std::make_unique<NumberToken>(log(atok));
	}
	default:
		throw std::runtime_error("Token was not Zero, Unity, NegUnity, Nan and Number");
	}
}

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