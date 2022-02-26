#include "../../../pch.hpp"

#include "pow.hpp"
#include "../token_algebra.hpp"

namespace tc {
namespace expression {

// <====================================== POW ============================================>

torch::Tensor pow(const torch::Tensor& a, const Token& b)
{
	switch (b.get_token_type()) {
	case TokenType::ZERO_TYPE:
	{
		const ZeroToken& btok = static_cast<const ZeroToken&>(b);
		auto outsizes = tc_broadcast_shapes(a.sizes(), btok.sizes);
		return torch::broadcast_to(torch::ones_like(a), outsizes);
	}
	case TokenType::UNITY_TYPE:
	{
		const UnityToken& btok = static_cast<const UnityToken&>(b);
		auto outsizes = tc_broadcast_shapes(a.sizes(), btok.sizes);
		return torch::broadcast_to(a, outsizes);
	}
	case TokenType::NEG_UNITY_TYPE:
	{
		const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
		auto outsizes = tc_broadcast_shapes(a.sizes(), btok.sizes);
		return torch::broadcast_to(1.0f / a, outsizes);
	}
	case TokenType::NAN_TYPE:
	{
		const NanToken& btok = static_cast<const NanToken&>(b);
		auto outsizes = tc_broadcast_shapes(a.sizes(), btok.sizes);
		return torch::broadcast_to(a + std::numeric_limits<float>::quiet_NaN(), outsizes);
	}
	case TokenType::NUMBER_TYPE:
	{
		const NumberToken& btok = static_cast<const NumberToken&>(b);
		auto outsizes = tc_broadcast_shapes(a.sizes(), btok.sizes);
		if (btok.is_imaginary)
			return torch::broadcast_to(torch::pow(a,c10::complex<float>(btok.num)), outsizes);
		return torch::broadcast_to(torch::pow(a, btok.num.real()), outsizes);
	}
	default:
		throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
	}
}

torch::Tensor pow(const Token& a, const torch::Tensor& b)
{
	switch (a.get_token_type()) {
	case TokenType::ZERO_TYPE:
	{
		const ZeroToken& atok = static_cast<const ZeroToken&>(a);
		auto outsizes = tc_broadcast_shapes(b.sizes(), atok.sizes);
		return torch::broadcast_to(torch::pow(0.0f, b), outsizes);
	}
	case TokenType::UNITY_TYPE:
	{
		const UnityToken& atok = static_cast<const UnityToken&>(a);
		auto outsizes = tc_broadcast_shapes(b.sizes(), atok.sizes);
		return torch::broadcast_to(torch::pow(1.0f, b), outsizes);
	}
	case TokenType::NEG_UNITY_TYPE:
	{
		const NegUnityToken& atok = static_cast<const NegUnityToken&>(a);
		auto outsizes = tc_broadcast_shapes(b.sizes(), atok.sizes);
		return torch::broadcast_to(torch::pow(-1.0f, b), outsizes);
	}
	case TokenType::NAN_TYPE:
	{
		const NanToken& atok = static_cast<const NanToken&>(a);
		auto outsizes = tc_broadcast_shapes(b.sizes(), atok.sizes);
		return torch::broadcast_to(std::numeric_limits<float>::quiet_NaN() - b, outsizes);
	}
	case TokenType::NUMBER_TYPE:
	{
		const NumberToken& atok = static_cast<const NumberToken&>(a);
		auto outsizes = tc_broadcast_shapes(b.sizes(), atok.sizes);
		if (atok.is_imaginary)
			return torch::broadcast_to(torch::pow(c10::complex<float>(atok.num),b), outsizes);
		return torch::broadcast_to(torch::pow(atok.num.real(), b), outsizes);
	}
	default:
		throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
	}
}


std::unique_ptr<Token> pow(const Token& a, const Token& b)
{
	switch (a.get_token_type()) {
	case TokenType::ZERO_TYPE:
	{
		const ZeroToken& atok = static_cast<const ZeroToken&>(a);
		return pow(atok, b);
	}
	case TokenType::UNITY_TYPE:
	{
		const UnityToken& atok = static_cast<const UnityToken&>(a);
		return pow(atok, b);
	}
	case TokenType::NEG_UNITY_TYPE:
	{
		const NegUnityToken& atok = static_cast<const NegUnityToken&>(a);
		return pow(atok, b);
	}
	case TokenType::NAN_TYPE:
	{
		const NanToken& atok = static_cast<const NanToken&>(a);
		return pow(atok, b);
	}
	case TokenType::NUMBER_TYPE:
	{
		const NumberToken& atok = static_cast<const NumberToken&>(a);
		return pow(atok, b);
	}
	default:
		throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
	}
}

std::unique_ptr<Token> pow(const ZeroToken& a, const Token& b)
{
	switch (b.get_token_type()) {
	case TokenType::ZERO_TYPE:
	{
		const ZeroToken& btok = static_cast<const ZeroToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::UNITY_TYPE:
	{
		const UnityToken& btok = static_cast<const UnityToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::NEG_UNITY_TYPE:
	{
		const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::NAN_TYPE:
	{
		const NanToken& btok = static_cast<const NanToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::NUMBER_TYPE:
	{
		const NumberToken& btok = static_cast<const NumberToken&>(b);
		return to_ptr(pow(a, btok));
	}
	default:
		throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
	}
}

std::unique_ptr<Token> pow(const UnityToken& a, const Token& b)
{
	switch (b.get_token_type()) {
	case TokenType::ZERO_TYPE:
	{
		const ZeroToken& btok = static_cast<const ZeroToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::UNITY_TYPE:
	{
		const UnityToken& btok = static_cast<const UnityToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::NEG_UNITY_TYPE:
	{
		const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::NAN_TYPE:
	{
		const NanToken& btok = static_cast<const NanToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::NUMBER_TYPE:
	{
		const NumberToken& btok = static_cast<const NumberToken&>(b);
		return to_ptr(pow(a, btok));
	}
	default:
		throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
	}
}

std::unique_ptr<Token> pow(const NegUnityToken& a, const Token& b)
{
	switch (b.get_token_type()) {
	case TokenType::ZERO_TYPE:
	{
		const ZeroToken& btok = static_cast<const ZeroToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::UNITY_TYPE:
	{
		const UnityToken& btok = static_cast<const UnityToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::NEG_UNITY_TYPE:
	{
		const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::NAN_TYPE:
	{
		const NanToken& btok = static_cast<const NanToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::NUMBER_TYPE:
	{
		const NumberToken& btok = static_cast<const NumberToken&>(b);
		return to_ptr(pow(a, btok));
	}
	default:
		throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
	}
}

std::unique_ptr<Token> pow(const NanToken& a, const Token& b)
{
	switch (b.get_token_type()) {
	case TokenType::ZERO_TYPE:
	{
		const ZeroToken& btok = static_cast<const ZeroToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::UNITY_TYPE:
	{
		const UnityToken& btok = static_cast<const UnityToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::NEG_UNITY_TYPE:
	{
		const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::NAN_TYPE:
	{
		const NanToken& btok = static_cast<const NanToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::NUMBER_TYPE:
	{
		const NumberToken& btok = static_cast<const NumberToken&>(b);
		return to_ptr(pow(a, btok));
	}
	default:
		throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
	}
}

std::unique_ptr<Token> pow(const NumberToken& a, const Token& b)
{
	switch (b.get_token_type()) {
	case TokenType::ZERO_TYPE:
	{
		const ZeroToken& btok = static_cast<const ZeroToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::UNITY_TYPE:
	{
		const UnityToken& btok = static_cast<const UnityToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::NEG_UNITY_TYPE:
	{
		const NegUnityToken& btok = static_cast<const NegUnityToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::NAN_TYPE:
	{
		const NanToken& btok = static_cast<const NanToken&>(b);
		return to_ptr(pow(a, btok));
	}
	case TokenType::NUMBER_TYPE:
	{
		const NumberToken& btok = static_cast<const NumberToken&>(b);
		return to_ptr(pow(a, btok));
	}
	default:
		throw std::runtime_error("Negation can only be applied to tokens Zero, Unity, NegUnity, Nan and Number");
	}
}


// 1
UnityToken pow(const ZeroToken& a, const ZeroToken& b)
{
	return UnityToken(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

NanToken pow(const ZeroToken& a, const NegUnityToken& b)
{
	return NanToken(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

ZeroToken pow(const ZeroToken& a, const UnityToken& b)
{
	return ZeroToken(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

NanToken pow(const ZeroToken& a, const NanToken& b)
{
	return NanToken(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

NumberToken pow(const ZeroToken& a, const NumberToken& b)
{
	return NumberToken(std::pow(0.0f, b.num), b.is_imaginary, tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

// 2
UnityToken pow(const NegUnityToken& a, const ZeroToken& b)
{
	return UnityToken(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

NegUnityToken pow(const NegUnityToken& a, const NegUnityToken& b)
{
	return NegUnityToken(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

NegUnityToken pow(const NegUnityToken& a, const UnityToken& b)
{
	return NegUnityToken(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

NanToken pow(const NegUnityToken& a, const NanToken& b)
{
	return NanToken(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

NumberToken pow(const NegUnityToken& a, const NumberToken& b)
{
	return NumberToken(std::pow(-1.0f, b.num), b.is_imaginary, tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

// 3
UnityToken pow(const UnityToken& a, const ZeroToken& b)
{
	return UnityToken(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

UnityToken pow(const UnityToken& a, const NegUnityToken& b)
{
	return UnityToken(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

UnityToken pow(const UnityToken& a, const UnityToken& b)
{
	return UnityToken(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

NanToken pow(const UnityToken& a, const NanToken& b)
{
	return NanToken(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

NumberToken pow(const UnityToken& a, const NumberToken& b)
{
	return NumberToken(std::pow(1.0f, b.num), b.is_imaginary, tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

// 4
UnityToken pow(const NanToken& a, const ZeroToken& b)
{
	return UnityToken(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

NanToken pow(const NanToken& a, const NegUnityToken& b)
{
	return NanToken(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

NanToken pow(const NanToken& a, const UnityToken& b)
{
	return NanToken(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

NanToken pow(const NanToken& a, const NanToken& b)
{
	return NanToken(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

NanToken pow(const NanToken& a, const NumberToken& b)
{
	return NanToken(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

// 5
NumberToken pow(const NumberToken& a, const ZeroToken& b)
{
	return NumberToken(std::pow(a.num, 0.0f), a.is_imaginary, tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

NumberToken pow(const NumberToken& a, const NegUnityToken& b)
{
	return NumberToken(std::pow(a.num, -1.0f), a.is_imaginary, tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

NumberToken pow(const NumberToken& a, const UnityToken& b)
{
	return NumberToken(std::pow(a.num, 1.0f), a.is_imaginary, tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

NanToken pow(const NumberToken& a, const NanToken& b)
{
	return NanToken(tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

NumberToken pow(const NumberToken& a, const NumberToken& b)
{
	return NumberToken(std::pow(a.num, b.num), a.is_imaginary || b.is_imaginary, tc::tc_broadcast_shapes(a.sizes, b.sizes));
}

}
}