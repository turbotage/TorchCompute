#pragma once

#include <torch/torch.h>

#include <memory>
#include <vector>
#include <string>
#include <functional>

#include <cstdint>
#include <optional>

typedef std::uint_fast64_t ui64;
typedef std::uint_fast32_t ui32;
typedef std::uint_fast16_t ui16;
typedef std::uint_fast8_t ui8;

typedef std::int_fast64_t i64;
typedef std::int_fast32_t i32;
typedef std::int_fast16_t i16;
typedef std::int_fast8_t i8;

// Used to signal output, functions with these parameters will fill the variable which the
// reference points to
template<typename T>
using OutRef = T&;

// Used to signal output, functions with these parameters will fill the variable which the
// reference points to if OptOutRef isn't std::nullopt
template<typename T>
using OptOutRef = std::optional<std::reference_wrapper<T>>;

enum eBuildMode {
	Release,
	Debug
};

constexpr ui8 BUILD_MODE = eBuildMode::Debug;

/*
Statements such as
if constexpr (BUILD_MODE == eBuildMode::Debug) {
	doSomething();
}
should be remove by Dead-Code-Compilation
*/