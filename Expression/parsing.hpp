#pragma once

#include "../pch.hpp"

#include <functional>

namespace tc {
	namespace expression {

		enum eNumberType {
			NO_NUMBER,
			REAL,
			IMAGINARY,
			COMPLEX
		};
		typedef std::uint8_t NumberTypeBits;

		/// <summary>
		/// Could be used to construct a number tokenizer
		/// </summary>
		/// <param name="str"></param>
		/// <returns>(tuple) = <remaining string after extraxtion, extracted string, flag (eNumberType)> </returns>
		std::tuple<std::string, std::string, NumberTypeBits> extractNumberString(const std::string& str);

		/// <summary>
		/// Extracts a number from a string and converts it to a double
		/// </summary>
		/// <param name="str"></param>
		/// <returns></returns>
		std::tuple<std::string, double, NumberTypeBits> getNumberFromString(const std::string& str);

		/// <summary>
		/// Converts a string to a double and indicates if the number was a imaginary number of a real number
		/// OBS! This function should only be called with well formatted number strings, otherwise it throws
		/// </summary>
		/// <param name="str"></param>
		/// <returns></returns>
		std::tuple<double, NumberTypeBits> getNumber(std::string str);

		/// <summary>
		/// Used to construct number resolver
		/// </summary>
		/// <param name="str"></param>
		/// <param name="ops"></param>
		/// <returns></returns>
		std::function<torch::Tensor()> defaultNumberResolver(std::string str, torch::TensorOptions& ops);


	}
}