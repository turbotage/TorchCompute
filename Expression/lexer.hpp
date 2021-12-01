#pragma once

#include "../pch.hpp"

#include <tuple>
#include <map>
#include <functional>

namespace expression {
    
    std::map<std::string, std::string> lexfix(std::string& expression);


    class Lexer {
    public:

        Lexer() = default;

        std::tuple<std::map<std::string, torch::Tensor>, std::map<std::string, std::function<torch::Tensor(torch::Tensor)>>> operator()(std::string& expression)
        {
            std::map<std::string, std::string> namemap = lexfix(expression);

            std::map<std::string, torch::Tensor> numbermap;
            std::map<std::string, std::function<torch::Tensor(torch::Tensor)>> varmap;

            std::function<torch::Tensor(torch::Tensor)> normal_set = [](torch::Tensor in) {
                return in;
            };

            std::function<torch::Tensor(torch::Tensor)> negate_set = [](torch::Tensor in) {
                return -in;
            };

            for (auto& var : namemap) {
                
                if (var.first == var.second) { // This was a non negated variable
                    varmap.insert({var.first, normal_set});
                }
                else if ("-" + var.first == var.second)  { // This was a negated variable
                    varmap.insert({var.first, negate_set});
                }
                else  { // This was a number
                    if (var.second.back() == 'i') { // This was a imaginary number
                        std::string str = var.second;
                        str.pop_back();
                        numbermap.insert({var.first, torch::complex(torch::full({}, 0), torch::full({}, std::stod(str))) });
                    }
                    else {
                        numbermap.insert({var.first, torch::full({}, std::stod(var.second))});
                    }
                }

            }

            return std::make_tuple(numbermap, varmap);
        } 

    };

}