#include "lexer.hpp"

#include <regex>


bool is_negative(std::string& expression, int i) {
    if (i > 0) {
        if (expression[i-1] == '-') {
            if (i > 1) {
                char c = expression[i-2];
                if ( c == ',' || c == '(' ) {
                    return true;
                }
            }
            else {
                return true;
            }
        }
    }
    return false;
}

std::map<std::string, std::string> expression::lexfix(std::string& expression)
{
    std::string regstr = "^\\d+(([.]\\d+))?([eE][+-]?\\d+)?([i]?)";
    std::regex r(regstr);
    std::smatch m;

    std::map<std::string, std::string> varmap;

    int numbers = 0;
    for (int i = 0; i < expression.length(); ++i) {
        
        // This is a variable, read until it terminates
        if (expression[i] == '$') {
            bool is_neg = is_negative(expression, i);

            std::string varname;
            do {
                varname += expression[i];
                ++i;
            }
            while (i < expression.length() && std::isalnum(expression[i]));
            
            if (is_neg) {
                expression.erase(i-varname.length()-1, 1);
                varmap.insert({varname, "-" + varname});
            }
            else {
                varmap.insert({varname, varname});
            }

        }
        
        std::string substr = expression.substr(i);
        
        std::regex_search(substr, m, r);
        
        if (m[0].matched) {
            std::string match_str = m[0].str();
            
            std::string variable_name = "$NUMBERVAR" + std::to_string(numbers);
            
            bool is_neg = is_negative(expression, i);
            
            if (is_neg) {
                match_str = "-" + match_str;
                --i;
            }
            
            expression.erase(i, match_str.length());
            expression.insert(i, variable_name);
            
            varmap.insert({variable_name, match_str});
            
            i += variable_name.length();
            ++numbers;
        }
    }
    
    return varmap;
}



