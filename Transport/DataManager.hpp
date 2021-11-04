#pragma once

#include "../pch.hpp"
#include "Threading/ThreadsafeQueue.hpp"

namespace transport {


    class DataManager {
    public:

    private:
        
        std::unordered_map<std::string,
            std::unordered_map<std::string, 
                std::tuple<
                    std::future<bool>,
                    ThreadsafeQueue<std::string>
                >
            >
        > m_SerializedPageMap;
    };

}