#pragma once

#include "../pch.hpp"
#include "Threading/ThreadsafeQueue.hpp"
#include "gRPC/TransportService.hpp"

namespace transport {

    class Data {
    public:
        std::string bytes;
    };

    enum eDataType {
        REQUEST_DATA,
        PUSH_DATA,
    }

    class DataManager {
    public:

        DataManager();

        void AddData(std::string path, std::shared_ptr<Data> pData);

        std::shared_ptr<Data> GetData(std::string path);


        void HandleDataTransport(::Data* fromClient, ::Data* toClient);

    private:
        
        // ID(PATH(FUTURE,QUEUE(PAGE)))

        template<typename T>
        using Map = std::unordered_map<std::string, T>;

        using PathMap = Map<std::pair<std::future<bool>, std::queue<std::string>>;

        using IdMap = map<PathMap>;

        void DataRequested(::Data* fromClient, ::Data* toClient);

        void DataPushed(::Data* fromClient, ::Data* toClient);

    private:

        mutable std::mutex m_DataMutex;
        Map<std::shared_ptr<Data>> m_Data;
        
    
        mutable std::mutex m_RequestMutex;
        IdMap m_RequestMap;

        mutable std::mutex m_PushMutex;
        IdMap m_PushMap;

        
    };

}