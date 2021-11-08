#pragma once

#include "pch.hpp"
#include "Threading/ThreadsafeQueue.hpp"
#include "gRPC/TransportService.hpp"
#include "Threading/ThreadPool.hpp"

namespace transport {

    const ui32 DEFAULT_PAGE_SIZE = 64; // 64*1024

    class Data {
    public:

        Data() = default;
        Data(Data&&) = default;

        std::string NextPage();

    private:

        std::string bytes;
    };

    enum eDataType {
        MESSAGE,
        REQUEST_DATA,
        RESPONSE_DATA,
        NO_DATA,
        PUSH_DATA,
    };

    class DataManager {
    public:

        DataManager(std::shared_ptr<transport::ThreadPool> threadpool);

        void AddData(std::string id, std::string path, Data pData);

        Data& GetData(std::string id, std::string path);

        void HandleDataTransport(::Data* fromClient, ::Data* toClient);

        void DataRequested(::Data* fromClient, ::Data* toClient);

        // Only guarenteed to be atomic
        void DataPushed(::Data* fromClient, ::Data* toClient);

    private:
        
        // ID(PATH(FUTURE,QUEUE(PAGE)))
        enum eDataState {
            DEFAULT,
            SERIALIZING,
            SERIALIZED,
            WAITING_DATAPUSH,
            DESERIALIZING,
            DESERIALIZED
        };

        struct DataTup {
            
            DataTup() = default;

            Data data;
            ui16 data_state;
            std::queue<std::string> pages;
        };

    private:

        std::shared_ptr<transport::ThreadPool> m_pThreadPool;
    
        mutable std::mutex m_RequestMutex;
        std::map<std::string, std::map<std::string, DataTup>> m_RequestMap;

        mutable std::mutex m_PushMutex;
        std::map<std::string, std::map<std::string, DataTup>> m_PushMap;

        
    };

}