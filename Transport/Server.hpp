#pragma once

#include "Threading/ThreadPool.hpp"
#include "gRPC/TransportService.hpp"

namespace transport {

    class Server {
    public:

        Server();

        void RunServer();

    private:
        std::unique_ptr<ThreadPool> m_pThreadPool;

        std::function<void(const ::Data*, ::Data*)> m_DataHandler;

        std::unique_ptr<::grpc::Server> m_pServer;


    }

}