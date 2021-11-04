#pragma once

#include "Threading/ThreadPool.hpp"

namespace transport {

    class Server {
    public:

        Server();

        void RunServer();

    private:
        std::unique_ptr<ThreadPool> m_pThreadPool;

        DataHandler m_DataHandler;

        std::unique_ptr<::grpc::Server> m_pServer;


    }

}