#pragma once

#include "Transport.grpc.pb.h"

#include <functional>

namespace transport {

    // ----------------------------- REQUEST-DATA --- RESPONSE-DATA -----------------
    using DataHandler = std::function<void(const ::Data*,::Data*)>;

    class TransportServiceImpl final : public TransportService::Service {
    public:

        TransportServiceImpl(DataHandler dataHandler);

        ::grpc::Status TransportRPC(::grpc::ServerContext* context, const ::Data* request, ::Data* response);

    private:

        DataHandler m_DataHandler;

    };


}


