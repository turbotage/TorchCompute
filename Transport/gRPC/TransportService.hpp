#pragma once

#include "Transport.grpc.pb.h"

#include <functional>

namespace transport {

    struct Req {
        std::string_view path;
    };

    struct Ack {
        std::string_view request_id;
    };

    struct PageReq {
        std::string_view request_id;
        int32_t page_num;
    };

    struct PageRep {
        std::unique_ptr<std::string> page_bytes;
        bool success;
    };

    class TransportServiceImpl final : public TransportService::Service {
    public:

        TransportServiceImpl(
            std::function<Ack(Req)> dataRequestSignaler,
            std::function<PageRep(PageReq)> pageFetcher);

        ::grpc::Status DataRequest(::grpc::ServerContext* context, 
            const ::Req* request, ::Ack* response) override;

        ::grpc::Status PageRequest(::grpc::ServerContext* context,
            const ::PageReq* request, ::PageRep* response) override;

    private:

        std::function<Ack(Req)> m_DataRequestSignaler;
        std::function<PageRep(PageReq)> m_PageFetcher;

    };


}


