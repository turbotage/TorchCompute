#pragma once

#include "Tensor.grpc.pb.h"

#include <functional>

namespace transport {


    class TensorSenderImpl final : public TensorSender::Service {
    public:

        TensorSenderImpl(std::function<std::string(std::string)> tensorStringFetcher);

        ::grpc::Status SendTensor(::grpc::ServerContext* context, 
            const TensorRequest* request, ::grpc::ServerWriter<TensorReply>* writer) override;

    private:

        std::function<std::string(std::string)> m_TensorStringFetcher;

    };


}


