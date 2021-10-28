#pragma once

#include "Tensor.grpc.pb.h"

#include <functional>

namespace transport {

    namespace grpc {


        class TensorSenderImpl final : public TensorSender::Service {
        public:

            TensorSenderImpl(std::function<std::string(std::string)> tensorStringFetcher);

            ::grpc::Status SendTensor(::grpc::ServerContext* context, const ::TensorRequest* request, ::TensorReply* response) override;

        private:

            std::function<std::string(std::string)> m_TensorStringFetcher;

        };

    }

}


