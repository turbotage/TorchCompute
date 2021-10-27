#pragma once

#include "Tensor.grpc.pb.h"

class TensorSenderImpl final : public TensorSender::Service {
public:
    

    ::grpc::Status SendTensor(::grpc::ServerContext* context, const ::TensorRequest* request, ::TensorReply* response) override;
};

