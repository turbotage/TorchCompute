#include "TensorSender.hpp"

transport::TensorSenderImpl::TensorSenderImpl(std::function<std::string(std::string)> tensorStringFetcher) 
    : m_TensorStringFetcher(tensorStringFetcher)
{
}

::grpc::Status transport::TensorSenderImpl::SendTensor(::grpc::ServerContext* context, 
    const ::TensorRequest* request, ::grpc::ServerWriter<::TensorReply>* writer)
{
    std::string tensor_name = request->tensor_name();
    std::string tensor;
    try {
        tensor = m_TensorStringFetcher(tensor_name);
    }
    catch (std::runtime_error e) {
        std::cout << e.what() << std::endl;
        return ::grpc::Status::CANCELLED;
    }
    
    uint64_t packetSize = 10; // 10 kChar
    uint64_t nPackets = std::ceil((double)tensor.length() / (double)packetSize);

    for (int i = 0; i < nPackets; ++i) {
        uint64_t startIdx = i*packetSize;
        uint64_t endIdx = (i+1)*packetSize;
        if (endIdx > tensor.length())
            endIdx = tensor.length();

        TensorReply reply;
        reply.set_tensor_str(tensor.substr(startIdx,endIdx));

        writer->Write(reply);
    }

    return ::grpc::Status::OK;
}
