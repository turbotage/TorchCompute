#include "TensorSender.hpp"

transport::grpc::TensorSenderImpl::TensorSenderImpl(std::function<std::string(std::string)> tensorStringFetcher) 
    : m_TensorStringFetcher(tensorStringFetcher)
{
}

::grpc::Status transport::grpc::TensorSenderImpl::SendTensor(
    ::grpc::ServerContext* context, 
    const ::TensorRequest* request, 
    ::TensorReply* response) 
{
    
    
}
