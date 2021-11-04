#include "TransportService.hpp"



transport::TransportServiceImpl::TransportServiceImpl(DataHandler dataHandler)
    : m_DataHandler(dataHandler)
{

}

::grpc::Status transport::TransportServiceImpl::TransportRPC(
    ::grpc::ServerContext* context, const ::Data* request, ::Data* response)
{
    m_DataHandler(request, response);
    return ::grpc::Status::OK;
}