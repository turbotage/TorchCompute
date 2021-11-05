#include "TransportService.hpp"



transport::TransportServiceImpl::TransportServiceImpl(DataHandler dataHandler)
    : m_DataHandler(dataHandler)
{

}

::grpc::Status transport::TransportServiceImpl::TransportRPC(
    ::grpc::ServerContext* context, const ::Data* fromClient, ::Data* toClient)
{
    m_DataHandler(fromClient, toClient);
    return ::grpc::Status::OK;
}