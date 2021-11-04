#include "Server.hpp"
#include "gRPC/TransportService.hpp"

#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>


transport::Server::Server()
{


    m_DataHandler = [](const ::Data* request, ::Data* response) {
        
    };

}


void transport::Server::RunServer() {
    std::string server_adress("0.0.0.0:50051");
    TransportServiceImpl service(m_DataHandler);

    ::grpc::ServerBuilder builder;

    builder.AddListeningPort(server_adress, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    m_pServer = builder.BuildAndStart();

    m_pServer->Wait();
}