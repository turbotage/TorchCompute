#include "Transport.hpp"

#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include "gRPC/TransportService.hpp"
#include "compute.hpp"


class Server {
public:

    

private:

    std::unordered_map<std::string, std::queue<PageRep>> m_Pages;

};

void RunServer(std::function<std::string(std::string)> tensorFetcher) {
    std::string server_adress("0.0.0.0:50051");

    transport::TransportServiceImpl tsService(tensorFetcher);

    ::grpc::ServerBuilder builder;
    builder.AddListeningPort(server_adress, grpc::InsecureServerCredentials());
    builder.RegisterService(&tsService);

    std::unique_ptr<::grpc::Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_adress << std::endl;
    server->Wait();
}



int main() {

    std::unordered_map<std::string, torch::Tensor> tensorMap;

    std::function<std::string(std::string)> tensorFetcher = [&tensorMap](std::string tensor_name) {
        torch::Tensor tensor = tensorMap[tensor_name];

        std::stringstream stream;
        torch::save(tensor, stream);

        std::string tensor_str = stream.str();
        return tensor_str;
    };

    torch::Tensor a = torch::rand({3,3});
    tensorMap.insert({"T1", a});

    RunServer(tensorFetcher);

    return 0;
}