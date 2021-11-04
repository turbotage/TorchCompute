#include "Transport.hpp"



#include "gRPC/TransportService.hpp"
#include "compute.hpp"


class Server {
public:

    

private:

    //std::unordered_map<std::string, std::queue<PageRep>> m_Pages;

};

void RunServer(std::function<std::string(std::string)> tensorFetcher) {
    std::string server_adress("0.0.0.0:50051");

    //transport::TransportServiceImpl tsService(tensorFetcher);

    ::grpc::ServerBuilder builder;
    builder.AddListeningPort(server_adress, grpc::InsecureServerCredentials());
    //builder.RegisterService(&tsService);

    std::unique_ptr<::grpc::Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_adress << std::endl;
    server->Wait();
}



int main() {

    std::unordered_map<std::string, torch::Tensor> tensorMap;


    torch::Tensor a = torch::rand({3,3});
    tensorMap.insert({"T1", a});


    return 0;
}