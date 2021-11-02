#include "Transport.hpp"

#include "gRPC/TensorSender.hpp"

#include "compute.hpp"

void RunServer(std::function<std::string(std::string)> tensorFetcher) {
    std::string server_adress("0.0.0.0:50051");

    transport::TensorSenderImpl tsService(tensorFetcher);
}



int main() {

    std::unordered_map<std::string, torch::Tensor> tensorMap;

    std::function<std::string(std::string)> tensorFetcher = [&tensorMap](std::string tensor_name) {
        return "Hello";
    };

    torch::Tensor a = torch::rand({100,3});
    tensorMap.insert({"T1", a});

    RunServer(tensorFetcher);

    return 0;
}