#include "TransportService.hpp"

transport::TransportServiceImpl::TransportServiceImpl(
    std::function<transport::Ack(transport::Req)> dataRequestSignaler,
    std::function<transport::PageRep(transport::PageReq)> pageFetcher) 
    : m_DataRequestSignaler(dataRequestSignaler), m_PageFetcher(pageFetcher)
{

}


/*
::grpc::Status transport::TensorSenderImpl::DataRequest(::grpc::ServerContext* context, 
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

    std::cout << "\ntensor_str:\n";
    for (int i = 0; i < nPackets; ++i) {
        uint64_t startIdx = i*packetSize;
        uint64_t endIdx = (i+1)*packetSize;
        if (endIdx > tensor.length())
            endIdx = tensor.length();

        TensorReply reply;
        std::string substr = tensor.substr(startIdx,endIdx);
        std::cout << substr;
        reply.set_tensor_str(substr);

        writer->Write(reply);
    }
    std::cout << std::endl;

    return ::grpc::Status::OK;
}
*/
