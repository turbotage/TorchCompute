#include "DataManager.hpp"


transport::DataManager::DataManager() 
{

}

void transport::DataManager::AddData(std::string path, std::shared_ptr<transport::Data> pData)
{
    std::lock_guard<std::mutex> lock(m_DataMutex);
    m_Data.insert_or_assign(path, pData);
}

std::shared_ptr<transport::Data> transport::DataManager::GetData(std::string path)
{
    std::lock_guard<std::mutex> lock(m_DataMutex);
    auto search = m_Data.find(path);
    if (search != m_Data.end())
        return search->second;
    else
        return nullptr;
}

void transport::DataManager::HandleDataTransport(::Data* fromClient, ::Data* toClient)
{
    if (fromClient->data_type() == REQUEST_DATA) {
        DataRequested(fromClient, toClient);
    }
    else if (fromClient->data_type() == PUSH_DATA) {
        DataPushed(fromClient, toClient);
    }


}

void transport::DataManager::DataRequested(::Data* fromClient, ::Data* toClient)
{
    const std::string& id = fromClient->id();
    const std::string& path = fromClient->data_path();

    std::lock_guard<std::mutex> lock(m_RequestMutex);

    bool start_serializing = false;
    auto id_search = m_RequestMap.find(fromClient->id());
    // This client has already done some requests
    if (id_search == m_RequestMap.end()) {
        start_serializing = true;
        
        m_RequestMap.emplace()
    }

}

void transport::DataManager::DataPushed(::Data* fromClient, ::Data* toClient)
{

}
