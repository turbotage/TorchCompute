#include "DataManager.hpp"


transport::DataManager::DataManager(std::shared_ptr<transport::ThreadPool> threadpool)
    : m_pThreadPool(threadpool)
{

}

void transport::DataManager::AddData(std::string id, std::string path, Data pData)
{
    std::lock_guard<std::mutex> lock(m_RequestMutex);
    auto id_it = m_RequestMap.try_emplace(id);
}

transport::Data& transport::DataManager::GetData(std::string id, std::string path)
{
    std::lock_guard<std::mutex> lock(m_RequestMutex);
    auto id_search = m_RequestMap.find(path);
    if (id_search != m_RequestMap.end()) {
        auto path_search = id_search->second.find(path);
        if (path_search != id_search->second.end()) {
            return path_search->second.data;
        }

    }
    throw std::runtime_error("Requested non-existing data");
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
    // Return ID and Path should always be the same as Request
    toClient->set_id(fromClient->id());
    toClient->set_data_path(fromClient->data_path());

    // Check the data exists
    std::lock_guard<std::mutex> request_lock(m_RequestMutex);
    auto id_search = m_RequestMap.find(path);
    if (id_search == m_RequestMap.end()) {
        toClient->set_data_type(eDataType::MESSAGE);
        toClient->set_meta("REQUESTED NON-EXISTING DATA");
        return;
    }
    auto path_search = id_search->second.find(path);
    if (path_search == id_search->second.end()) {
        toClient->set_data_type(eDataType::MESSAGE);
        toClient->set_meta("REQUESTED NON-EXISTING DATA");
        return;
    }
    
    DataTup& data_tup = path_search->second;

    if (data_tup.data_state == eDataState::DEFAULT) {
        // We should begin serializing
        data_tup.data_state = eDataState::SERIALIZING;

        std::function<void()> serializer;
        serializer = [&data_tup, &serializer, this]() {
            ui32 npages;
            std::lock_guard<std::mutex> lock(this->m_RequestMutex);
            npages = data_tup.pages.size();

            if (npages > 3) {
                m_pThreadPool->push_task(serializer);
                return;
            }
            else {

                // Get the next page
                std::string page = std::move(data_tup.data.NextPage());
                ui32 page_size = page.size();

                // Check if this was the last page in which case we are done serializing
                if (page_size == 0) {
                    data_tup.data_state = eDataState::SERIALIZED;
                    return;
                }
                
                //Push data page to queue
                data_tup.pages.push(page);

                // If the page_size is smaller than the default page size it is probable that this was the last page
                // in which case we would like to finalize serializing now
                if (page_size < transport::DEFAULT_PAGE_SIZE) {
                    page = std::move(data_tup.data.NextPage());
                    page_size = page.size();
                    if (page_size == 0) {
                        data_tup.data_state = eDataState::SERIALIZED;
                        return;
                    }
                    data_tup.pages.push(page);
                }

                m_pThreadPool->push_task(serializer);
            }
            
            
        };

        m_pThreadPool->push_task(serializer);

        return;
    }
    else if ((data_tup.data_state == eDataState::SERIALIZING) || (data_tup.data_state == eDataState::SERIALIZED) ) {
        if (data_tup.pages.size() > 0) {
            toClient->set_data(data_tup.pages.front());
            data_tup.pages.pop();
            toClient->set_data_type(eDataType::RESPONSE_DATA);

            if ((data_tup.pages.size() == 1) && (data_tup.data_state == eDataState::SERIALIZED)) {
                toClient->set_meta("END");
                data_tup.data_state = eDataState::DEFAULT;
            }
            return;
        }
    }
    
    toClient->set_data_type(eDataType::NO_DATA);
}

void transport::DataManager::DataPushed(::Data* fromClient, ::Data* toClient)
{

}
