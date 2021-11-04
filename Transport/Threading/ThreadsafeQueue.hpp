#pragma once

namespace transport {

    template<typename T>
    class ThreadsafeQueue {
    public:

        ThreadsafeQueue() = default;
        ThreadsafeQueue(const ThreadsafeQueue<T>&) = delete;
        ThreadsafeQueue& operator=(const ThreadsafeQueue<T>&) = delete;

        ThreadsafeQueue(ThreadsafeQueue<T>&& other) {
            std::lock_guard<std::mutex> lock(m_Mutex);
        }

        virtual ~ThreadsafeQueue() {}

        unsigned long size() const {
            std::lock_guard<std::mutex> lock(m_Mutex);
            return m_Queue.size();
        }

        std::optional<T> pop() {
            std::lock_guard<std::mutex> lock(m_Mutex);
            if (m_Queue.empty()) {
                return {};
            }
            T tmp = m_Queue.front();
            m_Queue.pop();
            return tmp;
        }

        void push(const T& item) {
            std::lock_guard<std::mutex> lock(m_Mutex);
            m_Queue.push(item);
        }

    private:
        std::queue<T> m_Queue;
        mutable std::mutex m_Mutex;

    };

}

