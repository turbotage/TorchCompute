#include "ThreadPool.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>


transport::ThreadPool::ThreadPool(const ui32& thread_count) 
    : m_ThreadCount(thread_count ? thread_count : std::thread::hardware_concurrency()), 
    m_Threads(std::make_unique<std::thread[]>(m_ThreadCount))
{
    create_threads();
}

transport::ThreadPool::~ThreadPool()
{
    wait_for_tasks();
    m_Running = false;
    destroy_threads();
}

ui32 transport::ThreadPool::get_tasks_queued() const
{
    const std::scoped_lock lock(m_QueueMutex);
    return m_Tasks.size();
}

ui32 transport::ThreadPool::get_tasks_running() const
{
    return m_TasksTotal - get_tasks_queued();
}

ui32 transport::ThreadPool::get_tasks_total() const
{
    return m_TasksTotal;
}

ui32 transport::ThreadPool::get_thread_count() const
{
    return m_ThreadCount;
}

void transport::ThreadPool::reset(const ui32& thread_count)
{
    bool was_paused = m_Paused;
    m_Paused = true;
    wait_for_tasks();
    m_Running = false;
    destroy_threads();
    m_ThreadCount = thread_count ? thread_count : std::thread::hardware_concurrency();
    m_Threads = std::make_unique<std::thread[]>(m_ThreadCount);
    m_Paused = was_paused;
    m_Running = true;
    create_threads();
}

void transport::ThreadPool::wait_for_tasks()
{
    while (true)
    {
        if (!m_Paused) {
            if (m_TasksTotal == 0)
                break;
        } else {
            if (get_tasks_running() == 0)
                break;
        }
        sleep_or_yield();
    }
}























void transport::ThreadPool::create_threads()
{
    for (ui32 i = 0; i < m_ThreadCount; ++i) {
        m_Threads[i] = std::thread(&ThreadPool::worker, this);
    }
}

void transport::ThreadPool::destroy_threads()
{
    for (ui32 i = 0; i < m_ThreadCount; ++i) {
        m_Threads[i].join();
    }
}

bool transport::ThreadPool::pop_task(std::function<void()>& task) 
{
    const std::scoped_lock lock(m_QueueMutex);
    if (m_Tasks.empty()) {
        return false;
    } else {
        task = std::move(m_Tasks.front());
        m_Tasks.pop();
        return true;
    }
}

void transport::ThreadPool::sleep_or_yield()
{
    if (m_SleepDuration) {
        std::this_thread::sleep_for(std::chrono::microseconds(m_SleepDuration));
    } else {
        std::this_thread::yield();
    }
}

void transport::ThreadPool::worker() {
    while (m_Running)
    {
        std::function<void()> task;
        if (!m_Paused && pop_task(task)) {
            task();
            m_TasksTotal--;
        }
        else {
            sleep_or_yield();
        }
    }
    
}
