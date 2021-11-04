#pragma once

#include "../pch.hpp"

#include <atomic>
#include <queue>
#include <thread>
#include <mutex>
#include <type_traits>
#include <utility>
#include <future>


namespace transport {

    class ThreadPool {
    public:

        ThreadPool(const ui32& thread_count = std::thread::hardware_concurrency());

        ~ThreadPool();

        ui32 get_tasks_queued() const;

        ui32 get_tasks_running() const;

        ui32 get_tasks_total() const;

        ui32 get_thread_count() const;

        template<typename F>
        void push_task(const F& task);

        template<typename F, typename... A>
        void push_task(const F& task, const A &...args);

        void reset(const ui32& thread_count = std::thread::hardware_concurrency());

        template<typename F, typename... A, typename = 
            std::enable_if<std::is_void_v<std::invoke_result_t<std::decay_t<F>, std::decay_t<A>...>>>>
        std::future<bool> submit(const F& task, const A& ...args);

        template<typename F, typename... A, typename R = 
            std::invoke_result_t<std::decay_t<F>, std::decay_t<A>...>, typename = std::enable_if_t<!std::is_void_v<R>>>
        std::future<R> submit(const F& task, const A& ...args);

        void wait_for_tasks();

    private:

        void create_threads();

        void destroy_threads();

        bool pop_task(std::function<void()>& task);

        void sleep_or_yield();

        void worker();

    private:

        mutable std::mutex m_QueueMutex;

        std::atomic<bool> m_Running = true;
        std::atomic<bool> m_Paused = false;
        std::atomic<ui32> m_SleepDuration = 1000;

        std::queue<std::function<void()>> m_Tasks;

        ui32 m_ThreadCount;

        std::unique_ptr<std::thread[]> m_Threads;

        std::atomic<ui32> m_TasksTotal = 0;


    };


    template<typename F>
    inline void ThreadPool::push_task(const F& task) {
        m_TasksTotal++;
        {
            const std::scoped_lock lock(m_QueueMutex);
            m_Tasks.push(std::function<void()>(task));
        }
    }

    template<typename F, typename... A>
    inline void ThreadPool::push_task(const F& task, const A &...args) {
        push_task([task, args...]{
            task(args...);
        });
    }

    template<typename F, typename... A, typename = std::enable_if_t<std::is_void_v<std::invoke_result_t<std::decay_t<F>, std::decay_t<A>...>>>>
    inline std::future<bool> ThreadPool::submit(const F& task, const A& ...args)
    {
        std::shared_ptr<std::promise<bool>> task_promise = std::make_shared<std::promise<bool>>();
        std::future<bool> future = task_promise->get_future();
        push_task([task, args..., task_promise]{
            try {
                task(args...);
                task_promise->set_value(true);
            } catch(...) {
                try {
                    task_promise->set_exception(std::current_exception());
                } catch(...) {}
            }
        });
        return future;
    }

    template <typename F, typename... A, typename R = std::invoke_result_t<std::decay_t<F>, std::decay_t<A>...>, typename = std::enable_if_t<!std::is_void_v<R>>>
    inline std::future<R> submit(const F &task, const A &...args)
    {
        std::shared_ptr<std::promise<R>> task_promise(new std::promise<R>);
        std::future<R> future = task_promise->get_future();
        push_task([task, args..., task_promise] {
            try {
                task_promise->set_value(task(args...));
            } catch (...) {
                try {
                    task_promise->set_exception(std::current_exception());
                } catch (...) {}
            }
        });
        return future;
    }

}

