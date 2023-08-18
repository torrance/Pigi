#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

template<typename T>
class Channel {
public:
    enum class State { closed, open };

    Channel(size_t buffersize = 0) : buffersize(buffersize) {};
    Channel(const Channel<T>&) = delete;
    Channel& operator=(const Channel<T>&) = delete;

    ~Channel() {}

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            state = State::closed;
        }
        cv_popped.notify_one();
        cv_pushed.notify_one();
    }

    std::optional<T> pop() {
        T tmp;
        // Release the mutex before notifying
        {
            std::unique_lock lock(mutex);
            cv_pushed.wait(lock, [this] {
                return closed() || !empty();
            });

            if (empty()) {
                return {};
            }

            tmp = std::move(queue.front());
            queue.pop();
        }
        cv_popped.notify_one();
        return tmp;
    }

    bool push(const T& item) {
        // Release the mutex before notifying
        {
            std::unique_lock lock(mutex);
            cv_popped.wait(lock, [this] {
                return closed() || !full();
            });

            if (closed()) {
                return false;
            }
            queue.push(item);
        }
        cv_pushed.notify_one();
        return true;
    }

    bool push(T&& item) {
        // Release the mutex before notifying
        {
            std::unique_lock lock(mutex);
            cv_popped.wait(lock, [this] {
                return closed() || !full();
            });

            if (closed()) {
                return false;
            }
            queue.push(std::move(item));
        }
        cv_pushed.notify_one();
        return true;
    }

private:
    size_t buffersize {};
    std::queue<T> queue;
    State state { State::open };
    mutable std::mutex mutex;
    mutable std::condition_variable cv_popped;
    mutable std::condition_variable cv_pushed;

    bool empty() const {
        return queue.empty();
    }

    bool full() const {
        return buffersize > 0 && queue.size() >= buffersize;
    }

    bool closed() const {
        return state == State::closed;
    }
};