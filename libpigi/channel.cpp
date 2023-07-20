#include <condition_variable>
#include <mutex>
#include <optional>
#include <iostream>
#include <queue>

template<typename T>
class Channel {
public:
    enum class State { closed, open };

    Channel() = default;
    Channel(const Channel<T>&) = delete;
    Channel& operator=(const Channel<T>&) = delete;

    ~Channel() {}

    void close() {
        std::lock_guard<std::mutex> lock(mutex);
        state = State::closed;
    }

    std::optional<T> pop() {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return closed() || !empty(); });

        if (empty()) {
            return {};
        }

        T tmp = std::move(queue.front());
        queue.pop();
        return tmp;
    }

    bool push(const T& item) {
        // Release the mutex before notifying
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (closed()) {
                return false;
            }
            queue.push(item);
        }
        cv.notify_one();
        return true;
    }

    bool push(T&& item) {
        // Release the mutex before notifying
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (closed()) {
                return false;
            }
            queue.push(std::move(item));
        }
        cv.notify_one();
        return true;
    }

private:
    std::queue<T> queue;
    State state { State::open };
    mutable std::mutex mutex;
    mutable std::condition_variable cv;

    bool empty() const {
        return queue.empty();
    }

    bool closed() const {
        return state == State::closed;
    }
};