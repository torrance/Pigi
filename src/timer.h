#pragma once

#include <atomic>
#include <chrono>
#include <string>
#include <unordered_map>

#include <fmt/format.h>

#include "hip.h"

#define PIGI_TIMER(label, fn) { auto timer = Timer::get(label); fn; }

#ifdef PIGI_TIMER_ENABLE

class Timer {
public:
    using Counter = std::atomic<long long>;

    class StopWatch {
    public:
        StopWatch() = default;
        StopWatch(Counter* counter) : counter(counter) {
            HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
            start = std::chrono::steady_clock::now();
        }

        ~StopWatch() {
            if (counter == nullptr) return;

            HIPCHECK( hipStreamSynchronize(hipStreamPerThread) );
            *counter += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start
            ).count();
        }

        // Delete all copy constructors
        StopWatch(const StopWatch&) = delete;
        StopWatch& operator=(const StopWatch&) = delete;

        StopWatch(StopWatch&& other) {
            (*this) = std::move(other);
        }

        StopWatch& operator=(StopWatch&& other) {
            using std::swap;
            std::swap(counter, other.counter);
            std::swap(start, other.start);
            return *this;
        }

    private:
        Counter* counter {nullptr};
        std::chrono::time_point<std::chrono::steady_clock> start {};
    };

    // Timer is a singleton
    static StopWatch get(std::string label) {
        static Timer instance;
        return StopWatch(&instance.counters[label]);
    }

    // Delete all other copy/move constructors
    Timer(const Timer&) = delete;
    Timer(Timer&&) = delete;
    Timer& operator=(const Timer&) = delete;
    Timer& operator=(Timer&&) = delete;

    ~Timer() {
        if (counters.empty()) return;

        // Sort labels
        std::vector<std::string> labels;
        for (auto& [label, _] : counters) labels.push_back(label);
        std::sort(labels.begin(), labels.end());

        // At program exit, print times for any counters
        fmt::println("======= COUNTERS =======");
        for (auto& label : labels) {
            fmt::println(
                "{}: {:.6f} s",
                label, counters[label].load() / 1e9
            );
        }
        fmt::println("========================");
    }

private:
    Timer() = default;
    std::unordered_map<std::string, Counter> counters;
};

#else

class Timer {
public:
    class StopWatch {
    public:
        StopWatch() = default;
        ~StopWatch() {}  // non-default destructor stops unused warnigns
    };

    static StopWatch get(std::string) { return StopWatch(); }
};

#endif