#pragma once

#include <mutex>

#include <fmt/color.h>
#include <fmt/format.h>

class Logger {
public:
    enum class Level {none, error, warning, info, debug, verbose};

    static void setLevel(Level level) {
        std::lock_guard l(writelock);
        maxlevel = level;
    }

    template <typename... Args>
    static void setName(const fmt::format_string<Args...>& name, Args&&... args) {
        threadname = fmt::format(name, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void log(Level level, const fmt::format_string<Args...>& msg, Args&&... args) {
        fmt::color color;
        switch (level) {
        case Level::error:
            color = fmt::color::red;
            break;
        case Level::warning:
            color = fmt::color::orange;
            break;
        case Level::info:
            color = fmt::color::lime_green;
            break;
        case Level::debug:
        case Level::verbose:
        case Level::none:
            color = fmt::color::slate_gray;
            break;
        }

        std::lock_guard l(writelock);
        if (static_cast<int>(level) <= static_cast<int>(maxlevel)) {
            fmt::print(fmt::fg(color), "[{}] ", threadname);
            fmt::println(msg, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    static void error(const fmt::format_string<Args...>& msg, Args&&... args) {
        log(Level::error, msg, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void warning(const fmt::format_string<Args...>& msg, Args&&... args) {
        log(Level::warning, msg, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void info(const fmt::format_string<Args...>& msg, Args&&... args) {
        log(Level::info, msg, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void debug(const fmt::format_string<Args...>& msg, Args&&... args) {
        log(Level::debug, msg, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void verbose(const fmt::format_string<Args...>& msg, Args&&... args) {
        log(Level::verbose, msg, std::forward<Args>(args)...);
    }

private:
    static inline std::mutex writelock {};
    thread_local static inline std::string threadname {"main"};
    static inline Level maxlevel {Level::info};
};