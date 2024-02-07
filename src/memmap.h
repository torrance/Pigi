#pragma once

#include <errno.h>     // errno
#include <fcntl.h>     // open
#include <filesystem>  // temp_directory_path
#include <mutex>       // std::mutex
#include <stdexcept>   // runtime_error
#include <string.h>    // strerror
#include <sys/mman.h>  // mmap
#include <type_traits> // true_type
#include <unistd.h>    // sysconf, ftruncate

#include <fmt/format.h>

#include "util.h"

template <typename T, int N = 0>
class MMapAllocator {
public:
    using value_type = T;

    MMapAllocator() noexcept = default;

    template <typename U>
    MMapAllocator(const MMapAllocator<U, N>&) noexcept {}

    bool operator==(MMapAllocator<T, N>&) { return true; }
    bool operator!=(MMapAllocator<T, N>&) { return false; }

    template <typename U>
    struct rebind { using other = MMapAllocator<U, N>; };

    T* allocate(size_t n) {
        // Convert n from units of T to bytes
        n = sizeof(T) * n;

        std::lock_guard lock(m);

        // Increase count to the next multiple of the system page size
        n = cld<size_t>(n, pagesz) * pagesz;

        // Increase the tmp file size and record new file size
        const size_t offset {fsize};
        fsize += n;
        int err = ftruncate(fd, fsize);

        if (err == -1) {
            throw std::runtime_error(fmt::format(
                "Failed to increase size of tmp file used for mmap in directory {}. Error: {}",
                std::filesystem::temp_directory_path().string(), strerror(errno)
            ));
        }

        // Create mmap pointer, offset by the file size prior to the call to ftruncate
        void* ptr =  mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
        if (ptr == (void*) -1) {
            throw std::runtime_error(fmt::format(
                "An error occurred calling mmap(): {}", strerror(errno)
            ));
        }

        return reinterpret_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_t n) {
        // Convert n from units of T to bytes
        n = sizeof(T) * n;

        std::lock_guard lock(m);

        // Increase count to the next multiple of the system page size
        n = cld<size_t>(n, pagesz) * pagesz;
        freed += n;

        int err = munmap(reinterpret_cast<void*>(ptr), n);
        if (err == -1) {
            std::runtime_error(fmt::format(
                "An error occurred unmapping pointer. Error: {}", strerror(errno)
            ));
        }

        if (fsize == freed) {
            int err = ftruncate(fd, 0);
            if (err == -1) {
                std::runtime_error(fmt::format(
                    "Failed to truncate tmp file used for mmap in directory {}. Error: {}",
                    std::filesystem::temp_directory_path().string(), strerror(errno)
                ));
            }

            // Reset size counters
            fsize = 0;
            freed = 0;
        }
    }

private:
    // Get the system's memory page size. This call is only valid on Linux
    static inline const size_t pagesz { static_cast<size_t>(sysconf(_SC_PAGESIZE)) };

    // Open backing file
    static inline const int fd = [] {
        int fd = open(
            std::filesystem::temp_directory_path().c_str(),
            O_TMPFILE | O_RDWR,
            S_IRUSR | S_IWUSR
        );
        if (fd == -1) {
            std::runtime_error(fmt::format(
                "Failed to create tmp file used for mmap in directory {}. Error: {}",
                std::filesystem::temp_directory_path().string(), strerror(errno)
            ));
        }
        return fd;
    }();

    // This mutex ensures all access to static state are serialized
    static inline std::mutex m;

    // Open file and track filesize
    static inline size_t fsize {};

    // Track total bytes freed
    static inline size_t freed {};
};