#pragma once

#include <stdexcept>

#include <fmt/format.h>
#include <mpi.h>

namespace MPI {
    enum class Tag { Lock, Release };

    // Create this as a singleton
    class MPI {
    public:
        static MPI& getInstance() {
            static MPI instance;
            return instance;
        }

        MPI(const MPI&) = delete;
        MPI& operator=(const MPI&) = delete;

        ~MPI() {
            int err = MPI_Finalize();

            // If an error occurred, we don't throw an exception since we are in a
            // destructor; but make a note of it at least.
            if (err != MPI_SUCCESS) {
                int length {};
                char msg[MPI_MAX_ERROR_STRING] {};
                MPI_Error_string(err, msg, &length);
                fmt::println(
                    stderr, "An error occurred calling MPI_Finalize(): {}", msg
                );
            }
        }

    private:
        MPI() {
            int provided;
            MPI_Init_thread(0, NULL, MPI_THREAD_FUNNELED, &provided);
            if (provided != MPI_THREAD_FUNNELED) {
                throw std::runtime_error(
                    "Failed to initialize MPI with MPI_THREAD_FUNNELED support"
                );
            }
        }
    };

    const bool init = [] {
        MPI::getInstance();
        return true;
    }();

    int getsize(MPI_Comm comm = MPI_COMM_WORLD) {
        int size;
        MPI_Comm_size(comm, &size);
        return size;
    }

    int getrank(MPI_Comm comm = MPI_COMM_WORLD) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        return rank;
    }

    void barrier(MPI_Comm comm = MPI_COMM_WORLD) {
        MPI_Barrier(comm);
    }

    template <typename T>
    void send(const T&, MPI_Comm, int) {
        static_assert(
            static_cast<int>(sizeof(T)) == -1,
            "No MPI::send specialisation provided"
        );
    }

    template <typename T>
    T recv(MPI_Comm, int) {
        static_assert(
            static_cast<int>(sizeof(T)) == -1,
            "No MPI::recv specialisation provided"
        );
        return T();
    }

    template <>
    void send(const bool& payload, const MPI_Comm comm, const int rank) {
        MPI_Send(&payload, 1, MPI_CXX_BOOL, rank, 0, comm);
    }

    template <>
    bool recv(const MPI_Comm comm, const int rank) {
        bool payload;
        MPI_Recv(&payload, 1, MPI_CXX_BOOL, rank, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
        return payload;
    }

    template <>
    void send(const double& payload, const MPI_Comm comm, const int rank) {
        MPI_Send(&payload, 1, MPI_DOUBLE, rank, 0, comm);
    }

    template <>
    double recv(const MPI_Comm comm, const int rank) {
        double payload;
        MPI_Recv(&payload, 1, MPI_DOUBLE, rank, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
        return payload;
    }

    template <>
    void send(const GridSpec& payload, const MPI_Comm comm, const int rank) {
        int size = sizeof(long long) * 2 + sizeof(double) * 2;
        char buffer[size];
        int position {};

        MPI_Pack(&payload.Nx, 1, MPI_LONG_LONG, buffer, size, &position, comm);
        MPI_Pack(&payload.Ny, 1, MPI_LONG_LONG, buffer, size, &position, comm);
        MPI_Pack(&payload.scalelm, 1, MPI_DOUBLE, buffer, size, &position, comm);
        MPI_Pack(&payload.scaleuv, 1, MPI_DOUBLE, buffer, size, &position, comm);
        MPI_Send(buffer, position, MPI_PACKED, rank, 0, comm);
    }

    template <>
    GridSpec recv(const MPI_Comm comm, const int rank) {
        int size = sizeof(long long) * 2 + sizeof(double) * 2;
        char buffer[size];
        MPI_Recv(buffer, size, MPI_PACKED, rank, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);

        GridSpec payload;
        int position {};

        MPI_Unpack(buffer, size, &position, &payload.Nx, 1, MPI_LONG_LONG, comm);
        MPI_Unpack(buffer, size, &position, &payload.Ny, 1, MPI_LONG_LONG, comm);
        MPI_Unpack(buffer, size, &position, &payload.scalelm, 1, MPI_DOUBLE, comm);
        MPI_Unpack(buffer, size, &position, &payload.scaleuv, 1, MPI_DOUBLE, comm);

        return payload;
    }

    template <typename T, int N>
    void send(const HostArray<T, N>& payload, const MPI_Comm comm, const int rank) {
        auto dims = payload.shape();
        MPI_Send(dims.data(), N, MPI_LONG_LONG, rank, 0, comm);

        // To avoid issues with exceeding the maximum int value, we send as contiguous rows
        MPI_Datatype row;
        MPI_Type_contiguous(payload.size(0) * sizeof(T), MPI_CHAR, &row);
        MPI_Type_commit(&row);

        long long Nrows = payload.size() / payload.size(0);
        MPI_Send(payload.data(), Nrows, row, rank, 0, comm);

        MPI_Type_free(&row);
    }

    template <typename T, int N>
    HostArray<T, N> recv(const MPI_Comm comm, const int rank) {
        std::array<long long, N> dims {};
        MPI_Recv(dims.data(), N, MPI_LONG_LONG, rank, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);

        HostArray< T, N> payload(dims);

        // To avoid issues with exceeding the maximum int value, we send as contiguous rows
        MPI_Datatype row;
        MPI_Type_contiguous(payload.size(0) * sizeof(T), MPI_CHAR, &row);
        MPI_Type_commit(&row);

        long long Nrows = payload.size() / payload.size(0);
        MPI_Recv(payload.data(), Nrows, row, rank, MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);

        MPI_Type_free(&row);

        return payload;
    }

    class Lock {
    public:
        Lock(MPI_Comm comm = MPI_COMM_WORLD) : comm(comm) {
            size = getsize(comm);
            rank = getrank(comm);

            if (rank == 0) {
                // No lock; Rank 0 always defaults to obtaining the lock first
                fmt::println("Rank {} acquired lock", rank);
                return;
            }

            // Aquire lock
            MPI_Recv(
                NULL, 0, MPI_INT,  rank - 1, static_cast<int>(Tag::Lock),
                comm, MPI_STATUS_IGNORE
            );
            fmt::println("Rank {} acquired lock", rank);
        }

        ~Lock() {
            fmt::println("Rank {} releasing lock", rank);

            // Release the lock by sending on to the next rank, unless we are the last rank
            if (rank + 1 < size) {
                MPI_Send(
                    NULL, 0, MPI_INT, (rank + 1) % size, static_cast<int>(Tag::Lock), comm
                );
            }

            // Wait for all to finish
            MPI_Barrier(comm);
        }

    private:
        MPI_Comm comm;
        int rank {};
        int size {};
    };
}