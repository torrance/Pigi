#pragma once

#include <stdexcept>

#include <fmt/format.h>
#include <boost/archive/basic_archive.hpp>
#include <boost/mpi.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <thrust/complex.h>

#include "config.h"
#include "coordinates.h"
#include "clean.h"
#include "gridspec.h"
#include "memory.h"
#include "mset.h"
#include "outputtypes.h"

BOOST_IS_BITWISE_SERIALIZABLE(GridConfig);
BOOST_IS_BITWISE_SERIALIZABLE(GridSpec);
BOOST_IS_BITWISE_SERIALIZABLE(RaDec);
BOOST_IS_BITWISE_SERIALIZABLE(Config::Field);
BOOST_IS_BITWISE_SERIALIZABLE(clean::LMpx);
BOOST_IS_BITWISE_SERIALIZABLE(MeasurementSet::FreqRange);

namespace boost {
    namespace serialization {
        template <typename Archive>
        void serialize(Archive& ar, Config& payload, const unsigned) {
            ar & payload.precision;
            ar & payload.chanlow;
            ar & payload.chanhigh;
            ar & payload.channelsOut;
            ar & payload.maxDuration;
            ar & payload.msets;
            ar & payload.weight;
            ar & payload.robust;
            ar & payload.scale;
            ar & payload.phasecenter;
            ar & payload.fields;
            ar & payload.kernelsize;
            ar & payload.paddingfactor;
            ar & payload.wstep;
            ar & payload.majorgain;
            ar & payload.minorgain;
            ar & payload.cleanThreshold;
            ar & payload.autoThreshold;
            ar & payload.nMajor;
            ar & payload.nMinor;
        }

        template <typename Archive, typename T, int N>
        void serialize(Archive& ar, HostArray<T, N>& payload, const unsigned) {
            auto dims = payload.shape();
            ar & dims;
            if (dims != payload.shape()) {
                payload = HostArray<T, N>(dims);
            }

            for (size_t i {}; i < payload.size(); ++i) {
                ar & payload.data()[i];
            }
        }

        template <typename Archive, typename T>
        void serialize(Archive& ar, thrust::complex<T>& payload, const unsigned) {
            T real = payload.real();
            T imag = payload.imag();
            ar & real;
            ar & imag;
            payload.real(real);
            payload.imag(imag);
        }

        template <typename Archive, typename T>
        void serialize(Archive& ar, StokesI<T>& payload, const unsigned) {
            ar & payload.I;
        }

        template <typename Archive, typename T>
        void serialize(Archive& ar, std::optional<T>& payload, const unsigned) {
            bool hasvalue = payload.has_value();
            ar & hasvalue;
            if (hasvalue) {
                T val = payload.value_or(T());
                ar & val;
                payload = val;
            } else {
                payload.reset();
            }
        }
    }
}

namespace mpi {
    class Lock {
    public:
        Lock(boost::mpi::communicator& comm)
            : comm(comm), rank(comm.rank()), size(comm.size()) {

            if (rank == 0) {
                // No lock; Rank 0 always defaults to obtaining the lock first
                return;
            }

            // Aquire lock
            comm.recv(rank - 1, 0);
        }

        ~Lock() {
            // Release the lock by sending on to the next rank, unless we are the last rank
            if (rank + 1 < size) {
                comm.send(rank + 1, 0);
            }

            // Wait for all to finish
            comm.barrier();
        }

    private:
        boost::mpi::communicator& comm;
        int rank;
        int size;
    };
}