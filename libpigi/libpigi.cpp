#include <algorithm>
#include <complex>
#include <span>
#include <unordered_map>
#include <vector>
#include <ranges>

#include <fmt/format.h>
#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include "hip.cpp"
#include "span.cpp"

template <typename T>
struct fmt::formatter<std::complex<T>> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const std::complex<T>& value, FormatContext& ctx) {
        return fmt::format_to(
            ctx.out(), "{:.2f} + {:.2f}i", value.real(), value.imag()
        );
    }
};

template <typename T>
struct Matrix2x2 {
    // COLUMN MAJOR
    T xx {}, yx {}, xy {}, yy {};

    static int size() { return 4; }
};

template <typename T, typename S>
__host__ __device__
auto operator*(const Matrix2x2<T>& A, const S& c) {
    Matrix2x2< decltype(A.xx * c) > B {A.xx, A.yx, A.xy, A.yy};
    return B *= c;
}

template <typename T, typename S>
__host__ __device__
auto& operator*=(Matrix2x2<T>& A, const S& c) {
    A.xx *= c;
    A.yx *= c;
    A.xy *= c;
    A.yy *= c;
    return A;
}

template <typename T, typename S>
__host__ __device__
auto operator*(Matrix2x2<T>& A, Matrix2x2<S> B) {
    return Matrix2x2< decltype(A.xx * B.xx) > {
        A.xx * B.xx, A.yx * B.yx, A.xy * B.xy, A.yy * B.yy
    };
}

template <typename T, typename S>
__host__ __device__
Matrix2x2<T>& operator+=(Matrix2x2<T>& A, const Matrix2x2<S>& B) {
    A.xx += B.xx;
    A.yx += B.yx;
    A.xy += B.xy;
    A.yy += B.yy;
    return A;
}

template <typename T>
struct fmt::formatter<Matrix2x2<T>> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const Matrix2x2<T>& value, FormatContext& ctx) {
        return fmt::format_to(
            ctx.out(), "[{}, {}; {}, {}]", value.xx, value.xy, value.yx, value.yy
        );
    }
};

template <typename T>
struct LinearData : public Matrix2x2<std::complex<T>> {
    __host__ __device__
    LinearData() : Matrix2x2<std::complex<T>>({}) {}

    __host__ __device__
    LinearData(
        std::complex<T> xx, std::complex<T> yx,
        std::complex<T> xy, std::complex<T> yy
    ) : Matrix2x2<std::complex<T>>({xx, yx, xy, yy}) {}

    __host__ __device__
    LinearData(Matrix2x2<std::complex<T>> val) :
        Matrix2x2<std::complex<T>>({val.xx, val.yx, val.xy, val.yy}) {}
};

template <typename T>
struct fmt::formatter<LinearData<T>> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const LinearData<T>& value, FormatContext& ctx) {
        return fmt::format_to(
            ctx.out(), "[{}, {}; {}, {}]", value.xx, value.xy, value.yx, value.yy
        );
    }
};

template <typename T>
struct StokesI {
    std::complex<T> I;

    static int size() { return 1; }

    __host__ __device__ StokesI<T>& operator=(const LinearData<T> data) {
        I = (T) 0.5 * (data.xx + data.yy);
        return *this;
    }

    template<typename S>
    __host__ __device__
    StokesI<T>& operator*=(const S x) {
        I *= x;
        return *this;
    }

    __host__ __device__
    StokesI<T>& operator +=(const StokesI<T> x) {
        I += x.I;
        return *this;
    }

    template <typename S>
    __host__ __device__
    StokesI<T>& operator /=(const S x) {
        I /= x;
        return *this;
    }
};

struct GridSpec {
    long long Nx;
    long long Ny;
    double scalelm;
    double scaleuv;
};

template <typename T>
struct UVDatum {
    size_t row;
    size_t chan;
    T u;
    T v;
    T w;
    Matrix2x2<T> weights;
    Matrix2x2<std::complex<T>> data;
};

template <typename T>
struct WorkUnit {
    long long u0px;
    long long v0px;
    T u0;
    T v0;
    T w0;
    GridSpec subgridspec;
    SpanMatrix< Matrix2x2<std::complex<T>> > Aleft;
    SpanMatrix< Matrix2x2<std::complex<T>> > Aright;
    SpanVector< UVDatum<T> > data;
};

template <typename T>
struct UVWOrigin {
    T u0, v0, w0;

    UVWOrigin(T* ptr) : u0(ptr[0]), v0(ptr[1]), w0(ptr[2]) {}
    UVWOrigin(T u0, T v0, T w0) : u0(u0), v0(v0), w0(w0) {}
};

template <typename T>
__host__ __device__
inline T ndash(T l, T m) {
    auto r2 = l*l + m*m;
    return r2 > 1 ? 1 : r2 / (1 + sqrt(1 - r2));
}

template <typename T>
inline T* hipMalloc(size_t n) {
    T* ptr {};
    HIPCHECK( hipMalloc(&ptr, sizeof(T) * n) );
    return ptr;
}

template <typename T>
inline T* hipMemcpyHtoD(const T* const hostptr, size_t n) {
    T* deviceptr {};
    HIPCHECK( hipMalloc(&deviceptr, sizeof(T) * n) );
    HIPCHECK( hipMemcpyHtoD((void*) deviceptr, (void*) hostptr, sizeof(T) * n) );
    return deviceptr;
}

template <typename T>
auto getKernelConfig(T fn, int N, size_t sharedMem=0) {

    static int nblocksmax, nthreads;

    [[maybe_unused]] static auto _ = [&]() {
        fmt::println("Calculating kernel configuration...");
        HIPCHECK( hipOccupancyMaxPotentialBlockSize(
            &nblocksmax, &nthreads, fn, sharedMem, 0 
        ) );
        fmt::println("Recommended launch config: blocksmax={}, threads={}", nblocksmax, nthreads);
        return true;
    }();

    return std::make_tuple(
        std::min<int>(nblocksmax, N / nthreads + 1), nthreads
    );
};

template <typename T, typename S>
__global__
void gpudift(
    T * __restrict__ subgrid,
    const Matrix2x2<std::complex<S>> * __restrict__ Aleft,
    const Matrix2x2<std::complex<S>> * __restrict__ Aright,
    const UVWOrigin<S> origin,
    const UVDatum<S> * __restrict__ uvdata,
    const size_t uvdata_n,
    const GridSpec subgridspec
) {
    for (
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < subgridspec.Nx * subgridspec.Ny;
        idx += blockDim.x * gridDim.x
    ) {
        // WARN: column-major ordering
        long long lpx {idx % subgridspec.Nx};
        long long mpx {idx / subgridspec.Nx};

        S l {(lpx - subgridspec.Nx / 2) * (S) subgridspec.scalelm};
        S m {(mpx - subgridspec.Ny / 2) * (S) subgridspec.scalelm};

        LinearData<S> cell {};
        for (size_t n = 0; n < uvdata_n; ++n) {
            auto uvdatum = uvdata[n];
            double real, imag;
            sincospi(
                2 * ((uvdatum.u - origin.u0) * l + (uvdatum.v - origin.v0) * m + (uvdatum.w - origin.w0) * ndash(l, m)),
                &imag, &real
            );

            cell += uvdatum.data * uvdatum.weights * std::complex<S>(real, imag);
        }

        // TODO: add beam correction and normalize
        subgrid[idx] = cell;
    }
}

template <typename T, typename S>
__global__
void applytaper(
    T* __restrict__ subgrid, S* __restrict__ taper, GridSpec subgridspec
) {
    auto N = subgridspec.Nx * subgridspec.Ny;
    for (
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < N;
        idx += blockDim.x * gridDim.x
    ) {
        subgrid[idx] *= (taper[idx] / N);
    }
}

template <typename T>
__global__
void applyCheckerboard(T* grid, long long Nx, long long Ny) {
    for (
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < Nx * Ny;
        idx += blockDim.x * gridDim.x
    ) {
        // COLUMN MAJOR
        auto lpx {idx % Nx};
        auto  mpx {idx / Nx};

        auto lfactor {1 - 2 * (lpx % 2)};
        auto mfactor {1 - 2 * (mpx % 2)};

        grid[idx] *= (lfactor * mfactor);
    }
}


auto fftExec(hipfftHandle plan, LinearData<double>* grid, int direction) { 
    return hipfftExecZ2Z(
        plan, 
        (hipfftDoubleComplex*) grid,
        (hipfftDoubleComplex*) grid,
        direction
    );
}
auto fftExec(hipfftHandle plan, StokesI<float>* grid, int direction) { 
    return hipfftExecC2C(
        plan, 
        (hipfftComplex*) grid,
        (hipfftComplex*) grid,
        direction
    );
}
auto fftExec(hipfftHandle plan, StokesI<double>* grid, int direction) { 
    return hipfftExecZ2Z(
        plan, 
        (hipfftDoubleComplex*) grid,
        (hipfftDoubleComplex*) grid,
        direction
    );
}

auto fftType(LinearData<double>* grid) { return HIPFFT_Z2Z; }
auto fftType(StokesI<float>* grid) { return HIPFFT_C2C; }
auto fftType(StokesI<double>* grid) { return HIPFFT_Z2Z; }

template<typename T>
void fftshift(T* grid, GridSpec gridspec) {
    auto [nblocks, nthreads] = getKernelConfig(
        applyCheckerboard<T>, gridspec.Nx * gridspec.Ny
    );

    hipLaunchKernelGGL(
        applyCheckerboard<T>, nblocks, nthreads, 0, 0,
        grid, gridspec.Nx, gridspec.Ny
    );
}

/**
 * Add a subgrid back onto the larger master grid
 */
template <typename T>
__global__
void addsubgrid(
    T* __restrict__ grid, GridSpec gridspec,
    T* __restrict__ subgrid, GridSpec subgridspec,
    long long u0px, long long v0px
) {
    // Iterate over each element of the subgrid
    auto N = subgridspec.Nx * subgridspec.Ny;
    for (
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < N;
        idx += blockDim.x * gridDim.x
    ) {
        // COLUMN MAJOR
        long long upx = idx % subgridspec.Nx;
        long long vpx = idx / subgridspec.Nx;

        // Transform to pixel position wrt to master grid
        upx += u0px - subgridspec.Nx / 2;
        vpx += v0px - subgridspec.Ny / 2;

        if (
            0 <= upx && upx < gridspec.Nx &&
            0 <= vpx && vpx < gridspec.Ny
        ) {
            // COLUMN MAJOR
            grid[vpx * gridspec.Nx + upx] += subgrid[idx];
        }
    }
}

template <typename T, typename S>
__global__
void wcorrect(T* grid, GridSpec gridspec, S w0) {
    for (
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < gridspec.Nx * gridspec.Ny;
        idx += blockDim.x * gridDim.x
    ) {
        // COLUMN MAJOR
        long long lpx {idx % gridspec.Nx};
        long long mpx {idx / gridspec.Nx};

        auto l {static_cast<S>((lpx - gridspec.Nx / 2) * gridspec.scalelm)};
        auto m {static_cast<S>((mpx - gridspec.Ny / 2) * gridspec.scalelm)};

        S real, imag;
        sincospi(
            2 * w0 * ndash(l, m),
            &imag, &real
        );

        grid[idx] *= std::complex<S>{real, imag};
    }
}

template<typename T, typename S>
void gridder(
    SpanMatrix<T> grid,
    const SpanVector<WorkUnit<S>> workunits,
    const SpanMatrix<S> subtaper
) {
    auto subgridspec = workunits[0].subgridspec;
    GPUArray<T, 2> subgrid({(size_t) subgridspec.Nx, (size_t) subgridspec.Ny});

    // Make FFT plan
    hipfftHandle plan {};
    int rank[] {(int) subgridspec.Ny, (int) subgridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftPlanMany(
        &plan, 2, rank,
        rank, T::size(), 1,
        rank, T::size(), 1,
        fftType(subgrid.data()), T::size()
    ) );

    for (const WorkUnit<S>& workunit : workunits) {
        UVWOrigin origin {workunit.u0, workunit.v0, workunit.w0};

        auto uvdata = workunit.data;
        auto Aleft = workunit.Aleft;
        auto Aright = workunit.Aright;

        // DFT
        {
            auto fn = gpudift<T, S>;
            auto [nblocks, nthreads] = getKernelConfig(
                fn, subgridspec.Nx * subgridspec.Ny
            );
            hipLaunchKernelGGL(
                fn, nblocks, nthreads, 0, 0,
                subgrid.data(), Aleft.data(),
                Aright.data(), origin, uvdata.data(),
                uvdata.size(), subgridspec
            );
        }

        // Taper
        {
            auto fn = applytaper<T, S>;
            auto [nblocks, nthreads] = getKernelConfig(
                fn, subgridspec.Nx * subgridspec.Ny
            );
            hipLaunchKernelGGL(
                fn, nblocks, nthreads, 0, 0,
                subgrid.data(), subtaper.data(), subgridspec
            );
        }

        // FFT
        fftshift(subgrid.data(), subgridspec);
        fftExec(plan, subgrid.data(), HIPFFT_FORWARD);
        fftshift(subgrid.data(), subgridspec);

        // Add back to master grid
        {
            auto fn = addsubgrid<T>;
            auto [nblocks, nthreads] = getKernelConfig(
                fn, subgridspec.Nx, subgridspec.Ny
            );
            GridSpec gridspec {(long long) grid.size(0), (long long) grid.size(1), 0, 0};
            hipLaunchKernelGGL(
                fn, nblocks, nthreads, 0, 0,
                grid.data(), gridspec, subgrid.data(), subgridspec, workunit.u0px, workunit.v0px
            );
        }
    }
    HIPCHECK( hipDeviceSynchronize() );

    HIPFFTCHECK( hipfftDestroy(plan) );
}

template<template<typename> typename T, typename S>
Matrix<T<S>> invert(
    const SpanVector<WorkUnit<S>> workunits,
    const GridSpec gridspec,
    const SpanMatrix<S> taper,
    const SpanMatrix<S> subtaper
) {
    Matrix<T<S>> img {{(size_t) gridspec.Nx, (size_t) gridspec.Ny}};
    Matrix<T<S>> wlayer {{(size_t) gridspec.Nx, (size_t) gridspec.Ny}};

    GPUMatrix<T<S>> wlayerd {{(size_t) gridspec.Nx, (size_t) gridspec.Ny}};
    GPUMatrix<S> subtaperd {subtaper};

    // Construct FFT plan
    hipfftHandle plan {};
    int rank[] {(int) gridspec.Ny, (int) gridspec.Nx}; // COL MAJOR
    HIPFFTCHECK( hipfftPlanMany(
        &plan, 2, rank,
        rank, T<S>::size(), 1,
        rank, T<S>::size(), 1,
        fftType(wlayerd.data()), T<S>::size()
    ) );

    // Get unique w terms
    std::vector<S> ws(workunits.size());
    std::transform(
        workunits.begin(), workunits.end(),
        ws.begin(),
        [](const auto& workunit) { return workunit.w0; }
    );
    std::sort(ws.begin(), ws.end());
    ws.resize(std::unique(ws.begin(), ws.end()) - ws.begin());

    for (const auto w0 : ws) {
        fmt::println("Processing w={} layer...", w0);

        // TOFIX: This makes a copy of workunits, which is fine for now
        // so long as data is a span, but not when it is owned.
        std::vector<WorkUnit<S>> wworkunits;
        std::copy_if(
            workunits.begin(), workunits.end(), std::back_inserter(wworkunits),
            [=](const auto& workunit) { return workunit.w0 == w0; }
        );

        wlayerd.zero();
        gridder(
            SpanMatrix<T<S>>(wlayerd),
            SpanVector<WorkUnit<S>>(wworkunits),
            SpanMatrix<S>(subtaperd)
        );

        // FFT the full wlayer
        fftshift(wlayerd.data(), gridspec);
        fftExec(plan, wlayerd.data(), HIPFFT_BACKWARD);
        fftshift(wlayerd.data(), gridspec);

        // Apply w correction
        {
            auto fn = wcorrect<StokesI<double>, double>;
            auto [nblocks, nthreads] = getKernelConfig(
                fn, gridspec.Nx, gridspec.Ny
            );
            hipLaunchKernelGGL(
                fn, nblocks, nthreads, 0, 0,
                wlayerd.data(), gridspec, w0
            );
        }

        wlayer = wlayerd;
        img += wlayer;
    }

    // The final image still has a taper applied. It's time to remove it.
    img /= taper;

    hipfftDestroy(plan);

    return img;
}

extern "C" {
    void gridder_lineardouble(
        SpanMatrix<LinearData<double>>* grid,
        WorkUnit<double>* workunits_ptr,
        size_t workunits_n,
        SpanMatrix<double>* subtaper
    ) {
        SpanVector<WorkUnit<double>> workunits(workunits_ptr, {workunits_n});
        gridder(*grid, workunits, *subtaper);
    }

    void gridder_stokesIdouble(
        SpanMatrix<StokesI<double>>* grid,
        WorkUnit<double>* workunits_ptr,
        size_t workunits_n,
        SpanMatrix<double>* subtaper
    ) {
        SpanVector<WorkUnit<double>> workunits(workunits_ptr, {workunits_n});
        gridder(*grid, workunits, *subtaper);
    }

    void gridder_stokesIfloat(
        SpanMatrix<StokesI<float>>* grid,
        WorkUnit<float>* workunits_ptr,
        size_t workunits_n,
        SpanMatrix<float>* subtaper
    ) {
        SpanVector<WorkUnit<float>> workunits(workunits_ptr, {workunits_n});
        gridder(*grid, workunits, *subtaper);
    }

    void invert_stokesIdouble(
        StokesI<double>* img_ptr,
        WorkUnit<double>* workunits_ptr,
        size_t workunits_n,
        GridSpec* gridspec,
        double* taper_ptr,
        double* subtaper_ptr
    ) {
        SpanVector<WorkUnit<double>> workunits {workunits_ptr, {workunits_n}};

        auto subgridspec = workunits[0].subgridspec;

        SpanMatrix<double> taper {
            taper_ptr, {(size_t) gridspec->Nx, (size_t) gridspec->Ny}
        };
        SpanMatrix<double> subtaper {
            subtaper_ptr, {(size_t) subgridspec.Nx, (size_t) subgridspec.Ny}
        };

        auto img = invert<StokesI, double>(
            workunits, *gridspec, taper, subtaper
        );
        memcpy(img_ptr, img.data(), img.size() * sizeof(StokesI<double>));
    }
}