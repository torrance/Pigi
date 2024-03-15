#pragma once

#include <limits>
#include <type_traits>

#include <fmt/format.h>
#include <hip/hip_runtime.h>
#include <thrust/complex.h>

#include "gridspec.h"
#include "memory.h"

// This is a std::numbers polyfill
// TODO: remove when we can use a more up to date standard library
template <typename T>
constexpr T pi_v = static_cast<T>(3.14159265358979323846264338327950288419716939937510L);

__device__ void sincospif(float, float*, float*);  // Supress IDE warning about device function
__device__ void sincospi(double, double*, double*);  // Suppress IDE warning about device function

// Non Clang compilers don't allow specialisation based
// on compute location, so we need to conditionally include these
#if defined(__clang__) || defined(__CUDA_ARCH__)
__device__ inline auto cispi(const float& theta) {
    float real, imag;
    sincospif(theta, &imag, &real);
    return thrust::complex(real, imag);
}

__device__ inline auto cispi(const double& theta) {
    double real, imag;
    sincospi(theta, &imag, &real);
    return thrust::complex(real, imag);
}
#endif
#if !defined(__CUDA_ARCH__)
template <typename T>
__host__ inline auto cispi(const T& theta) {
    auto pi = ::pi_v<T>;
    return thrust::complex {
        std::cos(theta * pi), std::sin(theta * pi)
    };
}
#endif

template <typename T>
struct fmt::formatter<thrust::complex<T>> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const thrust::complex<T>& value, FormatContext& ctx) {
        return fmt::format_to(
            ctx.out(), "{:.2f} + {:.2f}i", value.real(), value.imag()
        );
    }
};

template <typename T>
struct fmt::formatter<std::array<T, 1>> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const std::array<T, 1>& value, FormatContext& ctx) {
        return fmt::format_to(
            ctx.out(), "{}", value[0]
        );
    }
};

template <typename T>
struct fmt::formatter<std::array<T, 2>> {
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const std::array<T, 2>& value, FormatContext& ctx) {
        return fmt::format_to(
            ctx.out(), "{},{}", value[0], value[1]
        );
    }
};

template <typename P>
__host__ __device__
P abs(thrust::complex<P> x) { return thrust::abs(x); }

template <typename T>
inline bool isfinite(const T& x) requires(std::is_floating_point<T>::value) {
    return std::isfinite(x);
}

template <typename T>
inline bool isfinite(const thrust::complex<T>& x) {
    return std::isfinite(x.real()) && std::isfinite(x.imag());
}

template <typename T>
__host__ __device__
inline T conj(const T& x) requires(std::is_floating_point<T>::value) {
    return x;
}

template <typename T>
__host__ __device__
inline thrust::complex<T> conj(const thrust::complex<T>& x) { return thrust::conj(x); }

template <typename T>
__host__ __device__
inline T ndash(const T l, const T m) {
    auto r2 = std::min<T>(
        l*l + m*m, 1
    );
    return r2 / (1 + sqrt(1 - r2));
}

template <typename T> requires(std::is_floating_point<T>::value)
constexpr inline T deg2rad(const T& x) { return x * ::pi_v<T> / 180; }

template <typename T> requires(std::is_floating_point<T>::value)
constexpr inline T rad2deg(const T& x) { return x * 180 / ::pi_v<T>; }

template <typename T>
auto resize(HostSpan<T, 2> src, GridSpec srcGridspec, GridSpec dstGridspec) {
    HostArray<T, 2> dst {{dstGridspec.Nx, dstGridspec.Ny}, false};

    long long edgeX {(dstGridspec.Nx - srcGridspec.Nx) / 2};
    long long edgeY {(dstGridspec.Ny - srcGridspec.Ny) / 2};

    // This is a small optimization: only zero the memory if the dst array is larger
    // than the source
    if (edgeX > 0 || edgeY > 0) { dst.zero(); }

    // The row length to copy is simply the minimum of the dst or src row length
    size_t count = std::min(srcGridspec.Nx, dstGridspec.Nx) * sizeof(T);

    // Determine y-bounds of dst array
    const long long ymin = std::max(0ll, edgeY);
    const long long ymax = dst.size(1) - std::max(0ll, edgeY);

    // Copy row by row
    for (long long nyDst {ymin}; nyDst < ymax; ++nyDst) {
        long long nySrc = nyDst - edgeY;

        long long nxDst = std::max(0ll, edgeX);
        long long nxSrc = std::max(0ll, -edgeX);

        void* ptrDst =  dst.data() + dstGridspec.gridToLinear(nxDst, nyDst);
        void* ptrSrc = src.data() + srcGridspec.gridToLinear(nxSrc, nySrc);
        memcpy(ptrDst, ptrSrc, count);
    }

    return dst;
}

template <typename T, typename S>
HostArray<T, 2> convolve(const HostSpan<T, 2> img, const HostSpan<S, 2> kernel) {
    shapecheck(img, kernel);

    // Pad img and kernel with zeros
    GridSpec gridspec {.Nx=img.size(0), .Ny=img.size(1)};
    GridSpec gridspecPadded {2 * img.size(0), 2 * img.size(1), 0, 0};

    auto imgPadded = resize(img, gridspec, gridspecPadded);
    auto kernelPadded = resize(kernel, gridspec, gridspecPadded);

    // Create fft plans
    auto planImg = fftPlan<T>(gridspecPadded);
    auto planKernel = fftPlan<S>(gridspecPadded);

    // Send to device
    DeviceArray<T, 2> img_d {imgPadded};
    DeviceArray<S, 2> kernel_d {kernelPadded};

    // FT forward
    fftExec(planImg, img_d, HIPFFT_FORWARD);
    fftExec(planKernel, kernel_d, HIPFFT_FORWARD);

    // Multiply in FT domain and normalize
    map([=] __device__ (auto& img, auto kernel) {
        img *= (kernel /= gridspecPadded.size());
    }, img_d, kernel_d);

    // FT backward
    fftExec(planImg, img_d, HIPFFT_BACKWARD);

    HIPFFTCHECK( hipfftDestroy(planImg) );
    HIPFFTCHECK( hipfftDestroy(planKernel) );

    // Copy back from device
    copy(imgPadded, img_d);

    // Remove padding and return
    return resize(imgPadded, gridspecPadded, gridspec);
}

template <typename T>
HostArray<T, 2> rescale(
    const HostSpan<T, 2> img, const GridSpec from, const GridSpec to
) {
    if (from.scaleu != to.scaleu || from.scalev != to.scalev) {
        throw std::runtime_error("Resampling requires matching UV scales");
    };

    // Perform FFT on device
    DeviceArray<T, 2> img_d {img};
    {
        auto plan = fftPlan<T>(from);
        fftExec(plan, img_d, HIPFFT_FORWARD);
        hipfftDestroy(plan);
    }

    // Transfer to host and resize/pad with zeros
    // TODO: Implement resize() on device
    HostArray<T, 2> img_h {img_d};
    img_h = resize(img_h, from, to);  // move

    // Normalise
    img_h /= T(from.size());

    // Back to host and FFT
    img_d = DeviceArray<T, 2>{img_h};  // move
    {
        auto plan = fftPlan<T>(to);
        fftExec(plan, img_d, HIPFFT_BACKWARD);
        hipfftDestroy(plan);
    }

    return HostArray<T, 2>{img_d};
}

template <typename T=size_t>
class Iota {
public:
    class Iterator {
    public:
        Iterator(T val) : val(val) {}

        Iterator& operator++() {
            ++val;
            return *this;
        }

        T operator*() const { return val; }

        bool operator==(const Iterator& other) const { return val == other.val; }
        bool operator!=(const Iterator& other) const { return !(*this == other); }

    private:
        T val;
    };

    __host__ __device__
    T operator[](size_t i) { return i; }

    size_t size() { return std::numeric_limits<T>::max(); }

    Iterator begin() { return Iterator{ 0 }; }
    Iterator end() { return Iterator{ std::numeric_limits<T>::max() }; }
};

template <typename F, typename T, typename... Ts>
__global__
void _map(size_t N, F f, T x, Ts... xs) {
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < N;
        idx += blockDim.x * gridDim.x
    ) {
        f(x[idx], xs[idx]...);
    }
}

auto _mapforward(const auto& x) { return x; }

template <typename T, int N, typename Pointer>
Span<T, N, Pointer> _mapforward(const Array<T, N, Pointer>& x) { return x; }

template<typename F, typename T, typename... Ts>
void map(F f, T&& x, Ts&&... xs) {
    size_t N { std::min({x.size(), xs.size()...}) };
    auto fn = _map<F, decltype(_mapforward(x)), decltype(_mapforward(xs))...>;
    auto [nblocks, nthreads] = getKernelConfig(fn, N);
    hipLaunchKernelGGL(
        fn, nblocks, nthreads, 0, hipStreamPerThread,
        N, f, _mapforward(x), _mapforward(xs)...
    );
}

/**
 * Ceiling integer division
 */
template <typename T>
requires(std::is_integral<T>::value)
__host__ __device__
inline auto cld(T x, T y) {
    return (x + y - 1) / y;
}

struct Interval {
    double start;
    double end;

    double mid() const {
        return (start + end) / 2;
    }

    bool contains(double val) const {
        return start <= val && val < end;
    }
};