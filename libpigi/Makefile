CXX = hipcc
CXXFLAGS = --std=c++20 --gcc-toolchain=/usr/local -Wall -O1 -D__HIP_PLATFORM_HCC__= -D__HIP_PLATFORM_AMD__= -I/opt/rocm-5.5.0/include -I/opt/rocm-5.5.0/llvm/bin/../lib/clang/16.0.0
LDFLAGS = -shared -fPIC -L/opt/rocm-5.5.0/lib -lamdhip64 -lhipfft -lhipblas -L/usr/local/lib -lfmt

all: libpigi.so

libpigi.so: libpigi.cpp
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $^