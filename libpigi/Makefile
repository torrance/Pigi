# Disable automatic rules based on prefixes
.SUFFIXES:

CXX = hipcc
CXXFLAGS = -fPIC --std=c++20 -Wall -Wextra -Wpedantic -O3 -D__HIP_PLATFORM_HCC__= -D__HIP_PLATFORM_AMD__= -I/opt/rocm-5.2.5/include -I/opt/rocm-5.2.5/llvm/bin/../lib/clang/14.0.0 -I/opt/rocm-5.2.5/hsa/include -I/usr/include
LDFLAGS = -L/opt/rocm-5.2.5/lib -lrocm-core -lamdhip64 -lhipfft -lhipblas -L/usr/local/lib -lpthread -lfmt -L/usr/lib -lcasa_casa -lcasa_tables -lcfitsio -lgsl

# Do nothing
all:

.PHONY: test
test: test.cpp *.h Makefile
	rm -f test
	$(CXX) -o test test.cpp $(CXXFLAGS) $(LDFLAGS) -lCatch2Main -lCatch2

.PHONY: clean
clean:
	rm -f test