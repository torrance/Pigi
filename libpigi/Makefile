# Disable automatic rules based on prefixes
.SUFFIXES:

CXXFLAGS = -fPIC --std=c++20 -Wall -Wextra -Wpedantic -O3 -D__HIP_PLATFORM_HCC__= -D__HIP_PLATFORM_AMD__= -I/opt/rocm-5.6.0/include -I/opt/rocm-5.6.0/llvm/bin/../lib/clang/16.0.0 -I/usr/include
LDFLAGS = -L/opt/rocm-5.6.0/lib -lamdhip64 -lhipfft -lhipblas -L/usr/local/lib -lpthread -lfmt -L/usr/lib -lcasa_casa -lcasa_tables -lcfitsio -lgsl

OBJECTS = dft.o fft.o invert.o predict.o findabsmax.o clean.o

# Do nothing
all:

libpigi.so: *.h $(OBJECTS) Makefile
	g++ -c libpigi.cpp $(CXXFLAGS)
	g++ -o libpigi.so libpigi.o $(OBJECTS) -shared -fPIC $(LDFLAGS)

%.o: %.cpp %.h Makefile
	hipcc -c $< --gcc-toolchain=/usr/local $(CXXFLAGS)

# Special fucking case since thrust is not C++20 compatible
findabsmax.o: findabsmax.cpp findabsmax.h Makefile
	hipcc -c $< --gcc-toolchain=/usr/local $(CXXFLAGS) --std=c++17

.PHONY: test
test: test.cpp *.h $(OBJECTS) Makefile
	rm -f test
	g++ -c test.cpp $(CXXFLAGS)
	g++ -o test test.o $(OBJECTS) $(LDFLAGS) -lCatch2Main -lCatch2
	# ./test

.PHONY: clean
clean:
	rm -f *.o libpigi.so test