CC := gcc
CXX := g++
CYTHON := cython

NVCC_FOUND := $(shell type nvcc > /dev/null)$(shell echo $$?)
NVCC := nvcc

PYTHON_INCLUDES := $(shell python3 -c "from sysconfig import get_paths as gp; print(gp()[\"include\"])")
CFLAGS := -pthread -DNDEBUG -g -fwrapv -O3 -march=native -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -I$(PYTHON_INCLUDES) -Isrc
LDFLAGS := -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-Bsymbolic-functions -Wl,-z,relro -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2
LDFLAGS_CUDA := -L/usr/local/cuda/lib64
LDLIBS := -lm
LDLIBS_CUDA := -lcudart
NVCCFLAGS := --generate-code arch=compute_72,code=sm_72 --generate-code arch=compute_75,code=sm_75

SOURCE_PREFIX := src
BUILD_PREFIX := build

.PHONY: all clean

all: trajutil.so trajeometry.so fish2bird.so fuzzylines.so linetrack.so trajectorybuild.so #positioning.so

ifeq ($(NVCC_FOUND), 0)
fish2bird.so: $(SOURCE_PREFIX)/fish2bird.pyx $(SOURCE_PREFIX)/fish2bird_cuda.cu
	echo $(NVCC_FOUND)
	cython -3 -a -E CUDA_ENABLED=1 $(SOURCE_PREFIX)/fish2bird.pyx -o $(BUILD_PREFIX)/fish2bird.c
	$(CC) -c $(CFLAGS) $(BUILD_PREFIX)/fish2bird.c -o $(BUILD_PREFIX)/fish2bird.o
	$(NVCC) -c $(NVCCFLAGS) $(SOURCE_PREFIX)/fish2bird_cuda.cu -o $(BUILD_PREFIX)/fish2bird_cuda.o
	$(CXX) $(CFLAGS) $(LDFLAGS) $(LDFLAGS_CUDA) $(BUILD_PREFIX)/fish2bird.o $(BUILD_PREFIX)/fish2bird_cuda.o $(LDLIBS) $(LDLIBS_CUDA) -o fish2bird.so
else
fish2bird.so: $(BUILD_PREFIX)/fish2bird.c
	$(CC) $(CFLAGS) $(LDFLAGS) $(BUILD_PREFIX)/fish2bird.c $(LDLIBS) -o fish2bird.so
endif

fuzzylines.so: $(BUILD_PREFIX)/fuzzylines.c
	$(CC) $(CFLAGS) $(LDFLAGS) $(BUILD_PREFIX)/fuzzylines.c $(LDLIBS) -o fuzzylines.so

linetrack.so: $(BUILD_PREFIX)/linetrack.cpp
	$(CXX) $(CFLAGS) $(LDFLAGS) $(BUILD_PREFIX)/linetrack.cpp $(LDLIBS) -o linetrack.so

# positioning.so: $(BUILD_PREFIX)/positioning.cpp
#	$(CXX) $(CFLAGS) $(LDFLAGS) $(BUILD_PREFIX)/positioning.cpp $(LDLIBS) -o positioning.so

trajectorybuild.so: $(BUILD_PREFIX)/trajectorybuild.cpp
	$(CXX) $(CFLAGS) $(LDFLAGS) $(BUILD_PREFIX)/trajectorybuild.cpp $(LDLIBS) -o trajectorybuild.so

trajeometry.so: $(BUILD_PREFIX)/trajeometry.cpp
	$(CXX) $(CFLAGS) $(LDFLAGS) $(BUILD_PREFIX)/trajeometry.cpp $(LDLIBS) -o trajeometry.so

trajutil.so: $(BUILD_PREFIX)/trajutil.cpp
	$(CXX) $(CFLAGS) $(LDFLAGS) $(BUILD_PREFIX)/trajutil.cpp $(LDLIBS) -o trajutil.so

$(BUILD_PREFIX)/%.c : $(SOURCE_PREFIX)/%.pyx
	cython -3 -a $< -o $@

$(BUILD_PREFIX)/%.cpp : $(SOURCE_PREFIX)/%.pyx
	cython -3 -a --cplus $< -o $@

clean:
	rm $(BUILD_PREFIX)/* *.so