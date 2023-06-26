LIB = libblock_sparse.a

LIB_OBJS =\
spmm/blockwise_library.cu.o \
util/random_mask.cc.o

BENCHMARK =\
benchmark/benchmark.spmm_nm.out

TEST =\
test/test.spmm.blockwise.out

CC = g++
CC_INC = -I.. -I/usr/local/cuda/include
CC_FLAG = -O2 -std=c++11

NVCC = nvcc
NVCC_INC = -I..
NVCC_FLAG = -O3 \
  -std=c++17 \
  -arch=sm_$(GPU_CC) \
  -DGPU_CC=$(GPU_CC)

all: lib benchmark test

lib: $(LIB)

$(LIB): $(LIB_OBJS)
	ar cr $@ $^
	ranlib $@

benchmark: $(BENCHMARK)

test: $(TEST)

%.out: %.cu.o $(LIB)
	$(NVCC) $^ -o $@

%.cu.o: %.cu
	$(NVCC) $(NVCC_INC) $(NVCC_FLAG) -c $< -o $@

%.cc.o: %.cc
	$(CC) $(CC_INC) $(CC_FLAG) -c $< -o $@

clean:
	rm -f $(LIB) $(BENCHMARK) $(LIB_OBJS) $(TEST)
