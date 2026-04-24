# Optional: CUDA_ARCH="-gencode arch=compute_89,code=sm_89"
NVCC ?= nvcc
CU = csrc/kmeans_pp_l2.cu
PKG = q_kmeanspp/libq_kmeanspp_cuda.so

$(PKG): $(CU)
	$(NVCC) -O3 -std=c++17 -Xcompiler -fPIC -shared $(CU) -o $(PKG) -lcudart $(CUDA_ARCH)

# Standalone timing binary (same as before)
kmeans_pp_test: $(CU)
	$(NVCC) -O3 -std=c++17 $(CU) -o kmeans_pp_test -D KMEANS_PP_STANDALONE_MAIN -lcudart $(CUDA_ARCH)

clean:
	rm -f $(PKG) kmeans_pp_test

.PHONY: clean
