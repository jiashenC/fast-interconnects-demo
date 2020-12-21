#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cassert>

// Add a scalar to the vector
__global__ void vadd(int* const v, int const a, size_t const len) {
  const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int gsize = gridDim.x * blockDim.x;

  for (size_t i = gid; i < len; i += gsize) {
    v[i] += a;
  }
}

int main(int argc, char* argv[]) {
  // GPU device ID
  const int gpu_id = 0;
  cudaSetDevice(gpu_id);

  // Vector length
  size_t LEN = std::stol(argv[1]);

  // GPU kernel parameters
  int grid;
  int block;
  cudaOccupancyMaxPotentialBlockSize(&grid, &block, vadd, 0, 0);

  // Allocate vector
  int* h_data = static_cast<int*>(malloc(sizeof(int) * LEN));
  for (size_t i = 0; i < LEN; ++i) {
    h_data[i] = 1;
  }
  int* d_data;

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

#ifdef CUDA_COHERENT
  d_data = h_data;
#elif CUDA_UM
  cudaMallocManaged(&d_data, LEN * sizeof(int));
  cudaMemAdvise(d_data, LEN * sizeof(int), cudaMemAdviseSetAccessedBy, gpu_id);
  memcpy(d_data, h_data, LEN * sizeof(int));
#elif CUDA_ZEROCOPY
  cudaHostRegister(h_data, LEN * sizeof(int), cudaHostRegisterMapped);
  cudaHostGetDevicePointer(&d_data, h_data, 0);
#else
  cudaMalloc(&d_data, LEN * sizeof(int));
  cudaMemcpyAsync(d_data, h_data, sizeof(int) * LEN, cudaMemcpyHostToDevice);
#endif

  // Call a function to do some work
#ifdef CUDA_UM
  cudaMemPrefetchAsync(d_data, LEN * sizeof(int), gpu_id);
#endif
  vadd<<<grid, block>>>(d_data, 1, LEN);

#if CUDA_UM
  cudaMemPrefetchAsync(d_data, LEN * sizeof(int), cudaCpuDeviceId);
  memcpy(h_data, d_data, LEN * sizeof(int));
#elif CUDA_ZEROCOPY
#else
  cudaMemcpyAsync(h_data, d_data, sizeof(int) * LEN, cudaMemcpyDeviceToHost);
#endif

  cudaDeviceSynchronize();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

  // Verify that result is correct
  unsigned long long sum = 0;
  for (size_t i = 0; i < LEN; ++i) {
    sum += h_data[i];
  }
  assert(sum / 2 == LEN);

#ifdef CUDA_COHERENT
  d_data = nullptr;
#elif CUDA_UM
  cudaFree(d_data);
#elif CUDA_ZEROCOPY
  cudaHostUnregister(d_data);
  h_data = nullptr;
#else
  cudaFree(d_data);
#endif

  std::exit(EXIT_SUCCESS);
}
