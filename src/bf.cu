#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

#include "bf_config.h"
// #include "transpose_kernel.cuh"
#include "gemm_kernel.cuh"
#include "packing_kernel.cuh"

#define CEILDIV(A, B) ((A) / (B) + ((A) % (B) != 0))
#define ALIGN(A, B) (CEILDIV(A, B) * B)

using Tin = unsigned char;
using Tout = int32_t;

#define cuda_check(err) { cuda_assert((err),  __FILE__, __LINE__); }
inline void cuda_assert(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::cout << "ERROR: " << cudaGetErrorString(err) << " " << file << " " << line << std::endl;
  }
}

void pack(unsigned char *d_output, const unsigned char* input, unsigned long long n_elements) {
    unsigned char *d_input;
    cuda_check(cudaMalloc(&d_input,  n_elements * sizeof(unsigned char)));

    dim3 threads(WARP_SIZE);
    dim3 grid(CEILDIV(n_elements, threads.x));

    cuda_check(cudaMemcpy(d_input, input, n_elements * sizeof(unsigned char), cudaMemcpyHostToDevice));

    cudaEvent_t start, end;
    cuda_check(cudaEventCreate(&start));
    cuda_check(cudaEventCreate(&end));
    cuda_check(cudaEventRecord(start));
    packBits<<<grid, threads>>>(d_output, d_input, n_elements);
    cudaDeviceSynchronize();
    cuda_check(cudaGetLastError());
    cuda_check(cudaEventRecord(end));
    cuda_check(cudaEventSynchronize(end));

    float time;
    cudaEventElapsedTime(&time, start, end);

    std::cout << "Packing kernel took " << time << " ms" << std::endl;
}


void beamform(Tout *d_BF_arg, const Tin *d_A_arg, const Tin *d_RF_arg) {
  dim3 threads(WARP_SIZE, FRAMES_PER_BLOCK / FRAMES_PER_WARP, BEAMS_PER_BLOCK / BEAMS_PER_WARP);
  dim3 grid(FRAMES / FRAMES_PER_BLOCK, BEAMS / BEAMS_PER_BLOCK);

  const A_t *d_A = reinterpret_cast<const A_t *>(d_A_arg);
  const B_t *d_RF = reinterpret_cast<const B_t *>(d_RF_arg);
  C_t *d_BF = reinterpret_cast<C_t *>(d_BF_arg);

  cudaEvent_t start, end;
  cuda_check(cudaEventCreate(&start));
  cuda_check(cudaEventCreate(&end));
  cuda_check(cudaEventRecord(start));
  beamform_basic<<<grid, threads>>>(*d_BF, *d_A, *d_RF);
  cudaDeviceSynchronize();
  cuda_check(cudaGetLastError());
  cuda_check(cudaEventRecord(end));
  cuda_check(cudaEventSynchronize(end));

  float time;
  cudaEventElapsedTime(&time, start, end);

  std::cout << "Beamforming kernel took " << time << " ms" << std::endl;
  // float tops = 2ULL * FRAMES * BEAMS * SAMPLES * 1e-9 / time;
  // std::cout << "TOPS: " << tops << std::endl;
}


int main() {
  std::cout << "Beamform main" << std::endl;

  // load A
  std::cout << "Loading A matrix from disk" << std::endl;
  Tin *A;
  const size_t bytes_a = sizeof(Tin) * N_A;
  cuda_check(cudaHostAlloc(&A, bytes_a, cudaHostAllocWriteCombined));

  std::string path_a_real = DATA_PATH_INPUT "/" FILENAME_A_REAL;
  std::string path_a_imag = DATA_PATH_INPUT "/" FILENAME_A_IMAG;

  std::ifstream in;
  in = std::ifstream(path_a_real, std::ios::binary | std::ios::in);
  if (!in) {
    std::cerr << "Failed to open A matrix file " << path_a_real << std::endl;
    return 1;
  }
  in.read(reinterpret_cast<char *>(A), bytes_a / COMPLEX);
  in.close();


  in = std::ifstream(path_a_imag, std::ios::binary | std::ios::in);
  if (!in) {
    std::cerr << "Failed to open A matrix file " << path_a_imag << std::endl;
    return 1;
  }
  in.read(reinterpret_cast<char *>(&A[N_A / 2]), bytes_a / COMPLEX);
  in.close();

  // pack A into bits on the GPU
  std::cout << "Packing A matrix" << std::endl;
  Tin *d_A;
  const size_t bytes_a_packed = bytes_a / CHAR_BIT;
  cuda_check(cudaMalloc(&d_A, bytes_a_packed));
  pack(reinterpret_cast<unsigned char *>(d_A), reinterpret_cast<const unsigned char *>(A), N_A);

  // no longer need original data on the host
  cuda_check(cudaFreeHost(A));

  // Repeat for RF
  std::cout << "Loading RF from disk" << std::endl;
  Tin *RF;
  const size_t bytes_rf = sizeof(Tin) * N_RF;
  cuda_check(cudaHostAlloc(&RF, bytes_rf, cudaHostAllocWriteCombined));

  std::string path_rf_real = DATA_PATH_INPUT "/" FILENAME_RF_REAL;
  std::string path_rf_imag = DATA_PATH_INPUT "/" FILENAME_RF_IMAG;
  in = std::ifstream(path_rf_real, std::ios::binary | std::ios::in);
  if (!in) {
    std::cerr << "Failed to open RF file " << path_rf_real << std::endl;
    return 1;
  }
  in.read(reinterpret_cast<char *>(RF), bytes_rf / COMPLEX);
  in.close();

  in = std::ifstream(path_rf_imag, std::ios::binary | std::ios::in);
  if (!in) {
    std::cerr << "Failed to open RF file " << path_rf_imag << std::endl;
    return 1;
  }
  in.read(reinterpret_cast<char *>(&RF[N_RF / 2]), bytes_rf / COMPLEX);
  in.close();

  // pack RF into bits on the GPU
  std::cout << "Packing RF" << std::endl;
  Tin *d_RF;
  const size_t bytes_rf_packed = bytes_rf / CHAR_BIT;
  cuda_check(cudaMalloc(&d_RF, bytes_rf_packed));
  pack(reinterpret_cast<unsigned char *>(d_RF), reinterpret_cast<const unsigned char *>(RF), N_RF);

  // no longer need original data on the host
  cuda_check(cudaFreeHost(RF));

  // allocate BF on host and device
  const size_t bytes_bf = sizeof(Tout) * N_BF;
  Tout *BF, *d_BF;
  cuda_check(cudaHostAlloc(&BF, bytes_bf, cudaHostAllocDefault));
  cuda_check(cudaMalloc(&d_BF, bytes_bf));
  // initialize BF to zero
  cuda_check(cudaMemset(d_BF, 0, bytes_bf));

  // run the beamformer
  std::cout << "Running tensor-core beamformer" << std::endl;
  beamform(d_BF, d_A, d_RF);

  // copy the result to the host
  cuda_check(cudaMemcpy(BF, d_BF, bytes_bf, cudaMemcpyDeviceToHost));

  // write BF to disk
  std::string path_bf = DATA_PATH_OUTPUT "/" FILENAME_BF;
  std::cout << "Writing BF to " << path_bf << std::endl;
  std::ofstream out(path_bf, std::ios::binary | std::ios::out);
  if (!out) {
    std::cerr << "Failed to open BF file " << path_bf << std::endl;
    return 1;
  }
  out.write(reinterpret_cast<char *>(BF), bytes_bf);

  // free BF host memory
  cuda_check(cudaFreeHost(BF));

  // free all device memory
  cuda_check(cudaFree(d_A));
  cuda_check(cudaFree(d_RF));
  cuda_check(cudaFree(d_BF));

  return 0;
}