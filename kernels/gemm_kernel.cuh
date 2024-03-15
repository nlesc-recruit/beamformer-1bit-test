#ifndef GEMM_KERNEL_CUH
#define GEMM_KERNEL_CUH

#include <mma.h>

#include "async_copies.h"
#include "wmma_extension.h"

using namespace nvcuda;

// All values related to data layout must be defined at compile time
#if !defined BEAMS || !defined FRAMES || !defined SAMPLES
#error                                                                         \
    "BEAMS, FRAMES, SAMPLES and values per block, warp, tensor core must be defined at compile time"
#endif

#if NBIT == 1
using Tin = unsigned char;
using Ttc = wmma::experimental::precision::b1;
using Tout = int32_t;
#else
#error NBIT must be 1
#endif

// basic data layout
using A_t = Tin[COMPLEX][BEAMS][SAMPLES/CHAR_BIT];
using B_t = Tin[COMPLEX][FRAMES][SAMPLES/CHAR_BIT];
// data layout for optimal transfer to shared memory
using A_opt_t = Tin[BEAMS/BEAMS_PER_BLOCK][SAMPLES/SAMPLES_PER_WMMA][COMPLEX][BEAMS_PER_BLOCK][SAMPLES_PER_WMMA/CHAR_BIT];
using B_opt_t = Tin[FRAMES/FRAMES_PER_BLOCK][SAMPLES/SAMPLES_PER_WMMA][COMPLEX][FRAMES_PER_BLOCK][SAMPLES_PER_WMMA/CHAR_BIT];

using C_t = Tout[COMPLEX][BEAMS][FRAMES];


extern "C"
__global__ void beamform_basic(C_t &C, const A_t &A, const B_t &B) {
    const unsigned blockFrame = blockIdx.x;
    const unsigned blockBeam = blockIdx.y;
    const unsigned warpFrame = threadIdx.y;
    const unsigned warpBeam = threadIdx.z;

    // number of tiles processed by each warp
    constexpr unsigned BEAM_TILES = BEAMS_PER_WARP / BEAMS_PER_WMMA;
    constexpr unsigned FRAME_TILES = FRAMES_PER_WARP / FRAMES_PER_WMMA;
    constexpr unsigned SAMPLE_TILES = SAMPLES / SAMPLES_PER_WMMA;

    wmma::fragment<wmma::accumulator, BEAMS_PER_WMMA, FRAMES_PER_WMMA, SAMPLES_PER_WMMA, Tout> sum[COMPLEX][BEAM_TILES][FRAME_TILES];
    for (int c = 0; c < COMPLEX; c++) {
        for (int beam = 0; beam < BEAM_TILES; beam++) {
            for (int frame = 0; frame < FRAME_TILES; frame++) {
                wmma::fill_fragment(sum[c][beam][frame], 0);
            }
        }
    }

    for (int sample = 0; sample < SAMPLE_TILES; sample++) {
        // declare input fragments
        wmma::fragment<wmma::matrix_a, BEAMS_PER_WMMA, FRAMES_PER_WMMA, SAMPLES_PER_WMMA, Ttc, wmma::row_major> a[COMPLEX][BEAM_TILES];
        wmma::fragment<wmma::matrix_b, BEAMS_PER_WMMA, FRAMES_PER_WMMA, SAMPLES_PER_WMMA, Ttc, wmma::col_major> b[COMPLEX][FRAME_TILES];

        // load matrices from global memory
        for (int c = 0; c < COMPLEX; c++){
            for (int beam = 0; beam < BEAM_TILES; beam++) {
                int sample_index = sample * SAMPLES_PER_WMMA;
#if NBIT == 1
                sample_index /= CHAR_BIT;
#endif
                wmma::load_matrix_sync(a[c][beam], &(A[c][blockBeam * BEAMS_PER_BLOCK + warpBeam * BEAMS_PER_WARP + beam * BEAMS_PER_WMMA][sample_index]), SAMPLES);

            }
        }

        for (int c = 0; c < COMPLEX; c++){
            for (int frame = 0; frame<FRAME_TILES; frame++) {
                int sample_index = sample * SAMPLES_PER_WMMA;
#if NBIT == 1
                sample_index /= CHAR_BIT;
#endif
                wmma::load_matrix_sync(b[c][frame], &(B[c][blockFrame * FRAMES_PER_BLOCK + warpFrame * FRAMES_PER_WARP + frame * FRAMES_PER_WMMA][sample_index]), SAMPLES);
            }
        }

        /*
        tensor cores don't have subtraction
        (ar + ai i) * (br + bi i) ->
        real part is ar * br - ai * bi
        imag part is ar * bi + ai * br

        need to negate either ai or bi before calculating ai*bi for real part

        1. real part += ar * br
        2. imag part += ar * bi
        3. negate bi
        4. real part += ai * (-bi)
        5. imag part += ai * br

        */

        // step 1 and 2
        for (int beam = 0; beam < BEAM_TILES; beam++) {
            for (int frame = 0; frame < FRAME_TILES; frame++) {
#if NBIT == 1
                wmma::bmma_sync(sum[REAL][beam][frame], a[REAL][beam], b[REAL][frame], sum[REAL][beam][frame]);
                wmma::bmma_sync(sum[IMAG][beam][frame], a[REAL][beam], b[IMAG][frame], sum[IMAG][beam][frame]);
#else
                wmma::mma_sync(sum[REAL][beam][frame], a[REAL][beam], b[REAL][frame], sum[REAL][beam][frame]);
                wmma::mma_sync(sum[IMAG][beam][frame], a[REAL][beam], b[IMAG][frame], sum[IMAG][beam][frame]);
#endif
            }
        }

        // step 3
        __syncwarp();
        for (int frame = 0; frame < FRAME_TILES; frame++) {
            for (auto &element : b[IMAG][frame].x) {
#if NBIT == 1
                element = ~element;
#else
                element = -element;
#endif
            }
        }
        __syncwarp();

        // step 4 and 5
        for (int beam = 0; beam < BEAM_TILES; beam++) {
            for (int frame = 0; frame < FRAME_TILES; frame++) {
#if NBIT == 1
                wmma::bmma_sync(sum[REAL][beam][frame], a[IMAG][beam], b[IMAG][frame], sum[REAL][beam][frame]);
                wmma::bmma_sync(sum[IMAG][beam][frame], a[IMAG][beam], b[REAL][frame], sum[IMAG][beam][frame]);
#else
                wmma::mma_sync(sum[REAL][beam][frame], a[IMAG][beam], b[IMAG][frame], sum[REAL][beam][frame]);
                wmma::mma_sync(sum[IMAG][beam][frame], a[IMAG][beam], b[REAL][frame], sum[IMAG][beam][frame]);
#endif
            }
        }
    }

#if NBIT == 1
    // Fix result: a dot b = K - 2 * popc(a xor b)
    // 2 K here because we do two TC operations per sum fragment, so 2 K values were added together
    // should also take care of padding: extra samples are zero, interpreted as -1. So need to add the number of extra samples to output
    __syncwarp();
    for (int c = 0; c < COMPLEX; c++) {
        for (int beam = 0; beam < BEAM_TILES; beam++) {
            for (int frame = 0; frame < FRAME_TILES; frame++) {
                for (auto &element : sum[c][beam][frame].x) {
                    element = 2 * (SAMPLES + NUM_EXTRA_SAMPLES - element);
                }
            }
        }
    }
    __syncwarp();
#endif

    // store the result to global memory
    for (int c = 0; c < COMPLEX; c++) {
        for (int beam = 0; beam < BEAM_TILES; beam++) {
            for (int frame = 0; frame < FRAME_TILES; frame++) {
                Tout *c_ptr = &(C[c][blockBeam * BEAMS_PER_BLOCK + warpBeam * BEAMS_PER_WARP + beam * BEAMS_PER_WMMA][blockFrame * FRAMES_PER_BLOCK + warpFrame * FRAMES_PER_WARP + frame * FRAMES_PER_WMMA]);
                wmma::store_matrix_sync(c_ptr, sum[c][beam][frame], FRAMES, wmma::mem_row_major);
            }
        }
    }
}
#endif  // GEMM_KERNEL_CUH