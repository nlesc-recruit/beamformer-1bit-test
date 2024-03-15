#define DATA_PATH_INPUT "E:/eScience/sign_data"
#define DATA_PATH_OUTPUT "E:/eScience/sign_data"
#define FILENAME_A_REAL "A_Matrix_Real_524288x38880.bin"
#define FILENAME_A_IMAG "A_Matrix_Imag_524288x38880.bin"
#define FILENAME_RF_REAL "RF_Real_524288_8041.bin"
#define FILENAME_RF_IMAG "RF_Imag_524288_8041.bin"
#define FILENAME_BF "BF_test.bin"

// input shapes, put here the number of pixels (BEAMS), number of repeats/slow time samples (FRAMES) and
// number of fast time samples (SAMPLES). All values must be multiples of the value per block (for beams and frames)
// or the value per wmma (samples)
// all numbers must be followed by ULL
#define BEAMS 38880ULL
#define FRAMES 8041ULL
#define SAMPLES 524288ULL

// Values below this line should not need to be modified
#define N_A (COMPLEX * BEAMS * SAMPLES)
// #define N_A 40768634880
#define N_RF (COMPLEX * FRAMES * SAMPLES)
#define N_BF (COMPLEX * BEAMS * FRAMES)

#define BEAMS_PER_BLOCK 64
#define BEAMS_PER_WARP 32
#define BEAMS_PER_WMMA 16

#define FRAMES_PER_BLOCK 64
#define FRAMES_PER_WARP 32
#define FRAMES_PER_WMMA 8

#define SAMPLES_PER_WMMA 256
#define NUM_EXTRA_SAMPLES 0
#define WARP_SIZE 32
#define NBUFFER 4

#define NBIT 1
#define COMPLEX 2
#define REAL 0
#define IMAG 1