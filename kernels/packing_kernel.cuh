#ifndef PACKING_KERNEL_CUH
#define PACKING_KERNEL_CUH


__global__ void packBits(void *output, const unsigned char *input, const unsigned long long n) {
    unsigned long long tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n)
        return;

    unsigned output_value = __ballot_sync(__activemask(), input[tid]);
    reinterpret_cast<unsigned *>(output)[tid / warpSize] = output_value;
    }
}

#endif  // PACKING_KERNEL_CUH