/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve.cuh"


/* 
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source: 
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data,
    int padded_length) {


    /* TODO: Implement the point-wise multiplication and scaling for the
    FFT'd input and impulse response. 

    Recall that these are complex numbers, so you'll need to use the
    appropriate rule for multiplying them. 

    Also remember to scale by the padded length of the signal
    (see the notes for Question 1).

    As in Assignment 1 and Week 1, remember to make your implementation
    resilient to varying numbers of threads.

    */

   uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
   float a, b, c, d;
   for (int i = thread_index; i < padded_length; i += blockDim.x * gridDim.x) {
        a = raw_data[i].x;
        b = raw_data[i].y;
        c = impulse_v[i].x;
        d = impulse_v[i].y;
        out_data[i].x = (a * c - b * d) / ((float) padded_length);
        out_data[i].y = (a * d + b * c) / ((float) padded_length);
   }
}

__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the maximum-finding.

    There are many ways to do this reduction, and some methods
    have much better performance than others. 

    For this section: Please explain your approach to the reduction,
    including why you chose the optimizations you did
    (especially as they relate to GPU hardware).

    You'll likely find the above atomicMax function helpful.
    (CUDA's atomicMax function doesn't work for floating-point values.)
    It's based on two principles:
        1) From Week 2, any atomic function can be implemented using
        atomic compare-and-swap.
        2) One can "represent" floating-point values as integers in
        a way that preserves comparison, if the sign of the two
        values is the same. (see http://stackoverflow.com/questions/
        29596797/can-the-return-value-of-float-as-int-be-used-to-
        compare-float-in-cuda)

    */

    // with the `extern` keyword, we can declare a shared memory array whose size is known at runtime
    // the size of the array is specified in the kernel call 
    extern __shared__ float sdata[];
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    float local_max = -INFINITY;
    for (int i = thread_index; i < padded_length; i += blockDim.x * gridDim.x) {
        local_max = max(local_max, out_data[i].x);
    }

    sdata[threadIdx.x] = local_max;
    __syncthreads();

    // reduction logic: each thread will compare its value with the value of the thread at the other end of the block
    // the thread with the smaller value will be discarded. This process is repeated until only one value remains.
    // the final value is stored in the first element of the shared memory array.
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + i]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // the first thread of each block will compare the value in the shared memory array with the value in the global memory array
        atomicMax(max_abs_val, sdata[0]);
    }
}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the division kernel. Divide all
    data by the value pointed to by max_abs_val. 

    This kernel should be quite short.
    */

    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = thread_index; i < padded_length; i += blockDim.x * gridDim.x) {
        out_data[i].x /= *max_abs_val;
        out_data[i].y /= *max_abs_val;
    }

}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
        
    /* TODO: Call the element-wise product and scaling kernel. */

    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>(raw_data, impulse_v, out_data, padded_length);

}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        

    /* TODO 2: Call the max-finding kernel. */

    // following the cuda syntax, kernelName<<<numBlocks, numThreads, sharedMemSize>>>
    cudaMaximumKernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(out_data, max_abs_val, padded_length);
}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* TODO 2: Call the division kernel. */

    cudaDivideKernel<<<blocks, threadsPerBlock>>>(out_data, max_abs_val, padded_length);
}
