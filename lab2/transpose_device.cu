#include <cassert>
#include <cuda_runtime.h>
#include "transpose_device.cuh"

/*
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
 */

/*
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304    matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */

const int TILE_DIM = 64;
const int BLOCK_ROWS = 16;

__global__ void naiveTransposeKernel(const float *input, float *output)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        output[x * width + (y + j)] = input[(y + j) * width + x];
}

__global__ void shmemTransposeKernel(const float *input, float *output)
{
    // TODO: Modify transpose kernel to use shared memory. All global memory
    // reads and writes should be coalesced. Minimize the number of shared
    // memory bank conflicts (0 bank conflicts should be possible using
    // padding). Again, comment on all sub-optimal accesses.

    // __shared__ float data[???];

    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        output[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

__global__ void optimalTransposeKernel(const float *input, float *output)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        output[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

/**
 * Block size is set to 64 x 16 such that we could parallelise over
 * the column with four operations, aka, end_j = j + 4
 */
void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    if (type == NAIVE)
    {
        dim3 blockSize(TILE_DIM, BLOCK_ROWS);
        dim3 gridSize(n / TILE_DIM, n / TILE_DIM);
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output);
    }
    else if (type == SHMEM)
    {
        dim3 blockSize(TILE_DIM, BLOCK_ROWS);
        dim3 gridSize(n / TILE_DIM, n / TILE_DIM);
        shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output);
    }
    else if (type == OPTIMAL)
    {
        dim3 blockSize(TILE_DIM, BLOCK_ROWS);
        dim3 gridSize(n / TILE_DIM, n / TILE_DIM);
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output);
    }
    // Unknown type
    else
        assert(false);
}
