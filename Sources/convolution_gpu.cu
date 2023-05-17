#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define TILE_W 16
#define TILE_H 16
#define BLOCK 512

inline int get_number_of_blocks(int array_size, int block_size)
{
    return array_size / block_size + ((array_size % block_size > 0) ? 1 : 0);
}

__device__ __constant__ float kernel_gauss5x5[25] =
{
    0.00296902,      0.0133062,       0.0219382,       0.0133062,       0.00296902,
    0.0133062,       0.0596343,       0.0983203,       0.0596343,       0.0133062,
    0.0219382,       0.0983203,       0.162103,        0.0983203,       0.0219382,
    0.0133062,       0.0596343,       0.0983203,       0.0596343,       0.0133062,
    0.00296902,      0.0133062,       0.0219382,       0.0133062,       0.00296902
};

__device__ __constant__ float kernel_gauss7x7[49] =
{
    0.00001965,	0.00023941,	0.00107296,	0.00176901,	0.00107296,	0.00023941,	0.00001965,
    0.00023941,	0.0029166,	0.01307131,	0.02155094,	0.01307131,	0.0029166,	0.00023941,
    0.00107296,	0.01307131,	0.05858154,	0.09658462,	0.05858154,	0.01307131,	0.00107296,
    0.00176901,	0.02155094,	0.09658462,	0.15924113,	0.09658462,	0.02155094,	0.00176901,
    0.00107296,	0.01307131,	0.05858154,	0.09658462,	0.05858154,	0.01307131,	0.00107296,
    0.00023941,	0.0029166,	0.01307131,	0.02155094,	0.01307131,	0.0029166,	0.00023941,
    0.00001965,	0.00023941,	0.00107296,	0.00176901,	0.00107296,	0.00023941,	0.00001965,
};


__global__ void gauss5x5_kernel(const float* __restrict__ in, float *out, int w, int h)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=  w * h * 3)
        return;

    unsigned int y = idx / (3*w);
    unsigned int offset = idx - y * 3 * w;
    unsigned int x = offset % w;
    unsigned int c = offset / w;
    
    unsigned int idx4 = y * w * 4 + c*w + x;

    if (x <= 1 || y <= 1 || x >= w-2 || y >= h-2)
    {
        out[idx4] = in[idx4];
        return;
    }

    float sum = 0;
#pragma unroll
    for(int i = -2; i <=2; i++)
    for(int j = -2; j <=2; j++)
    {
        sum += in[idx4 + j + 4*w*i] *kernel_gauss5x5[(i + 2) * 5 + (j + 2)];
    }

    out[idx4] = sum;
}

__global__ void gauss5x5_tiles_kernel(const float* __restrict__ in, float *out, int w, int h)
{   
    const int R = 2;
    const int BLOCK_W = (TILE_W + 2*R);
    const int BLOCK_H = (TILE_H + 2*R);
    __shared__ float smem[BLOCK_W*BLOCK_H];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int offset = blockIdx.x * TILE_W + tx-R;
    int x = offset % w;
    int c = offset / w;
    int y = blockIdx.y * TILE_H + ty-R;

    // clamp to edge of image
    x = max(0, x);
    x = min(x, w-1);
    y = max(y, 0);
    y = min(y, h-1);
    //x = clamp(x, 0, w - 1);
    //y = clamp(y, 0, h - 1);

    unsigned int idx = y*w*4 + c*w + x;
    unsigned int bindex = threadIdx.y*BLOCK_W+threadIdx.x;
    
    // each thread copies its pixel of the block to shared memory
    smem[bindex] = in[idx];
    __syncthreads();

    float sum = 0;

    // only threads inside the apron will write results
    if (threadIdx.x >= R && threadIdx.x < (BLOCK_W-R) && threadIdx.y >= R && threadIdx.y < (BLOCK_H-R))
    {
#pragma unroll
        for(int i = -R; i <=R; i++)
        for(int j = -R; j <=R; j++)
        {
            sum += smem[bindex + (i*blockDim.x) + j] * kernel_gauss5x5[(i + R) * 5 + (j + R)];
        }

        out[idx] = sum;
    }
}

__global__ void gauss7x7_tiles_kernel(const float* __restrict__ in, float *out, int w, int h)
{   
    const int R = 3;
    const int BLOCK_W = (TILE_W + 2*R);
    const int BLOCK_H = (TILE_H + 2*R);
    __shared__ float smem[BLOCK_W*BLOCK_H];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int offset = blockIdx.x * TILE_W + tx-R;
    unsigned int x = offset % w;
    unsigned int c = offset / w;
    int y = blockIdx.y * TILE_H + ty-R;

    // clamp to edge of image
    x = max(0, x);
    x = min(x, w-1);
    y = max(y, 0);
    y = min(y, h-1);

    unsigned int idx = y*w*4 + c*w + x;
    unsigned int bindex = threadIdx.y*BLOCK_W+threadIdx.x;
    
    // each thread copies its pixel of the block to shared memory
    smem[bindex] = in[idx];
    __syncthreads();

    float sum = 0;

    // only threads inside the apron will write results
    if (threadIdx.x >= R && threadIdx.x < (BLOCK_W-R) && threadIdx.y >= R && threadIdx.y < (BLOCK_H-R))
    {
#pragma unroll
        for(int i = -R; i <=R; i++)
        for(int j = -R; j <=R; j++)
        {
            sum += smem[bindex + (i*blockDim.x) + j] *kernel_gauss7x7[(i + R) * 7 + (j + R)];
        }

        out[idx] = sum;
    }
}

__global__ void gauss7x7_kernel(const  float* __restrict__ in, float *out, int w, int h)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >=  w * h * 3)
        return;

    unsigned int y = idx / (3*w);
    unsigned int offset = idx - y * 3 * w;
    unsigned int x = offset % w;
    unsigned int c = offset / w;
    
    unsigned int idx4 = y * w * 4 + c*w + x;

    if (x <= 2 || y <= 2 || x >= w-3 || y >= h-3)
    {
        out[idx4] = in[idx4];
        return;
    }

    float sum = 0;
#pragma unroll
    for(int i = -3; i <=3; i++)
    for(int j = -3; j <=3; j++)
    {
        sum += in[idx4 + j + 4*w*i] *kernel_gauss7x7[(i + 3) * 7 + (j + 3)];
    }

    out[idx4] = sum;
}

extern "C" void gauss5x5_gpu_tiles(float* d_src, float* d_dest, unsigned char* d_result, int w, int h, int cycles)
{
    dim3 dimGrid ((w*3) / TILE_W, h / TILE_H);
    dim3 dimBlock(TILE_W+4, TILE_H+4);

    float* src = d_src;
    while (cycles--)
    {
        gauss5x5_tiles_kernel << < dimGrid, dimBlock, 0 >> > (src, d_dest, w, h);
        src = d_dest;
    }
}

extern "C" void gauss5x5_gpu(float* d_src, float* d_dest, unsigned char* d_result, int w, int h, int cycles)
{
    float* src = d_src;
    while (cycles--)
    {
        gauss5x5_kernel << < get_number_of_blocks(w*h * 3, BLOCK), BLOCK, 0 >> > (src, d_dest, w, h);
        src = d_dest;
    }
}

extern "C" void gauss7x7_gpu_tiles(float* d_src, float* d_dest, unsigned char* d_result, int w, int h, int cycles)
{
    dim3 dimGrid ((w*3) / TILE_W, h / TILE_H);
    dim3 dimBlock(TILE_W+6, TILE_H+6);

    float* src = d_src;
    while (cycles--)
    {
        gauss7x7_tiles_kernel << < dimGrid, dimBlock, 0 >> > (src, d_dest, w, h);
        src = d_dest;
    }
}

extern "C" void gauss7x7_gpu(float* d_src, float* d_dest, unsigned char* d_result, int w, int h, int cycles)
{
    float* src = d_src;   
    while (cycles--)
    {
        gauss7x7_kernel << < get_number_of_blocks(w*h*3, BLOCK), BLOCK, 0 >> > (src, d_dest, w, h);
        src = d_dest;
    }
}
