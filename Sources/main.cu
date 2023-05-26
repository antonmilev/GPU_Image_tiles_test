#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <iostream>

#ifndef RF
#define RF 3
#endif
const int R = RF;
const int D = 2*R+1;
const int LF = D*D;
const float FC = 1.0f/LF;
const int TILE_W = 32;
const int TILE_H = 16;

__constant__ float kernel_blur[LF] ;

__global__ void conv_kernel(const float* __restrict__ in, float * __restrict__ out, int w, int h)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int x = idx % w;
    unsigned int c = idx / w;   
    unsigned int w3 = 3 * w;

    if ((idx >= w3) || (y >= h))
        return;

    unsigned int id = y * w3 + c*w + x;   // image index RGB/NCWH

    if (x <= R-1 || y <= R-1 || x >= w-R || y >= h-R)
    {
        out[id] = in[id];
        return;
    }
    
    float sum = 0;
    for(int i = -R; i <=R; i++)
    for(int j = -R; j <=R; j++)
    {
        sum += in[id + j + w3*i] *kernel_blur[(i + R) * D + (j + R)];
    }

    out[id] = sum;
}

__global__ void conv_tiles_kernel(const float* __restrict__ in, float * __restrict__ out, int w, int h)
{
    const int BLOCK_W = (TILE_W + 2*R);
    const int BLOCK_H = (TILE_H + 2*R);
    __shared__ float smem[BLOCK_H][BLOCK_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int idy = blockIdx.y*blockDim.y+threadIdx.y;

    unsigned int x = idx % w;
    unsigned int c = idx / w;
    unsigned int y = idy;

    int w3 = w * 3;
    int idx_out = y * w3 + c * w + x;

    // load central block
    smem[ty+R][tx+R] = in[idx_out]; 
    // load left bar
    if ((x >= R) && (tx < R)) smem[ty+R][tx] = in[idx_out-R];
    // load right bar
    if ((x < w-R) && (tx >= blockDim.x-R) ) smem[ty+R][tx+2*R] = in[idx_out+R];
    // load top bar
    if ((y > R-1) && (ty < R)) smem[ty][tx+R] = in[idx_out-R*w3];
    // load bottom bar
    if ((y < h-R) && (ty >=blockDim.y-R)) smem[ty+2*R][tx+R] = in[idx_out+R*w3];
    // load UL corner
    if ((x > R-1) && (y > R-1) && (tx < R) && (ty < R)) smem[ty][tx] = in[idx_out - R*w3 - R];
    // load UR corner
    if ((x < w-R) && (y > R-1) && (tx >= blockDim.x-R) && (ty < R) ) smem[ty][tx+2*R] = in[idx_out - R * w3 + R];
    // load LL corner
    if ((x > R-1) && (y < h-R) && (tx < R) && (ty >= blockDim.y-R) ) smem[ty+2*R][tx] = in[idx_out + R * w3 - R];
    // load LR corner
    if ((x < w-R) && (y < h-R) && (tx >=blockDim.x-R) && (ty >= blockDim.y-R)) smem[ty+2*R][tx+2*R] = in[idx_out + R * w3 + R];
    
    __syncthreads();

    if (x <= R-1 || y <= R-1 || x >= w-R || y >= h-R)
    {
        out[idx_out] = in[idx_out];
        return;
    }

    float sum = 0;
    for(int i = -R; i <=R; i++)
    for(int j = -R; j <=R; j++)
    {
        sum += smem[ty+R+i][tx+R+j] * kernel_blur[(i+R)*D+(j+R)];
    }
    out[idx_out] = sum;
}

void blur_gpu_tiles(float* d_src, float* d_dest, unsigned char* d_result, int w, int h, int cycles)
{
    dim3 dimGrid ((w*3) / TILE_W, h / TILE_H);
    dim3 dimBlock(TILE_W, TILE_H);

    float* src = d_src,*dst = d_dest, *tmp = d_dest;
    while (cycles--)
    {
        conv_tiles_kernel << < dimGrid, dimBlock>> > (src, dst, w, h);
        tmp = dst;
        dst = src;
        src = tmp;
    }
}

void blur_gpu(float* d_src, float* d_dest, unsigned char* d_result, int w, int h, int cycles)
{
    dim3 dimGrid ((w*3) / TILE_W, h / TILE_H);
    dim3 dimBlock(TILE_W, TILE_H);
    float* src = d_src,*dst = d_dest, *tmp = d_dest;
    while (cycles--)
    {
        conv_kernel << < dimGrid, dimBlock>> > (src, dst, w, h);
        tmp = dst;
        dst = src;
        src = tmp;
    }
}

void check_error(cudaError_t status)
{
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        printf("\n CUDA Error: %s\n", s);
        getchar();
    }
    if (status2 != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status2);
        printf("\n CUDA Error Prev: %s\n", s);
        getchar();
    }
}

void printTime(const char* name, double time)
{
    float fps = 1000 / time;
    printf("%-#40s",name);
    char tmp[32];
    sprintf(tmp, "%0.2f [ms]", time);
    printf("%-#20s%0.2f\n", tmp, fps);
}

#define CHECK_CUDA(X) check_error((cudaError_t)X);

int main()
{
    const int IMAGE_W = 5200 ; // pixels
    const int IMAGE_H = 6500 ;
    const int N = IMAGE_W * IMAGE_H * 3;
    const int cycles = 100;
    printf("Filter: %d x %d\n", D, D);

    // image is loaded as RGB, fill with random values
    float* img_cpu = new float[N];
    float *t1 = new float[N];
    float *t2 = new float[N];

    // fill kernel
    float* kernel_cpu = new float[LF];
    for (int k = 0; k < LF; k++)
        kernel_cpu[k] = 1.0f/LF;

    unsigned char* kernel_gpu_adr;
    CHECK_CUDA(cudaGetSymbolAddress((void **)&kernel_gpu_adr, kernel_blur));
    cudaMemcpy(kernel_gpu_adr, kernel_cpu, LF*sizeof(float),cudaMemcpyHostToDevice);

    for (int k = 0; k < N; k++)
        img_cpu[k] = std::rand() % 255;

    float* img_gpu = nullptr;
    CHECK_CUDA(cudaMalloc((void **) &img_gpu, (N * sizeof(float))));
    cudaMemcpy(img_gpu, img_cpu, (N*sizeof(float)), cudaMemcpyHostToDevice);
    float* temp_gpu = nullptr;
    CHECK_CUDA(cudaMalloc((void **) &temp_gpu, (N * sizeof(float))));

    printf("image size: %d x %d\n", IMAGE_W, IMAGE_H);
    printf("%-#40s%-#20s%0s\n", "filter", "time", "FPS");
    printf("---------------------------------------------------------------------\n");

    // warmup
    blur_gpu_tiles(img_gpu, temp_gpu, nullptr, IMAGE_W, IMAGE_H, 1);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaMemcpy(t1, temp_gpu, N*sizeof(float),cudaMemcpyDeviceToHost);
    auto timeStart = std::chrono::system_clock::now();
    blur_gpu_tiles(img_gpu, temp_gpu, nullptr, IMAGE_W, IMAGE_H, cycles);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto timeEnd = std::chrono::system_clock::now();
    double dProcessingTime = (double)std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count() / cycles;
    printTime("gauss_gpu_tiles", dProcessingTime);

    // warmup
    cudaMemcpy(img_gpu, img_cpu, (N*sizeof(float)), cudaMemcpyHostToDevice);
    blur_gpu(img_gpu, temp_gpu, nullptr, IMAGE_W, IMAGE_H, 1);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaMemcpy(t2, temp_gpu, N*sizeof(float),cudaMemcpyDeviceToHost);
    timeStart = std::chrono::system_clock::now();
    blur_gpu(img_gpu, temp_gpu, nullptr, IMAGE_W, IMAGE_H, cycles);
    CHECK_CUDA(cudaDeviceSynchronize());
    timeEnd = std::chrono::system_clock::now();
    dProcessingTime = (double)std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count() / cycles;
    printTime("gauss_gpu", dProcessingTime);
   
    for (int i = 0; i < N; i++)
    {
        if (t1[i] != t2[i])
        {
            std::cout << "mismatch at: " << i << " t1: " << t1[i] << " t2: " << t2[i] << std::endl; 
        }
    }

    delete img_cpu;
    delete t1;
    delete t2;
    cudaFree(img_gpu);
    cudaFree(temp_gpu);

    return 0;
}

