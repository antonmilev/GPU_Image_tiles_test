#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>    
#include <chrono>
#include <algorithm>
// CUDA utilities and system includes
#include <cuda_runtime.h>

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

extern "C" void gauss5x5_gpu_tiles(float* d_src, float* d_dest, unsigned char* d_result, int w, int h, int cycles);
extern "C" void gauss5x5_gpu(float* d_src, float* d_dest, unsigned char* d_result, int w, int h, int cycles);
extern "C" void gauss7x7_gpu_tiles(float* d_src, float* d_dest, unsigned char* d_result, int w, int h, int cycles);
extern "C" void gauss7x7_gpu(float* d_src, float* d_dest, unsigned char* d_result, int w, int h, int cycles);

int main(void)
{
    const int IMAGE_W = 5202 ; // pixels
    const int IMAGE_H = 6502 ;   
    const int N = 5202 * 6502 * 4;
    const int cycles = 100;

    // image is loaded as RGBA. fill with random values
    float* img_cpu = new float[N];
    for (int k = 0; k < N; k++)
        img_cpu[k] = std::rand() % 255;
  
    float* img_gpu = nullptr;
    CHECK_CUDA(cudaMalloc((void **) &img_gpu, (N * sizeof(float))));

    float* temp_gpu = nullptr;
    CHECK_CUDA(cudaMalloc((void **) &temp_gpu, (N * sizeof(float))));

    printf("image size: %d x %d\n", IMAGE_W, IMAGE_H);
    printf("%-#40s%-#20s%0s\n", "filter", "time", "FPS");
    printf("---------------------------------------------------------------------\n");

    // warmup
    gauss5x5_gpu_tiles(img_gpu, temp_gpu, nullptr, IMAGE_W, IMAGE_H, cycles);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto timeStart = std::chrono::system_clock::now();
    gauss5x5_gpu_tiles(img_gpu, temp_gpu, nullptr, IMAGE_W, IMAGE_H, cycles);
    CHECK_CUDA(cudaDeviceSynchronize());  
    auto timeEnd = std::chrono::system_clock::now();
    double dProcessingTime = (double)std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count() / cycles;
    printTime("gauss5x5_gpu_tiles", dProcessingTime);

    // warmup
    gauss5x5_gpu(img_gpu, temp_gpu, nullptr, IMAGE_W, IMAGE_H, cycles);
    CHECK_CUDA(cudaDeviceSynchronize());
    timeStart = std::chrono::system_clock::now();
    gauss5x5_gpu(img_gpu, temp_gpu, nullptr, IMAGE_W, IMAGE_H, cycles);
    CHECK_CUDA(cudaDeviceSynchronize());  
    timeEnd = std::chrono::system_clock::now();
    dProcessingTime = (double)std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count() / cycles;
    printTime("gauss5x5_gpu", dProcessingTime);

    // warmup
    gauss7x7_gpu_tiles(img_gpu, temp_gpu, nullptr, IMAGE_W, IMAGE_H, cycles);
    CHECK_CUDA(cudaDeviceSynchronize());
    timeStart = std::chrono::system_clock::now();
    gauss7x7_gpu_tiles(img_gpu, temp_gpu, nullptr, IMAGE_W, IMAGE_H, cycles);
    CHECK_CUDA(cudaDeviceSynchronize());  
    timeEnd = std::chrono::system_clock::now();
    dProcessingTime = (double)std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count() / cycles;
    printTime("gauss7x7_gpu_tiles", dProcessingTime);

    // warmup
    gauss7x7_gpu(img_gpu, temp_gpu, nullptr, IMAGE_W, IMAGE_H, cycles);
    CHECK_CUDA(cudaDeviceSynchronize());
    timeStart = std::chrono::system_clock::now();
    gauss7x7_gpu(img_gpu, temp_gpu, nullptr, IMAGE_W, IMAGE_H, cycles);
    CHECK_CUDA(cudaDeviceSynchronize());  
    timeEnd = std::chrono::system_clock::now();
    dProcessingTime = (double)std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count() / cycles;
    printTime("gauss7x7_gpu", dProcessingTime);


    delete img_cpu;
    cudaFree(img_gpu);
    cudaFree(temp_gpu);

    return 0;
}
