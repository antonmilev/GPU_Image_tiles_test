# GPU Image Performance Tiles Test

CUDA Convolution test comparing image tiled and untiled blur filters.


**Run the demo**
<p>This demo can be build for Windows and Linus.

**Windows**
<br>Jse the Visual Studio solution.

**Linux**
Create folder <b>build_linux</b> .
<br>cmake ../
<br>make

Requirements:
<br>CUDA Toolkit
<br>CMake (>=3.26)

# Results

Typical output:

```
Filter: 7 x 7
image size: 5200 x 6500
filter                                  time                FPS
---------------------------------------------------------------------
gauss_gpu_tiles                         9.57 [ms]           104.49
gauss_gpu                               8.82 [ms]           113.38
```


# Benchmark comparison

The figure below shows the comparison of the tiled and untiled CUDA convolution image filter for kernel radius between 1-16.

<p align="center">
  <img src="bulr_performance.png" width="500px"/>
</p>

These results shows that tiled implementation with CUDA, using effectively the shared memory is accually not better for Filters less than 11 x 11. For detailed discussion [<b>See</b>,](https://forums.developer.nvidia.com/t/how-to-use-more-efficiently-the-shared-memory-and-2d-tiles/253551/2)








