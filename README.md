# GPU Image Performance Tiles Test

CUDA Perfomance test comparing image tiled and untiled Gauss filters.


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

# Results

Typical output:

```
image size: 5202 x 6502
filter                                  time                FPS
---------------------------------------------------------------------
gauss5x5_gpu_tiles                      6.85 [ms]           145.99
gauss5x5_gpu                            4.67 [ms]           214.13
gauss7x7_gpu_tiles                      10.63 [ms]          94.07
gauss7x7_gpu                            7.61 [ms]           131.41
```












