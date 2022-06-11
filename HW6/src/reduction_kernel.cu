#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "reduction.h"

void allocateDeviceMemory(void** M, int size)
{
    cudaError_t err = cudaMalloc(M, size);
    assert(err==cudaSuccess);
}


void deallocateDeviceMemory(void* M)
{
    cudaError_t err = cudaFree(M);
    assert(err==cudaSuccess);
}

void cudaMemcpyToDevice(void* dst, void* src, int size) {
    cudaError_t err = cudaMemcpy((void*)dst, (void*)src, size, cudaMemcpyHostToDevice);
    assert(err==cudaSuccess);
}

void cudaMemcpyToHost(void* dst, void* src, int size) {
    cudaError_t err = cudaMemcpy((void*)dst, (void*)src, size, cudaMemcpyDeviceToHost);
    assert(err==cudaSuccess);
}

void reduce_ref(const int* const g_idata, int* const g_odata, const int n) {
    for (int i = 0; i < n; i++)
        g_odata[0] += g_idata[i];
}

__global__ void reduceV1(const int* const d_idata, int* const d_odata, const int n, int reduceSize) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < reduceSize) {
        if (n == reduceSize) {
            sdata[tid] = d_idata[i];
        } else {
            sdata[tid] = d_odata[i];
        }
        __syncthreads();


        int maxS = blockDim.x;
        if (blockDim.x > reduceSize) {
            maxS = reduceSize;
        }
        for (unsigned int s=1; s < maxS; s*=2) {
            if (tid % (2*s) == 0) {
                sdata[tid] += sdata[tid+s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            d_odata[blockIdx.x] = sdata[0];
        }
    }
}

__global__ void reduceV2(const int* const d_idata, int* const d_odata, const int n, int reduceSize) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < reduceSize) {
        if (n == reduceSize) {
            sdata[tid] = d_idata[i];
        } else {
            sdata[tid] = d_odata[i];
        }
        __syncthreads();

        int maxS = blockDim.x;
        if (blockDim.x > reduceSize) {
            maxS = reduceSize;
        }
        for (unsigned int s=1; s < maxS; s*=2) {
            int index = 2*s*tid;
            if (index < blockDim.x) {
                sdata[index] += sdata[index+s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            d_odata[blockIdx.x] = sdata[0];
        }
    }
}

__global__ void reduceV3(const int* const d_idata, int* const d_odata, const int n, int reduceSize) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < reduceSize) {
        if (n == reduceSize) {
            sdata[tid] = d_idata[i];
        } else {
            sdata[tid] = d_odata[i];
        }
        __syncthreads();

        int maxS = blockDim.x/2;
        if (blockDim.x > reduceSize) {
            maxS = reduceSize/2;
        }
        for (unsigned int s=maxS; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid+s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            d_odata[blockIdx.x] = sdata[0];
        }
    }
}

__global__ void reduceV4(const int* const d_idata, int* const d_odata, const int n, int reduceSize) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    if (i < reduceSize) {
        if (n == reduceSize) {
            if (i+blockDim.x < reduceSize)
                sdata[tid] = d_idata[i] + d_idata[i + blockDim.x];
            else
                sdata[tid] = d_idata[i];
        } else {
            if (i+blockDim.x < reduceSize)
                sdata[tid] = d_odata[i] + d_odata[i + blockDim.x];
            else
                sdata[tid] = d_odata[i];
        }
        __syncthreads();

        int maxS = blockDim.x/2;
        if (blockDim.x > reduceSize) {
            maxS = reduceSize/2;
        }
        for (unsigned int s=maxS; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid+s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            d_odata[blockIdx.x] = sdata[0];
        }
    }
}

__global__ void reduceV5(const int* const d_idata, int* const d_odata, const int n, int reduceSize) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    if (n == reduceSize) {
        if (i+blockDim.x < reduceSize)
            sdata[tid] = d_idata[i] + d_idata[i + blockDim.x];
        else
            sdata[tid] = d_idata[i];
    } else {
        if (i+blockDim.x < reduceSize)
            sdata[tid] = d_odata[i] + d_odata[i + blockDim.x];
        else
            sdata[tid] = d_odata[i];
    }
    __syncthreads();


    for (unsigned int s=blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    if (reduceSize > 64) {
        if (tid < 32) {
            sdata[tid] += sdata[tid + 32];
            __syncwarp();
            sdata[tid] += sdata[tid + 16];
            __syncwarp();
            sdata[tid] += sdata[tid + 8];
            __syncwarp();
            sdata[tid] += sdata[tid + 4];
            __syncwarp();
            sdata[tid] += sdata[tid + 2];
            __syncwarp();
            sdata[tid] += sdata[tid + 1];
            __syncwarp();
        }
    } else {
        if (tid < 32) {
            if (tid + 32 < reduceSize) {
                sdata[tid] += sdata[tid + 32];
                __syncwarp();
            }
            if (tid + 16 < reduceSize) {
                sdata[tid] += sdata[tid + 16];
                __syncwarp();
            }
            if (tid + 8 < reduceSize) {
                sdata[tid] += sdata[tid + 8];
                __syncwarp();
            }
            if (tid + 4 < reduceSize) {
                sdata[tid] += sdata[tid + 4];
                __syncwarp();
            }
            if (tid + 2 < reduceSize) {
                sdata[tid] += sdata[tid + 2];
                __syncwarp();
            }
            if (tid + 1 < reduceSize) {
                sdata[tid] += sdata[tid + 1];
                __syncwarp();
            }
        }
    }

    if (tid == 0) {
        d_odata[blockIdx.x] = sdata[0];
    }
    
}

template <unsigned int blockSize>
__global__ void reduceV6(const int* const d_idata, int* const d_odata, const int n, int reduceSize) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    if (n == reduceSize) {
        if (i+blockDim.x < reduceSize)
            sdata[tid] = d_idata[i] + d_idata[i + blockDim.x];
        else
            sdata[tid] = d_idata[i];
    } else {
        if (i+blockDim.x < reduceSize)
            sdata[tid] = d_odata[i] + d_odata[i + blockDim.x];
        else
            sdata[tid] = d_odata[i];
    }
    __syncthreads();

    if (blockSize >= 1024) {
        if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }

    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        __syncwarp();
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        __syncwarp();
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        __syncwarp();
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
        __syncwarp();
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
        __syncwarp();
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
        __syncwarp();
    }

    if (tid == 0) {
        d_odata[blockIdx.x] = sdata[0];
    }
}

template <unsigned int blockSize>
__global__ void reduceV7(const int* const d_idata, int* const d_odata, const int n, int reduceSize) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;

    int cnt = 0;
    if (n == reduceSize) {
        while (i< reduceSize) {
            if (i + blockSize < reduceSize)
                sdata[tid] += d_idata[i] + d_idata[i + blockSize];
            else
                sdata[tid] = d_idata[i];
            i += gridSize;
            cnt += 1;
        }
    } else {
        while (i< reduceSize) {
            if (i + blockSize < reduceSize)
                sdata[tid] += d_odata[i] + d_odata[i + blockSize];
            else
                sdata[tid] += d_odata[i];
            i += gridSize;
            cnt += 1;
        }
    }
    
    __syncthreads();

    if (blockSize >= 1024) {
        if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }

    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        __syncwarp();
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        __syncwarp();
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        __syncwarp();
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
        __syncwarp();
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
        __syncwarp();
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
        __syncwarp();
    }

    if (tid == 0) {
        d_odata[blockIdx.x] = sdata[0];
    }
}


void reduce_optimize(const int* const g_idata, int* const g_odata, const int* const d_idata, int* const d_odata, const int n) {
    // TODO: Implement your CUDA code
    // Reduction result must be stored in d_odata[0] 
    // You should run the best kernel in here but you must remain other kernels as evidence.
    int blockSize = 1024;
    if (n >= 1024)
        blockSize = 1024;
    else if (n >= 512)
        blockSize = 512;
    else if (n >= 256)
        blockSize = 256;
    else if (n >= 128)
        blockSize = 128;
    else if (n >= 64)
        blockSize = 64;
    else if (n >= 32)
        blockSize = 32;
    else if (n >= 16)
        blockSize = 16;
    else if (n >= 8)
        blockSize = 8;
    else if (n >= 4)
        blockSize = 4;
    else
        blockSize = 2;

    int reduceSize = n;

    while(reduceSize > 1) {
        int dimGrid = (int)ceil((double)reduceSize/(blockSize));
        if (dimGrid > 128) {
            dimGrid = dimGrid / 128;
        }
        switch (blockSize)
            {
                case 1024:
                    reduceV7<1024><<<dimGrid, blockSize, blockSize*4>>>(d_idata, d_odata, n, reduceSize);
                    break;
                case 512:
                    reduceV7<512><<<dimGrid, blockSize, blockSize*4>>>(d_idata, d_odata, n, reduceSize);
                    break;
                case 256:
                    reduceV7<256><<<dimGrid, blockSize, blockSize*4>>>(d_idata, d_odata, n, reduceSize);
                    break;
                case 128:
                    reduceV7<128><<<dimGrid, blockSize, blockSize*4>>>(d_idata, d_odata, n, reduceSize);
                    break;
                case 64:
                    reduceV7<64><<<dimGrid, blockSize, blockSize*4>>>(d_idata, d_odata, n, reduceSize);
                    break;
                case 32:
                    reduceV7<32><<<dimGrid, blockSize, blockSize*4>>>(d_idata, d_odata, n, reduceSize);
                    break;
                case 16:
                    reduceV7<16><<<dimGrid, blockSize, blockSize*4>>>(d_idata, d_odata, n, reduceSize);
                    break;
                case 8:
                    reduceV7<8><<<dimGrid, blockSize, blockSize*4>>>(d_idata, d_odata, n, reduceSize);
                    break;
                case 4:
                    reduceV7<4><<<dimGrid, blockSize, blockSize*4>>>(d_idata, d_odata, n, reduceSize);
                    break;
                case 2:
                    reduceV7<2><<<dimGrid, blockSize, blockSize*4>>>(d_idata, d_odata, n, reduceSize);
                    break;
                default:
                    break;
            }
        reduceSize = dimGrid;
        if (reduceSize >= 1024)
            blockSize = 1024;
        else if (reduceSize >= 512)
            blockSize = 512;
        else if (reduceSize >= 256)
            blockSize = 256;
        else if (reduceSize >= 128)
            blockSize = 128;
        else if (reduceSize >= 64)
            blockSize = 64;
        else if (reduceSize >= 32)
            blockSize = 32;
        else if (reduceSize >= 16)
            blockSize = 16;
        else if (reduceSize >= 8)
            blockSize = 8;
        else if (reduceSize >= 4)
            blockSize = 4;
        else
            blockSize = 2;
    }
}
