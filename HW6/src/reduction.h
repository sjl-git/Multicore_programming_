#ifndef __REDUCTION_H__
#define __REDUCTION_H__

void reduce_ref(const int* const g_idata, int* const g_odata, const int n);

void reduce_optimize(const int* const g_idata, int* const g_odata, const int* const d_idata, int* const d_odata, const int n);

void allocateDeviceMemory(void** M, int width);
void deallocateDeviceMemory(void* M);

void cudaMemcpyToDevice(void* dst, void* src, int size);
void cudaMemcpyToHost(void* dst, void* src, int size);

#endif