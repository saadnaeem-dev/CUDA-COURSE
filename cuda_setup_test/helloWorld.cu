#include <stdio.h>
#include <stdbool.h>
#include <cuda_runtime.h>

int main() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);

    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    // The loop is used in this example to iterate over all the available CUDA devices on the system and print the properties of each one.
    for (int i = 0; i < count; i++) {
        // Create a struct to hold the device properties
        cudaDeviceProp prop;
        // Get the device properties for device 'i'
        cudaGetDeviceProperties(&prop, i);
        printf("i %d:\n", i);

        // Print out some of the properties
        printf("CUDA Device #%d:\n", i);
        printf("  Name: %s\n", prop.name);
        printf("  Total global memory: %zu bytes\n", prop.totalGlobalMem);
        printf("  Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Registers per block: %d\n", prop.regsPerBlock);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Clock rate: %d kHz\n", prop.clockRate);
        printf("  Device Overlap: %d\n", prop.deviceOverlap);
        printf("  Can Map Host Memory: %d\n", prop.canMapHostMemory);
        printf("  Total constant memory: %zu bytes\n", prop.totalConstMem);
        printf("  Multi-processor count: %d\n", prop.multiProcessorCount);
        printf("  Maximum threads dimensions: [%d, %d, %d]\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Maximum grid size: [%d, %d, %d]\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }

    return 0;
}
