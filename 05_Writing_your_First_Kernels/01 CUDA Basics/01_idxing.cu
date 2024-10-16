#include <stdio.h>

__global__ void whoami(void) {
    int block_id = // trying to find where we are in the apartment complex. you go hig, then go deep in the panes and finally stop at x dim offset. this is how you find your block id. read this starting from bottom (panes deep z, how high y, stop at x offset to get the block id)
        blockIdx.x +    // apartment number on this floor (points across)
        blockIdx.y * gridDim.x +    // gridDim.x is floor number in this building (blockIdx.y rows high)
        blockIdx.z * gridDim.x * gridDim.y;   // building number in this city (panes deep) gridDim.x = x dimesion in the grid, blockIdx.z = depth. in the pane which has depth, first get x, y and however deep the z is

    int block_offset =
        block_id * // times our apartment number
        blockDim.x * blockDim.y * blockDim.z; // total threads per block (people per apartment)

    int thread_offset =
        threadIdx.x +  
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    int id = block_offset + thread_offset; // global person id in the entire apartment complex

    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
    // printf("blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(int argc, char **argv) {
    const int b_x = 2, b_y = 3, b_z = 4; // grid/block dim with x, y, z. inside the grid x dim = 2, the height dim is 3, and depth z = 4 (A grid volume of this shape)
    const int t_x = 4, t_y = 4, t_z = 4; // block dim the max warp size is 32, so 
    // we will get 2 warp of 32 threads per block (In each individual block inside that grid, we have these thread dims i.e. each block is 4 long, 4 high, and 4 deep)

    int blocks_per_grid = b_x * b_y * b_z; // base times width times height
    int threads_per_block = t_x * t_y * t_z;

    printf("%d blocks/grid\n", blocks_per_grid); // total number of these
    printf("%d threads/block\n", threads_per_block);
    printf("%d total threads\n", blocks_per_grid * threads_per_block);

    dim3 blocksPerGrid(b_x, b_y, b_z); // 3d cube of shape 2*3*4 = 24. dim3 type is specific to cuda
    dim3 threadsPerBlock(t_x, t_y, t_z); // 3d cube of shape 4*4*4 = 64

    whoami<<<blocksPerGrid, threadsPerBlock>>>(); // we plug these into our kernel = whoami
    cudaDeviceSynchronize(); // to ensure everything is caught up before we continue
}
