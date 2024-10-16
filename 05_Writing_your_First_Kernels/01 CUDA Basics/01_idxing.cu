#include <stdio.h>

__global__ void whoami(void) {
    int block_id = // trying to find where we are in the apartment complex. you go hig, then go deep in the panes and finally stop at x dim offset. this is how you find your block id. read this starting from bottom (panes deep z, how high y, stop at x offset to get the block id)
        blockIdx.x +    // apartment number on this floor (points across)
        blockIdx.y * gridDim.x +    // gridDim.x is floor number in this building (blockIdx.y rows high)
        blockIdx.z * gridDim.x * gridDim.y;   // building number in this city (panes deep) gridDim.x = x dimesion in the grid, blockIdx.z = depth. in the pane which has depth, first get x, y and however deep the z is

    int block_offset = // read from bottom to build up
        block_id * // times our apartment number. the below line gets how many threads before your block (apt number) until you get to your block
        blockDim.x * blockDim.y * blockDim.z; // total threads per block (people per apartment). blockDim.x = threads in x, blockDim.y threads in y, blockDim.z threads in z we multiply these together to get the threads per block, this line is thread level off set before we build up to our block.

    int thread_offset = // same analogy as block_id except for thread_offset
        threadIdx.x +  // offset in x
        threadIdx.y * blockDim.x + // offset for y accounting for x dimension
        threadIdx.z * blockDim.x * blockDim.y; // offset for z accounting for both x and y dimensions
        // threadIdx.x: The thread's position in the x dimension within the block.
        // threadIdx.y: The thread's position in the y dimension within the block.
        // threadIdx.z: The thread's position in the z dimension within the block.
        // blockDim.x: The size (number of threads) in the x dimension of the block.
        // blockDim.y: The size (number of threads) in the y dimension of the block.
        // threadIdx.x: This is the thread's position within the x dimension. It directly contributes to the offset.
        // threadIdx.y * blockDim.x: This accounts for all the threads in the y dimension before the current y position.
        // For every increment in threadIdx.z, you have blockDim.x * blockDim.y threads in the entire 2D plane of the x and y dimensions that have already been counted. So, you multiply threadIdx.z by the total number of threads in the x and y dimensions to shift the x and y offsets for threads in previous z positions.
        // Summary: The formula effectively flattens the 3D thread indices while accounting for offset (x, y, z) into a 1D linear index by accounting for the size of the block in each dimension. The formula works as

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
