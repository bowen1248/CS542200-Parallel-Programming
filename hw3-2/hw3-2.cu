#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define TILE_SIZE 128 // Should be multiplication of BLOCK_SIZE
#define BLOCK_SIZE 32
#define SHORT_INF 65535
#define INF 1073741823 // 2^30 - 1

__device__ void block_APSP(u_int16_t* C, u_int16_t* A, u_int16_t* B, int x, int y) {
    for (int k = 0; k < TILE_SIZE; k++) {
        for (int i = 0; i < TILE_SIZE; i += 32) {
        // printf("%d %d %d %d %d %d\n", blockIdx.y, blockIdx.x, y, x, A[y * BLOCK_SIZE + k], B[k * BLOCK_SIZE + x]);
        // uint32_t tmp = B[x * BLOCK_SIZE + k] << 16 + B[(x + 1) * BLOCK_SIZE + k];
            C[(y + i) * TILE_SIZE + x] = min(C[(y + i) * TILE_SIZE + x], A[(y + i) * TILE_SIZE + k] + B[k * TILE_SIZE + x]);
            C[(y + i) * TILE_SIZE + x + 32] = min(C[(y + i) * TILE_SIZE + x + 32], A[(y + i) * TILE_SIZE + k] + B[k * TILE_SIZE + (x + 32)]);
            C[(y + i) * TILE_SIZE + x + 64] = min(C[(y + i) * TILE_SIZE + x + 64], A[(y + i) * TILE_SIZE + k] + B[k * TILE_SIZE + (x + 64)]);
            C[(y + i) * TILE_SIZE + x + 96] = min(C[(y + i) * TILE_SIZE + x + 96], A[(y + i) * TILE_SIZE + k] + B[k * TILE_SIZE + (x + 96)]);
        }
        __syncthreads();
    }
}

__global__ void stage1(short2 *devMat, int startIdx, int n) {
    printf("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    // __shared__ u_int16_t mat[TILE_SIZE * TILE_SIZE];

    // // Load adj. matrix from global memory to shared memory
    // int cursorX = startIdx + threadIdx.x;
    // int cursorY = startIdx + threadIdx.y;
    // for (int i = 0; i < TILE_SIZE; i += 32) {
    //     mat[(threadIdx.y + i) * TILE_SIZE + threadIdx.x] = devMat[(cursorY + i) * n + cursorX];
    //     mat[(threadIdx.y + i) * TILE_SIZE + (threadIdx.x + 32)] = devMat[(cursorY + i) * n + (cursorX + 32)];
    //     mat[(threadIdx.y + i) * TILE_SIZE + threadIdx.x + 64] = devMat[(cursorY + i) * n + cursorX + 64];
    //     mat[(threadIdx.y + i) * TILE_SIZE + (threadIdx.x + 96)] = devMat[(cursorY + i) * n + (cursorX + 96)];
    // }
    // __syncthreads();
    
    // // Perform APSP on the block
    // block_APSP((u_int16_t *) mat, (u_int16_t *) mat, (u_int16_t *) mat, threadIdx.x, threadIdx.y);

    // // Write data back to global memory
    // for (int i = 0; i < TILE_SIZE; i += 32) {
    //     devMat[(cursorY + i) * n + cursorX] = mat[(threadIdx.y + i) * TILE_SIZE + threadIdx.x];
    //     devMat[(cursorY + i) * n + (cursorX + 32)] = mat[(threadIdx.y + i) * TILE_SIZE + (threadIdx.x + 32)];
    //     devMat[(cursorY + i) * n + cursorX + 64] = mat[(threadIdx.y + i) * TILE_SIZE + threadIdx.x + 64];
    //     devMat[(cursorY + i) * n + (cursorX + 96)] = mat[(threadIdx.y + i) * TILE_SIZE + (threadIdx.x + 96)];
    // }
    // devMat[cursorY * n + cursorX] = mat[threadIdx.y * BLOCK_SIZE + threadIdx.x];
    __syncthreads();
}

// __global__ void stage2_row(int *devMat, int startIdx, int n) {
//     // Load adj. matrix from global memory to shared memory
//     int cursorX = blockIdx.x * BLOCK_SIZE + threadIdx.x;
//     int cursorY = startIdx + threadIdx.y;

//     if ((blockIdx.x * BLOCK_SIZE) == startIdx)
//         return;

//     // C[BLOCK_SIZE * BLOCK_SIZE], B[BLOCK_SIZE * BLOCK_SIZE]
//     __shared__ int mat[2 * BLOCK_SIZE * BLOCK_SIZE];

//     // Load adj. matrixs from global memory to shared memory
//     mat[threadIdx.y * BLOCK_SIZE + threadIdx.x] = devMat[cursorY * n + cursorX];
//     mat[threadIdx.y * BLOCK_SIZE + threadIdx.x + BLOCK_SIZE * BLOCK_SIZE] = devMat[cursorY * n + startIdx + threadIdx.x];
//     __syncthreads();

//     // Perform APSP on the block
//     block_APSP(mat, &mat[BLOCK_SIZE * BLOCK_SIZE], mat, threadIdx.x, threadIdx.y);

//     // Write data back to global memory
//     devMat[cursorY * n + cursorX] = mat[threadIdx.y * BLOCK_SIZE + threadIdx.x];
//     __syncthreads();
// }

// __global__ void stage2_col(int *devMat, int startIdx, int n) {
//     // Load adj. matrix from global memory to shared memory
//     int cursorX = startIdx + threadIdx.x;
//     int cursorY = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    
//     if ((blockIdx.x * BLOCK_SIZE) == startIdx)
//         return;

//     // C[BLOCK_SIZE * BLOCK_SIZE], B[BLOCK_SIZE * BLOCK_SIZE]
//     __shared__ int mat[2 * BLOCK_SIZE * BLOCK_SIZE];

//     // Load adj. matrixs from global memory to shared memory
//     mat[threadIdx.y * BLOCK_SIZE + threadIdx.x] = devMat[cursorY * n + cursorX];
//     mat[threadIdx.y * BLOCK_SIZE + threadIdx.x + BLOCK_SIZE * BLOCK_SIZE] = devMat[(startIdx + threadIdx.y) * n + cursorX];
//     __syncthreads();

//     // Perform APSP on the block
//     block_APSP(mat, mat, &mat[BLOCK_SIZE * BLOCK_SIZE], threadIdx.x, threadIdx.y);

//     // Write data back to global memory
//     devMat[cursorY * n + cursorX] = mat[threadIdx.y * BLOCK_SIZE + threadIdx.x];
//     __syncthreads();
// }

// __global__ void stage3(int *devMat, int startIdx, int n) {
//     // Load adj. matrix from global memory to shared memory
//     int cursorX = blockIdx.x * BLOCK_SIZE + threadIdx.x;
//     int cursorY = blockIdx.y * BLOCK_SIZE + threadIdx.y;
//     // long long int t1 = clock64();
//     if ((blockIdx.x * BLOCK_SIZE) == startIdx || (blockIdx.y * BLOCK_SIZE) == startIdx)
//         return;

//     // C[BLOCK_SIZE * BLOCK_SIZE], B[BLOCK_SIZE * BLOCK_SIZE], A[BLOCK_SIZE * BLOCK_SIZE]
//     __shared__ int mat[3 * BLOCK_SIZE * BLOCK_SIZE];

//     // Load adj. matrixs from global memory to shared memory
//     mat[threadIdx.y * BLOCK_SIZE + threadIdx.x] = devMat[cursorY * n + cursorX];
//     mat[threadIdx.y * BLOCK_SIZE + threadIdx.x + BLOCK_SIZE * BLOCK_SIZE] = devMat[cursorY * n + startIdx + threadIdx.x];
//     mat[threadIdx.y * BLOCK_SIZE + threadIdx.x + 2 * BLOCK_SIZE * BLOCK_SIZE] = devMat[(startIdx + threadIdx.y) * n + cursorX];
//     __syncthreads();
//     // long long int t2 = clock64();
//     // printf("%lld ", (t2 - t1));
//     // Perform APSP on the block
//     block_APSP(mat, &mat[BLOCK_SIZE * BLOCK_SIZE], &mat[2 * BLOCK_SIZE * BLOCK_SIZE], threadIdx.x, threadIdx.y);
//     // long long int t3 = clock64();
//     // printf("%lld ", (t3 - t2));
//     // Write data back to global memory
//     devMat[cursorY * n + cursorX] = mat[threadIdx.y * BLOCK_SIZE + threadIdx.x];
//     __syncthreads();
//     // long long int t4 = clock64();
//     // printf("%lld ", (t4 - t3));
// }

int main(int argc, char **argv) {
    freopen("log.txt","w",stdout);
    /* detect how many CPUs are available */
    // cpu_set_t cpu_set;
    // int ncpus;
    // sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    // ncpus = CPU_COUNT(&cpu_set);

    // // Thread handlers
    // pthread_t threads[ncpus];

    /* argument parsing */
    assert(argc == 3);
    const char *inputFile = argv[1];
    const char *outputFile = argv[2];

    FILE *inFp = fopen(inputFile, "rb");
    FILE *outFp = fopen(outputFile, "wb");
    if( inFp == NULL ) {
        fprintf(stderr, "Couldn't open %s: %s\n", inputFile, strerror(errno));
        exit(1);
    }
    int verticesTotal;
    int edgesTotal;

    // Get input vertices and edges number
    size_t _;
    _ = fread(&verticesTotal, sizeof(int), 1, inFp);
    _ = fread(&edgesTotal, sizeof(int), 1, inFp);
    static int block_dim = (verticesTotal + TILE_SIZE - 1) / TILE_SIZE;
    static int n = TILE_SIZE * block_dim;

    // Create adjanency matrix
    uint16_t *adjMat = (uint16_t *) malloc(n * n * sizeof(uint16_t));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j)
                adjMat[i * n + j] = 0;
            else
                adjMat[i * n + j] = SHORT_INF;
        }
    }

    // Put edges into adjanency matrix
    int tmp[3];
    for (int i = 0; i < edgesTotal; i++) {
        _ = fread(&tmp, sizeof(int), 3, inFp);
        adjMat[tmp[0] * n + tmp[1]] = (uint16_t) tmp[2];
    }
    fclose(inFp);

    // Print input graph
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         if (adjMat[i * n + j] != 65535)
    //             std::cout << adjMat[i * n + j] << " ";
    //         else
    //             std::cout << "INF" << " ";
    //     }
    //     std::cout << std::endl;
    // }

    short2* device_adjMat;
    cudaMalloc((void **) &device_adjMat, n * n * sizeof(uint16_t));

    // Put adjanency matrix to GPU
    cudaMemcpy(device_adjMat, adjMat, n * n * sizeof(uint16_t), cudaMemcpyHostToDevice);

    // stages
    stage1<<< 1, dim3(BLOCK_SIZE, BLOCK_SIZE) >>> (device_adjMat, 0, n);
        // stage2_row<<< block_dim, dim3(BLOCK_SIZE, BLOCK_SIZE), 0 >>> (device_adjMat, k_start, n);
        // stage2_col<<< block_dim, dim3(BLOCK_SIZE, BLOCK_SIZE), 0 >>> (device_adjMat, k_start, n);
        // stage3<<< dim3(block_dim, block_dim), dim3(BLOCK_SIZE, BLOCK_SIZE), 0 >>> (device_adjMat, k_start, n);
    cudaDeviceSynchronize();
    // output
    cudaMemcpy(adjMat, device_adjMat, n * n * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    // Print input graph
    for (int i = 0; i < verticesTotal; i++) {
        for (int j = 0; j < verticesTotal; j++) {
            if (adjMat[i * n + j] != 1073741823)
                std::cout << adjMat[i * n + j] << " ";
            else
                std::cout << "INF" << " ";
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < verticesTotal; i++) {
        for (int j = 0; j < verticesTotal; j++) {
            int tmp = (u_int32_t) adjMat[i * n + j];
            fwrite(&tmp, sizeof(u_int32_t), 1, outFp);
        }
    }
    fclose(outFp);

    return 0;
}