#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define BLOCK_SIZE 32
#define INF 1073741823 // 2^30 - 1

__device__ void block_APSP(int* C, int* A, int* B, int x, int y) {
    for (int k = 0; k < BLOCK_SIZE; k++) {
        // printf("%d %d %d %d %d %d\n", blockIdx.y, blockIdx.x, y, x, A[y * BLOCK_SIZE + k], B[k * BLOCK_SIZE + x]);
        C[y * BLOCK_SIZE + x] = min(C[y * BLOCK_SIZE + x], A[y * BLOCK_SIZE + k] + B[k * BLOCK_SIZE + x]);
        __syncthreads();
    }
}

__global__ void stage1(int *devMat, int startIdx, int n) {
    __shared__ int mat[BLOCK_SIZE * BLOCK_SIZE];

    // Load adj. matrix from global memory to shared memory
    int cursorX = startIdx + threadIdx.x;
    int cursorY = startIdx + threadIdx.y;
    mat[threadIdx.y * BLOCK_SIZE + threadIdx.x] = devMat[cursorY * n + cursorX];
    __syncthreads();

    // Perform APSP on the block
    block_APSP(mat, mat, mat, threadIdx.x, threadIdx.y);

    // Write data back to global memory
    devMat[cursorY * n + cursorX] = mat[threadIdx.y * BLOCK_SIZE + threadIdx.x];
    __syncthreads();
}

__global__ void stage2_row(int *devMat, int startIdx, int n) {
    // Load adj. matrix from global memory to shared memory
    int cursorX = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int cursorY = startIdx + threadIdx.y;

    if ((blockIdx.x * BLOCK_SIZE) == startIdx)
        return;

    // C[BLOCK_SIZE * BLOCK_SIZE], B[BLOCK_SIZE * BLOCK_SIZE]
    __shared__ int mat[2 * BLOCK_SIZE * BLOCK_SIZE];

    // Load adj. matrixs from global memory to shared memory
    mat[threadIdx.y * BLOCK_SIZE + threadIdx.x] = devMat[cursorY * n + cursorX];
    mat[threadIdx.y * BLOCK_SIZE + threadIdx.x + BLOCK_SIZE * BLOCK_SIZE] = devMat[cursorY * n + startIdx + threadIdx.x];
    __syncthreads();

    // Perform APSP on the block
    block_APSP(mat, &mat[BLOCK_SIZE * BLOCK_SIZE], mat, threadIdx.x, threadIdx.y);

    // Write data back to global memory
    devMat[cursorY * n + cursorX] = mat[threadIdx.y * BLOCK_SIZE + threadIdx.x];
    __syncthreads();
}

__global__ void stage2_col(int *devMat, int startIdx, int n) {
    // Load adj. matrix from global memory to shared memory
    int cursorX = startIdx + threadIdx.x;
    int cursorY = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    
    if ((blockIdx.x * BLOCK_SIZE) == startIdx)
        return;

    // C[BLOCK_SIZE * BLOCK_SIZE], B[BLOCK_SIZE * BLOCK_SIZE]
    __shared__ int mat[2 * BLOCK_SIZE * BLOCK_SIZE];

    // Load adj. matrixs from global memory to shared memory
    mat[threadIdx.y * BLOCK_SIZE + threadIdx.x] = devMat[cursorY * n + cursorX];
    mat[threadIdx.y * BLOCK_SIZE + threadIdx.x + BLOCK_SIZE * BLOCK_SIZE] = devMat[(startIdx + threadIdx.y) * n + cursorX];
    __syncthreads();

    // Perform APSP on the block
    block_APSP(mat, mat, &mat[BLOCK_SIZE * BLOCK_SIZE], threadIdx.x, threadIdx.y);

    // Write data back to global memory
    devMat[cursorY * n + cursorX] = mat[threadIdx.y * BLOCK_SIZE + threadIdx.x];
    __syncthreads();
}

__global__ void stage3(int *devMat, int startIdx, int n) {
    // Load adj. matrix from global memory to shared memory
    int cursorX = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int cursorY = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    // long long int t1 = clock64();
    if ((blockIdx.x * BLOCK_SIZE) == startIdx || (blockIdx.y * BLOCK_SIZE) == startIdx)
        return;

    // C[BLOCK_SIZE * BLOCK_SIZE], B[BLOCK_SIZE * BLOCK_SIZE], A[BLOCK_SIZE * BLOCK_SIZE]
    __shared__ int mat[3 * BLOCK_SIZE * BLOCK_SIZE];

    // Load adj. matrixs from global memory to shared memory
    mat[threadIdx.y * BLOCK_SIZE + threadIdx.x] = devMat[cursorY * n + cursorX];
    mat[threadIdx.y * BLOCK_SIZE + threadIdx.x + BLOCK_SIZE * BLOCK_SIZE] = devMat[cursorY * n + startIdx + threadIdx.x];
    mat[threadIdx.y * BLOCK_SIZE + threadIdx.x + 2 * BLOCK_SIZE * BLOCK_SIZE] = devMat[(startIdx + threadIdx.y) * n + cursorX];
    __syncthreads();
    // long long int t2 = clock64();
    // printf("%lld ", (t2 - t1));
    // Perform APSP on the block
    block_APSP(mat, &mat[BLOCK_SIZE * BLOCK_SIZE], &mat[2 * BLOCK_SIZE * BLOCK_SIZE], threadIdx.x, threadIdx.y);
    // long long int t3 = clock64();
    // printf("%lld ", (t3 - t2));
    // Write data back to global memory
    devMat[cursorY * n + cursorX] = mat[threadIdx.y * BLOCK_SIZE + threadIdx.x];
    __syncthreads();
    // long long int t4 = clock64();
    // printf("%lld ", (t4 - t3));
}

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
    static int block_dim = (verticesTotal + BLOCK_SIZE - 1) / BLOCK_SIZE;
    static int n = BLOCK_SIZE * block_dim;

    // Create adjanency matrix
    int *adjMat = (int *) malloc(n * n * sizeof(int));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j)
                adjMat[i * n + j] = 0;
            else
                adjMat[i * n + j] = INF;
        }
    }

    // Put edges into adjanency matrix
    int tmp[3];
    for (int i = 0; i < edgesTotal; i++) {
        _ = fread(&tmp, sizeof(int), 3, inFp);
        adjMat[tmp[0] * n + tmp[1]] = tmp[2];
    }
    fclose(inFp);

    // Print input graph
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         if (adjMat[i * n + j] != 1073741823)
    //             std::cout << adjMat[i * n + j] << " ";
    //         else
    //             std::cout << "INF" << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Stream number
    // const int nStreams = 8;

    // Init streams
    // cudaStream_t streams[nStreams];
    // for (int i = 0; i < nStreams; i++) {
    //     cudaStreamCreate(&streams[i]);
    // }

    int* device_adjMat;
    cudaMalloc((void **) &device_adjMat, n * n * sizeof(int));

    // Put adjanency matrix to GPU
    cudaMemcpy(device_adjMat, adjMat, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // stages
    for (int k_start = 0; k_start < n; k_start += BLOCK_SIZE) {
        stage1<<< 1, dim3(BLOCK_SIZE, BLOCK_SIZE), 0 >>> (device_adjMat, k_start, n);
        stage2_row<<< block_dim, dim3(BLOCK_SIZE, BLOCK_SIZE), 0 >>> (device_adjMat, k_start, n);
        stage2_col<<< block_dim, dim3(BLOCK_SIZE, BLOCK_SIZE), 0 >>> (device_adjMat, k_start, n);
        stage3<<< dim3(block_dim, block_dim), dim3(BLOCK_SIZE, BLOCK_SIZE), 0 >>> (device_adjMat, k_start, n);
    }

    // output
    cudaMemcpy(adjMat, device_adjMat, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print input graph
    // for (int i = 0; i < verticesTotal; i++) {
    //     for (int j = 0; j < verticesTotal; j++) {
    //         if (ansMat[i * n + j] != 1073741823)
    //             std::cout << ansMat[i * n + j] << " ";
    //         else
    //             std::cout << "INF" << " ";
    //     }
    //     std::cout << std::endl;
    // }

    for (int i = 0; i < verticesTotal; i++) {
        fwrite(&adjMat[i * n], sizeof(int), verticesTotal, outFp);
    }
    fclose(outFp);

    return 0;
}