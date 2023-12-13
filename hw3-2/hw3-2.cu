#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define BLOCK_SIZE 64
#define INF 1073741823 // 2^30 - 1

__device__ void block_APSP(short *C, short *A, short *B, int x, int y) {
    for (int k = 0; k < BLOCK_SIZE; k++) {
        // printf("%d %d %d %d %d %d\n", blockIdx.y, blockIdx.x, y, x, A[y * BLOCK_SIZE + k], B[k * BLOCK_SIZE + x]);
        C[threadIdx.y * BLOCK_SIZE + threadIdx.x] = min(A[threadIdx.y * BLOCK_SIZE + k] + B[k * BLOCK_SIZE + threadIdx.x], C[threadIdx.y * BLOCK_SIZE + threadIdx.x]);
        C[threadIdx.y * BLOCK_SIZE + (threadIdx.x + 32)] = min(A[threadIdx.y * BLOCK_SIZE + k] + B[k * BLOCK_SIZE + (threadIdx.x + 32)], C[threadIdx.y * BLOCK_SIZE + (threadIdx.x + 32)]);
        C[(threadIdx.y + 32) * BLOCK_SIZE + threadIdx.x] = min(A[(threadIdx.y + 32) * BLOCK_SIZE + k] + B[k * BLOCK_SIZE + threadIdx.x], C[(threadIdx.y + 32) * BLOCK_SIZE + threadIdx.x]);
        C[(threadIdx.y + 32) * BLOCK_SIZE + (threadIdx.x + 32)] = min(A[(threadIdx.y + 32) * BLOCK_SIZE + k] + B[k * BLOCK_SIZE + (threadIdx.x + 32)], C[(threadIdx.y + 32) * BLOCK_SIZE + (threadIdx.x + 32)]);
                // C[i * BLOCK_SIZE + j] = __viaddmax_s16x2((unsigned int) A[i * BLOCK_SIZE + k], (unsigned int) B[k * BLOCK_SIZE + j]);
        __syncthreads();
    }
}

__global__ void stage1(short *devMat, int startIdx, int n) {
    __shared__ short mat[BLOCK_SIZE * BLOCK_SIZE];
    // Start idx is the most left upper coordinate
    // Load adj. matrix from global memory to shared memory
    for (int i = threadIdx.y; i < BLOCK_SIZE; i += blockDim.y) {
        for (int j = threadIdx.x; j < BLOCK_SIZE; j += blockDim.x) {
            mat[i * BLOCK_SIZE + j] = devMat[(startIdx + i) * n + (startIdx + j)];
        }
    }
    __syncthreads();

    // Perform APSP on the block
    block_APSP(mat, mat, mat, threadIdx.x, threadIdx.y);

    // Write data back to global memory
    for (int i = threadIdx.y; i < BLOCK_SIZE; i += blockDim.y) {
        for (int j = threadIdx.x; j < BLOCK_SIZE; j += blockDim.x) {
            devMat[(startIdx + i) * n + (startIdx + j)] = mat[i * BLOCK_SIZE + j];
        }
    }

    //devMat[cursorY * n + cursorX] = mat[threadIdx.y * BLOCK_SIZE + threadIdx.x];
    __syncthreads();
}

__global__ void stage2_row(short *devMat, int startIdx, int n) {
    // Load adj. matrix from global memory to shared memory
    // int cursorX = blockIdx.x * 32 + threadIdx.x;
    // int cursorY = startIdx + threadIdx.y;

    if ((blockIdx.x * BLOCK_SIZE) == startIdx)
        return;

    // C[BLOCK_SIZE * BLOCK_SIZE], B[BLOCK_SIZE * BLOCK_SIZE]
    __shared__ short mat[2 * BLOCK_SIZE * BLOCK_SIZE];

    // Load adj. matrixs from global memory to shared memory
    for (int i = threadIdx.y; i < BLOCK_SIZE; i += blockDim.y) {
        for (int j = threadIdx.x; j < BLOCK_SIZE; j += blockDim.x) {
            mat[i * BLOCK_SIZE + j] = devMat[(startIdx + i) * n + (blockIdx.x * BLOCK_SIZE + j)];
            mat[i * BLOCK_SIZE + j + BLOCK_SIZE * BLOCK_SIZE] = devMat[(startIdx + i) * n + startIdx + j];
        }
    }
    __syncthreads();

    // Perform APSP on the block
    block_APSP(mat, &mat[BLOCK_SIZE * BLOCK_SIZE], mat, threadIdx.x, threadIdx.y);

    // Write data back to global memory
    for (int i = threadIdx.y; i < BLOCK_SIZE; i += blockDim.y) {
        for (int j = threadIdx.x; j < BLOCK_SIZE; j += blockDim.x) {
            devMat[(startIdx + i) * n + (blockIdx.x * BLOCK_SIZE + j)] = mat[i * BLOCK_SIZE + j];
        }
    }
    __syncthreads();
}

__global__ void stage2_col(short *devMat, int startIdx, int n) {
    // Load adj. matrix from global memory to shared memory
    // int cursorX = startIdx + threadIdx.x;
    // int cursorY = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    
    if ((blockIdx.x * BLOCK_SIZE) == startIdx)
        return;

    // C[BLOCK_SIZE * BLOCK_SIZE], B[BLOCK_SIZE * BLOCK_SIZE]
    __shared__ short mat[2 * BLOCK_SIZE * BLOCK_SIZE];

    // Load adj. matrixs from global memory to shared memory
    for (int i = threadIdx.y; i < BLOCK_SIZE; i += blockDim.y) {
        for (int j = threadIdx.x; j < BLOCK_SIZE; j += blockDim.x) {
            mat[i * BLOCK_SIZE + j] = devMat[(blockIdx.x * BLOCK_SIZE + i) * n + (startIdx + j)];
            mat[i * BLOCK_SIZE + j + BLOCK_SIZE * BLOCK_SIZE] = devMat[(startIdx + i) * n + (startIdx + j)];
        }
    }
    __syncthreads();

    // Perform APSP on the block
    block_APSP(mat, mat, &mat[BLOCK_SIZE * BLOCK_SIZE], threadIdx.x, threadIdx.y);

    // Write data back to global memory
    for (int i = threadIdx.y; i < BLOCK_SIZE; i += blockDim.y) {
        for (int j = threadIdx.x; j < BLOCK_SIZE; j += blockDim.x) {
            devMat[(blockIdx.x * BLOCK_SIZE + i) * n + (startIdx + j)] = mat[i * BLOCK_SIZE + j];
        }
    }
    __syncthreads();
}

__global__ void stage3(short *devMat, int startIdx, int n) {
    // Load adj. matrix from global memory to shared memory
    // int cursorX = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    // int cursorY = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    // long long int t1 = clock64();
    if ((blockIdx.x * BLOCK_SIZE) == startIdx || (blockIdx.y * BLOCK_SIZE) == startIdx)
        return;

    // C[BLOCK_SIZE * BLOCK_SIZE], B[BLOCK_SIZE * BLOCK_SIZE], A[BLOCK_SIZE * BLOCK_SIZE]
    __shared__ short mat[3 * BLOCK_SIZE * BLOCK_SIZE];

    // Load adj. matrixs from global memory to shared memory
    for (int i = threadIdx.y; i < BLOCK_SIZE; i += blockDim.y) {
        for (int j = threadIdx.x; j < BLOCK_SIZE; j += blockDim.x) {
            mat[i * BLOCK_SIZE + j] = devMat[(blockIdx.y * BLOCK_SIZE + i) * n + (blockIdx.x * BLOCK_SIZE + j)];
            mat[i * BLOCK_SIZE + j + BLOCK_SIZE * BLOCK_SIZE] = devMat[(blockIdx.y * BLOCK_SIZE + i) * n + (startIdx + j)];
            mat[i * BLOCK_SIZE + j + 2 * BLOCK_SIZE * BLOCK_SIZE] = devMat[(startIdx + i) * n + (blockIdx.x * BLOCK_SIZE + j)];
        }
    }

    __syncthreads();
    // long long int t2 = clock64();
    // printf("%lld ", (t2 - t1));
    // Perform APSP on the block
    block_APSP(mat, &mat[BLOCK_SIZE * BLOCK_SIZE], &mat[2 * BLOCK_SIZE * BLOCK_SIZE], threadIdx.x, threadIdx.y);
    // long long int t3 = clock64();
    // printf("%lld ", (t3 - t2));
    // Write data back to global memory
    for (int i = threadIdx.y; i < BLOCK_SIZE; i += 32) {
        for (int j = threadIdx.x; j < BLOCK_SIZE; j += 32) {
            devMat[(blockIdx.y * BLOCK_SIZE + i) * n + (blockIdx.x * BLOCK_SIZE + j)] = mat[i * BLOCK_SIZE + j];
        }
    }
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
printf("%d %d ", verticesTotal, edgesTotal);
    // Create adjanency matrix
    short *adjMat = (short *) malloc(n * n * sizeof(short));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j)
                adjMat[i * n + j] = 0;
            else
                adjMat[i * n + j] = 10000; // INF
        }
    }

    // Put edges into adjanency matrix
    int tmp[3];
    for (int i = 0; i < edgesTotal; i++) {
        _ = fread(&tmp, sizeof(int), 3, inFp);
        adjMat[tmp[0] * n + tmp[1]] = (short) tmp[2];
    }
    fclose(inFp);

    // Print input graph
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         if (adjMat[i * n + j] != 10000)
    //             std::cout << adjMat[i * n + j] << " ";
    //         else
    //             std::cout << "INF" << " ";
    //     }
    //     std::cout << std::endl;
    // }

    short* device_adjMat;
    cudaMalloc((void **) &device_adjMat, n * n * sizeof(short));

    // Put adjanency matrix to GPU
    cudaMemcpy(device_adjMat, adjMat, n * n * sizeof(short), cudaMemcpyHostToDevice);

    // stages
    for (int k_start = 0; k_start < n; k_start += BLOCK_SIZE) {
        stage1<<< 1, dim3(32, 32), 0 >>> (device_adjMat, k_start, n);
        stage2_row<<< block_dim, dim3(32, 32), 0 >>> (device_adjMat, k_start, n);
        stage2_col<<< block_dim, dim3(32, 32), 0 >>> (device_adjMat, k_start, n);
        stage3<<< dim3(block_dim, block_dim), dim3(32, 32), 0 >>> (device_adjMat, k_start, n);
    }

    // output
    cudaMemcpy(adjMat, device_adjMat, n * n * sizeof(short), cudaMemcpyDeviceToHost);

    // Print input graph
    // for (int i = 0; i < verticesTotal; i++) {
    //     for (int j = 0; j < verticesTotal; j++) {
    //         if (adjMat[i * n + j] != 1073741823)
    //             std::cout << adjMat[i * n + j] << " ";
    //         else
    //             std::cout << "INF" << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // for (int i = 0; i < verticesTotal; i++) {
    //     for (int j = 0; j < verticesTotal; j++) {
    //         int tmp = (int) adjMat[i * n + j];
    //         if (tmp == 10000)
    //             tmp = INF;
    //         fwrite(&tmp, sizeof(int), 1, outFp);
    //     }
    // }
    fclose(outFp);

    return 0;
}