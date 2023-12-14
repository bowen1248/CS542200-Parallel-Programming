#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define BLK_n 32
#define THREAD_NUMS 32
#define INF 1073741823 // 2^30 - 1

__device__ void block_APSP(int *C, int *A, int *B, int x, int y) {
    for (int k = 0; k < BLK_n; k++) {
        // printf("%d %d %d %d %d %d\n", blockIdx.y, blockIdx.x, y, x, A[y * BLK_n + k], B[k * BLK_n + x]);
        C[y * BLK_n + x] = min(C[y * BLK_n + x], A[y * BLK_n + k] + B[k * BLK_n + x]);
        __syncthreads();
    }
}

__global__ void stage1(int *devMat, int g_k, int n) {
    // Upperleft coordinate of this thread to global memory
    int g_x = g_k + threadIdx.x;
    int g_y = g_k + threadIdx.y;
    // Upperleft coordinate of this thread to shared memory
    int s_x = threadIdx.x;
    int s_y = threadIdx.y;

    __shared__ int mat[BLK_n * BLK_n];

    // Load adj. matrix from global memory to shared memory
    mat[s_y * BLK_n + s_x] = devMat[g_y * n + g_x];

    // Perform APSP on the block
    for (int k = 0; k < BLK_n; k++) {
        __syncthreads();
        mat[s_y * BLK_n + s_x] = min(mat[s_y * BLK_n + s_x], mat[s_y * BLK_n + k] + mat[k * BLK_n + s_x]);
    }

    // Write data back to global memory
    devMat[g_y * n + g_x] = mat[s_y * BLK_n + s_x];
    __syncthreads();
}

__global__ void stage2(int *devMat, int g_k, int n) {
    // Matrix to be changed
    // Note blockidx.y 0 is row, 1 is column
    int g_x1 = (blockIdx.x * BLK_n + threadIdx.x) * !blockIdx.y + (g_k + threadIdx.x) * blockIdx.y;
    int g_y1 = (g_k + threadIdx.y) * !blockIdx.y + (blockIdx.x * BLK_n + threadIdx.x) * blockIdx.y;
    // Matrix that is for refenerce
    int g_x2 = g_k + threadIdx.x;
    int g_y2 = g_k + threadIdx.y;
    int s_x = threadIdx.x;
    int s_y = threadIdx.y;
    if ((blockIdx.x * BLK_n) == g_k)
        return;

    __shared__ int mat[2 * BLK_n * BLK_n];

    // Load adj. matrixs from global memory to shared memory
    mat[s_y * BLK_n + s_x] = devMat[g_y1 * n + g_x1];
    mat[s_y * BLK_n + s_x + BLK_n * BLK_n] = devMat[g_y2 * n + g_x2];

    // Perform APSP on the block
    for (int k = 0; k < BLK_n; k++) {
        __syncthreads();
        mat[s_y * BLK_n + s_x] = min(mat[s_y * BLK_n + s_x], mat[s_y * BLK_n + k] + mat[k * BLK_n + s_x + BLK_n * BLK_n]);
    }

    // Write data back to global memory
    devMat[g_y1 * n + g_x1] = mat[s_y * BLK_n + s_x];
    devMat[g_y2 * n + g_x2] = mat[s_y * BLK_n + s_x + BLK_n * BLK_n];
    __syncthreads();
}

__global__ void stage3(int *devMat, int startIdx, int n) {
    // Load adj. matrix from global memory to shared memory
    // int cursorX = blockIdx.x * BLK_n + threadIdx.x;
    // int cursorY = blockIdx.y * BLK_n + threadIdx.y;
    // long long int t1 = clock64();
    if ((blockIdx.x * BLK_n) == startIdx || (blockIdx.y * BLK_n) == startIdx)
        return;

    // C[BLK_n * BLK_n], B[BLK_n * BLK_n], A[BLK_n * BLK_n]
    __shared__ int mat[3 * BLK_n * BLK_n];

    // Load adj. matrixs from global memory to shared memory
    mat[threadIdx.y * BLK_n + threadIdx.x] = devMat[(blockIdx.y * BLK_n + threadIdx.y) * n + (blockIdx.x * BLK_n + threadIdx.x)];
    mat[threadIdx.y * BLK_n + threadIdx.x + BLK_n * BLK_n] = devMat[(blockIdx.y * BLK_n + threadIdx.y) * n + (startIdx + threadIdx.x)];
    mat[threadIdx.y * BLK_n + threadIdx.x + 2 * BLK_n * BLK_n] = devMat[(startIdx + threadIdx.y) * n + (blockIdx.x * BLK_n + threadIdx.x)];

    __syncthreads();

    // Perform APSP on the block
    int tmp = mat[threadIdx.y * BLK_n + threadIdx.x];
    for (int k = 0; k < BLK_n; k++) {
        tmp = min(tmp, mat[threadIdx.y * BLK_n + k + BLK_n * BLK_n] + mat[k * BLK_n + threadIdx.x + 2 * BLK_n * BLK_n]);
    }

    // Write data back to global memory
    devMat[(blockIdx.y * BLK_n + threadIdx.y) * n + (blockIdx.x * BLK_n + threadIdx.x)] = tmp;
    __syncthreads();

    // long long int t4 = clock64();
    // printf("%lld ", (t4 - t3));
}

int main(int argc, char **argv) {
    // freopen("log.txt","w",stdout);

    /* argument parsing */
    assert(argc == 3);
    const char *inputFile = argv[1];
    const char *outputFile = argv[2];

    FILE *inFp = fopen(inputFile, "rb");
    FILE *outFp = fopen(outputFile, "wb");

    int verticesTotal;
    int edgesTotal;
    
    // Get input vertices and edges number
    size_t _;
    _ = fread(&verticesTotal, sizeof(int), 1, inFp);
    _ = fread(&edgesTotal, sizeof(int), 1, inFp);
    static int block_dim = (verticesTotal + BLK_n - 1) / BLK_n;
    static int n = BLK_n * block_dim;
    // printf("%d %d ", verticesTotal, edgesTotal);

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
    int tmp[15];
    int i; 
    for (i = 0; i < (edgesTotal - 5); i += 5) {
        _ = fread(&tmp, sizeof(int), 15, inFp);
        adjMat[tmp[0] * n + tmp[1]] = tmp[2];
        adjMat[tmp[3] * n + tmp[4]] = tmp[5];
        adjMat[tmp[6] * n + tmp[7]] = tmp[8];
        adjMat[tmp[9] * n + tmp[10]] = tmp[11];
        adjMat[tmp[12] * n + tmp[13]] = tmp[14];
    }
    for (i = i; i < edgesTotal; i += 1) {
        _ = fread(&tmp, sizeof(int), 3, inFp);
        adjMat[tmp[0] * n + tmp[1]] = tmp[2];
    }
    fclose(inFp);

    // Print input graph
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         if (adjMat[i * n + j] != INF)
    //             std::cout << adjMat[i * n + j] << " ";
    //         else
    //             std::cout << "INF" << " ";
    //     }
    //     std::cout << std::endl;
    // }

    int* devMat;
    cudaMalloc((void **) &devMat, n * n * sizeof(int));

    // Put adjanency matrix to GPU
    cudaMemcpy(devMat, adjMat, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // stages
    for (int g_k = 0; g_k < n; g_k += BLK_n) {
        stage1<<< 1, dim3(THREAD_NUMS, THREAD_NUMS), 0 >>> (devMat, g_k, n);
        stage2<<< dim3(2, block_dim), dim3(THREAD_NUMS, THREAD_NUMS), 0 >>> (devMat, g_k, n);
        stage3<<< dim3(block_dim, block_dim), dim3(THREAD_NUMS, THREAD_NUMS), 0 >>> (devMat, g_k, n);
    }

    // output
    cudaMemcpy(adjMat, devMat, n * n * sizeof(int), cudaMemcpyDeviceToHost);

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

    for (int i = 0; i < verticesTotal; i++) {
        fwrite(&adjMat[i * n], sizeof(int), verticesTotal, outFp);
    }
    fclose(outFp);

    return 0;
}