#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define BLK_n 64
#define log2BLK_n 6
#define THREAD_n 32
#define INF 1073741823 // 2^30 - 1

__global__ void stage1(int *devMat, int g_k, int n);
__global__ void stage2(int *devMat, int g_k, int n);
__global__ void stage3(int *devMat, int g_k, int n);

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
    fread(&verticesTotal, sizeof(int), 1, inFp);
    fread(&edgesTotal, sizeof(int), 1, inFp);
    static int block_dim = (verticesTotal + BLK_n - 1) / BLK_n;
    static int n = BLK_n * block_dim;

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
        fread(&tmp, sizeof(int), 15, inFp);
        adjMat[tmp[0] * n + tmp[1]] = tmp[2];
        adjMat[tmp[3] * n + tmp[4]] = tmp[5];
        adjMat[tmp[6] * n + tmp[7]] = tmp[8];
        adjMat[tmp[9] * n + tmp[10]] = tmp[11];
        adjMat[tmp[12] * n + tmp[13]] = tmp[14];
    }
    for (i = i; i < edgesTotal; i += 1) {
        fread(&tmp, sizeof(int), 3, inFp);
        adjMat[tmp[0] * n + tmp[1]] = tmp[2];
    }
    fclose(inFp);

    int* devMat;
    cudaMalloc((void **) &devMat, n * n * sizeof(int));
    cudaMemcpy(devMat, adjMat, n * n * sizeof(int), cudaMemcpyHostToDevice);

    for (int g_k = 0; g_k < n; g_k += BLK_n) {
        stage1<<< 1, dim3(THREAD_n, THREAD_n), 0 >>> (devMat, g_k, n);
        stage2<<< dim3(block_dim - 1, 2), dim3(THREAD_n, THREAD_n), 0 >>> (devMat, g_k, n);
        stage3<<< dim3(block_dim - 1, block_dim - 1), dim3(THREAD_n, THREAD_n), 0 >>> (devMat, g_k, n);
    }

    cudaMemcpy(adjMat, devMat, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(devMat);

    // Output to result file
    for (int i = 0; i < verticesTotal; i++) {
        fwrite(&adjMat[i * n], sizeof(int), verticesTotal, outFp);
    }
    fclose(outFp);

    return 0;
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
    mat[s_y * BLK_n + (s_x + 32)] = devMat[g_y * n + (g_x + 32)];
    mat[(s_y + 32) * BLK_n + s_x] = devMat[(g_y + 32) * n + g_x];
    mat[(s_y + 32) * BLK_n + (s_x + 32)] = devMat[(g_y + 32) * n + (g_x + 32)];

    // Perform APSP on the block
    for (int k = 0; k < BLK_n; k++) {
        __syncthreads();
        mat[s_y * BLK_n + s_x] = min(mat[s_y * BLK_n + s_x], mat[s_y * BLK_n + k] + mat[k * BLK_n + s_x]);
        mat[s_y * BLK_n + (s_x + 32)] = min(mat[s_y * BLK_n + (s_x + 32)], mat[s_y * BLK_n + k] + mat[k * BLK_n + (s_x + 32)]);
        mat[(s_y + 32) * BLK_n + s_x] = min(mat[(s_y + 32) * BLK_n + s_x], mat[(s_y + 32) * BLK_n + k] + mat[k * BLK_n + s_x]);
        mat[(s_y + 32) * BLK_n + (s_x + 32)] = min(mat[(s_y + 32) * BLK_n + (s_x + 32)], mat[(s_y + 32) * BLK_n + k] + mat[k * BLK_n + (s_x + 32)]);
    }

    // Write data back to global memory
    devMat[g_y * n + g_x] = mat[s_y * BLK_n + s_x];
    devMat[g_y * n + (g_x + 32)] = mat[s_y * BLK_n + (s_x + 32)];
    devMat[(g_y + 32) * n + g_x] = mat[(s_y + 32) * BLK_n + s_x];
    devMat[(g_y + 32) * n + (g_x + 32)] = mat[(s_y + 32) * BLK_n + (s_x + 32)];
}

__global__ void stage2(int *devMat, int g_k, int n) {
    // Matrix to be changed
    // Note blockidx.y 0 is row, 1 is column
    int s_x = threadIdx.x;
    int s_y = threadIdx.y;

    __shared__ int mat[2 * BLK_n * BLK_n];

    // Load adj. matrixs from global memory to shared memory
    if (blockIdx.y) {
        // column
        int g_x = g_k + s_x;
        int g_y = (blockIdx.x + (blockIdx.x >= (g_k >> log2BLK_n))) * BLK_n + s_y;

        mat[s_y * BLK_n + s_x] =  devMat[g_y * n + g_x];
        mat[s_y * BLK_n + (s_x + 32)] =  devMat[g_y * n + (g_x + 32)];
        mat[(s_y + 32) * BLK_n + s_x] =  devMat[(g_y + 32) * n + g_x];
        mat[(s_y + 32) * BLK_n + (s_x + 32)] =  devMat[(g_y + 32) * n + (g_x + 32)];
        
        mat[s_y * BLK_n + s_x + BLK_n * BLK_n] = devMat[(g_k + s_y) * n + g_x];
        mat[s_y * BLK_n + (s_x + 32) + BLK_n * BLK_n] = devMat[(g_k + s_y) * n + (g_x + 32)];
        mat[(s_y + 32) * BLK_n + s_x + BLK_n * BLK_n] = devMat[(g_k + s_y + 32) * n + g_x];
        mat[(s_y + 32) * BLK_n + (s_x + 32) + BLK_n * BLK_n] = devMat[(g_k + s_y + 32) * n + (g_x + 32)];

        for (int k = 0; k < BLK_n; k++) {
            __syncthreads();
            mat[s_y * BLK_n + s_x] = min(mat[s_y * BLK_n + s_x], mat[s_y * BLK_n + k] + mat[k * BLK_n + s_x + BLK_n * BLK_n]);
            mat[s_y * BLK_n + (s_x + 32)] = min(mat[s_y * BLK_n + (s_x + 32)], mat[s_y * BLK_n + k] + mat[k * BLK_n + (s_x + 32) + BLK_n * BLK_n]);
            mat[(s_y + 32) * BLK_n + s_x] = min(mat[(s_y + 32) * BLK_n + s_x], mat[(s_y + 32) * BLK_n + k] + mat[k * BLK_n + s_x + BLK_n * BLK_n]);
            mat[(s_y + 32) * BLK_n + (s_x + 32)] = min(mat[(s_y + 32) * BLK_n + (s_x + 32)], mat[(s_y + 32) * BLK_n + k] + mat[k * BLK_n + (s_x + 32) + BLK_n * BLK_n]);
        }

        // Write data back to global memory
        devMat[g_y * n + g_x] = mat[s_y * BLK_n + s_x];
        devMat[g_y * n + (g_x + 32)] = mat[s_y * BLK_n + (s_x + 32)];
        devMat[(g_y + 32) * n + g_x] = mat[(s_y + 32) * BLK_n + s_x];
        devMat[(g_y + 32) * n + (g_x + 32)] = mat[(s_y + 32) * BLK_n + (s_x + 32)];
    } else {
        // row
        int g_x = (blockIdx.x + (blockIdx.x >= (g_k >> log2BLK_n))) * BLK_n + s_x;
        int g_y = g_k + s_y;

        mat[s_y * BLK_n + s_x] = devMat[g_y * n + g_x];
        mat[s_y * BLK_n + (s_x + 32)] = devMat[g_y * n + (g_x + 32)];
        mat[(s_y + 32) * BLK_n + s_x] = devMat[(g_y + 32) * n + g_x];
        mat[(s_y + 32) * BLK_n + (s_x + 32)] = devMat[(g_y + 32) * n + (g_x + 32)];
        
        mat[s_y * BLK_n + s_x + BLK_n * BLK_n] = devMat[g_y * n + (g_k + s_x)];
        mat[s_y * BLK_n + (s_x + 32) + BLK_n * BLK_n] = devMat[g_y * n + (g_k + s_x + 32)];
        mat[(s_y + 32) * BLK_n + s_x + BLK_n * BLK_n] = devMat[(g_y + 32) * n + (g_k + s_x)];
        mat[(s_y + 32) * BLK_n + (s_x + 32) + BLK_n * BLK_n] = devMat[(g_y + 32) * n + (g_k + s_x + 32)];


        for (int k = 0; k < BLK_n; k++) {
            __syncthreads();
            mat[s_y * BLK_n + s_x] = min(mat[s_y * BLK_n + s_x], mat[s_y * BLK_n + k + BLK_n * BLK_n] + mat[k * BLK_n + s_x]);
            mat[s_y * BLK_n + (s_x + 32)] = min(mat[s_y * BLK_n + (s_x + 32)], mat[s_y * BLK_n + k + BLK_n * BLK_n] + mat[k * BLK_n + (s_x + 32)]);
            mat[(s_y + 32) * BLK_n + s_x] = min(mat[(s_y + 32) * BLK_n + s_x], mat[(s_y + 32) * BLK_n + k + BLK_n * BLK_n] + mat[k * BLK_n + s_x]);
            mat[(s_y + 32) * BLK_n + (s_x + 32)] = min(mat[(s_y + 32) * BLK_n + (s_x + 32)], mat[(s_y + 32) * BLK_n + k + BLK_n * BLK_n] + mat[k * BLK_n + (s_x + 32)]);
        }

        // Write data back to global memory
        devMat[g_y * n + g_x] = mat[s_y * BLK_n + s_x];
        devMat[g_y * n + (g_x + 32)] = mat[s_y * BLK_n + (s_x + 32)];
        devMat[(g_y + 32) * n + g_x] = mat[(s_y + 32) * BLK_n + s_x];
        devMat[(g_y + 32) * n + (g_x + 32)] = mat[(s_y + 32) * BLK_n + (s_x + 32)];
    }
}

__global__ void stage3(int *devMat, int g_k, int n) {
    // Load adj. matrix from global memory to shared memory
    int s_x = threadIdx.x;
    int s_y = threadIdx.y;
    int g_x = (blockIdx.x + (blockIdx.x >= (g_k >> log2BLK_n))) * BLK_n + s_x;
    int g_y = (blockIdx.y + (blockIdx.y >= (g_k >> log2BLK_n))) * BLK_n + s_y;

    __shared__ int mat[2 * BLK_n * BLK_n];

    // Load adj. matrixs from global memory to shared memory
    int num1 = devMat[g_y * n + g_x];
    int num2 = devMat[g_y * n + (g_x + 32)];
    int num3 = devMat[(g_y + 32) * n + g_x];
    int num4 = devMat[(g_y + 32) * n + (g_x + 32)];

    mat[s_y * BLK_n + s_x] = devMat[g_y * n + (g_k + s_x)];
    mat[s_y * BLK_n + (s_x + 32)] = devMat[g_y * n + (g_k + s_x + 32)];
    mat[(s_y + 32) * BLK_n + s_x] = devMat[(g_y + 32) * n + (g_k + s_x)];
    mat[(s_y + 32) * BLK_n + (s_x + 32)] = devMat[(g_y + 32) * n + (g_k + s_x + 32)];

    mat[s_y * BLK_n + s_x + BLK_n * BLK_n] = devMat[(g_k + s_y) * n + g_x];
    mat[s_y * BLK_n + (s_x + 32) + BLK_n * BLK_n] = devMat[(g_k + s_y) * n + (g_x + 32)];
    mat[(s_y + 32) * BLK_n + s_x + BLK_n * BLK_n] = devMat[(g_k + s_y + 32) * n + g_x];
    mat[(s_y + 32) * BLK_n + (s_x + 32) + BLK_n * BLK_n] = devMat[(g_k + s_y + 32) * n + (g_x + 32)];

    __syncthreads();

    // Perform APSP on the block
    for (int k = 0; k < BLK_n; k++) {
        num1 = min(num1, mat[s_y * BLK_n + k] + mat[k * BLK_n + s_x + BLK_n * BLK_n]);
        num2 = min(num2, mat[s_y * BLK_n + k] + mat[k * BLK_n + (s_x + 32) + BLK_n * BLK_n]);
        num3 = min(num3, mat[(s_y + 32) * BLK_n + k] + mat[k * BLK_n + s_x + BLK_n * BLK_n]);
        num4 = min(num4, mat[(s_y + 32) * BLK_n + k] + mat[k * BLK_n + (s_x + 32) + BLK_n * BLK_n]);
    }

    // Write data back to global memory
    devMat[g_y * n + g_x] = num1;
    devMat[g_y * n + (g_x + 32)] = num2;
    devMat[(g_y + 32) * n + g_x] = num3;
    devMat[(g_y + 32) * n + (g_x + 32)] = num4;
}