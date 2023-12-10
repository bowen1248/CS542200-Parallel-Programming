#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define BLOCK_SIZE 85
#define INF 1073741823 // 2^30 - 1

int main(int argc, char **argv) {
    // freopen("log.txt","w",stdout);
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    int ncpus;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    ncpus = CPU_COUNT(&cpu_set);

    // // Thread handlers
    // pthread_t threads[ncpus];

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
        fread(&tmp, sizeof(int), 3, inFp);
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

#pragma omp parallel num_threads(12) shared(adjMat)
{
    for (int k_start = 0; k_start < n; k_start += BLOCK_SIZE) {
        // stage 1
        # pragma omp single 
        {

        int end = k_start + BLOCK_SIZE;
        for (int k = k_start; k < end; k++) {
            for (int y = k_start; y < end; y++) {
                for (int x = k_start; x < end; x++) {
                    adjMat[y * n + x] = std::min(adjMat[y * n + x], adjMat[y * n + k] + adjMat[k * n + x]);
                }
            }
        }
        }

        // stage 2
        // row
        #pragma omp for schedule(dynamic) nowait 
        for (int x_start = 0; x_start < n; x_start += BLOCK_SIZE) {
            if (x_start == k_start)
                continue;

            int x_end = x_start + BLOCK_SIZE;
            int y_end = k_start + BLOCK_SIZE;
            int k_end = k_start + BLOCK_SIZE;
            for (int k = k_start; k < k_end; k++) {
                for (int y = k_start; y < y_end; y++) {
                    for (int x = x_start; x < x_end; x++) {
                        adjMat[y * n + x] = std::min(adjMat[y * n + x], adjMat[y * n + k] + adjMat[k * n + x]);
                    }
                }
            }
        }

        // column
        #pragma omp for schedule(dynamic)
        for (int y_start = 0; y_start < n; y_start += BLOCK_SIZE) {
            if (y_start == k_start)
                continue;

            int x_end = k_start + BLOCK_SIZE;
            int y_end = y_start + BLOCK_SIZE;
            int k_end = k_start + BLOCK_SIZE;
            for (int k = k_start; k < k_end; k++) {
                for (int y = y_start; y < y_end; y++) {
                    for (int x = k_start; x < x_end; x++) {
                        adjMat[y * n + x] = std::min(adjMat[y * n + x], adjMat[y * n + k] + adjMat[k * n + x]);
                    }
                }
            }
        }

        // stage 3
        #pragma omp for schedule(dynamic) collapse(2)
        for (int x_start = 0; x_start < n; x_start += BLOCK_SIZE) {
            for (int y_start = 0; y_start < n; y_start += BLOCK_SIZE) {
                if (x_start == k_start || y_start == k_start)
                    continue;

                int x_end = x_start + BLOCK_SIZE;
                int y_end = y_start + BLOCK_SIZE;
                int k_end = k_start + BLOCK_SIZE;
                for (int k = k_start; k < k_end; k++) {
                    for (int y = y_start; y < y_end; y++) {
                        for (int x = x_start; x < x_end; x++) {
                            adjMat[y * n + x] = std::min(adjMat[y * n + x], adjMat[y * n + k] + adjMat[k * n + x]);
                        }
                    }
                }
            }
        }
    }
}

    for (int i = 0; i < verticesTotal; i++) {
        fwrite(&adjMat[i * n], sizeof(int), verticesTotal, outFp);
    }
    fclose(outFp);

    return 0;
}