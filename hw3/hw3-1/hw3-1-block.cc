#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define BLOCK_SIZE 128

int main(int argc, char **argv) {
    freopen("log.txt","w",stdout);
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

    // Create adjanency matrix
    int *adjMat = (int *) malloc(verticesTotal * verticesTotal * sizeof(int));
    for (int i = 0; i < verticesTotal; i++) {
        for (int j = 0; j < verticesTotal; j++) {
            if (i == j)
                adjMat[i * verticesTotal + j] = 0;
            else
                adjMat[i * verticesTotal + j] = 1073741823;
        }
    }

    // Put edges into adjanency matrix
    int tmp[3];
    for (int i = 0; i < edgesTotal; i++) {
        fread(&tmp, sizeof(int), 3, inFp);
        adjMat[tmp[0] * verticesTotal + tmp[1]] = tmp[2];
    }
    fclose(inFp);
    // Print input graph
    // for (int i = 0; i < verticesTotal; i++) {
    //     for (int j = 0; j < verticesTotal; j++) {
    //         if (adjMat[i * verticesTotal + j] != 1073741823)
    //             std::cout << adjMat[i * verticesTotal + j] << " ";
    //         else
    //             std::cout << "INF" << " ";
    //     }
    //     std::cout << std::endl;
    // }

#pragma omp parallel num_threads(12) shared(adjMat)
{
    for (int k_start = 0; k_start < verticesTotal; k_start += BLOCK_SIZE) {
        // stage 1
        # pragma omp single 
        {
        int end = std::min(k_start + BLOCK_SIZE, verticesTotal);
        for (int k = k_start; k < end; k++) {
            for (int y = k_start; y < end; y++) {
                for (int x = k_start; x < end; x++) {
                    adjMat[y * verticesTotal + x] = std::min(adjMat[y * verticesTotal + x], adjMat[y * verticesTotal + k] + adjMat[k * verticesTotal + x]);
                }
            }
        }
        }

        // stage 2
        // row
        #pragma omp for nowait 
        for (int x_start = 0; x_start < verticesTotal; x_start += BLOCK_SIZE) {
            if (x_start == k_start)
                continue;
            int x_end = std::min(x_start + BLOCK_SIZE, verticesTotal);
            int y_end = std::min(k_start + BLOCK_SIZE, verticesTotal);
            int k_end = std::min(k_start + BLOCK_SIZE, verticesTotal);
            for (int k = k_start; k < k_end; k++) {
                for (int y = k_start; y < y_end; y++) {
                    for (int x = x_start; x < x_end; x++) {
                        adjMat[y * verticesTotal + x] = std::min(adjMat[y * verticesTotal + x], adjMat[y * verticesTotal + k] + adjMat[k * verticesTotal + x]);
                        
                    }
                }
            }
        }

        // column
        #pragma omp for
        for (int y_start = 0; y_start < verticesTotal; y_start += BLOCK_SIZE) {
            if (y_start == k_start)
                continue;
            int x_end = std::min(k_start + BLOCK_SIZE, verticesTotal);
            int y_end = std::min(y_start + BLOCK_SIZE, verticesTotal);
            int k_end = std::min(k_start + BLOCK_SIZE, verticesTotal);
            for (int k = k_start; k < k_end; k++) {
                for (int y = y_start; y < y_end; y++) {
                    for (int x = k_start; x < x_end; x++) {
                        adjMat[y * verticesTotal + x] = std::min(adjMat[y * verticesTotal + x], adjMat[y * verticesTotal + k] + adjMat[k * verticesTotal + x]);
                        //std::cout << "id: " << omp_get_thread_num() << std::endl;
                    }
                }
            }
        }

        // stage 3
        #pragma omp for 
        for (int x_start = 0; x_start < verticesTotal; x_start += BLOCK_SIZE) {
            for (int y_start = 0; y_start < verticesTotal; y_start += BLOCK_SIZE) {
                if (x_start == k_start || y_start == k_start)
                    continue;
                int x_end = std::min(x_start + BLOCK_SIZE, verticesTotal);
                int y_end = std::min(y_start + BLOCK_SIZE, verticesTotal);
                int k_end = std::min(k_start + BLOCK_SIZE, verticesTotal);
                for (int k = k_start; k < k_end; k++) {
                    for (int y = y_start; y < y_end; y++) {
                        for (int x = x_start; x < x_end; x++) {
                            adjMat[y * verticesTotal + x] = std::min(adjMat[y * verticesTotal + x], adjMat[y * verticesTotal + k] + adjMat[k * verticesTotal + x]);
                        }
                    }
                }
            }
        }
    }
}

    fwrite(adjMat, sizeof(int), verticesTotal * verticesTotal, outFp);
    fclose(outFp);

    return 0;
}