// srun -N1 -n1 -cCPUS ./hw3-1 INPUTFILE OUTPUTFILE

#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <pthread.h>
#include <omp.h>

#define BLOCK_SIZE 2

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
    // std::cout << verticesTotal << " " << edgesTotal << std::endl;

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

#pragma omp parallel
{
    for (int k = 0; k < verticesTotal; k++) {
        #pragma omp for
        for (int i = 0; i < verticesTotal; i++) {
            for (int j = 0; j < verticesTotal; j++) {
                adjMat[i * verticesTotal + j] = std::min(adjMat[i * verticesTotal + j], adjMat[i * verticesTotal + k] + adjMat[k * verticesTotal + j]);
                //std::cout << "Thread id: " << omp_get_thread_num() << std::endl;
                //std::cout << i << " " << j << std::endl;
            }
        }
    }
}
    // Print output graph
    // for (int i = 0; i < verticesTotal; i++) {
    //     for (int j = 0; j < verticesTotal; j++) {
    //         if (adjMat[i * verticesTotal + j] != 1073741823)
    //             std::cout << adjMat[i * verticesTotal + j] << " ";
    //         else
    //             std::cout << "INF" << " ";
    //     }
    //     std::cout << std::endl;
    // }

    fwrite(adjMat, sizeof(int), verticesTotal * verticesTotal, outFp);

    return 0;
}