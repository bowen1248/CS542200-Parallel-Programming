#include <sched.h>
#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

int main(int argc, char **argv) {
    freopen("result.txt","w",stdout);
    /* detect how many CPUs are available */
    // cpu_set_t cpu_set;
    // int ncpus;
    // sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    // ncpus = CPU_COUNT(&cpu_set);

    // // Thread handlers
    // pthread_t threads[ncpus];

    /* argument parsing */
    assert(argc == 4);
    const char *inputFile = argv[1];
    const char *answerFile = argv[2];
    const char *userFile = argv[3];

    FILE *inFp1 = fopen(inputFile, "rb");
    FILE *inFp2 = fopen(answerFile, "rb");
    FILE *inFp3 = fopen(userFile, "rb");
    if( inFp1 == NULL ) {
        fprintf(stderr, "Couldn't open %s: %s\n", inputFile, strerror(errno));
        exit(1);
    }
    if( inFp2 == NULL ) {
        fprintf(stderr, "Couldn't open %s: %s\n", answerFile, strerror(errno));
        exit(1);
    }
    if( inFp3 == NULL ) {
        fprintf(stderr, "Couldn't open %s: %s\n", userFile, strerror(errno));
        exit(1);
    }


    int verticesTotal;
    int edgesTotal;
    
    fread(&verticesTotal, sizeof(int), 1, inFp1);
    fread(&edgesTotal, sizeof(int), 1, inFp1);

    std::cout << verticesTotal << " " << edgesTotal << std::endl;

    int *adjMat1 = (int *) calloc(verticesTotal * verticesTotal, sizeof(int));
    int *adjMat2 = (int *) calloc(verticesTotal * verticesTotal, sizeof(int));

    fread(adjMat1, sizeof(int), verticesTotal * verticesTotal, inFp2);
    fread(adjMat2, sizeof(int), verticesTotal * verticesTotal, inFp3);

    std::cout << "Standard answer matrix: " << std::endl;
    for (int i = 0; i < verticesTotal; i++) {
        for (int j = 0; j < verticesTotal; j++) {
            if (adjMat1[i * verticesTotal + j] != 1073741823)
                std::cout << adjMat1[i * verticesTotal + j] << " ";
            else
                std::cout << "INF" << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "User answer matrix: " << std::endl;
    for (int i = 0; i < verticesTotal; i++) {
        for (int j = 0; j < verticesTotal; j++) {
            if (adjMat2[i * verticesTotal + j] != 1073741823)
                std::cout << adjMat2[i * verticesTotal + j] << " ";
            else
                std::cout << "INF" << " ";
        }
        std::cout << std::endl;
    }

    bool ans = true;
    for (int i = 0; i < verticesTotal * verticesTotal; i++) {
        if (adjMat1[i] != adjMat2[i]) {
            std::cout << "Result is wrong" << std::endl;
            return 0;
        }
    }

    std::cout << "Result is correct" << std::endl;

    return 0;
}