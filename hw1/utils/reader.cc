// ***************************************************
// A small tool to see binary file
// ***************************************************

#include <cstdio>
#include <cstdlib>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char *input_filename = argv[1];
    char *output_filename = argv[2];

    MPI_File input_file, output_file;
    float data[65536];

    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, 0, data, 65536, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    for (int j = 0; j < 4; j++) {
        printf("%f, ", data[j]);
    }
    printf("---initial data %d, end---\n", rank);

    MPI_Finalize();
    return 0;
}