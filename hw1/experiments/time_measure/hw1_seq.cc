#include <cstdio>
#include <cstdlib>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <mpi.h>

int rank, size;

double io_time = 0;
double compute_time = 0;
double t2;

float * sort_left_half (float * arr1, float * arr2, float * result, int arr1_len, int arr2_len) {
    int index1 = 0;
    int index2 = 0;
    
    if (arr1_len == 0) {
        return arr2;
    }
    if (arr2_len == 0) {
        return arr1;
    }

    for (int i = 0; i < arr1_len; i++) {
        if (index2 >= arr2_len){
            result[i] = arr1[index1++];
        } else if (arr1[index1] > arr2[index2]) {
            result[i] = arr2[index2++];
        } else {
            result[i] = arr1[index1++];
        }
    }

    return result;
}

float * sort_right_half (float * arr1, float * arr2, float * result, int arr1_len, int arr2_len) {
    int index1 = arr1_len - 1;
    int index2 = arr2_len - 1;

    if (arr1_len == 0) {
        return arr2;
    }
    if (arr2_len == 0) {
        return arr1;
    }

    for (int i = (arr1_len - 1); i >= 0; i--) {
        if (index2 < 0){
            result[i] = arr1[index1--];
        } else if (arr1[index1] > arr2[index2]) {
            result[i] = arr1[index1--];
        } else {
            result[i] = arr2[index2--];
        }
    }

    return result;
}

int main (int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    double t1 = MPI_Wtime();
    // Get current process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get input args
    char *n_string = argv[1];
    char *input_filename = argv[2];
    char *output_filename = argv[3];
    MPI_File input_file, output_file;

    // Allocate used buffers
    
    int n = atoi(n_string);
    float * data = (float *) malloc(n * sizeof(float));
    // Get this process data's by its rank

    t2 = MPI_Wtime();
    compute_time += (t2 - t1);
    t1 = MPI_Wtime();

    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, 0, data, n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    t2 = MPI_Wtime();
    io_time += (t2 - t1);
    t1 = MPI_Wtime();

    // Sort local value by process
    if (n >= 2) {
        boost::sort::spreadsort::spreadsort(data, data + n);
    }

    t2 = MPI_Wtime();
    compute_time += (t2 - t1);
    t1 = MPI_Wtime();

    // Write result to output file
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, 0, data, n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    t2 = MPI_Wtime();
    io_time += (t2 - t1);
    t1 = MPI_Wtime();

    printf("%f %f", compute_time, io_time);

    MPI_Finalize();
    return 0;
}

