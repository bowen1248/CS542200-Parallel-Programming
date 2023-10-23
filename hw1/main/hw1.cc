#include <cstdio>
#include <cstdlib>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <mpi.h>

int rank, size;

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
            result[i] = arr1[index1];
            index1++;
        } else if (arr1[index1] > arr2[index2]) {
            result[i] = arr2[index2];
            index2++;
        } else {
            result[i] = arr1[index1];
            index1++;
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
            result[i] = arr1[index1];
            index1--;
        } else if (arr1[index1] > arr2[index2]) {
            result[i] = arr1[index1];
            index1--;
        } else {
            result[i] = arr2[index2];
            index2--;
        }
    }

    return result;
}

int main (int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    // Get current process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get input args
    char *n_string = argv[1];
    char *input_filename = argv[2];
    char *output_filename = argv[3];
    MPI_File input_file, output_file;

    // Calculate fetched data range
    int n = atoi(n_string);
    int num_local_elem;
    int local_start_index;
    int local_end_index;
    int additional_elem = 0;

    // Local element number in this array
    num_local_elem = n / size;
    if (rank < (n % size)) {
        num_local_elem += 1;
        additional_elem = 1;
    }

    // Starting index to fetch in .in file
    local_start_index = (n / size) * rank;
    if (rank < (n % size)) {
        local_start_index += rank;
    } else {
        local_start_index += (n % size);
    }

    local_end_index = local_start_index + num_local_elem;

    
    // Calculate right and left neighborhood process's local element number
    int right_elem_num = n / size;
    int left_elem_num = n / size;
    if ((rank + 1) < (n % size)) {
        right_elem_num += 1;
    }
    if (rank == (size - 1)) {
        right_elem_num = 0;
    }

    if ((rank - 1) < (n % size)) {
        left_elem_num += 1;
    }
    if (rank == 0) {
        left_elem_num = 0;
    }

    // Allocate used buffers
    float * data = (float *) malloc(num_local_elem * sizeof(float));
    float * comm_right = (float *) malloc(right_elem_num * sizeof(float));
    float * comm_left = (float *) malloc(left_elem_num * sizeof(float));
    float * result = (float *) malloc(num_local_elem * sizeof(float));
    
    // Get this process data's by its rank
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(float) * local_start_index, data, num_local_elem, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    // Sort local value by process
    if (num_local_elem >= 2) {
        boost::sort::spreadsort::spreadsort(data, data + num_local_elem);
    }

    // Even and odd phases
    MPI_Status status;
    float * tmp = data;
    
    for (int i = 0; i <= (size / 2); i++) {
        // Even phase
        if (size % 2 == 0 || size % 2 == 1 && rank != (size - 1)) {
            if (rank % 2 == 0) {
                MPI_Sendrecv(data, num_local_elem, MPI_FLOAT, rank + 1, 0,
                            comm_right, right_elem_num, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &status);
                tmp = sort_left_half(data, comm_right, result, num_local_elem, right_elem_num);
                result = data;
                data = tmp;
            } else if (rank % 2 == 1) {
                MPI_Sendrecv(data, num_local_elem, MPI_FLOAT, rank - 1, 0,
                            comm_left, left_elem_num, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &status);
                tmp = sort_right_half(data, comm_left, result, num_local_elem, left_elem_num);
                result = data;
                data = tmp;
            }
        }

        // Odd phase
        if (size % 2 == 0 && rank != 0 && rank != (size - 1) || (size % 2 == 1 && rank != 0)) {
            if (rank % 2 == 0) {
                MPI_Sendrecv(data, num_local_elem, MPI_FLOAT, rank - 1, 0,
                            comm_left, left_elem_num, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &status);
                tmp = sort_right_half(data, comm_left, result, num_local_elem, left_elem_num);
                result = data;
                data = tmp;
            } else if (rank % 2 == 1) {
                MPI_Sendrecv(data, num_local_elem, MPI_FLOAT, rank + 1, 0,
                comm_right, right_elem_num, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &status);
                tmp = sort_left_half(data, comm_right, result, num_local_elem, right_elem_num);
                result = data;
                data = tmp;
            }
        }
    }

    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, sizeof(float) * local_start_index, data, num_local_elem, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    MPI_Finalize();
    return 0;
}

