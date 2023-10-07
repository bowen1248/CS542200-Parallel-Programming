#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <mpi.h>

int rank, size;

int compare (const void * a, const void * b) {
  float fa = *(const float*) a;
  float fb = *(const float*) b;
  return (fa > fb) - (fa < fb);
}

float * sort_left_half (float * arr1, float * arr2, float * result, int n) {
    int index1 = 0;
    int index2 = 0;
    
    // for (int j = 0; j < n; j++) {
    //     printf("%f, ", arr1[j]);
    // }
    // printf("---rank %d, arr1---\n", rank);

    // for (int j = 0; j < n; j++) {
    //     printf("%f, ", arr2[j]);
    // }
    // printf("---rank %d, arr2---\n", rank);
    for (int i = 0; i < n; i++) {
        if (arr1[index1] == INFINITY || index1 > n) {
            result[i] = arr2[index2];
            index2++;
            index1++;
        } else if (arr2[index2] == INFINITY || index2 > n){
            result[i] = arr1[index1];
            index1++;
            index2++;
        }
        else if (arr1[index1] > arr2[index2]) {
            result[i] = arr2[index2];
            index2++;
        } else {
            result[i] = arr1[index1];
            index1++;
        }
    }

    if (n == 0) {
        result[0] = INFINITY;
    }

    // for (int j = 0; j < n; j++) {
    //     printf("%f, ", result[j]);
    // }
    //     printf("---rank %d, sort left---\n", rank);

    return result;
}

float * sort_right_half (float * arr1, float * arr2, float * result, int n) {
    int index1 = n - 1;
    int index2 = n - 1;

    // for (int j = 0; j < n; j++) {
    //     printf("%f, ", arr1[j]);
    // }
    // printf("---rank %d, arr1---\n", rank);

    // for (int j = 0; j < n; j++) {
    //     printf("%f, ", arr2[j]);
    // }
    // printf("---rank %d, arr2---\n", rank);
    for (int i = (n - 1); i >= 0; i--) {
        if (arr1[index1] == INFINITY || index1 < 0) {
            result[i] = arr2[index2];
            index2--;
            index1--;
        } else if (arr2[index2] == INFINITY || index2 < 0){
            result[i] = arr1[index1];
            index1--;
            index2--;
        } else if (arr1[index1] > arr2[index2]) {
            result[i] = arr1[index1];
            index1--;
        } else {
            result[i] = arr2[index2];
            index2--;
        }
    }

    if (n == 0) {
        result[0] = INFINITY;
    }

    // for (int j = 0; j < n; j++) {
    //     printf("%f, ", result[j]);
    // }
    //     printf("---rank %d, sort right---\n", rank);

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

    num_local_elem = n / size;
    if (rank < (n % size)) {
        num_local_elem += 1;
        additional_elem = 1;
    }

    local_start_index = (n / size) * rank;
    if (rank < (n % size)) {
        local_start_index += rank;
    } else {
        local_start_index += (n % size);
    }

    local_end_index = local_start_index + num_local_elem;
    // printf("Start index: %d, End index: %d, Num of element: %d\n", local_start_index, local_end_index, num_local_elem);
    // Allocate used buffers
    int allocate_num = ((n / size) + 1);
    float* data = (float *) malloc(allocate_num * sizeof(float));
    float* temp1 = (float *) malloc(allocate_num * sizeof(float));
    float* temp2 = (float *) malloc(allocate_num * sizeof(float));
    
    // Get this process data's by its rank
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(float) * local_start_index, data, num_local_elem, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);
    if (additional_elem == 0 || num_local_elem == 0) {
        data[allocate_num - 1] = INFINITY;
    }

    // Sort local value by process
    if (num_local_elem >= 2) {
        qsort(data, num_local_elem, sizeof(float), compare);
    }

    // for (int j = 0; j < allocate_num; j++) {
    //     printf("%f, ", data[j]);
    // }
    // printf("---initial data %d, end---\n", rank);

    // Even and odd phases
    MPI_Status status;
    float * tmp = data;
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < size; i++) {
        // even phase
        if (size % 2 == 0 || size % 2 == 1 && rank != (size - 1)) {
            if (rank % 2 == 0) {
                MPI_Sendrecv(data, allocate_num, MPI_FLOAT, rank + 1, 0,
                            temp1, allocate_num, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &status);
                data = sort_left_half(data, temp1, temp2, num_local_elem);
                temp2 = tmp;
                tmp = data;
            } else if (rank % 2 == 1) {
                MPI_Sendrecv(data, allocate_num, MPI_FLOAT, rank - 1, 0,
                            temp1, allocate_num, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &status);
                data = sort_right_half(data, temp1, temp2, num_local_elem);
                temp2 = tmp; 
                tmp = data;
            }
        }

        // for (int j = 0; j < allocate_num; j++) {
        //     printf("%f, ", data[j]);
        // }
        // printf("---rank %d, even phase end---\n", rank);

        MPI_Barrier(MPI_COMM_WORLD);
        // odd phase
        if ((size % 2 == 0 && rank != 0 && rank != (size - 1)) || (size % 2 == 1 && rank != 0)) {
            if (rank % 2 == 0) {
                MPI_Sendrecv(data, allocate_num, MPI_FLOAT, rank - 1, 0,
                            temp1, allocate_num, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &status);
                data = sort_right_half(data, temp1, temp2, num_local_elem);
                temp2 = tmp;
                tmp = data;
            } else if (rank % 2 == 1) {
                MPI_Sendrecv(data, allocate_num, MPI_FLOAT, rank + 1, 0,
                temp1, allocate_num, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &status);
                data = sort_left_half(data, temp1, temp2, num_local_elem);
                temp2 = tmp;
                tmp = data;
            }
        }

        // for (int j = 0; j < allocate_num; j++) {
        //     printf("%f, ", data[i]);
        // }
        // printf("---rank %d, odd phase end---\n", rank);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // for (int i = 0; i < num_local_elem; i++) {
    //     printf("rank %d got float: %f\n", rank, data[i]);
    // }

    // printf("rank %d float size: %d\n", rank, num_local_elem);

    // int * count = (int *) malloc(size * sizeof(int));
    // for (int i = 0; i < size - 1; i++) {
    //     count[i] = data_size;
    // }
    // int * displs = (int *) malloc(size * sizeof(int));
    // for (int i = 0; i < size; i++) {
    //     displs[i] = data_size * i;
    // }

    // count[size - 1] = n - (n / size) * (size - 1);
    // for (int i = 0; i < size; i++) {
    //     printf("rank: %d %d ,", rank, count[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < size; i++) {
    //     printf("rank: %d %d ,", rank, displs[i]);
    // }
    // printf("\n");

    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, sizeof(float) * local_start_index, data, num_local_elem, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    // if (rank == 0) {
    //     int ans[4] = {1, 2, 3, 4}; 
    //     // float * ans = (float *) malloc(size * sizeof(float));
    //     // MPI_Gatherv ( data, data_size, MPI_FLOAT,
    //     //             ans, count, displs, MPI_FLOAT,
    //     //             0, MPI_COMM_WORLD );
    //     printf("fml, nigga");
    //     // Write final result to output file
    //     MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    //     MPI_File_write_at(output_file, 0, ans, n, MPI_FLOAT, MPI_STATUS_IGNORE);
    //     MPI_File_close(&output_file);
    // } else {
    //     // MPI_Gatherv ( data, data_size, MPI_FLOAT,
    //     //             NULL, NULL, NULL, MPI_FLOAT,
    //     //             0, MPI_COMM_WORLD );
        
    // }


    MPI_Finalize();
    return 0;

    // Get current process rank / v
    // Open input file / v
    // Get input file size / v
    // Get this process data's by its rank / v
    // Sort local value by process / v
    // Implement odd phase by comparing and sort to neighbor
    // 1 -> 2 3 -> 4 5 -> 6 7
        // two pointer
    // Implement even phase by comparing and sort to neighbor
    // 1 2 -> 3 4 -> 5 6 -> 7
    // Organize to output
}

