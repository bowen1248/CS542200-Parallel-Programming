#include <assert.h>
#include <stdio.h>
#include <math.h>

#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
	MPI_Init(&argc,&argv);
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	int num_tasks;
	int cur_rank;
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long start_x;
	unsigned long long end_x;
	unsigned long long pixels = 0;
	
    MPI_Comm_rank(MPI_COMM_WORLD, &cur_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

	start_x = (r / num_tasks) * cur_rank;
	if (cur_rank == (num_tasks - 1)) {
		end_x = r;
	} else {
		end_x = (r / num_tasks) * (cur_rank + 1);
	}

	#pragma omp parallel
		#pragma omp for reduction(+:pixels)
			for (unsigned long long x = start_x; x < end_x; x++) {
				pixels += ceil(sqrtl(r*r - x*x));
				if (pixels > k) {
					pixels -= k;
				}
			}

	unsigned long long global_pixels;
	MPI_Reduce(&pixels, &global_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0,
           MPI_COMM_WORLD);

	if (cur_rank == 0)
		printf("%llu\n", (4 * global_pixels) % k);

	MPI_Finalize();
}