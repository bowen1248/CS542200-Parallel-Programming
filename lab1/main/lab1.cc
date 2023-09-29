#include "mpi.h"
#include <math.h>
#include <stdio.h>

int main(int argc, char** argv) {
	int numtasks;
	int cur_rank;
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long allocated_range1;
	unsigned long long allocated_range2;
	unsigned long long ans;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &cur_rank);
	
	allocated_range1 = (r / numtasks) * cur_rank;
	if (cur_rank == (numtasks - 1)) {
		allocated_range2 = r;
	} else {
		allocated_range2 = (r / numtasks) * (cur_rank + 1);
	}

	unsigned long long pixels = 0;
	unsigned long long r_square = r * r;
	unsigned long long sqrt_num = r_square - allocated_range1 * allocated_range1;
	for (unsigned long long x = allocated_range1; x < allocated_range2; x++) {
		unsigned long long y = ceil(sqrtl(sqrt_num));
		sqrt_num = sqrt_num - x - x - 1;
		pixels += y;
		if (pixels >= k) {
			pixels -= k;
		}
	}

	MPI_Reduce(&pixels, &ans, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if (cur_rank == 0) {
		printf("%llu", (4 * ans) % k);
	}

	MPI_Finalize();
}
