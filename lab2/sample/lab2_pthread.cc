#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <math.h>

unsigned long long r;
unsigned long long k;

typedef struct thread_data {
   unsigned long long start_x;
   unsigned long long end_x;
   unsigned long long result;

} thread_data;

void* calculate_by_range(void* arg) {
    thread_data* tdata = (thread_data*) arg;
	unsigned long long start = tdata->start_x;
	unsigned long long end = tdata->end_x;
	unsigned long long pixels = 0;

	for (unsigned long long x = start; x < end; x++) {
		pixels += ceil(sqrtl(r*r - x*x));
		if (pixels > k) {
			pixels -= k;
		}
	}
	tdata->result = pixels;
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	r = atoll(argv[1]);
	k = atoll(argv[2]);

	// Get CPU numbers
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	int ncpus = CPU_COUNT(&cpuset);

	// Thread data
	thread_data tdata[ncpus];
	pthread_t threads[ncpus];
	int rc;

	for (int i = ncpus - 1; i >= 0; i--) {
		// Calculate boundaries of the thread
		tdata[i].start_x = (r / ncpus) * i;
		if (i == (ncpus - 1)) {
			tdata[i].end_x = r;
		} else {
			tdata[i].end_x = (r / ncpus) * (i + 1);
		}
		tdata[i].result = 0;
		
		rc = pthread_create(&threads[i], NULL, calculate_by_range, (void*)&tdata[i]);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
	}

	for (int i = 0; i < ncpus; i++) {
		pthread_join(threads[i], NULL); 
	}

	unsigned long long pixels = 0;
	for (int i = 0; i < ncpus; i++) {
		pixels += tdata[i].result;
		pixels %= k;
	}

	printf("%llu\n", (4 * pixels) % k);
}
