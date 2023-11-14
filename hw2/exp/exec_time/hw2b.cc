#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP

#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <atomic>
#include <emmintrin.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

int *tmpImage, *fullImage, *finalImage;
int iters;
double left, right, lower, upper;
int height, width;

int taskCount;
std::atomic_int curTask;

double xDelta, yDelta;
int yRange;

int rank, size;

double calculateInterval(timespec& start, timespec& end) {
    struct timespec temp;
    double time_used;

    if ((end.tv_nsec - start.tv_nsec) < 0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    } else {
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
    return time_used;
}

void write_png(const char *filename, int iters, int width, int height, const int *buffer)
{
    FILE *fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y)
    {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x)
        {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters)
            {
                if (p & 16)
                {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                }
                else
                {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char **argv)
{
    struct timespec start, end, preprocess_time, compute_time, mpi_time, io_time;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    MPI_Init(&argc, &argv);

    // Get current process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    int ncpus;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    ncpus = CPU_COUNT(&cpu_set);

    // Thread handlers
    pthread_t threads[ncpus];

    /* argument parsing */
    assert(argc == 9);
    const char *filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    // Calculate delta
    xDelta = (right - left) / width;
    yDelta = (upper - lower) / height;

    // Calculate y range
    yRange = ceil((double) height / size);

    /* allocate memory for image */
    tmpImage = (int *)malloc(width * yRange * sizeof(int));
    if (rank == 0) {
        fullImage = (int *)malloc(width * yRange * size * sizeof(int));
        finalImage = (int *)malloc(width * height * sizeof(int));
    }

/* mandelbrot set */
#pragma omp parallel num_threads(ncpus)
    {   
        clock_gettime(CLOCK_MONOTONIC, &preprocess_time);
        printf("Rank: %d, preprocess time: %f\n", rank, calculateInterval(start, preprocess_time));
        
        int fetchedTask;
        long long int i;
        long long int repeats[2], curPixel[2];
        __m128d x, y, x0, y0, temp, length_squared;
        __m128d two_m128 = _mm_set1_pd(2);

        // Fetch task
        fetchedTask = curTask.fetch_add(1);
        
        while (fetchedTask < yRange)
        {
            y0 = _mm_set1_pd((fetchedTask * size + rank) * yDelta + lower);
            // y0 = _mm_set1_pd((yRange * rank + fetchedTask) * yDelta + lower);
            x0[0] = left;
            x0[1] = xDelta + left;
            curPixel[0] = 0;
            curPixel[1] = 1;
            i = 2;

            // init
            x = y = length_squared = _mm_setzero_pd();
            repeats[0] = repeats[1] = 0;

            while (true)
            {
                if (length_squared[0] >= 4 || repeats[0] >= iters)
                {
                    tmpImage[fetchedTask * width + curPixel[0]] = repeats[0];
                    if (i >= width)
                        break;
                    curPixel[0] = i;
                    x0[0] = xDelta * i + left;
                    x[0] = 0;
                    y[0] = 0;
                    repeats[0] = 0;
                    i++;
                }

                if (length_squared[1] >= 4 || repeats[1] >= iters)
                {
                    tmpImage[fetchedTask * width + curPixel[1]] = repeats[1];
                    if (i >= width)
                        break;
                    curPixel[1] = i;
                    x0[1] = xDelta * i + left;
                    x[1] = 0;
                    y[1] = 0;
                    repeats[1] = 0;
                    i++;
                }

                temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y)), x0);
                y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(x, y), two_m128), y0);
                x = temp;
                length_squared = _mm_add_pd(_mm_mul_pd(x, x), _mm_mul_pd(y, y));

                repeats[0]++;
                repeats[1]++;
            }

            if (curPixel[0] < width)
            {
                x0[0] = xDelta * curPixel[0] + left;
                x[0] = 0;
                y[0] = 0;
                repeats[0] = 0;
                length_squared[0] = 0;
                while (repeats[0] < iters && length_squared[0] < 4)
                {
                    temp[0] = x[0] * x[0] - y[0] * y[0] + x0[0];
                    y[0] = 2 * x[0] * y[0] + y0[0];
                    x[0] = temp[0];
                    length_squared[0] = x[0] * x[0] + y[0] * y[0];
                    ++repeats[0];
                }
                tmpImage[fetchedTask * width + curPixel[0]] = repeats[0];
            }

            if (curPixel[1] < width)
            {
                x0[1] = xDelta * curPixel[1] + left;
                x[1] = 0;
                y[1] = 0;
                repeats[1] = 0;
                length_squared[1] = 0;
                while (repeats[1] < iters && length_squared[1] < 4)
                {
                    temp[1] = x[1] * x[1] - y[1] * y[1] + x0[1];
                    y[1] = 2 * x[1] * y[1] + y0[1];
                    x[1] = temp[1];
                    length_squared[1] = x[1] * x[1] + y[1] * y[1];
                    ++repeats[1];
                }
                tmpImage[fetchedTask * width + curPixel[1]] = repeats[1];
            }
            // Fetch new task
            
            fetchedTask = curTask.fetch_add(1);
        }

        int omp_thread = omp_get_thread_num();
        clock_gettime(CLOCK_MONOTONIC, &compute_time);
        printf("Rank: %d, Process: %d, compute time: %f\n", rank, omp_thread, calculateInterval(preprocess_time, compute_time));
    }

    MPI_Gather(tmpImage, width * yRange, MPI_INT, fullImage, width * yRange, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();

    clock_gettime(CLOCK_MONOTONIC, &mpi_time);
    printf("Rank: %d, mpi time: %f\n", rank, calculateInterval(compute_time, mpi_time));

    /* draw and cleanup */
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            int tmp = height / size;
            if (i < (height % size))
                tmp++;
            for (int j = 0; j < tmp; j++) {
                for (int k = 0; k < width; k++) {
                    finalImage[(j * size + i) * width + k] = fullImage[(i * yRange + j) * width + k];
                }
            }
        }

        write_png(filename, iters, width, height, finalImage);
    }

    clock_gettime(CLOCK_MONOTONIC, &io_time);
    printf("Rank: %d, io time: %f\n", rank, calculateInterval(mpi_time, io_time));
    printf("Rank: %d, total time: %f\n", rank, calculateInterval(start, io_time));

    return 0;
}
