#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP

#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <atomic>
#include <emmintrin.h>

#define TASK_HEIGHT 1
#define TASK_WIDTH 80

int *image;
int *image2;
int iters;
double left;
double right;
double lower;
double upper;
int height;
int width;

int taskCount;
int taskWidthNum;
int totalTask;
std::atomic_int curTask;

double xDelta;
double yDelta;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void *slaveSIMD(void *args)
{
    int fetchedTask, xStart, xEnd, taskHeight;
    long long int i;
    long long int repeats[2], curPixel[2];
    __m128d x, y, x0, y0, temp, length_squared;
    __m128d two_m128 = _mm_set1_pd(2);

    // Fetch task
    fetchedTask = curTask.fetch_add(1);

    while (fetchedTask < totalTask)
    {
        taskHeight = fetchedTask / taskWidthNum;
        xStart = (fetchedTask % taskWidthNum) * TASK_WIDTH;
        if (xStart + TASK_WIDTH >= width) 
            xEnd = width;
        else
            xEnd = xStart + TASK_WIDTH;

        y0 = _mm_set1_pd(taskHeight * yDelta + lower);
        x0[0] = xDelta * xStart + left;
        x0[1] = xDelta * (xStart + 1) + left;
        curPixel[0] = xStart;
        curPixel[1] = xStart + 1;
        i = xStart + 2;

        // init
        x = y = length_squared = _mm_setzero_pd();
        repeats[0] = repeats[1] = 0;

        while (true)
        {
            if (repeats[0] >= iters || length_squared[0] >= 4) {
                image[taskHeight * width + curPixel[0]] = repeats[0];
                if (i >= xEnd)
                    break;
                curPixel[0] = i;
                x0[0] = xDelta * i + left;
                x[0] = 0;
                y[0] = 0;
                repeats[0] = 0;
                i++;
            }
            
            if (repeats[1] >= iters || length_squared[1] >= 4) {
                image[taskHeight * width + curPixel[1]] = repeats[1];
                if (i >= xEnd)
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
            length_squared = _mm_add_pd(_mm_mul_pd(x, x),_mm_mul_pd(y, y));

            repeats[0]++;
            repeats[1]++;
        }

        if (curPixel[0] < xEnd) {
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
            image[taskHeight * width + curPixel[0]] = repeats[0];
        }

        if (curPixel[1] < xEnd) {
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
            image[taskHeight * width + curPixel[1]] = repeats[1];
        }

        // Fetch new task
        fetchedTask = curTask.fetch_add(1);
    }

    return NULL;
}

void *slaveSISD(void *args)
{
    int fetchedTask;
    int repeats = 0;
    double x, x0, y, y0;
    double length_squared;
    double temp;

    // Fetch task
    fetchedTask = curTask.fetch_add(1);

    while (fetchedTask < height)
    {
        y0 = fetchedTask * yDelta + lower;

        for (int i = 0; i < width; ++i)
        {
            repeats = 0;
            x = 0;
            x0 = xDelta * i + left;
            y = 0;
            length_squared = 0;

            while (repeats < iters && length_squared < 4)
            {
                temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            image2[fetchedTask * width + i] = repeats;
            
        }
        fetchedTask = curTask.fetch_add(1);
    }

    return NULL;
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

    /* allocate memory for image */
    image = (int *)malloc(width * height * sizeof(int));
    assert(image);
    image2 = (int *)malloc(width * height * sizeof(int));
    assert(image2);

    // Calculate task count
    taskWidthNum = width / TASK_WIDTH + 1;
    totalTask = taskWidthNum * height;

    // Calculate delta
    xDelta = (right - left) / width;
    yDelta = (upper - lower) / height;

    /* mandelbrot set */
    for (int i = 0; i < ncpus; i++)
    {
        pthread_create(&threads[i], NULL, slaveSIMD, NULL);
    }

    for (int i = 0; i < ncpus; i++)
    {
        pthread_join(threads[i], NULL);
    }

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}
