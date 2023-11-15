#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define PROCESS_LENGTH 32
#define SCALE 8

/* Hint 7 */
// this variable is used by device
__constant__ int mask[MASK_N][MASK_X][MASK_Y];
int cpuMask[MASK_N][MASK_X][MASK_Y] = { 
        {{ -1, -4, -6, -4, -1},
        { -2, -8,-12, -8, -2},
        {  0,  0,  0,  0,  0}, 
        {  2,  8, 12,  8,  2}, 
        {  1,  4,  6,  4,  1}},
        {{ -1, -2,  0,  2,  1}, 
        { -4, -8,  0,  8,  4}, 
        { -6,-12,  0, 12,  6}, 
        { -4, -8,  0,  8,  4}, 
        { -1, -2,  0,  2,  1}} 
    };

int read_png(const char* filename, unsigned char** image, unsigned* height, 
             unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

/* Hint 5 */
// this function is called by host and executed by device
__global__ void sobel (unsigned char* s, unsigned char* t, unsigned height, unsigned width, unsigned channels) {
    int  x, y, i, v, u;
    int  R, G, B;
    double val[MASK_N * 3] = {0.0};
    int adjustX, adjustY, xBound, yBound;

    // Thread statistic
    int threadIndex = threadIdx.y * blockDim.y + threadIdx.x;
    int blkIndex = blockIdx.y * blockIdx.y + blockIdx.x; 

    // Fetch this block's image
    int imageSize = (PROCESS_LENGTH + 2) * (PROCESS_LENGTH + 2) * 3;
    __shared__ unsigned char image[imageSize];

    int cursorX = blockIdx.x * 32 - 2 + (threadIndex % 36);
    int cursorY = blockIdx.y * 32 - 2 + (threadIndex / 36);
    int writtenIndex = threadIndex;

    if (cursorX >= 0 && cursorX < width && cursorY >= 0 && cursorY < height) {
        image[writtenIndex * 3] = s[(cursorY * width + cursorX) * 3];
        image[writtenIndex * 3 + 1] = s[(cursorY * width + cursorX) * 3 + 1];
        image[writtenIndex * 3 + 2] = s[(cursorY * width + cursorX) * 3 + 2];
    } else {
        image[writtenIndex * 3] = 0;
        image[writtenIndex * 3 + 1] = 0;
        image[writtenIndex * 3 + 2] = 0;
    }

    cursorX = blockIdx.x * 32 - 2 + ((threadIndex + blockDim.x * blockDim.y) % 36);
    cursorY = blockIdx.y * 32 - 2 + ((threadIndex + blockDim.x * blockDim.y) / 36);
    writtenIndex = threadIndex + blockDim.x * blockDim.y;

    if (cursorX >= 0 && cursorX < width && cursorY >= 0 && cursorY < height) {
        image[writtenIndex * 3] = s[(cursorY * width + cursorX) * 3];
        image[writtenIndex * 3 + 1] = s[(cursorY * width + cursorX) * 3 + 1];
        image[writtenIndex * 3 + 2] = s[(cursorY * width + cursorX) * 3 + 2];
    } else {
        image[writtenIndex * 3] = 0;
        image[writtenIndex * 3 + 1] = 0;
        image[writtenIndex * 3 + 2] = 0;
    }

    __syncthreads();

    /* Hint 6 */
    // parallel job by blockIdx, blockDim, threadIdx
    for (v = -yBound; v < yBound + adjustY; ++v) {
        for (u = -xBound; u < xBound + adjustX; ++u) {
            R = image[channels * (width * (threadIdx.y + v) + (threadIdx.x + u)) + 2];
            G = image[channels * (width * (threadIdx.y + v) + (threadIdx.x + u)) + 1];
            B = image[channels * (width * (threadIdx.y + v) + (threadIdx.x + u)) + 0];

            for (i = 0; i < MASK_N; ++i) {
                val[i * 3 + 2] += R * mask[i][u + xBound][v + yBound];
                val[i * 3 + 1] += G * mask[i][u + xBound][v + yBound];
                val[i * 3] += B * mask[i][u + xBound][v + yBound];
                val[i * 3 + 2] += R * mask[i][u + xBound][v + yBound];
                val[i * 3 + 1] += G * mask[i][u + xBound][v + yBound];
                val[i * 3] += B * mask[i][u + xBound][v + yBound];
            }
        }
    }

    float totalR = 0.0;
    float totalG = 0.0;
    float totalB = 0.0;
    for (i = 0; i < MASK_N; ++i) {
        totalR += val[i * 3 + 2] * val[i * 3 + 2];
        totalG += val[i * 3 + 1] * val[i * 3 + 1];
        totalB += val[i * 3 + 0] * val[i * 3 + 0];
    }

    totalR = sqrt(totalR) / SCALE;
    totalG = sqrt(totalG) / SCALE;
    totalB = sqrt(totalB) / SCALE;
    const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
    const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
    const unsigned char cB = (totalB > 255.0) ? 255 : totalB;
    cursorX = blockIdx.x * 32 + threadIdx.x;
    cursorY = blockIdx.y * 32 + threadIdx.y;

    if (cursorX >= 0 && cursorX < width && cursorY >= 0 && cursorY < height) {
        t[channels * (width * y + x) + 2] = cR;
        t[channels * (width * y + x) + 1] = cG;
        t[channels * (width * y + x) + 0] = cB;
    }
}

int main(int argc, char** argv) {
    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char* host_s = NULL;
    read_png(argv[1], &host_s, &height, &width, &channels);
    unsigned char* host_t = (unsigned char *) malloc(height * width * channels * sizeof(unsigned char));
    
    /* Hint 1 */
    // cudaMalloc(...) for device src and device dst
    unsigned char* device_s;
    cudaMalloc((void **) &device_s, height * width * channels * sizeof(unsigned char));
    unsigned char* device_t;
    cudaMalloc((void **) &device_t, height * width * channels * sizeof(unsigned char));

    /* Hint 2 */
    // cudaMemcpy(...) copy source image to device (filter matrix if necessary)
    cudaMemcpy(device_s, host_s, height * width * channels * sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpyToSymbol(mask, cpuMask, MASK_N * MASK_X * MASK_Y * sizeof(int));

    /* Hint 3 */
    // acclerate this function
    dim3 blk(PROCESS_LENGTH, PROCESS_LENGTH);
    dim3 grid(width / PROCESS_LENGTH + 1, height / PROCESS_LENGTH + 1);
    
    sobel<<< grid, blk >>>(device_s, device_t, height, width, channels);
    
    /* Hint 4 */
    // cudaMemcpy(...) copy result image to host
    cudaMemcpy(host_t, device_t, height * width * channels * sizeof(int), cudaMemcpyDeviceToHost);

    write_png(argv[2], host_t, height, width, channels);

    return 0;
}
