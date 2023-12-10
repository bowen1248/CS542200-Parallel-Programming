#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>
#include <cuda_fp16.h>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define PROCESS_LENGTH 28
#define SCALE 8
#define CAL_TYPE float

inline __device__ int bound_check(int val, int lower, int upper) {
    if (val >= lower && val < upper)
        return 1;
    else
        return 0;
}

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
    int  i, x, y;
    int  R, G, B;
    float val[MASK_N * 3] = {0.0};

    // Create a 32 * 32 image for this block to run
    __shared__ unsigned char image[32 * 32 * 3];

    int cursorX = blockIdx.x * 28 + threadIdx.x - 2;
    int cursorY = blockIdx.y * 28 + threadIdx.y - 2;
    int writtenIndex = (threadIdx.y * blockDim.x + threadIdx.x) * 3;

    if (bound_check(cursorX, 0, width) && bound_check(cursorY, 0, height)) {
        image[writtenIndex] = s[(cursorY * width + cursorX) * 3];
        image[writtenIndex + 1] = s[(cursorY * width + cursorX) * 3 + 1];
        image[writtenIndex + 2] = s[(cursorY * width + cursorX) * 3 + 2];
    } else {
        image[writtenIndex] = 0;
        image[writtenIndex + 1] = 0;
        image[writtenIndex + 2] = 0;
    }

    __syncthreads();

    // /* Hint 6 */
    // Do convolution for this specific output pixel by blockIdx, blockDim, threadIdx
    for (y = -2; y <= 2; y++) {
        for (x = -2; x <= 2; x++) {
            if (bound_check(threadIdx.x, 2, 30) && bound_check(threadIdx.y, 2, 30)) {
                R = image[channels * (32 * (y + threadIdx.y) + (x + threadIdx.x)) + 2];
                G = image[channels * (32 * (y + threadIdx.y) + (x + threadIdx.x)) + 1];
                B = image[channels * (32 * (y + threadIdx.y) + (x + threadIdx.x))];

                val[5] += R * mask[1][x + 2][y + 2];
                val[4] += G * mask[1][x + 2][y + 2];
                val[3] += B * mask[1][x + 2][y + 2];
                val[2] += R * mask[0][x + 2][y + 2];
                val[1] += G * mask[0][x + 2][y + 2];
                val[0] += B * mask[0][x + 2][y + 2];
            }
        }
    }

    CAL_TYPE totalR = 0.0;
    CAL_TYPE totalG = 0.0;
    CAL_TYPE totalB = 0.0;
    for (i = 0; i < MASK_N; ++i) {
        totalR += val[i * 3 + 2] * val[i * 3 + 2];
        totalG += val[i * 3 + 1] * val[i * 3 + 1];
        totalB += val[i * 3 + 0] * val[i * 3 + 0];
    }

    totalR = sqrt(totalR) / SCALE;
    totalG = sqrt(totalG) / SCALE;
    totalB = sqrt(totalB) / SCALE;
    const unsigned char cR = (totalR > 255) ? 255 : totalR;
    const unsigned char cG = (totalG > 255) ? 255 : totalG;
    const unsigned char cB = (totalB > 255) ? 255 : totalB;

    if (bound_check(threadIdx.x, 2, 30) && bound_check(threadIdx.y, 2, 30) && bound_check(cursorX, 0, width) && bound_check(cursorY, 0, height)) {
        t[channels * (width * cursorY + cursorX) + 2] = cR;
        t[channels * (width * cursorY + cursorX) + 1] = cG;
        t[channels * (width * cursorY + cursorX) + 0] = cB;
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
    unsigned char* device_t;
    cudaMalloc((void **) &device_s, height * width * channels * sizeof(unsigned char));
    cudaMalloc((void **) &device_t, height * width * channels * sizeof(unsigned char));

    // /* Hint 2 */
    // // cudaMemcpy(...) copy source image to device (filter matrix if necessary)
    cudaMemcpy(device_s, host_s, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice); 
    cudaMemcpyToSymbol(mask, cpuMask, MASK_N * MASK_X * MASK_Y * sizeof(int));

    // /* Hint 3 */
    // // // acclerate this function
    dim3 blk(PROCESS_LENGTH + 4, PROCESS_LENGTH + 4);
    dim3 grid(width / PROCESS_LENGTH + 1, height / PROCESS_LENGTH + 1);

    sobel<<< grid, blk >>>(device_s, device_t, height, width, channels);
    
    /* Hint 4 */
    // cudaMemcpy(...) copy result image to host
    cudaMemcpy(host_t, device_t, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    write_png(argv[2], host_t, height, width, channels);

    return 0;
}

