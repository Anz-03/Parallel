//
// Created by Liu Yuxuan on 2024/9/11
// Modified from Zhong Yebin's PartB on 2023/9/16
//
// Email: yebinzhong@link.cuhk.edu.cn
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// CUDA implementation of bilateral filtering on JPEG image
//

#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#include "../utils.hpp"
using namespace std;
/**
 * Demo kernel device function to clamp pixel value
 * 
 * You may mimic this to implement your own kernel device functions
 */
__device__ unsigned char d_clamp_pixel_value(float value)
{
    return value > 255 ? 255
           : value < 0 ? 0
                       : static_cast<unsigned char>(value);
}

__device__ float d_bilateral_filter(const ColorValue* values, int row, int col,
                            int width)
{
    ColorValue value_11 = values[(row - 1) * width + (col - 1)];
    ColorValue value_12 = values[(row - 1) * width + col];
    ColorValue value_13 = values[(row - 1) * width + (col + 1)];
    ColorValue value_21 = values[row * width + (col - 1)];
    ColorValue value_22 = values[row * width + col];
    ColorValue value_23 = values[row * width + (col + 1)];
    ColorValue value_31 = values[(row + 1) * width + (col - 1)];
    ColorValue value_32 = values[(row + 1) * width + col];
    ColorValue value_33 = values[(row + 1) * width + (col + 1)];
    // Spatial Weights
    float w_spatial_border = expf(-1 / 2 * powf_SIGMA_D_2);
    float w_spatial_corner = expf(2 * -1 / 2 * powf_SIGMA_D_2);
    // Intensity Weights
    ColorValue center_value = value_22;
    float w_11 = w_spatial_corner * expf(powf(center_value - value_11, 2) *
                                         minus_half_powf_SIGMA_R_2);
    float w_12 = w_spatial_border * expf(powf(center_value - value_12, 2) *
                                         minus_half_powf_SIGMA_R_2);
    float w_13 = w_spatial_corner * expf(powf(center_value - value_13, 2) *
                                         minus_half_powf_SIGMA_R_2);
    float w_21 = w_spatial_border * expf(powf(center_value - value_21, 2) *
                                         minus_half_powf_SIGMA_R_2);
    float w_22 = 1.0;
    float w_23 = w_spatial_border * expf(powf(center_value - value_23, 2) *
                                         minus_half_powf_SIGMA_R_2);
    float w_31 = w_spatial_corner * expf(powf(center_value - value_31, 2) *
                                         minus_half_powf_SIGMA_R_2);
    float w_32 = w_spatial_border * expf(powf(center_value - value_32, 2) *
                                         minus_half_powf_SIGMA_R_2);
    float w_33 = w_spatial_border * expf(powf(center_value - value_33, 2) *
                                         minus_half_powf_SIGMA_R_2);
    float sum_weights =
        w_11 + w_12 + w_13 + w_21 + w_22 + w_23 + w_31 + w_32 + w_33;
    // Calculate filtered value
    float filtered_value =
        (w_11 * value_11 + w_12 * value_12 + w_13 * value_13 + w_21 * value_21 +
         w_22 * center_value + w_23 * value_23 + w_31 * value_31 +
         w_32 * value_32 + w_33 * value_33) /
        sum_weights;
    return d_clamp_pixel_value(filtered_value);
}


__global__ void apply_filter_kernel(unsigned char* inputRvalues,
    unsigned char* inputGvalues,
    unsigned char* inputBvalues,
    unsigned char* outputRChannel,
    unsigned char* outputGChannel,
    unsigned char* outputBChannel,
    int width,
    int height,
    int start_row,
    int end_row)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1)
    {
            int id = y * width + x;

            // ColorValue r_sum = d_bilateral_filter(inputRvalues, y, x, width);
            // ColorValue g_sum = d_bilateral_filter(inputBvalues, y, x, width);
            // ColorValue b_sum = d_bilateral_filter(inputGvalues, y, x, width);
            ColorValue r_sum = d_bilateral_filter(inputRvalues, y, x, width);
            ColorValue g_sum = d_bilateral_filter(inputGvalues, y, x, width);
            ColorValue b_sum = d_bilateral_filter(inputBvalues, y, x, width); // 交换过绿色和蓝色就正常
            

            outputRChannel[id] = r_sum;
            outputGChannel[id] = g_sum;
            outputBChannel[id] = b_sum;
    }
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read input JPEG image in structure-of-array form
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    // if (input_jpeg.r_values == nullptr)
    // {
    //     std::cerr << "Failed to read input JPEG image\n";
    //     return -1;
    // }

    // Apply the filter to the image
    size_t buffer_size =
        input_jpeg.width * input_jpeg.height;
    unsigned char* filteredImageR = new unsigned char[buffer_size];
    unsigned char* filteredImageG = new unsigned char[buffer_size];
    unsigned char* filteredImageB = new unsigned char[buffer_size];

    // Allocate GPU memory
    unsigned char* d_input_buffer_r;
    unsigned char* d_input_buffer_g;
    unsigned char* d_input_buffer_b;
    unsigned char* d_filtered_imageR;
    unsigned char* d_filtered_imageG;
    unsigned char* d_filtered_imageB;
    float(*d_filter)[FILTERSIZE];

    cudaMalloc((void**)&d_input_buffer_r, buffer_size);
    cudaMalloc((void**)&d_input_buffer_g, buffer_size);
    cudaMalloc((void**)&d_input_buffer_b, buffer_size);
    cudaMalloc((void**)&d_filtered_imageR, buffer_size);
    cudaMalloc((void**)&d_filtered_imageG, buffer_size);
    cudaMalloc((void**)&d_filtered_imageB, buffer_size);
    cudaMalloc((void**)&d_filter, FILTERSIZE * FILTERSIZE * sizeof(float));

    cudaMemset(d_filtered_imageR, 0, buffer_size);
    cudaMemset(d_filtered_imageG, 0, buffer_size);
    cudaMemset(d_filtered_imageB, 0, buffer_size);

    // Copy input data from host to device
    cudaMemcpy(d_input_buffer_r, input_jpeg.r_values,buffer_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_buffer_g, input_jpeg.g_values,buffer_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_buffer_b, input_jpeg.b_values,buffer_size,
               cudaMemcpyHostToDevice);

    // Set CUDA grid and block sizes
    dim3 blockDim(32, 32);
    dim3 gridDim((input_jpeg.width + blockDim.x - 1) / blockDim.x,
                 (input_jpeg.height + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    // Perform filtering on GPU
    cudaEventRecord(start, 0); // GPU start time
    // Launch CUDA kernel
    apply_filter_kernel<<<gridDim, blockDim>>>(
        d_input_buffer_r,d_input_buffer_g,d_input_buffer_b,
        d_filtered_imageR,d_filtered_imageG,d_filtered_imageB,
        // d_input_buffer_r,d_input_buffer_g,d_input_buffer_b,
        input_jpeg.width,input_jpeg.height,0,input_jpeg.height);
    cudaEventRecord(stop, 0); // GPU end time
    cudaEventSynchronize(stop);
    // Print the result of the GPU computation
    cudaEventElapsedTime(&gpuDuration, start, stop);
    // Copy output data from GPU
    cudaMemcpy(filteredImageR, d_filtered_imageR, buffer_size,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(filteredImageG, d_filtered_imageG, buffer_size,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(filteredImageB, d_filtered_imageB, buffer_size,
               cudaMemcpyDeviceToHost);

    // Save output JPEG image
    const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";
        JpegSOA output_jpeg{filteredImageR,filteredImageG,filteredImageB,
        input_jpeg.width, input_jpeg.height,
        input_jpeg.num_channels, input_jpeg.color_space};
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    // Post-processing
    // delete[] input_jpeg.buffer;
    // delete[] filteredImage;
    // Release GPU memory
    cudaFree(d_input_buffer_r);
    cudaFree(d_input_buffer_g);
    cudaFree(d_input_buffer_b);
    cudaFree(d_filtered_imageR);
    cudaFree(d_filtered_imageG);
    cudaFree(d_filtered_imageB);
    cudaFree(d_filter);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds"
              << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
