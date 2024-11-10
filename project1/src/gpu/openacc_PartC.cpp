//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// OpenACC implementation of image filtering on JPEG
//

#include <memory.h>
#include <cstring>
#include <chrono>
#include <cmath>
#include <iostream>
#include <openacc.h>

#include "../utils.hpp"
using namespace std;
#pragma acc routine seq
ColorValue acc_clamp_pixel_value(float value)
{
    return value > 255 ? 255
           : value < 0 ? 0
                       : static_cast<unsigned char>(value);
}

#pragma acc routine seq
float acc_bilateral_filter(const ColorValue* values, int row, int col,
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
    return acc_clamp_pixel_value(filtered_value);
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read JPEG File
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    // if (bufferR == nullptr)
    // {
    //     std::cerr << "Failed to read input JPEG image\n";
    //     return -1;
    // }
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    // int num_channels = input_jpeg.num_channels;
    size_t buffer_size = width * height;
    unsigned char* filteredImageR = new unsigned char[buffer_size];
    unsigned char* filteredImageG = new unsigned char[buffer_size];
    unsigned char* filteredImageB = new unsigned char[buffer_size];
    unsigned char* bufferR = new unsigned char[buffer_size];
    unsigned char* bufferG = new unsigned char[buffer_size];
    unsigned char* bufferB = new unsigned char[buffer_size];

    memset(filteredImageR, 0, buffer_size);
    memset(filteredImageG, 0, buffer_size);
    memset(filteredImageB, 0, buffer_size);
    memcpy(bufferR, input_jpeg.r_values, buffer_size);
    memcpy(bufferG, input_jpeg.g_values, buffer_size);
    memcpy(bufferB, input_jpeg.b_values, buffer_size);
    delete[] input_jpeg.r_values;
    delete[] input_jpeg.g_values;
    delete[] input_jpeg.b_values;

#pragma acc enter data copyin(filteredImageR[0 : buffer_size], \
    filteredImageG[0 : buffer_size], \
    filteredImageB[0 : buffer_size], \
                              bufferR[0 : buffer_size],        \
                              bufferG[0 : buffer_size],        \
                              bufferB[0 : buffer_size],)
#pragma acc update device(\
filteredImageR[0 : buffer_size], \
    filteredImageG[0 : buffer_size], \
    filteredImageB[0 : buffer_size], \
                              bufferR[0 : buffer_size],        \
                              bufferG[0 : buffer_size],        \
                              bufferB[0 : buffer_size],\
                              )

    auto start_time = std::chrono::high_resolution_clock::now();
// cout<<"kkkkk"<<endl;
#pragma acc parallel present(                                \
filteredImageR[0 : buffer_size], \
    filteredImageG[0 : buffer_size], \
    filteredImageB[0 : buffer_size], \
                              bufferR[0 : buffer_size],        \
                              bufferG[0 : buffer_size],        \
                              bufferB[0 : buffer_size]) num_gangs(1024)
    {
#pragma acc loop independent
    for (int y = 1; y < height-1; y++)
    {
#pragma acc loop independent
        for (int x = 1; x < width-1; x++)
        {
            int id = y * width + x;

            ColorValue r_sum = acc_bilateral_filter(bufferR, y, x, width);
            ColorValue g_sum = acc_bilateral_filter(bufferG, y, x, width);
            ColorValue b_sum = acc_bilateral_filter(bufferB, y, x, width);
            // ColorValue r_sum = bilateral_filter(inputRvalues, y, x, width);
            // ColorValue g_sum = bilateral_filter(inputBvalues, y, x, width);
            // ColorValue b_sum = bilateral_filter(inputGvalues, y, x, width); // 交换过绿色和蓝色就正常
            

            filteredImageR[id] = r_sum;
            filteredImageG[id] = g_sum;
            filteredImageB[id] = b_sum;
        }
    }
    }
    // cout<<"qqqqq"<<endl;
#pragma acc update self(\
filteredImageR[0 : buffer_size], \
    filteredImageG[0 : buffer_size], \
    filteredImageB[0 : buffer_size], \
)

#pragma acc exit data copyout(\
filteredImageR[0 : buffer_size], \
    filteredImageG[0 : buffer_size], \
    filteredImageB[0 : buffer_size], \
)

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    // Save output JPEG image
    // cout<<"ascxsacx"<<endl;
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JpegSOA output_jpeg{filteredImageR, filteredImageG, filteredImageB, input_jpeg.width, input_jpeg.height,
                         input_jpeg.num_channels, input_jpeg.color_space};
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Post-processing
    // delete[] buffer;
    // delete[] filteredImage;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}

