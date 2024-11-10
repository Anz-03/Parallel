//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// Sequential implementation of converting a JPEG from RGB to gray
// (Strcture-of-Array)
//

#include <memory.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>

#include "../utils.hpp"

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg NUM_THREADS\n";
        return -1;
    }
    // Read JPEG File
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    int NUM_THREADS = std::stoi(argv[3]);
    omp_set_num_threads(NUM_THREADS);

    unsigned char* rChannel =
        new unsigned char[input_jpeg.width * input_jpeg.height];
    unsigned char* gChannel =
        new unsigned char[input_jpeg.width * input_jpeg.height];
    unsigned char* bChannel =
        new unsigned char[input_jpeg.width * input_jpeg.height];

    auto start_time = std::chrono::high_resolution_clock::now();
    ////////////
#pragma omp parallel for shared(input_jpeg, rChannel, gChannel, bChannel)
    for (int y = 0; y < input_jpeg.height; y++)
    {
        for (int x = 0; x < input_jpeg.width; x++)
        {
            int id = y * input_jpeg.width + x;

            rChannel[id] = bilateral_filter(input_jpeg.r_values, y, x, input_jpeg.width);
            gChannel[id] = bilateral_filter(input_jpeg.g_values, y, x, input_jpeg.width);
            bChannel[id] = bilateral_filter(input_jpeg.b_values, y, x, input_jpeg.width);
        }
    }

    ////////////
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    end_time - start_time);
    const char* output_filepath = argv[2];
    st