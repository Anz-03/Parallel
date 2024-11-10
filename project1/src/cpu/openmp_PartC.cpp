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
using namespace std;

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg NUM_THREADS\n";
        return -1;
    }
    int NUM_THREADS = std::stoi(argv[3]);
    omp_set_num_threads(NUM_THREADS);
    // cout<<"aaaaa"<<endl;
    // Read JPEG File
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    auto filteredImageR =
        new unsigned char[input_jpeg.width * input_jpeg.height];
    auto filteredImageG =
        new unsigned char[input_jpeg.width * input_jpeg.height];
    auto filteredImageB =
        new unsigned char[input_jpeg.width * input_jpeg.height];
    auto start_time = std::chrono::high_resolution_clock::now();
    ////////////
#pragma omp parallel for shared(input_jpeg)
    for (int y = 0; y < input_jpeg.height; y++)
    {
        for (int x = 0; x < input_jpeg.width; x++)
        {
            int id = y * input_jpeg.width + x;
            // cout<<"bbbbb"<<endl;
            ColorValue r_sum = bilateral_filter(input_jpeg.r_values, y, x, input_jpeg.width);
            ColorValue g_sum = bilateral_filter(input_jpeg.g_values, y, x, input_jpeg.width);
            ColorValue b_sum = bilateral_filter(input_jpeg.b_values, y, x, input_jpeg.width);
            // ColorValue r_sum = bilateral_filter(inputRvalues, offset+y, x, width);
            // ColorValue g_sum = bilateral_filter(inputBvalues, offset+y, x, width);
            // ColorValue b_sum = bilateral_filter(inputGvalues, offset+y, x, width); // 交换过绿色和蓝色就正常
            
            // cout<<(int)r_sum<<" ";
            filteredImageR[id] = r_sum;
            filteredImageG[id] = g_sum;
            filteredImageB[id] = b_sum;
            // cout<<"ccccc"<<endl;
        }
    }
    ////////////
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    end_time - start_time);
    const char* output_filepath = argv[2];
        JpegSOA output_jpeg{filteredImageR,filteredImageG,filteredImageB,
        input_jpeg.width, input_jpeg.height,
        input_jpeg.num_channels, input_jpeg.color_space};
        if (export_jpeg(output_jpeg, output_filepath))
        {
            std::cerr << "Failed to write output JPEG\n";
            return -1;
        }
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
