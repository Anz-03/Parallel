//
// Created by Liu Yuxuan on 2024/9/10
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Row-wise Pthread parallel implementation of smooth image filtering of JPEG
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <pthread.h>

#include "../utils.hpp"

struct ThreadData
{
    unsigned char* inputRvalues;
    unsigned char* inputGvalues;
    unsigned char* inputBvalues;
    unsigned char* outputRChannel;
    unsigned char* outputGChannel;
    unsigned char* outputBChannel;
    int width;
    int height;
    int start_row;
    int end_row;
};

void* bilateral_filter_thread_function(void* arg)
{
    ThreadData* data = (ThreadData*)arg;

    for (int y = data->start_row; y < data->end_row; y++)
    {
        for (int x = 0; x < data->width; x++)
        {
            int id = y * data->width + x;

            ColorValue r_sum = bilateral_filter(data->inputRvalues, y, x, data->width);
            ColorValue g_sum = bilateral_filter(data->inputBvalues, y, x, data->width);
            ColorValue b_sum = bilateral_filter(data->inputGvalues, y, x, data->width);
            // ColorValue r_sum = bilateral_filter(data->inputRvalues, y, x, data->width);
            // ColorValue g_sum = bilateral_filter(data->inputBvalues, y, x, data->width);
            // ColorValue b_sum = bilateral_filter(data->inputGvalues, y, x, data->width); // 交换过绿色和蓝色就正常
            

            data->outputRChannel[id] = r_sum;
            data->outputGChannel[id] = g_sum;
            data->outputBChannel[id] = b_sum;
        }
    }
    return nullptr;
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg NUM_THREADS\n";
        return -1;
    }
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    // Read input JPEG image
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }

    int NUM_THREADS = std::stoi(argv[3]); // Convert the input to integer

    unsigned char* rChannel =
        new unsigned char[input_jpeg.width * input_jpeg.height];
    unsigned char* gChannel =
        new unsigned char[input_jpeg.width * input_jpeg.height];
    unsigned char* bChannel =
        new unsigned char[input_jpeg.width * input_jpeg.height];

    pthread_t* threads = new pthread_t[NUM_THREADS];
    ThreadData* threadData = new ThreadData[NUM_THREADS];
    int rowsPerThread = input_jpeg.height / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; i++)
    {
        threadData[i] = {input_jpeg.r_values, input_jpeg.b_values, input_jpeg.g_values,
                         rChannel, gChannel, bChannel,
                         input_jpeg.width,
                         input_jpeg.height,
                         i * rowsPerThread,
                         (i == NUM_THREADS - 1) ? input_jpeg.height
                                                : (i + 1) * rowsPerThread};
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_create(&threads[i], NULL, bilateral_filter_thread_function,
                       &threadData[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";

    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JpegSOA output_jpeg{rChannel, gChannel, bChannel, input_jpeg.width, input_jpeg.height,
                         input_jpeg.num_channels, input_jpeg.color_space};
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    delete[] input_jpeg.r_values;
    delete[] input_jpeg.g_values;
    delete[] input_jpeg.b_values;
    delete[] rChannel;
    delete[] gChannel;
    delete[] bChannel;
    delete[] threads;
    delete[] threadData;
    return 0;
}
