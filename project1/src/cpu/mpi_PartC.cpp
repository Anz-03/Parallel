//
// Created by Yang Yufan on 2023/9/16.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI implementation of transforming a JPEG image from RGB to gray
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <mpi.h> // MPI Header

#include "../utils.hpp"

#define MASTER 0
#define TAG_GATHER 0
using namespace std;

void bilateral_filter_thread_function(unsigned char* inputRvalues,unsigned char* inputGvalues,unsigned char* inputBvalues,
                                        unsigned char* outputRChannel,unsigned char* outputGChannel,unsigned char* outputBChannel,
                                        int width, int height, int start_row, int end_row, int offset)
{
    for (int y = start_row; y < end_row; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int id = y * width + x;

            ColorValue r_sum = bilateral_filter(inputRvalues, offset+y, x, width);
            ColorValue g_sum = bilateral_filter(inputGvalues, offset+y, x, width);
            ColorValue b_sum = bilateral_filter(inputBvalues, offset+y, x, width);
            // ColorValue r_sum = bilateral_filter(inputRvalues, offset+y, x, width);
            // ColorValue g_sum = bilateral_filter(inputBvalues, offset+y, x, width);
            // ColorValue b_sum = bilateral_filter(inputGvalues, offset+y, x, width); // 交换过绿色和蓝色就正常
            
            // cout<<(int)r_sum<<" ";
            outputRChannel[id] = r_sum;
            outputGChannel[id] = g_sum;
            outputBChannel[id] = b_sum;
        }
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

    // Start the MPI
    MPI_Init(&argc, &argv);
    // How many processes are running
    int numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // What's my rank?
    int taskid;
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    // Which node am I running on?
    int len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;
    
    // Read input JPEG File
    const char* input_filepath = argv[1];
    JpegSOA input_jpeg = read_jpeg_soa(input_filepath);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    // unsigned char* rChannel =
    //     new unsigned char[input_jpeg.width * input_jpeg.height];
    // unsigned char* gChannel =
    //     new unsigned char[input_jpeg.width * input_jpeg.height];
    // unsigned char* bChannel =
    //     new unsigned char[input_jpeg.width * input_jpeg.height];

    // Divide the task
    // For example, there are 11 lines and 3 tasks,
    // we try to divide to 4 4 3 instead of 3 3 5
    int total_line_num = input_jpeg.height - 2;
    int line_per_task = total_line_num / numtasks;
    int left_line_num = total_line_num % numtasks;

    std::vector<int> cuts(numtasks + 1, 1);
    int divided_left_line_num = 0;

    for (int i = 0; i < numtasks; i++)
    {
        if (divided_left_line_num < left_line_num)
        {
            cuts[i + 1] = cuts[i] + line_per_task + 1;
            divided_left_line_num++;
        }
        else
            cuts[i + 1] = cuts[i] + line_per_task;
    }

    if (taskid == MASTER)
    {
        std::cout << "Input file from: " << input_filepath << "\n";
        auto filteredImageR =
            new unsigned char[input_jpeg.width * input_jpeg.height];
        auto filteredImageG =
            new unsigned char[input_jpeg.width * input_jpeg.height];
        auto filteredImageB =
            new unsigned char[input_jpeg.width * input_jpeg.height];
        memset(filteredImageR, 0, input_jpeg.width * input_jpeg.height);
        memset(filteredImageG, 0, input_jpeg.width * input_jpeg.height);
        memset(filteredImageB, 0, input_jpeg.width * input_jpeg.height);

        auto start_time = std::chrono::high_resolution_clock::now();

        // // Filter the first division of the contents
        bilateral_filter_thread_function(input_jpeg.r_values,input_jpeg.g_values,input_jpeg.b_values,
        filteredImageR,filteredImageG,filteredImageB,
        input_jpeg.width,input_jpeg.height,
        cuts[taskid],cuts[taskid + 1],0);

        // Receive the transformed Gray contents from each slave executors
        for (int i = MASTER + 1; i < numtasks; i++)
        {
            int line_width = input_jpeg.width;
            unsigned char* start_posR = filteredImageR + cuts[i] * line_width;
            unsigned char* start_posG = filteredImageG + cuts[i] * line_width;
            unsigned char* start_posB = filteredImageB + cuts[i] * line_width;
            int length = ((i == numtasks-1) ? input_jpeg.height- cuts[i]: cuts[i+1] - cuts[i]) * line_width;
            // cout<<"i="<<i<<endl;
            // cout<<"length="<<length<<endl;
            // cout<<"000000"<<endl;
            MPI_Recv(start_posR, length, MPI_CHAR, i, 0, MPI_COMM_WORLD,
                     &status);
            // cout<<"111111"<<endl;
            MPI_Recv(start_posG, length, MPI_CHAR, i, 1, MPI_COMM_WORLD,
                     &status);
            // cout<<"222222"<<endl;
            MPI_Recv(start_posB, length, MPI_CHAR, i, 2, MPI_COMM_WORLD,
                     &status);
            // cout<<"333333"<<endl;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                  start_time);

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
        // delete[] filteredImageR;
        // delete[] filteredImageG;
        // delete[] filteredImageB;
        std::cout << "Transformation Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count()
                  << " milliseconds\n";
    }
    // The tasks for the slave executor
    // 1. Filter a division of image
    // 2. Send the Filterd contents back to the master executor
    else
    {
        // Intialize the filtered image
        int length = input_jpeg.width * ((taskid == numtasks-1) ? input_jpeg.height- cuts[taskid]: cuts[taskid+1] - cuts[taskid]);
        int offset = input_jpeg.width * cuts[taskid];

        auto filteredImageR = new unsigned char[length];
        auto filteredImageG = new unsigned char[length];
        auto filteredImageB = new unsigned char[length];
        memset(filteredImageR, 0, length);
        memset(filteredImageG, 0, length);
        memset(filteredImageB, 0, length);

        // Filter a coresponding division
        // // Filter the first division of the contents
        bilateral_filter_thread_function(input_jpeg.r_values,input_jpeg.g_values,input_jpeg.b_values,
        filteredImageR,filteredImageG,filteredImageB,
        input_jpeg.width,input_jpeg.height,
        0,(taskid == numtasks-1) ? input_jpeg.height-cuts[taskid]
                                                : cuts[taskid+1]-cuts[taskid],cuts[taskid]);

        // Send the filtered image back to the master
        // cout<<"taskid="<<taskid<<" length="<<length<<endl;
        // for(int i=0;i<length;++i){
        //     cout<<int(filteredImageR[i])<<' ';
        // }
        // cout<<"fcbweiucbwqi"<<endl;
        MPI_Send(filteredImageR, length, MPI_CHAR, MASTER, 0,
                 MPI_COMM_WORLD);
        MPI_Send(filteredImageG, length, MPI_CHAR, MASTER, 1,
                 MPI_COMM_WORLD);
        MPI_Send(filteredImageB, length, MPI_CHAR, MASTER, 2,
                 MPI_COMM_WORLD);

        // Release the memory
        // delete[] filteredImageR;
        // delete[] filteredImageG;
        // delete[] filteredImageB;
    }

    MPI_Finalize();
    return 0;
}
