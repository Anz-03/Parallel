//
// Created by Liu Yuxuan on 2024/9/10
// Modified on Yang Yufan's simd_PartB.cpp on 2023/9/16
// Email: yufanyang1@link.cuhk.edu.cn
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// SIMD (AVX2) implementation of transferring a JPEG picture from RGB to gray
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>
// #include <assert.h>

#include "../utils.hpp"
using namespace std;
inline void sum_vector(float& result,__m256 values) {
   // 方法一：使用水平加法指令

    __m128 hi = _mm256_extractf128_ps(values, 1); // 获取高128位
    __m128 lo = _mm256_extractf128_ps(values, 0); // 获取低128位
    __m128 sum = _mm_add_ps(hi, lo);           // 高低128位相加

    sum = _mm_hadd_ps(sum, sum);              // 水平加法
    sum = _mm_hadd_ps(sum, sum);
    result = _mm_cvtss_f32(sum);
}
inline ColorValue my_bilateral_filter(const ColorValue* values, int row, int col,
                            int widtconstexpr float powf_SIGMA_D_2=2.89;
constexpr float minus_half_powf_SIGMA_R_2=-1250;
h)
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
    constexpr float powf_SIGMA_D_2=2.89;
    // cout<<powf_SIGMA_D_2<<endl;
    constexpr float minus_half_powf_SIGMA_R_2=-1250;
    // cout<<minus_half_powf_SIGMA_R_2<<endl;
    // exit(1);
    __m256 vec_values =
        _mm256_set_ps(value_11, value_12, value_13, value_21,
                      value_23, value_31, value_32, value_33);
    constexpr float w_spatial_border = 1;
    // cout<<w_spatial_border<<endl;
    constexpr float w_spatial_corner = 0.0555762;
    // cout<<w_spatial_corner<<endl;
    // exit(1);
    // Intensity Weights
    ColorValue center_value = value_22;
    
    __m256 tmp_sign_mask = _mm256_set1_ps(-0.0f);
    __m256 vec_weights =_mm256_xor_ps(vec_values, tmp_sign_mask);

    __m256 tmp_vec_adder = _mm256_set1_ps(center_value);
    vec_weights=_mm256_add_ps(vec_weights,tmp_vec_adder);

    float* f = (float*)&vec_weights;
    __m256 tmp_result = _mm256_set_ps(
        powf(f[0],2), powf(f[1],2), powf(f[2],2), powf(f[3],2),
        powf(f[4],2), powf(f[5],2), powf(f[6],2), powf(f[7],2)
    );
    vec_weights=tmp_result;

    __m256 tmp_vec_multiplier = _mm256_set1_ps(minus_half_powf_SIGMA_R_2);
    vec_weights=_mm256_mul_ps(vec_weights,tmp_vec_multiplier);

    f = (float*)&vec_weights;
    tmp_result = _mm256_set_ps(
        expf(f[0]), expf(f[1]), expf(f[2]), expf(f[3]),
        expf(f[4]), expf(f[5]), expf(f[6]), expf(f[7])
    );
    vec_weights=tmp_result;

    tmp_vec_multiplier = _mm256_set_ps(
        w_spatial_corner,w_spatial_border,
        w_spatial_corner,w_spatial_border,
        w_spatial_corner,w_spatial_border,
        w_spatial_corner,w_spatial_border
    );
    vec_weights=_mm256_mul_ps(vec_weights,tmp_vec_multiplier);

    // float w_11 = w_spatial_corner * expf(powf(center_value - value_11, 2) *
    //                                      minus_half_powf_SIGMA_R_2);
    // float w_12 = w_spatial_border * expf(powf(center_value - value_12, 2) *
    //                                      minus_half_powf_SIGMA_R_2);
    // float w_13 = w_spatial_corner * expf(powf(center_value - value_13, 2) *
    //                                      minus_half_powf_SIGMA_R_2);
    // float w_21 = w_spatial_border * expf(powf(center_value - value_21, 2) *
    //                                      minus_half_powf_SIGMA_R_2);
    // // float w_22 = 1.0;
    // float w_23 = w_spatial_border * expf(powf(center_value - value_23, 2) *
    //                                      minus_half_powf_SIGMA_R_2);
    // float w_31 = w_spatial_corner * expf(powf(center_value - value_31, 2) *
    //                                      minus_half_powf_SIGMA_R_2);
    // float w_32 = w_spatial_border * expf(powf(center_value - value_32, 2) *
    //                                      minus_half_powf_SIGMA_R_2);
    // float w_33 = w_spatial_border * expf(powf(center_value - value_33, 2) *
    //                                      minus_half_powf_SIGMA_R_2);
    // w_11 = 1.1631;
        
    // w_12 = 1.1632;
        
    // w_13 = 1.1633;
        
    // w_21 = 1.1634;
        
    // w_23 = 1.1635;
        
    // w_31 = 1.1636;
        
    // w_32 = 1.1637;
        
    // w_33 = 1.1638;
    // __m256 vec_weights =
    //     _mm256_set_ps(w_11, w_12, w_13, w_21,
    //                   w_23, w_31, w_32, w_33);
    // float sum_weights_prev =
    //     w_11 + w_12 + w_13 + w_21 + 1 + w_23 + w_31 + w_32 + w_33;
    float sum_weights;
    sum_vector(sum_weights,vec_weights);
    sum_weights+=1;
    // if(sum_weights!=sum_weights_prev){
    //     cout<<sum_weights<<endl;
    //     cout<<sum_weights_prev<<endl;
    //     assert(0);
    // }
    // Calculate filtered value
    // float filtered_value_prev =
    //     (w_11 * value_11 + 
    //     w_12 * value_12 + 
    //     w_13 * value_13 + 
    //     w_21 * value_21 +
    //      center_value + 
    //      w_23 * value_23 + 
    //      w_31 * value_31 +
    //      w_32 * value_32 + 
    //      w_33 * value_33) /
    //     sum_weights;
    // 计算点积
    __m256 product = _mm256_mul_ps(vec_weights, vec_values); // Multiply element-wise
    float tmp;
    sum_vector(tmp,product);
    float filtered_value = (tmp+center_value)/sum_weights;
    return clamp_pixel_value(filtered_value);
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
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    // Apply the filter to the image
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    int num_channels = input_jpeg.num_channels;
    ColorValue* output_r_values = new ColorValue[width * height];
    ColorValue* output_g_values = new ColorValue[width * height];
    ColorValue* output_b_values = new ColorValue[width * height];
    JpegSOA output_jpeg{
        output_r_values, output_g_values, output_b_values,       width,
        height,          num_channels,    input_jpeg.color_space};
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int channel = 0; channel < num_channels; ++channel)
    {
        for (int row = 1; row < height - 1; ++row)
        {
            for (int col = 1; col < width - 1; ++col)
            {
                int index = row * width + col;
                ColorValue filtered_value = my_bilateral_filter(
                    input_jpeg.get_channel(channel), row, col, width);
                output_jpeg.set_value(channel, index, filtered_value);
            }
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Cleanup
    delete[] output_r_values;
    delete[] output_g_values;
    delete[] output_b_values;
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}