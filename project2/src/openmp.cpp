//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// OpenMp + SIMD + Reordering Matrix Multiplication
//scan

#include <immintrin.h>
#include <omp.h> 
#include <stdexcept>
#include <chrono>
#include <array>
#include "matrix.hpp"

#define TILE_SIZE 2048
using namespace std;

// Matrix quadrant_combine(const Matrix& q1, const Matrix& q2, const Matrix& q3, const Matrix& q4) {
//     if (q1.rows != q2.rows || q1.rows != q3.rows || q1.rows != q4.rows ||
//         q1.cols != q2.cols || q1.cols != q3.cols || q1.cols != q4.cols) {
//         // Handle error: quadrant dimensions must match
//         // (Throw exception, return error, etc.)
//         throw std::runtime_error("Quadrant dimensions must match for combination.");
//     }

//     size_t half_rows = q1.rows;
//     size_t half_cols = q1.cols;
//     size_t combined_rows = half_rows * 2;
//     size_t combined_cols = half_cols * 2;


//     Matrix combined(combined_rows, combined_cols);

//     #pragma omp parallel for
//     for (size_t i = 0; i < half_rows; ++i) {
//         #pragma omp parallel for
//         for (size_t j = 0; j < half_cols; ++j) {
//             combined.data[i][j] = q1.data[i][j];
//             combined.data[i][j + half_cols] = q2.data[i][j];
//             combined.data[i + half_rows][j] = q3.data[i][j];
//             combined.data[i + half_rows][j + half_cols] = q4.data[i][j];
//         }
//     }

//     return combined;
// }
// void quadrant_split(const Matrix& mainM,int length, Matrix& q1, Matrix& q2, Matrix& q3, Matrix& q4) {
// //   if (rows % 2 != 0 || cols % 2 != 0) {
// //     // Handle error: rows and cols must be even
// //     // You might throw an exception, return an error code, or print an error message.
// //     // Example:
// //     // throw std::runtime_error("Matrix dimensions must be even for quadrant split.");
// //     return; // Or handle the error differently
// //   }

//   q1 = Matrix(length,length);
//   q2 = Matrix(length,length);
//   q3 = Matrix(length,length);
//   q4 = Matrix(length,length);

//     #pragma omp parallel for
//   for (size_t i = 0; i < length; ++i) {
//     #pragma omp parallel for
//     for (size_t j = 0; j < length; ++j) {
//       q1.data[i][j] = mainM.data[i][j];
//       q2.data[i][j] = mainM.data[i][j + length];
//       q3.data[i][j] = mainM.data[i + length][j];
//       q4.data[i][j] = mainM.data[i + length][j + length];
//     }
//   }
// }
// void add_matrix(const Matrix& matrix1,
//                 const Matrix& matrix2,
//                 Matrix& result
//                 )
// {
//     int length=matrix1.cols;
//     for (auto i = 0; i < length; i++)
//         for (auto j = 0; j < length; j++)
//             result[i][j]
//                 = matrix1[i][j] + matrix2[i][j];
// }

// Matrix add_matrix(const Matrix& matrix1,
//                 const Matrix& matrix2
//                 )
// {   
//     int length=matrix1.cols;
//     Matrix result(length,length);
//     for (auto i = 0; i < length; i++)
//         for (auto j = 0; j < length; j++)
//             result[i][j]
//                 = matrix1[i][j] + matrix2[i][j];
//     return result;
// }

// Matrix sub_matrix(const Matrix& matrix1,
//                 const Matrix& matrix2
//                 )
// {
//     int length=matrix1.cols;
//     Matrix result(length,length);
//     for (auto i = 0; i < length; i++)
//         for (auto j = 0; j < length; j++)
//             result[i][j]
//                 = matrix1[i][j] - matrix2[i][j];
//     return result;
// }
// void sub_matrix(const Matrix& matrix1,
//                 const Matrix& matrix2,
//                 Matrix& result
//                 )
// {
//     int length=matrix1.cols;
//     for (auto i = 0; i < length; i++)
//         for (auto j = 0; j < length; j++)
//             result[i][j]
//                 = matrix1[i][j] - matrix2[i][j];
// }

Matrix matrix_multiply_o(const Matrix& matrix1, const Matrix& matrix2) {
    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    int num;
    #pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        auto tmp3=matrix1[i];
        #pragma omp parallel for
        for (size_t k = 0; k < K; ++k) {
            int num = tmp3[k];
            auto tmp = matrix2[k];
            auto tmp2 = result[i];
            for (size_t j = 0; j < N; j += 8) {
                __m256i numVec = _mm256_set1_epi32(num);
                __m256i vecB = _mm256_loadu_si256((__m256i*)&tmp[j]);
                __m256i vecC = _mm256_loadu_si256((__m256i*)&tmp2[j]);
                __m256i product = _mm256_mullo_epi32(numVec, vecB);
                __m256i resultVec = _mm256_add_epi32(vecC, product);
                _mm256_storeu_si256((__m256i*)&tmp2[j], resultVec);
            }
        }
    }
    return result;
}

// Matrix matrix_multiply_openmp(const Matrix& matrix1, const Matrix& matrix2){
//     size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
//     if(M<=TILE_SIZE){
//         return matrix_multiply_o(move(matrix1),move(matrix2));
//     }else{
//         int length=M/2;
        
//         Matrix A11(length,length);
//         Matrix A12(length,length);
//         Matrix A21(length,length);
//         Matrix A22(length,length);
//         Matrix B11(length,length);
//         Matrix B12(length,length);
//         Matrix B21(length,length);
//         Matrix B22(length,length);
//         quadrant_split(matrix1,length,A11,A12,A21,A22);
//         quadrant_split(matrix2,length,B11,B12,B21,B22);

//         array<Matrix, 10>  S {
//             Matrix(length,length),
//             Matrix(length,length),
//             Matrix(length,length),
//             Matrix(length,length),
//             Matrix(length,length),
//             Matrix(length,length),
//             Matrix(length,length),
//             Matrix(length,length),
//             Matrix(length,length),
//             Matrix(length,length)
//         };
//         #pragma omp parallel for
//         for(int i=0;i<10;++i){
//             switch (i)
//             {
//             case 0:
//                 sub_matrix(B12,B22,S[i]);
//                 break;
//             case 1:
//                 add_matrix(A11,A12,S[i]);
//                 break;
//             case 2:
//                 add_matrix(A21,A22,S[i]);
//                 break;
//             case 3:
//                 sub_matrix(B21,B11,S[i]);
//                 break;
//             case 4:
//                 add_matrix(A11,A22,S[i]);
//                 break;

//             case 5:
//                 add_matrix(B11,B22,S[i]);
//                 break;
//             case 6:
//                 sub_matrix(A12,A22,S[i]);
//                 break;
//             case 7:
//                 add_matrix(B21,B22,S[i]);
//                 break;
//             case 8:
//                 sub_matrix(A11,A21,S[i]);
//                 break;
//             case 9:
//                 add_matrix(B11,B12,S[i]);
//                 break;
            
//             default:
//                 break;
//             }
//         }
//         array<Matrix, 7>  P {
//             Matrix(length,length),
//             Matrix(length,length),
//             Matrix(length,length),
//             Matrix(length,length),
//             Matrix(length,length),
//             Matrix(length,length),
//             Matrix(length,length)
//         };
//         #pragma omp parallel for
//         for(int i=0;i<7;++i){
//             switch (i)
//             {
//             case 0:
//                 P[i]=matrix_multiply_openmp(A11,S[0]);
//                 break;
//             case 1:
//                 P[i]=matrix_multiply_openmp(S[1],B22);
//                 break;
//             case 2:
//                 P[i]=matrix_multiply_openmp(S[2],B11);
//                 break;
//             case 3:
//                 P[i]=matrix_multiply_openmp(A22,S[3]);
//                 break;
//             case 4:
//                 P[i]=matrix_multiply_openmp(S[4],S[5]);
//                 break;
//             case 5:
//                 P[i]=matrix_multiply_openmp(S[6],S[7]);
//                 break;
//             case 6:
//                 P[i]=matrix_multiply_openmp(S[8],S[9]);
//                 break;
            
//             default:
//                 break;
//             }
//         }
//         Matrix C11=add_matrix((sub_matrix(add_matrix(P[4],P[3]),P[1])),P[5]);
//         Matrix C12=add_matrix(P[0],P[1]);
//         Matrix C21=add_matrix(P[2],P[3]);
//         Matrix C22=sub_matrix(add_matrix(P[4],P[0]),add_matrix(P[2],P[6]));


//         return quadrant_combine(C11,C12,C21,C22);
//     }
// }

int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 5) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable thread_num"
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    int thread_num = atoi(argv[1]);
    omp_set_num_threads(thread_num);

    const std::string matrix1_path = argv[2];

    const std::string matrix2_path = argv[3];

    const std::string result_path = argv[4];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_o(matrix1, matrix2);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    result.saveToFile(result_path);

    std::cout << "Output file to: " << result_path << std::endl;

    std::cout << "Multiplication Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds"
              << std::endl;

    return 0;
}