//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// Reordering Matrix Multiplication
//

#include <stdexcept>
#include <chrono>
#include "matrix.hpp"
#define TILE_SIZE 256
using namespace std;
Matrix quadrant_combine(const Matrix& q1, const Matrix& q2, const Matrix& q3, const Matrix& q4) {
    if (q1.rows != q2.rows || q1.rows != q3.rows || q1.rows != q4.rows ||
        q1.cols != q2.cols || q1.cols != q3.cols || q1.cols != q4.cols) {
        // Handle error: quadrant dimensions must match
        // (Throw exception, return error, etc.)
        throw std::runtime_error("Quadrant dimensions must match for combination.");
    }

    size_t half_rows = q1.rows;
    size_t half_cols = q1.cols;
    size_t combined_rows = half_rows * 2;
    size_t combined_cols = half_cols * 2;


    Matrix combined(combined_rows, combined_cols);


    for (size_t i = 0; i < half_rows; ++i) {
        for (size_t j = 0; j < half_cols; ++j) {
            combined.data[i][j] = q1.data[i][j];
            combined.data[i][j + half_cols] = q2.data[i][j];
            combined.data[i + half_rows][j] = q3.data[i][j];
            combined.data[i + half_rows][j + half_cols] = q4.data[i][j];
        }
    }

    return combined;
}
void quadrant_split(const Matrix& mainM,int length, Matrix& q1, Matrix& q2, Matrix& q3, Matrix& q4) {
//   if (rows % 2 != 0 || cols % 2 != 0) {
//     // Handle error: rows and cols must be even
//     // You might throw an exception, return an error code, or print an error message.
//     // Example:
//     // throw std::runtime_error("Matrix dimensions must be even for quadrant split.");
//     return; // Or handle the error differently
//   }

  q1 = Matrix(length,length);
  q2 = Matrix(length,length);
  q3 = Matrix(length,length);
  q4 = Matrix(length,length);


  for (size_t i = 0; i < length; ++i) {
    for (size_t j = 0; j < length; ++j) {
      q1.data[i][j] = mainM.data[i][j];
      q2.data[i][j] = mainM.data[i][j + length];
      q3.data[i][j] = mainM.data[i + length][j];
      q4.data[i][j] = mainM.data[i + length][j + length];
    }
  }
}
void add_matrix(const Matrix& matrix1,
                const Matrix& matrix2,
                Matrix& result
                )
{
    int length=matrix1.cols;
    for (auto i = 0; i < length; i++)
        for (auto j = 0; j < length; j++)
            result[i][j]
                = matrix1[i][j] + matrix2[i][j];
}

Matrix add_matrix(const Matrix& matrix1,
                const Matrix& matrix2
                )
{   
    int length=matrix1.cols;
    Matrix result(length,length);
    for (auto i = 0; i < length; i++)
        for (auto j = 0; j < length; j++)
            result[i][j]
                = matrix1[i][j] + matrix2[i][j];
    return result;
}

Matrix sub_matrix(const Matrix& matrix1,
                const Matrix& matrix2
                )
{
    int length=matrix1.cols;
    Matrix result(length,length);
    for (auto i = 0; i < length; i++)
        for (auto j = 0; j < length; j++)
            result[i][j]
                = matrix1[i][j] - matrix2[i][j];
    return result;
}
void sub_matrix(const Matrix& matrix1,
                const Matrix& matrix2,
                Matrix& result
                )
{
    int length=matrix1.cols;
    for (auto i = 0; i < length; i++)
        for (auto j = 0; j < length; j++)
            result[i][j]
                = matrix1[i][j] - matrix2[i][j];
}

Matrix matrix_multiply_o(const Matrix& matrix1, const Matrix& matrix2) {
    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    int num;
    for (size_t i_block = 0; i_block < M; i_block += TILE_SIZE) {
        size_t I_max = std::min(i_block + TILE_SIZE, M);
        for (size_t k_block = 0; k_block < K; k_block += TILE_SIZE){
            size_t K_max = std::min(k_block + TILE_SIZE, M);
            for (size_t j_block = 0; j_block < N; j_block += TILE_SIZE) {
                    size_t J_max = std::min(j_block + TILE_SIZE, M);

                for (size_t i = i_block; i < I_max; ++i) {
                    for (size_t k = k_block; k < K_max; ++k) {
                        num = matrix1[i][k];
                        auto tmp=matrix2[k];
                        auto tmp2=result[i];
                        for (size_t j = j_block; j < J_max; ++j) {
                            tmp2[j] += num * tmp[j];
                        }
                    }
                }
            }   
        }
    }
    return result;
}

Matrix matrix_multiply_locality(const Matrix& matrix1, const Matrix& matrix2){
    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();
    if(M<=TILE_SIZE){
        return matrix_multiply_o(move(matrix1),move(matrix2));
    }else{
        int length=M/2;
        Matrix A11(length,length);
        Matrix A12(length,length);
        Matrix A21(length,length);
        Matrix A22(length,length);
        Matrix B11(length,length);
        Matrix B12(length,length);
        Matrix B21(length,length);
        Matrix B22(length,length);
        quadrant_split(matrix1,length,A11,A12,A21,A22);
        quadrant_split(matrix2,length,B11,B12,B21,B22);

        Matrix S1(length,length);
        sub_matrix(B12,B22,S1);
        Matrix S2(length,length);
        add_matrix(A11,A12,S2);
        Matrix S3(length,length);
        add_matrix(A21,A22,S3);
        Matrix S4(length,length);
        sub_matrix(B21,B11,S4);
        Matrix S5(length,length);
        add_matrix(A11,A22,S5);

        Matrix S6(length,length);
        add_matrix(B11,B22,S6);
        Matrix S7(length,length);
        sub_matrix(A12,A22,S7);
        Matrix S8(length,length);
        add_matrix(B21,B22,S8);
        Matrix S9(length,length);
        sub_matrix(A11,A21,S9);
        Matrix S10(length,length);
        add_matrix(B11,B12,S10);

        Matrix P1=matrix_multiply_locality(A11,S1);
        Matrix P2=matrix_multiply_locality(S2,B22);
        Matrix P3=matrix_multiply_locality(S3,B11);
        Matrix P4=matrix_multiply_locality(A22,S4);
        Matrix P5=matrix_multiply_locality(S5,S6);
        Matrix P6=matrix_multiply_locality(S7,S8);
        Matrix P7=matrix_multiply_locality(S9,S10);

        Matrix C11=add_matrix((sub_matrix(add_matrix(P5,P4),P2)),P6);
        Matrix C12=add_matrix(P1,P2);
        Matrix C21=add_matrix(P3,P4);
        Matrix C22=sub_matrix(add_matrix(P5,P1),add_matrix(P3,P7));


        return quadrant_combine(C11,C12,C21,C22);
    }
}


int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 4) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
    }

    const std::string matrix1_path = argv[1];

    const std::string matrix2_path = argv[2];

    const std::string result_path = argv[3];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);

    auto start_time = std::chrono::high_resolution_clock::now();

    Matrix result = matrix_multiply_locality(move(matrix1), move(matrix2));

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