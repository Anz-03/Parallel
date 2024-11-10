//
// Created by Yang Yufan on 2023/10/07.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI + OpenMp + SIMD + Reordering Matrix Multiplication
//

#include <mpi.h>  // MPI Header
#include <omp.h> 
#include <immintrin.h>
#include <stdexcept>
#include <chrono>
#include "matrix.hpp"

#define MASTER 0

Matrix matrix_multiply_mpi(const Matrix& matrix1, const Matrix& matrix2, int taskid, int numtasks) {
    if (matrix1.getCols() != matrix2.getRows()) {
        throw std::invalid_argument(
            "Matrix dimensions are not compatible for multiplication.");
    }

    size_t M = matrix1.getRows(), K = matrix1.getCols(), N = matrix2.getCols();

    Matrix result(M, N);

    // Your Code Here!
    // Optimizing Matrix Multiplication 
    // In addition to OpenMP, SIMD, Memory Locality and Cache Missing,
    // Further Applying MPI
    // Note:
    // You can change the argument of the function 
    // for your convenience of task division
    size_t n_rows = M/numtasks;
    size_t star = taskid * n_rows;
    size_t en = (taskid == numtasks - 1) ? M : (taskid + 1) * n_rows;

    const size_t TILE_SIZE = 128;
    const size_t SIMD_WIDTH = 8; 
    int num;

    #pragma omp parallel for
    for (size_t i = star; i < en; ++i) {
        auto tmp1=matrix1[i];
        for (size_t k = 0; k < K; ++k) {
            int num = tmp1[k];
            auto tmp2 = matrix2[k];
            auto tmp3 = result[i];

            for (size_t j = 0; j < N; j += SIMD_WIDTH) {
                __m256i numVec = _mm256_set1_epi32(num);
                __m256i vecB = _mm256_loadu_si256((__m256i*)&tmp2[j]);
                __m256i vecC = _mm256_loadu_si256((__m256i*)&tmp3[j]);
                __m256i product = _mm256_mullo_epi32(numVec, vecB);
                __m256i resultVec = _mm256_add_epi32(vecC, product);
                _mm256_storeu_si256((__m256i*)&tmp3[j], resultVec);
            }
        }
    }
    return result;
}
int main(int argc, char** argv) {
    // Verify input argument format
    if (argc != 5) {
        throw std::invalid_argument(
            "Invalid argument, should be: ./executable thread_num "
            "/path/to/matrix1 /path/to/matrix2 /path/to/multiply_result\n");
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
    int thread_num = atoi(argv[1]);
    omp_set_num_threads(thread_num);

    // Read Matrix
    const std::string matrix1_path = argv[2];

    const std::string matrix2_path = argv[3];

    const std::string result_path = argv[4];

    Matrix matrix1 = Matrix::loadFromFile(matrix1_path);

    Matrix matrix2 = Matrix::loadFromFile(matrix2_path);
    
    size_t M = matrix1.getRows();
    int N = matrix2.getCols(); 
    int num = M/numtasks;

    auto start_time = std::chrono::high_resolution_clock::now();

    if (taskid == MASTER) {
        Matrix result = matrix_multiply_mpi(matrix1, matrix2, taskid, numtasks);

        for (int taskid = 1; taskid < numtasks; ++taskid) {
            int start_row = taskid * num;
            int end_row = (taskid == numtasks - 1) ? M : (taskid + 1) * num;
            for (int i = start_row; i < end_row; ++i) {
                MPI_Recv(&result[i][0], N, MPI_INT, taskid, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        result.saveToFile(result_path);

        std::cout << "Output file to: " << result_path << std::endl;
        std::cout << "Multiplication Complete!" << std::endl;
        std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds" << std::endl;

    } else {
        Matrix partial_result = matrix_multiply_mpi(matrix1, matrix2, taskid, numtasks);
        int start_row = taskid * num;
        int end_row = (taskid == numtasks - 1) ? M : (taskid + 1) * num;
        for (int i = start_row; i < end_row; ++i) {
            MPI_Send(&partial_result[i][0], N, MPI_INT, MASTER, i, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
