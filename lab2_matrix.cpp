#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <mpi.h>

constexpr int ROWS_A = 8;    // Строки матрицы A
constexpr int COLS_A = 5;    // Столбцы матрицы A (строки B)
constexpr int COLS_B = 3;    // Столбцы матрицы B
constexpr int BLOCK_ROWS = 2; // Блоки по 2 строки
constexpr int PROCESS_COUNT = 4; // 4 обрабатывающих процесса

void fillMatrix(double* matrix, int rows, int cols, int seed = 0) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(1.0, 10.0);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = std::round(dis(gen) * 10) / 10.0;
        }
    }
}

void printMatrix(double* matrix, int rows, int cols, const std::string& name) {
    std::cout << "\n" << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::fixed << std::setprecision(1) 
                      << std::setw(6) << matrix[i * cols + j];
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void sequentialMultiply(double* A, double* B, double* C, 
                       int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            C[i * colsB + j] = 0;
            for (int k = 0; k < colsA; k++) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (world_size != PROCESS_COUNT + 1) {
        if (world_rank == 0) {
            std::cerr << "Ошибка: требуется " << (PROCESS_COUNT + 1) 
                      << " процессов (1 корневой + " << PROCESS_COUNT << " рабочих)\n";
            std::cerr << "Запустите: mpiexec -n 5 ./lab2_matrix\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    double start_time = MPI_Wtime();
    
    if (world_rank == 0) {
        // КОРНЕВОЙ ПРОЦЕСС
        std::cout << "==================================================\n";
        std::cout << "ЛАБОРАТОРНАЯ РАБОТА №2 - Вариант 1\n";
        std::cout << "Параллельное перемножение матриц " 
                  << ROWS_A << "x" << COLS_A << " и " 
                  << COLS_A << "x" << COLS_B << "\n";
        std::cout << "==================================================\n";
        
        std::vector<double> A(ROWS_A * COLS_A);
        std::vector<double> B(COLS_A * COLS_B);
        std::vector<double> C_seq(ROWS_A * COLS_B, 0);
        std::vector<double> C_par(ROWS_A * COLS_B, 0);
        
        fillMatrix(A.data(), ROWS_A, COLS_A, 123);
        fillMatrix(B.data(), COLS_A, COLS_B, 456);
        
        printMatrix(A.data(), ROWS_A, COLS_A, "Матрица A");
        printMatrix(B.data(), COLS_A, COLS_B, "Матрица B");
        
        // Последовательное умножение
        sequentialMultiply(A.data(), B.data(), C_seq.data(), ROWS_A, COLS_A, COLS_B);
        printMatrix(C_seq.data(), ROWS_A, COLS_B, "Результат C (последовательный)");
        
        // Рассылка блоков матрицы A
        for (int proc = 1; proc < world_size; proc++) {
            int start_row = (proc - 1) * BLOCK_ROWS;
            double* block_start = A.data() + start_row * COLS_A;
            MPI_Send(block_start, BLOCK_ROWS * COLS_A, MPI_DOUBLE, 
                    proc, 0, MPI_COMM_WORLD);
            std::cout << "Корень -> Процесс " << proc 
                      << ": блок A[" << start_row << ":" 
                      << (start_row + BLOCK_ROWS - 1) << ",:]\n";
        }
        
        // Широковещательная рассылка матрицы B
        MPI_Bcast(B.data(), COLS_A * COLS_B, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Получение результатов
        for (int proc = 1; proc < world_size; proc++) {
            int start_row = (proc - 1) * BLOCK_ROWS;
            double* result_block = C_par.data() + start_row * COLS_B;
            MPI_Recv(result_block, BLOCK_ROWS * COLS_B, MPI_DOUBLE, 
                    proc, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Корень <- Процесс " << proc 
                      << ": блок C[" << start_row << ":" 
                      << (start_row + BLOCK_ROWS - 1) << ",:]\n";
        }
        
        printMatrix(C_par.data(), ROWS_A, COLS_B, "Результат C (параллельный)");
        
        // Проверка
        bool correct = true;
        for (int i = 0; i < ROWS_A * COLS_B; i++) {
            if (std::abs(C_seq[i] - C_par[i]) > 0.001) {
                correct = false;
                break;
            }
        }
        
        std::cout << "\n=== ПРОВЕРКА ===\n";
        if (correct) {
            std::cout << "✓ Результаты совпадают!\n";
        } else {
            std::cout << "✗ Ошибка: результаты не совпадают!\n";
        }
        
    } else {
        // РАБОЧИЕ ПРОЦЕССЫ
        int process_id = world_rank;
        
        std::vector<double> A_block(BLOCK_ROWS * COLS_A);
        MPI_Recv(A_block.data(), BLOCK_ROWS * COLS_A, MPI_DOUBLE, 
                0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::vector<double> B_full(COLS_A * COLS_B);
        MPI_Bcast(B_full.data(), COLS_A * COLS_B, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Вычисление блока C
        std::vector<double> C_block(BLOCK_ROWS * COLS_B, 0);
        for (int i = 0; i < BLOCK_ROWS; i++) {
            for (int j = 0; j < COLS_B; j++) {
                for (int k = 0; k < COLS_A; k++) {
                    C_block[i * COLS_B + j] += 
                        A_block[i * COLS_A + k] * B_full[k * COLS_B + j];
                }
            }
        }
        
        // Синхронизация
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Отправка результата
        MPI_Send(C_block.data(), BLOCK_ROWS * COLS_B, MPI_DOUBLE, 
                0, 1, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    
    if (world_rank == 0) {
        std::cout << "\n==================================================\n";
        std::cout << "Время выполнения: " << std::fixed << std::setprecision(4) 
                  << (end_time - start_time) * 1000 << " мс\n";
        std::cout << "==================================================\n";
    }
    
    MPI_Finalize();
    return 0;
}
