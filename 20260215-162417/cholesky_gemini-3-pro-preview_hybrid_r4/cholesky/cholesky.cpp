#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

__global__ void extractAndScaleColumnK(double* A, double* send_buf, int n, int k, int size, int rank, int local_rows, double diag_inv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < local_rows) {
        int global_row = idx * size + rank;
        if (global_row > k) {
            double val = A[idx * n + k];
            val *= diag_inv;
            A[idx * n + k] = val;
            
            // Calculate start index in send_buf
            // start_idx is the first local index such that (local_idx * size + rank) > k
            // global > k  =>  local * size > k - rank
            // if k >= rank: local > (k - rank) / size  =>  local >= (k - rank)/size + 1
            // if k < rank:  local >= 0
            int start_idx;
            if (k >= rank) start_idx = (k - rank) / size + 1;
            else start_idx = 0;
            
            if (idx >= start_idx) {
                send_buf[idx - start_idx] = val;
            }
        }
    }
}

__global__ void updateTrailingSimple(double* A, const double* d_col_k, int n, int k, int size, int rank, int local_rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int global_row_i = idx * size + rank;
    
    if (idx < local_rows && global_row_i > k) {
        double l_ik = A[idx * n + k];
        for (int j = k + 1; j <= global_row_i; ++j) {
            A[idx * n + j] -= l_ik * d_col_k[j];
        }
    }
}

void generatePositiveDefiniteMatrixParallel(std::vector<double>& local_A, size_t n, int rank, int size) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    
    size_t local_rows = n / size + (rank < (int)(n % size) ? 1 : 0);
    local_A.resize(local_rows * n);
    
    #pragma omp parallel for
    for (size_t loc_i = 0; loc_i < local_rows; ++loc_i) {
        size_t global_i = loc_i * size + rank;
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[global_i * n + k] * B[j * n + k];
            }
            local_A[loc_i * n + j] = sum;
        }
        if (global_i < n) local_A[loc_i * n + global_i] += n;
    }
}

void gatherGlobalA(const std::vector<double>& local_A, std::vector<double>& global_A, size_t n, int rank, int size) {
    size_t local_rows = local_A.size() / n;
    int sendcount = local_rows * n;
    
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            size_t r_rows = n / size + (r < (int)(n % size) ? 1 : 0);
            recvcounts[r] = r_rows * n;
            displs[r] = (r == 0) ? 0 : displs[r-1] + recvcounts[r-1];
        }
        global_A.resize(n * n);
    }
    
    std::vector<double> global_buffer;
    if (rank == 0) global_buffer.resize(n * n);
    
    MPI_Gatherv(local_A.data(), sendcount, MPI_DOUBLE, 
                global_buffer.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 
                0, MPI_COMM_WORLD);
                
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            size_t r_rows = n / size + (r < (int)(n % size) ? 1 : 0);
            double* src = global_buffer.data() + displs[r];
            for (size_t loc = 0; loc < r_rows; ++loc) {
                size_t global = loc * size + r;
                memcpy(&global_A[global * n], src + loc * n, n * sizeof(double));
            }
        }
    }
}

bool validateCholeskyParallel(const std::vector<double>& local_A, size_t n, int rank, int size) {
    std::vector<double> L;
    gatherGlobalA(local_A, L, n, rank, size);
    
    if (rank == 0) {
        std::vector<double> A_orig(n * n);
        std::vector<double> B(n * n);
        unsigned int seed = 42;
        for (size_t i = 0; i < n * n; ++i) B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) sum += B[i * n + k] * B[j * n + k];
                A_orig[i * n + j] = sum;
            }
            A_orig[i * n + i] += n;
        }
        
        double maxError = 0.0, relError = 0.0;
        #pragma omp parallel for reduction(max:maxError, relError)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k <= std::min(i, j); ++k) {
                    double val_ik = (k > i) ? 0.0 : L[i*n + k];
                    double val_jk = (k > j) ? 0.0 : L[j*n + k];
                    sum += val_ik * val_jk;
                }
                double error = fabs(sum - A_orig[i*n + j]);
                if (error > maxError) maxError = error;
                double rel = error / (fabs(A_orig[i*n + j]) + 1e-10);
                if (rel > relError) relError = rel;
            }
        }
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
        return relError <= 1e-5;
    }
    return true;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark (Hybrid: MPI + OpenMP + CUDA)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI Ranks: %d\n", size);
        printf("OpenMP Threads: %d\n", omp_get_max_threads());
    }

    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 0) cudaSetDevice(rank % num_devices);

    if (rank == 0) printf("Generating matrix...\n");
    std::vector<double> local_A;
    generatePositiveDefiniteMatrixParallel(local_A, n, rank, size);

    size_t local_rows = local_A.size() / n;
    
    double *d_A, *d_send_buf, *d_col_k;
    CUDA_CHECK(cudaMalloc(&d_A, local_rows * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, local_A.data(), local_rows * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_send_buf, local_rows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_col_k, n * sizeof(double)));

    std::vector<double> host_send_buf(local_rows);
    std::vector<double> host_recv_buf(n);
    std::vector<double> sorted_col_k(n);
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t k = 0; k < n; ++k) {
        int root = k % size;
        size_t root_local_idx = k / size;
        
        double l_kk = 0.0;
        
        if (rank == root) {
            CUDA_CHECK(cudaMemcpy(&l_kk, d_A + root_local_idx * n + k, sizeof(double), cudaMemcpyDeviceToHost));
            if (l_kk <= 0.0) l_kk = -1.0;
            else {
                l_kk = sqrt(l_kk);
                CUDA_CHECK(cudaMemcpy(d_A + root_local_idx * n + k, &l_kk, sizeof(double), cudaMemcpyHostToDevice));
            }
        }
        
        MPI_Bcast(&l_kk, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
        if (l_kk < 0) {
            if (rank == 0) printf("Error: Matrix not positive definite at %zu\n", k);
            break;
        }

        int threads = 256;
        int blocks = (local_rows + threads - 1) / threads;
        extractAndScaleColumnK<<<blocks, threads>>>(d_A, d_send_buf, n, k, size, rank, local_rows, 1.0/l_kk);
        CUDA_CHECK(cudaGetLastError());

        int start_idx_gt_k = (k >= (size_t)rank) ? (k - rank) / size + 1 : 0;
        int my_count = (start_idx_gt_k < (int)local_rows) ? local_rows - start_idx_gt_k : 0;
        
        if (my_count > 0) {
            CUDA_CHECK(cudaMemcpy(host_send_buf.data(), d_send_buf, my_count * sizeof(double), cudaMemcpyDeviceToHost));
        }

        int total_count = 0;
        for (int r = 0; r < size; ++r) {
            int r_start = (k >= (size_t)r) ? (k - r) / size + 1 : 0;
            size_t r_rows = n / size + (r < (int)(n % size) ? 1 : 0);
            int r_count = (r_start < (int)r_rows) ? r_rows - r_start : 0;
            recvcounts[r] = r_count;
            displs[r] = total_count;
            total_count += r_count;
        }
        
        MPI_Allgatherv(host_send_buf.data(), my_count, MPI_DOUBLE,
                       host_recv_buf.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        #pragma omp parallel for
        for (int r = 0; r < size; ++r) {
            int count = recvcounts[r];
            int offset = displs[r];
            int r_start_idx = (k >= (size_t)r) ? (k - r) / size + 1 : 0;
            
            for (int i = 0; i < count; ++i) {
                int local_idx = r_start_idx + i;
                int global_idx = local_idx * size + r;
                sorted_col_k[global_idx] = host_recv_buf[offset + i];
            }
        }
        
        if (n > k + 1) {
            CUDA_CHECK(cudaMemcpy(d_col_k + k + 1, sorted_col_k.data() + k + 1, (n - k - 1) * sizeof(double), cudaMemcpyHostToDevice));
        }
        
        updateTrailingSimple<<<blocks, threads>>>(d_A, d_col_k, n, k, size, rank, local_rows);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    
    CUDA_CHECK(cudaMemcpy(local_A.data(), d_A, local_rows * n * sizeof(double), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_send_buf));
    CUDA_CHECK(cudaFree(d_col_k));

    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        std::vector<double> global_A;
        gatherGlobalA(local_A, global_A, n, rank, size);
        if (rank == 0) {
            print_results(global_A, "CholeskyL");
        }
    }

    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateCholeskyParallel(local_A, n, rank, size);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    
    MPI_Finalize();
    return 0;
}
