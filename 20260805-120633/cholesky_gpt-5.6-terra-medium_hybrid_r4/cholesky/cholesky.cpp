#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// A block is owned by one rank for its entire lifetime.  All ranks retain the
// input, but only touch their owned block rows; this avoids an all-to-all for
// every rank-k update while retaining simple, scalable panel broadcasts.
constexpr int BLOCK = 64;

#define CUDA_CHECK(call) do { \
    cudaError_t e_ = (call); \
    if (e_ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
        MPI_Abort(MPI_COMM_WORLD, 2); \
    } \
} while (0)

__global__ void factor_panel(double* a, int n, int k, int p, int* failed) {
    // Panel sizes are deliberately small; sequential dependencies make one
    // thread the most efficient synchronization mechanism here.
    if (blockIdx.x || threadIdx.x) return;
    for (int j = 0; j < p; ++j) {
        double sum = 0.0;
        for (int q = 0; q < j; ++q) {
            const double x = a[(k + j) * n + k + q];
            sum += x * x;
        }
        const int diag = (k + j) * n + k + j;
        const double v = a[diag] - sum;
        if (v <= 0.0) { *failed = 1; return; }
        a[diag] = sqrt(v);
        for (int i = j + 1; i < p; ++i) {
            double s = 0.0;
            for (int q = 0; q < j; ++q)
                s += a[(k + i) * n + k + q] * a[(k + j) * n + k + q];
            a[(k + i) * n + k + j] = (a[(k + i) * n + k + j] - s) / a[diag];
        }
    }
}

__global__ void solve_panel_rows(double* a, int n, int k, int p, int begin, int end) {
    const int row = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= end) return;
    for (int j = 0; j < p; ++j) {
        double sum = 0.0;
        for (int q = 0; q < j; ++q) sum += a[row * n + k + q] * a[(k + j) * n + k + q];
        a[row * n + k + j] = (a[row * n + k + j] - sum) / a[(k + j) * n + k + j];
    }
}

__global__ void update_block(double* a, int n, int k, int p, int row0, int rows, int col0, int cols) {
    const int i = row0 + blockIdx.y * blockDim.y + threadIdx.y;
    const int j = col0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= row0 + rows || j >= col0 + cols || j > i) return;
    double sum = 0.0;
    #pragma unroll 4
    for (int q = 0; q < p; ++q) sum += a[i * n + k + q] * a[j * n + k + q];
    a[i * n + j] -= sum;
}

void generatePositiveDefiniteMatrix(std::vector<double>& a, size_t n) {
    std::vector<double> b(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) b[i] = rand_r(&seed) / static_cast<double>(RAND_MAX) - .5;
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t q = 0; q < n; ++q) s += b[i * n + q] * b[j * n + q];
            a[i * n + j] = s + (static_cast<size_t>(i) == j ? n : 0.0);
        }
}

bool validateCholesky(const std::vector<double>& l, const std::vector<double>& original, size_t n) {
    double maxAbs = 0.0, maxRel = 0.0;
    #pragma omp parallel for reduction(max:maxAbs,maxRel) schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t q = 0; q <= std::min(static_cast<size_t>(i), j); ++q) s += l[i*n+q] * l[j*n+q];
            const double err = std::fabs(s - original[i*n+j]);
            maxAbs = std::max(maxAbs, err);
            maxRel = std::max(maxRel, err / (std::fabs(original[i*n+j]) + 1e-10));
        }
    }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxAbs, maxRel);
    return maxRel <= 1e-6;
}

void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -n <num>  Matrix size (default: 512)\n  -v        Enable validation\n  -r        Print results for external validation\n  -h        Show this help message\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t n = 512; bool validate = false, printResults = false;
    for (int i=1; i<argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i+1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!n || n > static_cast<size_t>(std::numeric_limits<int>::max()) || n*n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (!rank) std::fprintf(stderr, "Matrix size is unsupported\n"); MPI_Finalize(); return 1;
    }
    int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) std::fprintf(stderr, "A CUDA device is required\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(rank % devices));

    const int nn = static_cast<int>(n);
    std::vector<double> a(n*n), original;
    if (!rank) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nMPI ranks: %d, CUDA block size: %d\nGenerating positive definite matrix...\n", n, n, validate ? "enabled" : "disabled", ranks, BLOCK);
        generatePositiveDefiniteMatrix(a, n);
        if (validate) original = a;
    }
    MPI_Bcast(a.data(), nn*nn, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double* d = nullptr; int* dFailed = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n*n*sizeof(double))); CUDA_CHECK(cudaMalloc(&dFailed, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, a.data(), n*n*sizeof(double), cudaMemcpyHostToDevice));
    MPI_Barrier(MPI_COMM_WORLD); const double start = MPI_Wtime();
    int failed = 0;
    const int blocks = (nn + BLOCK - 1) / BLOCK;
    for (int kb=0; kb<blocks; ++kb) {
        const int k=kb*BLOCK, p=std::min(BLOCK, nn-k), owner=kb%ranks;
        if (rank == owner) {
            CUDA_CHECK(cudaMemset(dFailed, 0, sizeof(int))); factor_panel<<<1,1>>>(d,nn,k,p,dFailed);
            CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaMemcpy(&failed,dFailed,sizeof(int),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(a.data()+static_cast<size_t>(k)*nn, d+static_cast<size_t>(k)*nn, static_cast<size_t>(p)*nn*sizeof(double), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(&failed, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (failed) break;
        // Broadcast the complete panel rows: needed both for TRSM and updates.
        MPI_Bcast(a.data()+static_cast<size_t>(k)*nn, p*nn, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (rank != owner) CUDA_CHECK(cudaMemcpy(d+static_cast<size_t>(k)*nn, a.data()+static_cast<size_t>(k)*nn, static_cast<size_t>(p)*nn*sizeof(double), cudaMemcpyHostToDevice));
        for (int ib=kb+1; ib<blocks; ++ib) if (ib%ranks == rank) {
            const int row=ib*BLOCK, rows=std::min(BLOCK,nn-row);
            solve_panel_rows<<<(rows+255)/256,256>>>(d,nn,k,p,row,row+rows);
            if (k + p < nn) {
                const dim3 threads(16,16), grid((nn-k-p+15)/16,(rows+15)/16);
                update_block<<<grid,threads>>>(d,nn,k,p,row,rows,k+p,nn-k-p);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    // Gather only the owning rank's final block rows to rank zero.
    for (int ib=0; ib<blocks; ++ib) {
        const int row=ib*BLOCK, rows=std::min(BLOCK,nn-row), owner=ib%ranks;
        if (rank == owner) CUDA_CHECK(cudaMemcpy(a.data()+static_cast<size_t>(row)*nn,d+static_cast<size_t>(row)*nn,static_cast<size_t>(rows)*nn*sizeof(double),cudaMemcpyDeviceToHost));
        MPI_Bcast(a.data()+static_cast<size_t>(row)*nn, rows*nn, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    }
    double elapsed=MPI_Wtime()-start, maxElapsed=0; MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    CUDA_CHECK(cudaFree(d)); CUDA_CHECK(cudaFree(dFailed));
    int result = failed ? 1 : 0;
    if (!rank) {
        if (failed) std::printf("Cholesky decomposition failed\n");
        else {
            #pragma omp parallel for schedule(static)
            for (long long i = 0; i < static_cast<long long>(n); ++i)
                for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j) a[i*n+j] = 0.0;
            std::printf("Computation time: %.3f ms\nPerformance: %.3f GFLOPS\n", maxElapsed*1000.0, (double(n)*n*n/3.0)/(maxElapsed*1e9));
            if (printResults) print_results(a, "CholeskyL");
            if (validate) { const bool ok=validateCholesky(a,original,n); std::printf("Validation: %s\n",ok?"PASSED":"FAILED"); result=ok?0:1; }
        }
    }
    MPI_Bcast(&result,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Finalize(); return result;
}
