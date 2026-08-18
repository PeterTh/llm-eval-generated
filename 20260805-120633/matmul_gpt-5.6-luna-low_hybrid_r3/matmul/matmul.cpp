#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
#pragma omp parallel for collapse(2) schedule(static)
    for (long long i = 0; i < static_cast<long long>(N); ++i)
        for (long long j = 0; j < static_cast<long long>(N); ++j)
            mat[static_cast<size_t>(i) * N + static_cast<size_t>(j)] =
                getPseudoRndValue(N, static_cast<size_t>(i), static_cast<size_t>(j));
}

__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, size_t rows, size_t N) {
    constexpr int TILE = 32;
    __shared__ double as[TILE][TILE];
    __shared__ double bs[TILE][TILE];
    const size_t row = static_cast<size_t>(blockIdx.y) * TILE + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + threadIdx.x;
    double sum = 0.0;
    for (size_t t = 0; t < N; t += TILE) {
        as[threadIdx.y][threadIdx.x] = row < rows && t + threadIdx.x < N
            ? A[row * N + t + threadIdx.x] : 0.0;
        bs[threadIdx.y][threadIdx.x] = t + threadIdx.y < N && col < N
            ? B[(t + threadIdx.y) * N + col] : 0.0;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) sum += as[threadIdx.y][k] * bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < rows && col < N) C[row * N + col] = sum;
}

static void cudaCheck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error in %s: %s\n", where, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); }
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    constexpr size_t points[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) for (size_t pj = 0; pj < 5; ++pj) {
        const size_t i = points[pi] % N, j = points[pj] % N;
        double expected = 0.0;
        for (size_t k = 0; k < N; ++k) expected += A[i*N+k] * B[k*N+j];
        const double error = std::abs((C[i*N+j] - expected) / (expected + 1e-10));
        if (error > 1e-6) { std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n", i,j,expected,C[i*N+j],error); return false; }
    }
    return true;
}

void printUsage(const char* p) { std::printf("Usage: %s [options]\n  -n <num> Matrix size (default: 512)\n  -v Validate\n  -r Print results\n  -h Show this help\n", p); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    size_t N = 512; bool validate = false, printResults = false;
    for (int i=1; i<argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i+1<argc) N = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate=true;
        else if (!std::strcmp(argv[i], "-r")) printResults=true;
        else if (!std::strcmp(argv[i], "-h")) { if(rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (N == 0) { if(rank==0) std::fprintf(stderr,"Matrix size must be positive\n"); MPI_Finalize(); return 1; }
    int devices=0; cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (!devices) { if(rank==0) std::fprintf(stderr,"No CUDA device available\n"); MPI_Finalize(); return 1; }
    cudaCheck(cudaSetDevice(rank % devices), "cudaSetDevice");
    std::vector<double> A(N*N), B(N*N), C(N*N);
    if(rank==0) { initMatrix(A,N); initMatrix(B,N); }
    MPI_Bcast(A.data(), static_cast<int>(A.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), static_cast<int>(B.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const size_t first = static_cast<size_t>(rank) * N / nranks;
    const size_t last = static_cast<size_t>(rank+1) * N / nranks, rows = last-first;
    std::vector<double> local(rows*N);
    double *dA=nullptr,*dB=nullptr,*dC=nullptr;
    cudaCheck(cudaMalloc(&dA,A.size()*sizeof(double)), "cudaMalloc A"); cudaCheck(cudaMalloc(&dB,B.size()*sizeof(double)), "cudaMalloc B"); cudaCheck(cudaMalloc(&dC,local.size()*sizeof(double)), "cudaMalloc C");
    cudaCheck(cudaMemcpy(dA,A.data(),A.size()*sizeof(double),cudaMemcpyHostToDevice),"copy A"); cudaCheck(cudaMemcpy(dB,B.data(),B.size()*sizeof(double),cudaMemcpyHostToDevice),"copy B");
    MPI_Barrier(MPI_COMM_WORLD); const auto start=std::chrono::steady_clock::now();
    dim3 block(32,32), grid((N+31)/32,(rows+31)/32); matmulKernel<<<grid,block>>>(dA+first*N,dB,dC,rows,N); cudaCheck(cudaGetLastError(),"kernel"); cudaCheck(cudaDeviceSynchronize(),"kernel sync");
    const auto end=std::chrono::steady_clock::now(); cudaCheck(cudaMemcpy(local.data(),dC,local.size()*sizeof(double),cudaMemcpyDeviceToHost),"copy C");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    std::vector<int> counts(nranks), displs(nranks); for(int r=0;r<nranks;++r){ counts[r]=static_cast<int>((static_cast<size_t>(r+1)*N/nranks-static_cast<size_t>(r)*N/nranks)*N); displs[r]=static_cast<int>(static_cast<size_t>(r)*N/nranks*N); }
    MPI_Gatherv(local.data(),static_cast<int>(local.size()),MPI_DOUBLE,C.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    const double localSeconds = std::chrono::duration<double>(end-start).count();
    double seconds = 0.0; MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int status = 0;
    if(rank==0){ std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nComputation time: %ld ms\nPerformance: %.3f GFLOPS\n",N,N,validate?"enabled":"disabled",static_cast<long>(seconds*1000),2.0*N*N*N/seconds/1e9); if(printResults) print_results(C,"MatrixC"); if(validate){ const bool valid=validateResult(A,B,C,N); std::printf("Validating result...\nValidation: %s\n",valid?"PASSED":"FAILED"); status=valid?0:1; } }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize(); return status;
}
