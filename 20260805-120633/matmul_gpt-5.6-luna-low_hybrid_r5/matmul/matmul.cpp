#include <cuda_runtime.h>
#include <mpi.h>
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

void initMatrix(std::vector<double>& mat, const size_t rows, const size_t N,
                const size_t rowOffset = 0) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(rows); ++i)
        for (size_t j = 0; j < N; ++j)
            mat[static_cast<size_t>(i) * N + j] = getPseudoRndValue(N, rowOffset + i, j);
}

__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, size_t rows, size_t N) {
    constexpr int TILE = 32;
    __shared__ double As[TILE][TILE], Bs[TILE][TILE];
    const size_t row = static_cast<size_t>(blockIdx.y) * TILE + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + threadIdx.x;
    double sum = 0.0;
    for (size_t tile = 0; tile < N; tile += TILE) {
        As[threadIdx.y][threadIdx.x] = row < rows && tile + threadIdx.x < N
            ? A[row * N + tile + threadIdx.x] : 0.0;
        Bs[threadIdx.y][threadIdx.x] = tile + threadIdx.y < N && col < N
            ? B[(tile + threadIdx.y) * N + col] : 0.0;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE && tile + k < N; ++k) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < rows && col < N) C[row * N + col] = sum;
}

static void cudaCheck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    constexpr size_t points[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) for (size_t pj = 0; pj < 5; ++pj) {
        const size_t i = points[pi] % N, j = points[pj] % N;
        double expected = 0.0;
#pragma omp parallel for reduction(+:expected) schedule(static)
        for (long long k = 0; k < static_cast<long long>(N); ++k) expected += A[i*N+k] * B[static_cast<size_t>(k)*N+j];
        const double error = std::abs((C[i*N+j] - expected) / (expected + 1e-10));
        if (error > 1e-6) { std::printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n", i,j,expected,C[i*N+j],error); return false; }
    }
    return true;
}

void printUsage(const char* p) { std::printf("Usage: %s [-n num] [-v] [-r] [-h]\n", p); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, world; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &world);
    size_t N = 512; bool validate = false, printResults = false;
    for (int i=1; i<argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i+1<argc) N = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank==0) printUsage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (N == 0) { if (rank==0) std::fprintf(stderr, "N must be positive\n"); MPI_Finalize(); return 1; }
    const size_t base=N/world, extra=N%world, row0=rank*base+(rank<static_cast<int>(extra)?rank:extra);
    const size_t rows=base+(rank<static_cast<int>(extra));
    std::vector<double> A(rows*N), B(N*N), C(rows*N), allA, allC;
    if (rank==0) allA.resize(N*N); // retain the original global inputs on the root
    initMatrix(B,N,N); initMatrix(A,rows,N,row0);
    if (rank==0) initMatrix(allA,N,N);
    MPI_Bcast(B.data(), static_cast<int>(B.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int deviceCount=0; cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    cudaCheck(cudaSetDevice(rank % deviceCount), "cudaSetDevice");
    double *dA=nullptr,*dB=nullptr,*dC=nullptr;
    cudaCheck(cudaMalloc(&dA, std::max<size_t>(1,A.size())*sizeof(double)), "cudaMalloc A");
    cudaCheck(cudaMalloc(&dB, B.size()*sizeof(double)), "cudaMalloc B");
    cudaCheck(cudaMalloc(&dC, std::max<size_t>(1,C.size())*sizeof(double)), "cudaMalloc C");
    cudaCheck(cudaMemcpy(dA,A.data(),A.size()*sizeof(double),cudaMemcpyHostToDevice), "copy A");
    cudaCheck(cudaMemcpy(dB,B.data(),B.size()*sizeof(double),cudaMemcpyHostToDevice), "copy B");
    MPI_Barrier(MPI_COMM_WORLD); const auto start=std::chrono::high_resolution_clock::now();
    dim3 block(32,32), grid((N+31)/32,(rows+31)/32); matmulKernel<<<grid,block>>>(dA,dB,dC,rows,N);
    cudaCheck(cudaGetLastError(), "kernel launch"); cudaCheck(cudaDeviceSynchronize(), "kernel");
    cudaCheck(cudaMemcpy(C.data(),dC,C.size()*sizeof(double),cudaMemcpyDeviceToHost), "copy C");
    const auto end=std::chrono::high_resolution_clock::now(); cudaFree(dA); cudaFree(dB); cudaFree(dC);
    std::vector<int> counts(world), displs(world); for(int r=0;r<world;++r){size_t rr=base+(r<static_cast<int>(extra)); counts[r]=static_cast<int>(rr*N); displs[r]=static_cast<int>((r*base+(r<static_cast<int>(extra)?r:extra))*N);}
    if(rank==0) allC.resize(N*N); MPI_Gatherv(C.data(),counts[rank],MPI_DOUBLE,rank==0?allC.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    int validationStatus = 0;
    if(rank==0){ double sec=std::chrono::duration<double>(end-start).count(); std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nComputation time: %.0f ms\nPerformance: %.3f GFLOPS\n",N,N,sec*1000,2.0*N*N*N/sec/1e9); if(printResults) print_results(allC,"MatrixC"); if(validate) { const bool valid=validateResult(allA,B,allC,N); std::printf("Validation: %s\n",valid?"PASSED":"FAILED"); validationStatus=valid?0:1; } }
    MPI_Bcast(&validationStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize(); return validationStatus;
}
