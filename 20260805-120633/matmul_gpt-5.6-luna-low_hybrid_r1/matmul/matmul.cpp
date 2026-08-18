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

void initMatrix(std::vector<double>& mat, size_t N) {
    #pragma omp parallel for schedule(static)
    for (long long x = 0; x < static_cast<long long>(N * N); ++x)
        mat[static_cast<size_t>(x)] = getPseudoRndValue(N, static_cast<size_t>(x) / N, static_cast<size_t>(x) % N);
}

__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, size_t N, size_t firstRow, size_t rows) {
    constexpr int TILE = 32;
    __shared__ double As[TILE][TILE], Bs[TILE][TILE];
    const size_t row = firstRow + blockIdx.y * TILE + threadIdx.y;
    const size_t col = blockIdx.x * TILE + threadIdx.x;
    double sum = 0.0;
    for (size_t tile = 0; tile < N; tile += TILE) {
        const size_t ak = tile + threadIdx.x, bk = tile + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < firstRow + rows && ak < N) ? A[row * N + ak] : 0.0;
        Bs[threadIdx.y][threadIdx.x] = (bk < N && col < N) ? B[bk * N + col] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < firstRow + rows && col < N) C[(row - firstRow) * N + col] = sum;
}

static void checkCuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& localC,
                    size_t N, size_t firstRow, size_t rows, int rank, int deviceCount) {
    checkCuda(cudaSetDevice(rank % deviceCount), "cudaSetDevice");
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    checkCuda(cudaMalloc(&dA, A.size() * sizeof(double)), "cudaMalloc(A)");
    checkCuda(cudaMalloc(&dB, B.size() * sizeof(double)), "cudaMalloc(B)");
    checkCuda(cudaMalloc(&dC, localC.size() * sizeof(double)), "cudaMalloc(C)");
    checkCuda(cudaMemcpy(dA, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice), "copy A");
    checkCuda(cudaMemcpy(dB, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice), "copy B");
    dim3 block(32, 32), grid((N + 31) / 32, (rows + 31) / 32);
    matmulKernel<<<grid, block>>>(dA, dB, dC, N, firstRow, rows);
    checkCuda(cudaGetLastError(), "matmulKernel launch");
    checkCuda(cudaDeviceSynchronize(), "matmulKernel");
    checkCuda(cudaMemcpy(localC.data(), dC, localC.size() * sizeof(double), cudaMemcpyDeviceToHost), "copy C");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

bool validateResult(const std::vector<double>& A, const std::vector<double>& B, const std::vector<double>& C, size_t N) {
    bool valid = true;
    #pragma omp parallel for collapse(2) reduction(&:valid)
    for (int pi = 0; pi < 5; ++pi) for (int pj = 0; pj < 5; ++pj) {
        size_t i = static_cast<size_t>(pi) % N, j = static_cast<size_t>(pj) % N; double expected = 0.0;
        for (size_t k = 0; k < N; ++k) expected += A[i*N+k] * B[k*N+j];
        double error = std::abs((C[i*N+j] - expected) / (expected + 1e-10));
        if (error > 1e-6) valid = false;
    }
    return valid;
}

void printUsage(const char* p) { std::printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results\n  -h           Show this help\n", p); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv); int rank, world; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &world);
    size_t N = 512; bool validate = false, printResults = false;
    for (int i=1; i<argc; ++i) { if (!std::strcmp(argv[i], "-n") && i+1<argc) N=std::strtoull(argv[++i],nullptr,10); else if (!std::strcmp(argv[i],"-v")) validate=true; else if (!std::strcmp(argv[i],"-r")) printResults=true; else if (!std::strcmp(argv[i],"-h")) { if(rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; } else { if(rank==0) printUsage(argv[0]); MPI_Finalize(); return 1; } }
    int devices=0; checkCuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount"); if (!devices) { if(rank==0) std::fprintf(stderr,"No CUDA devices found\n"); MPI_Finalize(); return 2; }
    size_t first = N * static_cast<size_t>(rank) / world, last = N * static_cast<size_t>(rank+1) / world, rows=last-first;
    std::vector<double> A(N*N), B(N*N), localC(rows*N), C(N*N);
    initMatrix(A,N); initMatrix(B,N); MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
    matrixMultiply(A,B,localC,N,first,rows,rank,devices);
    std::vector<int> counts(world), displs(world); for(int r=0;r<world;++r){ size_t f=N*static_cast<size_t>(r)/world,l=N*static_cast<size_t>(r+1)/world; counts[r]=static_cast<int>((l-f)*N); displs[r]=static_cast<int>(f*N); }
    MPI_Gatherv(localC.data(), static_cast<int>(localC.size()), MPI_DOUBLE, C.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    auto end=std::chrono::high_resolution_clock::now(); double seconds=std::chrono::duration<double>(end-start).count(), total=0; MPI_Reduce(&seconds,&total,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0){ std::printf("Matrix Multiplication Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nComputation time: %ld ms\nPerformance: %.3f GFLOPS\n",N,N,validate?"enabled":"disabled",static_cast<long>(total*1000),2.0*N*N*N/total/1e9); if(printResults) print_results(C,"MatrixC"); if(validate) { std::printf("Validating result...\nValidation: %s\n",validateResult(A,B,C,N)?"PASSED":"FAILED"); } }
    bool ok=true; if(rank==0 && validate) ok=validateResult(A,B,C,N); MPI_Bcast(&ok,1,MPI_C_BOOL,0,MPI_COMM_WORLD); MPI_Finalize(); return ok?0:1;
}
