#include <algorithm>
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

namespace {
[[noreturn]] void cudaFailure(cudaError_t e, const char* call, const char* file, int line) {
    std::fprintf(stderr, "CUDA error at %s:%d (%s): %s\n", file, line, call, cudaGetErrorString(e));
    MPI_Abort(MPI_COMM_WORLD, 1); std::abort();
}
#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) cudaFailure(e, #call, __FILE__, __LINE__); } while (0)

__global__ void diagonalKernel(double* a, std::size_t n, std::size_t j) {
    extern __shared__ double p[]; double sum = 0.0;
    for (std::size_t k = threadIdx.x; k < j; k += blockDim.x) sum += a[j*n+k] * a[j*n+k];
    p[threadIdx.x] = sum; __syncthreads();
    for (unsigned s = blockDim.x/2; s; s >>= 1) { if (threadIdx.x < s) p[threadIdx.x] += p[threadIdx.x+s]; __syncthreads(); }
    if (threadIdx.x == 0) { double v = a[j*n+j] - p[0]; a[j*n+j] = v > 0.0 ? sqrt(v) : -sqrt(-v); }
}

__global__ void updateRowsKernel(double* a, std::size_t n, std::size_t j, std::size_t first, std::size_t last) {
    extern __shared__ double p[]; const std::size_t i = first + blockIdx.x; if (i >= last) return;
    double sum = 0.0;
    for (std::size_t k = threadIdx.x; k < j; k += blockDim.x) sum += a[i*n+k] * a[j*n+k];
    p[threadIdx.x] = sum; __syncthreads();
    for (unsigned s = blockDim.x/2; s; s >>= 1) { if (threadIdx.x < s) p[threadIdx.x] += p[threadIdx.x+s]; __syncthreads(); }
    if (threadIdx.x == 0) a[i*n+j] = (a[i*n+j] - p[0]) / a[j*n+j];
}

__global__ void zeroUpperKernel(double* a, std::size_t n) {
    const std::size_t x = static_cast<std::size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    if (x/n < n && x%n > x/n) a[x] = 0.0;
}

std::size_t rowBegin(std::size_t n, int r, int p) { return n*static_cast<std::size_t>(r)/p; }
std::size_t rowEnd(std::size_t n, int r, int p) { return n*static_cast<std::size_t>(r+1)/p; }

bool choleskyDecomposition(std::vector<double>& a, std::size_t n, int rank, int ranks) {
    const std::size_t first = rowBegin(n, rank, ranks), last = rowEnd(n, rank, ranks);
    double* dA = nullptr; CUDA_CHECK(cudaMalloc(&dA, n*n*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dA, a.data(), n*n*sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double> pivot(n); bool success = true; constexpr unsigned threads = 256;
    for (std::size_t j = 0; j < n; ++j) {
        int owner = 0;
        while (owner + 1 < ranks && j >= rowEnd(n, owner, ranks)) ++owner;
        if (rank == owner) {
            diagonalKernel<<<1, threads, threads*sizeof(double)>>>(dA, n, j); CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&pivot[j], dA+j*n+j, sizeof(double), cudaMemcpyDeviceToHost));
            success = pivot[j] > 0.0 && std::isfinite(pivot[j]);
        }
        MPI_Bcast(&success, 1, MPI_C_BOOL, owner, MPI_COMM_WORLD);
        if (!success) { if (rank == 0) std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", j); CUDA_CHECK(cudaFree(dA)); return false; }
        MPI_Bcast(&pivot[j], 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (rank == owner) CUDA_CHECK(cudaMemcpy(pivot.data(), dA+j*n, (j+1)*sizeof(double), cudaMemcpyDeviceToHost));
        MPI_Bcast(pivot.data(), static_cast<int>(j+1), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(dA+j*n, pivot.data(), (j+1)*sizeof(double), cudaMemcpyHostToDevice));
        const std::size_t begin = std::max(first, j+1);
        if (begin < last) { updateRowsKernel<<<static_cast<unsigned>(last-begin), threads, threads*sizeof(double)>>>(dA,n,j,begin,last); CUDA_CHECK(cudaGetLastError()); }
    }
    const std::size_t elements = n*n;
    zeroUpperKernel<<<static_cast<unsigned>((elements+255)/256),256>>>(dA,n); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    const int count = static_cast<int>((last-first)*n); std::vector<int> counts(ranks), displacements(ranks);
    for (int r=0; r<ranks; ++r) { counts[r]=static_cast<int>((rowEnd(n,r,ranks)-rowBegin(n,r,ranks))*n); displacements[r]=static_cast<int>(rowBegin(n,r,ranks)*n); }
    CUDA_CHECK(cudaMemcpy(a.data()+first*n, dA+first*n, count*sizeof(double), cudaMemcpyDeviceToHost));
    MPI_Allgatherv(MPI_IN_PLACE,count,MPI_DOUBLE,a.data(),counts.data(),displacements.data(),MPI_DOUBLE,MPI_COMM_WORLD);
    CUDA_CHECK(cudaFree(dA)); return true;
}

void generatePositiveDefiniteMatrix(std::vector<double>& a, std::size_t n) {
    std::vector<double> b(n*n); unsigned seed=42;
    for (double& v : b) v = rand_r(&seed)/static_cast<double>(RAND_MAX)-0.5;
#pragma omp parallel for schedule(static)
    for (long long i=0; i<static_cast<long long>(n); ++i) { for (std::size_t j=0; j<n; ++j) { double sum=0.0;
#pragma omp simd reduction(+:sum)
        for (std::size_t k=0; k<n; ++k) sum += b[i*n+k]*b[j*n+k]; a[i*n+j]=sum; } a[i*n+i]+=static_cast<double>(n); }
}

bool validateCholesky(const std::vector<double>& l, const std::vector<double>& original, std::size_t n) {
    double maxError=0.0, relativeError=0.0;
#pragma omp parallel for reduction(max:maxError,relativeError) schedule(static)
    for (long long x=0; x<static_cast<long long>(n*n); ++x) { const std::size_t i=x/n, j=x%n; double sum=0.0;
#pragma omp simd reduction(+:sum)
        for (std::size_t k=0; k<n; ++k) sum += l[i*n+k]*l[j*n+k]; const double e=std::fabs(sum-original[x]); maxError=std::max(maxError,e); relativeError=std::max(relativeError,e/(std::fabs(original[x])+1e-10)); }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n",maxError,relativeError);
    if (relativeError>1e-6) { std::printf("Validation failed: relative error too large\n"); return false; } return true;
}

void printUsage(const char* name) { std::printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n",name); }
} // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided); int rank=0,ranks=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    std::size_t n=512; bool validate=false,printResults=false;
    for(int i=1;i<argc;++i) { if(!std::strcmp(argv[i],"-n")&&i+1<argc) n=std::strtoull(argv[++i],nullptr,10); else if(!std::strcmp(argv[i],"-v")) validate=true; else if(!std::strcmp(argv[i],"-r")) printResults=true; else if(!std::strcmp(argv[i],"-h")){if(rank==0)printUsage(argv[0]);MPI_Finalize();return 0;} else {if(rank==0){std::printf("Unknown option: %s\n",argv[i]);printUsage(argv[0]);}MPI_Finalize();return 1;} }
    if(!n||n>static_cast<std::size_t>(std::numeric_limits<int>::max())||n*n>static_cast<std::size_t>(std::numeric_limits<int>::max())) MPI_Abort(MPI_COMM_WORLD,1);
    int devices=0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices <= 0) { if (rank == 0) std::fprintf(stderr, "No CUDA device is available.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    const char* lr=std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"); int local=lr?std::atoi(lr):rank; CUDA_CHECK(cudaSetDevice(local%devices));
    std::vector<double> a(n*n),original; if(rank==0){std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nGenerating positive definite matrix...\n",n,n,validate?"enabled":"disabled");generatePositiveDefiniteMatrix(a,n);if(validate)original=a;}
    MPI_Bcast(a.data(),static_cast<int>(n*n),MPI_DOUBLE,0,MPI_COMM_WORLD); if(validate&&rank!=0)original=a; if(rank==0)std::printf("Computing Cholesky decomposition...\n"); MPI_Barrier(MPI_COMM_WORLD); const double start=MPI_Wtime(); bool ok=choleskyDecomposition(a,n,rank,ranks); const double elapsed=MPI_Wtime()-start, gflops=(static_cast<double>(n)*n*n/3.0)/1e9; double worst=0.0; MPI_Reduce(&elapsed,&worst,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD); if(!ok){MPI_Finalize();return 1;}
    if(rank==0){std::printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n",static_cast<long>(worst*1000.0),gflops/worst);if(printResults)print_results(a,"CholeskyL");if(validate){std::printf("Validating result...\n");bool valid=validateCholesky(a,original,n);std::printf("Validation: %s\n",valid?"PASSED":"FAILED");MPI_Finalize();return valid?0:1;}} MPI_Finalize();return 0;
}
