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

#define CUDA_CHECK(call) do { const cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

// Row-block distributed, right-looking Cholesky.  Each process keeps only its
// rows on its GPU; the current column is all-gathered before the rank-1 update.
__global__ void scale_column(double* a, int rows, int n, int first, int k, double diag) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    const int global = first + local;
    if (local < rows && global > k) a[(size_t)local * n + k] /= diag;
}

__global__ void trailing_update(double* a, const double* column, int rows, int n, int first, int k) {
    const int j = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    const int local = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = first + local;
    if (local < rows && i > k && j <= i && j < n)
        a[(size_t)local * n + j] -= a[(size_t)local * n + k] * column[j];
}

static void generatePositiveDefiniteMatrix(std::vector<double>& a, int n) {
    std::vector<double> b((size_t)n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < b.size(); ++i) b[i] = rand_r(&seed) / (double)RAND_MAX - .5;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j) {
            double sum = 0.;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < n; ++k) sum += b[(size_t)i*n+k] * b[(size_t)j*n+k];
            a[(size_t)i*n+j] = a[(size_t)j*n+i] = sum;
        }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) a[(size_t)i*n+i] += n;
}

static void usage(const char* p) { printf("Usage: %s [-n <num>] [-v] [-r] [-h]\n", p); }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int n = 512; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) usage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (n <= 0) { if (!rank) fprintf(stderr, "Matrix size must be positive\n"); MPI_Finalize(); return 1; }
    std::vector<int> rowCounts(ranks), displs(ranks), counts(ranks), elemDispls(ranks);
    for (int r=0; r<ranks; ++r) { rowCounts[r] = n/ranks + (r < n%ranks); displs[r] = r ? displs[r-1]+rowCounts[r-1] : 0; counts[r]=rowCounts[r]*n; elemDispls[r]=displs[r]*n; }
    const int rows = rowCounts[rank], first = displs[rank];
    std::vector<double> full, original, local((size_t)rows*n);
    if (!rank) { full.resize((size_t)n*n); generatePositiveDefiniteMatrix(full,n); if (validate) original=full; }
    MPI_Scatterv(rank ? nullptr : full.data(), counts.data(), elemDispls.data(), MPI_DOUBLE, local.data(), rows*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int devices=0; CUDA_CHECK(cudaGetDeviceCount(&devices)); if (!devices) { if (!rank) fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    CUDA_CHECK(cudaSetDevice(rank % devices));
    double *dA=nullptr, *dColumn=nullptr; CUDA_CHECK(cudaMalloc(&dA, local.size()*sizeof(double))); CUDA_CHECK(cudaMalloc(&dColumn, (size_t)n*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dA, local.data(), local.size()*sizeof(double), cudaMemcpyHostToDevice));
    if (!rank) { printf("Cholesky Decomposition Benchmark (MPI + OpenMP + CUDA)\nMatrix size: %d x %d, MPI ranks: %d\n",n,n,ranks); }
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now(); bool ok=true;
    std::vector<double> column(n,0.);
    for (int k=0; k<n; ++k) {
        const int owner = int(std::upper_bound(displs.begin(), displs.end(), k) - displs.begin()) - 1;
        double diag = 0.;
        if (rank == owner) CUDA_CHECK(cudaMemcpy(&diag, dA + (size_t)(k-first)*n+k, sizeof(double), cudaMemcpyDeviceToHost));
        MPI_Bcast(&diag,1,MPI_DOUBLE,owner,MPI_COMM_WORLD);
        if (diag <= 0.) { ok=false; break; } diag=std::sqrt(diag);
        if (rank == owner) CUDA_CHECK(cudaMemcpy(dA + (size_t)(k-first)*n+k, &diag, sizeof(double), cudaMemcpyHostToDevice));
        if (rows) { scale_column<<<(rows+255)/256,256>>>(dA,rows,n,first,k,diag); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
        // Each process contributes the part of column k stored in its row block.
        // Obtain contiguous column data with a packed device-to-host gather.
        std::vector<double> contribution(rows);
        for (int li=0; li<rows; ++li) CUDA_CHECK(cudaMemcpy(&contribution[li], dA+(size_t)li*n+k, sizeof(double), cudaMemcpyDeviceToHost));
        MPI_Allgatherv(contribution.data(),rows,MPI_DOUBLE,column.data(),rowCounts.data(),displs.data(),MPI_DOUBLE,MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(dColumn,column.data(),(size_t)n*sizeof(double),cudaMemcpyHostToDevice));
        if (rows && k + 1 < n) { dim3 block(16,16), grid((n-k-1+15)/16,(rows+15)/16);
            trailing_update<<<grid,block>>>(dA,dColumn,rows,n,first,k); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
    }
    CUDA_CHECK(cudaMemcpy(local.data(),dA,local.size()*sizeof(double),cudaMemcpyDeviceToHost));
    auto end=std::chrono::high_resolution_clock::now();
    int allok=ok; MPI_Allreduce(MPI_IN_PLACE,&allok,1,MPI_INT,MPI_LAND,MPI_COMM_WORLD);
    if (validate || printResults) { if (!rank) full.resize((size_t)n*n); MPI_Gatherv(local.data(),rows*n,MPI_DOUBLE,rank?nullptr:full.data(),counts.data(),elemDispls.data(),MPI_DOUBLE,0,MPI_COMM_WORLD); }
    if (!rank) {
        auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n",ms, (double)n*n*n/3.0/(std::max<long>(1,ms)/1000.)/1e9);
        if (printResults) print_results(full,"CholeskyL");
        if (validate) { double mx=0., rel=0.; for(int i=0;i<n;++i) for(int j=0;j<n;++j) { double s=0.; for(int q=0;q<=std::min(i,j);++q)s+=full[(size_t)i*n+q]*full[(size_t)j*n+q]; double e=fabs(s-original[(size_t)i*n+j]); mx=std::max(mx,e); rel=std::max(rel,e/(fabs(original[(size_t)i*n+j])+1e-10)); } printf("Max absolute error: %.10e\nMax relative error: %.10e\nValidation: %s\n",mx,rel,(allok&&rel<=1e-6)?"PASSED":"FAILED"); allok &= rel<=1e-6; }
    }
    CUDA_CHECK(cudaFree(dColumn)); CUDA_CHECK(cudaFree(dA)); MPI_Finalize(); return allok?0:1;
}
