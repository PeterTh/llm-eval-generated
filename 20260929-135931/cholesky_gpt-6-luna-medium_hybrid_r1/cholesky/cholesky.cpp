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

// Each MPI rank owns a contiguous set of rows. At step k, the owner of row k
// broadcasts the panel row; every GPU updates its local rows independently.
__global__ void normalize_column(double* a, const double* panel, size_t n,
                                 size_t first, size_t count, size_t k) {
    const size_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < count) a[r*n+k] /= panel[k];
}

__global__ void update_rows(double* a, const double* panel, size_t n,
                            size_t first, size_t count, size_t k) {
    const size_t q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= count * n) return;
    const size_t r = q / n, j = q % n, i = first + r;
    if (i <= k || j <= k || j >= n) return;
    const double lik = a[r * n + k];
    a[r * n + j] -= lik * panel[j];
}

static void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

bool choleskyDecomposition(std::vector<double>& local, size_t n, size_t first,
                           int rank, int ranks, int device) {
    const size_t rows = local.size() / n;
    int local_rank = 0;
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &local_rank);
    int devices = 0; cuda_check(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    cuda_check(cudaSetDevice((local_rank + device) % devices));
    double *d_a = nullptr, *d_panel = nullptr;
    cuda_check(cudaMalloc(&d_a, std::max<size_t>(1, local.size()) * sizeof(double)));
    cuda_check(cudaMalloc(&d_panel, n * sizeof(double)));
    if (!local.empty()) cuda_check(cudaMemcpy(d_a, local.data(), local.size()*sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double> panel(n);
    bool ok = true;
    for (size_t k = 0; k < n; ++k) {
        const int owner = std::min(ranks - 1, static_cast<int>((k * ranks) / n));
        if (rank == owner) {
            const size_t off = (k - first) * n;
            double sum = 0.0;
            for (size_t j = 0; j < k; ++j) sum += local[off+j] * local[off+j];
            const double d = local[off+k] - sum;
            if (!(d > 0.0)) { ok = false; panel[k] = 0.0; }
            else {
                panel[k] = std::sqrt(d); local[off+k] = panel[k];
                for (size_t j = k + 1; j < n; ++j) local[off+j] /= panel[k];
            }
            if (ok) {
                // The row is already on the device and has received all previous updates.
                cuda_check(cudaMemcpy(d_a + (k-first)*n, local.data()+off, n*sizeof(double), cudaMemcpyHostToDevice));
                cuda_check(cudaMemcpy(panel.data(), local.data()+off, n*sizeof(double), cudaMemcpyHostToHost));
                panel[k] = std::sqrt(d);
            }
        }
        int good = ok ? 1 : 0;
        MPI_Bcast(&good, 1, MPI_INT, owner, MPI_COMM_WORLD);
        MPI_Bcast(panel.data(), static_cast<int>(n), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (!good) { ok = false; break; }
        cuda_check(cudaMemcpy(d_panel, panel.data(), n*sizeof(double), cudaMemcpyHostToDevice));
        const size_t work_rows = (first + rows > k + 1) ? first + rows - std::max(first, k+1) : 0;
        if (work_rows) {
            const size_t start = std::max(first, k+1);
            const size_t offset_rows = start - first;
            normalize_column<<<static_cast<unsigned>((work_rows+255)/256),256>>>(d_a+offset_rows*n, d_panel, n, start, work_rows, k);
            cuda_check(cudaGetLastError());
            update_rows<<<static_cast<unsigned>((work_rows*n+255)/256),256>>>(d_a+offset_rows*n, d_panel, n, start, work_rows, k);
            cuda_check(cudaGetLastError()); cuda_check(cudaDeviceSynchronize());
            cuda_check(cudaMemcpy(local.data()+offset_rows*n, d_a+offset_rows*n, work_rows*n*sizeof(double), cudaMemcpyDeviceToHost));
        }
    }
    if (ok) {
        // Ensure every rank's full local rows are current for the final gather.
        if (!local.empty()) cuda_check(cudaMemcpy(local.data(), d_a, local.size()*sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t r=0;r<rows;++r) for(size_t j=first+r+1;j<n;++j) local[r*n+j]=0.0;
    }
    cudaFree(d_panel); cudaFree(d_a); MPI_Comm_free(&node);
    return ok;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n*n);
    unsigned int seed=42;
    for(size_t i=0;i<n*n;++i) B[i]=(rand_r(&seed)/(double)RAND_MAX)-0.5;
    #pragma omp parallel for schedule(static)
    for(long long ij=0;ij<static_cast<long long>(n*n);++ij) {
        size_t i=ij/n,j=ij%n; double sum=0;
        for(size_t k=0;k<n;++k) sum+=B[i*n+k]*B[j*n+k];
        A[ij]=sum;
    }
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<static_cast<long long>(n);++i) A[i*n+i]+=n;
}

bool validateCholesky(const std::vector<double>& L,const std::vector<double>& A,size_t n) {
    double maxError=0, relError=0;
    #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
    for(long long ij=0;ij<static_cast<long long>(n*n);++ij) {
        size_t i=ij/n,j=ij%n; double sum=0;
        for(size_t k=0;k<n;++k) sum+=L[i*n+k]*L[j*n+k];
        double e=fabs(sum-A[ij]); maxError=std::max(maxError,e);
        relError=std::max(relError,e/(fabs(A[ij])+1e-10));
    }
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n",maxError,relError);
    return relError<=1e-6;
}

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv); int rank=0,ranks=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    size_t n=512; bool validate=false,printResults=false; int device=0;
    for(int i=1;i<argc;++i) {
        if(strcmp(argv[i],"-n")==0&&i+1<argc) n=static_cast<size_t>(strtoull(argv[++i],nullptr,10));
        else if(strcmp(argv[i],"-v")==0) validate=true;
        else if(strcmp(argv[i],"-r")==0) printResults=true;
        else if(strcmp(argv[i],"-h")==0) { if(rank==0) printf("Usage: %s [-n size] [-v] [-r] [-h]\n",argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) fprintf(stderr,"Unknown option: %s\n",argv[i]); MPI_Finalize(); return 1; }
    }
    if(n==0 || n>static_cast<size_t>(INT_MAX)) { if(rank==0) fprintf(stderr,"Invalid matrix size\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    if(rank==0) { printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nGenerating positive definite matrix...\n",n,n,validate?"enabled":"disabled"); }
    const size_t first=(n*rank)/ranks, last=(n*(rank+1))/ranks, rows=last-first;
    std::vector<double> full;
    if(rank==0) { full.resize(n*n); generatePositiveDefiniteMatrix(full,n); }
    std::vector<double> original;
    if(rank==0&&validate) original=full;
    std::vector<int> counts(ranks),displs(ranks);
    for(int r=0;r<ranks;++r) { counts[r]=static_cast<int>((n*(r+1)/ranks-n*r/ranks)*n); displs[r]=static_cast<int>((n*r/ranks)*n); }
    std::vector<double> local(rows*n);
    MPI_Scatterv(rank==0?full.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,local.data(),static_cast<int>(local.size()),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(rank==0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
    bool success=choleskyDecomposition(local,n,first,rank,ranks,device);
    int allok=success; MPI_Allreduce(MPI_IN_PLACE,&allok,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    std::vector<double> result(rank==0?n*n:0);
    MPI_Gatherv(local.data(),static_cast<int>(local.size()),MPI_DOUBLE,rank==0?result.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    auto end=std::chrono::high_resolution_clock::now();
    if(rank==0) {
        if(!allok) { fprintf(stderr,"Cholesky decomposition failed\n"); MPI_Abort(MPI_COMM_WORLD,1); }
        long ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        printf("Computation time: %ld ms\n",ms);
        double sec=std::chrono::duration<double>(end-start).count(); printf("Performance: %.3f GFLOPS\n",(double)n*n*n/3.0/sec/1e9);
        if(printResults) print_results(result,"CholeskyL");
        if(validate) { printf("Validating result...\n"); bool valid=validateCholesky(result,original,n); printf("Validation: %s\n",valid?"PASSED":"FAILED"); MPI_Finalize(); return valid?0:1; }
    }
    MPI_Finalize(); return 0;
}
