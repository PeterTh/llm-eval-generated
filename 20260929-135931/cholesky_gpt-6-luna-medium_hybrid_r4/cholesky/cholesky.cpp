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

// Each MPI rank owns a cyclic set of matrix rows. CUDA updates those rows;
// the pivot row is broadcast at every step, which is the dependency boundary.
__global__ void update_column(double* a, const double* pivot, size_t n, size_t j,
                              int rank, int ranks) {
    const size_t q = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = n * n;
    if (q >= count) return;
    const size_t i = q / n, col = q % n;
    if (i <= j || col > j || int(i % ranks) != rank) return;
    double sum = 0.0;
    for (size_t k = 0; k < j; ++k) sum += a[i*n+k] * pivot[k];
    if (col == j) a[q] = (a[q] - sum) / pivot[j];
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n*n);
    unsigned int seed = 42;
    for (size_t i=0;i<n*n;++i) B[i]=(rand_r(&seed)/(double)RAND_MAX)-0.5;
    #pragma omp parallel for schedule(static)
    for (long long ii=0; ii<(long long)n; ++ii) {
        size_t i=(size_t)ii;
        for(size_t j=0;j<n;++j) {
            double s=0.0;
            for(size_t k=0;k<n;++k) s+=B[i*n+k]*B[j*n+k];
            A[i*n+j]=s;
        }
        A[i*n+i]+=n;
    }
}

bool choleskyDecomposition(std::vector<double>& A, size_t n, int rank, int ranks) {
    double *da=nullptr, *dp=nullptr;
    if (cudaMalloc(&da,n*n*sizeof(double)) != cudaSuccess || cudaMalloc(&dp,n*sizeof(double)) != cudaSuccess) return false;
    cudaMemcpy(da,A.data(),n*n*sizeof(double),cudaMemcpyHostToDevice);
    std::vector<double> pivot(n);
    bool ok=true;
    for(size_t j=0;j<n;++j) {
        const int owner=int(j%ranks);
        if(rank==owner) {
            double s=0.0;
            for(size_t k=0;k<j;++k) s+=A[j*n+k]*A[j*n+k];
            double v=A[j*n+j]-s;
            if(v<=0.0) ok=false;
            else A[j*n+j]=std::sqrt(v);
            for(size_t k=0;k<j;++k) pivot[k]=A[j*n+k];
            pivot[j]=ok?A[j*n+j]:0.0;
            for(size_t k=j+1;k<n;++k) pivot[k]=0.0;
        }
        MPI_Bcast(&ok,1,MPI_C_BOOL,owner,MPI_COMM_WORLD);
        if(!ok) break;
        MPI_Bcast(pivot.data(),(int)n,MPI_DOUBLE,owner,MPI_COMM_WORLD);
        // Distribute this column's values by exchanging updated rows in the
        // next pivot broadcast. Update all owned rows for column j on device.
        cudaMemcpy(dp,pivot.data(),n*sizeof(double),cudaMemcpyHostToDevice);
        update_column<<<(n*n+255)/256,256>>>(da,dp,n,j,rank,ranks);
        if(cudaDeviceSynchronize()!=cudaSuccess) { ok=false; break; }
        // Keep the owned host rows authoritative for MPI and subsequent pivots.
        // This OpenMP pass also avoids accumulating device/host synchronization
        // roundoff differently across ranks.
        #pragma omp parallel for schedule(static)
        for(long long ii=(long long)j+1;ii<(long long)n;++ii) {
            size_t i=(size_t)ii;
            if(int(i%ranks)!=rank) continue;
            double sum=0.0;
            for(size_t k=0;k<j;++k) sum+=A[i*n+k]*pivot[k];
            A[i*n+j]=(A[i*n+j]-sum)/pivot[j];
        }
        // Pivot owner supplies the row values required for later columns.
        MPI_Bcast(A.data()+j*n,(int)n,MPI_DOUBLE,owner,MPI_COMM_WORLD);
        cudaMemcpy(da,A.data(),n*n*sizeof(double),cudaMemcpyHostToDevice);
        A[j*n+j]=pivot[j];
    }
    cudaFree(da); cudaFree(dp);
    return ok;
}

bool validateCholesky(const std::vector<double>& L,const std::vector<double>& A,size_t n) {
    double maxError=0.0,relError=0.0;
    #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
    for(long long q=0;q<(long long)(n*n);++q) {
        size_t i=(size_t)q/n,j=(size_t)q%n; double s=0.0;
        for(size_t k=0;k<n;++k) s+=L[i*n+k]*L[j*n+k];
        double e=fabs(s-A[q]); maxError=std::max(maxError,e);
        relError=std::max(relError,e/(fabs(A[q])+1e-10));
    }
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n",maxError,relError);
    if(relError>1e-6) { printf("Validation failed: relative error too large\n"); return false; }
    return true;
}

void printUsage(const char* p) { printf("Usage: %s [-n num] [-v] [-r] [-h]\n",p); }
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv); int rank,ranks; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    size_t n=512; bool validate=false,printResults=false;
    for(int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"-n")&&i+1<argc) n=(size_t)atoi(argv[++i]);
        else if(!strcmp(argv[i],"-v")) validate=true;
        else if(!strcmp(argv[i],"-r")) printResults=true;
        else if(!strcmp(argv[i],"-h")) { if(rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) { printf("Unknown option: %s\n",argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    int deviceCount=0; cudaGetDeviceCount(&deviceCount); if(deviceCount>0) cudaSetDevice(rank%deviceCount);
    std::vector<double> A(n*n),original;
    if(rank==0) generatePositiveDefiniteMatrix(A,n);
    MPI_Bcast(A.data(),(int)(n*n),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(validate) original=A;
    if(rank==0) { printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nGenerating positive definite matrix...\nComputing Cholesky decomposition...\n",n,n,validate?"enabled":"disabled"); }
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
    bool success=choleskyDecomposition(A,n,rank,ranks);
    MPI_Allreduce(MPI_IN_PLACE,&success,1,MPI_C_BOOL,MPI_LAND,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD); auto end=std::chrono::high_resolution_clock::now();
    if(rank==0&&!success) printf("Cholesky decomposition failed\n");
    if(!success) { MPI_Finalize(); return 1; }
    // Form the complete factor from row owners for output and validation.
    std::vector<double> local(n*n,0.0),L(n*n);
    for(size_t i=0;i<n;++i) if(int(i%ranks)==rank) std::copy(A.begin()+i*n,A.begin()+(i+1)*n,local.begin()+i*n);
    MPI_Reduce(local.data(),L.data(),(int)(n*n),MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    if(rank==0) {
        for(size_t i=0;i<n;++i) for(size_t j=i+1;j<n;++j) L[i*n+j]=0.0;
        auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        printf("Computation time: %ld ms\n",(long)ms);
        double sec=ms/1000.0; printf("Performance: %.3f GFLOPS\n",sec>0?((double)n*n*n/3.0)/sec/1e9:0.0);
        if(printResults) print_results(L,"CholeskyL");
        if(validate) { printf("Validating result...\n"); bool v=validateCholesky(L,original,n); printf("Validation: %s\n",v?"PASSED":"FAILED"); if(!v) success=false; }
    }
    MPI_Bcast(&success,1,MPI_C_BOOL,0,MPI_COMM_WORLD); MPI_Finalize(); return success?0:1;
}
