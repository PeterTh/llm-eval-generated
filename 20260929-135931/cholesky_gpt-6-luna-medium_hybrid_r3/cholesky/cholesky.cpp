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

// Row-cyclic distributed, right-looking scalar Cholesky. Each rank owns
// complete rows, while CUDA updates those rows in parallel. Pivot rows are
// broadcast so every rank has the values needed for its local updates.
__global__ void updateRows(double *a, const double *pivot, size_t n,
                           size_t k, size_t count) {
    size_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= count) return;
    size_t i = k + 1 + r;
    if (i <= k) return;
    double x = a[i*n+k];
    for (size_t q = 0; q < k; ++q) x -= a[i*n+q] * pivot[q];
    a[i*n+k] = x / pivot[k];
}

__global__ void formPivot(double *a, size_t n, size_t k) {
    size_t j = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    double s=0.;
    for(size_t q=0;q<k;++q) s += a[k*n+q]*a[j*n+q];
    a[k*n+j]=(a[k*n+j]-s)/a[k*n+k];
}

static void cudaCheck(cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

bool choleskyDecomposition(std::vector<double>& A, size_t n, int rank, int ranks) {
    // Keep a full replicated matrix for validation/result compatibility; only
    // owner rows are modified and broadcast after completion.
    double *deviceA=nullptr, *devicePivot=nullptr;
    int deviceCount=0;
    cudaCheck(cudaGetDeviceCount(&deviceCount));
    if(deviceCount==0){fprintf(stderr,"No CUDA device available\n");MPI_Abort(MPI_COMM_WORLD,2);}
    cudaCheck(cudaSetDevice(rank % deviceCount));
    cudaCheck(cudaMalloc(&deviceA, A.size()*sizeof(double)));
    cudaCheck(cudaMalloc(&devicePivot, n*sizeof(double)));
    cudaCheck(cudaMemcpy(deviceA, A.data(), A.size()*sizeof(double), cudaMemcpyHostToDevice));
    std::vector<double> pivot(n);
    bool success=true;
    for (size_t k=0; k<n; ++k) {
        int owner=static_cast<int>(k % ranks);
        double *row=A.data()+k*n;
        double sum=0.;
        for(size_t q=0;q<k;++q) sum += row[q]*row[q];
        double val=row[k]-sum;
        if (!(val>0.)) success=false;
        if(success) row[k]=std::sqrt(val);
        int ok=success; MPI_Bcast(&ok,1,MPI_INT,owner,MPI_COMM_WORLD);
        if(!ok){success=false; break;}
        cudaCheck(cudaMemcpy(deviceA,A.data(),A.size()*sizeof(double),cudaMemcpyHostToDevice));
        if(k+1<n){formPivot<<<(n-k-1+255)/256,256>>>(deviceA,n,k);cudaCheck(cudaGetLastError());cudaCheck(cudaDeviceSynchronize());}
        cudaCheck(cudaMemcpy(A.data(),deviceA,A.size()*sizeof(double),cudaMemcpyDeviceToHost));
        for(size_t j=0;j<n;++j) pivot[j]=A[k*n+j];
        MPI_Bcast(pivot.data(),static_cast<int>(n),MPI_DOUBLE,owner,MPI_COMM_WORLD);
        if(rank!=owner) std::copy(pivot.begin(),pivot.end(),A.begin()+k*n);
        // Update local rows on GPU. For a row i, the dot uses its stored L row
        // and the pivot's L row; row i's current A(i,k) is updated in place.
        cudaCheck(cudaMemcpy(devicePivot,pivot.data(),n*sizeof(double),cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(deviceA,A.data(),A.size()*sizeof(double),cudaMemcpyHostToDevice));
        if(k+1<n) {
            updateRows<<<(n-k-1+255)/256,256>>>(deviceA,devicePivot,n,k,n-k-1);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaDeviceSynchronize());
        }
        cudaCheck(cudaMemcpy(A.data(),deviceA,A.size()*sizeof(double),cudaMemcpyDeviceToHost));
        MPI_Bcast(A.data()+k*n, static_cast<int>(n), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        // Replicas perform identical GPU updates; the collective establishes a
        // common completed column before advancing to the next pivot.
        MPI_Barrier(MPI_COMM_WORLD);
    }
    cudaFree(deviceA); cudaFree(devicePivot);
    return success;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A,size_t n) {
    std::vector<double>B(n*n); unsigned int seed=42;
    for(size_t i=0;i<n*n;++i) B[i]=(rand_r(&seed)/(double)RAND_MAX)-.5;
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<(long long)n;++i) for(size_t j=0;j<n;++j){double s=0.;for(size_t k=0;k<n;++k)s+=B[i*n+k]*B[j*n+k];A[i*n+j]=s;}
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<(long long)n;++i) A[i*n+i]+=n;
}

bool validateCholesky(const std::vector<double>& L,const std::vector<double>& A,size_t n){
    double maxErr=0.,relErr=0.;
    #pragma omp parallel for reduction(max:maxErr,relErr) schedule(static)
    for(long long i=0;i<(long long)(n*n);++i){size_t r=i/n,c=i%n;double s=0.;for(size_t k=0;k<n;++k)s+=L[r*n+k]*L[c*n+k];double e=fabs(s-A[i]);maxErr=std::max(maxErr,e);relErr=std::max(relErr,e/(fabs(A[i])+1e-10));}
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n",maxErr,relErr);
    if(relErr>1e-6){printf("Validation failed: relative error too large\n");return false;}return true;
}
void printUsage(const char*p){printf("Usage: %s [options]\n  -n <num> Matrix size (default: 512)\n  -v Enable validation\n  -r Print results for external validation\n  -h Show this help message\n",p);}

int main(int argc,char**argv){
    int provided; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank,ranks; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    size_t n=512;bool validate=false,printResults=false;int bad=0;
    for(int i=1;i<argc;++i){if(!strcmp(argv[i],"-n")&&i+1<argc)n=atoi(argv[++i]);else if(!strcmp(argv[i],"-v"))validate=true;else if(!strcmp(argv[i],"-r"))printResults=true;else if(!strcmp(argv[i],"-h")){if(!rank)printUsage(argv[0]);MPI_Finalize();return 0;}else{if(!rank){printf("Unknown option: %s\n",argv[i]);printUsage(argv[0]);}bad=1;}}
    if(bad){MPI_Finalize();return 1;}
    if(!rank){printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nGenerating positive definite matrix...\n",n,n,validate?"enabled":"disabled");}
    std::vector<double>A(n*n),orig;
    if(!rank)generatePositiveDefiniteMatrix(A,n);
    MPI_Bcast(A.data(),static_cast<int>(A.size()),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(validate)orig=A;
    if(!rank)printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);auto start=std::chrono::high_resolution_clock::now();
    bool success=choleskyDecomposition(A,n,rank,ranks);
    MPI_Allreduce(MPI_IN_PLACE,&success,1,MPI_C_BOOL,MPI_LAND,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);auto end=std::chrono::high_resolution_clock::now();
    if(rank){MPI_Finalize();return success?0:1;}
    if(!success){printf("Cholesky decomposition failed\n");MPI_Finalize();return 1;}
    long ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();printf("Computation time: %ld ms\n",ms);
    double secs=std::max(1e-9,ms/1000.);printf("Performance: %.3f GFLOPS\n",((double)n*n*n/3.)/secs/1e9);
    if(printResults) print_results(A,"CholeskyL");
    if(validate){printf("Validating result...\n");bool valid=validateCholesky(A,orig,n);printf("Validation: %s\n",valid?"PASSED":"FAILED");MPI_Finalize();return valid?0:1;}
    MPI_Finalize();return 0;
}
