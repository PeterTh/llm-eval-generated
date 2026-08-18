#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_OK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
  MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

// One rank owns a contiguous set of complete rows.  The pivot row is broadcast
// once and every GPU independently updates the column entries in its row set.
__global__ void update_column(double* a, size_t n, size_t first, size_t rows,
                              size_t j, const double* pivot, double diagonal) {
    size_t li = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = first + li;
    if (li >= rows || i <= j) return;
    double s = 0.0;
    for (size_t k = 0; k < j; ++k) s = fma(a[li*n+k], pivot[k], s);
    a[li*n+j] = (a[li*n+j] - s) / diagonal;
}

static void distribution(size_t n, int p, std::vector<int>& counts,
                         std::vector<int>& displs) {
    counts.resize(p); displs.resize(p);
    size_t off = 0;
    for (int r=0; r<p; ++r) {
        size_t rows = n/(size_t)p + ((size_t)r < n%(size_t)p);
        if (rows*n > (size_t)INT_MAX || off*n > (size_t)INT_MAX) {
            std::fprintf(stderr, "Matrix is too large for MPI counts\n");
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
        counts[r] = (int)(rows*n); displs[r] = (int)(off*n); off += rows;
    }
}

static void generate(std::vector<double>& A, size_t n) {
    std::vector<double> B(n*n);
    #pragma omp parallel for schedule(static)
    for (long long i=0; i<(long long)n; ++i) {
        unsigned int seed = 42u + (unsigned int)i * 747796405u;
        for (size_t k=0; k<n; ++k)
            B[(size_t)i*n+k] = rand_r(&seed)/(double)RAND_MAX - 0.5;
    }
    #pragma omp parallel for schedule(static)
    for (long long ii=0; ii<(long long)n; ++ii) {
        size_t i=(size_t)ii;
        for (size_t j=0; j<n; ++j) {
            double s=0.0;
            #pragma omp simd reduction(+:s)
            for (size_t k=0; k<n; ++k) s += B[i*n+k]*B[j*n+k];
            A[i*n+j]=s;
        }
        A[i*n+i] += (double)n;
    }
}

static bool validate(const std::vector<double>& L, const std::vector<double>& A,
                     size_t n) {
    double maxAbs=0.0, maxRel=0.0;
    #pragma omp parallel for reduction(max:maxAbs,maxRel) schedule(static)
    for (long long ii=0; ii<(long long)n; ++ii) {
        size_t i=(size_t)ii;
        for (size_t j=0; j<n; ++j) {
            double s=0.0; size_t end=std::min(i,j);
            #pragma omp simd reduction(+:s)
            for (size_t k=0; k<=end; ++k) s += L[i*n+k]*L[j*n+k];
            double e=std::fabs(s-A[i*n+j]);
            maxAbs=std::max(maxAbs,e); maxRel=std::max(maxRel,e/(std::fabs(A[i*n+j])+1e-10));
        }
    }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n",maxAbs,maxRel);
    if (maxRel>1e-6) std::printf("Validation failed: relative error too large\n");
    return maxRel<=1e-6;
}

static void usage(const char* p) {
    std::printf("Usage: %s [options]\n  -n <num>     Matrix size (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n",p);
}

int main(int argc,char** argv) {
    int provided=0; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank=0,np=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&np);
    size_t n=512; bool doValidate=false, printResults=false; int parseOk=1, help=0;
    for(int i=1;i<argc;++i) {
        if(!std::strcmp(argv[i],"-n") && i+1<argc) n=std::strtoull(argv[++i],nullptr,10);
        else if(!std::strcmp(argv[i],"-v")) doValidate=true;
        else if(!std::strcmp(argv[i],"-r")) printResults=true;
        else if(!std::strcmp(argv[i],"-h")) help=1;
        else parseOk=0;
    }
    if(rank==0 && (help||!parseOk)) usage(argv[0]);
    if(help||!parseOk||n==0) { MPI_Finalize(); return parseOk?0:1; }

    int localRank=0; MPI_Comm node; MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&node);
    MPI_Comm_rank(node,&localRank); int devices=0; CUDA_OK(cudaGetDeviceCount(&devices));
    if(devices==0) { if(rank==0) std::fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    CUDA_OK(cudaSetDevice(localRank%devices)); MPI_Comm_free(&node);

    std::vector<int> counts,displs; distribution(n,np,counts,displs);
    size_t localRows=(size_t)counts[rank]/n, first=(size_t)displs[rank]/n;
    std::vector<double> full, original, local((size_t)counts[rank]), pivot(n);
    if(rank==0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",n,n,doValidate?"enabled":"disabled");
        std::printf("Generating positive definite matrix...\n"); full.resize(n*n); generate(full,n);
        if(doValidate) original=full;
    }
    MPI_Scatterv(rank==0?full.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,
                 local.data(),counts[rank],MPI_DOUBLE,0,MPI_COMM_WORLD);
    double* dA=nullptr; double* dPivot=nullptr;
    CUDA_OK(cudaMalloc(&dA,std::max<size_t>(1,local.size())*sizeof(double)));
    CUDA_OK(cudaMalloc(&dPivot,n*sizeof(double)));
    if(!local.empty()) CUDA_OK(cudaMemcpy(dA,local.data(),local.size()*sizeof(double),cudaMemcpyHostToDevice));

    if(rank==0) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD); double start=MPI_Wtime(); int success=1;
    for(size_t j=0;j<n && success;++j) {
        int owner=0;
        while (owner+1<np && (size_t)(displs[owner]+counts[owner])/n<=j) ++owner;
        if(rank==owner) {
            size_t li=j-first;
            if(j) CUDA_OK(cudaMemcpy(pivot.data(),dA+li*n,j*sizeof(double),cudaMemcpyDeviceToHost));
            double s=0.0;
            #pragma omp simd reduction(+:s)
            for(size_t k=0;k<j;++k) s+=pivot[k]*pivot[k];
            double v=local[li*n+j]-s; // diagonal has never been modified on device
            if(v<=0.0) success=0; else pivot[j]=std::sqrt(v);
        }
        MPI_Bcast(&success,1,MPI_INT,owner,MPI_COMM_WORLD); if(!success) break;
        MPI_Bcast(pivot.data(),(int)(j+1),MPI_DOUBLE,owner,MPI_COMM_WORLD);
        if(rank==owner) CUDA_OK(cudaMemcpy(dA+(j-first)*n+j,&pivot[j],sizeof(double),cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dPivot,pivot.data(),(j+1)*sizeof(double),cudaMemcpyHostToDevice));
        if(localRows) update_column<<<(unsigned)((localRows+255)/256),256>>>(dA,n,first,localRows,j,dPivot,pivot[j]);
        CUDA_OK(cudaGetLastError());
    }
    CUDA_OK(cudaDeviceSynchronize()); double elapsed=MPI_Wtime()-start, worst=0.0;
    MPI_Reduce(&elapsed,&worst,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(!local.empty()) CUDA_OK(cudaMemcpy(local.data(),dA,local.size()*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(dA)); CUDA_OK(cudaFree(dPivot));
    #pragma omp parallel for schedule(static)
    for(long long li=0;li<(long long)localRows;++li) {
        size_t gi=first+(size_t)li;
        for(size_t j=gi+1;j<n;++j) local[(size_t)li*n+j]=0.0;
    }
    if(rank==0 && full.size()!=n*n) full.resize(n*n);
    MPI_Gatherv(local.data(),counts[rank],MPI_DOUBLE,rank==0?full.data():nullptr,
                counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    int rc=0;
    if(rank==0) {
        if(!success) { std::printf("Cholesky decomposition failed\n"); rc=1; }
        else {
            long long ms=(long long)(worst*1000.0);
            std::printf("Computation time: %lld ms\nPerformance: %.3f GFLOPS\n",ms,(double)n*n*n/3.0/worst/1e9);
            if(printResults) print_results(full,"CholeskyL");
            if(doValidate) { std::printf("Validating result...\n"); bool ok=validate(full,original,n);
                std::printf("Validation: %s\n",ok?"PASSED":"FAILED"); rc=ok?0:1; }
        }
    }
    MPI_Bcast(&rc,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Finalize(); return rc;
}
