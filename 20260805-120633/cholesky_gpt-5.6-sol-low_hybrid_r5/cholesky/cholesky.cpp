#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../common/results_output.hpp"

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){std::fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,2);} } while(0)

// Rows are distributed cyclically.  At step k, its owner factors the pivot;
// every rank updates its independent rows on its GPU.  This is a communication-
// efficient 1-D right-looking Cholesky (one O(n) panel broadcast per step).
__global__ void update_column(double *a, int local_n, int n, int k,
                              double diag, int rank, int nranks) {
    int lr=blockIdx.x*blockDim.x+threadIdx.x;
    if(lr>=local_n) return;
    int i=rank+lr*nranks;
    if(i>k) a[(size_t)lr*n+k]/=diag;
}

__global__ void trailing_update(double *a, int local_n, int n, int k,
                                const double *col, int rank, int nranks) {
    int j=k+1+blockIdx.x*blockDim.x+threadIdx.x;
    int lr=blockIdx.y*blockDim.y+threadIdx.y;
    if(lr>=local_n || j>=n) return;
    int i=rank+lr*nranks;
    if(i>=j) a[(size_t)lr*n+j]-=col[i]*col[j];
}

static void generate(std::vector<double>& A, size_t n) {
    std::vector<double>B(n*n); unsigned seed=42;
    for(size_t i=0;i<n*n;i++) B[i]=rand_r(&seed)/(double)RAND_MAX-.5;
    #pragma omp parallel for schedule(static)
    for(long i=0;i<(long)n;i++) for(size_t j=0;j<n;j++) {
        double s=0;
        #pragma omp simd reduction(+:s)
        for(size_t k=0;k<n;k++) s+=B[(size_t)i*n+k]*B[j*n+k];
        A[(size_t)i*n+j]=s+(i==(long)j?n:0);
    }
}

static bool factor(std::vector<double>& local, int n, int rank, int size) {
    int ln=(n-rank+size-1)/size;
    double *d_a=nullptr,*d_col=nullptr;
    CUDA_OK(cudaMalloc(&d_a,(size_t)std::max(1,ln)*n*sizeof(double)));
    CUDA_OK(cudaMalloc(&d_col,(size_t)n*sizeof(double)));
    if(ln) CUDA_OK(cudaMemcpy(d_a,local.data(),(size_t)ln*n*sizeof(double),cudaMemcpyHostToDevice));
    std::vector<double> row(n), col(ln);
    for(int k=0;k<n;k++) {
        int owner=k%size, kl=k/size; double diag=0;
        if(rank==owner) {
            CUDA_OK(cudaMemcpy(row.data(),d_a+(size_t)kl*n,(size_t)n*sizeof(double),cudaMemcpyDeviceToHost));
            double v=row[k];
            if(v<=0) diag=-1; else diag=std::sqrt(v);
            CUDA_OK(cudaMemcpy(d_a+(size_t)kl*n+k,&diag,sizeof(double),cudaMemcpyHostToDevice));
        }
        MPI_Bcast(&diag,1,MPI_DOUBLE,owner,MPI_COMM_WORLD);
        if(diag<=0) { cudaFree(d_col); cudaFree(d_a); return false; }
        int threads=256;
        if(ln) {
            update_column<<<(ln+threads-1)/threads,threads>>>(d_a,ln,n,k,diag,rank,size);
            CUDA_OK(cudaGetLastError());
        }
        // Gather the newly computed distributed column to all ranks.  The
        // compact allgatherv buffer is then expanded for coalesced GPU updates.
        if(ln) CUDA_OK(cudaMemcpy2D(col.data(),sizeof(double),d_a+k,(size_t)n*sizeof(double),sizeof(double),ln,cudaMemcpyDeviceToHost));
        std::vector<int> counts(size),displs(size); int total=0;
        for(int r=0;r<size;r++){counts[r]=(n-r+size-1)/size;displs[r]=total;total+=counts[r];}
        std::vector<double> packed(n), full(n);
        MPI_Allgatherv(col.data(),ln,MPI_DOUBLE,packed.data(),counts.data(),displs.data(),MPI_DOUBLE,MPI_COMM_WORLD);
        for(int r=0;r<size;r++) for(int q=0;q<counts[r];q++) full[r+q*size]=packed[displs[r]+q];
        CUDA_OK(cudaMemcpy(d_col,full.data(),(size_t)n*sizeof(double),cudaMemcpyHostToDevice));
        if(k+1<n && ln) {
            dim3 block(32,8), grid((n-k-1+31)/32,(ln+7)/8);
            trailing_update<<<grid,block>>>(d_a,ln,n,k,d_col,rank,size);
            CUDA_OK(cudaGetLastError());
        }
        CUDA_OK(cudaDeviceSynchronize());
    }
    if(ln) CUDA_OK(cudaMemcpy(local.data(),d_a,(size_t)ln*n*sizeof(double),cudaMemcpyDeviceToHost));
    cudaFree(d_col); cudaFree(d_a); return true;
}

static bool validate(const std::vector<double>&L,const std::vector<double>&A,int n){
    double me=0,mr=0;
    #pragma omp parallel for reduction(max:me,mr) schedule(static)
    for(int i=0;i<n;i++) for(int j=0;j<n;j++){double s=0; for(int k=0;k<=std::min(i,j);k++)s+=L[(size_t)i*n+k]*L[(size_t)j*n+k]; double e=fabs(s-A[(size_t)i*n+j]); me=std::max(me,e);mr=std::max(mr,e/(fabs(A[(size_t)i*n+j])+1e-10));}
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n",me,mr); return mr<=1e-6;
}
static void usage(const char*p){printf("Usage: %s [-n num] [-v] [-r] [-h]\n",p);}

int main(int argc,char**argv){
    int provided; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided); int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    int devices=0; CUDA_OK(cudaGetDeviceCount(&devices)); if(!devices){if(!rank)fprintf(stderr,"No CUDA device\n");MPI_Abort(MPI_COMM_WORLD,2);} CUDA_OK(cudaSetDevice(rank%devices));
    int n=512; bool val=false,pr=false; for(int i=1;i<argc;i++){if(!strcmp(argv[i],"-n")&&i+1<argc)n=atoi(argv[++i]);else if(!strcmp(argv[i],"-v"))val=true;else if(!strcmp(argv[i],"-r"))pr=true;else if(!strcmp(argv[i],"-h")){if(!rank)usage(argv[0]);MPI_Finalize();return 0;}else{if(!rank)usage(argv[0]);MPI_Finalize();return 1;}}
    if(n<=0){if(!rank)fprintf(stderr,"Matrix size must be positive\n");MPI_Finalize();return 1;}
    std::vector<double>A,orig; if(!rank){printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\nGenerating positive definite matrix...\n",n,n,val?"enabled":"disabled",size,omp_get_max_threads());A.resize((size_t)n*n);generate(A,n);if(val)orig=A;}
    int ln=(n-rank+size-1)/size; std::vector<double>local((size_t)ln*n); std::vector<int>cnt(size),disp(size);if(!rank)for(int r=0;r<size;r++){cnt[r]=((n-r+size-1)/size)*n;disp[r]=r?disp[r-1]+cnt[r-1]:0;}
    std::vector<double>packed;if(!rank){packed.resize((size_t)n*n);for(int r=0;r<size;r++)for(int q=0;q<cnt[r]/n;q++)memcpy(packed.data()+disp[r]+(size_t)q*n,A.data()+(size_t)(r+q*size)*n,(size_t)n*sizeof(double));}
    MPI_Scatterv(packed.data(),cnt.data(),disp.data(),MPI_DOUBLE,local.data(),ln*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD); if(!rank)printf("Computing Cholesky decomposition...\n"); double t=MPI_Wtime(); bool ok=factor(local,n,rank,size);int allok;int iok=ok;MPI_Allreduce(&iok,&allok,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);t=MPI_Wtime()-t;
    MPI_Gatherv(local.data(),ln*n,MPI_DOUBLE,packed.data(),cnt.data(),disp.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    int rc=allok?0:1;if(!rank&&allok){for(int r=0;r<size;r++)for(int q=0;q<cnt[r]/n;q++)memcpy(A.data()+(size_t)(r+q*size)*n,packed.data()+disp[r]+(size_t)q*n,(size_t)n*sizeof(double));for(int i=0;i<n;i++)std::fill(A.begin()+(size_t)i*n+i+1,A.begin()+(size_t)(i+1)*n,0.0);printf("Computation time: %.3f ms\nPerformance: %.3f GFLOPS\n",t*1e3,(double)n*n*n/(3e9*t));if(pr)print_results(A,"CholeskyL");if(val){printf("Validating result...\n");bool v=validate(A,orig,n);printf("Validation: %s\n",v?"PASSED":"FAILED");rc=v?0:1;}}
    MPI_Bcast(&rc,1,MPI_INT,0,MPI_COMM_WORLD);MPI_Finalize();return rc;
}
