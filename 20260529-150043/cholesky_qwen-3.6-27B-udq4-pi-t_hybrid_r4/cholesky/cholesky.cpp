// Hybrid MPI + OpenMP + CUDA Cholesky Decomposition Benchmark
//
// Parallelization: MPI row-wise distribution + CUDA kernels + OpenMP
// Algorithm: Blocked Cholesky with panel factorization, solve, and SYRK update.
// After solving, all panel rows are Allgathered so every rank can do the full
// SYRK update on its local trailing submatrix.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include <mpi.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#include "../common/results_output.hpp"

#ifdef HAVE_CUDA
#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); exit(1); } } while(0)
static constexpr int BLOCK_DIM = 256;

__global__ void genRandK(double* B, size_t n, unsigned int seed) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * n) {
        unsigned int s = seed + i * 2654435761u;
        s = s * 1664525u + 1013904223u;
        B[i] = (double)(s & 0x7FFFFFFF) / (double)0x7FFFFFFF - 0.5;
    }
}
__global__ void mmK(const double* B, double* A, size_t n) {
    size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n && c < n) {
        double s = 0;
        for (size_t k = 0; k < n; ++k) s += B[r*n+k] * B[c*n+k];
        A[r*n+c] = s;
    }
}
__global__ void diagK(double* A, size_t n, double v) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i*n+i] += v;
}
__global__ void gatherK(const double* A, double* P, size_t m, size_t n, size_t k, size_t p) {
    size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < m && c < p) P[r*p+c] = A[r*n+k+c];
}
__global__ void scatterK(const double* P, double* A, size_t m, size_t n, size_t k, size_t p) {
    size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < m && c < p) A[r*n+k+c] = P[r*p+c];
}
__global__ void solveK(double* P, const double* U, size_t m, size_t p) {
    size_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < m) {
        for (size_t c = 0; c < p; ++c) {
            double v = P[r*p+c];
            for (size_t l = 0; l < c; ++l) v -= P[r*p+l] * U[c*p+l];
            P[r*p+c] = v / U[c*p+c];
        }
    }
}
// SYRK: C = C - A*A^T, C is m x ntr, A is m x p
__global__ void syrK(const double* A, double* C, size_t m, size_t ntr, size_t p) {
    size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < m && c < m && c < ntr) {
        double s = 0;
        for (size_t l = 0; l < p; ++l) s += A[r*p+l] * A[c*p+l];
        C[r*ntr+c] -= s;
    }
}
// Extended SYRK: C[r][c] -= sum_l(FP[off+r][l] * FP[c][l])
// FP is trailR x p, C is m x ntr, off is row offset into FP
__global__ void syrExtK(const double* FP, double* C, size_t m, size_t ntr, size_t p, size_t off) {
    size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < m && c < ntr) {
        double s = 0;
        for (size_t l = 0; l < p; ++l) s += FP[(off+r)*p+l] * FP[c*p+l];
        C[r*ntr+c] -= s;
    }
}
__global__ void valK(const double* L, double* R, size_t n) {
    size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < n && c < n) {
        double s = 0;
        for (size_t k = 0; k < n; ++k) s += L[r*n+k] * L[c*n+k];
        R[r*n+c] = s;
    }
}
__global__ void zeroUK(double* A, size_t n, size_t mLN, size_t mRS) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < mLN*n) { size_t r=mRS+i/n, c=i%n; if (c>r) A[i]=0; }
}

#endif

static bool choleskyPanelCPU(double* L, size_t nb) {
    for (size_t i = 0; i < nb; ++i)
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0;
            if (i == j) {
                for (size_t k = 0; k < j; ++k) sum += L[j*nb+k]*L[j*nb+k];
                double val = L[j*nb+j] - sum;
                if (val <= 0) { printf("Error: not PD at %zu\n",j); return false; }
                L[j*nb+j] = std::sqrt(val);
            } else {
                for (size_t k = 0; k < j; ++k) sum += L[i*nb+k]*L[j*nb+k];
                L[i*nb+j] = (L[i*nb+j] - sum) / L[j*nb+j];
            }
        }
    return true;
}

static size_t rowStart(size_t n, int r, int nr) {
    size_t rp = n/nr, ex = n%nr, s = 0;
    for (int i = 0; i < r; ++i) s += rp + (i < (int)ex ? 1 : 0);
    return s;
}
static size_t localN(size_t n, int r, int nr) {
    return n/nr + (r < (int)(n%nr) ? 1 : 0);
}

static void generateMatrix(std::vector<double>& Al, size_t n, size_t mN, size_t mRS) {
#ifdef HAVE_CUDA
    double *dB, *dA; size_t nn=n*n;
    CUDA_CHECK(cudaMalloc(&dB,nn*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dA,nn*sizeof(double)));
    int th=BLOCK_DIM, bl=(nn+th-1)/th;
    genRandK<<<bl,th>>>(dB,n,42u); CUDA_CHECK(cudaDeviceSynchronize());
    dim3 td(16,16), bd((n+td.x-1)/td.x,(n+td.y-1)/td.y);
    mmK<<<bd,td>>>(dB,dA,n); CUDA_CHECK(cudaDeviceSynchronize());
    diagK<<<(n+th-1)/th,th>>>(dA,n,(double)n); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(Al.data(),dA+mRS*n,mN*n*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dA));
#else
    std::vector<double> B(n*n); unsigned int seed=42;
    for(size_t i=0;i<n*n;++i) B[i]=(rand_r(&seed)/(double)RAND_MAX)-0.5;
    std::vector<double> Af(n*n);
    #pragma omp parallel for collapse(2) schedule(static)
    for(size_t i=0;i<n;++i) for(size_t j=0;j<n;++j){double s=0;for(size_t k=0;k<n;++k)s+=B[i*n+k]*B[j*n+k];Af[i*n+j]=s;}
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<n;++i) Af[i*n+i]+=(double)n;
    for(size_t i=0;i<mN;++i) memcpy(&Al[i*n],&Af[(mRS+i)*n],n*sizeof(double));
#endif
}

static bool choleskyDecomposition(std::vector<double>& A, size_t n, int rank, int numRanks) {
    const size_t nb = 64;
    size_t mRS = rowStart(n, rank, numRanks);
    size_t mLN = localN(n, rank, numRanks);

#ifdef HAVE_CUDA
    double *dA, *dP, *dPS, *dFP, *dC;
    CUDA_CHECK(cudaMalloc(&dA, mLN*n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dP, nb*nb*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dPS, mLN*nb*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dFP, n*nb*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC, n*n*sizeof(double)));
    std::vector<double> pf(nb*nb);
    std::vector<size_t> rRS(numRanks), rLN(numRanks);
    for(int r=0;r<numRanks;++r){rRS[r]=rowStart(n,r,numRanks);rLN[r]=localN(n,r,numRanks);}

    CUDA_CHECK(cudaMemcpy(dA,A.data(),mLN*n*sizeof(double),cudaMemcpyHostToDevice));

    for (size_t k = 0; k < n; k += nb) {
        size_t ps = std::min(nb, n-k);
        size_t trailR = n - k - ps;

        // Find owning rank for panel rows [k, k+ps)
        int pRank = -1;
        for(int r=0;r<numRanks;++r) if(rRS[r]<=k && k<rRS[r]+rLN[r]){pRank=r;break;}

        // Step 1: Factor panel
        if (rank == pRank) {
            dim3 td(16,16), bd((ps+td.x-1)/td.x,(ps+td.y-1)/td.y);
            size_t pLR = k - mRS;
            gatherK<<<bd,td>>>(dA+pLR*n, dP, ps, n, k, ps);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(pf.data(),dP,ps*ps*sizeof(double),cudaMemcpyDeviceToHost));
            if (!choleskyPanelCPU(pf.data(), ps)) {
                CUDA_CHECK(cudaFree(dA));CUDA_CHECK(cudaFree(dP));
                CUDA_CHECK(cudaFree(dPS));CUDA_CHECK(cudaFree(dFP));
                CUDA_CHECK(cudaFree(dC));
                return false;
            }
            CUDA_CHECK(cudaMemcpy(dP,pf.data(),ps*ps*sizeof(double),cudaMemcpyHostToDevice));
            scatterK<<<bd,td>>>(dP, dA+pLR*n, ps, n, k, ps);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(pf.data(),(int)(ps*ps),MPI_DOUBLE,pRank,MPI_COMM_WORLD);

        // Step 2: Solve panel for rows below panel
        size_t rBP = std::max(mRS, k+ps);
        size_t rEnd = mRS + mLN;
        size_t sRows = (rBP < rEnd) ? (rEnd-rBP) : 0;
        size_t sOff = (rBP < rEnd) ? (rBP-mRS) : 0;

        if (sRows > 0) {
            double* sp = dA + sOff*n;
            {dim3 td(16,16),bd((ps+td.x-1)/td.x,(sRows+td.y-1)/td.y);
             gatherK<<<bd,td>>>(sp,dPS,sRows,n,k,ps);CUDA_CHECK(cudaDeviceSynchronize());}
            {int st=BLOCK_DIM,sb=(sRows+st-1)/st;
             solveK<<<sb,st>>>(dPS,pf.data(),sRows,ps);CUDA_CHECK(cudaDeviceSynchronize());}
            {dim3 td(16,16),bd((ps+td.x-1)/td.x,(sRows+td.y-1)/td.y);
             scatterK<<<bd,td>>>(dPS,sp,sRows,n,k,ps);CUDA_CHECK(cudaDeviceSynchronize());}
        }

        // Step 3: Allgather solved panel rows so every rank has full L(k+ps:n, k:k+ps)
        if (trailR > 0) {
            std::vector<int> sc(numRanks), rc(numRanks), sd(numRanks), rd(numRanks);
            int rOff = 0;
            for(int r=0;r<numRanks;++r) {
                size_t rBP2 = std::max(rRS[r], k+ps);
                size_t rEnd2 = rRS[r] + rLN[r];
                size_t sR = (rBP2 < rEnd2) ? (rEnd2-rBP2) : 0;
                sc[r] = (int)(sR * ps);
                sd[r] = (int)(rOff * ps);
                rc[r] = sc[r];
                rd[r] = (int)((rBP2 - (k+ps)) * ps);
                rOff += sR;
            }

            // Copy local solved panel to send buffer
            if (sRows > 0) {
                // dPS already has the solved panel
                CUDA_CHECK(cudaMemcpy(dFP, dPS, sRows*ps*sizeof(double), cudaMemcpyDeviceToDevice));
            }

            // Allgatherv
            std::vector<double> hSend(sRows > 0 ? sRows*ps : 0);
            std::vector<double> hRecv(trailR * ps);
            if (sRows > 0) {
                CUDA_CHECK(cudaMemcpy(hSend.data(), dPS, sRows*ps*sizeof(double), cudaMemcpyDeviceToHost));
            }
            MPI_Allgatherv(hSend.data(), sc[rank], MPI_DOUBLE,
                          hRecv.data(), rc.data(), rd.data(), MPI_DOUBLE, MPI_COMM_WORLD);

            // Copy full panel to GPU
            CUDA_CHECK(cudaMemcpy(dFP, hRecv.data(), trailR*ps*sizeof(double), cudaMemcpyHostToDevice));

            // Step 4: SYRK update on local trailing submatrix
            if (sRows > 0) {
                size_t fpOff = mRS + sOff - (k+ps);
                dim3 td2(16,16);
                // Gather trailing submatrix into dC
                dim3 bdG((trailR+td2.x-1)/td2.x,(sRows+td2.y-1)/td2.y);
                gatherK<<<bdG,td2>>>(dA+sOff*n, dC, sRows, n, k+ps, trailR);
                CUDA_CHECK(cudaDeviceSynchronize());
                // Extended SYRK: C[r][c] -= sum_l(FP[fpOff+r][l] * FP[c][l])
                dim3 bdS((trailR+td2.x-1)/td2.x,(sRows+td2.y-1)/td2.y);
                syrExtK<<<bdS,td2>>>(dFP, dC, sRows, trailR, ps, fpOff);
                CUDA_CHECK(cudaDeviceSynchronize());
                // Scatter back
                scatterK<<<bdG,td2>>>(dC, dA+sOff*n, sRows, n, k+ps, trailR);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }
    }

    // Zero upper triangular
    {int th=BLOCK_DIM,bl=(mLN*n+th-1)/th;
     zeroUK<<<bl,th>>>(dA,n,mLN,mRS);CUDA_CHECK(cudaDeviceSynchronize());}
    CUDA_CHECK(cudaMemcpy(A.data(),dA,mLN*n*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));CUDA_CHECK(cudaFree(dP));
    CUDA_CHECK(cudaFree(dPS));CUDA_CHECK(cudaFree(dFP));CUDA_CHECK(cudaFree(dC));
#else
    for(size_t i=0;i<n;++i){
        for(size_t j=0;j<=i;++j){
            double s=0;
            if(i==j){for(size_t k=0;k<j;++k)s+=A[j*n+k]*A[j*n+k];double v=A[j*n+j]-s;if(v<=0)return false;A[j*n+j]=std::sqrt(v);}
            else{for(size_t k=0;k<j;++k)s+=A[i*n+k]*A[j*n+k];A[i*n+j]=(A[i*n+j]-s)/A[j*n+j];}
        }
        for(size_t j=i+1;j<n;++j)A[i*n+j]=0;
    }
#endif
    return true;
}

static bool validateCholesky(const std::vector<double>& L, const std::vector<double>& Ao, size_t n) {
    std::vector<double> R(n*n);
#ifdef HAVE_CUDA
    double *dL,*dR;
    CUDA_CHECK(cudaMalloc(&dL,n*n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dR,n*n*sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dL,L.data(),n*n*sizeof(double),cudaMemcpyHostToDevice));
    dim3 td(16,16),bd((n+td.x-1)/td.x,(n+td.y-1)/td.y);
    valK<<<bd,td>>>(dL,dR,n);CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(R.data(),dR,n*n*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dL));CUDA_CHECK(cudaFree(dR));
#else
    #pragma omp parallel for collapse(2) schedule(static)
    for(size_t i=0;i<n;++i)for(size_t j=0;j<n;++j){double s=0;for(size_t k=0;k<n;++k)s+=L[i*n+k]*L[j*n+k];R[i*n+j]=s;}
#endif
    double me=0,re=0;
    #pragma omp parallel for reduction(max:me) reduction(max:re) schedule(static)
    for(size_t i=0;i<n*n;++i){double e=std::fabs(R[i]-Ao[i]);if(e>me)me=e;double r=e/(std::fabs(Ao[i])+1e-10);if(r>re)re=r;}
    printf("Max absolute error: %.10e\n",me);
    printf("Max relative error: %.10e\n",re);
    return re <= 1e-6;
}

static void printUsage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>  Matrix size (default: 512)\n  -v  Validate\n  -r  Print results\n  -h  Help\n",p);
}

int main(int argc, char** argv) {
    size_t n=512; bool validate=false, printResults=false;
    for(int i=1;i<argc;++i){
        if(strcmp(argv[i],"-n")==0&&i+1<argc)n=atoi(argv[++i]);
        else if(strcmp(argv[i],"-v")==0)validate=true;
        else if(strcmp(argv[i],"-r")==0)printResults=true;
        else if(strcmp(argv[i],"-h")==0){printUsage(argv[0]);return 0;}
        else{printf("Unknown: %s\n",argv[i]);printUsage(argv[0]);return 1;}
    }
    int provided;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank,numRanks;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&numRanks);

#ifdef HAVE_CUDA
    int nd=0;cudaGetDeviceCount(&nd);
    if(nd>0){
        CUDA_CHECK(cudaSetDevice(rank%nd));
        int ot=1;
        #pragma omp parallel
        {
#pragma omp master
            ot = omp_get_num_threads();
        }
        if(rank==0){
            cudaDeviceProp prop;cudaGetDeviceProperties(&prop,rank%nd);
            printf("Cholesky Decomposition Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
            printf("Matrix size: %zu x %zu\n",n,n);
            printf("MPI ranks: %d\n",numRanks);
            printf("CUDA devices: %d, using device %d (%s)\n",nd,rank%nd,prop.name);
            printf("OpenMP threads per rank: %d\n",ot);
            printf("Validation: %s\n",validate?"enabled":"disabled");
        }
    } else {
        if(rank==0)printf("No CUDA devices, CPU fallback\n");
    }
#else
    if(rank==0)printf("Cholesky (MPI+OpenMP only)\n");
#endif

    size_t mLN=localN(n,rank,numRanks),mRS=rowStart(n,rank,numRanks);
    std::vector<double> Al(mLN*n);
    if(rank==0)printf("Generating positive definite matrix...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    generateMatrix(Al,n,mLN,mRS);

    if(rank==0)printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto t0=std::chrono::high_resolution_clock::now();
    bool ok=choleskyDecomposition(Al,n,rank,numRanks);
    auto t1=std::chrono::high_resolution_clock::now();
    auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0);
    MPI_Barrier(MPI_COMM_WORLD);
    if(!ok){printf("Cholesky failed\n");MPI_Finalize();return 1;}
    double gflops=(double)n*n*n/3.0/(ms.count()/1000.0)/1e9;
    if(rank==0){printf("Computation time: %ld ms\n",ms.count());printf("Performance: %.3f GFLOPS\n",gflops);}

    // Gather to rank 0
    std::vector<double> Af;
    std::vector<int> rc(numRanks),rd(numRanks);
    size_t rp=n/numRanks,ex=n%numRanks;
    for(int r=0;r<numRanks;++r){
        size_t rr=rp+(r<(int)ex?1:0);rc[r]=(int)(rr*n);
        size_t rs=0;for(int i=0;i<r;++i)rs+=rp+(i<(int)ex?1:0);
        rd[r]=(int)(rs*n);
    }
    if(rank==0)Af.resize(n*n);
    MPI_Gatherv(Al.data(),(int)(mLN*n),MPI_DOUBLE,Af.data(),rc.data(),rd.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);

    if(printResults&&rank==0)print_results(Af,"CholeskyL");
    if(validate&&rank==0){
        printf("Validating result...\n");
        std::vector<double> Ao(n*n);generateMatrix(Ao,n,n,0);
        if(validateCholesky(Af,Ao,n))printf("Validation: PASSED\n");
        else{printf("Validation: FAILED\n");MPI_Finalize();return 1;}
    }
    MPI_Finalize();return 0;
}
