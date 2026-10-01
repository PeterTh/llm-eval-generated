#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept { return j * n + i; }

static inline void cudaCheck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

__global__ void updateRows(unsigned int* d, unsigned int* p, const unsigned int* pivot,
                           size_t n, size_t rows, size_t first, size_t k) {
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows || j >= n) return;
    const unsigned int dik = d[k * rows + i];
    const unsigned int candidate = dik + pivot[j];
    const size_t at = j * rows + i;
    if (candidate < d[at]) { d[at] = candidate; p[at] = static_cast<unsigned int>(k); }
    (void)first;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n, unsigned int lo, unsigned int hi) {
    unsigned int seed = 42;
    const double range = static_cast<double>(hi - lo) + 1.0;
    for (size_t i = 0; i < n * n; ++i) dist[i] = lo + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    for (size_t i = 0; i < n; ++i) dist[idx2(i, i, n)] = 0;
}
void initializePathMatrix(std::vector<unsigned int>& path, size_t n) {
    for (size_t j = 0; j < n; ++j) for (size_t i = 0; i < n; ++i) {
        path[idx2(i, j, n)] = static_cast<unsigned int>(j);
        path[idx2(j, i, n)] = static_cast<unsigned int>(i);
    }
}

bool validateResult(const std::vector<unsigned int>& d, size_t n) {
    for (size_t i = 0; i < n; ++i) if (d[idx2(i, i, n)] != 0) { printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i); return false; }
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i) for (size_t j = 0; j < std::min(n, size_t(10)); ++j)
        for (size_t k = 0; k < n; ++k) {
            unsigned int a = d[idx2(k, i, n)], b = d[idx2(j, k, n)], c = d[idx2(j, i, n)];
            if (a < INF && b < INF && a + b < c) { printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i,j,k); return false; }
        }
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &world);
    size_t n = 512; bool validate = false, printResults = false;
    for (int i=1; i<argc; ++i) {
        if (!strcmp(argv[i], "-n") && i+1<argc) n = static_cast<size_t>(strtoull(argv[++i], nullptr, 10));
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if(rank==0) printf("Usage: %s [-n nodes] [-v] [-r] [-h]\n", argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) fprintf(stderr,"Unknown option: %s\n",argv[i]); MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    if (n == 0 || n > static_cast<size_t>(INT_MAX)) { if(rank==0) fprintf(stderr,"Invalid node count\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    int localRank = 0; MPI_Comm local; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local); MPI_Comm_rank(local,&localRank);
    int devices=0; cudaCheck(cudaGetDeviceCount(&devices),"device query");
    if (!devices) { if(rank==0) fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    cudaCheck(cudaSetDevice(localRank % devices),"device selection");

    const size_t base=n/static_cast<size_t>(world), rem=n%static_cast<size_t>(world);
    const size_t rows=base+(static_cast<size_t>(rank)<rem), first=static_cast<size_t>(rank)*base+std::min(static_cast<size_t>(rank),rem);
    std::vector<unsigned int> fullD, fullP, packedD, packedP;
    if(rank==0) { fullD.resize(n*n); fullP.resize(n*n); initializeDistanceMatrix(fullD,n,1,MAX_DISTANCE); initializePathMatrix(fullP,n); }
    std::vector<int> counts(world), displs(world), elemCounts(world), elemDispls(world);
    for(int r=0;r<world;++r) { size_t rr=base+(static_cast<size_t>(r)<rem), ff=static_cast<size_t>(r)*base+std::min(static_cast<size_t>(r),rem); counts[r]=static_cast<int>(rr*n); displs[r]=static_cast<int>(ff*n); elemCounts[r]=static_cast<int>(rr); elemDispls[r]=static_cast<int>(ff); }
    if(rank==0) {
        packedD.resize(n*n); packedP.resize(n*n);
        #pragma omp parallel for collapse(3) schedule(static)
        for(int r=0;r<world;++r) for(size_t i=0;i<static_cast<size_t>(elemCounts[r]);++i) for(size_t j=0;j<n;++j) {
            size_t from=elemDispls[r]+i, to=displs[r]+i*n+j;
            packedD[to]=fullD[idx2(from,j,n)]; packedP[to]=fullP[idx2(from,j,n)];
        }
    }
    std::vector<unsigned int> d(rows*n), p(rows*n), pivot(n);
    MPI_Scatterv(rank==0?packedD.data():nullptr,counts.data(),displs.data(),MPI_UNSIGNED,d.data(),static_cast<int>(rows*n),MPI_UNSIGNED,0,MPI_COMM_WORLD);
    MPI_Scatterv(rank==0?packedP.data():nullptr,counts.data(),displs.data(),MPI_UNSIGNED,p.data(),static_cast<int>(rows*n),MPI_UNSIGNED,0,MPI_COMM_WORLD);
    unsigned int *dd=nullptr,*dp=nullptr,*dv=nullptr;
    cudaCheck(cudaMalloc(&dd, std::max(size_t(1),d.size())*sizeof(unsigned int)),"distance allocation");
    cudaCheck(cudaMalloc(&dp, std::max(size_t(1),p.size())*sizeof(unsigned int)),"path allocation");
    cudaCheck(cudaMalloc(&dv,n*sizeof(unsigned int)),"pivot allocation");
    if(!d.empty()) { cudaCheck(cudaMemcpy(dd,d.data(),d.size()*sizeof(unsigned int),cudaMemcpyHostToDevice),"distance upload"); cudaCheck(cudaMemcpy(dp,p.data(),p.size()*sizeof(unsigned int),cudaMemcpyHostToDevice),"path upload"); }
    if(rank==0) { printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\nInitializing graph...\n", n, validate?"enabled":"disabled"); printf("Computing shortest paths...\n"); }
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
    dim3 block(32,8), grid((static_cast<unsigned>(n)+block.x-1)/block.x,(static_cast<unsigned>(rows)+block.y-1)/block.y);
    for(size_t k=0;k<n;++k) {
        if(k>=first && k<first+rows) {
            std::vector<unsigned int> row(n);
            cudaCheck(cudaMemcpy2D(row.data(),sizeof(unsigned int),dd+(k-first),rows*sizeof(unsigned int),sizeof(unsigned int),1,cudaMemcpyDeviceToHost),"pivot download");
            pivot.swap(row);
        }
        const size_t wide = (base + 1) * rem;
        const int pivotOwner = k < wide ? static_cast<int>(k / (base + 1)) : static_cast<int>(rem + (k - wide) / base);
        MPI_Bcast(pivot.data(),static_cast<int>(n),MPI_UNSIGNED,pivotOwner,MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(dv,pivot.data(),n*sizeof(unsigned int),cudaMemcpyHostToDevice),"pivot upload");
        if(rows) { updateRows<<<grid,block>>>(dd,dp,dv,n,rows,first,k); cudaCheck(cudaGetLastError(),"kernel launch"); }
    }
    cudaCheck(cudaDeviceSynchronize(),"kernel completion"); auto end=std::chrono::high_resolution_clock::now();
    if(!d.empty()) { cudaCheck(cudaMemcpy(d.data(),dd,d.size()*sizeof(unsigned int),cudaMemcpyDeviceToHost),"distance download"); cudaCheck(cudaMemcpy(p.data(),dp,p.size()*sizeof(unsigned int),cudaMemcpyDeviceToHost),"path download"); }
    MPI_Gatherv(d.data(),static_cast<int>(rows*n),MPI_UNSIGNED,rank==0?packedD.data():nullptr,counts.data(),displs.data(),MPI_UNSIGNED,0,MPI_COMM_WORLD);
    MPI_Gatherv(p.data(),static_cast<int>(rows*n),MPI_UNSIGNED,rank==0?packedP.data():nullptr,counts.data(),displs.data(),MPI_UNSIGNED,0,MPI_COMM_WORLD);
    if(rank==0) {
    #pragma omp parallel for collapse(3) schedule(static)
    for(int r=0;r<world;++r) for(size_t i=0;i<static_cast<size_t>(elemCounts[r]);++i) for(size_t j=0;j<n;++j) {
        size_t from=displs[r]+i*n+j, to=idx2(elemDispls[r]+i,j,n);
        fullD[to]=packedD[from]; fullP[to]=packedP[from];
    }
    }
    if(rank==0) {
        auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        printf("Computation time: %ld ms\n",ms); double secs=std::chrono::duration<double>(end-start).count(); printf("Performance: %.3f GOPS\n",secs?double(n)*n*n/secs/1e9:0.0);
        if(printResults) print_results_int(fullD,"DistanceMatrix");
        if(validate) { printf("Validating result...\n"); bool ok=validateResult(fullD,n); printf("Validation: %s\n",ok?"PASSED":"FAILED"); if(!ok) MPI_Abort(MPI_COMM_WORLD,1); }
    }
    cudaFree(dd); cudaFree(dp); cudaFree(dv); MPI_Comm_free(&local); MPI_Finalize(); return 0;
}
