#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept { return j * n + i; }

__global__ void fw_step(unsigned int* d, unsigned int* p, size_t n, size_t k,
                        size_t first, size_t count) {
    size_t q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= count * n) return;
    size_t i = first + q / n, j = q % n;
    unsigned int a = d[idx2(k, i, n)], b = d[idx2(j, k, n)];
    unsigned int old = d[idx2(j, i, n)];
    if (a < INF && b < INF && a + b < old) {
        d[idx2(j, i, n)] = a + b;
        p[idx2(j, i, n)] = static_cast<unsigned int>(k);
    }
}

static void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

void initializeDistanceMatrix(std::vector<unsigned int>& d, size_t n, unsigned int lo, unsigned int hi) {
    unsigned int seed = 42;
    const double range = static_cast<double>(hi - lo) + 1.0;
    for (size_t x = 0; x < n*n; ++x) d[x] = lo + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    #pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(n); ++i) d[idx2(i, i, n)] = 0;
}
void initializePathMatrix(std::vector<unsigned int>& p, size_t n) {
    #pragma omp parallel for
    for (long long x = 0; x < static_cast<long long>(n*n); ++x) {
        size_t i = static_cast<size_t>(x) % n, j = static_cast<size_t>(x) / n;
        p[static_cast<size_t>(x)] = (i == j) ? static_cast<unsigned int>(i) : static_cast<unsigned int>(j);
    }
}

void floydWarshall(std::vector<unsigned int>& d, std::vector<unsigned int>& p, size_t n, int rank, int ranks) {
    const size_t first = n * static_cast<size_t>(rank) / ranks;
    const size_t last = n * static_cast<size_t>(rank + 1) / ranks;
    const size_t count = last - first, total = n*n;
    unsigned int *gd, *gp;
    cuda_check(cudaMalloc(&gd, total*sizeof(unsigned int)));
    cuda_check(cudaMalloc(&gp, total*sizeof(unsigned int)));
    cuda_check(cudaMemcpy(gd, d.data(), total*sizeof(unsigned int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(gp, p.data(), total*sizeof(unsigned int), cudaMemcpyHostToDevice));
    std::vector<unsigned int> localPath(total), mergedPath(total);
    for (size_t k = 0; k < n; ++k) {
        if (count) {
            size_t work = count*n;
            fw_step<<<static_cast<unsigned int>((work+255)/256),256>>>(gd,gp,n,k,first,count);
            cuda_check(cudaGetLastError()); cuda_check(cudaDeviceSynchronize());
        }
        cuda_check(cudaMemcpy(d.data(), gd, total*sizeof(unsigned int), cudaMemcpyDeviceToHost));
        // Every rank has identical distances except for its owned source rows; MIN merges updates.
        MPI_Allreduce(MPI_IN_PLACE, d.data(), static_cast<int>(total), MPI_UNSIGNED, MPI_MIN, MPI_COMM_WORLD);
        cuda_check(cudaMemcpy(gd, d.data(), total*sizeof(unsigned int), cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(localPath.data(), gp, total*sizeof(unsigned int), cudaMemcpyDeviceToHost));
        for (size_t i=0;i<n;++i) if (i < first || i >= last)
            for (size_t j=0;j<n;++j) localPath[idx2(j,i,n)] = 0;
        MPI_Allreduce(localPath.data(), mergedPath.data(), static_cast<int>(total), MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
        p.swap(mergedPath); mergedPath.resize(total);
        cuda_check(cudaMemcpy(gp, p.data(), total*sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    cuda_check(cudaMemcpy(d.data(), gd, total*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaFree(gd); cudaFree(gp);
}

bool validateResult(const std::vector<unsigned int>& d, size_t n) {
    for (size_t i=0;i<n;++i) if(d[idx2(i,i,n)] != 0) { printf("Validation failed: diagonal element [%zu,%zu] is not zero\n",i,i); return false; }
    int valid=1;
    #pragma omp parallel for reduction(&:valid)
    for (int i=0;i<static_cast<int>(std::min(n,size_t(10)));++i) for(int j=0;j<static_cast<int>(std::min(n,size_t(10)));++j)
        for(size_t k=0;k<n;++k) { auto a=d[idx2(k,i,n)], b=d[idx2(j,k,n)]; if(a<INF && b<INF && a+b<d[idx2(j,i,n)]) valid=false; }
    if(!valid) printf("Validation failed: triangle inequality violated\n");
    return valid != 0;
}
void printUsage(const char* name) { printf("Usage: %s [options]\nOptions:\n  -n <num>     Number of nodes in the graph (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n",name); }

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv); int rank,ranks; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    size_t n=512; bool validate=false, printResults=false; int bad=0;
    for(int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"-n")&&i+1<argc) n=static_cast<size_t>(atoi(argv[++i]));
        else if(!strcmp(argv[i],"-v")) validate=true; else if(!strcmp(argv[i],"-r")) printResults=true;
        else if(!strcmp(argv[i],"-h")) { if(rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) { printf("Unknown option: %s\n",argv[i]); printUsage(argv[0]); } bad=1; }
    }
    MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD); if(bad){MPI_Finalize();return 1;}
    if(n==0 || n*n>static_cast<size_t>(INT_MAX)){if(rank==0)fprintf(stderr,"Invalid matrix size\n");MPI_Finalize();return 1;}
    std::vector<unsigned int> dist(n*n),path(n*n);
    if(rank==0) { printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\nInitializing graph...\n",n,validate?"enabled":"disabled"); initializeDistanceMatrix(dist,n,1,MAX_DISTANCE); initializePathMatrix(path,n); }
    MPI_Bcast(dist.data(),static_cast<int>(n*n),MPI_UNSIGNED,0,MPI_COMM_WORLD); MPI_Bcast(path.data(),static_cast<int>(n*n),MPI_UNSIGNED,0,MPI_COMM_WORLD);
    if(rank==0) printf("Computing shortest paths...\n"); MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
    floydWarshall(dist,path,n,rank,ranks); MPI_Barrier(MPI_COMM_WORLD); auto end=std::chrono::high_resolution_clock::now();
    if(rank==0) {
        auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); printf("Computation time: %ld ms\n",static_cast<long>(ms));
        double gops=static_cast<double>(n)*n*n/(ms/1000.0)/1e9; printf("Performance: %.3f GOPS\n",gops);
        if(printResults) print_results_int(dist,"DistanceMatrix");
        if(validate) { printf("Validating result...\n"); bool ok=validateResult(dist,n); printf("Validation: %s\n",ok?"PASSED":"FAILED"); MPI_Finalize(); return ok?0:1; }
    }
    MPI_Finalize(); return 0;
}
