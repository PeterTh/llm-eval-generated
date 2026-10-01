#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include "../common/results_output.hpp"

constexpr unsigned INF = 1000000000;
constexpr int B = 32;
static void gpu(cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1); }
}
static void mpi(int e) {
    if (e != MPI_SUCCESS) { fprintf(stderr, "MPI operation failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
}

// A block updates the pivot tile, synchronizing between intermediate vertices.
__global__ void diagonal(unsigned* dist, unsigned* path, int pitch, int local, int pivot) {
    __shared__ unsigned s[B][B+1];
    int x=threadIdx.x, y=threadIdx.y;
    size_t at=size_t(local*B+y)*pitch+pivot*B+x;
    unsigned value=dist[at], via=path[at];
    s[y][x]=value;
    __syncthreads();
    for (int k=0;k<B;++k) {
        unsigned candidate=s[y][k]+s[k][x];
        if (candidate<value) { value=candidate; via=pivot*B+k; }
        __syncthreads();
        s[y][x]=value;
        __syncthreads();
    }
    dist[at]=value; path[at]=via;
}

// Pivot row and pivot column use the completed diagonal tile.
template<bool ROW>
__global__ void edge(unsigned* dist, unsigned* path, const unsigned* diag,
                     int pitch, int localPivot, int startTile, int pivot) {
    int tile=blockIdx.x;
    if (ROW ? tile==pivot : startTile+tile==pivot) return;
    int x=threadIdx.x, y=threadIdx.y;
    __shared__ unsigned d[B][B+1], c[B][B+1];
    size_t at=ROW ? size_t(localPivot*B+y)*pitch+tile*B+x
                  : size_t(tile*B+y)*pitch+pivot*B+x;
    d[y][x]=diag[y*pitch+pivot*B+x];
    unsigned value=dist[at], via=path[at];
    c[y][x]=value;
    __syncthreads();
    for (int k=0;k<B;++k) {
        unsigned candidate=ROW ? d[y][k]+c[k][x] : c[y][k]+d[k][x];
        if (candidate<value) { value=candidate; via=pivot*B+k; }
        __syncthreads();
        c[y][x]=value;
        __syncthreads();
    }
    dist[at]=value; path[at]=via;
}

__global__ void remaining(unsigned* dist, unsigned* path, const unsigned* pivotRow,
                          int pitch, int startTile, int pivot) {
    int col=blockIdx.x, tile=blockIdx.y;
    if (col==pivot || startTile+tile==pivot) return;
    int x=threadIdx.x, y=threadIdx.y;
    __shared__ unsigned left[B][B+1], top[B][B+1];
    size_t base=size_t(tile*B+y)*pitch;
    left[y][x]=dist[base+pivot*B+x];
    top[y][x]=pivotRow[y*pitch+col*B+x];
    size_t at=base+col*B+x;
    unsigned value=dist[at], via=path[at];
    __syncthreads();
    for (int k=0;k<B;++k) {
        unsigned candidate=left[y][k]+top[k][x];
        if (candidate<value) { value=candidate; via=pivot*B+k; }
    }
    dist[at]=value; path[at]=via;
}

static bool validate(const std::vector<unsigned>& d, size_t n) {
    for (size_t i=0;i<n;++i) if (d[i*n+i]!=0) {
        printf("Validation failed: diagonal element [%zu,%zu] is not zero\n",i,i); return false;
    }
    for (size_t i=0;i<std::min(n,size_t(10));++i)
        for (size_t j=0;j<std::min(n,size_t(10));++j)
            for (size_t k=0;k<n;++k)
                if (d[i*n+k]<INF && d[k*n+j]<INF && d[i*n+k]+d[k*n+j]<d[i*n+j]) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",i,j,k);
                    return false;
                }
    return true;
}
static void usage(const char* prog) {
    printf("Usage: %s [options]\nOptions:\n",prog);
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    mpi(MPI_Init(&argc,&argv));
    int rank,ranks; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    size_t n=512; bool check=false,print=false;
    for (int i=1;i<argc;++i) {
        if (!strcmp(argv[i],"-n") && i+1<argc) {
            char* end=nullptr; unsigned long long value=strtoull(argv[++i],&end,10);
            if (*end || !value || value>INT_MAX-B) {
                if (!rank) fprintf(stderr,"Invalid number of nodes\n");
                MPI_Finalize(); return 1;
            }
            n=size_t(value);
        } else if (!strcmp(argv[i],"-v")) check=true;
        else if (!strcmp(argv[i],"-r")) print=true;
        else if (!strcmp(argv[i],"-h")) { if (!rank) usage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n",argv[i]); usage(argv[0]); }
               MPI_Finalize(); return 1; }
    }
    int tiles=int((n+B-1)/B), pitch=tiles*B;
    auto first=[&](int r) { return r*(tiles/ranks)+std::min(r,tiles%ranks); };
    auto count=[&](int r) { return tiles/ranks+(r<tiles%ranks); };
    int localTiles=count(rank), startTile=first(rank);
    size_t localSize=size_t(localTiles)*B*pitch;
    if (size_t(pitch)*pitch>INT_MAX) {
        if (!rank) fprintf(stderr,"Graph exceeds MPI message size\n");
        MPI_Finalize(); return 1;
    }
    MPI_Comm shared;
    mpi(MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&shared));
    int localRank=0, devices=0; MPI_Comm_rank(shared,&localRank);
    gpu(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) fprintf(stderr,"A CUDA GPU is required\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    gpu(cudaSetDevice(localRank%devices)); MPI_Comm_free(&shared);
    if (!rank) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\nValidation: %s\n",n,check?"enabled":"disabled");
        printf("Initializing graph...\n");
    }
    std::vector<unsigned> packed;
    if (!rank) {
        std::vector<unsigned> input(n*n);
        unsigned seed=42;
        for (size_t i=0;i<n*n;++i)
            input[i]=1+unsigned(200.0*rand_r(&seed)/double(RAND_MAX));
        for (size_t i=0;i<n;++i) input[i*n+i]=0;
        packed.assign(size_t(pitch)*pitch,INF);
#pragma omp parallel for schedule(static)
        for (size_t row=0;row<n;++row)
            std::copy_n(input.data()+row*n,n,packed.data()+row*pitch);
    }
    std::vector<int> counts(ranks),offsets(ranks);
    for (int r=0;r<ranks;++r) { counts[r]=count(r)*B*pitch; offsets[r]=first(r)*B*pitch; }
    std::vector<unsigned> hostDist(localSize),hostPath(localSize);
    mpi(MPI_Scatterv(rank==0?packed.data():nullptr,counts.data(),offsets.data(),MPI_UNSIGNED,
                     hostDist.data(),int(localSize),MPI_UNSIGNED,0,MPI_COMM_WORLD));
    std::vector<unsigned>().swap(packed);
#pragma omp parallel for schedule(static)
    for (int row=0;row<localTiles*B;++row)
        std::fill_n(hostPath.data()+size_t(row)*pitch,pitch,unsigned(startTile*B+row));
    unsigned *dDist=nullptr,*dPath=nullptr,*dPivot=nullptr,*pivotHost=nullptr;
    gpu(cudaMalloc(&dDist,std::max(size_t(1),localSize)*sizeof(unsigned)));
    gpu(cudaMalloc(&dPath,std::max(size_t(1),localSize)*sizeof(unsigned)));
    gpu(cudaMalloc(&dPivot,size_t(B)*pitch*sizeof(unsigned)));
    gpu(cudaMallocHost(&pivotHost,size_t(B)*pitch*sizeof(unsigned)));
    if (localSize) {
        gpu(cudaMemcpy(dDist,hostDist.data(),localSize*sizeof(unsigned),cudaMemcpyHostToDevice));
        gpu(cudaMemcpy(dPath,hostPath.data(),localSize*sizeof(unsigned),cudaMemcpyHostToDevice));
    }
    std::vector<unsigned>().swap(hostPath);
    std::vector<unsigned>().swap(hostDist);
    if (!rank) printf("Computing shortest paths...\n");
    mpi(MPI_Barrier(MPI_COMM_WORLD));
    double begin=MPI_Wtime();
    dim3 threads(B,B);
    for (int p=0;p<tiles;++p) {
        int owner=0;
        while (p>=first(owner)+count(owner)) ++owner;
        if (rank==owner) {
            int local=p-startTile;
            diagonal<<<1,threads>>>(dDist,dPath,pitch,local,p);
            edge<true><<<tiles,threads>>>(dDist,dPath,dDist+size_t(local)*B*pitch,
                                           pitch,local,startTile,p);
            gpu(cudaGetLastError());
            gpu(cudaMemcpy(pivotHost,dDist+size_t(local)*B*pitch,
                           size_t(B)*pitch*sizeof(unsigned),cudaMemcpyDeviceToHost));
        }
        mpi(MPI_Bcast(pivotHost,B*pitch,MPI_UNSIGNED,owner,MPI_COMM_WORLD));
        gpu(cudaMemcpy(dPivot,pivotHost,size_t(B)*pitch*sizeof(unsigned),cudaMemcpyHostToDevice));
        if (localTiles) {
            edge<false><<<localTiles,threads>>>(dDist,dPath,dPivot,pitch,0,startTile,p);
            remaining<<<dim3(tiles,localTiles),threads>>>(dDist,dPath,dPivot,pitch,startTile,p);
            gpu(cudaGetLastError());
        }
    }
    gpu(cudaDeviceSynchronize());
    double elapsed=MPI_Wtime()-begin,seconds=0;
    mpi(MPI_Reduce(&elapsed,&seconds,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD));
    hostDist.resize(localSize);
    if (localSize) gpu(cudaMemcpy(hostDist.data(),dDist,localSize*sizeof(unsigned),cudaMemcpyDeviceToHost));
    if (!rank) packed.resize(size_t(pitch)*pitch);
    mpi(MPI_Gatherv(hostDist.data(),int(localSize),MPI_UNSIGNED,rank==0?packed.data():nullptr,
                    counts.data(),offsets.data(),MPI_UNSIGNED,0,MPI_COMM_WORLD));
    gpu(cudaFreeHost(pivotHost)); gpu(cudaFree(dPivot)); gpu(cudaFree(dPath)); gpu(cudaFree(dDist));
    int result=0;
    if (!rank) {
        std::vector<unsigned> dist(n*n);
#pragma omp parallel for schedule(static)
        for (size_t row=0;row<n;++row)
            std::copy_n(packed.data()+row*pitch,n,dist.data()+row*n);
        printf("Computation time: %ld ms\n",long(seconds*1000));
        printf("Performance: %.3f GOPS\n",double(n)*n*n/seconds/1e9);
        if (print) print_results_int(dist,"DistanceMatrix");
        if (check) {
            printf("Validating result...\n");
            result=validate(dist,n)?0:1;
            printf("Validation: %s\n",result?"FAILED":"PASSED");
        }
    }
    mpi(MPI_Bcast(&result,1,MPI_INT,0,MPI_COMM_WORLD));
    MPI_Finalize(); return result;
}
