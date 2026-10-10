#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <exception>
#include <utility>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../common/results_output.hpp"

namespace {
constexpr int tile = 256;
constexpr size_t area = size_t(tile) * tile;
int rankId = 0;
void fail(const char* message) {
    fprintf(stderr, "Rank %d: %s\n", rankId, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}
void cudaCheck(cudaError_t e) { if (e != cudaSuccess) fail(cudaGetErrorString(e)); }
void blasCheck(cublasStatus_t e) { if (e != CUBLAS_STATUS_SUCCESS) fail("cuBLAS operation failed"); }
void mpiCheck(int e) { if (e != MPI_SUCCESS) fail("MPI operation failed"); }
struct Device {
    double* p = nullptr;
    explicit Device(size_t count) { cudaCheck(cudaMalloc(&p, std::max(size_t(1), count) * sizeof(double))); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
struct Pinned {
    double* p = nullptr;
    explicit Pinned(size_t count) { cudaCheck(cudaMallocHost(&p, std::max(size_t(1), count) * sizeof(double))); }
    ~Pinned() { cudaFreeHost(p); }
};
// Chunk collectives so neither matrix size nor process count overflows MPI's int count.
void broadcast(double* p, size_t count, int root, MPI_Comm comm) {
    while (count) {
        int chunk = int(std::min(count, size_t(INT_MAX)));
        mpiCheck(MPI_Bcast(p, chunk, MPI_DOUBLE, root, comm));
        p += chunk; count -= chunk;
    }
}

__global__ void addDiagonal(double* a, int valid, int n) {
    int i = threadIdx.x;
    if (i < tile) a[i + i * tile] += i < valid ? double(n) : 1.0;
}
// The diagonal tile is small; one cooperative block avoids a kernel launch per column.
// Rows within a column are independent after its pivot has been published.
__global__ void factorDiagonal(double* a, int* info, int offset, int valid) {
    __shared__ double pivot;
    int i = threadIdx.x;
    for (int j = 0; j < valid; ++j) {
        if (i == j) {
            double v = a[j + j * tile];
            for (int k = 0; k < j; ++k) v -= a[j + k * tile] * a[j + k * tile];
            if (!(v > 0.0) || !isfinite(v)) *info = offset + j + 1;
            pivot = sqrt(v);
            a[j + j * tile] = pivot;
        }
        __syncthreads();
        if (*info) return;
        if (i > j) {
            double v = a[i + j * tile];
            for (int k = 0; k < j; ++k) v -= a[i + k * tile] * a[j + k * tile];
            a[i + j * tile] = v / pivot;
        }
        if (i < j) a[i + j * tile] = 0.0;
        __syncthreads();
    }
}

// glibc rand_r uses three advances of this LCG for each output. Jumping the
// state preserves the original seed-42 sequence independently of scheduling.
#if defined(__GLIBC__)
unsigned int rowSeed(uint64_t row, uint64_t n) {
    uint32_t mul = 1103515245u, add = 12345u, am = 1, aa = 0;
    uint64_t steps = 3 * row * n;
    while (steps) {
        if (steps & 1) { aa = mul * aa + add; am = mul * am; }
        add = mul * add + add; mul *= mul; steps >>= 1;
    }
    return am * 42u + aa;
}
#endif
void generateRows(double* b, int groups, int coordinate, int stride, int n,
                  const std::vector<unsigned int>& seeds) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < groups * tile; ++r) {
        uint64_t global = uint64_t(r / tile * stride + coordinate) * tile + r % tile;
        double* out = b + size_t(r) * n;
        if (global >= unsigned(n)) { std::fill(out, out + n, 0.0); continue; }
#if defined(__GLIBC__)
        (void)seeds;
        unsigned int seed = rowSeed(global, n);
#else
        unsigned int seed = seeds[global];
#endif
        for (int k = 0; k < n; ++k) out[k] = rand_r(&seed) / double(RAND_MAX) - 0.5;
    }
}

struct DistributedMatrix {
    int n, nt, pr, pc, row, col, nr, nc;
    MPI_Comm rows, cols;
    std::vector<int> index;
    std::vector<std::pair<int,int>> coordinates;
    cublasHandle_t handle;
    cudaStream_t stream;
    Device a, rowPanel, colPanel, diagonal;
    Pinned hostRows, hostCols, scratch;
    double** batchPointers = nullptr;
    std::vector<const double*> batchLeft, batchRight;
    std::vector<double*> batchOutput;

    static size_t tileCount(int nt, int pr, int pc, int row, int col) {
        size_t count = 0;
        for (int i = row; i < nt; i += pr)
            for (int j = col; j <= i; j += pc) ++count;
        return count;
    }
    DistributedMatrix(int size, int pRows, int pCols)
        : n(size), nt((n + tile - 1) / tile), pr(pRows), pc(pCols),
          row(rankId / pc), col(rankId % pc),
          nr((nt + pr - 1 - row) / pr), nc((nt + pc - 1 - col) / pc),
          index(size_t(nr)*nc, -1), a(tileCount(nt,pr,pc,row,col)*area),
          rowPanel(size_t(nr)*area), colPanel(size_t(nc)*area), diagonal(area),
          hostRows(size_t(nr)*area), hostCols(size_t(nc)*area), scratch(size_t(nc)*area) {
        cudaCheck(cudaMalloc(&batchPointers, std::max(size_t(1), tileCount(nt,pr,pc,row,col)*3)*sizeof(double*)));
        batchLeft.reserve(tileCount(nt,pr,pc,row,col));
        batchRight.reserve(tileCount(nt,pr,pc,row,col));
        batchOutput.reserve(tileCount(nt,pr,pc,row,col));
        std::fill(hostRows.p,hostRows.p+size_t(nr)*area,0.0);
        std::fill(hostCols.p,hostCols.p+size_t(nc)*area,0.0);
        mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, row, col, &rows));
        mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, col, row, &cols));
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        blasCheck(cublasCreate(&handle));
        blasCheck(cublasSetStream(handle, stream));
        blasCheck(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
        for (int i = row; i < nt; i += pr) for (int j = col; j <= i; j += pc) {
            index[size_t(i/pr)*nc + j/pc] = int(coordinates.size());
            coordinates.emplace_back(i,j);
        }
    }
    ~DistributedMatrix() {
        cudaFree(batchPointers); cublasDestroy(handle); cudaStreamDestroy(stream);
        MPI_Comm_free(&rows); MPI_Comm_free(&cols);
    }
    double* at(int i, int j) { return a.p + size_t(index[size_t(i/pr)*nc + j/pc])*area; }
    void sync() { cudaCheck(cudaStreamSynchronize(stream)); }
    void generate() {
        std::vector<unsigned int> seeds;
#if !defined(__GLIBC__)
        seeds.resize(n);
        unsigned int seed = 42;
        for (int i = 0; i < n; ++i) {
            seeds[i] = seed;
            for (int k = 0; k < n; ++k) rand_r(&seed);
        }
#endif
        // Only the rows of B needed by this process row/column are materialized.
        Pinned bRows(size_t(nr)*tile*n), bCols(size_t(nc)*tile*n);
        generateRows(bRows.p, nr, row, pr, n, seeds);
        generateRows(bCols.p, nc, col, pc, n, seeds);
        Device dr(size_t(nr)*tile*n), dc(size_t(nc)*tile*n);
        cudaCheck(cudaMemcpyAsync(dr.p,bRows.p,size_t(nr)*tile*n*sizeof(double),cudaMemcpyHostToDevice,stream));
        cudaCheck(cudaMemcpyAsync(dc.p,bCols.p,size_t(nc)*tile*n*sizeof(double),cudaMemcpyHostToDevice,stream));
        const double one=1, zero=0;
        for (auto ij : coordinates) {
            int i=ij.first, j=ij.second;
            blasCheck(cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,tile,tile,n,&one,
                dr.p+size_t(i/pr)*tile*n,n,dc.p+size_t(j/pc)*tile*n,n,&zero,at(i,j),tile));
            if (i==j) addDiagonal<<<1,tile,0,stream>>>(at(i,j),std::min(tile,n-i*tile),n);
        }
        cudaCheck(cudaGetLastError()); sync();
    }
    // Broadcast L_ik along its process row. Redistribute the transposed panel
    // along process columns. Communication is O(n*b/sqrt(P)) per rank/step,
    // and pinned staging works with MPI implementations lacking CUDA awareness.
    void panels(int k) {
        if (col == k%pc) {
            for (int i=row; i<nt; i+=pr) if (i>=k)
                cudaCheck(cudaMemcpyAsync(hostRows.p+size_t(i/pr)*area,at(i,k),area*sizeof(double),cudaMemcpyDeviceToHost,stream));
            sync();
        }
        int firstRow=std::max(0,(k-row+pr-1)/pr);
        int firstCol=std::max(0,(k-col+pc-1)/pc);
        broadcast(hostRows.p+size_t(firstRow)*area,size_t(nr-firstRow)*area,k%pc,rows);
        for (int r=0; r<pr; ++r) {
            int count=0;
            for (int j=col; j<nt; j+=pc) if (j>=k && j%pr==r) ++count;
            if (!count) continue;
            if (row==r) {
                #pragma omp parallel for schedule(static) if(nc >= 8)
                for (int t=0; t<nc; ++t) {
                    int j=t*pc+col;
                    if (j<nt && j>=k && j%pr==r)
                        std::memcpy(hostCols.p+size_t(t)*area,hostRows.p+size_t(j/pr)*area,area*sizeof(double));
                }
                int s=0;
                for (int j=col; j<nt; j+=pc) if (j>=k && j%pr==r)
                    std::memcpy(scratch.p+size_t(s++)*area,hostCols.p+size_t(j/pc)*area,area*sizeof(double));
            }
            broadcast(scratch.p,size_t(count)*area,r,cols);
            int s=0;
            for (int j=col; j<nt; j+=pc) if (j>=k && j%pr==r)
                std::memcpy(hostCols.p+size_t(j/pc)*area,scratch.p+size_t(s++)*area,area*sizeof(double));
        }
        cudaCheck(cudaMemcpyAsync(rowPanel.p+size_t(firstRow)*area,hostRows.p+size_t(firstRow)*area,size_t(nr-firstRow)*area*sizeof(double),cudaMemcpyHostToDevice,stream));
        cudaCheck(cudaMemcpyAsync(colPanel.p+size_t(firstCol)*area,hostCols.p+size_t(firstCol)*area,size_t(nc-firstCol)*area*sizeof(double),cudaMemcpyHostToDevice,stream));
    }
    // Batch independent tile products to amortize launch overhead and keep
    // many GPU multiprocessors busy even when the trailing matrix is small.
    void update(int k, double alpha, double* target, bool includeDiagonal) {
        batchLeft.clear(); batchRight.clear(); batchOutput.clear();
        for (size_t t=0;t<coordinates.size();++t) {
            int i=coordinates[t].first,j=coordinates[t].second;
            if (j<k || (!includeDiagonal && j==k)) continue;
            batchLeft.push_back(rowPanel.p+size_t(i/pr)*area);
            batchRight.push_back(colPanel.p+size_t(j/pc)*area);
            batchOutput.push_back(target+t*area);
        }
        size_t count=batchOutput.size();
        if (!count) return;
        cudaCheck(cudaMemcpyAsync(batchPointers,batchLeft.data(),count*sizeof(double*),cudaMemcpyHostToDevice,stream));
        cudaCheck(cudaMemcpyAsync(batchPointers+count,batchRight.data(),count*sizeof(double*),cudaMemcpyHostToDevice,stream));
        cudaCheck(cudaMemcpyAsync(batchPointers+2*count,batchOutput.data(),count*sizeof(double*),cudaMemcpyHostToDevice,stream));
        const double one=1;
        blasCheck(cublasDgemmBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,tile,tile,tile,&alpha,
            batchPointers,tile,batchPointers+count,tile,&one,batchPointers+2*count,tile,int(count)));
    }
    bool factor() {
        int* info=nullptr;
        cudaCheck(cudaMalloc(&info,sizeof(int)));
        cudaCheck(cudaMemsetAsync(info,0,sizeof(int),stream));
        const double one=1, minus=-1;
        for (int k=0; k<nt; ++k) {
            int owner=(k%pr)*pc+k%pc, status=0;
            if (rankId==owner) {
                factorDiagonal<<<1,tile,0,stream>>>(at(k,k),info,k*tile,std::min(tile,n-k*tile));
                cudaCheck(cudaGetLastError());
                cudaCheck(cudaMemcpyAsync(&status,info,sizeof(int),cudaMemcpyDeviceToHost,stream));
                cudaCheck(cudaMemcpyAsync(diagonal.p,at(k,k),area*sizeof(double),cudaMemcpyDeviceToDevice,stream));
                cudaCheck(cudaMemcpyAsync(scratch.p,diagonal.p,area*sizeof(double),cudaMemcpyDeviceToHost,stream));
                sync();
            }
            mpiCheck(MPI_Bcast(&status,1,MPI_INT,owner,MPI_COMM_WORLD));
            if (status) {
                if (!rankId) printf("Error: Matrix is not positive definite at diagonal element %d\n",status-1);
                cudaFree(info); return false;
            }
            // Only the process column owning the panel needs the diagonal tile.
            if (col==k%pc) {
                broadcast(scratch.p,area,k%pr,cols);
                cudaCheck(cudaMemcpyAsync(diagonal.p,scratch.p,area*sizeof(double),cudaMemcpyHostToDevice,stream));
                for (int i=row; i<nt; i+=pr) if (i>k)
                    blasCheck(cublasDtrsm(handle,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,tile,tile,&one,diagonal.p,tile,at(i,k),tile));
            }
            panels(k);
            update(k,minus,a.p,false);
            // Finish before reusing the pinned panel staging buffers.
            sync();
        }
        cudaFree(info); return true;
    }
    bool validate(const Device& original) {
        Device reconstructed(coordinates.size()*area);
        cudaCheck(cudaMemsetAsync(reconstructed.p,0,coordinates.size()*area*sizeof(double),stream));
        const double one=1;
        for (int k=0; k<nt; ++k) {
            panels(k);
            update(k,one,reconstructed.p,true);
            sync();
        }
        Pinned rec(coordinates.size()*area), ref(coordinates.size()*area);
        cudaCheck(cudaMemcpyAsync(rec.p,reconstructed.p,coordinates.size()*area*sizeof(double),cudaMemcpyDeviceToHost,stream));
        cudaCheck(cudaMemcpyAsync(ref.p,original.p,coordinates.size()*area*sizeof(double),cudaMemcpyDeviceToHost,stream));
        sync();
        double maxError=0,relError=0;
        int bad=0;
        #pragma omp parallel for reduction(max:maxError,relError,bad) schedule(static)
        for (size_t t=0;t<coordinates.size();++t) {
            int i=coordinates[t].first,j=coordinates[t].second;
            for (int c=0;c<tile && j*tile+c<n;++c) for (int r=0;r<tile && i*tile+r<n;++r) {
                size_t x=t*area+c*tile+r;
                double error=std::fabs(rec.p[x]-ref.p[x]);
                if (!std::isfinite(error)) bad=1;
                maxError=std::max(maxError,error);
                relError=std::max(relError,error/(std::fabs(ref.p[x])+1e-10));
            }
        }
        double local[2]={maxError,relError},global[2];
        mpiCheck(MPI_Allreduce(local,global,2,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD));
        int invalid=0;
        mpiCheck(MPI_Allreduce(&bad,&invalid,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD));
        if (!rankId) {
            printf("Max absolute error: %.10e\nMax relative error: %.10e\n",global[0],global[1]);
            if (invalid || global[1]>1e-6) printf("Validation failed: relative error too large\n");
        }
        return !invalid && global[1]<=1e-6;
    }
    void results(int processes) {
        std::vector<double> output;
        if (!rankId) output.assign(size_t(n)*n,0.0);
        Pinned buffer(area);
        for (int p=0;p<processes;++p) {
            int r=p/pc,c=p%pc;
            for (int i=r;i<nt;i+=pr) for (int j=c;j<=i;j+=pc) {
                if (rankId==p) {
                    cudaCheck(cudaMemcpy(buffer.p,at(i,j),area*sizeof(double),cudaMemcpyDeviceToHost));
                    if (p) mpiCheck(MPI_Send(buffer.p,int(area),MPI_DOUBLE,0,0,MPI_COMM_WORLD));
                }
                if (!rankId) {
                    if (p) mpiCheck(MPI_Recv(buffer.p,int(area),MPI_DOUBLE,p,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE));
                    for (int rr=0;rr<std::min(tile,n-i*tile);++rr)
                        for (int cc=0;cc<tile && j*tile+cc<n;++cc)
                            if (i*tile+rr>=j*tile+cc)
                                output[size_t(i*tile+rr)*n+j*tile+cc]=buffer.p[cc*tile+rr];
                }
            }
        }
        if (!rankId) print_results(output,"CholeskyL");
    }
};
void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n",name);
    printf("  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n  -h           Show this help message\n");
}
}
int main(int argc,char** argv) {
    int provided=0;
    mpiCheck(MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided));
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD,&rankId));
    if (provided<MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED is required");
    int processes=1;
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD,&processes));
    int n=512; bool validation=false,print=false;
    for (int i=1;i<argc;++i) {
        if (!strcmp(argv[i],"-n") && i+1<argc) {
            char* end=nullptr; errno=0;
            unsigned long long value=strtoull(argv[++i],&end,10);
            if (errno || *end || !*argv[i] || value>INT_MAX-tile || argv[i][0]=='-' ||
                (value && value>std::numeric_limits<size_t>::max()/sizeof(double)/value)) {
                if (!rankId) fprintf(stderr,"Invalid matrix size: %s\n",argv[i]);
                MPI_Finalize(); return 1;
            }
            n=int(value);
        } else if (!strcmp(argv[i],"-v")) validation=true;
        else if (!strcmp(argv[i],"-r")) print=true;
        else if (!strcmp(argv[i],"-h")) { if (!rankId) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rankId) { printf("Unknown option: %s\n",argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    MPI_Comm node;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rankId,MPI_INFO_NULL,&node));
    int localRank=0,devices=0;
    mpiCheck(MPI_Comm_rank(node,&localRank));
    cudaCheck(cudaGetDeviceCount(&devices));
    if (!devices) fail("CUDA GPU required");
    cudaCheck(cudaSetDevice(localRank%devices));
    MPI_Comm_free(&node);
    int dims[2]={0,0}; mpiCheck(MPI_Dims_create(processes,2,dims));
    if (!rankId) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n",n,n,validation?"enabled":"disabled");
        printf("MPI grid: %d x %d; OpenMP threads per rank: %d; CUDA tile size: %d\n",dims[0],dims[1],omp_get_max_threads(),tile);
        printf("Generating positive definite matrix...\n");
    }
    int result=0;
    try {
        DistributedMatrix matrix(n,dims[0],dims[1]);
        matrix.generate();
        Device original(validation?matrix.coordinates.size()*area:0);
        if (validation) {
            cudaCheck(cudaMemcpyAsync(original.p,matrix.a.p,matrix.coordinates.size()*area*sizeof(double),cudaMemcpyDeviceToDevice,matrix.stream));
            matrix.sync();
        }
        if (!rankId) printf("Computing Cholesky decomposition...\n");
        mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
        double start=MPI_Wtime();
        bool success=matrix.factor();
        double elapsed=MPI_Wtime()-start,seconds=0;
        mpiCheck(MPI_Reduce(&elapsed,&seconds,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD));
        if (!success) { if (!rankId) printf("Cholesky decomposition failed\n"); result=1; }
        else {
            if (!rankId) printf("Computation time: %.3f ms\nPerformance: %.3f GFLOPS\n",seconds*1000,double(n)*n*n/3/seconds/1e9);
            if (print) matrix.results(processes);
            if (validation) {
                if (!rankId) printf("Validating result...\n");
                bool valid=matrix.validate(original);
                if (!rankId) printf("Validation: %s\n",valid?"PASSED":"FAILED");
                if (!valid) result=1;
            }
        }
    } catch (const std::exception& e) { fail(e.what()); }
    mpiCheck(MPI_Finalize());
    return result;
}
