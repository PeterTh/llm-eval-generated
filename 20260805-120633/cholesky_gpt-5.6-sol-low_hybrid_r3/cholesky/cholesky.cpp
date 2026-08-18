#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Block rows are distributed over MPI ranks.  Each rank keeps its rows on one
// GPU; panels are exchanged through host buffers so this also works with MPI
// implementations which are not CUDA-aware.
static constexpr int BLOCK_SIZE = 128;

static void cudaCheck(cudaError_t e, const char *what, MPI_Comm comm) {
    if (e == cudaSuccess) return;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    std::fprintf(stderr, "Rank %d: %s: %s\n", rank, what, cudaGetErrorString(e));
    MPI_Abort(comm, 2);
}

__global__ void factorDiagonal(double *a, size_t ld, size_t local0,
                               size_t k, int nb, int *ok) {
    if (blockIdx.x || threadIdx.x) return;
    for (int j = 0; j < nb; ++j) {
        double *row = a + (k + j - local0) * ld;
        double s = 0.0;
        for (int p = 0; p < j; ++p) s += row[k + p] * row[k + p];
        const double v = row[k + j] - s;
        if (!(v > 0.0)) { *ok = 0; return; }
        row[k + j] = sqrt(v);
        for (int i = j + 1; i < nb; ++i) {
            double *r = a + (k + i - local0) * ld;
            s = 0.0;
            for (int p = 0; p < j; ++p) s += r[k + p] * row[k + p];
            r[k + j] = (r[k + j] - s) / row[k + j];
        }
    }
}

__global__ void solvePanel(double *a, size_t ld, size_t local0,
                           size_t first, size_t rows, size_t k, int nb,
                           const double *diag) {
    const size_t q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= rows) return;
    const size_t gi = first + q;
    double *r = a + (gi - local0) * ld;
    for (int j = 0; j < nb; ++j) {
        double s = r[k + j];
        for (int p = 0; p < j; ++p) s -= r[k + p] * diag[j * nb + p];
        r[k + j] = s / diag[j * nb + j];
    }
}

__global__ void trailingUpdate(double *a, size_t ld, size_t local0,
                               size_t first, size_t rows, size_t n,
                               size_t kend, int nb, const double *panel) {
    const size_t j = kend + blockIdx.x * blockDim.x + threadIdx.x;
    const size_t q = blockIdx.y * blockDim.y + threadIdx.y;
    if (q >= rows || j >= n) return;
    const size_t i = first + q;
    if (j > i) return;
    double s = 0.0;
    const double *pi = panel + i * nb;
    const double *pj = panel + j * nb;
#pragma unroll 4
    for (int p = 0; p < nb; ++p) s += pi[p] * pj[p];
    a[(i - local0) * ld + j] -= s;
}

__global__ void clearUpper(double *a, size_t ld, size_t local0, size_t rows) {
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < rows && j < ld && j > local0 + r) a[r * ld + j] = 0.0;
}

static size_t rowStart(int rank, int size, size_t n) {
    return n * static_cast<size_t>(rank) / static_cast<size_t>(size);
}
static int rowOwner(size_t row, int size, size_t n) {
    // The loop is inexpensive and handles ranks owning zero rows.
    for (int r = 0; r < size; ++r)
        if (row >= rowStart(r, size, n) && row < rowStart(r + 1, size, n)) return r;
    return size - 1;
}

static void generateLocal(std::vector<double>& a, size_t n, size_t first,
                          size_t rows) {
    std::vector<double> b(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        b[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
#pragma omp parallel for schedule(static)
    for (long long rr = 0; rr < static_cast<long long>(rows); ++rr) {
        const size_t i = first + static_cast<size_t>(rr);
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
#pragma omp simd reduction(+:s)
            for (size_t p = 0; p < n; ++p) s += b[i*n+p] * b[j*n+p];
            a[static_cast<size_t>(rr)*n+j] = s + (i == j ? static_cast<double>(n) : 0.0);
        }
    }
}

static bool decompose(std::vector<double>& local, size_t n, size_t first,
                      size_t rows, int rank, int ranks, MPI_Comm comm) {
    double *d_a = nullptr, *d_diag = nullptr, *d_panel = nullptr;
    int *d_ok = nullptr;
    const size_t allocRows = std::max<size_t>(rows, 1);
    cudaCheck(cudaMalloc(&d_a, allocRows*n*sizeof(double)), "cudaMalloc(matrix)", comm);
    cudaCheck(cudaMalloc(&d_diag, BLOCK_SIZE*BLOCK_SIZE*sizeof(double)), "cudaMalloc(diagonal)", comm);
    cudaCheck(cudaMalloc(&d_panel, std::max<size_t>(n*BLOCK_SIZE, 1)*sizeof(double)), "cudaMalloc(panel)", comm);
    cudaCheck(cudaMalloc(&d_ok, sizeof(int)), "cudaMalloc(status)", comm);
    if (rows) cudaCheck(cudaMemcpy(d_a, local.data(), rows*n*sizeof(double), cudaMemcpyHostToDevice), "copy matrix to GPU", comm);

    std::vector<double> diag(BLOCK_SIZE*BLOCK_SIZE), send(allocRows*BLOCK_SIZE), panel(n*BLOCK_SIZE);
    std::vector<int> counts(ranks), displs(ranks);
    bool success = true;
    for (size_t k = 0; k < n; ) {
        const int owner = rowOwner(k, ranks, n);
        const size_t ownerEnd = rowStart(owner + 1, ranks, n);
        const int nb = static_cast<int>(std::min<size_t>(BLOCK_SIZE, std::min(n, ownerEnd) - k));
        const size_t kend = k + static_cast<size_t>(nb);
        int ok = 1;
        if (rank == owner) {
            cudaCheck(cudaMemcpy(d_ok, &ok, sizeof(int), cudaMemcpyHostToDevice), "initialize status", comm);
            factorDiagonal<<<1,1>>>(d_a, n, first, k, nb, d_ok);
            cudaCheck(cudaGetLastError(), "factor diagonal", comm);
            cudaCheck(cudaMemcpy(&ok, d_ok, sizeof(int), cudaMemcpyDeviceToHost), "read status", comm);
            cudaCheck(cudaMemcpy2D(diag.data(), nb*sizeof(double),
                                  d_a + (k-first)*n + k, n*sizeof(double),
                                  nb*sizeof(double), nb, cudaMemcpyDeviceToHost), "copy diagonal block", comm);
        }
        MPI_Bcast(&ok, 1, MPI_INT, owner, comm);
        if (!ok) { success = false; break; }
        MPI_Bcast(diag.data(), nb*nb, MPI_DOUBLE, owner, comm);
        cudaCheck(cudaMemcpy(d_diag, diag.data(), nb*nb*sizeof(double), cudaMemcpyHostToDevice), "copy diagonal to GPU", comm);

        const size_t solveFirst = std::max(first, kend);
        const size_t solveEnd = first + rows;
        const size_t solveRows = solveEnd > solveFirst ? solveEnd - solveFirst : 0;
        if (solveRows) solvePanel<<<(solveRows+255)/256,256>>>(d_a,n,first,solveFirst,solveRows,k,nb,d_diag);
        cudaCheck(cudaGetLastError(), "solve panel", comm);
        if (rows) cudaCheck(cudaMemcpy2D(send.data(), nb*sizeof(double), d_a+k, n*sizeof(double),
                                         nb*sizeof(double), rows, cudaMemcpyDeviceToHost), "stage panel", comm);
        for (int r = 0; r < ranks; ++r) {
            counts[r] = static_cast<int>((rowStart(r+1,ranks,n)-rowStart(r,ranks,n))*nb);
            displs[r] = static_cast<int>(rowStart(r,ranks,n)*nb);
        }
        MPI_Allgatherv(send.data(), static_cast<int>(rows*nb), MPI_DOUBLE,
                       panel.data(), counts.data(), displs.data(), MPI_DOUBLE, comm);
        cudaCheck(cudaMemcpy(d_panel, panel.data(), n*nb*sizeof(double), cudaMemcpyHostToDevice), "copy global panel", comm);
        const size_t updateFirst = std::max(first, kend);
        const size_t updateRows = solveEnd > updateFirst ? solveEnd-updateFirst : 0;
        if (updateRows) {
            dim3 block(16,16), grid((n-kend+15)/16, (updateRows+15)/16);
            trailingUpdate<<<grid,block>>>(d_a,n,first,updateFirst,updateRows,n,kend,nb,d_panel);
            cudaCheck(cudaGetLastError(), "trailing update", comm);
        }
        k = kend;
    }
    if (success && rows) {
        dim3 block(16,16), grid((n+15)/16,(rows+15)/16);
        clearUpper<<<grid,block>>>(d_a,n,first,rows);
        cudaCheck(cudaMemcpy(local.data(), d_a, rows*n*sizeof(double), cudaMemcpyDeviceToHost), "copy result from GPU", comm);
    }
    cudaFree(d_ok); cudaFree(d_panel); cudaFree(d_diag); cudaFree(d_a);
    return success;
}

static std::vector<double> gatherMatrix(const std::vector<double>& local, size_t n,
                                        int rank, int ranks, MPI_Comm comm) {
    std::vector<int> counts(ranks), displs(ranks);
    for (int r=0; r<ranks; ++r) {
        counts[r]=static_cast<int>((rowStart(r+1,ranks,n)-rowStart(r,ranks,n))*n);
        displs[r]=static_cast<int>(rowStart(r,ranks,n)*n);
    }
    std::vector<double> all(rank == 0 ? n*n : 0);
    MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                rank == 0 ? all.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    return all;
}

static bool validate(const std::vector<double>& l, const std::vector<double>& original,
                     size_t n, size_t first, size_t rows, int rank, int ranks, MPI_Comm comm) {
    std::vector<int> counts(ranks), displs(ranks);
    for (int r=0;r<ranks;++r) { counts[r]=static_cast<int>((rowStart(r+1,ranks,n)-rowStart(r,ranks,n))*n); displs[r]=static_cast<int>(rowStart(r,ranks,n)*n); }
    std::vector<double> all(n*n);
    MPI_Allgatherv(l.data(), static_cast<int>(l.size()), MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE, comm);
    double maxAbs=0.0, maxRel=0.0;
#pragma omp parallel for reduction(max:maxAbs,maxRel) schedule(static)
    for (long long rr=0; rr<static_cast<long long>(rows); ++rr) {
        const size_t i=first+static_cast<size_t>(rr);
        for (size_t j=0;j<n;++j) {
            double s=0.0;
#pragma omp simd reduction(+:s)
            for (size_t p=0;p<=std::min(i,j);++p) s += all[i*n+p]*all[j*n+p];
            const double e=std::fabs(s-original[static_cast<size_t>(rr)*n+j]);
            maxAbs=std::max(maxAbs,e); maxRel=std::max(maxRel,e/(std::fabs(original[static_cast<size_t>(rr)*n+j])+1e-10));
        }
    }
    double global[2]={maxAbs,maxRel}, reduced[2];
    MPI_Reduce(global,reduced,2,MPI_DOUBLE,MPI_MAX,0,comm);
    if (rank==0) { std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n",reduced[0],reduced[1]); if(reduced[1]>1e-6) std::printf("Validation failed: relative error too large\n"); }
    return rank != 0 || reduced[1] <= 1e-6;
}

static void usage(const char *p) { std::printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n",p); }

int main(int argc, char **argv) {
    int provided=0; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int rank=0,ranks=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank=0; MPI_Comm_rank(nodeComm,&localRank);
    int devices=0; cudaCheck(cudaGetDeviceCount(&devices),"query CUDA devices",MPI_COMM_WORLD);
    if (!devices) { if(!rank) std::fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    // Bind by node-local rank: global-rank modulo device-count would collide on
    // multi-node jobs whose rank layout is not an exact multiple of GPU count.
    cudaCheck(cudaSetDevice(localRank%devices),"select CUDA device",MPI_COMM_WORLD);
    size_t n=512; bool doValidate=false, printResults=false; int parseOk=1, help=0;
    for(int i=1;i<argc;++i) {
        if(!std::strcmp(argv[i],"-n") && i+1<argc) { char *end=nullptr; unsigned long long v=std::strtoull(argv[++i],&end,10); if(!end||*end||v==0) parseOk=0; else n=static_cast<size_t>(v); }
        else if(!std::strcmp(argv[i],"-v")) doValidate=true;
        else if(!std::strcmp(argv[i],"-r")) printResults=true;
        else if(!std::strcmp(argv[i],"-h")) help=1;
        else parseOk=0;
    }
    if(help||!parseOk) { if(!rank) usage(argv[0]); MPI_Finalize(); return parseOk?0:1; }
    if (n > static_cast<size_t>(std::numeric_limits<int>::max()/std::max<size_t>(n,1))) { if(!rank) std::fprintf(stderr,"Matrix is too large for MPI counts\n"); MPI_Finalize(); return 1; }
    const size_t first=rowStart(rank,ranks,n), rows=rowStart(rank+1,ranks,n)-first;
    if(!rank) std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\n",n,n,doValidate?"enabled":"disabled",ranks,omp_get_max_threads());
    if(!rank) std::printf("Generating positive definite matrix...\n");
    std::vector<double> a(rows*n); generateLocal(a,n,first,rows);
    std::vector<double> original; if(doValidate) original=a;
    if(!rank) std::printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD); const double start=MPI_Wtime();
    const bool localSuccess=decompose(a,n,first,rows,rank,ranks,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD); const double elapsed=MPI_Wtime()-start;
    int success=localSuccess?1:0, allSuccess=0; MPI_Allreduce(&success,&allSuccess,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    if(!allSuccess) { if(!rank) std::printf("Cholesky decomposition failed\n"); MPI_Finalize(); return 1; }
    if(!rank) std::printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n",static_cast<long>(elapsed*1000.0),(static_cast<double>(n)*n*n/3.0)/elapsed/1e9);
    if(printResults) { auto full=gatherMatrix(a,n,rank,ranks,MPI_COMM_WORLD); if(!rank) print_results(full,"CholeskyL"); }
    int valid=1;
    if(doValidate) { if(!rank) std::printf("Validating result...\n"); valid=validate(a,original,n,first,rows,rank,ranks,MPI_COMM_WORLD)?1:0; MPI_Bcast(&valid,1,MPI_INT,0,MPI_COMM_WORLD); if(!rank) std::printf("Validation: %s\n",valid?"PASSED":"FAILED"); }
    MPI_Comm_free(&nodeComm); MPI_Finalize(); return valid?0:1;
}
