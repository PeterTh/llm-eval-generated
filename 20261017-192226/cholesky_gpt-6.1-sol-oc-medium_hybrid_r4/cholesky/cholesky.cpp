#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

namespace {
constexpr int tileSize = 256;
constexpr size_t tileElements = size_t(tileSize) * tileSize;
int rankId;

void check(bool ok, const char* operation) {
    if (!ok) {
        fprintf(stderr, "Rank %d: %s failed\n", rankId, operation);
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}
#define CUDA(call) check((call) == cudaSuccess, #call)
#define BLAS(call) check((call) == CUBLAS_STATUS_SUCCESS, #call)
#define SOLVER(call) check((call) == CUSOLVER_STATUS_SUCCESS, #call)
#define MPI_CHECK(call) check((call) == MPI_SUCCESS, #call)

template<class T> T* deviceAllocate(size_t count) {
    T* p = nullptr;
    check(count <= std::numeric_limits<size_t>::max() / sizeof(T), "device allocation size");
    CUDA(cudaMalloc(reinterpret_cast<void**>(&p), std::max(size_t(1), count) * sizeof(T)));
    return p;
}
double* pinnedAllocate(size_t count) {
    double* p = nullptr;
    CUDA(cudaMallocHost(reinterpret_cast<void**>(&p), std::max(size_t(1), count) * sizeof(double)));
    return p;
}
// Chunking avoids MPI's int count limit, including for very large panels.
void broadcast(double* p, size_t count, int root, MPI_Comm comm) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        MPI_CHECK(MPI_Bcast(p, chunk, MPI_DOUBLE, root, comm));
        p += chunk;
        count -= chunk;
    }
}
void transfer(double* p, size_t count, int source, int target) {
    if (source == target) return;
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        if (rankId == source)
            MPI_CHECK(MPI_Send(p, chunk, MPI_DOUBLE, target, 0, MPI_COMM_WORLD));
        if (rankId == target)
            MPI_CHECK(MPI_Recv(p, chunk, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        p += chunk;
        count -= chunk;
    }
}

__global__ void finishInput(double* a, int tiles, const int* rows, const int* cols, int n) {
    size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= size_t(tiles) * tileElements) return;
    int tile = int(t / tileElements);
    int r = int(t % tileSize), c = int(t % tileElements / tileSize);
    int i = rows[tile] * tileSize + r, j = cols[tile] * tileSize + c;
    if (i >= n || j >= n) a[t] = (i == j) ? 1.0 : 0.0;
    else if (i == j) a[t] += n;
}
__global__ void zeroUpper(double* a) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < tileSize)
        for (int c = r + 1; c < tileSize; ++c) a[r + c * tileSize] = 0.0;
}
__global__ void identity(double* a) {
    int r = threadIdx.x;
    a[r + r * tileSize] = 1.0;
}

struct Tile { int row, col; double* data; };

// Only lower-triangular tiles are stored. Tile (i,j) belongs to (i%Pr,j%Pc).
struct Matrix {
    int n, nt, pr, pc, row, col, nr, nc;
    MPI_Comm rowComm, colComm;
    cudaStream_t stream;
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    std::vector<Tile> tiles;
    std::vector<double*> lookup;
    double *data, *dRow, *dCol, *diag, *hRow, *hCol, *scratch, *hDiag;
    double **dPointers;
    double* work;
    int *dInfo, workSize;
    std::vector<double*> pointers;
    size_t capacity;

    Matrix(int size, int processes) : n(size), nt((n - 1) / tileSize + 1) {
        int dims[2] = {0, 0};
        MPI_CHECK(MPI_Dims_create(processes, 2, dims));
        pr = dims[0]; pc = dims[1]; row = rankId / pc; col = rankId % pc;
        MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm));
        MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm));
        nr = row < nt ? (nt - 1 - row) / pr + 1 : 0;
        nc = col < nt ? (nt - 1 - col) / pc + 1 : 0;
        lookup.resize(size_t(nr) * nc, nullptr);
        for (int i = row; i < nt; i += pr)
            for (int j = col; j <= i; j += pc) tiles.push_back({i, j, nullptr});
        data = deviceAllocate<double>(tiles.size() * tileElements);
        for (size_t t = 0; t < tiles.size(); ++t) {
            tiles[t].data = data + t * tileElements;
            lookup[size_t(tiles[t].row / pr) * nc + tiles[t].col / pc] = tiles[t].data;
        }
        CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        BLAS(cublasCreate(&blas)); BLAS(cublasSetStream(blas, stream));
        SOLVER(cusolverDnCreate(&solver)); SOLVER(cusolverDnSetStream(solver, stream));
        dRow = deviceAllocate<double>(size_t(nr) * tileElements);
        dCol = deviceAllocate<double>(size_t(nc) * tileElements);
        diag = deviceAllocate<double>(tileElements);
        hRow = pinnedAllocate(size_t(nr) * tileElements);
        hCol = pinnedAllocate(size_t(nc) * tileElements);
        scratch = pinnedAllocate(std::max(size_t(n) * tileSize, size_t(nc) * tileElements));
        hDiag = pinnedAllocate(tileElements);
        capacity = std::max(size_t(1), tiles.size());
        pointers.resize(3 * capacity);
        dPointers = deviceAllocate<double*>(3 * capacity);
        check(tiles.size() <= size_t(INT_MAX), "local tile count fits CUDA BLAS batch size");
        // Allocate and warm up the solver outside the benchmark's timed region.
        SOLVER(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, tileSize, diag, tileSize, &workSize));
        work = deviceAllocate<double>(workSize);
        dInfo = deviceAllocate<int>(1);
        CUDA(cudaMemsetAsync(diag, 0, tileElements * sizeof(double), stream));
        identity<<<1, tileSize, 0, stream>>>(diag);
        CUDA(cudaGetLastError());
        SOLVER(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, tileSize, diag, tileSize, work, workSize, dInfo));
        if (!tiles.empty()) {
            // Also load the triangular-solve kernels before timing the factor.
            CUDA(cudaMemsetAsync(dRow, 0, tileElements * sizeof(double), stream));
            pointers[0] = diag;
            pointers[capacity] = dRow;
            uploadPointers();
            double one = 1.0;
            BLAS(cublasDtrsmBatched(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                 CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, tileSize, tileSize, &one,
                 dPointers, tileSize, dPointers + capacity, tileSize, 1));
        }
        CUDA(cudaStreamSynchronize(stream));
    }
    ~Matrix() {
        CUDA(cudaStreamSynchronize(stream));
        CUDA(cudaFree(data)); CUDA(cudaFree(dRow)); CUDA(cudaFree(dCol)); CUDA(cudaFree(diag));
        CUDA(cudaFree(dPointers));
        CUDA(cudaFree(work)); CUDA(cudaFree(dInfo));
        CUDA(cudaFreeHost(hRow)); CUDA(cudaFreeHost(hCol));
        CUDA(cudaFreeHost(scratch)); CUDA(cudaFreeHost(hDiag));
        BLAS(cublasDestroy(blas)); SOLVER(cusolverDnDestroy(solver));
        CUDA(cudaStreamDestroy(stream));
        MPI_CHECK(MPI_Comm_free(&rowComm)); MPI_CHECK(MPI_Comm_free(&colComm));
    }
    double* at(int i, int j) { return lookup[size_t(i / pr) * nc + j / pc]; }
    void uploadPointers() {
        CUDA(cudaMemcpyAsync(dPointers, pointers.data(), pointers.size() * sizeof(double*),
                             cudaMemcpyHostToDevice, stream));
        // The host pointer lists are reused by the next batch; complete their
        // transfer before returning, independently of pageable-memory behavior.
        CUDA(cudaStreamSynchronize(stream));
    }

    void generate() {
        // Stream the original rand_r sequence through a single tile-row buffer.
        // Each B row is retained only by its process row and process column;
        // neither host nor device ever replicates the full dense input.
        double* left = deviceAllocate<double>(size_t(nr) * tileSize * n);
        double* right = deviceAllocate<double>(size_t(nc) * tileSize * n);
        unsigned int seed = 42;
        const size_t count = size_t(tileSize) * n;
        for (int t = 0; t < nt; ++t) {
            if (rankId == 0) {
                int realRows = std::min(tileSize, n - t * tileSize);
                for (size_t x = 0; x < size_t(realRows) * n; ++x)
                    scratch[x] = rand_r(&seed) / double(RAND_MAX) - 0.5;
                std::fill(scratch + size_t(realRows) * n, scratch + count, 0.0);
            }
            int owner = (t % pr) * pc + t % pc;
            transfer(scratch, count, 0, owner);
            if (row == t % pr) broadcast(scratch, count, t % pc, rowComm);
            if (col == t % pc) broadcast(scratch, count, t % pr, colComm);
            if (row == t % pr)
                CUDA(cudaMemcpyAsync(left + size_t(t / pr) * count, scratch, count * sizeof(double),
                                     cudaMemcpyHostToDevice, stream));
            if (col == t % pc)
                CUDA(cudaMemcpyAsync(right + size_t(t / pc) * count, scratch, count * sizeof(double),
                                     cudaMemcpyHostToDevice, stream));
            // MPI may overwrite the pinned staging buffer in the next iteration.
            CUDA(cudaStreamSynchronize(stream));
        }
        #pragma omp parallel for schedule(static) num_threads(std::max(1, std::min(omp_get_max_threads(), int(tiles.size()))))
        for (size_t t = 0; t < tiles.size(); ++t) {
            pointers[t] = left + size_t(tiles[t].row / pr) * count;
            pointers[capacity + t] = right + size_t(tiles[t].col / pc) * count;
            pointers[2 * capacity + t] = tiles[t].data;
        }
        if (!tiles.empty()) {
            uploadPointers();
            double one = 1.0, zero = 0.0;
            BLAS(cublasDgemmBatched(blas, CUBLAS_OP_T, CUBLAS_OP_N, tileSize, tileSize, n,
                 &one, dPointers, n, dPointers + capacity, n, &zero,
                 dPointers + 2 * capacity, tileSize, int(tiles.size())));
            std::vector<int> rows(tiles.size()), cols(tiles.size());
            for (size_t t = 0; t < tiles.size(); ++t) { rows[t] = tiles[t].row; cols[t] = tiles[t].col; }
            int* indices = deviceAllocate<int>(2 * tiles.size());
            CUDA(cudaMemcpyAsync(indices, rows.data(), rows.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
            CUDA(cudaMemcpyAsync(indices + tiles.size(), cols.data(), cols.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
            finishInput<<<(tiles.size() * tileElements + 255) / 256, 256, 0, stream>>>
                (data, int(tiles.size()), indices, indices + tiles.size(), n);
            CUDA(cudaGetLastError());
            CUDA(cudaStreamSynchronize(stream));
            CUDA(cudaFree(indices));
        }
        CUDA(cudaFree(left)); CUDA(cudaFree(right));
    }

    // Broadcast L_ik across process rows, then L_jk down process columns.
    // Host staging works with ordinary MPI; CUDA-aware MPI is not required.
    void panel(int k, bool includeDiagonal) {
        int first = k + (includeDiagonal ? 0 : 1);
        int r0 = std::min(nr, std::max(0, (first - row + pr - 1) / pr));
        int c0 = std::min(nc, std::max(0, (first - col + pc - 1) / pc));
        if (col == k % pc) {
            for (int r = r0; r < nr; ++r)
                CUDA(cudaMemcpyAsync(hRow + size_t(r) * tileElements, at(row + r * pr, k),
                     tileElements * sizeof(double), cudaMemcpyDeviceToHost, stream));
        }
        CUDA(cudaStreamSynchronize(stream));
        broadcast(hRow + size_t(r0) * tileElements, size_t(nr - r0) * tileElements, k % pc, rowComm);
        for (int root = 0; root < pr; ++root) {
            std::vector<int> columns;
            for (int c = c0; c < nc; ++c)
                if ((col + c * pc) % pr == root) columns.push_back(c);
            if (row == root) {
                #pragma omp parallel for schedule(static) num_threads(std::max(1, std::min(omp_get_max_threads(), int(columns.size()))))
                for (size_t x = 0; x < columns.size(); ++x) {
                    int j = col + columns[x] * pc;
                    std::memcpy(scratch + x * tileElements, hRow + size_t(j / pr) * tileElements,
                                tileElements * sizeof(double));
                }
            }
            broadcast(scratch, columns.size() * tileElements, root, colComm);
            #pragma omp parallel for schedule(static) num_threads(std::max(1, std::min(omp_get_max_threads(), int(columns.size()))))
            for (size_t x = 0; x < columns.size(); ++x)
                std::memcpy(hCol + size_t(columns[x]) * tileElements, scratch + x * tileElements,
                            tileElements * sizeof(double));
        }
        CUDA(cudaMemcpyAsync(dRow + size_t(r0) * tileElements, hRow + size_t(r0) * tileElements,
             size_t(nr - r0) * tileElements * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA(cudaMemcpyAsync(dCol + size_t(c0) * tileElements, hCol + size_t(c0) * tileElements,
             size_t(nc - c0) * tileElements * sizeof(double), cudaMemcpyHostToDevice, stream));
    }

    void update(int k, double* destination, bool reconstruction) {
        size_t batch = 0;
        for (size_t t = 0; t < tiles.size(); ++t) {
            const Tile& tile = tiles[t];
            if (tile.col < k + (reconstruction ? 0 : 1)) continue;
            pointers[batch] = dRow + size_t(tile.row / pr) * tileElements;
            pointers[capacity + batch] = dCol + size_t(tile.col / pc) * tileElements;
            pointers[2 * capacity + batch] = destination + t * tileElements;
            ++batch;
        }
        if (batch) {
            uploadPointers();
            double alpha = reconstruction ? 1.0 : -1.0, one = 1.0;
            BLAS(cublasDgemmBatched(blas, CUBLAS_OP_N, CUBLAS_OP_T, tileSize, tileSize, tileSize,
                 &alpha, dPointers, tileSize, dPointers + capacity, tileSize, &one,
                 dPointers + 2 * capacity, tileSize, int(batch)));
        }
    }

    bool factor() {
        bool success = true;
        for (int k = 0; k < nt; ++k) {
            int owner = (k % pr) * pc + k % pc;
            int info = 0;
            if (rankId == owner) {
                double* a = at(k, k);
                SOLVER(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, tileSize, a, tileSize, work, workSize, dInfo));
                zeroUpper<<<1, tileSize, 0, stream>>>(a);
                CUDA(cudaGetLastError());
                CUDA(cudaMemcpyAsync(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));
                CUDA(cudaMemcpyAsync(hDiag, a, tileElements * sizeof(double), cudaMemcpyDeviceToHost, stream));
                CUDA(cudaStreamSynchronize(stream));
            }
            MPI_CHECK(MPI_Bcast(&info, 1, MPI_INT, owner, MPI_COMM_WORLD));
            if (info) {
                if (rankId == 0) printf("Error: Matrix is not positive definite at diagonal element %d\n", k * tileSize + info - 1);
                success = false;
                break;
            }
            // Only the owning process column needs the diagonal factor.
            if (col == k % pc) {
                broadcast(hDiag, tileElements, k % pr, colComm);
                CUDA(cudaMemcpyAsync(diag, hDiag, tileElements * sizeof(double), cudaMemcpyHostToDevice, stream));
                size_t batch = 0;
                for (int i = row; i < nt; i += pr) if (i > k) {
                    pointers[batch] = diag;
                    pointers[capacity + batch] = at(i, k);
                    ++batch;
                }
                if (batch) {
                    uploadPointers();
                    double one = 1.0;
                    BLAS(cublasDtrsmBatched(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                         CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, tileSize, tileSize, &one,
                         dPointers, tileSize, dPointers + capacity, tileSize, int(batch)));
                }
            }
            panel(k, false);
            update(k, data, false);
        }
        CUDA(cudaStreamSynchronize(stream));
        return success;
    }

    bool validate(double* original) {
        size_t count = tiles.size() * tileElements;
        double* reconstructed = deviceAllocate<double>(count);
        CUDA(cudaMemsetAsync(reconstructed, 0, count * sizeof(double), stream));
        for (int k = 0; k < nt; ++k) {
            panel(k, true);
            update(k, reconstructed, true);
        }
        std::vector<double> reference(count), result(count);
        CUDA(cudaMemcpyAsync(reference.data(), original, count * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA(cudaMemcpyAsync(result.data(), reconstructed, count * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA(cudaStreamSynchronize(stream));
        double maxError = 0.0, relError = 0.0;
        #pragma omp parallel for reduction(max:maxError,relError) schedule(static) num_threads(std::max(1, std::min(omp_get_max_threads(), int(tiles.size()))))
        for (size_t t = 0; t < tiles.size(); ++t) {
            for (int c = 0; c < tileSize; ++c) for (int r = 0; r < tileSize; ++r) {
                int i = tiles[t].row * tileSize + r, j = tiles[t].col * tileSize + c;
                if (i >= n || j >= n || i < j) continue;
                size_t x = t * tileElements + c * tileSize + r;
                double error = std::abs(result[x] - reference[x]);
                double relative = error / (std::abs(reference[x]) + 1e-10);
                if (!std::isfinite(error) || !std::isfinite(relative)) error = relative = std::numeric_limits<double>::infinity();
                maxError = std::max(maxError, error);
                relError = std::max(relError, relative);
            }
        }
        double local[2] = {maxError, relError}, global[2];
        MPI_CHECK(MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
        if (rankId == 0) {
            printf("Max absolute error: %.10e\nMax relative error: %.10e\n", global[0], global[1]);
            if (global[1] > 1e-6) printf("Validation failed: relative error too large\n");
        }
        CUDA(cudaFree(reconstructed));
        return global[1] <= 1e-6;
    }

    void printResults() {
        std::vector<double> result;
        if (rankId == 0) result.assign(size_t(n) * n, 0.0);
        for (int i = 0; i < nt; ++i) for (int j = 0; j <= i; ++j) {
            int owner = (i % pr) * pc + j % pc;
            if (rankId == owner) {
                CUDA(cudaMemcpyAsync(hDiag, at(i, j), tileElements * sizeof(double), cudaMemcpyDeviceToHost, stream));
                CUDA(cudaStreamSynchronize(stream));
            }
            transfer(hDiag, tileElements, owner, 0);
            if (rankId == 0) {
                #pragma omp parallel for schedule(static) num_threads(std::min(omp_get_max_threads(), tileSize))
                for (int r = 0; r < std::min(tileSize, n - i * tileSize); ++r)
                    for (int c = 0; c < std::min(tileSize, n - j * tileSize); ++c)
                        if (i * tileSize + r >= j * tileSize + c)
                            result[size_t(i * tileSize + r) * n + j * tileSize + c] = hDiag[r + c * tileSize];
            }
        }
        if (rankId == 0) print_results(result, "CholeskyL");
    }
};

void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}
} // namespace

int main(int argc, char** argv) {
    int provided;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rankId));
    check(provided >= MPI_THREAD_FUNNELED, "MPI_THREAD_FUNNELED support");
    int processes;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &processes));
    int n = 512;
    bool validation = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end;
            errno = 0;
            long value = strtol(argv[++i], &end, 10);
            if (errno || *end || value <= 0 || value > INT_MAX - tileSize) {
                if (rankId == 0) fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                MPI_Finalize(); return 1;
            }
            n = int(value);
        } else if (strcmp(argv[i], "-v") == 0) validation = true;
        else if (strcmp(argv[i], "-r") == 0) results = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rankId == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rankId == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    MPI_Comm local;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rankId, MPI_INFO_NULL, &local));
    int localRank, localSize, devices;
    MPI_CHECK(MPI_Comm_rank(local, &localRank));
    MPI_CHECK(MPI_Comm_size(local, &localSize));
    // Packing is bandwidth-bound; avoid huge implicit OpenMP teams on hosts
    // with many cores. Explicit OMP_NUM_THREADS settings always take precedence.
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::min(8, std::max(1, omp_get_num_procs() / localSize)));
    CUDA(cudaGetDeviceCount(&devices));
    check(devices > 0, "CUDA device availability");
    CUDA(cudaSetDevice(localRank % devices));
    MPI_CHECK(MPI_Comm_free(&local));
    int exitCode = 0;
    {
        Matrix matrix(n, processes);
        if (rankId == 0) {
            printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\n", n, n);
            printf("Validation: %s\n", validation ? "enabled" : "disabled");
            printf("MPI grid: %d x %d; OpenMP threads/rank: %d; CUDA tile size: %d\n",
                   matrix.pr, matrix.pc, omp_get_max_threads(), tileSize);
            printf("Generating positive definite matrix...\n");
        }
        matrix.generate();
        double* original = nullptr;
        if (validation) {
            original = deviceAllocate<double>(matrix.tiles.size() * tileElements);
            CUDA(cudaMemcpyAsync(original, matrix.data, matrix.tiles.size() * tileElements * sizeof(double),
                                 cudaMemcpyDeviceToDevice, matrix.stream));
        }
        CUDA(cudaStreamSynchronize(matrix.stream));
        if (rankId == 0) printf("Computing Cholesky decomposition...\n");
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        double start = MPI_Wtime();
        bool success = matrix.factor();
        double elapsed = MPI_Wtime() - start, seconds;
        MPI_CHECK(MPI_Allreduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
        if (!success) {
            if (rankId == 0) printf("Cholesky decomposition failed\n");
            exitCode = 1;
        } else {
            if (rankId == 0) {
                printf("Computation time: %.3f ms\n", seconds * 1000.0);
                printf("Performance: %.3f GFLOPS\n", double(n) * n * n / (3.0e9 * seconds));
            }
            if (results) matrix.printResults();
            if (validation) {
                if (rankId == 0) printf("Validating result...\n");
                bool valid = matrix.validate(original);
                if (rankId == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                if (!valid) exitCode = 1;
            }
        }
        if (original) CUDA(cudaFree(original));
    }
    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
