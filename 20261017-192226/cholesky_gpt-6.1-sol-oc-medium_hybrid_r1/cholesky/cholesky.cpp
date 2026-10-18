#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
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
int rankId = 0;

void check(bool ok, const char* operation) {
    if (!ok) {
        fprintf(stderr, "Rank %d: %s failed\n", rankId, operation);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) check((call) == cudaSuccess, #call)
#define BLAS(call) check((call) == CUBLAS_STATUS_SUCCESS, #call)
#define SOLVER(call) check((call) == CUSOLVER_STATUS_SUCCESS, #call)
#define MPI_CHECK(call) check((call) == MPI_SUCCESS, #call)

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        CUDA(cudaMalloc(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct PinnedBuffer {
    double* data = nullptr;
    explicit PinnedBuffer(size_t count) {
        CUDA(cudaMallocHost(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(double)));
    }
    ~PinnedBuffer() { cudaFreeHost(data); }
};

__global__ void addDiagonal(double* a, int first, int n) {
    int i = threadIdx.x;
    if (i < tileSize) a[size_t(i) * tileSize + i] += first + i < n ? double(n) : 1.0;
}

struct Tile { int i, j; };

// Tiles are column-major for CUDA BLAS. A nearly square process grid gives
// O(n^2/P) matrix storage and O(n^2/sqrt(P)) panel traffic per rank.
struct DistributedMatrix {
    int n, nt, rows, cols, row, col, nr, nc;
    MPI_Comm rowComm, colComm;
    std::vector<Tile> tiles;
    std::vector<int> index;
    cudaStream_t stream;
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    double* matrix = nullptr;

    DistributedMatrix(int size, int processes) : n(size), nt((n - 1) / tileSize + 1) {
        int dims[2] = {0, 0};
        MPI_CHECK(MPI_Dims_create(processes, 2, dims));
        rows = dims[0]; cols = dims[1];
        row = rankId / cols; col = rankId % cols;
        nr = row < nt ? (nt - 1 - row) / rows + 1 : 0;
        nc = col < nt ? (nt - 1 - col) / cols + 1 : 0;
        MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm));
        MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm));
        index.assign(size_t(nr) * nc, -1);
        for (int i = row; i < nt; i += rows)
            for (int j = col; j <= i; j += cols) {
                index[size_t(i / rows) * nc + j / cols] = int(tiles.size());
                tiles.push_back({i, j});
            }
        check(tiles.size() <= size_t(INT_MAX) / tileElements &&
              size_t(std::max(nr, nc)) <= size_t(INT_MAX) / tileElements,
              "MPI tile counts fit in an int");
        CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        BLAS(cublasCreate(&blas));
        BLAS(cublasSetStream(blas, stream));
        SOLVER(cusolverDnCreate(&solver));
        SOLVER(cusolverDnSetStream(solver, stream));
        CUDA(cudaMalloc(reinterpret_cast<void**>(&matrix),
                        std::max(size_t(1), tiles.size() * tileElements) * sizeof(double)));
    }

    ~DistributedMatrix() {
        cudaFree(matrix);
        cusolverDnDestroy(solver);
        cublasDestroy(blas);
        cudaStreamDestroy(stream);
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }

    double* tile(int i, int j) {
        return matrix + size_t(index[size_t(i / rows) * nc + j / cols]) * tileElements;
    }

    // Preserve the original rand_r sequence exactly, including its seed of 42.
    // Only O(n) RNG states are broadcast; each process generates the B rows
    // required by its own tile rows/columns in parallel.
    void generate() {
        std::vector<unsigned int> seeds(n);
        if (rankId == 0) {
            unsigned int seed = 42;
            for (int i = 0; i < n; ++i) {
                seeds[i] = seed;
                for (int j = 0; j < n; ++j) (void)rand_r(&seed);
            }
        }
        MPI_CHECK(MPI_Bcast(seeds.data(), n, MPI_UNSIGNED, 0, MPI_COMM_WORLD));
        auto makeRows = [&](int count, int coordinate, int stride, double* destination) {
            #pragma omp parallel for schedule(static)
            for (int r = 0; r < count * tileSize; ++r) {
                int global = (r / tileSize * stride + coordinate) * tileSize + r % tileSize;
                double* out = destination + size_t(r) * n;
                if (global < n) {
                    unsigned int seed = seeds[global];
                    for (int k = 0; k < n; ++k)
                        out[k] = rand_r(&seed) / double(RAND_MAX) - 0.5;
                } else std::fill(out, out + n, 0.0);
            }
        };
        DeviceBuffer<double> br(size_t(nr) * tileSize * n);
        DeviceBuffer<double> bc(size_t(nc) * tileSize * n);
        // Reuse host storage between the two sets of B rows.
        PinnedBuffer host(size_t(std::max(nr, nc)) * tileSize * n);
        makeRows(nr, row, rows, host.data);
        CUDA(cudaMemcpyAsync(br.data, host.data, size_t(nr) * tileSize * n * sizeof(double),
                             cudaMemcpyHostToDevice, stream));
        CUDA(cudaStreamSynchronize(stream));
        makeRows(nc, col, cols, host.data);
        CUDA(cudaMemcpyAsync(bc.data, host.data, size_t(nc) * tileSize * n * sizeof(double),
                             cudaMemcpyHostToDevice, stream));
        const double one = 1.0, zero = 0.0;
        for (const Tile& t : tiles) {
            BLAS(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, tileSize, tileSize, n,
                             &one, br.data + size_t(t.i / rows) * tileSize * n, n,
                             bc.data + size_t(t.j / cols) * tileSize * n, n,
                             &zero, tile(t.i, t.j), tileSize));
            if (t.i == t.j) addDiagonal<<<1, tileSize, 0, stream>>>(tile(t.i, t.j), t.i * tileSize, n);
        }
        CUDA(cudaGetLastError());
        CUDA(cudaStreamSynchronize(stream));
    }

    bool factor() {
        DeviceBuffer<double> diagonal(tileElements);
        DeviceBuffer<double> rowPanel(size_t(nr) * tileElements);
        DeviceBuffer<double> colPanel(size_t(nc) * tileElements);
        PinnedBuffer hdiag(tileElements), hrow(size_t(nr) * tileElements);
        PinnedBuffer send(size_t(nc) * tileElements), hcol(size_t(nc) * tileElements);
        DeviceBuffer<int> info(1);
        int lwork = 0;
        SOLVER(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, tileSize,
                                          diagonal.data, tileSize, &lwork));
        DeviceBuffer<double> work(lwork);
        std::vector<int> counts(rows, 0), displacements(rows, 0), columnSlot(nt, -1);
        std::vector<int> offsets(rows);
        std::vector<const double*> left(tiles.size()), right(tiles.size());
        std::vector<double*> output(tiles.size());
        DeviceBuffer<const double*> dleft(tiles.size()), dright(tiles.size());
        DeviceBuffer<double*> doutput(tiles.size());
        const double one = 1.0, minusOne = -1.0;
        // Pinned host staging works with ordinary MPI, without requiring
        // CUDA-aware MPI or assuming that asynchronous GPU work has completed.
        for (int k = 0; k < nt; ++k) {
            int owner = (k % rows) * cols + k % cols;
            int status = 0;
            if (rankId == owner) {
                double* a = tile(k, k);
                SOLVER(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, tileSize,
                                       a, tileSize, work.data, lwork, info.data));
                CUDA(cudaMemcpyAsync(&status, info.data, sizeof(int), cudaMemcpyDeviceToHost, stream));
                CUDA(cudaMemcpyAsync(hdiag.data, a, tileElements * sizeof(double),
                                     cudaMemcpyDeviceToHost, stream));
                CUDA(cudaStreamSynchronize(stream));
            }
            MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, owner, MPI_COMM_WORLD));
            if (status != 0) {
                if (rankId == 0) printf("Error: Matrix is not positive definite at diagonal element %d\n",
                                        k * tileSize + std::abs(status) - 1);
                return false;
            }
            if (col == k % cols) {
                // Only the process column doing triangular solves needs L_kk.
                MPI_CHECK(MPI_Bcast(hdiag.data, int(tileElements), MPI_DOUBLE, k % rows, colComm));
                CUDA(cudaMemcpyAsync(diagonal.data, hdiag.data, tileElements * sizeof(double),
                                     cudaMemcpyHostToDevice, stream));
                for (int i = row; i < nt; i += rows) if (i > k) {
                    double* a = tile(i, k);
                    BLAS(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, tileSize, tileSize,
                                     &one, diagonal.data, tileSize, a, tileSize));
                    CUDA(cudaMemcpyAsync(hrow.data + size_t(i / rows) * tileElements, a,
                                         tileElements * sizeof(double), cudaMemcpyDeviceToHost, stream));
                }
            }
            CUDA(cudaStreamSynchronize(stream));
            // Broadcast L_ik across process rows, then redistribute L_jk down
            // process columns. Each panel tile is communicated in bulk.
            int firstRow = k < row ? 0 : std::min(nr, (k - row) / rows + 1);
            MPI_CHECK(MPI_Bcast(hrow.data + size_t(firstRow) * tileElements,
                                int(size_t(nr - firstRow) * tileElements), MPI_DOUBLE,
                                k % cols, rowComm));
            std::fill(counts.begin(), counts.end(), 0);
            for (int j = col; j < nt; j += cols) if (j > k) counts[j % rows] += int(tileElements);
            for (int r = 1; r < rows; ++r) displacements[r] = displacements[r - 1] + counts[r - 1];
            offsets = displacements;
            for (int j = col; j < nt; j += cols) if (j > k) {
                columnSlot[j] = offsets[j % rows] / int(tileElements);
                offsets[j % rows] += int(tileElements);
            }
            int packed = 0;
            for (int j = col; j < nt; j += cols) if (j > k && j % rows == row) {
                std::memcpy(send.data + size_t(packed) * tileElements,
                            hrow.data + size_t(j / rows) * tileElements,
                            tileElements * sizeof(double));
                ++packed;
            }
            MPI_CHECK(MPI_Allgatherv(send.data, counts[row], MPI_DOUBLE, hcol.data,
                                     counts.data(), displacements.data(), MPI_DOUBLE, colComm));
            CUDA(cudaMemcpyAsync(rowPanel.data + size_t(firstRow) * tileElements,
                                 hrow.data + size_t(firstRow) * tileElements,
                                 size_t(nr - firstRow) * tileElements * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
            size_t columnElements = size_t(displacements.back()) + counts.back();
            CUDA(cudaMemcpyAsync(colPanel.data, hcol.data, columnElements * sizeof(double),
                                 cudaMemcpyHostToDevice, stream));
            int batch = 0;
            for (size_t t = 0; t < tiles.size(); ++t) if (tiles[t].j > k) {
                left[batch] = rowPanel.data + size_t(tiles[t].i / rows) * tileElements;
                right[batch] = colPanel.data + size_t(columnSlot[tiles[t].j]) * tileElements;
                output[batch] = matrix + t * tileElements;
                ++batch;
            }
            if (batch) {
                CUDA(cudaMemcpyAsync(dleft.data, left.data(), size_t(batch) * sizeof(double*), cudaMemcpyHostToDevice, stream));
                CUDA(cudaMemcpyAsync(dright.data, right.data(), size_t(batch) * sizeof(double*), cudaMemcpyHostToDevice, stream));
                CUDA(cudaMemcpyAsync(doutput.data, output.data(), size_t(batch) * sizeof(double*), cudaMemcpyHostToDevice, stream));
                BLAS(cublasDgemmBatched(blas, CUBLAS_OP_N, CUBLAS_OP_T, tileSize, tileSize, tileSize,
                                        &minusOne, dleft.data, tileSize, dright.data, tileSize,
                                        &one, doutput.data, tileSize, batch));
            }
            // This also protects the reusable pinned communication buffers.
            CUDA(cudaStreamSynchronize(stream));
        }
        return true;
    }

    // Only the reporting rank materializes a dense matrix, and only when
    // requested by -v or -r. Point-to-point chunks avoid MPI_Gatherv int offsets.
    std::vector<double> gather(bool symmetric) {
        PinnedBuffer local(tiles.size() * tileElements);
        CUDA(cudaMemcpyAsync(local.data, matrix, tiles.size() * tileElements * sizeof(double),
                             cudaMemcpyDeviceToHost, stream));
        CUDA(cudaStreamSynchronize(stream));
        std::vector<double> result;
        if (rankId == 0) result.assign(size_t(n) * n, 0.0);
        int processes = rows * cols;
        for (int source = 0; source < processes; ++source) {
            if (rankId != 0 && rankId != source) continue;
            std::vector<Tile> sourceTiles;
            for (int i = source / cols; i < nt; i += rows)
                for (int j = source % cols; j <= i; j += cols) sourceTiles.push_back({i, j});
            size_t count = sourceTiles.size() * tileElements;
            if (rankId == source && source != 0) {
                MPI_CHECK(MPI_Send(local.data, int(count), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
            } else if (rankId == 0) {
                std::vector<double> receive(source == 0 ? 0 : count);
                const double* data = local.data;
                if (source != 0) {
                    MPI_CHECK(MPI_Recv(receive.data(), int(count), MPI_DOUBLE, source, 0,
                                       MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                    data = receive.data();
                }
                #pragma omp parallel for schedule(static)
                for (size_t t = 0; t < sourceTiles.size(); ++t) {
                    int i0 = sourceTiles[t].i * tileSize, j0 = sourceTiles[t].j * tileSize;
                    for (int j = 0; j < tileSize && j0 + j < n; ++j)
                        for (int i = 0; i < tileSize && i0 + i < n; ++i) if (i0 + i >= j0 + j) {
                            double value = data[t * tileElements + size_t(j) * tileSize + i];
                            result[size_t(i0 + i) * n + j0 + j] = value;
                            if (symmetric) result[size_t(j0 + j) * n + i0 + i] = value;
                        }
                }
            }
        }
        return result;
    }
};

bool validateCholesky(const std::vector<double>& l, const std::vector<double>& original, int n) {
    double maxError = 0.0, relError = 0.0;
    int finite = 1;
    #pragma omp parallel for schedule(dynamic) reduction(max:maxError,relError) reduction(&:finite)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k <= j; ++k) sum += l[size_t(i) * n + k] * l[size_t(j) * n + k];
            double expected = original[size_t(i) * n + j];
            double error = std::abs(sum - expected);
            finite &= std::isfinite(sum) && std::isfinite(error);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::abs(expected) + 1e-10));
        }
    }
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
    return finite && relError <= 1e-6;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int processes) {
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            long long parsed = strtoll(argv[++i], &end, 10);
            if (errno || !*argv[i] || *end || parsed <= 0 || parsed > INT_MAX - tileSize ||
                static_cast<unsigned long long>(parsed) > std::numeric_limits<size_t>::max() / sizeof(double) / parsed) {
                if (rankId == 0) fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            n = int(parsed);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rankId == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rankId == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    MPI_Comm local;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rankId, MPI_INFO_NULL, &local));
    int localRank = 0, devices = 0;
    MPI_CHECK(MPI_Comm_rank(local, &localRank));
    CUDA(cudaGetDeviceCount(&devices));
    check(devices > 0, "at least one CUDA device is required");
    CUDA(cudaSetDevice(localRank % devices));
    MPI_CHECK(MPI_Comm_free(&local));
    DistributedMatrix a(n, processes);
    if (rankId == 0) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n",
               n, n, validate ? "enabled" : "disabled");
        printf("MPI grid: %d x %d; OpenMP threads/rank: %d; CUDA tile size: %d\n",
               a.rows, a.cols, omp_get_max_threads(), tileSize);
        printf("Generating positive definite matrix...\n");
    }
    a.generate();
    std::vector<double> original;
    if (validate) original = a.gather(true);
    if (rankId == 0) printf("Computing Cholesky decomposition...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    bool success = a.factor();
    double elapsed = MPI_Wtime() - start, seconds = 0.0;
    MPI_CHECK(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!success) {
        if (rankId == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rankId == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Performance: %.3f GFLOPS\n", double(n) * n * n / (3.0 * seconds * 1e9));
    }
    std::vector<double> result;
    if (printResults || validate) result = a.gather(false);
    int exitCode = 0;
    if (rankId == 0) {
        if (printResults) print_results(result, "CholeskyL");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(result, original, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    }
    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return exitCode;
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rankId));
    check(provided >= MPI_THREAD_FUNNELED, "MPI_THREAD_FUNNELED support");
    int processes = 1;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &processes));
    int result = 1;
    try { result = run(argc, argv, processes); }
    catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rankId, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
