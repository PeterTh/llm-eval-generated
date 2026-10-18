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

// All matrix tiles are column-major; only the lower triangle is distributed.
// A rank owns tile (i,j) iff (i % rows, j % cols) is its grid coordinate.
// Host-staged collectives also work on MPI installations without CUDA awareness.
constexpr int blockSize = 256;
constexpr size_t tileElements = size_t(blockSize) * blockSize;
int worldRank = 0;

static void check(int status, const char* operation, int line) {
    if (status != 0) {
        fprintf(stderr, "Rank %d: %s failed (%d), line %d\n",
                worldRank, operation, status, line);
        MPI_Abort(MPI_COMM_WORLD, status);
    }
}
#define CHECK(call) check(static_cast<int>(call), #call, __LINE__)

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        CHECK(cudaMalloc(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct PinnedBuffer {
    double* data = nullptr;
    explicit PinnedBuffer(size_t count) {
        CHECK(cudaMallocHost(reinterpret_cast<void**>(&data),
                             std::max(size_t(1), count) * sizeof(double)));
    }
    ~PinnedBuffer() { cudaFreeHost(data); }
};

// Chunking avoids the int-count limit of pre-MPI-4 implementations.
static void broadcast(double* data, size_t count, int root, MPI_Comm comm) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        CHECK(MPI_Bcast(data, chunk, MPI_DOUBLE, root, comm));
        data += chunk;
        count -= chunk;
    }
}

__global__ void clearUpper(double* tile) {
    int row = threadIdx.x;
    int col = blockIdx.x;
    if (row < col) tile[row + col * blockSize] = 0.0;
}

struct Tile { int row, col; size_t offset; };

struct DistributedMatrix {
    int n, nt, rank, size, rows, cols, row, col, nr, nc;
    MPI_Comm rowComm, colComm;
    std::vector<Tile> tiles;
    std::vector<size_t> lookup;
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    cudaStream_t stream;
    DeviceBuffer<double> values;

    static size_t tileCount(int nt, int rows, int cols, int row, int col) {
        size_t count = 0;
        for (int i = row; i < nt; i += rows)
            for (int j = col; j <= i; j += cols) ++count;
        return count;
    }

    DistributedMatrix(int dimension, int processRows, int processCols, int rank_, int size_)
        : n(dimension), nt((n - 1) / blockSize + 1), rank(rank_), size(size_),
          rows(processRows), cols(processCols), row(rank / cols), col(rank % cols),
          nr(row < nt ? (nt - 1 - row) / rows + 1 : 0),
          nc(col < nt ? (nt - 1 - col) / cols + 1 : 0),
          lookup(size_t(nr) * nc, std::numeric_limits<size_t>::max()),
          values(tileCount(nt, rows, cols, row, col) * tileElements) {
        CHECK(MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm));
        CHECK(MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm));
        for (int i = row; i < nt; i += rows)
            for (int j = col; j <= i; j += cols) {
                size_t offset = tiles.size() * tileElements;
                tiles.push_back({i, j, offset});
                lookup[size_t(i / rows) * nc + j / cols] = offset;
            }
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK(cublasCreate(&blas));
        CHECK(cusolverDnCreate(&solver));
        CHECK(cublasSetStream(blas, stream));
        CHECK(cusolverDnSetStream(solver, stream));
        // Keep all products in FP64, independent of toolkit math-mode defaults.
        CHECK(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
    }

    ~DistributedMatrix() {
        cusolverDnDestroy(solver);
        cublasDestroy(blas);
        cudaStreamDestroy(stream);
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }

    double* tile(int i, int j) {
        return values.data + lookup[size_t(i / rows) * nc + j / cols];
    }

    void sync() { CHECK(cudaStreamSynchronize(stream)); }

    void generate() {
        // Row seeds preserve exactly the original rand_r sequence. Each rank
        // retains only the B row strips needed by its process row and column.
        std::vector<unsigned int> seeds(n);
        unsigned int seed = 42;
        for (int i = 0; i < n; ++i) {
            seeds[i] = seed;
            for (int j = 0; j < n; ++j) (void)rand_r(&seed);
        }
        std::vector<double> r(size_t(nr) * blockSize * n, 0.0);
        std::vector<double> c(size_t(nc) * blockSize * n, 0.0);
        #pragma omp parallel for schedule(static)
        for (int strip = 0; strip < nr + nc; ++strip) {
            bool isRow = strip < nr;
            int local = isRow ? strip : strip - nr;
            int global = isRow ? row + local * rows : col + local * cols;
            double* out = (isRow ? r.data() : c.data()) + size_t(local) * blockSize * n;
            for (int i = 0; i < blockSize && global * blockSize + i < n; ++i) {
                unsigned int state = seeds[global * blockSize + i];
                for (int j = 0; j < n; ++j)
                    out[size_t(i) * n + j] = rand_r(&state) / double(RAND_MAX) - 0.5;
            }
        }
        DeviceBuffer<double> dr(r.size()), dc(c.size());
        CHECK(cudaMemcpyAsync(dr.data, r.data(), r.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(dc.data, c.data(), c.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
        std::vector<double> initial(tiles.size() * tileElements, 0.0);
        #pragma omp parallel for schedule(static)
        for (size_t t = 0; t < tiles.size(); ++t) {
            const Tile& tile = tiles[t];
            if (tile.row == tile.col)
                for (int i = 0; i < blockSize; ++i)
                    initial[tile.offset + i + size_t(i) * blockSize] =
                        tile.row * blockSize + i < n ? double(n) : 1.0;
        }
        CHECK(cudaMemcpyAsync(values.data, initial.data(), initial.size() * sizeof(double),
                              cudaMemcpyHostToDevice, stream));
        const double one = 1.0;
        for (const Tile& t : tiles)
            CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, blockSize, blockSize, n,
                              &one, dr.data + size_t(t.row / rows) * blockSize * n, n,
                              dc.data + size_t(t.col / cols) * blockSize * n, n,
                              &one, values.data + t.offset, blockSize));
        sync();
    }
};

struct Panels {
    DistributedMatrix& a;
    DeviceBuffer<double> rowDevice, colDevice;
    PinnedBuffer rowHost, colHost;
    explicit Panels(DistributedMatrix& matrix)
        : a(matrix), rowDevice(size_t(a.nr) * tileElements),
          colDevice(size_t(a.nc) * tileElements),
          rowHost(size_t(a.nr) * tileElements), colHost(size_t(a.nc) * tileElements) {}

    void exchange(int k) {
        // One packed row broadcast, followed by packed column broadcasts from
        // each process row. No rank ever stores an entire global panel.
        int firstRow = k <= a.row ? 0 : (k - a.row + a.rows - 1) / a.rows;
        firstRow = std::min(firstRow, a.nr);
        size_t rowStart = size_t(firstRow) * tileElements;
        size_t rowCount = size_t(a.nr - firstRow) * tileElements;
        if (a.col == k % a.cols) {
            for (int i = firstRow; i < a.nr; ++i)
                CHECK(cudaMemcpyAsync(rowDevice.data + size_t(i) * tileElements,
                                      a.tile(a.row + i * a.rows, k), tileElements * sizeof(double),
                                      cudaMemcpyDeviceToDevice, a.stream));
            CHECK(cudaMemcpyAsync(rowHost.data + rowStart, rowDevice.data + rowStart,
                                  rowCount * sizeof(double), cudaMemcpyDeviceToHost, a.stream));
            a.sync();
        }
        broadcast(rowHost.data + rowStart, rowCount, k % a.cols, a.rowComm);
        CHECK(cudaMemcpyAsync(rowDevice.data + rowStart, rowHost.data + rowStart,
                              rowCount * sizeof(double), cudaMemcpyHostToDevice, a.stream));

        // Pack by source process row; each destination process column needs
        // only the panel tiles corresponding to its own matrix columns.
        for (int source = 0; source < a.rows; ++source) {
            std::vector<int> indices;
            for (int j = 0; j < a.nc; ++j) {
                int global = a.col + j * a.cols;
                if (global >= k && global % a.rows == source) indices.push_back(j);
            }
            if (a.row == source) {
                #pragma omp parallel for schedule(static) if(indices.size() > 4)
                for (size_t t = 0; t < indices.size(); ++t) {
                    int global = a.col + indices[t] * a.cols;
                    memcpy(colHost.data + t * tileElements,
                           rowHost.data + size_t(global / a.rows) * tileElements,
                           tileElements * sizeof(double));
                }
            }
            broadcast(colHost.data, indices.size() * tileElements, source, a.colComm);
            for (size_t t = 0; t < indices.size(); ++t)
                CHECK(cudaMemcpyAsync(colDevice.data + size_t(indices[t]) * tileElements,
                                      colHost.data + t * tileElements, tileElements * sizeof(double),
                                      cudaMemcpyHostToDevice, a.stream));
            // The pinned staging buffer must not be reused before DMA ends.
            a.sync();
        }
    }
};

// Batched BLAS amortizes launch overhead over all local trailing tiles.
struct Updates {
    DistributedMatrix& a;
    std::vector<const double*> left, right;
    std::vector<double*> target;
    DeviceBuffer<const double*> dl, dr;
    DeviceBuffer<double*> dt;
    explicit Updates(DistributedMatrix& matrix)
        : a(matrix), dl(a.tiles.size()), dr(a.tiles.size()), dt(a.tiles.size()) {
        left.reserve(a.tiles.size()); right.reserve(a.tiles.size()); target.reserve(a.tiles.size());
    }
    void apply(Panels& panels, int first, double alpha, double* destination) {
        left.clear(); right.clear(); target.clear();
        for (const Tile& t : a.tiles) {
            if (t.col < first) continue;
            left.push_back(panels.rowDevice.data + size_t(t.row / a.rows) * tileElements);
            right.push_back(panels.colDevice.data + size_t(t.col / a.cols) * tileElements);
            target.push_back(destination + t.offset);
        }
        if (target.empty()) return;
        CHECK(cudaMemcpyAsync(dl.data, left.data(), left.size() * sizeof(double*), cudaMemcpyHostToDevice, a.stream));
        CHECK(cudaMemcpyAsync(dr.data, right.data(), right.size() * sizeof(double*), cudaMemcpyHostToDevice, a.stream));
        CHECK(cudaMemcpyAsync(dt.data, target.data(), target.size() * sizeof(double*), cudaMemcpyHostToDevice, a.stream));
        const double one = 1.0;
        CHECK(cublasDgemmBatched(a.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                                 blockSize, blockSize, blockSize, &alpha,
                                 dl.data, blockSize, dr.data, blockSize, &one,
                                 dt.data, blockSize, static_cast<int>(target.size())));
    }
};

struct FactorWorkspace {
    Panels panels;
    Updates updates;
    DeviceBuffer<double> diagonal;
    PinnedBuffer hostDiagonal;
    DeviceBuffer<int> info;
    int workSize = 0;
    static int workspaceSize(DistributedMatrix& a, double* diagonal) {
        int count = 0;
        CHECK(cusolverDnDpotrf_bufferSize(a.solver, CUBLAS_FILL_MODE_LOWER, blockSize,
                                         diagonal, blockSize, &count));
        return count;
    }
    DeviceBuffer<double> work;
    explicit FactorWorkspace(DistributedMatrix& a)
        : panels(a), updates(a), diagonal(tileElements), hostDiagonal(tileElements), info(1),
          workSize(workspaceSize(a, diagonal.data)), work(workSize) {
        // Warm up cuSOLVER (including lazy module loading) before measurement.
        std::fill(hostDiagonal.data, hostDiagonal.data + tileElements, 0.0);
        for (int i = 0; i < blockSize; ++i)
            hostDiagonal.data[i + size_t(i) * blockSize] = 1.0;
        CHECK(cudaMemcpyAsync(diagonal.data, hostDiagonal.data, tileElements * sizeof(double),
                              cudaMemcpyHostToDevice, a.stream));
        CHECK(cusolverDnDpotrf(a.solver, CUBLAS_FILL_MODE_LOWER, blockSize,
                               diagonal.data, blockSize, work.data, workSize, info.data));
        DeviceBuffer<double> warm(tileElements);
        CHECK(cudaMemsetAsync(warm.data, 0, tileElements * sizeof(double), a.stream));
        const double one = 1.0;
        CHECK(cublasDtrsm(a.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                          CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, blockSize, blockSize,
                          &one, diagonal.data, blockSize, warm.data, blockSize));
        clearUpper<<<blockSize, blockSize, 0, a.stream>>>(warm.data);
        CHECK(cudaGetLastError());
        a.sync();
    }
};

static bool choleskyDecomposition(DistributedMatrix& a, FactorWorkspace& workspace) {
    Panels& panels = workspace.panels;
    Updates& updates = workspace.updates;
    auto& diagonal = workspace.diagonal;
    auto& hostDiagonal = workspace.hostDiagonal;
    auto& info = workspace.info;
    auto& work = workspace.work;
    for (int k = 0; k < a.nt; ++k) {
        int status = 0;
        int owner = (k % a.rows) * a.cols + k % a.cols;
        if (a.rank == owner) {
            CHECK(cusolverDnDpotrf(a.solver, CUBLAS_FILL_MODE_LOWER, blockSize,
                                   a.tile(k, k), blockSize, work.data, workspace.workSize, info.data));
            CHECK(cudaMemcpyAsync(&status, info.data, sizeof(int), cudaMemcpyDeviceToHost, a.stream));
            clearUpper<<<blockSize, blockSize, 0, a.stream>>>(a.tile(k, k));
            CHECK(cudaGetLastError());
            CHECK(cudaMemcpyAsync(hostDiagonal.data, a.tile(k, k), tileElements * sizeof(double),
                                  cudaMemcpyDeviceToHost, a.stream));
            a.sync();
        }
        CHECK(MPI_Bcast(&status, 1, MPI_INT, owner, MPI_COMM_WORLD));
        if (status != 0) {
            if (a.rank == 0)
                printf("Error: Matrix is not positive definite at diagonal element %d\n",
                       k * blockSize + std::max(0, status - 1));
            return false;
        }
        if (a.col == k % a.cols) {
            broadcast(hostDiagonal.data, tileElements, k % a.rows, a.colComm);
            CHECK(cudaMemcpyAsync(diagonal.data, hostDiagonal.data, tileElements * sizeof(double),
                                  cudaMemcpyHostToDevice, a.stream));
            const double one = 1.0;
            for (int i = a.row; i < a.nt; i += a.rows)
                if (i > k)
                    CHECK(cublasDtrsm(a.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                      CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, blockSize, blockSize,
                                      &one, diagonal.data, blockSize, a.tile(i, k), blockSize));
        }
        panels.exchange(k);
        updates.apply(panels, k + 1, -1.0, a.values.data);
    }
    a.sync();
    return true;
}

static bool validateCholesky(DistributedMatrix& a, const std::vector<double>& original) {
    DeviceBuffer<double> reconstructed(a.tiles.size() * tileElements);
    CHECK(cudaMemsetAsync(reconstructed.data, 0, a.tiles.size() * tileElements * sizeof(double), a.stream));
    Panels panels(a);
    Updates updates(a);
    for (int k = 0; k < a.nt; ++k) {
        panels.exchange(k);
        updates.apply(panels, k, 1.0, reconstructed.data);
    }
    std::vector<double> result(original.size());
    CHECK(cudaMemcpyAsync(result.data(), reconstructed.data, result.size() * sizeof(double),
                          cudaMemcpyDeviceToHost, a.stream));
    a.sync();
    double maxError = 0.0, relError = 0.0;
    int invalid = 0;
    #pragma omp parallel for reduction(max:maxError,relError) reduction(|:invalid) schedule(static)
    for (size_t t = 0; t < a.tiles.size(); ++t) {
        const Tile& tile = a.tiles[t];
        for (int j = 0; j < blockSize && tile.col * blockSize + j < a.n; ++j)
            for (int i = 0; i < blockSize && tile.row * blockSize + i < a.n; ++i) {
                if (tile.row == tile.col && i < j) continue;
                size_t index = tile.offset + i + size_t(j) * blockSize;
                double error = fabs(result[index] - original[index]);
                invalid |= !std::isfinite(error);
                maxError = std::max(maxError, error);
                relError = std::max(relError, error / (fabs(original[index]) + 1e-10));
            }
    }
    double errors[2] = {maxError, relError}, global[2];
    int anyInvalid;
    CHECK(MPI_Allreduce(errors, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    CHECK(MPI_Allreduce(&invalid, &anyInvalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    if (a.rank == 0) {
        printf("Max absolute error: %.10e\n", global[0]);
        printf("Max relative error: %.10e\n", global[1]);
    }
    return !anyInvalid && global[1] <= 1e-6;
}

static void printFactor(DistributedMatrix& a) {
    std::vector<double> local(a.tiles.size() * tileElements);
    CHECK(cudaMemcpyAsync(local.data(), a.values.data, local.size() * sizeof(double),
                          cudaMemcpyDeviceToHost, a.stream));
    a.sync();
    std::vector<double> result(a.rank == 0 ? size_t(a.n) * a.n : 0, 0.0);
    std::vector<double> buffer(tileElements);
    for (int i = 0; i < a.nt; ++i)
        for (int j = 0; j <= i; ++j) {
            int owner = (i % a.rows) * a.cols + j % a.cols;
            const double* source = nullptr;
            if (a.rank == owner) {
                source = local.data() + a.lookup[size_t(i / a.rows) * a.nc + j / a.cols];
                if (owner != 0)
                    CHECK(MPI_Send(source, int(tileElements), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
            }
            if (a.rank == 0) {
                if (owner != 0) {
                    CHECK(MPI_Recv(buffer.data(), int(tileElements), MPI_DOUBLE, owner, 0,
                                   MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                    source = buffer.data();
                }
                for (int c = 0; c < blockSize && j * blockSize + c < a.n; ++c)
                    for (int r = 0; r < blockSize && i * blockSize + r < a.n; ++r)
                        if (i != j || r >= c)
                            result[size_t(i * blockSize + r) * a.n + j * blockSize + c] =
                                source[r + size_t(c) * blockSize];
            }
        }
    if (a.rank == 0) print_results(result, "CholeskyL");
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static int run(int argc, char** argv, int size) {
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            long value = strtol(argv[++i], &end, 10);
            if (errno || !end || *end || value <= 0 || value > INT_MAX - blockSize) {
                if (worldRank == 0) fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            n = static_cast<int>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    MPI_Comm node;
    CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &node));
    int localRank, devices = 0;
    CHECK(MPI_Comm_rank(node, &localRank));
    CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: a CUDA GPU is required\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CHECK(cudaSetDevice(localRank % devices));
    CHECK(MPI_Comm_free(&node));
    int dims[2] = {0, 0};
    CHECK(MPI_Dims_create(size, 2, dims));
    DistributedMatrix a(n, dims[0], dims[1], worldRank, size);
    if (worldRank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %d x %d\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI grid: %d x %d; CUDA tile: %d; OpenMP threads/rank: %d\n",
               dims[0], dims[1], blockSize, omp_get_max_threads());
        printf("Generating positive definite matrix...\n");
    }
    a.generate();
    std::vector<double> original;
    if (validate) {
        original.resize(a.tiles.size() * tileElements);
        CHECK(cudaMemcpy(original.data(), a.values.data, original.size() * sizeof(double), cudaMemcpyDeviceToHost));
    }
    FactorWorkspace workspace(a);
    if (worldRank == 0) printf("Computing Cholesky decomposition...\n");
    CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    bool success = choleskyDecomposition(a, workspace);
    double elapsed = MPI_Wtime() - start, duration;
    CHECK(MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!success) {
        if (worldRank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (worldRank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Performance: %.3f GFLOPS\n", double(n) * n * n / (3.0 * duration * 1e9));
    }
    if (printResults) printFactor(a);
    if (validate) {
        if (worldRank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(a, original);
        if (worldRank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    int provided = 0;
    CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int result = 1;
    try {
        int size;
        CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
        result = run(argc, argv, size);
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", worldRank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CHECK(MPI_Finalize());
    return result;
}
