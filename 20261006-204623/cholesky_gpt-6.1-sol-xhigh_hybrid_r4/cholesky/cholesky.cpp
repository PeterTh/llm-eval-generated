#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

namespace {

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
void blasCheck(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " + std::to_string(status));
}
void solverCheck(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuSOLVER status " + std::to_string(status));
}
void mpiCheck(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        throw std::runtime_error(std::string(operation) + ": " + std::string(message, length));
    }
}
#define CUDA(call) cudaCheck((call), #call)
#define BLAS(call) blasCheck((call), #call)
#define SOLVER(call) solverCheck((call), #call)
#define MPI(call) mpiCheck((call), #call)

template<class T, bool pinned = false> struct Buffer {
    T* data = nullptr;
    explicit Buffer(size_t count) {
        count = std::max(size_t{1}, count);
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::runtime_error("Matrix allocation size overflow");
        if constexpr (pinned)
            CUDA(cudaMallocHost(reinterpret_cast<void**>(&data), count * sizeof(T)));
        else
            CUDA(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~Buffer() {
        if constexpr (pinned) cudaFreeHost(data);
        else cudaFree(data);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

// MPI rank order is (process row, process column). Each GPU keeps its blocks
// in a column-major local matrix, so a whole panel uses one TRSM and each
// trailing block column uses a tall GEMM rather than many tiny tile kernels.
struct Grid {
    int rank, size, pr, pc, row, col, n, b, blocks, rows, cols, lda;
    MPI_Comm rowComm = MPI_COMM_NULL, colComm = MPI_COMM_NULL;
    Grid(int order, int worldRank, int worldSize) : rank(worldRank), size(worldSize), n(order) {
        int dims[2] = {0, 0};
        MPI(MPI_Dims_create(size, 2, dims));
        pr = dims[0]; pc = dims[1]; row = rank / pc; col = rank % pc;
        b = std::min(256, std::max(1, n / pr + (n % pr != 0)));
        blocks = n / b + (n % b != 0);
        rows = localBlocks(row, pr) * b;
        cols = localBlocks(col, pc) * b;
        lda = std::max(1, rows);
        MPI(MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm));
        MPI(MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm));
    }
    ~Grid() { MPI_Comm_free(&rowComm); MPI_Comm_free(&colComm); }
    int localBlocks(int coordinate, int stride) const {
        return blocks > coordinate ? 1 + (blocks - 1 - coordinate) / stride : 0;
    }
    // First local element whose global block index is at least k.
    int first(int k, int coordinate, int stride) const {
        return (k <= coordinate ? 0 : 1 + (k - 1 - coordinate) / stride) * b;
    }
    int global(int local, int coordinate, int stride) const {
        return (local / b * stride + coordinate) * b + local % b;
    }
    size_t elements() const { return size_t(rows) * cols; }
    double* tile(double* a, int i, int j) const {
        return a + (i / pr) * b + size_t((j / pc) * b) * lda;
    }
};

struct Accelerator {
    cudaStream_t panel, update;
    cublasHandle_t panelBlas, updateBlas;
    cusolverDnHandle_t solver;
    cudaEvent_t nextReady[2], updateDone[2], dataReady[2];
    Accelerator() {
        int low, high;
        CUDA(cudaDeviceGetStreamPriorityRange(&low, &high));
        CUDA(cudaStreamCreateWithPriority(&panel, cudaStreamNonBlocking, high));
        CUDA(cudaStreamCreateWithPriority(&update, cudaStreamNonBlocking, low));
        BLAS(cublasCreate(&panelBlas)); BLAS(cublasCreate(&updateBlas));
        BLAS(cublasSetStream(panelBlas, panel)); BLAS(cublasSetStream(updateBlas, update));
        // Keep FP64 throughout, including generation and validation.
        BLAS(cublasSetMathMode(panelBlas, CUBLAS_DEFAULT_MATH));
        BLAS(cublasSetMathMode(updateBlas, CUBLAS_DEFAULT_MATH));
        SOLVER(cusolverDnCreate(&solver)); SOLVER(cusolverDnSetStream(solver, panel));
        for (int s = 0; s < 2; ++s) {
            CUDA(cudaEventCreateWithFlags(&nextReady[s], cudaEventDisableTiming));
            CUDA(cudaEventCreateWithFlags(&updateDone[s], cudaEventDisableTiming));
            CUDA(cudaEventCreateWithFlags(&dataReady[s], cudaEventDisableTiming));
        }
    }
    ~Accelerator() {
        for (int s = 0; s < 2; ++s) {
            cudaEventDestroy(nextReady[s]); cudaEventDestroy(updateDone[s]);
            cudaEventDestroy(dataReady[s]);
        }
        cusolverDnDestroy(solver); cublasDestroy(panelBlas); cublasDestroy(updateBlas);
        cudaStreamDestroy(panel); cudaStreamDestroy(update);
    }
};

__global__ void matrixBoundary(double* a, size_t count, int rows, int b,
                               int pr, int pc, int row, int col, int n, bool clearUpper) {
    for (size_t p = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         p < count; p += size_t(blockDim.x) * gridDim.x) {
        int lr = p % rows, lc = p / rows;
        int i = (lr / b * pr + row) * b + lr % b;
        int j = (lc / b * pc + col) * b + lc % b;
        if (clearUpper) {
            if (i < j) a[p] = 0.0;
        } else if (i == j) {
            // Embed an incomplete last tile in an identity block. It has no
            // coupling to the original n-by-n matrix and is excluded from output.
            a[p] += i < n ? double(n) : 1.0;
        }
    }
}

int kernelBlocks(size_t count) {
    return int(std::min(size_t{65535}, std::max(size_t{1}, (count + 255) / 256)));
}

void boundary(const Grid& g, Accelerator& gpu, double* a, bool clearUpper) {
    matrixBoundary<<<kernelBlocks(g.elements()), 256, 0, gpu.panel>>>(
        a, g.elements(), g.rows, g.b, g.pr, g.pc, g.row, g.col, g.n, clearUpper);
    CUDA(cudaGetLastError());
}

// glibc rand_r advances a 32-bit LCG three times per sample. Jumping that
// affine recurrence preserves the original seed-42 row-major random stream
// while allowing independent OpenMP generation of every distributed row.
unsigned int randomSeed(uint64_t samples) {
    uint32_t a = 1103515245u, c = 12345u, multiplier = 1u, increment = 0u;
    for (uint64_t steps = samples * 3; steps; steps >>= 1) {
        if (steps & 1) { increment = a * increment + c; multiplier = a * multiplier; }
        c = a * c + c; a = a * a;
    }
    return multiplier * 42u + increment;
}

std::vector<unsigned int> rowSeeds(const Grid& g, int count, int coordinate, int stride) {
    std::vector<unsigned int> seeds(count);
#if defined(__GLIBC__)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i)
        seeds[i] = randomSeed(uint64_t(g.global(i, coordinate, stride)) * g.n);
#else
    // rand_r is implementation-defined outside glibc. Retain its exact stream
    // there too, by scanning to the starts of the locally owned rows.
    unsigned int seed = 42;
    int local = 0;
    for (int row = 0; row < g.n; ++row) {
        if (local < count && g.global(local, coordinate, stride) == row)
            seeds[local++] = seed;
        for (int j = 0; j < g.n; ++j) (void)rand_r(&seed);
    }
#endif
    return seeds;
}

void generatePositiveDefiniteMatrix(const Grid& g, Accelerator& gpu, double* a) {
    const int width = std::min(1024, std::max(1, g.n));
    Buffer<double, true> hostRows(size_t(g.rows) * width), hostCols(size_t(g.cols) * width);
    Buffer<double> deviceRows(size_t(g.rows) * width), deviceCols(size_t(g.cols) * width);
    auto rowState = rowSeeds(g, g.rows, g.row, g.pr);
    auto colState = rowSeeds(g, g.cols, g.col, g.pc);
    const double one = 1.0, zero = 0.0;
    CUDA(cudaMemsetAsync(a, 0, g.elements() * sizeof(double), gpu.panel));
    // Stream B in bounded-width panels: neither the full random matrix nor
    // the full global A is replicated on any node.
    for (int start = 0; start < g.n;) {
        int k = std::min(width, g.n - start);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < g.rows; ++i) {
            unsigned int seed = rowState[i];
            bool valid = g.global(i, g.row, g.pr) < g.n;
            for (int j = 0; j < k; ++j)
                hostRows.data[size_t(i) * width + j] = valid ? rand_r(&seed) / double(RAND_MAX) - 0.5 : 0.0;
            rowState[i] = seed;
        }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < g.cols; ++i) {
            unsigned int seed = colState[i];
            bool valid = g.global(i, g.col, g.pc) < g.n;
            for (int j = 0; j < k; ++j)
                hostCols.data[size_t(i) * width + j] = valid ? rand_r(&seed) / double(RAND_MAX) - 0.5 : 0.0;
            colState[i] = seed;
        }
        CUDA(cudaMemcpyAsync(deviceRows.data, hostRows.data, size_t(g.rows) * width * sizeof(double),
                             cudaMemcpyHostToDevice, gpu.panel));
        CUDA(cudaMemcpyAsync(deviceCols.data, hostCols.data, size_t(g.cols) * width * sizeof(double),
                             cudaMemcpyHostToDevice, gpu.panel));
        if (g.rows && g.cols)
            BLAS(cublasDgemm(gpu.panelBlas, CUBLAS_OP_T, CUBLAS_OP_N, g.rows, g.cols, k,
                &one, deviceRows.data, width, deviceCols.data, width, start ? &one : &zero, a, g.lda));
        // Pinned input cannot be reused until its asynchronous DMA has finished.
        CUDA(cudaStreamSynchronize(gpu.panel));
        start += k;
    }
    boundary(g, gpu, a, false);
    CUDA(cudaStreamSynchronize(gpu.panel));
}

void broadcast(double* data, size_t count, int root, MPI_Comm comm) {
    for (size_t offset = 0; offset < count;) {
        int chunk = int(std::min(size_t(INT_MAX), count - offset));
        MPI(MPI_Bcast(data + offset, chunk, MPI_DOUBLE, root, comm));
        offset += chunk;
    }
}

struct Panel {
    Buffer<double, true> leftHost, rightHost, packed;
    Buffer<double> left, right;
    const double* deviceLeft = nullptr;
    const double* deviceRight = nullptr;
    int leftLeading = 0, rightLeading = 0;
    Panel(const Grid& g) : leftHost(size_t(g.rows) * g.b), rightHost(size_t(g.cols) * g.b),
        packed(size_t(g.cols) * g.b), left(size_t(g.rows) * g.b), right(size_t(g.cols) * g.b) {}
};

__global__ void identityMatrix(double* a, int n) {
    for (size_t p = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         p < size_t(n) * n; p += size_t(blockDim.x) * gridDim.x)
        a[p] = p % n == p / n ? 1.0 : 0.0;
}

struct FactorizationWorkspace {
    Panel even, odd;
    Buffer<double> diagonal;
    Buffer<double, true> diagonalHost;
    Buffer<int> info;
    int workspaceSize;
    Buffer<double> workspace;
    static int query(Accelerator& gpu, double* diagonal, int b) {
        int count = 0;
        SOLVER(cusolverDnDpotrf_bufferSize(gpu.solver, CUBLAS_FILL_MODE_LOWER,
                                         b, diagonal, b, &count));
        return count;
    }
    FactorizationWorkspace(const Grid& g, Accelerator& gpu) : even(g), odd(g),
        diagonal(size_t(g.b) * g.b), diagonalHost(size_t(g.b) * g.b), info(1),
        workspaceSize(query(gpu, diagonal.data, g.b)), workspace(workspaceSize) {
        // Allocate communication/work buffers and initialize library kernels
        // before timing, as with the original benchmark's matrix allocation.
        // Warm-up uses an identity matrix, never the benchmark's input.
        Buffer<double> scratch(size_t(g.b) * g.b);
        identityMatrix<<<kernelBlocks(size_t(g.b) * g.b), 256, 0, gpu.panel>>>(diagonal.data, g.b);
        CUDA(cudaGetLastError());
        CUDA(cudaMemsetAsync(scratch.data, 0, size_t(g.b) * g.b * sizeof(double), gpu.panel));
        SOLVER(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_LOWER, g.b,
            diagonal.data, g.b, workspace.data, workspaceSize, info.data));
        const double one = 1.0, zero = 0.0;
        BLAS(cublasDtrsm(gpu.panelBlas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, g.b, g.b, &one,
            diagonal.data, g.b, scratch.data, g.b));
        BLAS(cublasDgemm(gpu.panelBlas, CUBLAS_OP_N, CUBLAS_OP_T, g.b, g.b, g.b,
            &one, diagonal.data, g.b, diagonal.data, g.b, &zero, scratch.data, g.b));
        BLAS(cublasDsyrk(gpu.panelBlas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, g.b, g.b,
            &one, diagonal.data, g.b, &zero, scratch.data, g.b));
        CUDA(cudaStreamSynchronize(gpu.panel));
    }
};

// First broadcast L(:,k) across process rows. Its tiles are then redistributed
// across process columns, grouped by source row, giving the two operands of
// each rank's local trailing update. Pinned staging also works with MPI builds
// that do not support device pointers.
void exchangePanel(const Grid& g, Accelerator& gpu, double* a, Panel& p,
                   int k, int firstBlock, int slot, bool reuse) {
    int rowFirst = g.first(firstBlock, g.row, g.pr);
    int colFirst = g.first(firstBlock, g.col, g.pc);
    int m = g.rows - rowFirst, nc = g.cols - colFirst;
    if (g.size == 1) {
        // With one rank both operands already reside on its GPU. The MPI
        // factorization protocol is unchanged, but no panel staging is needed.
        p.deviceLeft = g.tile(a, 0, k) + rowFirst;
        p.deviceRight = p.deviceLeft;
        p.leftLeading = p.rightLeading = g.lda;
        CUDA(cudaEventRecord(gpu.dataReady[slot], gpu.panel));
        CUDA(cudaStreamWaitEvent(gpu.update, gpu.dataReady[slot], 0));
        return;
    }
    if (reuse) CUDA(cudaStreamWaitEvent(gpu.panel, gpu.updateDone[slot], 0));
    if (g.col == k % g.pc && m)
        CUDA(cudaMemcpy2DAsync(p.leftHost.data, size_t(m) * sizeof(double),
            g.tile(a, 0, k) + rowFirst, size_t(g.lda) * sizeof(double),
            size_t(m) * sizeof(double), g.b, cudaMemcpyDeviceToHost, gpu.panel));
    CUDA(cudaStreamSynchronize(gpu.panel));
    broadcast(p.leftHost.data, size_t(m) * g.b, k % g.pc, g.rowComm);

    // The panel owner can use its already resident factor directly. Other
    // process columns upload the panel received through MPI.
    bool resident = g.col == k % g.pc && m;
    p.deviceLeft = resident ? g.tile(a, 0, k) + rowFirst : p.left.data;
    p.leftLeading = resident ? g.lda : m;
    if (!resident)
        CUDA(cudaMemcpyAsync(p.left.data, p.leftHost.data, size_t(m) * g.b * sizeof(double),
                             cudaMemcpyHostToDevice, gpu.panel));

    if (g.pr == g.pc) {
        // In a square process grid all right-hand tiles for a process column
        // come from its diagonal rank. Its row panel is already in the exact
        // packed layout needed by that column; no CPU repacking is necessary.
        double* host = g.row == g.col ? p.leftHost.data : p.rightHost.data;
        broadcast(host, size_t(nc) * g.b, g.col, g.colComm);
        if (g.row == g.col) {
            p.deviceRight = p.deviceLeft;
            p.rightLeading = p.leftLeading;
        } else {
            p.deviceRight = p.right.data;
            p.rightLeading = nc;
            CUDA(cudaMemcpyAsync(p.right.data, host, size_t(nc) * g.b * sizeof(double),
                                 cudaMemcpyHostToDevice, gpu.panel));
        }
        CUDA(cudaEventRecord(gpu.dataReady[slot], gpu.panel));
        CUDA(cudaStreamWaitEvent(gpu.update, gpu.dataReady[slot], 0));
        return;
    }

    int tiles = nc / g.b;
    std::vector<int> counts(g.pr, 0), offsets(g.pr + 1, 0), locations(tiles);
    for (int q = 0; q < tiles; ++q) {
        int j = (colFirst / g.b + q) * g.pc + g.col;
        locations[q] = counts[j % g.pr]++;
    }
    for (int r = 0; r < g.pr; ++r) offsets[r + 1] = offsets[r] + counts[r];
    #pragma omp parallel for schedule(static)
    for (int q = 0; q < tiles; ++q) {
        int j = (colFirst / g.b + q) * g.pc + g.col;
        int source = j % g.pr;
        locations[q] += offsets[source];
        if (source == g.row) {
            int rowOffset = (j / g.pr) * g.b - rowFirst;
            double* dest = p.packed.data + size_t(locations[q]) * g.b * g.b;
            for (int c = 0; c < g.b; ++c)
                std::memcpy(dest + c * g.b, p.leftHost.data + size_t(c) * m + rowOffset,
                            size_t(g.b) * sizeof(double));
        }
    }
    std::vector<MPI_Request> requests;
    for (int r = 0; r < g.pr; ++r) {
        size_t count = size_t(counts[r]) * g.b * g.b;
        double* data = p.packed.data + size_t(offsets[r]) * g.b * g.b;
        for (size_t offset = 0; offset < count;) {
            int chunk = int(std::min(size_t(INT_MAX), count - offset));
            MPI_Request request;
            MPI(MPI_Ibcast(data + offset, chunk, MPI_DOUBLE, r, g.colComm, &request));
            requests.push_back(request); offset += chunk;
        }
    }
    if (!requests.empty()) MPI(MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
    #pragma omp parallel for schedule(static)
    for (int q = 0; q < tiles; ++q) {
        const double* source = p.packed.data + size_t(locations[q]) * g.b * g.b;
        for (int c = 0; c < g.b; ++c)
            std::memcpy(p.rightHost.data + size_t(c) * nc + q * g.b,
                        source + c * g.b, size_t(g.b) * sizeof(double));
    }
    p.deviceRight = p.right.data;
    p.rightLeading = nc;
    CUDA(cudaMemcpyAsync(p.right.data, p.rightHost.data, size_t(nc) * g.b * sizeof(double),
                         cudaMemcpyHostToDevice, gpu.panel));
    CUDA(cudaEventRecord(gpu.dataReady[slot], gpu.panel));
    CUDA(cudaStreamWaitEvent(gpu.update, gpu.dataReady[slot], 0));
}

void updateColumn(const Grid& g, Accelerator& gpu, double* a, const Panel& p,
                  int rowFirst, int colFirst, int j, double alpha) {
    int begin = g.first(j, g.row, g.pr);
    int m = g.rows - begin;
    const double one = 1.0;
    if (m)
        BLAS(cublasDgemm(gpu.updateBlas, CUBLAS_OP_N, CUBLAS_OP_T, m, g.b, g.b,
            &alpha, p.deviceLeft + begin - rowFirst, p.leftLeading,
            p.deviceRight + (j / g.pc) * g.b - colFirst, p.rightLeading,
            &one, g.tile(a, 0, j) + begin, g.lda));
}

void updateTrailing(const Grid& g, Accelerator& gpu, double* a, const Panel& p,
                    int firstBlock, int slot, double alpha) {
    int rowFirst = g.first(firstBlock, g.row, g.pr);
    int colFirst = g.first(firstBlock, g.col, g.pc);
    int firstCol = colFirst / g.b * g.pc + g.col;
    // Update the next factorization panel first. Its event lets the high
    // priority stream factor it while this stream updates the remaining matrix.
    bool next = firstCol == firstBlock && firstCol < g.blocks;
    if (next) updateColumn(g, gpu, a, p, rowFirst, colFirst, firstCol, alpha);
    CUDA(cudaEventRecord(gpu.nextReady[slot], gpu.update));
    int rest = colFirst + (next ? g.b : 0);
    if (g.pr == g.pc && g.row == g.col) {
        // On diagonal grid ranks the local trailing matrix is symmetric.
        // A single large SYRK replaces all its individual column updates.
        int m = g.rows - rest;
        const double one = 1.0;
        if (m)
            BLAS(cublasDsyrk(gpu.updateBlas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                m, g.b, &alpha, p.deviceLeft + rest - rowFirst, p.leftLeading,
                &one, a + rest + size_t(rest) * g.lda, g.lda));
    } else {
        for (int c = rest; c < g.cols; c += g.b)
            updateColumn(g, gpu, a, p, rowFirst, colFirst, c / g.b * g.pc + g.col, alpha);
    }
    CUDA(cudaEventRecord(gpu.updateDone[slot], gpu.update));
}

bool choleskyDecomposition(const Grid& g, Accelerator& gpu, double* a,
                          FactorizationWorkspace& work) {
    const double one = 1.0;
    for (int k = 0; k < g.blocks; ++k) {
        int slot = k % 2, ownerRow = k % g.pr, ownerCol = k % g.pc;
        if (k) CUDA(cudaStreamWaitEvent(gpu.panel, gpu.nextReady[1 - slot], 0));
        int info = 0;
        if (g.row == ownerRow && g.col == ownerCol) {
            SOLVER(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_LOWER, g.b,
                g.tile(a, k, k), g.lda, work.workspace.data, work.workspaceSize, work.info.data));
            CUDA(cudaMemcpyAsync(&info, work.info.data, sizeof(int), cudaMemcpyDeviceToHost, gpu.panel));
            CUDA(cudaMemcpy2DAsync(work.diagonalHost.data, size_t(g.b) * sizeof(double),
                g.tile(a, k, k), size_t(g.lda) * sizeof(double),
                size_t(g.b) * sizeof(double), g.b, cudaMemcpyDeviceToHost, gpu.panel));
            CUDA(cudaStreamSynchronize(gpu.panel));
        }
        MPI(MPI_Bcast(&info, 1, MPI_INT, ownerRow * g.pc + ownerCol, MPI_COMM_WORLD));
        if (info != 0) {
            if (g.rank == 0) {
                if (info > 0) std::printf("Error: Matrix is not positive definite at diagonal element %d\n", k * g.b + info - 1);
                else std::printf("Error: invalid cuSOLVER factorization argument %d\n", -info);
            }
            CUDA(cudaStreamSynchronize(gpu.update));
            return false;
        }
        if (g.col == ownerCol) {
            broadcast(work.diagonalHost.data, size_t(g.b) * g.b, ownerRow, g.colComm);
            const double* diagonal = g.row == ownerRow ? g.tile(a, k, k) : work.diagonal.data;
            int diagonalLeading = g.row == ownerRow ? g.lda : g.b;
            if (g.row != ownerRow)
                CUDA(cudaMemcpyAsync(work.diagonal.data, work.diagonalHost.data, size_t(g.b) * g.b * sizeof(double),
                                     cudaMemcpyHostToDevice, gpu.panel));
            int begin = g.first(k + 1, g.row, g.pr), m = g.rows - begin;
            if (m)
                BLAS(cublasDtrsm(gpu.panelBlas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m, g.b, &one,
                    diagonal, diagonalLeading, g.tile(a, 0, k) + begin, g.lda));
        }
        if (k + 1 < g.blocks) {
            Panel& panel = slot ? work.odd : work.even;
            exchangePanel(g, gpu, a, panel, k, k + 1, slot, k >= 2);
            updateTrailing(g, gpu, a, panel, k + 1, slot, -1.0);
        }
    }
    CUDA(cudaStreamSynchronize(gpu.update));
    boundary(g, gpu, a, true);
    CUDA(cudaStreamSynchronize(gpu.panel));
    return true;
}

__global__ void validationError(const double* reconstructed, const double* original,
                                double* errors, size_t count, int rows, int b,
                                int pr, int pc, int row, int col, int n) {
    double absolute = 0.0, relative = 0.0;
    for (size_t p = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         p < count; p += size_t(blockDim.x) * gridDim.x) {
        int lr = p % rows, lc = p / rows;
        int i = (lr / b * pr + row) * b + lr % b;
        int j = (lc / b * pc + col) * b + lc % b;
        if (i >= n || j >= n || i < j) continue;
        double e = fabs(reconstructed[p] - original[p]);
        if (!isfinite(e)) e = CUDART_INF;
        absolute = fmax(absolute, e);
        relative = fmax(relative, e / (fabs(original[p]) + 1e-10));
    }
    __shared__ double absBlock[256], relBlock[256];
    absBlock[threadIdx.x] = absolute; relBlock[threadIdx.x] = relative;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            absBlock[threadIdx.x] = fmax(absBlock[threadIdx.x], absBlock[threadIdx.x + stride]);
            relBlock[threadIdx.x] = fmax(relBlock[threadIdx.x], relBlock[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // Positive IEEE doubles have the same ordering as their unsigned bits.
        atomicMax(reinterpret_cast<unsigned long long*>(errors),
                  static_cast<unsigned long long>(__double_as_longlong(absBlock[0])));
        atomicMax(reinterpret_cast<unsigned long long*>(errors + 1),
                  static_cast<unsigned long long>(__double_as_longlong(relBlock[0])));
    }
}

bool validateCholesky(const Grid& g, Accelerator& gpu, double* l, const double* original) {
    Buffer<double> reconstructed(g.elements()), errors(2);
    Panel even(g), odd(g);
    CUDA(cudaMemsetAsync(reconstructed.data, 0, g.elements() * sizeof(double), gpu.update));
    // Reconstruct the distributed lower triangle with exactly the same panel
    // exchanges. Symmetry makes its maximum errors equal to the full matrix's.
    for (int k = 0; k < g.blocks; ++k) {
        int slot = k % 2;
        Panel& panel = slot ? odd : even;
        exchangePanel(g, gpu, l, panel, k, k, slot, k >= 2);
        updateTrailing(g, gpu, reconstructed.data, panel, k, slot, 1.0);
    }
    CUDA(cudaStreamSynchronize(gpu.update));
    CUDA(cudaMemsetAsync(errors.data, 0, 2 * sizeof(double), gpu.panel));
    validationError<<<kernelBlocks(g.elements()), 256, 0, gpu.panel>>>(
        reconstructed.data, original, errors.data, g.elements(), g.rows,
        g.b, g.pr, g.pc, g.row, g.col, g.n);
    CUDA(cudaGetLastError());
    double local[2], global[2];
    CUDA(cudaMemcpyAsync(local, errors.data, sizeof(local), cudaMemcpyDeviceToHost, gpu.panel));
    CUDA(cudaStreamSynchronize(gpu.panel));
    MPI(MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    if (g.rank == 0) {
        std::printf("Max absolute error: %.10e\n", global[0]);
        std::printf("Max relative error: %.10e\n", global[1]);
        if (global[1] > 1e-6) std::printf("Validation failed: relative error too large\n");
    }
    return std::isfinite(global[1]) && global[1] <= 1e-6;
}

void printDistributedResults(const Grid& g, double* a) {
    std::vector<double> local(g.elements());
    CUDA(cudaMemcpy(local.data(), a, g.elements() * sizeof(double), cudaMemcpyDeviceToHost));
    if (g.rank != 0) {
        for (size_t offset = 0; offset < local.size();) {
            int count = int(std::min(size_t(INT_MAX), local.size() - offset));
            MPI(MPI_Send(local.data() + offset, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
            offset += count;
        }
        return;
    }
    std::vector<double> result(size_t(g.n) * g.n), received;
    for (int rank = 0; rank < g.size; ++rank) {
        int row = rank / g.pc, col = rank % g.pc;
        int rows = g.localBlocks(row, g.pr) * g.b, cols = g.localBlocks(col, g.pc) * g.b;
        const double* data = local.data();
        if (rank) {
            received.resize(size_t(rows) * cols);
            for (size_t offset = 0; offset < received.size();) {
                int count = int(std::min(size_t(INT_MAX), received.size() - offset));
                MPI(MPI_Recv(received.data() + offset, count, MPI_DOUBLE, rank, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                offset += count;
            }
            data = received.data();
        }
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r) {
            int i = g.global(r, row, g.pr);
            if (i >= g.n) continue;
            for (int c = 0; c < cols; ++c) {
                int j = g.global(c, col, g.pc);
                if (j < g.n) result[size_t(i) * g.n + j] = data[r + size_t(c) * rows];
            }
        }
    }
    print_results(result, "CholeskyL");
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n", name);
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int benchmark(int argc, char** argv, int rank, int size) {
    int n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            long long value = std::strtoll(argv[++i], &end, 10);
            // Leave room for a padded tile in the BLAS integer dimensions.
            if (errno || end == argv[i] || *end || value < 0 || value > INT_MAX - 256 ||
                uint64_t(value) * uint64_t(value) > std::numeric_limits<size_t>::max() / sizeof(double)) {
                if (rank == 0) std::fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            n = int(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }

    MPI_Comm node;
    MPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int localRank, localSize;
    MPI(MPI_Comm_rank(node, &localRank)); MPI(MPI_Comm_size(node, &localSize));
    MPI(MPI_Comm_free(&node));
    int devices = 0;
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("A CUDA GPU is required on every MPI rank");
    // This also handles launchers that expose one assigned GPU per rank.
    CUDA(cudaSetDevice(localRank % devices));
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::min(8, std::max(1, omp_get_num_procs() / localSize)));

    Grid g(n, rank, size);
    Accelerator gpu;
    Buffer<double> a(g.elements()), original(validate ? g.elements() : 0);
    if (rank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\n", n, n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI process grid: %d x %d; CUDA tile size: %d; OpenMP threads per rank: %d\n",
                    g.pr, g.pc, g.b, omp_get_max_threads());
        std::printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(g, gpu, a.data);
    if (validate) {
        CUDA(cudaMemcpyAsync(original.data, a.data, g.elements() * sizeof(double),
                             cudaMemcpyDeviceToDevice, gpu.panel));
        CUDA(cudaStreamSynchronize(gpu.panel));
    }
    bool success;
    double elapsed;
    {
        FactorizationWorkspace work(g, gpu);
        if (rank == 0) std::printf("Computing Cholesky decomposition...\n");
        MPI(MPI_Barrier(MPI_COMM_WORLD));
        double start = MPI_Wtime();
        success = choleskyDecomposition(g, gpu, a.data, work);
        elapsed = MPI_Wtime() - start;
    }
    double seconds = 0;
    MPI(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!success) {
        if (rank == 0) std::printf("Cholesky decomposition failed\n");
        return 1;
    }
    if (rank == 0) {
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        double ops = double(n) * n * n / 3.0;
        std::printf("Performance: %.3f GFLOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0.0);
    }
    if (printResults) printDistributedResults(g, a.data);
    if (validate) {
        if (rank == 0) std::printf("Validating result...\n");
        bool valid = validateCholesky(g, gpu, a.data, original.data);
        if (rank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    try {
        if (provided < MPI_THREAD_FUNNELED)
            throw std::runtime_error("MPI_THREAD_FUNNELED support is required");
        int status = benchmark(argc, argv, rank, size);
        MPI(MPI_Finalize());
        return status;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
