#include <algorithm>
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

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

namespace {
int rank, ranks;
constexpr double one = 1, zero = 0, minusOne = -1;

void check(int status, const char* operation) {
    if (status) {
        std::fprintf(stderr, "Rank %d: %s failed (status %d)\n", rank, operation, status);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CHECK(call) check((call), #call)

template <class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t count) {
        if (count)
            CHECK(cudaMalloc(&p, count * sizeof(T)));
    }
    ~Device() {
        if (p)
            CHECK(cudaFree(p));
    }
    Device(const Device&) = delete;
};

struct Grid {
    int n, tile, cols, rank, ranks;
    __host__ __device__ int global(int local) const {
        return int(((long long)(local / tile) * ranks + rank) * tile + local % tile);
    }
    int owner(int k) const { return (k / tile) % ranks; }
    int local(int k) const { return k / tile / ranks * tile; }
};

void broadcast(double* p, size_t count, int owner) {
    while (count) {
        int chunk = int(std::min<size_t>(count, INT_MAX));
        CHECK(MPI_Bcast(p, chunk, MPI_DOUBLE, owner, MPI_COMM_WORLD));
        p += chunk;
        count -= chunk;
    }
}

// glibc rand_r advances its 32-bit LCG three times per output. Jump to a row
// boundary, then use the original library function for every sample.
#ifdef __GLIBC__
unsigned rowSeed(size_t offset) {
    unsigned seed = 42, mul = 1103515245u, add = 12345u;
    for (size_t steps = 3 * offset; steps; steps >>= 1) {
        if (steps & 1)
            seed = mul * seed + add;
        add *= mul + 1;
        mul *= mul;
    }
    return seed;
}
#endif

void generate(Grid g, double* a, double* panel, double* host, cublasHandle_t blas) {
    size_t count = size_t(g.n) * g.cols;
    Device<double> b(count);
    std::vector<double> values(count);
#ifndef __GLIBC__
    std::vector<unsigned> seeds(g.n);
    unsigned state = 42;
    for (int i = 0; i < g.n; ++i) {
        seeds[i] = state;
        for (int j = 0; j < g.n; ++j)
            rand_r(&state);
    }
#endif
#pragma omp parallel for schedule(static)
    for (int c = 0; c < g.cols; ++c) {
#ifdef __GLIBC__
        unsigned seed = rowSeed(size_t(g.global(c)) * g.n);
#else
        unsigned seed = seeds[g.global(c)];
#endif
        for (int i = 0; i < g.n; ++i)
            values[size_t(c) * g.n + i] = rand_r(&seed) / double(RAND_MAX) - 0.5;
    }
    if (count)
        CHECK(cudaMemcpy(b.p, values.data(), count * sizeof(double), cudaMemcpyHostToDevice));
    if (ranks == 1 && g.n) {
        CHECK(cublasDsyrk(blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, g.n, g.n, &one, b.p, g.n,
                          &zero, a, g.n));
        return;
    }
    // Grouped trailing updates can touch entries just above the diagonal.
    if (count)
        CHECK(cudaMemset(a, 0, count * sizeof(double)));
    int prefix = 0;
    // B rows are contiguous vectors, interpreted by BLAS as columns of B^T.
    for (int k = 0; k < g.n;) {
        int width = ranks == 1 ? g.n : std::min(g.tile, g.n - k);
        size_t length = size_t(g.n) * width;
        const double* left = b.p;
        if (ranks > 1) {
            if (rank == g.owner(k))
                std::copy_n(values.data() + size_t(g.local(k)) * g.n, length, host);
            broadcast(host, length, g.owner(k));
            if (rank == g.owner(k))
                left = b.p + size_t(g.local(k)) * g.n;
            else {
                CHECK(cudaMemcpy(panel, host, length * sizeof(double), cudaMemcpyHostToDevice));
                left = panel;
            }
        }
        if (rank == g.owner(k))
            prefix += width;
        if (prefix)
            CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, width, prefix, g.n, &one, left, g.n,
                              b.p, g.n, &zero, a + k, g.n));
        k += width;
    }
}

__global__ void shiftDiagonal(double* a, Grid g) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < g.cols) {
        a[size_t(c) * g.n + g.global(c)] += g.n;
    }
}

// Pack the rows of a broadcast panel that correspond to this rank's columns.
__global__ void packRows(const double* panel, int ld, double* packed, Grid g, int first, int k) {
    int c = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (c < g.cols) {
        packed[size_t(blockIdx.y) * g.cols + c] = panel[size_t(blockIdx.y) * ld + g.global(c) - k];
    }
}

int factor(Grid g, double* a, double* panel, double* host, double* packed, cublasHandle_t blas,
           cusolverDnHandle_t solver) {
    int block = ranks == 1 ? g.n : g.tile;
    int workSize = 0;
    if (g.cols)
        CHECK(
            cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, block, a, g.n, &workSize));
    Device<double> work(workSize);
    Device<int> status(1);
    int first = 0;
    for (int k = 0; k < g.n;) {
        int width = std::min(block, g.n - k), ld = g.n - k, owner = g.owner(k);
        int below = ld - width;
        const double* left = panel;
        if (rank == owner) {
            double* diagonal = a + size_t(first) * g.n + k;
            CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, width, diagonal, g.n, work.p,
                                   workSize, status.p));
            int info;
            CHECK(cudaMemcpy(&info, status.p, sizeof(info), cudaMemcpyDeviceToHost));
            if (info < 0)
                check(info, "potrf argument");
            if (ranks == 1)
                return info ? k + info : 0;
            if (!info && below) {
                CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                  CUBLAS_DIAG_NON_UNIT, below, width, &one, diagonal, g.n,
                                  diagonal + width, g.n));
                CHECK(cudaMemcpy2D(host, ld * sizeof(double), diagonal, g.n * sizeof(double),
                                   ld * sizeof(double), width, cudaMemcpyDeviceToHost));
            }
            // The first pivot is positive. Use its broadcast slot to signal failure.
            if (info)
                host[0] = -(k + info);
            else if (!below)
                host[0] = 1;
            first += width;
            left = diagonal;
        }
        broadcast(host, below ? size_t(ld) * width : 1, owner);
        if (host[0] < 0)
            return int(-host[0]);
        if (first < g.cols) {
            if (rank != owner)
                CHECK(cudaMemcpy(panel, host, size_t(ld) * width * sizeof(double),
                                 cudaMemcpyHostToDevice));
            int pitch = rank == owner ? g.n : ld;
            packRows<<<dim3((g.cols - first + 255) / 256, width), 256>>>(left, pitch, packed, g,
                                                                         first, k);
            CHECK(cudaGetLastError());
            // Group adjacent local tiles for larger GEMMs, skipping rows above the group.
            for (int c = first; c < g.cols;) {
                int nc = std::min(4 * g.tile, g.cols - c), row = g.global(c);
                CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, g.n - row, nc, width, &minusOne,
                                  left + (row - k), pitch, packed + c, g.cols, &one,
                                  a + size_t(c) * g.n + row, g.n));
                c += nc;
            }
        }
        k += width;
    }
    return 0;
}

__global__ void clearUpper(double* a, Grid g) {
    int c = blockIdx.x;
    int global = g.global(c);
    for (int i = threadIdx.x; i < global; i += blockDim.x)
        a[size_t(c) * g.n + i] = 0;
}

void results(Grid g, const double* a, double* host) {
    std::vector<double> result;
    if (!rank)
        result.resize(size_t(g.n) * g.n);
    for (int k = 0; k < g.n;) {
        int width = std::min(g.tile, g.n - k), owner = g.owner(k);
        size_t length = size_t(width) * g.n;
        if (rank == owner)
            CHECK(cudaMemcpy(host, a + size_t(g.local(k)) * g.n, length * sizeof(double),
                             cudaMemcpyDeviceToHost));
        if (owner && (rank == owner || !rank)) {
            for (size_t off = 0; off < length;) {
                int chunk = int(std::min<size_t>(length - off, INT_MAX));
                if (rank)
                    CHECK(MPI_Send(host + off, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
                else
                    CHECK(MPI_Recv(host + off, chunk, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD,
                                   MPI_STATUS_IGNORE));
                off += chunk;
            }
        }
        if (!rank) {
#pragma omp parallel for schedule(static)
            for (int row = 0; row < g.n; ++row)
                for (int j = 0; j < width; ++j)
                    result[size_t(row) * g.n + k + j] = host[size_t(j) * g.n + row];
        }
        k += width;
    }
    if (!rank)
        print_results(result, "CholeskyL");
}

bool validate(Grid g, cublasHandle_t blas, const double* a, const double* original, double* host,
              double* panel, double* packed) {
    size_t count = size_t(g.n) * g.cols;
    Device<double> recon(count);
    for (int k = 0; k < g.n;) {
        int width = ranks == 1 ? g.n : std::min(g.tile, g.n - k);
        const double* left = a;
        const double* right = a;
        int rightLd = g.n;
        if (ranks > 1) {
            size_t bytes = size_t(width) * g.n * sizeof(double);
            if (rank == g.owner(k)) {
                left = a + size_t(g.local(k)) * g.n;
                CHECK(cudaMemcpy(host, left, bytes, cudaMemcpyDeviceToHost));
            }
            broadcast(host, bytes / sizeof(double), g.owner(k));
            if (rank != g.owner(k)) {
                CHECK(cudaMemcpy(panel, host, bytes, cudaMemcpyHostToDevice));
                left = panel;
            }
            if (g.cols) {
                packRows<<<dim3((g.cols + 255) / 256, width), 256>>>(left, g.n, packed, g, 0, 0);
                CHECK(cudaGetLastError());
            }
            right = packed;
            rightLd = g.cols;
        }
        if (g.cols)
            CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, g.n, g.cols, width, &one, left, g.n,
                              right, rightLd, k ? &one : &zero, recon.p, g.n));
        k += width;
    }
    std::vector<double> values(count);
    if (count)
        CHECK(cudaMemcpy(values.data(), recon.p, count * sizeof(double), cudaMemcpyDeviceToHost));
    double absolute = 0, relative = 0;
    // Both Gram matrices are symmetric, so each pair contributes once to the maxima.
#pragma omp parallel for reduction(max : absolute, relative)
    for (int c = 0; c < g.cols; ++c) {
        for (int row = g.global(c); row < g.n; ++row) {
            size_t i = size_t(c) * g.n + row;
            double err = std::fabs(values[i] - original[i]);
            if (!std::isfinite(err))
                err = std::numeric_limits<double>::infinity();
            absolute = std::max(absolute, err);
            relative = std::max(relative, err / (std::fabs(original[i]) + 1e-10));
        }
    }
    double errors[] = {absolute, relative};
    CHECK(MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    if (!rank)
        std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", errors[0], errors[1]);
    return errors[1] <= 1e-6;
}

int run(int argc, char** argv) {
    int n = 512;
    bool verify = false, print = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            long size = std::atol(argv[++i]);
            if (size < 0 || size > INT_MAX)
                return 1;
            n = int(size);
        } else if (!strcmp(argv[i], "-v"))
            verify = true;
        else if (!strcmp(argv[i], "-r"))
            print = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank)
                printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    MPI_Comm node;
    CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int localRank, devices;
    CHECK(MPI_Comm_rank(node, &localRank));
    CHECK(MPI_Comm_free(&node));
    CHECK(cudaGetDeviceCount(&devices));
    if (!devices)
        check(1, "no CUDA device");
    CHECK(cudaSetDevice(localRank % devices));
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    CHECK(cublasCreate(&blas));
    CHECK(cusolverDnCreate(&solver));

    int tile = std::min(128, std::max(1, int((n + (long long)ranks - 1) / ranks)));
    int blocks = n / tile;
    int cols = (blocks / ranks + (rank < blocks % ranks)) * tile;
    if (rank == blocks % ranks)
        cols += n % tile;
    Grid g{n, tile, cols, rank, ranks};
    size_t count = size_t(n) * cols, capacity = size_t(n) * tile;
    Device<double> a(count), panel(ranks > 1 ? capacity : 0),
        packed(ranks > 1 ? size_t(cols) * tile : 0);
    double* host = nullptr;
    if (capacity && (ranks > 1 || print))
        CHECK(cudaMallocHost(&host, capacity * sizeof(double)));
    if (!rank)
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n"
                    "Generating positive definite matrix...\n",
                    n, n, verify ? "enabled" : "disabled");
    generate(g, a.p, panel.p, host, blas);
    if (cols) {
        shiftDiagonal<<<(cols + 255) / 256, 256>>>(a.p, g);
        CHECK(cudaGetLastError());
    }
    std::vector<double> original;
    if (verify && count) {
        original.resize(count);
        CHECK(cudaMemcpy(original.data(), a.p, count * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CHECK(cudaDeviceSynchronize());
    if (!rank)
        std::puts("Computing Cholesky decomposition...");
    CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    int failure = n ? factor(g, a.p, panel.p, host, packed.p, blas, solver) : 0;
    if (!failure && cols) {
        clearUpper<<<cols, 256>>>(a.p, g);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, maximum = 0;
    CHECK(MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (failure) {
        if (!rank)
            std::printf("Error: Matrix is not positive definite at diagonal element %d\n"
                        "Cholesky decomposition failed\n",
                        failure - 1);
    } else {
        if (!rank)
            std::printf("Computation time: %.3f ms\nPerformance: %.3f GFLOPS\n", maximum * 1000,
                        maximum ? double(n) * n * n / (3e9 * maximum) : 0);
        if (print)
            results(g, a.p, host);
        if (verify) {
            if (!rank)
                std::puts("Validating result...");
            bool valid = validate(g, blas, a.p, original.data(), host, panel.p, packed.p);
            if (!rank) {
                if (!valid)
                    std::puts("Validation failed: relative error too large");
                std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
            failure = !valid;
        }
    }
    if (host)
        CHECK(cudaFreeHost(host));
    CHECK(cusolverDnDestroy(solver));
    CHECK(cublasDestroy(blas));
    return failure ? 1 : 0;
}
} // namespace

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED)
        check(1, "MPI_THREAD_FUNNELED unavailable");
    int result = run(argc, argv);
    MPI_Finalize();
    return result;
}
