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
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime_api.h>
#include "../common/results_output.hpp"

namespace {
void check(int status, const char *call) {
    if (status) {
        fprintf(stderr, "%s failed (status %d)\n", call, status);
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}
#define CHECK(call) check((call), #call)

template <class T> struct Device {
    T *p = nullptr;
    explicit Device(size_t n) {
        CHECK(cudaMalloc(reinterpret_cast<void **>(&p), std::max<size_t>(1, n) * sizeof(T)));
    }
    ~Device() { cudaFree(p); }
    Device(const Device &) = delete;
};

struct Pinned {
    double *p;
    explicit Pinned(size_t n) {
        CHECK(cudaMallocHost(reinterpret_cast<void **>(&p), std::max<size_t>(1, n) * sizeof(double)));
    }
    ~Pinned() { cudaFreeHost(p); }
    Pinned(const Pinned &) = delete;
};

struct Axis {
    int n, b, coord, parts;
    // Number of locally owned indices strictly before global index end.
    int before(int end) const {
        int tiles = end / b;
        return (tiles / parts) * b + (coord < tiles % parts ? b : coord == tiles % parts ? end % b : 0);
    }
    int size() const { return before(n); }
    int global(int local) const { return ((local / b) * parts + coord) * b + local % b; }
};

struct Grid {
    int rank, ranks, dims[2]{}, b;
    MPI_Comm row, col;
    Axis rows, cols;
    explicit Grid(int n) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ranks);
        MPI_Dims_create(ranks, 2, dims);
        b = n >= 8192 ? 256 : n >= 2048 ? 128 : 64;
        while (b > 8 && (n + (long long)b - 1) / b < dims[0])
            b /= 2;
        rows = {n, b, rank / dims[1], dims[0]};
        cols = {n, b, rank % dims[1], dims[1]};
        MPI_Comm_split(MPI_COMM_WORLD, rows.coord, cols.coord, &row);
        MPI_Comm_split(MPI_COMM_WORLD, cols.coord, rows.coord, &col);
    }
    ~Grid() {
        MPI_Comm_free(&row);
        MPI_Comm_free(&col);
    }
};

constexpr double one = 1, zero = 0, minusOne = -1;
size_t area(int a, int b) { return size_t(a) * b; }

// Each glibc rand_r draw advances its unsigned LCG three times. Jump to
// the seed at the beginning of each row without serializing the draws.
std::vector<unsigned> rowSeeds(int n) {
    std::vector<unsigned> seeds(n);
#ifdef __GLIBC__
    unsigned mul = 1, add = 0, a = 1103515245u, c = 12345u;
    for (unsigned long long steps = 3ull * n; steps; steps >>= 1) {
        if (steps & 1) {
            add = a * add + c;
            mul *= a;
        }
        c *= a + 1;
        a *= a;
    }
#endif
    unsigned seed = 42;
    for (auto &start : seeds) {
        start = seed;
#ifdef __GLIBC__
        seed = mul * seed + add;
#else
        for (int k = 0; k < n; ++k)
            rand_r(&seed);
#endif
    }
    return seeds;
}

void fillRandom(const Axis &axis, std::vector<unsigned> &seeds, int width, double *buf) {
#pragma omp parallel for schedule(static) if (area(axis.size(), width) >= 32768)
    for (int i = 0; i < axis.size(); ++i) {
        unsigned s = seeds[axis.global(i)];
        for (int k = 0; k < width; ++k)
            buf[area(i, width) + k] = rand_r(&s) / double(RAND_MAX) - 0.5;
        seeds[axis.global(i)] = s;
    }
}

void generate(const Grid &g, cublasHandle_t blas, double *matrix) {
    int m = g.rows.size(), l = g.cols.size(), n = g.rows.n;
    if (!m || !l)
        return;
    auto rs = rowSeeds(n), cs = rs;
    int chunk = std::min(n, 1024);
    bool same = g.rows.coord == g.cols.coord && g.rows.parts == g.cols.parts;
    Pinned host(area(std::max(m, l), chunk));
    Device<double> x(area(m, chunk)), y(same ? 0 : area(l, chunk));
    for (int k = 0; k < n;) {
        int w = std::min(chunk, n - k);
        fillRandom(g.rows, rs, w, host.p);
        CHECK(cudaMemcpy(x.p, host.p, area(m, w) * sizeof(double), cudaMemcpyHostToDevice));
        if (!same) {
            fillRandom(g.cols, cs, w, host.p);
            CHECK(cudaMemcpy(y.p, host.p, area(l, w) * sizeof(double), cudaMemcpyHostToDevice));
        }
        const double *beta = k ? &one : &zero;
        if (same) {
            CHECK(cublasDsyrk(blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, m, w, &one,
                              x.p, w, beta, matrix, m));
        } else {
            for (int j = 0; j < l;) {
                int width = std::min(4 * g.b, l - j);
                int i = g.rows.before(g.cols.global(j));
                if (i < m)
                    CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, m - i, width, w, &one,
                                      x.p + area(i, w), w, y.p + area(j, w), w, beta,
                                      matrix + area(j, m) + i, m));
                j += width;
            }
        }
        k += w;
    }
    // Add n to the diagonal using one strided BLAS operation per owned tile.
    std::vector<double> shift(g.b, double(n));
    CHECK(cudaMemcpy(x.p, shift.data(), std::min(n, g.b) * sizeof(double), cudaMemcpyHostToDevice));
    for (int j = 0; j < l;) {
        int global = g.cols.global(j), w = std::min(g.b, l - j);
        if ((global / g.b) % g.rows.parts == g.rows.coord)
            CHECK(
                cublasDaxpy(blas, w, &one, x.p, 1, matrix + area(j, m) + g.rows.before(global), m + 1));
        j += w;
    }
}

// A panel is broadcast along each process row. Within each process column
// redistribute the same values in local column order for the transposed operand.
struct Panels {
    const Grid &g;
    int m, l;
    Device<double> left, right, diag;
    Pinned hleft, hright, packed, diagHost;
    std::vector<int> counts, offsets;
    explicit Panels(const Grid &grid)
        : g(grid), m(g.rows.size()), l(g.cols.size()), left(area(m, g.b)), right(area(l, g.b)),
          diag(area(g.b, g.b)), hleft(area(m, g.b)), hright(area(l, g.b)), packed(area(l, g.b)),
          diagHost(area(g.b, g.b)), counts(g.rows.parts), offsets(g.rows.parts) {}

    void exchange(double *matrix, int k, int w, bool includeDiagonal) {
        int owner = (k / g.b) % g.cols.parts;
        int first = k + (includeDiagonal ? 0 : w);
        int i = g.rows.before(first), j = g.cols.before(first);
        int nr = m - i, nc = l - j;
        if (g.cols.coord == owner && nr)
            CHECK(cudaMemcpy2D(left.p, nr * sizeof(double), matrix + area(g.cols.before(k), m) + i,
                               m * sizeof(double), nr * sizeof(double), w, cudaMemcpyDeviceToDevice));
        if (g.ranks == 1)
            return;
        if (g.cols.coord == owner && nr)
            CHECK(cudaMemcpy(hleft.p, left.p, area(nr, w) * sizeof(double), cudaMemcpyDeviceToHost));
        if (g.cols.parts > 1) {
            CHECK(MPI_Bcast(hleft.p, nr * w, MPI_DOUBLE, owner, g.row));
            if (g.cols.coord != owner && nr)
                CHECK(cudaMemcpy(left.p, hleft.p, area(nr, w) * sizeof(double), cudaMemcpyHostToDevice));
        }
        if (!nc)
            return;
        if (g.rows.parts == g.cols.parts) {
            if (g.rows.coord == g.cols.coord)
                std::copy_n(hleft.p, area(nc, w), hright.p);
            CHECK(MPI_Bcast(hright.p, nc * w, MPI_DOUBLE, g.cols.coord, g.col));
        } else {
            // Pack whole tile rows contiguously, grouped by their row owner.
            std::fill(counts.begin(), counts.end(), 0);
            for (int t = j; t < l;) {
                int global = g.cols.global(t), len = std::min(g.b, l - t);
                int r = (global / g.b) % g.rows.parts;
                if (r == g.rows.coord) {
                    for (int d = 0; d < w; ++d)
                        std::copy_n(hleft.p + area(d, nr) + (g.rows.before(global) - i), len,
                                    hright.p + counts[r] + area(d, len));
                }
                counts[r] += len * w;
                t += len;
            }
            offsets[0] = 0;
            for (int r = 1; r < g.rows.parts; ++r)
                offsets[r] = offsets[r - 1] + counts[r - 1];
            CHECK(MPI_Allgatherv(hright.p, counts[g.rows.coord], MPI_DOUBLE, packed.p, counts.data(),
                                 offsets.data(), MPI_DOUBLE, g.col));
            for (int t = j; t < l;) {
                int global = g.cols.global(t), len = std::min(g.b, l - t),
                    r = (global / g.b) % g.rows.parts;
                for (int d = 0; d < w; ++d)
                    std::copy_n(packed.p + offsets[r] + area(d, len), len,
                                hright.p + area(d, nc) + (t - j));
                offsets[r] += len * w;
                t += len;
            }
        }
        CHECK(cudaMemcpy(right.p, hright.p, area(nc, w) * sizeof(double), cudaMemcpyHostToDevice));
    }

    void subtract(cublasHandle_t blas, double *a, int first, int w) {
        int i = g.rows.before(first), j = g.cols.before(first);
        // Group adjacent local tiles to amortize launches while skipping the
        // large rectangles wholly above the global diagonal.
        for (int t = j; t < l;) {
            int width = std::min(4 * g.b, l - t), row = g.rows.before(g.cols.global(t));
            if (row < m)
                CHECK(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, m - row, width, w, &minusOne,
                                  left.p + (row - i), m - i, (g.ranks == 1 ? left.p : right.p) + (t - j),
                                  l - j, &one, a + area(t, m) + row, m));
            t += width;
        }
    }
};

bool factor(const Grid &g, Panels &p, cublasHandle_t blas, cusolverDnHandle_t solver, double *a,
            double *work, int workspace, int *info) {
    int n = g.rows.n, m = p.m;
    for (int k = 0; k < n;) {
        // cuSOLVER can schedule the entire factorization on a single GPU.
        int w = g.ranks == 1 ? n : std::min(g.b, n - k);
        int r = (k / g.b) % g.rows.parts, c = (k / g.b) % g.cols.parts;
        int i = g.rows.before(k), j = g.cols.before(k);
        if (g.cols.coord == c) {
            if (g.rows.coord == r) {
                double *tile = a + area(j, m) + i;
                CHECK(
                    cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_LOWER, w, tile, m, work, workspace, info));
                int pivot;
                CHECK(cudaMemcpy(&pivot, info, sizeof(int), cudaMemcpyDeviceToHost));
                if (pivot) {
                    fprintf(stderr, "Error: Matrix is not positive definite at diagonal element %d\n",
                            k + pivot - 1);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return false;
                }
                if (g.rows.parts > 1)
                    CHECK(cudaMemcpy2D(p.diagHost.p, w * sizeof(double), tile, m * sizeof(double),
                                       w * sizeof(double), w, cudaMemcpyDeviceToHost));
            }
            if (g.rows.parts > 1) {
                CHECK(MPI_Bcast(p.diagHost.p, w * w, MPI_DOUBLE, r, g.col));
                CHECK(cudaMemcpy(p.diag.p, p.diagHost.p, area(w, w) * sizeof(double),
                                 cudaMemcpyHostToDevice));
            }
            int next = g.rows.before(k + w);
            if (next < m)
                CHECK(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                  CUBLAS_DIAG_NON_UNIT, m - next, w, &one,
                                  g.rows.parts > 1 ? p.diag.p : a + area(j, m) + i,
                                  g.rows.parts > 1 ? w : m, a + area(j, m) + next, m));
        }
        if (k + w < n) {
            p.exchange(a, k, w, false);
            p.subtract(blas, a, k + w, w);
        }
        k += w;
    }
    return true;
}

void results(const Grid &g, const std::vector<double> &local) {
    int n = g.rows.n;
    std::vector<double> full(g.rank == 0 ? area(n, n) : 0);
    // Gather one rank at a time and keep each MPI count representable.
    for (int rank = 0; rank < g.ranks; ++rank) {
        Axis rows{n, g.b, rank / g.cols.parts, g.rows.parts};
        Axis cols{n, g.b, rank % g.cols.parts, g.cols.parts};
        size_t length = area(rows.size(), cols.size());
        if (!length)
            continue;
        if (g.rank != 0 && g.rank != rank)
            continue;
        std::vector<double> received(g.rank == 0 && rank != 0 ? length : 0);
        if (rank != 0) {
            for (size_t offset = 0; offset < length;) {
                int count = static_cast<int>(std::min<size_t>(length - offset, INT_MAX));
                if (g.rank == 0)
                    CHECK(MPI_Recv(received.data() + offset, count, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD,
                                   MPI_STATUS_IGNORE));
                else
                    CHECK(MPI_Send(local.data() + offset, count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
                offset += count;
            }
        }
        if (g.rank == 0) {
            const auto &values = rank == 0 ? local : received;
#pragma omp parallel for schedule(static) if (length >= 32768)
            for (int i = 0; i < rows.size(); ++i)
                for (int j = 0; j < cols.size(); ++j)
                    full[area(rows.global(i), n) + cols.global(j)] = values[area(j, rows.size()) + i];
        }
    }
    if (g.rank == 0)
        print_results(full, "CholeskyL");
}

bool benchmark(int n, bool validate, bool print) {
    Grid g(n);
    if (!g.rank) {
        printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n", n, n,
               validate ? "enabled" : "disabled");
        puts("Generating positive definite matrix...");
    }
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g.rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CHECK(cudaSetDevice(localRank % devices));
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    CHECK(cublasCreate(&blas));
    CHECK(cusolverDnCreate(&solver));

    int m = g.rows.size(), l = g.cols.size();
    size_t length = area(m, l), bytes = length * sizeof(double);
    Device<double> a(length), original(validate ? length : 0);
    generate(g, blas, a.p);
    if (validate)
        CHECK(cudaMemcpy(original.p, a.p, bytes, cudaMemcpyDeviceToDevice));
    Panels panels(g);
    int workspace = 0;
    if (m && l) {
        int w = g.ranks == 1 ? n : std::min({g.b, m, l});
        CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, w, a.p, m, &workspace));
    }
    Device<double> work(workspace);
    Device<int> info(1);
    CHECK(cudaDeviceSynchronize());
    if (!g.rank)
        puts("Computing Cholesky decomposition...");
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (!factor(g, panels, blas, solver, a.p, work.p, workspace, info.p))
        return false;
    CHECK(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, slowest = 0;
    MPI_Reduce(&elapsed, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!g.rank) {
        printf("Computation time: %lld ms\n", static_cast<long long>(slowest * 1000));
        printf("Performance: %.3f GFLOPS\n", double(n) * n * n / (3e9 * slowest));
    }

    bool valid = true;
    if (print || validate) {
        std::vector<double> local(length);
        if (length)
            CHECK(cudaMemcpy(local.data(), a.p, bytes, cudaMemcpyDeviceToHost));
#pragma omp parallel for schedule(static) if (length >= 32768)
        for (int j = 0; j < l; ++j) {
            int end = g.rows.before(g.cols.global(j));
            if (end)
                std::fill_n(local.data() + area(j, m), end, 0.0);
        }
        if (print)
            results(g, local);
        if (validate) {
            if (!g.rank)
                puts("Validating result...");
            if (length)
                CHECK(cudaMemcpy(a.p, local.data(), bytes, cudaMemcpyHostToDevice));
            if (length)
                CHECK(cudaMemcpy(local.data(), original.p, bytes, cudaMemcpyDeviceToHost));
            // Accumulate A - L*L^T on its owners, then inspect the lower half
            // (the other half of this residual is its transpose).
            for (int k = 0; k < n;) {
                int w = std::min(g.b, n - k);
                panels.exchange(a.p, k, w, true);
                panels.subtract(blas, original.p, k, w);
                k += w;
            }
            std::vector<double> residual(length);
            if (length)
                CHECK(cudaMemcpy(residual.data(), original.p, bytes, cudaMemcpyDeviceToHost));
            double absError = 0, relError = 0;
#pragma omp parallel for reduction(max : absError, relError) if (length >= 32768)
            for (int j = 0; j < l; ++j) {
                for (int i = g.rows.before(g.cols.global(j)); i < m; ++i) {
                    size_t idx = area(j, m) + i;
                    double e = std::abs(residual[idx]);
                    if (!std::isfinite(e))
                        e = std::numeric_limits<double>::infinity();
                    absError = std::max(absError, e);
                    relError = std::max(relError, e / (std::abs(local[idx]) + 1e-10));
                }
            }
            double errors[] = {absError, relError};
            MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            valid = errors[1] <= 1e-6;
            if (!g.rank) {
                printf("Max absolute error: %.10e\nMax relative error: %.10e\n", errors[0], errors[1]);
                if (!valid)
                    puts("Validation failed: relative error too large");
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
        }
    }
    CHECK(cusolverDnDestroy(solver));
    CHECK(cublasDestroy(blas));
    return valid;
}

void usage(const char *name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    puts("  -n <num>     Matrix size (default: 512)\n"
         "  -v           Enable validation\n"
         "  -r           Print results for external validation\n"
         "  -h           Show this help message");
}
} // namespace

int main(int argc, char **argv) {
    int provided, rank;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (provided < MPI_THREAD_FUNNELED)
        MPI_Abort(MPI_COMM_WORLD, 1);
    int n = 512, status = 0;
    bool validate = false, print = false, run = true;
    for (int i = 1; i < argc && run; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char *end;
            errno = 0;
            long value = strtol(argv[++i], &end, 10);
            // MPI counts for a packed panel and BLAS dimensions are int.
            if (errno || end == argv[i] || *end || value < 0 || value > INT_MAX / 256) {
                if (!rank)
                    fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                run = false;
                status = 1;
            } else
                n = static_cast<int>(value);
        } else if (!strcmp(argv[i], "-v"))
            validate = true;
        else if (!strcmp(argv[i], "-r"))
            print = true;
        else {
            if (strcmp(argv[i], "-h")) {
                if (!rank)
                    printf("Unknown option: %s\n", argv[i]);
                status = 1;
            }
            if (!rank)
                usage(argv[0]);
            run = false;
        }
    }
    if (run) {
        try {
            status = benchmark(n, validate, print) ? 0 : 1;
        } catch (const std::exception &e) {
            fprintf(stderr, "Rank %d: %s\n", rank, e.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Finalize();
    return status;
}
