#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed-memory blocked Cholesky decomposition with 2D process grid.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.

namespace {

struct Grid {
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Comm row_comm = MPI_COMM_NULL;
    MPI_Comm col_comm = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    int dims[2] = {1, 1}; // pr, pc
    int coords[2] = {0, 0};
};

static Grid make_grid(MPI_Comm comm) {
    Grid g;
    g.comm = comm;
    MPI_Comm_rank(comm, &g.rank);
    MPI_Comm_size(comm, &g.size);

    int dims[2] = {0, 0};
    MPI_Dims_create(g.size, 2, dims);
    g.dims[0] = dims[0];
    g.dims[1] = dims[1];

    int periods[2] = {0, 0};
    MPI_Cart_create(comm, 2, g.dims, periods, /*reorder=*/1, &g.cart);
    MPI_Cart_coords(g.cart, g.rank, 2, g.coords);

    // row_comm: color = process row, key = process col
    MPI_Comm_split(g.cart, g.coords[0], g.coords[1], &g.row_comm);
    // col_comm: color = process col, key = process row
    MPI_Comm_split(g.cart, g.coords[1], g.coords[0], &g.col_comm);
    return g;
}

static void free_grid(Grid& g) {
    if (g.row_comm != MPI_COMM_NULL) MPI_Comm_free(&g.row_comm);
    if (g.col_comm != MPI_COMM_NULL) MPI_Comm_free(&g.col_comm);
    if (g.cart != MPI_COMM_NULL) MPI_Comm_free(&g.cart);
    g.row_comm = g.col_comm = g.cart = MPI_COMM_NULL;
}

static inline uint64_t tile_key(int ti, int tj) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(ti)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(tj));
}

struct Tile {
    int ti = 0;
    int tj = 0;
    int rows = 0;
    int cols = 0;
    std::vector<double> a; // row-major
};

static inline int tile_extent(int idx, int nb, int n) {
    const int start = idx * nb;
    const int rem = n - start;
    return rem > nb ? nb : rem;
}

static int owner_rank(const Grid& g, int ti, int tj) {
    int coords[2] = {ti % g.dims[0], tj % g.dims[1]};
    int r = 0;
    MPI_Cart_rank(g.cart, coords, &r);
    return r;
}

static bool local_potrf(Tile& Akk) {
    const int n = Akk.rows; // square tile
    double* A = Akk.a.data();

    for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int k = 0; k < j; ++k) {
            const double v = A[j * n + k];
            sum += v * v;
        }
        const double val = A[j * n + j] - sum;
        if (val <= 0.0) return false;
        const double dj = std::sqrt(val);
        A[j * n + j] = dj;

        for (int i = j + 1; i < n; ++i) {
            double s = 0.0;
            for (int k = 0; k < j; ++k) {
                s += A[i * n + k] * A[j * n + k];
            }
            A[i * n + j] = (A[i * n + j] - s) / dj;
        }

        // keep upper triangle clean
        for (int k = j + 1; k < n; ++k) {
            A[j * n + k] = 0.0;
        }
    }

    return true;
}

// Solve B := B * inv(L^T), where L is lower-triangular (n x n), B is (m x n).
static void local_trsm_right_lower_trans(double* B, int m, int n, const double* L) {
    for (int r = 0; r < m; ++r) {
        double* brow = B + r * n;
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            const double* Lj = L + j * n;
            for (int t = 0; t < j; ++t) {
                s += Lj[t] * brow[t];
            }
            brow[j] = (brow[j] - s) / Lj[j];
        }
    }
}

static void local_gemm_update(Tile& Aij,
                             const double* Lik, int mi, int kdim,
                             const double* Ljk, int mj) {
    // Aij(mi x mj) -= Lik(mi x kdim) * Ljk(mj x kdim)^T
    double* C = Aij.a.data();
    const int ldc = Aij.cols;

    for (int kk = 0; kk < kdim; ++kk) {
        for (int i = 0; i < mi; ++i) {
            const double aik = Lik[i * kdim + kk];
            double* Crow = C + i * ldc;
            const double* Bcol = Ljk + kk; // access Ljk[j * kdim + kk]
            for (int j = 0; j < mj; ++j) {
                Crow[j] -= aik * Bcol[j * kdim];
            }
        }
    }
}

static void scatter_lower_tiles(const Grid& g,
                               const std::vector<double>& A_root,
                               int n, int nb, int nt,
                               std::vector<Tile>& local_tiles,
                               std::unordered_map<uint64_t, size_t>& tile_index) {
    local_tiles.clear();
    tile_index.clear();

    // Pre-create local tiles so receives can go directly into place.
    for (int ti = 0; ti < nt; ++ti) {
        const int rows = tile_extent(ti, nb, n);
        for (int tj = 0; tj <= ti; ++tj) {
            const int cols = tile_extent(tj, nb, n);
            if (owner_rank(g, ti, tj) != g.rank) continue;
            Tile t;
            t.ti = ti;
            t.tj = tj;
            t.rows = rows;
            t.cols = cols;
            t.a.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
            tile_index.emplace(tile_key(ti, tj), local_tiles.size());
            local_tiles.emplace_back(std::move(t));
        }
    }

    if (g.rank == 0) {
        std::vector<double> buf;
        for (int ti = 0; ti < nt; ++ti) {
            const int rows = tile_extent(ti, nb, n);
            const int r0 = ti * nb;
            for (int tj = 0; tj <= ti; ++tj) {
                const int cols = tile_extent(tj, nb, n);
                const int c0 = tj * nb;
                buf.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0);

                for (int i = 0; i < rows; ++i) {
                    const double* src = &A_root[static_cast<size_t>(r0 + i) * n + c0];
                    double* dst = &buf[static_cast<size_t>(i) * cols];
                    std::memcpy(dst, src, static_cast<size_t>(cols) * sizeof(double));
                }

                const int owner = owner_rank(g, ti, tj);
                const int tag = ti * nt + tj;
                if (owner == 0) {
                    auto it = tile_index.find(tile_key(ti, tj));
                    std::memcpy(local_tiles[it->second].a.data(), buf.data(),
                                buf.size() * sizeof(double));
                } else {
                    MPI_Send(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, owner, tag, g.comm);
                }
            }
        }
    } else {
        for (auto& t : local_tiles) {
            const int tag = t.ti * nt + t.tj;
            MPI_Recv(t.a.data(), static_cast<int>(t.a.size()), MPI_DOUBLE, 0, tag, g.comm, MPI_STATUS_IGNORE);
        }
    }
}

static void gather_lower_tiles_root(const Grid& g,
                                   std::vector<double>& A_root,
                                   int n, int nb, int nt,
                                   const std::vector<Tile>& local_tiles,
                                   const std::unordered_map<uint64_t, size_t>& tile_index) {
    if (g.rank == 0) {
        std::fill(A_root.begin(), A_root.end(), 0.0);

        std::vector<double> buf;
        for (int ti = 0; ti < nt; ++ti) {
            const int rows = tile_extent(ti, nb, n);
            const int r0 = ti * nb;
            for (int tj = 0; tj <= ti; ++tj) {
                const int cols = tile_extent(tj, nb, n);
                const int c0 = tj * nb;
                const int owner = owner_rank(g, ti, tj);
                const int tag = ti * nt + tj;

                buf.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
                if (owner == 0) {
                    auto it = tile_index.find(tile_key(ti, tj));
                    std::memcpy(buf.data(), local_tiles[it->second].a.data(),
                                buf.size() * sizeof(double));
                } else {
                    MPI_Recv(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, owner, tag, g.comm, MPI_STATUS_IGNORE);
                }

                for (int i = 0; i < rows; ++i) {
                    double* dst = &A_root[static_cast<size_t>(r0 + i) * n + c0];
                    const double* src = &buf[static_cast<size_t>(i) * cols];
                    std::memcpy(dst, src, static_cast<size_t>(cols) * sizeof(double));
                }
            }
        }

        // Ensure exact lower-triangular semantics.
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                A_root[static_cast<size_t>(i) * n + j] = 0.0;
            }
        }
    } else {
        for (const auto& t : local_tiles) {
            const int tag = t.ti * nt + t.tj;
            MPI_Send(t.a.data(), static_cast<int>(t.a.size()), MPI_DOUBLE, 0, tag, g.comm);
        }
    }
}

static bool choleskyDecompositionMPI(std::vector<double>& A_root, const size_t n, const Grid& g, double* max_time_s) {
    const int nn = static_cast<int>(n);
    const int nb = std::min(128, nn);
    const int nt = (nn + nb - 1) / nb;

    std::vector<Tile> local_tiles;
    std::unordered_map<uint64_t, size_t> tile_index;

    scatter_lower_tiles(g, A_root, nn, nb, nt, local_tiles, tile_index);

    // Benchmark the factorization itself; distribution/gather are setup/teardown.
    MPI_Barrier(g.comm);
    const double t0 = MPI_Wtime();

    std::unordered_map<int, std::vector<double>> row_panel;
    std::unordered_map<int, std::vector<double>> col_panel;

    for (int k = 0; k < nt; ++k) {
        const int bk = tile_extent(k, nb, nn);

        // 1) Factor diagonal tile and broadcast it down process column k%pc.
        std::vector<double> Lkk;
        if (g.coords[1] == (k % g.dims[1])) {
            Lkk.resize(static_cast<size_t>(bk) * static_cast<size_t>(bk));

            int local_ok = 1;
            if (g.coords[0] == (k % g.dims[0])) {
                auto it = tile_index.find(tile_key(k, k));
                if (it == tile_index.end()) {
                    local_ok = 0;
                } else {
                    Tile& Akk = local_tiles[it->second];
                    local_ok = local_potrf(Akk) ? 1 : 0;
                    std::memcpy(Lkk.data(), Akk.a.data(), Lkk.size() * sizeof(double));
                }
            }

            int ok_all = 0;
            MPI_Allreduce(&local_ok, &ok_all, 1, MPI_INT, MPI_MIN, g.comm);
            if (!ok_all) return false;

            MPI_Bcast(Lkk.data(), static_cast<int>(Lkk.size()), MPI_DOUBLE, k % g.dims[0], g.col_comm);
        } else {
            int local_ok = 1;
            int ok_all = 0;
            MPI_Allreduce(&local_ok, &ok_all, 1, MPI_INT, MPI_MIN, g.comm);
            if (!ok_all) return false;
        }

        // 2) TRSM for tiles below the diagonal in column k (only in process column k%pc).
        if (g.coords[1] == (k % g.dims[1])) {
            for (int ti = k + 1; ti < nt; ++ti) {
                if ((ti % g.dims[0]) != g.coords[0]) continue;
                auto it = tile_index.find(tile_key(ti, k));
                if (it == tile_index.end()) continue;
                Tile& Aik = local_tiles[it->second];
                local_trsm_right_lower_trans(Aik.a.data(), Aik.rows, bk, Lkk.data());
            }
        }

        // 3) Broadcast panel tiles Lik along their process rows.
        row_panel.clear();
        for (int ti = k + 1; ti < nt; ++ti) {
            if ((ti % g.dims[0]) != g.coords[0]) continue; // only ranks in that process row participate

            const int mi = tile_extent(ti, nb, nn);
            std::vector<double> buf(static_cast<size_t>(mi) * static_cast<size_t>(bk));

            if (g.coords[1] == (k % g.dims[1])) {
                // root for this broadcast (col = k%pc) has the tile locally.
                auto it = tile_index.find(tile_key(ti, k));
                if (it != tile_index.end()) {
                    std::memcpy(buf.data(), local_tiles[it->second].a.data(), buf.size() * sizeof(double));
                }
            }

            MPI_Bcast(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, k % g.dims[1], g.row_comm);
            row_panel.emplace(ti, std::move(buf));
        }

        // 4) Broadcast panel tiles Ljk down their process columns.
        col_panel.clear();
        for (int tj = k + 1; tj < nt; ++tj) {
            if ((tj % g.dims[1]) != g.coords[1]) continue; // only ranks in that process column participate

            const int mj = tile_extent(tj, nb, nn);
            std::vector<double> buf(static_cast<size_t>(mj) * static_cast<size_t>(bk));

            if (g.coords[0] == (tj % g.dims[0])) {
                // root (row=tj%pr, col=tj%pc) uses the row-broadcasted copy from step 3.
                auto it = row_panel.find(tj);
                if (it != row_panel.end()) {
                    std::memcpy(buf.data(), it->second.data(), buf.size() * sizeof(double));
                }
            }

            MPI_Bcast(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, tj % g.dims[0], g.col_comm);
            col_panel.emplace(tj, std::move(buf));
        }

        // 5) Update trailing submatrix tiles owned by this rank.
        for (auto& t : local_tiles) {
            const int ti = t.ti;
            const int tj = t.tj;
            if (ti <= k || tj <= k) continue;
            if (ti < tj) continue;
            if ((ti % g.dims[0]) != g.coords[0]) continue;
            if ((tj % g.dims[1]) != g.coords[1]) continue;

            auto it_i = row_panel.find(ti);
            auto it_j = col_panel.find(tj);
            if (it_i == row_panel.end() || it_j == col_panel.end()) continue;

            const int mi = t.rows;
            const int mj = t.cols;
            local_gemm_update(t, it_i->second.data(), mi, bk, it_j->second.data(), mj);
        }

    }

    MPI_Barrier(g.comm);
    const double t1 = MPI_Wtime();

    double elapsed = t1 - t0;
    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, g.comm);
    if (max_time_s) *max_time_s = max_elapsed;

    gather_lower_tiles_root(g, A_root, nn, nb, nt, local_tiles, tile_index);

    return true;
}

} // namespace

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }

    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;

    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);

        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

    // Check if error is within tolerance
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    Grid g = make_grid(MPI_COMM_WORLD);

    size_t n = 512;
    int validate = 0;
    int printResults = 0;
    int parse_ok = 1;

    if (g.rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                parse_ok = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parse_ok = 0;
                break;
            }
        }
    }

    MPI_Bcast(&parse_ok, 1, MPI_INT, 0, g.comm);
    if (!parse_ok) {
        free_grid(g);
        MPI_Finalize();
        return 0;
    }

    // Broadcast parameters
    unsigned long long n_ull = static_cast<unsigned long long>(n);
    MPI_Bcast(&n_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, g.comm);
    n = static_cast<size_t>(n_ull);
    MPI_Bcast(&validate, 1, MPI_INT, 0, g.comm);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, g.comm);

    if (g.rank == 0) {
        printf("Cholesky Decomposition Benchmark (MPI)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d (grid %d x %d)\n", g.size, g.dims[0], g.dims[1]);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");

        printf("Generating positive definite matrix...\n");
    }

    std::vector<double> A;
    std::vector<double> A_orig;

    if (g.rank == 0) {
        A.resize(n * n);
        generatePositiveDefiniteMatrix(A, n);
        if (validate) A_orig = A;
    }

    if (g.rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }

    double max_factor_s = 0.0;
    bool success = choleskyDecompositionMPI(A, n, g, &max_factor_s);

    int ok = success ? 1 : 0;
    int ok_all = 0;
    MPI_Allreduce(&ok, &ok_all, 1, MPI_INT, MPI_MIN, g.comm);
    if (!ok_all) {
        if (g.rank == 0) printf("Cholesky decomposition failed\n");
        free_grid(g);
        MPI_Finalize();
        return 1;
    }

    if (g.rank == 0) {
        const long ms = static_cast<long>(max_factor_s * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        // Calculate GFLOPS (approximately n^3/3 operations for Cholesky)
        const double ops = (double)n * (double)n * (double)n / 3.0;
        const double secs = max_factor_s > 0.0 ? max_factor_s : 1e-12;
        const double gflops = ops / secs / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(A, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(A, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                free_grid(g);
                MPI_Finalize();
                return 1;
            }
        }
    }

    free_grid(g);
    MPI_Finalize();
    return 0;
}
