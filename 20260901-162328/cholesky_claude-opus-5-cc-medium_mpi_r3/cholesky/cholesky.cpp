#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Cholesky decomposition, distributed with MPI (left-looking, panel blocked).
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// Parallelization / optimization strategy:
//   * The matrix rows are distributed block-cyclically over the ranks in panels
//     of NB consecutive rows, so a whole panel is owned by a single rank.
//   * Each rank stores its rows in column-major order.  The update of the
//     trailing rows for one panel is then a GEMM-like operation (register
//     blocked, vectorized over the rows), while each individual matrix element
//     is still the result of one dot product accumulated in ascending k order
//     by a single rank.  The floating point operations per element and their
//     order are therefore identical to the sequential reference algorithm.
//   * Per panel there is exactly one broadcast (the pb x j1 panel block), i.e.
//     n/NB messages instead of one per column.
//   * Look-ahead: after receiving panel p, the owner of panel p+1 first updates
//     its own rows and factors panel p+1, starts the non-blocking broadcast and
//     only then updates its remaining rows, overlapping communication with
//     computation.

namespace {

// Panel width (rows per distribution block / columns per update step).
constexpr size_t NB = 24;
// Register blocking of the trailing update: RB rows x CB columns.
constexpr size_t RB = 8;
constexpr size_t CB = 4;

int g_rank = 0;
int g_nranks = 1;

inline size_t panelsOf(const size_t n) { return (n + NB - 1) / NB; }
inline int ownerOfPanel(const size_t p) { return (int)(p % (size_t)g_nranks); }

// Number of panels owned by this rank when there are np panels in total.
inline size_t localPanelCount(const size_t np, const int rank) {
    if ((size_t)rank >= np) return 0;
    return (np - (size_t)rank + (size_t)g_nranks - 1) / (size_t)g_nranks;
}

// Local row index of the first owned row belonging to a panel > p.
inline size_t localRowAfterPanel(const size_t p) {
    const size_t r = (size_t)g_rank;
    const size_t owned = (p < r) ? 0 : ((p - r) / (size_t)g_nranks + 1);
    return owned * NB;
}

// Global row index of local row l.
inline size_t globalRow(const size_t l) {
    return (l / NB * (size_t)g_nranks + (size_t)g_rank) * NB + l % NB;
}

// SIMD vector of VL doubles (plain compiler vector extension, no intrinsics).
constexpr size_t VL = 4;
typedef double vec __attribute__((vector_size(sizeof(double) * VL)));

inline vec vload(const double* p) {
    vec v;
    __builtin_memcpy(&v, p, sizeof(vec));
    return v;
}
inline void vstore(double* p, const vec v) { __builtin_memcpy(p, &v, sizeof(vec)); }
inline vec vsplat(const double x) { return vec{x, x, x, x}; }

// acc[c * RB + r] = sum_{k < j0} Aloc[k][lo + r] * R[c][k]   (k ascending)
// Fast path: exactly RB rows and CB columns, accumulators held in registers.
inline void gemmTileFast(const double* __restrict A, const size_t ldl, const size_t j0,
    const double* __restrict R, const size_t lenR, double* __restrict acc) {
    constexpr size_t NV = RB / VL;
    vec t[CB][NV];
    for (size_t c = 0; c < CB; ++c) {
        for (size_t h = 0; h < NV; ++h) {
            t[c][h] = vsplat(0.0);
        }
    }
    for (size_t k = 0; k < j0; ++k) {
        const double* __restrict a = A + k * ldl;
        vec av[NV];
        for (size_t h = 0; h < NV; ++h) {
            av[h] = vload(a + h * VL);
        }
        for (size_t c = 0; c < CB; ++c) {
            const vec rv = vsplat(R[c * lenR + k]);
            for (size_t h = 0; h < NV; ++h) {
                t[c][h] += av[h] * rv;
            }
        }
    }
    for (size_t c = 0; c < CB; ++c) {
        for (size_t h = 0; h < NV; ++h) {
            vstore(acc + c * RB + h * VL, t[c][h]);
        }
    }
}

// Generic variant for the remainder tiles.
inline void gemmTileGeneric(const double* __restrict A, const size_t ldl, const size_t j0,
    const double* __restrict R, const size_t lenR, double* __restrict acc, const size_t nr, const size_t nc) {
    for (size_t c = 0; c < nc; ++c) {
        for (size_t r = 0; r < nr; ++r) {
            acc[c * RB + r] = 0.0;
        }
    }
    for (size_t k = 0; k < j0; ++k) {
        const double* __restrict a = A + k * ldl;
        for (size_t c = 0; c < nc; ++c) {
            const double rv = R[c * lenR + k];
            for (size_t r = 0; r < nr; ++r) {
                acc[c * RB + r] += a[r] * rv;
            }
        }
    }
}

// Update the local rows [lo, hi) (all of them below the panel) with the columns
// [j0, j0 + pb) of the panel block R (pb rows of length lenR, row-major).
void updateTrailingRows(double* __restrict Aloc, const size_t ldl, const size_t lo, const size_t hi,
    const double* __restrict R, const size_t lenR, const size_t j0, const size_t pb, double* __restrict pack) {
    double acc[NB * RB];

    for (size_t l0 = lo; l0 < hi; l0 += RB) {
        const size_t nr = std::min(RB, hi - l0);
        const double* __restrict A = Aloc + l0;

        // Phase 1: contributions of the already finished columns k < j0.
        if (j0 > 0) {
            // Pack the RB x j0 slice of the local matrix contiguously; it is
            // read once per column group, and packing turns the ldl-strided
            // access into a single streaming pass.
            for (size_t k = 0; k < j0; ++k) {
                const double* __restrict src = A + k * ldl;
                double* __restrict dst = pack + k * RB;
                for (size_t r = 0; r < nr; ++r) {
                    dst[r] = src[r];
                }
            }
            for (size_t c0 = 0; c0 < pb; c0 += CB) {
                const size_t nc = std::min(CB, pb - c0);
                const double* __restrict Rc = R + c0 * lenR;
                if (nr == RB && nc == CB) {
                    gemmTileFast(pack, RB, j0, Rc, lenR, acc + c0 * RB);
                } else {
                    gemmTileGeneric(pack, RB, j0, Rc, lenR, acc + c0 * RB, nr, nc);
                }
            }
        } else {
            for (size_t c = 0; c < pb; ++c) {
                for (size_t r = 0; r < nr; ++r) {
                    acc[c * RB + r] = 0.0;
                }
            }
        }

        // Phase 2: the columns inside the panel, in ascending order.  Column
        // j0 + jj continues the very same accumulator with k = j0 .. j0+jj-1.
        // The freshly computed columns are kept in a small local buffer so that
        // this phase works entirely out of L1.
        double res[NB * RB];
        for (size_t jj = 0; jj < pb; ++jj) {
            const double* __restrict Rj = R + jj * lenR;
            double* __restrict a = acc + jj * RB;
            for (size_t k = 0; k < jj; ++k) {
                const double* __restrict col = res + k * RB;
                const double rv = Rj[j0 + k];
                for (size_t r = 0; r < nr; ++r) {
                    a[r] += col[r] * rv;
                }
            }
            const double d = Rj[j0 + jj];
            const double* __restrict colj = Aloc + (j0 + jj) * ldl + l0;
            double* __restrict out = res + jj * RB;
            for (size_t r = 0; r < nr; ++r) {
                out[r] = (colj[r] - a[r]) / d;
            }
        }
        for (size_t jj = 0; jj < pb; ++jj) {
            const double* __restrict src = res + jj * RB;
            double* __restrict colj = Aloc + (j0 + jj) * ldl + l0;
            for (size_t r = 0; r < nr; ++r) {
                colj[r] = src[r];
            }
        }
    }
}

// Factor the diagonal block of the panel starting at global row j0 (pb rows,
// local rows [lq, lq + pb)) and pack the finished panel rows (columns
// 0 .. j0+pb-1) into R.  Returns 1.0 on success and -(failing column + 1) if
// the matrix turned out not to be positive definite.
double factorPanel(double* __restrict Aloc, const size_t ldl, const size_t lq, const size_t j0, const size_t pb,
    double* __restrict R, const size_t lenR) {
    // Gather the already computed part of the panel rows.  Done as a blocked
    // transpose: the source is read in contiguous chunks of pb doubles and the
    // destination is written in contiguous runs of KT doubles.
    constexpr size_t KT = 64;
    double tmp[KT * NB];
    for (size_t k0 = 0; k0 < j0; k0 += KT) {
        const size_t kn = std::min(KT, j0 - k0);
        for (size_t kk = 0; kk < kn; ++kk) {
            const double* __restrict src = Aloc + (k0 + kk) * ldl + lq;
            double* __restrict dst = tmp + kk * NB;
            for (size_t jj = 0; jj < pb; ++jj) {
                dst[jj] = src[jj];
            }
        }
        for (size_t jj = 0; jj < pb; ++jj) {
            double* __restrict Rj = R + jj * lenR + k0;
            for (size_t kk = 0; kk < kn; ++kk) {
                Rj[kk] = tmp[kk * NB + jj];
            }
        }
    }

    for (size_t jj = 0; jj < pb; ++jj) {
        const size_t j = j0 + jj;
        const double* __restrict Rj = R + jj * lenR;

        // Diagonal element
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            sum += Rj[k] * Rj[k];
        }
        const double val = Aloc[j * ldl + lq + jj] - sum;
        if (val <= 0.0) {
            return -(double)(j + 1);
        }
        const double d = sqrt(val);
        Aloc[j * ldl + lq + jj] = d;
        R[jj * lenR + j] = d;

        // Remaining rows of the panel, column j.  Four rows at a time so that
        // the (strictly ordered) dot products can overlap in the pipeline.
        constexpr size_t IB = 4;
        size_t ii = jj + 1;
        for (; ii + IB <= pb; ii += IB) {
            double s[IB] = {};
            for (size_t k = 0; k < j; ++k) {
                const double rv = Rj[k];
                for (size_t b = 0; b < IB; ++b) {
                    s[b] += R[(ii + b) * lenR + k] * rv;
                }
            }
            for (size_t b = 0; b < IB; ++b) {
                const double v = (Aloc[j * ldl + lq + ii + b] - s[b]) / d;
                Aloc[j * ldl + lq + ii + b] = v;
                R[(ii + b) * lenR + j] = v;
            }
        }
        for (; ii < pb; ++ii) {
            double* __restrict Ri = R + ii * lenR;
            double s = 0.0;
            for (size_t k = 0; k < j; ++k) {
                s += Ri[k] * Rj[k];
            }
            const double v = (Aloc[j * ldl + lq + ii] - s) / d;
            Aloc[j * ldl + lq + ii] = v;
            Ri[j] = v;
        }
    }
    return 1.0;
}

} // namespace

// Distributed Cholesky decomposition.  Aloc holds the locally owned rows in
// column-major layout (leading dimension ldl, nloc valid rows).
bool choleskyDecomposition(std::vector<double>& Aloc, const size_t n, const size_t ldl, const size_t nloc) {
    if (n == 0) return true;

    // The upper triangular part is never read by the algorithm, so it can be
    // zeroed up front (the sequential version zeroes it row by row).
    for (size_t l = 0; l < nloc; ++l) {
        const size_t i = globalRow(l);
        for (size_t j = i + 1; j < n; ++j) {
            Aloc[j * ldl + l] = 0.0;
        }
    }

    const size_t np = panelsOf(n);

    // Panel broadcast buffers: pb rows of length j1, plus a trailing status flag.
    std::vector<double> bufA(NB * n + 1), bufB(NB * n + 1);
    std::vector<double> pack(RB * n);
    double* cur = bufA.data();
    double* next = bufB.data();

    MPI_Request req = MPI_REQUEST_NULL;

    // Start the pipeline with panel 0.
    {
        const size_t pb = std::min(NB, n);
        const int root = ownerOfPanel(0);
        if (g_rank == root) {
            cur[pb * pb] = factorPanel(Aloc.data(), ldl, 0, 0, pb, cur, pb);
        }
        MPI_Ibcast(cur, (int)(pb * pb + 1), MPI_DOUBLE, root, MPI_COMM_WORLD, &req);
    }

    bool ok = true;
    for (size_t p = 0; p < np; ++p) {
        const size_t j0 = p * NB;
        const size_t pb = std::min(NB, n - j0);
        const size_t lenR = j0 + pb;

        MPI_Wait(&req, MPI_STATUS_IGNORE);
        if (cur[pb * lenR] != 1.0) {
            if (g_rank == 0) {
                const size_t bad = (size_t)(-cur[pb * lenR]) - 1;
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", bad);
            }
            ok = false;
            break;
        }

        size_t lo = localRowAfterPanel(p);

        if (p + 1 < np) {
            const size_t j0n = j0 + pb;
            const size_t pbn = std::min(NB, n - j0n);
            const size_t lenRn = j0n + pbn;
            const int root = ownerOfPanel(p + 1);
            if (g_rank == root) {
                // Look-ahead: bring the next panel's own rows up to date and
                // factor them, so that its broadcast overlaps with the bulk of
                // the trailing update below.
                updateTrailingRows(Aloc.data(), ldl, lo, lo + pbn, cur, lenR, j0, pb, pack.data());
                next[pbn * lenRn] = factorPanel(Aloc.data(), ldl, lo, j0n, pbn, next, lenRn);
                lo += pbn;
            }
            MPI_Ibcast(next, (int)(pbn * lenRn + 1), MPI_DOUBLE, root, MPI_COMM_WORLD, &req);
        }

        updateTrailingRows(Aloc.data(), ldl, lo, nloc, cur, lenR, j0, pb, pack.data());

        std::swap(cur, next);
    }

    if (!ok && req != MPI_REQUEST_NULL) {
        MPI_Cancel(&req);
        MPI_Request_free(&req);
    }
    return ok;
}

// Generate the locally owned rows of a symmetric positive definite matrix.
void generatePositiveDefiniteMatrix(std::vector<double>& Aloc, const size_t n, const size_t ldl, const size_t nloc) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (identical on every rank, so no communication)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute the locally owned rows of A = B * B^T.  The dot products are
    // computed in tiles of GI local rows x GJ columns with the accumulators
    // held in registers; every single sum is still accumulated over ascending
    // k, exactly as in the sequential reference.
    constexpr size_t GI = 4, GJ = 4;
    for (size_t j0 = 0; j0 < n; j0 += GJ) {
        const size_t nj = std::min(GJ, n - j0);
        const double* Bj[GJ];
        for (size_t b = 0; b < nj; ++b) {
            Bj[b] = B.data() + (j0 + b) * n;
        }
        for (size_t l0 = 0; l0 < nloc; l0 += GI) {
            const size_t ni = std::min(GI, nloc - l0);
            const double* Bi[GI];
            for (size_t a = 0; a < ni; ++a) {
                Bi[a] = B.data() + globalRow(l0 + a) * n;
            }
            double t[GI][GJ] = {};
            if (ni == GI && nj == GJ) {
                for (size_t k = 0; k < n; ++k) {
                    for (size_t a = 0; a < GI; ++a) {
                        const double va = Bi[a][k];
                        for (size_t b = 0; b < GJ; ++b) {
                            t[a][b] += va * Bj[b][k];
                        }
                    }
                }
            } else {
                for (size_t k = 0; k < n; ++k) {
                    for (size_t a = 0; a < ni; ++a) {
                        const double va = Bi[a][k];
                        for (size_t b = 0; b < nj; ++b) {
                            t[a][b] += va * Bj[b][k];
                        }
                    }
                }
            }
            for (size_t b = 0; b < nj; ++b) {
                for (size_t a = 0; a < ni; ++a) {
                    Aloc[(j0 + b) * ldl + l0 + a] = t[a][b];
                }
            }
        }
    }

    // Add diagonal dominance to ensure positive definiteness
    for (size_t l = 0; l < nloc; ++l) {
        Aloc[globalRow(l) * ldl + l] += (double)n;
    }
}

// Collect the distributed matrix into a full row-major copy on every rank.
void gatherFullMatrix(const std::vector<double>& Aloc, std::vector<double>& full, const size_t n, const size_t ldl) {
    std::vector<double> recv((size_t)g_nranks * ldl * n);
    MPI_Allgather(Aloc.data(), (int)(ldl * n), MPI_DOUBLE, recv.data(), (int)(ldl * n), MPI_DOUBLE, MPI_COMM_WORLD);

    const size_t np = panelsOf(n);
    full.resize(n * n);
    for (int r = 0; r < g_nranks; ++r) {
        const double* block = recv.data() + (size_t)r * ldl * n;
        const size_t lp = localPanelCount(np, r);
        for (size_t q = 0; q < lp; ++q) {
            const size_t p = q * (size_t)g_nranks + (size_t)r;
            const size_t rows = std::min(NB, n - p * NB);
            for (size_t off = 0; off < rows; ++off) {
                const size_t i = p * NB + off;
                const size_t l = q * NB + off;
                double* dst = full.data() + i * n;
                for (size_t j = 0; j < n; ++j) {
                    dst[j] = block[j * ldl + l];
                }
            }
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig_loc, const size_t n,
    const size_t ldl, const size_t nloc) {
    // Validate by computing L * L^T and comparing with original matrix.
    // Each rank checks the rows it owns; L is available in full on every rank.

    double maxError = 0.0;
    double relError = 0.0;

    for (size_t l = 0; l < nloc; ++l) {
        const size_t i = globalRow(l);
        const double* __restrict Li = L.data() + i * n;
        for (size_t j = 0; j < n; ++j) {
            const double* __restrict Lj = L.data() + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += Li[k] * Lj[k];
            }
            const double orig = A_orig_loc[j * ldl + l];
            const double error = fabs(sum - orig);
            maxError = std::max(maxError, error);

            const double rel = error / (fabs(orig) + 1e-10);
            relError = std::max(relError, rel);
        }
    }

    double errors[2] = {maxError, relError};
    MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    maxError = errors[0];
    relError = errors[1];

    if (g_rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    // Check if error is within tolerance
    if (relError > 1e-6) {
        if (g_rank == 0) {
            printf("Validation failed: relative error too large\n");
        }
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
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nranks);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (g_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (g_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (g_rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d\n", g_nranks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Block-cyclic row distribution: panel p (rows [p*NB, (p+1)*NB)) is owned
    // by rank p % nranks.
    const size_t np = panelsOf(n);
    const size_t maxLocalPanels = std::max<size_t>(1, (np + (size_t)g_nranks - 1) / (size_t)g_nranks);
    const size_t ldl = maxLocalPanels * NB;
    size_t nloc = 0;
    for (size_t q = 0; q < localPanelCount(np, g_rank); ++q) {
        const size_t p = q * (size_t)g_nranks + (size_t)g_rank;
        nloc += std::min(NB, n - p * NB);
    }

    // Allocate the local part of the matrix (column-major, ldl x n)
    std::vector<double> Aloc(ldl * n);
    std::vector<double> A_orig_loc;

    // Generate positive definite matrix
    if (g_rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(Aloc, n, ldl, nloc);

    if (validate) {
        A_orig_loc = Aloc; // Save original for validation
    }

    // Perform Cholesky decomposition
    if (g_rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(Aloc, n, ldl, nloc);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (g_rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    long elapsed = (long)duration.count();
    MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (g_rank == 0) {
        printf("Computation time: %ld ms\n", elapsed);

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (elapsed / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> full;
    if (printResults || validate) {
        gatherFullMatrix(Aloc, full, n, ldl);
    }

    // Print results for external validation
    if (printResults && g_rank == 0) {
        print_results(full, "CholeskyL");
    }

    // Validation
    if (validate) {
        if (g_rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateCholesky(full, A_orig_loc, n, ldl, nloc);

        if (valid) {
            if (g_rank == 0) {
                printf("Validation: PASSED\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (g_rank == 0) {
                printf("Validation: FAILED\n");
            }
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
