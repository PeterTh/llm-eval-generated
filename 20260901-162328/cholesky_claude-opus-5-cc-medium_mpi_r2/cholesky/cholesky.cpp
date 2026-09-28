#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed-memory (MPI) Cholesky decomposition.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// Parallelization strategy
// ------------------------
// The matrix is distributed over the ranks in a one-dimensional block-cyclic
// fashion by rows: row block m (rows [m*NB, (m+1)*NB)) lives on rank m % nprocs.
// Only the rows owned by a rank are stored locally, so memory scales with 1/P.
//
// The factorization is a right-looking blocked algorithm over column blocks:
//   1. the owner of the diagonal block factorizes it,
//   2. the resulting "panel" (pivot rows of that block, columns 0..j0+kb) is
//      broadcast to all ranks (in transposed layout, see below),
//   3. every rank computes the block columns of all of its rows below the panel.
// A one-step lookahead is used: the owner of the *next* panel updates its own
// pivot rows and factorizes the next diagonal block first, then starts the
// non-blocking broadcast of that panel, and only afterwards updates the rest of
// its rows. This overlaps the panel factorization/communication of step s+1 with
// the trailing update of step s, keeping the critical path short.
//
// Numerics: the results are bit-for-bit identical to the sequential code.
// Element (i, j) is still computed as (A[i][j] - sum_{k<j} L[i][k]*L[j][k]) with
// the sum accumulated in a single accumulator over strictly increasing k. The
// micro kernels therefore only vectorize across *columns* j (one accumulator per
// column, k iterated sequentially), never across the reduction dimension k, and
// no partial sums are ever combined. The panel is broadcast transposed so that
// the vectorized column dimension is contiguous in memory. Because the block
// width is a multiple of four, splitting a dot product into "columns < j0" plus
// a tail does not disturb the grouping either.
//
// The exact rounding sequence of every reduction is reproduced by `dotStep4()` /
// the remainder loops: the reduction accumulates four separately rounded
// products at a time and then adds the last (length % 4) products with a fused
// multiply-add - which is precisely what the compiler generates for the plain
// `sum += a[k] * b[k]` loops of the sequential version (unroll by four with
// vmulpd, sequential vaddsd, fma remainder). The file is compiled with
// -ffp-contract=off so that this is not perturbed by additional contraction.

// Column block width (and row block height). Must be a multiple of four so that
// the reductions can be split at block boundaries without changing their
// grouping, see the note on numerics above.
static constexpr size_t NB = 16;

// Micro kernel register blocking: MR rows x NR columns, and the cache blocking
// factor for the reduction dimension (a multiple of four).
static constexpr size_t MR = 4;
static constexpr size_t NR = 16;
static constexpr size_t KC = 1024;

namespace {

struct Dist {
    size_t n = 0;
    size_t nb = 0;       // row/column block size
    size_t nbr = 0;      // number of block rows
    int rank = 0;
    int nprocs = 1;
    size_t localBlocks = 0;
    std::vector<double> A; // localBlocks * nb rows of n columns

    int owner(const size_t m) const { return static_cast<int>(m % static_cast<size_t>(nprocs)); }
    bool mine(const size_t m) const { return owner(m) == rank; }
    size_t rowsInBlock(const size_t m) const { return std::min(nb, n - m * nb); }
    // Pointer to the first row of block m (only valid if mine(m)).
    double* block(const size_t m) { return A.data() + (m / static_cast<size_t>(nprocs)) * nb * n; }
    const double* block(const size_t m) const {
        return A.data() + (m / static_cast<size_t>(nprocs)) * nb * n;
    }
};

// One group of four terms of a scalar reduction: four separately rounded
// products, added to the accumulator one after the other (see the note on
// numerics above). `b` is strided, `a` is contiguous.
inline double dotStep4(double s, const double* __restrict a, const double* __restrict b,
                       const size_t ldb) {
    const double p0 = a[0] * b[0];
    const double p1 = a[1] * b[ldb];
    const double p2 = a[2] * b[2 * ldb];
    const double p3 = a[3] * b[3 * ldb];
    s = s + p0;
    s = s + p1;
    s = s + p2;
    s = s + p3;
    return s;
}

// Phase A: partial dot products over the columns k < j0 (the part of the
// reduction that is common to all columns of the current block), for the
// columns [jc0, colLimit) of mr rows and the reduction range [kc, kend).
// part[m * kb + j] accumulates sum_{k<j0} rows[m][k] * L[j0+j][k]; `first`
// selects whether the accumulators start at zero or continue a previous chunk.
// Requires j0, kc and kend to be multiples of four (the block size is, and so
// is the k blocking factor), so that the grouping of the reduction is the same
// as in an unsplit loop.
void phaseA(const double* __restrict rows, const size_t mr, const size_t ldr,
            const double* __restrict pT, const size_t kb, const size_t kc, const size_t kend,
            const size_t colLimit, double* __restrict part, const bool first) {
    for (size_t jc = 0; jc < colLimit; jc += NR) {
        const size_t nr = std::min(NR, colLimit - jc);
        double acc[MR][NR] = {};

        if (!first) {
            for (size_t m = 0; m < mr; ++m) {
                for (size_t v = 0; v < nr; ++v) {
                    acc[m][v] = part[m * kb + jc + v];
                }
            }
        }

        if (mr == MR && nr == NR) {
            // Fully unrolled tile: MR * NR independent accumulators, the
            // reduction over k stays strictly sequential in each of them.
            for (size_t k = kc; k < kend; k += 4) {
                const double* __restrict pv = pT + k * kb + jc;
                for (size_t m = 0; m < MR; ++m) {
                    const double* __restrict r = rows + m * ldr + k;
                    const double a0 = r[0], a1 = r[1], a2 = r[2], a3 = r[3];
                    for (size_t v = 0; v < NR; ++v) {
                        double s = acc[m][v];
                        s = s + a0 * pv[v];
                        s = s + a1 * pv[kb + v];
                        s = s + a2 * pv[2 * kb + v];
                        s = s + a3 * pv[3 * kb + v];
                        acc[m][v] = s;
                    }
                }
            }
        } else {
            for (size_t k = kc; k < kend; k += 4) {
                const double* __restrict pv = pT + k * kb + jc;
                for (size_t m = 0; m < mr; ++m) {
                    const double* __restrict r = rows + m * ldr + k;
                    const double a0 = r[0], a1 = r[1], a2 = r[2], a3 = r[3];
                    for (size_t v = 0; v < nr; ++v) {
                        double s = acc[m][v];
                        s = s + a0 * pv[v];
                        s = s + a1 * pv[kb + v];
                        s = s + a2 * pv[2 * kb + v];
                        s = s + a3 * pv[3 * kb + v];
                        acc[m][v] = s;
                    }
                }
            }
        }

        for (size_t m = 0; m < mr; ++m) {
            for (size_t v = 0; v < nr; ++v) {
                part[m * kb + jc + v] = acc[m][v];
            }
        }
    }
}

// Computes the columns [j0, j0+kb) of nrows consecutive rows.
//
// diag == false: `rows` are rows strictly below the panel: pT[k * kb + j] ==
//   L[j0+j][k] (transposed) for the columns k < j0, followed by the diagonal
//   block pT[(j0 + j) * kb + k] == L[j0+j][j0+k] in row major order.
// diag == true: `rows` are the kb pivot rows themselves (global row j0 + i) and
//   `pT` only holds the columns k < j0 of those rows; column j is computed for
//   j <= i only, with j == i being the diagonal element.
// `part` is scratch space for nrows * kb accumulators.
bool computeBlockCols(double* __restrict rows, const size_t nrows, const size_t ldr,
                      const double* __restrict pT, const size_t kb, const size_t j0,
                      const bool diag, double* __restrict part, size_t* failIdx) {
    // Phase A, blocked over the reduction dimension so that the panel slice and
    // the row slice both stay in cache while all row tiles are swept.
    if (j0 == 0) {
        memset(part, 0, nrows * kb * sizeof(double));
    }
    for (size_t kc = 0; kc < j0; kc += KC) {
        const size_t kend = std::min(kc + KC, j0);
        for (size_t i0 = 0; i0 < nrows; i0 += MR) {
            const size_t mr = std::min(MR, nrows - i0);
            const size_t colLimit = diag ? std::min(kb, i0 + mr) : kb;
            phaseA(rows + i0 * ldr, mr, ldr, pT, kb, kc, kend, colLimit, part + i0 * kb, kc == 0);
        }
    }

    // Phase B: finish the reduction with the (triangular) tail k in
    // [j0, j0+j) and divide by the pivot.
    for (size_t i0 = 0; i0 < nrows; i0 += MR) {
        const size_t mr = std::min(MR, nrows - i0);
        double* rblk = rows + i0 * ldr;
        for (size_t m = 0; m < mr; ++m) {
            double* __restrict row = rblk + m * ldr;
            const size_t jmax = diag ? (i0 + m + 1) : kb;

            for (size_t j = 0; j < jmax; ++j) {
                double s = part[(i0 + m) * kb + j];

                if (diag) {
                    const double* __restrict pj = rows + j * ldr + j0;
                    const double* __restrict ri = row + j0;
                    size_t k = 0;
                    for (; k + 4 <= j; k += 4) {
                        s = dotStep4(s, ri + k, pj + k, 1);
                    }
                    for (; k < j; ++k) {
                        s = std::fma(ri[k], pj[k], s);
                    }
                    if (j == i0 + m) {
                        const double val = row[j0 + j] - s;
                        if (val <= 0.0) {
                            *failIdx = j0 + j;
                            return false;
                        }
                        row[j0 + j] = sqrt(val);
                    } else {
                        row[j0 + j] = (row[j0 + j] - s) / pj[j];
                    }
                } else {
                    const double* __restrict pj = pT + (j0 + j) * kb;
                    const double* __restrict ri = row + j0;
                    size_t k = 0;
                    for (; k + 4 <= j; k += 4) {
                        s = dotStep4(s, ri + k, pj + k, 1);
                    }
                    for (; k < j; ++k) {
                        s = std::fma(ri[k], pj[k], s);
                    }
                    row[j0 + j] = (row[j0 + j] - s) / pj[j];
                }
            }
        }
    }

    return true;
}

// Factorizes diagonal block s and packs the transposed panel into `buf`.
// Returns false if the matrix turns out not to be positive definite.
bool preparePanel(Dist& d, const size_t s, double* buf, double* part) {
    const size_t j0 = s * d.nb;
    const size_t kb = d.rowsInBlock(s);
    double* rows = d.block(s);

    // Transposed prefix: buf[k * kb + j] = L[j0+j][k] for k < j0.
    for (size_t j = 0; j < kb; ++j) {
        const double* __restrict src = rows + j * d.n;
        double* __restrict dst = buf + j;
        for (size_t k = 0; k < j0; ++k) {
            dst[k * kb] = src[k];
        }
    }

    size_t failIdx = 0;
    if (!computeBlockCols(rows, kb, d.n, buf, kb, j0, true, part, &failIdx)) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", failIdx);
        return false;
    }

    // Diagonal block, row major: buf[j0 * kb + j * kb + k] = L[j0+j][j0+k].
    // Phase B walks it along k, so unlike the prefix it is not transposed.
    for (size_t j = 0; j < kb; ++j) {
        double* __restrict dst = buf + (j0 + j) * kb;
        for (size_t k = 0; k < kb; ++k) {
            dst[k] = (k <= j) ? rows[j * d.n + j0 + k] : 0.0;
        }
    }

    return true;
}

} // namespace

bool choleskyDecomposition(Dist& d) {
    const size_t n = d.n;
    const size_t nb = d.nb;
    const size_t nbr = d.nbr;
    if (nbr == 0) {
        return true;
    }

    // Panel buffers (double buffered for the lookahead). The last element
    // carries the "not positive definite" flag.
    const size_t maxPanel = n * nb + 1;
    std::vector<double> panel[2] = {std::vector<double>(maxPanel), std::vector<double>(maxPanel)};
    std::vector<double> part(nb * nb); // phase A accumulators of one row block
    MPI_Request req[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    auto panelSize = [&](const size_t s) { return (s * nb + d.rowsInBlock(s)) * d.rowsInBlock(s); };

    bool failed = false;

    // Factorize and broadcast the first panel.
    {
        double* buf = panel[0].data();
        if (d.mine(0)) {
            const bool ok = preparePanel(d, 0, buf, part.data());
            buf[panelSize(0)] = ok ? 0.0 : 1.0;
        }
        MPI_Ibcast(buf, static_cast<int>(panelSize(0) + 1), MPI_DOUBLE, d.owner(0), MPI_COMM_WORLD,
                   &req[0]);
    }

    for (size_t s = 0; s < nbr; ++s) {
        const size_t cur = s % 2;
        MPI_Wait(&req[cur], MPI_STATUS_IGNORE);
        const double* __restrict pT = panel[cur].data();
        if (pT[panelSize(s)] != 0.0) {
            failed = true;
            break;
        }

        const size_t j0 = s * nb;
        const size_t kb = d.rowsInBlock(s);
        const bool hasNext = (s + 1 < nbr);
        const bool ownNext = hasNext && d.mine(s + 1);

        // Lookahead: bring the next panel forward so its factorization and
        // broadcast overlap with the bulk of this step's trailing update.
        if (ownNext) {
            size_t failIdx = 0;
            computeBlockCols(d.block(s + 1), d.rowsInBlock(s + 1), n, pT, kb, j0, false,
                             part.data(), &failIdx);
            double* buf = panel[(s + 1) % 2].data();
            const bool ok = preparePanel(d, s + 1, buf, part.data());
            buf[panelSize(s + 1)] = ok ? 0.0 : 1.0;
        }
        if (hasNext) {
            MPI_Ibcast(panel[(s + 1) % 2].data(), static_cast<int>(panelSize(s + 1) + 1),
                       MPI_DOUBLE, d.owner(s + 1), MPI_COMM_WORLD, &req[(s + 1) % 2]);
        }

        // Trailing update of all remaining locally owned row blocks.
        const size_t skip = ownNext ? s + 1 : nbr;
        for (size_t m = static_cast<size_t>(d.rank); m < nbr; m += static_cast<size_t>(d.nprocs)) {
            if (m <= s || m == skip) {
                continue;
            }
            size_t failIdx = 0;
            computeBlockCols(d.block(m), d.rowsInBlock(m), n, pT, kb, j0, false, part.data(),
                             &failIdx);
            // Give the MPI library a chance to make progress on the broadcast
            // of the next panel while there is still work left to do.
            if (hasNext) {
                int flag = 0;
                MPI_Test(&req[(s + 1) % 2], &flag, MPI_STATUS_IGNORE);
            }
        }
    }

    if (failed) {
        return false;
    }

    // Zero out the upper triangular part of the locally owned rows.
    for (size_t m = d.rank; m < nbr; m += static_cast<size_t>(d.nprocs)) {
        double* rows = d.block(m);
        const size_t rb = d.rowsInBlock(m);
        for (size_t ii = 0; ii < rb; ++ii) {
            const size_t i = m * nb + ii;
            for (size_t j = i + 1; j < n; ++j) {
                rows[ii * n + j] = 0.0;
            }
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix (distributed).
void generatePositiveDefiniteMatrix(Dist& d, std::vector<double>& Bloc) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    const size_t n = d.n;
    const size_t nb = d.nb;
    const size_t nbr = d.nbr;

    // B is generated by a strictly sequential RNG stream, so every rank can
    // reproduce any part of it. Each rank keeps only the rows it owns; the
    // remaining rows are regenerated on the fly in blocks below.
    Bloc.assign(d.localBlocks * nb * n, 0.0);
    {
        unsigned int seed = 42;
        std::vector<double> row(n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = 0; k < n; ++k) {
                row[k] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
            }
            const size_t m = i / nb;
            if (d.mine(m)) {
                double* dst = Bloc.data() + (m / static_cast<size_t>(d.nprocs)) * nb * n +
                              (i % nb) * n;
                memcpy(dst, row.data(), n * sizeof(double));
            }
        }
    }

    // Compute A = B * B^T for the locally owned rows, streaming over the rows
    // of B in groups of NR (transposed, so the group index vectorizes).
    d.A.assign(d.localBlocks * nb * n, 0.0);
    std::vector<double> bT(NR * n);
    unsigned int seed = 42;
    for (size_t j0 = 0; j0 < n; j0 += NR) {
        const size_t nr = std::min(NR, n - j0);
        for (size_t v = 0; v < nr; ++v) {
            for (size_t k = 0; k < n; ++k) {
                bT[k * NR + v] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
            }
        }

        for (size_t m = d.rank; m < nbr; m += static_cast<size_t>(d.nprocs)) {
            const size_t rb = d.rowsInBlock(m);
            const double* __restrict brows = Bloc.data() + (m / static_cast<size_t>(d.nprocs)) * nb * n;
            double* __restrict arows = d.block(m);
            for (size_t ii = 0; ii < rb; ++ii) {
                const double* __restrict b = brows + ii * n;
                double acc[NR] = {};
                size_t k = 0;
                for (; k + 4 <= n; k += 4) {
                    const double b0 = b[k], b1 = b[k + 1], b2 = b[k + 2], b3 = b[k + 3];
                    const double* __restrict t = bT.data() + k * NR;
                    for (size_t v = 0; v < NR; ++v) {
                        double s = acc[v];
                        s = s + b0 * t[v];
                        s = s + b1 * t[NR + v];
                        s = s + b2 * t[2 * NR + v];
                        s = s + b3 * t[3 * NR + v];
                        acc[v] = s;
                    }
                }
                for (; k < n; ++k) {
                    const double bv = b[k];
                    const double* __restrict t = bT.data() + k * NR;
                    for (size_t v = 0; v < NR; ++v) {
                        acc[v] = std::fma(bv, t[v], acc[v]);
                    }
                }
                for (size_t v = 0; v < nr; ++v) {
                    arows[ii * n + j0 + v] = acc[v];
                }
            }
        }
    }

    // Add diagonal dominance to ensure positive definiteness
    for (size_t m = d.rank; m < nbr; m += static_cast<size_t>(d.nprocs)) {
        double* rows = d.block(m);
        const size_t rb = d.rowsInBlock(m);
        for (size_t ii = 0; ii < rb; ++ii) {
            rows[ii * n + m * nb + ii] += (double)n;
        }
    }

    Bloc.clear();
    Bloc.shrink_to_fit();
}

bool validateCholesky(const Dist& d, const std::vector<double>& A_orig) {
    // Validate by computing L * L^T and comparing with original matrix.
    // Every rank checks its own rows; the rows of L it does not own are
    // streamed through in blocks.
    const size_t n = d.n;
    const size_t nb = d.nb;
    const size_t nbr = d.nbr;

    double maxError = 0.0;
    double relError = 0.0;

    std::vector<double> buf(nb * n);
    for (size_t m = 0; m < nbr; ++m) {
        const size_t rb = d.rowsInBlock(m);
        if (d.mine(m)) {
            memcpy(buf.data(), d.block(m), rb * n * sizeof(double));
        }
        MPI_Bcast(buf.data(), static_cast<int>(rb * n), MPI_DOUBLE, d.owner(m), MPI_COMM_WORLD);

        for (size_t lm = d.rank; lm < nbr; lm += static_cast<size_t>(d.nprocs)) {
            const size_t lrb = d.rowsInBlock(lm);
            const double* __restrict L = d.block(lm);
            const double* __restrict orig =
                A_orig.data() + (lm / static_cast<size_t>(d.nprocs)) * nb * n;
            for (size_t ii = 0; ii < lrb; ++ii) {
                const double* __restrict li = L + ii * n;
                for (size_t jj = 0; jj < rb; ++jj) {
                    const double* __restrict lj = buf.data() + jj * n;
                    double sum = 0.0;
                    size_t k = 0;
                    for (; k + 4 <= n; k += 4) {
                        sum = dotStep4(sum, li + k, lj + k, 1);
                    }
                    for (; k < n; ++k) {
                        sum = std::fma(li[k], lj[k], sum);
                    }
                    const size_t j = m * nb + jj;
                    const double a = orig[ii * n + j];
                    const double error = fabs(sum - a);
                    maxError = std::max(maxError, error);
                    relError = std::max(relError, error / (fabs(a) + 1e-10));
                }
            }
        }
    }

    double red[2] = {maxError, relError};
    MPI_Allreduce(MPI_IN_PLACE, red, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    maxError = red[0];
    relError = red[1];

    if (d.rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    // Check if error is within tolerance
    if (relError > 1e-6) {
        if (d.rank == 0) {
            printf("Validation failed: relative error too large\n");
        }
        return false;
    }

    return true;
}

// Collects the full factor on rank 0 (in global row-major order).
void gatherFullMatrix(const Dist& d, std::vector<double>& full) {
    if (d.rank == 0) {
        full.assign(d.n * d.n, 0.0);
    }
    for (size_t m = 0; m < d.nbr; ++m) {
        const size_t rb = d.rowsInBlock(m);
        const int owner = d.owner(m);
        if (owner == 0) {
            if (d.rank == 0) {
                memcpy(full.data() + m * d.nb * d.n, d.block(m), rb * d.n * sizeof(double));
            }
        } else if (d.rank == owner) {
            MPI_Send(d.block(m), static_cast<int>(rb * d.n), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        } else if (d.rank == 0) {
            MPI_Recv(full.data() + m * d.nb * d.n, static_cast<int>(rb * d.n), MPI_DOUBLE, owner, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
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

    Dist d;
    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.nprocs);

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
            if (d.rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (d.rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (d.rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Row/column block size (see NB).
    d.n = n;
    d.nb = std::max<size_t>(1, std::min(NB, n));
    d.nbr = (n + d.nb - 1) / d.nb;
    d.localBlocks = (d.nbr > static_cast<size_t>(d.rank))
                        ? (d.nbr - static_cast<size_t>(d.rank) + static_cast<size_t>(d.nprocs) - 1) /
                              static_cast<size_t>(d.nprocs)
                        : 0;

    // Generate positive definite matrix
    if (d.rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    std::vector<double> scratch;
    generatePositiveDefiniteMatrix(d, scratch);

    std::vector<double> A_orig;
    if (validate) {
        A_orig = d.A; // Save original for validation
    }

    // Perform Cholesky decomposition
    if (d.rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(d);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (d.rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (d.rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> full;
        gatherFullMatrix(d, full);
        if (d.rank == 0) {
            print_results(full, "CholeskyL");
        }
    }

    // Validation
    if (validate) {
        if (d.rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateCholesky(d, A_orig);

        if (valid) {
            if (d.rank == 0) {
                printf("Validation: PASSED\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (d.rank == 0) {
                printf("Validation: FAILED\n");
            }
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
