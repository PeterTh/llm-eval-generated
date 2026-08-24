#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr int TILE_SIZE = 128;

struct Tile {
    int row = 0, col = 0;
    std::vector<double> x;
};

bool potrf(Tile& a, int n) {
    for (int j = 0; j < n; ++j) {
        double s = 0.0;
        for (int k = 0; k < j; ++k) s += a.x[j * n + k] * a.x[j * n + k];
        const double value = a.x[j * n + j] - s;
        if (value <= 0.0) return false;
        a.x[j * n + j] = std::sqrt(value);
        for (int i = j + 1; i < n; ++i) {
            s = 0.0;
            for (int k = 0; k < j; ++k) s += a.x[i * n + k] * a.x[j * n + k];
            a.x[i * n + j] = (a.x[i * n + j] - s) / a.x[j * n + j];
        }
    }
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) a.x[i * n + j] = 0.0;
    return true;
}

// X <- X L^{-T}, where L is the diagonal tile.
void trsm_right_transpose(Tile& x, const std::vector<double>& l, int m, int b) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < b; ++j) {
            double s = x.x[i * b + j];
            for (int k = 0; k < j; ++k) s -= x.x[i * b + k] * l[j * b + k];
            x.x[i * b + j] = s / l[j * b + j];
        }
    }
}

void update(Tile& c, const std::vector<double>& a, const std::vector<double>& b,
            int m, int n, int k) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int q = 0; q < k; ++q) s += a[i * k + q] * b[j * k + q];
            c.x[i * n + j] -= s;
        }
}

void generatePositiveDefiniteMatrix(std::vector<double>& a, size_t n) {
    std::vector<double> b(n * n);
    unsigned int seed = 42;
    for (double& v : b) v = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) s += b[i * n + k] * b[j * n + k];
            a[i * n + j] = s;
        }
    for (size_t i = 0; i < n; ++i) a[i * n + i] += n;
}

bool validateCholesky(const std::vector<double>& l, const std::vector<double>& orig, size_t n) {
    double max_error = 0.0, rel_error = 0.0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) s += l[i * n + k] * l[j * n + k];
            double e = std::fabs(s - orig[i * n + j]);
            max_error = std::max(max_error, e);
            rel_error = std::max(rel_error, e / (std::fabs(orig[i * n + j]) + 1e-10));
        }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", max_error, rel_error);
    if (rel_error > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* p) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t n = 512;
    bool validate = false, output_results = false;
    int parse_error = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) output_results = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { parse_error = 1; if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } }
    }
    if (parse_error || n == 0) { MPI_Finalize(); return 1; }

    int dims[2] = {0, 0};
    MPI_Dims_create(nranks, 2, dims);
    const int nprow = dims[0], npcol = dims[1];
    const int nb = static_cast<int>((n + TILE_SIZE - 1) / TILE_SIZE);
    std::vector<int> local_slot(nb * nb, -1);
    std::vector<Tile> tiles;
    for (int bi = 0; bi < nb; ++bi)
        for (int bj = 0; bj <= bi; ++bj)
            if ((bi % nprow) * npcol + (bj % npcol) == rank) {
                int rows = std::min<int>(TILE_SIZE, n - size_t(bi) * TILE_SIZE);
                int cols = std::min<int>(TILE_SIZE, n - size_t(bj) * TILE_SIZE);
                local_slot[bi * nb + bj] = static_cast<int>(tiles.size());
                tiles.push_back({bi, bj, std::vector<double>(size_t(rows) * cols)});
            }

    std::vector<double> a, original;
    if (rank == 0) {
        a.resize(n * n);
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        std::printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(a, n);
        if (validate) original = a;
    }

    // Root packs and sends only the tiles owned by each rank. This preserves the
    // original generator bit-for-bit while keeping the factorization distributed.
    for (int dest = 0; dest < nranks; ++dest) {
        std::vector<double> packed;
        if (rank == 0) {
            for (int bi = 0; bi < nb; ++bi) for (int bj = 0; bj <= bi; ++bj)
                if ((bi % nprow) * npcol + (bj % npcol) == dest) {
                    int rows = std::min<int>(TILE_SIZE, n - size_t(bi) * TILE_SIZE);
                    int cols = std::min<int>(TILE_SIZE, n - size_t(bj) * TILE_SIZE);
                    for (int r = 0; r < rows; ++r)
                        for (int c = 0; c < cols; ++c) packed.push_back(a[size_t(bi) * TILE_SIZE * n + size_t(r) * n + bj * TILE_SIZE + c]);
                }
            if (dest == 0) {
                size_t p = 0; for (Tile& t : tiles) { std::copy(packed.begin() + p, packed.begin() + p + t.x.size(), t.x.begin()); p += t.x.size(); }
            } else MPI_Send(packed.data(), static_cast<int>(packed.size()), MPI_DOUBLE, dest, 7, MPI_COMM_WORLD);
        } else if (rank == dest) {
            size_t count = 0; for (const Tile& t : tiles) count += t.x.size();
            packed.resize(count); MPI_Recv(packed.data(), static_cast<int>(count), MPI_DOUBLE, 0, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            size_t p = 0; for (Tile& t : tiles) { std::copy(packed.begin() + p, packed.begin() + p + t.x.size(), t.x.begin()); p += t.x.size(); }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    bool local_ok = true;
    for (int k = 0; k < nb; ++k) {
        int diag_slot = local_slot[k * nb + k];
        int drows = std::min<int>(TILE_SIZE, n - size_t(k) * TILE_SIZE);
        std::vector<double> diag(size_t(drows) * drows);
        if (diag_slot >= 0) { local_ok = potrf(tiles[diag_slot], drows) && local_ok; diag = tiles[diag_slot].x; }
        MPI_Bcast(diag.data(), static_cast<int>(diag.size()), MPI_DOUBLE, (k % nprow) * npcol + (k % npcol), MPI_COMM_WORLD);
        for (int i = k + 1; i < nb; ++i) {
            int rows = std::min<int>(TILE_SIZE, n - size_t(i) * TILE_SIZE);
            int slot = local_slot[i * nb + k];
            std::vector<double> panel(size_t(rows) * drows);
            if (slot >= 0) { trsm_right_transpose(tiles[slot], diag, rows, drows); panel = tiles[slot].x; }
            MPI_Bcast(panel.data(), static_cast<int>(panel.size()), MPI_DOUBLE, (i % nprow) * npcol + (k % npcol), MPI_COMM_WORLD);
            for (int j = k + 1; j <= i; ++j) {
                int cols = std::min<int>(TILE_SIZE, n - size_t(j) * TILE_SIZE);
                int cslot = local_slot[i * nb + j];
                int jrows = std::min<int>(TILE_SIZE, n - size_t(j) * TILE_SIZE);
                int jslot = local_slot[j * nb + k];
                std::vector<double> other(size_t(jrows) * drows);
                if (jslot >= 0) other = tiles[jslot].x;
                MPI_Bcast(other.data(), static_cast<int>(other.size()), MPI_DOUBLE, (j % nprow) * npcol + (k % npcol), MPI_COMM_WORLD);
                if (cslot >= 0) update(tiles[cslot], panel, other, rows, cols, drows);
            }
        }
    }
    int bad = 0; int local_bad = local_ok ? 0 : 1; MPI_Allreduce(&local_bad, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;
    double global_elapsed = 0.0;
    MPI_Reduce(&elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) std::printf("Computing Cholesky decomposition...\nComputation time: %.0f ms\nPerformance: %.3f GFLOPS\n", global_elapsed * 1000.0, (double)n * n * n / 3.0 / global_elapsed / 1e9);
    if (rank == 0) {
        for (const Tile& t : tiles) { int rows = std::min<int>(TILE_SIZE, n - size_t(t.row) * TILE_SIZE), cols = std::min<int>(TILE_SIZE, n - size_t(t.col) * TILE_SIZE); for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) a[size_t(t.row)*TILE_SIZE*n + size_t(r)*n + t.col*TILE_SIZE+c] = t.x[size_t(r)*cols+c]; }
        for (int src = 1; src < nranks; ++src) { size_t count=0; for(int bi=0;bi<nb;++bi)for(int bj=0;bj<=bi;++bj)if((bi%nprow)*npcol+(bj%npcol)==src) count+=size_t(std::min<int>(TILE_SIZE,n-size_t(bi)*TILE_SIZE))*std::min<int>(TILE_SIZE,n-size_t(bj)*TILE_SIZE); std::vector<double> p(count); MPI_Recv(p.data(),(int)count,MPI_DOUBLE,src,8,MPI_COMM_WORLD,MPI_STATUS_IGNORE); size_t q=0; for(int bi=0;bi<nb;++bi)for(int bj=0;bj<=bi;++bj)if((bi%nprow)*npcol+(bj%npcol)==src){int rr=std::min<int>(TILE_SIZE,n-size_t(bi)*TILE_SIZE),cc=std::min<int>(TILE_SIZE,n-size_t(bj)*TILE_SIZE);for(int r=0;r<rr;++r)for(int c=0;c<cc;++c)a[size_t(bi)*TILE_SIZE*n+size_t(r)*n+bj*TILE_SIZE+c]=p[q++];} }
    } else { std::vector<double> p; for(const Tile&t:tiles)p.insert(p.end(),t.x.begin(),t.x.end()); MPI_Send(p.data(),(int)p.size(),MPI_DOUBLE,0,8,MPI_COMM_WORLD); }
    if (rank == 0)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j) a[i * n + j] = 0.0;
    bool validation_ok = true;
    if (rank == 0) { if (output_results) print_results(a, "CholeskyL"); if (validate) { std::printf("Validating result...\nValidation: %s\n", (validation_ok = validateCholesky(a, original, n)) ? "PASSED" : "FAILED"); } }
    int validation_bad = validation_ok ? 0 : 1;
    MPI_Bcast(&validation_bad, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize(); return (bad || validation_bad) ? 1 : 0;
}
