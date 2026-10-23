#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation (local, padded storage)
inline constexpr size_t ldix(const size_t x, const size_t y, const size_t z,
                             const size_t sx, const size_t sxy) noexcept {
    return z * sxy + y * sx + x;
}

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t sx, const size_t sy, const size_t sz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t sxy = sx * sy;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;

    for (size_t z = 1; z + 1 < sz; ++z) {
        for (size_t y = 1; y + 1 < sy; ++y) {
            const size_t row = z * sxy + y * sx;
            for (size_t x = 1; x + 1 < sx; ++x) {
                const size_t idx = row + x;
                const double cv = c[idx];

                const double cxx = (c[idx + 1] + c[idx - 1] - 2.0 * cv) / dx2;
                const double cyy = (c[idx + sx] + c[idx - sx] - 2.0 * cv) / dy2;
                const double czz = (c[idx + sxy] + c[idx - sxy] - 2.0 * cv) / dz2;

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * (cxx + cyy + czz);
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t sx, const size_t sy, const size_t sz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t sxy = sx * sy;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;

    for (size_t z = 1; z + 1 < sz; ++z) {
        for (size_t y = 1; y + 1 < sy; ++y) {
            const size_t row = z * sxy + y * sx;
            for (size_t x = 1; x + 1 < sx; ++x) {
                const size_t idx = row + x;
                const double muv = mu[idx];

                const double muxx = (mu[idx + 1] + mu[idx - 1] - 2.0 * muv) / dx2;
                const double muyy = (mu[idx + sx] + mu[idx - sx] - 2.0 * muv) / dy2;
                const double muzz = (mu[idx + sxy] + mu[idx - sxy] - 2.0 * muv) / dz2;

                cnew[idx] = cold[idx] + dt * D * (muxx + muyy + muzz);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c,
                             const size_t nx, const size_t ny, const size_t nz,
                             const size_t x0, const size_t y0, const size_t z0,
                             const size_t lx, const size_t ly, const size_t lz,
                             const size_t sx, const size_t sxy) {
    const size_t vol = nx * ny * nz;
    const size_t nxny = nx * ny;

    for (size_t z = 0; z < lz; ++z) {
        const size_t gz = z0 + z;
        for (size_t y = 0; y < ly; ++y) {
            const size_t gy = y0 + y;
            for (size_t x = 0; x < lx; ++x) {
                const size_t gx = x0 + x;
                const size_t linear_id = gz * nxny + gy * nx + gx;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[ldix(x + 1, y + 1, z + 1, sx, sxy)] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static void partitionAxis(const size_t n, const int dim, const int coord,
                          size_t& start, size_t& count) {
    const size_t d = static_cast<size_t>(dim);
    const size_t c = static_cast<size_t>(coord);
    const size_t base = n / d;
    const size_t rem = n % d;
    count = base + (c < rem ? 1 : 0);
    start = c * base + std::min(c, rem);
}

static void chooseDims(const size_t n[3], const int nprocs, int dims[3]) {
    int requested[3] = {0, 0, 0};
    MPI_Dims_create(nprocs, 3, requested);
    for (int i = 0; i < 3; ++i) dims[i] = requested[i];

    bool reduced = false;
    for (int i = 0; i < 3; ++i) {
        if (n[i] == 0) {
            dims[i] = 1;
            reduced = true;
        } else if (static_cast<size_t>(dims[i]) > n[i]) {
            dims[i] = static_cast<int>(n[i]);
            reduced = true;
        }
    }
    if (!reduced) return;

    long long prod = static_cast<long long>(dims[0]) * dims[1] * dims[2];
    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 0; i < 3; ++i) {
            const long long target = nprocs / prod;
            if (target <= 1) continue;
            const long long cap = n[i] == 0
                                ? 1
                                : static_cast<long long>(n[i] / static_cast<size_t>(dims[i]));
            const long long k = std::min(target, cap);
            if (k > 1) {
                dims[i] = static_cast<int>(static_cast<long long>(dims[i]) * k);
                prod = static_cast<long long>(dims[0]) * dims[1] * dims[2];
                changed = true;
            }
        }
    }
}

struct HaloBuffers {
    std::vector<double> sendX[2], recvX[2];
    std::vector<double> sendY[2], recvY[2];
    std::vector<double> sendZ[2], recvZ[2];
};

static void allocateHaloBuffers(HaloBuffers& h, const size_t lx, const size_t ly, const size_t lz) {
    const size_t cX = std::max<size_t>(1, ly * lz);
    const size_t cY = std::max<size_t>(1, lx * lz);
    const size_t cZ = std::max<size_t>(1, lx * ly);
    for (int i = 0; i < 2; ++i) {
        h.sendX[i].resize(cX);
        h.recvX[i].resize(cX);
        h.sendY[i].resize(cY);
        h.recvY[i].resize(cY);
        h.sendZ[i].resize(cZ);
        h.recvZ[i].resize(cZ);
    }
}

static void packX(const std::vector<double>& c, const size_t x,
                  const size_t sx, const size_t sxy,
                  const size_t ly, const size_t lz, std::vector<double>& b) {
    size_t p = 0;
    for (size_t z = 1; z <= lz; ++z) {
        for (size_t y = 1; y <= ly; ++y) {
            b[p++] = c[ldix(x, y, z, sx, sxy)];
        }
    }
}

static void unpackX(std::vector<double>& c, const size_t x,
                    const size_t sx, const size_t sxy,
                    const size_t ly, const size_t lz, const std::vector<double>& b) {
    size_t p = 0;
    for (size_t z = 1; z <= lz; ++z) {
        for (size_t y = 1; y <= ly; ++y) {
            c[ldix(x, y, z, sx, sxy)] = b[p++];
        }
    }
}

static void fillX(std::vector<double>& c, const size_t xdst, const size_t xsrc,
                  const size_t sx, const size_t sxy,
                  const size_t ly, const size_t lz) {
    for (size_t z = 1; z <= lz; ++z) {
        for (size_t y = 1; y <= ly; ++y) {
            c[ldix(xdst, y, z, sx, sxy)] = c[ldix(xsrc, y, z, sx, sxy)];
        }
    }
}

static void packY(const std::vector<double>& c, const size_t y,
                  const size_t sx, const size_t sxy,
                  const size_t lx, const size_t lz, std::vector<double>& b) {
    size_t p = 0;
    for (size_t z = 1; z <= lz; ++z) {
        for (size_t x = 1; x <= lx; ++x) {
            b[p++] = c[ldix(x, y, z, sx, sxy)];
        }
    }
}

static void unpackY(std::vector<double>& c, const size_t y,
                    const size_t sx, const size_t sxy,
                    const size_t lx, const size_t lz, const std::vector<double>& b) {
    size_t p = 0;
    for (size_t z = 1; z <= lz; ++z) {
        for (size_t x = 1; x <= lx; ++x) {
            c[ldix(x, y, z, sx, sxy)] = b[p++];
        }
    }
}

static void fillY(std::vector<double>& c, const size_t ydst, const size_t ysrc,
                  const size_t sx, const size_t sxy,
                  const size_t lx, const size_t lz) {
    for (size_t z = 1; z <= lz; ++z) {
        for (size_t x = 1; x <= lx; ++x) {
            c[ldix(x, ydst, z, sx, sxy)] = c[ldix(x, ysrc, z, sx, sxy)];
        }
    }
}

static void packZ(const std::vector<double>& c, const size_t z,
                  const size_t sx, const size_t sxy,
                  const size_t lx, const size_t ly, std::vector<double>& b) {
    size_t p = 0;
    for (size_t y = 1; y <= ly; ++y) {
        for (size_t x = 1; x <= lx; ++x) {
            b[p++] = c[ldix(x, y, z, sx, sxy)];
        }
    }
}

static void unpackZ(std::vector<double>& c, const size_t z,
                    const size_t sx, const size_t sxy,
                    const size_t lx, const size_t ly, const std::vector<double>& b) {
    size_t p = 0;
    for (size_t y = 1; y <= ly; ++y) {
        for (size_t x = 1; x <= lx; ++x) {
            c[ldix(x, y, z, sx, sxy)] = b[p++];
        }
    }
}

static void fillZ(std::vector<double>& c, const size_t zdst, const size_t zsrc,
                  const size_t sx, const size_t sxy,
                  const size_t lx, const size_t ly) {
    for (size_t y = 1; y <= ly; ++y) {
        for (size_t x = 1; x <= lx; ++x) {
            c[ldix(x, y, zdst, sx, sxy)] = c[ldix(x, y, zsrc, sx, sxy)];
        }
    }
}

static void exchangeHalos(std::vector<double>& c,
                          const int neg[3], const int pos[3],
                          const size_t lx, const size_t ly, const size_t lz,
                          const size_t sx, const size_t sy, const size_t sz,
                          MPI_Comm cart, HaloBuffers& h) {
    const size_t sxy = sx * sy;
    MPI_Request reqs[12];
    int nreq = 0;

    if (neg[0] != MPI_PROC_NULL) {
        packX(c, 1, sx, sxy, ly, lz, h.sendX[0]);
        const int n = static_cast<int>(h.sendX[0].size());
        MPI_Irecv(h.recvX[0].data(), n, MPI_DOUBLE, neg[0], 10, cart, &reqs[nreq++]);
        MPI_Isend(h.sendX[0].data(), n, MPI_DOUBLE, neg[0], 10, cart, &reqs[nreq++]);
    } else {
        fillX(c, 0, 1, sx, sxy, ly, lz);
    }
    if (pos[0] != MPI_PROC_NULL) {
        packX(c, lx, sx, sxy, ly, lz, h.sendX[1]);
        const int n = static_cast<int>(h.sendX[1].size());
        MPI_Irecv(h.recvX[1].data(), n, MPI_DOUBLE, pos[0], 10, cart, &reqs[nreq++]);
        MPI_Isend(h.sendX[1].data(), n, MPI_DOUBLE, pos[0], 10, cart, &reqs[nreq++]);
    } else {
        fillX(c, sx - 1, lx, sx, sxy, ly, lz);
    }

    if (neg[1] != MPI_PROC_NULL) {
        packY(c, 1, sx, sxy, lx, lz, h.sendY[0]);
        const int n = static_cast<int>(h.sendY[0].size());
        MPI_Irecv(h.recvY[0].data(), n, MPI_DOUBLE, neg[1], 11, cart, &reqs[nreq++]);
        MPI_Isend(h.sendY[0].data(), n, MPI_DOUBLE, neg[1], 11, cart, &reqs[nreq++]);
    } else {
        fillY(c, 0, 1, sx, sxy, lx, lz);
    }
    if (pos[1] != MPI_PROC_NULL) {
        packY(c, ly, sx, sxy, lx, lz, h.sendY[1]);
        const int n = static_cast<int>(h.sendY[1].size());
        MPI_Irecv(h.recvY[1].data(), n, MPI_DOUBLE, pos[1], 11, cart, &reqs[nreq++]);
        MPI_Isend(h.sendY[1].data(), n, MPI_DOUBLE, pos[1], 11, cart, &reqs[nreq++]);
    } else {
        fillY(c, sy - 1, ly, sx, sxy, lx, lz);
    }

    if (neg[2] != MPI_PROC_NULL) {
        packZ(c, 1, sx, sxy, lx, ly, h.sendZ[0]);
        const int n = static_cast<int>(h.sendZ[0].size());
        MPI_Irecv(h.recvZ[0].data(), n, MPI_DOUBLE, neg[2], 12, cart, &reqs[nreq++]);
        MPI_Isend(h.sendZ[0].data(), n, MPI_DOUBLE, neg[2], 12, cart, &reqs[nreq++]);
    } else {
        fillZ(c, 0, 1, sx, sxy, lx, ly);
    }
    if (pos[2] != MPI_PROC_NULL) {
        packZ(c, lz, sx, sxy, lx, ly, h.sendZ[1]);
        const int n = static_cast<int>(h.sendZ[1].size());
        MPI_Irecv(h.recvZ[1].data(), n, MPI_DOUBLE, pos[2], 12, cart, &reqs[nreq++]);
        MPI_Isend(h.sendZ[1].data(), n, MPI_DOUBLE, pos[2], 12, cart, &reqs[nreq++]);
    } else {
        fillZ(c, sz - 1, lz, sx, sxy, lx, ly);
    }

    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    if (neg[0] != MPI_PROC_NULL) unpackX(c, 0, sx, sxy, ly, lz, h.recvX[0]);
    if (pos[0] != MPI_PROC_NULL) unpackX(c, sx - 1, sx, sxy, ly, lz, h.recvX[1]);
    if (neg[1] != MPI_PROC_NULL) unpackY(c, 0, sx, sxy, lx, lz, h.recvY[0]);
    if (pos[1] != MPI_PROC_NULL) unpackY(c, sy - 1, sx, sxy, lx, lz, h.recvY[1]);
    if (neg[2] != MPI_PROC_NULL) unpackZ(c, 0, sx, sxy, lx, ly, h.recvZ[0]);
    if (pos[2] != MPI_PROC_NULL) unpackZ(c, sz - 1, sx, sxy, lx, ly, h.recvZ[1]);
}

static void gatherField(const std::vector<double>& local,
                        std::vector<double>& global,
                        const size_t nx, const size_t ny, const size_t nz,
                        const int dims[3],
                        const size_t lx, const size_t ly, const size_t lz,
                        const size_t sx, const size_t sxy,
                        const int rank, const int nprocs,
                        MPI_Comm comm, MPI_Comm cart) {
    std::vector<double> packed(lx * ly * lz);
    size_t p = 0;
    for (size_t z = 1; z <= lz; ++z) {
        for (size_t y = 1; y <= ly; ++y) {
            for (size_t x = 1; x <= lx; ++x) {
                packed[p++] = local[ldix(x, y, z, sx, sxy)];
            }
        }
    }

    const int localCount = static_cast<int>(packed.size());
    std::vector<int> counts(nprocs);
    std::vector<int> displs(nprocs);
    MPI_Allgather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    displs[0] = 0;
    for (int i = 1; i < nprocs; ++i) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }
    const int total = displs[nprocs - 1] + counts[nprocs - 1];

    std::vector<double> all;
    if (rank == 0) {
        all.resize(total);
        global.assign(nx * ny * nz, 0.0);
    }

    double* recvBuf = (rank == 0) ? all.data() : nullptr;
    int* recvCounts = (rank == 0) ? counts.data() : nullptr;
    int* recvDispls = (rank == 0) ? displs.data() : nullptr;
    MPI_Gatherv(packed.data(), localCount, MPI_DOUBLE,
                recvBuf, recvCounts, recvDispls, MPI_DOUBLE, 0, comm);

    if (rank != 0) return;

    const size_t nxny = nx * ny;
    for (int r = 0; r < nprocs; ++r) {
        int coords[3];
        MPI_Cart_coords(cart, r, 3, coords);

        size_t xs, xc, ys, yc, zs, zc;
        partitionAxis(nx, dims[0], coords[0], xs, xc);
        partitionAxis(ny, dims[1], coords[1], ys, yc);
        partitionAxis(nz, dims[2], coords[2], zs, zc);

        const double* src = all.data() + displs[r];
        size_t q = 0;
        for (size_t zi = 0; zi < zc; ++zi) {
            for (size_t yi = 0; yi < yc; ++yi) {
                const size_t gbase = (zs + zi) * nxny + (ys + yi) * nx + xs;
                for (size_t xi = 0; xi < xc; ++xi) {
                    global[gbase + xi] = src[q++];
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    int worldSize = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (worldRank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    const size_t gridSize = nx * ny * nz;

    const size_t n[3] = {nx, ny, nz};
    int dims[3] = {1, 1, 1};
    chooseDims(n, worldSize, dims);

    int periods[3] = {0, 0, 0};
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart);

    if (cart == MPI_COMM_NULL) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        return 0;
    }

    int rank = 0;
    MPI_Comm_rank(cart, &rank);
    int nprocs = 1;
    MPI_Comm_size(cart, &nprocs);

    int coords[3];
    MPI_Cart_coords(cart, rank, 3, coords);

    size_t x0, y0, z0;
    size_t lx, ly, lz;
    partitionAxis(nx, dims[0], coords[0], x0, lx);
    partitionAxis(ny, dims[1], coords[1], y0, ly);
    partitionAxis(nz, dims[2], coords[2], z0, lz);

    const size_t sx = lx + 2;
    const size_t sy = ly + 2;
    const size_t sz = lz + 2;
    const size_t sxy = sx * sy;

    int neg[3], pos[3];
    MPI_Cart_shift(cart, 0, 1, &neg[0], &pos[0]);
    MPI_Cart_shift(cart, 1, 1, &neg[1], &pos[1]);
    MPI_Cart_shift(cart, 2, 1, &neg[2], &pos[2]);

    const size_t localSize = sx * sy * sz;
    std::vector<double> cold(localSize);
    std::vector<double> cnew(localSize);
    std::vector<double> mu(localSize);

    HaloBuffers halo;
    allocateHaloBuffers(halo, lx, ly, lz);

    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, x0, y0, z0, lx, ly, lz, sx, sxy);

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");

    MPI_Barrier(cart);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(cold, neg, pos, lx, ly, lz, sx, sy, sz, cart, halo);

        computeChemicalPotential(cold, mu, sx, sy, sz, dx, dy, dz,
                                 gamma, e_AA, e_BB, e_AB);

        exchangeHalos(mu, neg, pos, lx, ly, lz, sx, sy, sz, cart, halo);

        cahnHilliardUpdate(cnew, cold, mu, sx, sy, sz, D, dt, dx, dy, dz);

        std::swap(cold, cnew);
    }

    MPI_Barrier(cart);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<double> global;
    if (printResults || validate) {
        gatherField(cold, global, nx, ny, nz, dims, lx, ly, lz, sx, sxy,
                    rank, nprocs, cart, cart);
    }

    if (printResults && rank == 0) {
        print_results(global, "Concentration");
    }

    int validFlag = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(global, nx, ny, nz);
            validFlag = valid ? 1 : 0;
        }
        MPI_Bcast(&validFlag, 1, MPI_INT, 0, cart);
        if (rank == 0) {
            if (validFlag) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    const int exitCode = validFlag ? 0 : 1;

    MPI_Comm_free(&cart);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
