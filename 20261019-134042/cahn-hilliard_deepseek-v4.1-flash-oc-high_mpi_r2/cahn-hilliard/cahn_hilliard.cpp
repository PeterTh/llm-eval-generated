#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

struct Domain {
    size_t nx;
    size_t ny;
    size_t nz;
    size_t plane;
    size_t zstart;
    size_t zend;
    size_t zlocal;
};

static int ownerRank(const size_t z, const size_t nz, const int nprocs) {
    const unsigned long long numerator =
        (static_cast<unsigned long long>(z) + 1ULL) * static_cast<unsigned long long>(nprocs) - 1ULL;
    return static_cast<int>(numerator / static_cast<unsigned long long>(nz));
}

static int startExchange(double* buffer, const Domain& d, const int nprocs, MPI_Comm comm,
                         MPI_Request* requests) {
    if (d.zlocal == 0) {
        return 0;
    }

    const int plane = static_cast<int>(d.plane);
    int n = 0;

    if (d.zstart == 0) {
        std::memcpy(buffer, buffer + d.plane, d.plane * sizeof(double));
    } else {
        const int neighbor = ownerRank(d.zstart - 1, d.nz, nprocs);
        MPI_Isend(buffer + d.plane, plane, MPI_DOUBLE, neighbor, 0, comm, &requests[n++]);
        MPI_Irecv(buffer, plane, MPI_DOUBLE, neighbor, 0, comm, &requests[n++]);
    }

    if (d.zend == d.nz) {
        std::memcpy(buffer + (d.zlocal + 1) * d.plane,
                    buffer + d.zlocal * d.plane,
                    d.plane * sizeof(double));
    } else {
        const int neighbor = ownerRank(d.zend, d.nz, nprocs);
        MPI_Isend(buffer + d.zlocal * d.plane, plane, MPI_DOUBLE, neighbor, 0, comm, &requests[n++]);
        MPI_Irecv(buffer + (d.zlocal + 1) * d.plane, plane, MPI_DOUBLE, neighbor, 0, comm, &requests[n++]);
    }

    return n;
}

static void finishExchange(MPI_Request* requests, const int n) {
    if (n > 0) {
        MPI_Waitall(n, requests, MPI_STATUSES_IGNORE);
    }
}

// Compute chemical potential
static void computeChemicalPotential(const double* __restrict c, double* __restrict mu,
                                     const Domain& d, const size_t lzBegin, const size_t lzEnd,
                                     const double invdx2, const double invdy2, const double invdz2,
                                     const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t nx = d.nx;
    const size_t ny = d.ny;
    const size_t plane = d.plane;

    for (size_t lz = lzBegin; lz < lzEnd; ++lz) {
        const size_t base = (lz + 1) * plane;
        for (size_t y = 0; y < ny; ++y) {
            const size_t yp = (y + 1 < ny) ? y + 1 : y;
            const size_t yn = (y > 0) ? y - 1 : 0;
            const size_t row = base + y * nx;
            const size_t rowp = base + yp * nx;
            const size_t rown = base + yn * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t xp = (x + 1 < nx) ? x + 1 : x;
                const size_t xn = (x > 0) ? x - 1 : 0;
                const size_t idx = row + x;
                const double cv = c[idx];

                const double cxx = (c[row + xp] + c[row + xn] - 2.0 * cv) * invdx2;
                const double cyy = (c[rowp + x] + c[rown + x] - 2.0 * cv) * invdy2;
                const double czz = (c[idx + plane] + c[idx - plane] - 2.0 * cv) * invdz2;

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * (cxx + cyy + czz);
            }
        }
    }
}

// Cahn-Hilliard update step
static void cahnHilliardUpdate(double* __restrict cnew, const double* __restrict cold,
                               const double* __restrict mu, const Domain& d,
                               const size_t lzBegin, const size_t lzEnd,
                               const double D, const double dt,
                               const double invdx2, const double invdy2, const double invdz2) {
    const size_t nx = d.nx;
    const size_t ny = d.ny;
    const size_t plane = d.plane;
    const double factor = dt * D;

    for (size_t lz = lzBegin; lz < lzEnd; ++lz) {
        const size_t base = (lz + 1) * plane;
        for (size_t y = 0; y < ny; ++y) {
            const size_t yp = (y + 1 < ny) ? y + 1 : y;
            const size_t yn = (y > 0) ? y - 1 : 0;
            const size_t row = base + y * nx;
            const size_t rowp = base + yp * nx;
            const size_t rown = base + yn * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t xp = (x + 1 < nx) ? x + 1 : x;
                const size_t xn = (x > 0) ? x - 1 : 0;
                const size_t idx = row + x;
                const double cv = mu[idx];

                const double mxx = (mu[row + xp] + mu[row + xn] - 2.0 * cv) * invdx2;
                const double myy = (mu[rowp + x] + mu[rown + x] - 2.0 * cv) * invdy2;
                const double mzz = (mu[idx + plane] + mu[idx - plane] - 2.0 * cv) * invdz2;

                cnew[idx] = cold[idx] + factor * (mxx + myy + mzz);
            }
        }
    }
}

// Initialize concentration field
static void initializeConcentration(double* c, const Domain& d, const size_t vol) {
    const size_t nx = d.nx;
    const size_t ny = d.ny;
    const size_t plane = d.plane;

    for (size_t lz = 0; lz < d.zlocal; ++lz) {
        const size_t z = d.zstart + lz;
        const size_t base = (lz + 1) * plane;
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = base + y * nx;
            const size_t linear_y = z * plane + y * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t linear_id = linear_y + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[row + x] = -1.0 + 2.0 * pseudo;
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (rank == 0) {
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

    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);
    
    size_t gridSize = nx * ny * nz;

    Domain d;
    d.nx = nx;
    d.ny = ny;
    d.nz = nz;
    d.plane = nx * ny;
    d.zstart = static_cast<size_t>(static_cast<unsigned long long>(nz) * static_cast<unsigned long long>(rank) / static_cast<unsigned long long>(nprocs));
    d.zend = static_cast<size_t>(static_cast<unsigned long long>(nz) * static_cast<unsigned long long>(rank + 1) / static_cast<unsigned long long>(nprocs));
    d.zlocal = d.zend - d.zstart;

    const size_t localSize = (d.zlocal + 2) * d.plane;
    
    // Allocate arrays
    std::vector<double> cold(localSize);
    std::vector<double> cnew(localSize);
    std::vector<double> mu(localSize);
    
    // Initialize concentration field
    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(cold.data(), d, gridSize);
    
    // Run simulation
    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const size_t interiorBegin = (d.zlocal >= 3) ? 1 : 0;
    const size_t interiorEnd = (d.zlocal >= 3) ? d.zlocal - 1 : 0;
    MPI_Request requests[4];
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        int nreq = startExchange(cold.data(), d, nprocs, MPI_COMM_WORLD, requests);
        computeChemicalPotential(cold.data(), mu.data(), d, interiorBegin, interiorEnd,
                                 invdx2, invdy2, invdz2, gamma, e_AA, e_BB, e_AB);
        finishExchange(requests, nreq);
        if (d.zlocal > 0) {
            if (interiorEnd == 0) {
                computeChemicalPotential(cold.data(), mu.data(), d, 0, d.zlocal,
                                         invdx2, invdy2, invdz2, gamma, e_AA, e_BB, e_AB);
            } else {
                computeChemicalPotential(cold.data(), mu.data(), d, 0, 1,
                                         invdx2, invdy2, invdz2, gamma, e_AA, e_BB, e_AB);
                computeChemicalPotential(cold.data(), mu.data(), d, d.zlocal - 1, d.zlocal,
                                         invdx2, invdy2, invdz2, gamma, e_AA, e_BB, e_AB);
            }
        }
        
        // Update concentration
        nreq = startExchange(mu.data(), d, nprocs, MPI_COMM_WORLD, requests);
        cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), d, interiorBegin, interiorEnd,
                           D, dt, invdx2, invdy2, invdz2);
        finishExchange(requests, nreq);
        if (d.zlocal > 0) {
            if (interiorEnd == 0) {
                cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), d, 0, d.zlocal,
                                   D, dt, invdx2, invdy2, invdz2);
            } else {
                cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), d, 0, 1,
                                   D, dt, invdx2, invdy2, invdz2);
                cahnHilliardUpdate(cnew.data(), cold.data(), mu.data(), d, d.zlocal - 1, d.zlocal,
                                   D, dt, invdx2, invdy2, invdz2);
            }
        }
        
        // Swap buffers
        std::swap(cold, cnew);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }
    
    int rc = 0;

    // Print results for external validation
    if (printResults || validate) {
        std::vector<double> global;
        std::vector<int> counts;
        std::vector<int> displacements;

        if (rank == 0) {
            global.resize(gridSize);
            counts.resize(nprocs);
            displacements.resize(nprocs);
            for (int r = 0; r < nprocs; ++r) {
                const size_t s = static_cast<size_t>(static_cast<unsigned long long>(nz) * static_cast<unsigned long long>(r) / static_cast<unsigned long long>(nprocs));
                const size_t e = static_cast<size_t>(static_cast<unsigned long long>(nz) * static_cast<unsigned long long>(r + 1) / static_cast<unsigned long long>(nprocs));
                counts[r] = static_cast<int>((e - s) * d.plane);
                displacements[r] = static_cast<int>(s * d.plane);
            }
        }

        const int localCount = static_cast<int>(d.zlocal * d.plane);
        MPI_Gatherv(cold.data() + d.plane, localCount, MPI_DOUBLE,
                    (rank == 0) ? global.data() : nullptr,
                    (rank == 0) ? counts.data() : nullptr,
                    (rank == 0) ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results(global, "Concentration");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(global, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                    rc = 0;
                } else {
                    printf("Validation: FAILED\n");
                    rc = 1;
                }
            }
        }

        if (validate) {
            MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return rc;
}
