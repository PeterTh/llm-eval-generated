#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation (global)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// MPI domain decomposition helpers
struct DomainDecomp {
    size_t nx, ny, nz;        // global dimensions
    int rank, nprocs;
    size_t local_nz;          // local z size (including halos)
    size_t nz_start;          // global z start of owned region (excluding lower halo)
    size_t nz_end;            // global z end of owned region (exclusive, excluding upper halo)
    size_t local_nxy;         // nx * ny
    int lower_neighbor, upper_neighbor;

    void compute(size_t gx, size_t gy, size_t gz, int r, int np) {
        nx = gx; ny = gy; nz = gz;
        rank = r; nprocs = np;
        local_nxy = nx * ny;

        // Decompose along Z axis
        size_t base = nz / nprocs;
        size_t remainder = nz % nprocs;

        // Compute global z start/end for each rank
        size_t total = 0;
        for (int i = 0; i < rank; ++i) {
            total += base + (i < static_cast<int>(remainder) ? 1 : 0);
        }
        nz_start = total;
        nz_end = nz_start + base + (rank < static_cast<int>(remainder) ? 1 : 0);

        // Local size includes 1-cell halos on each end (if not at boundary)
        local_nz = (nz_end - nz_start);
        if (rank > 0) local_nz += 1;       // lower halo
        if (rank < nprocs - 1) local_nz += 1;  // upper halo

        lower_neighbor = (rank > 0) ? rank - 1 : -1;
        upper_neighbor = (rank < nprocs - 1) ? rank + 1 : -1;
    }

    // Convert global z to local z index
    inline size_t global_to_local_z(size_t gz) const {
        if (rank == 0) {
            return gz - nz_start;
        } else {
            return gz - nz_start + 1;
        }
    }

    // Convert local z to global z
    inline size_t local_to_global_z(size_t lz) const {
        return nz_start + (lz - owned_start_local());
    }

    size_t owned_start_local() const {
        return (rank == 0) ? 0 : 1;
    }
    size_t owned_end_local() const {
        return owned_start_local() + (nz_end - nz_start);
    }
};

// Non-blocking halo exchange for a single field
struct HaloRequest {
    MPI_Request reqs[4];
    int count;
    HaloRequest() : count(0) {}
};

void exchangeHalos(std::vector<double>& field, const DomainDecomp& decomp, HaloRequest& req) {
    const size_t nxy = decomp.local_nxy;
    req.count = 0;
    const int tag = 100;

    // Post all receives first to avoid deadlock
    // Receive lower halo from lower_neighbor (their top owned layer)
    if (decomp.lower_neighbor >= 0) {
        MPI_Irecv(&field[0], nxy, MPI_DOUBLE, decomp.lower_neighbor, tag, MPI_COMM_WORLD, &req.reqs[req.count++]);
    }
    // Receive upper halo from upper_neighbor (their bottom owned layer)
    if (decomp.upper_neighbor >= 0) {
        size_t upper_halo_local = decomp.owned_end_local();
        MPI_Irecv(&field[upper_halo_local * nxy], nxy, MPI_DOUBLE, decomp.upper_neighbor, tag, MPI_COMM_WORLD, &req.reqs[req.count++]);
    }

    // Then post all sends
    // Send top owned layer to upper_neighbor (for their lower halo)
    if (decomp.upper_neighbor >= 0) {
        size_t top_owned = (decomp.owned_end_local() - 1) * nxy;
        MPI_Isend(&field[top_owned], nxy, MPI_DOUBLE, decomp.upper_neighbor, tag, MPI_COMM_WORLD, &req.reqs[req.count++]);
    }
    // Send bottom owned layer to lower_neighbor (for their upper halo)
    if (decomp.lower_neighbor >= 0) {
        MPI_Isend(&field[decomp.owned_start_local() * nxy], nxy, MPI_DOUBLE, decomp.lower_neighbor, tag, MPI_COMM_WORLD, &req.reqs[req.count++]);
    }
}

void waitHalos(HaloRequest& req) {
    if (req.count > 0) {
        MPI_Waitall(req.count, req.reqs, MPI_STATUSES_IGNORE);
    }
}

// Compute Laplacian using local z indices into a local array
// The local array layout: [lower_halo][owned][upper_halo] along z
// We pass the local z index and clamp to local boundaries
inline double computeLaplacianLocal(const std::vector<double>& c, const size_t nx, const size_t ny,
                                    const size_t local_nz, const size_t local_nxy,
                                    const double dx, const double dy, const double dz,
                                    const size_t x, const size_t y, const size_t lz) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (lz < local_nz - 1) ? lz + 1 : lz;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (lz > 0) ? lz - 1 : 0;

    const auto li = [&](size_t lx, size_t ly, size_t lzz) {
        return lzz * local_nxy + ly * nx + lx;
    };

    const double cxx = (c[li(xp, y, lz)] + c[li(xn, y, lz)] -
                  2.0 * c[li(x, y, lz)]) / (dx * dx);
    const double cyy = (c[li(x, yp, lz)] + c[li(x, yn, lz)] -
                  2.0 * c[li(x, y, lz)]) / (dy * dy);
    const double czz = (c[li(x, y, zp)] + c[li(x, y, zn)] -
                  2.0 * c[li(x, y, lz)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Compute chemical potential on local domain
void computeChemicalPotentialLocal(const std::vector<double>& c, std::vector<double>& mu,
                                   const DomainDecomp& decomp,
                                   const double dx, const double dy, const double dz,
                                   const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t nx = decomp.nx;
    const size_t ny = decomp.ny;
    const size_t local_nz = decomp.local_nz;
    const size_t local_nxy = decomp.local_nxy;
    const size_t owned_lo = decomp.owned_start_local();
    const size_t owned_hi = decomp.owned_end_local();

    for (size_t lz = owned_lo; lz < owned_hi; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = lz * local_nxy + y * nx + x;
                const double cv = c[idx];

                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacianLocal(c, nx, ny, local_nz, local_nxy, dx, dy, dz, x, y, lz);
            }
        }
    }
}

// Cahn-Hilliard update step on local domain
void cahnHilliardUpdateLocal(std::vector<double>& cnew, const std::vector<double>& cold,
                             const std::vector<double>& mu,
                             const DomainDecomp& decomp,
                             const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t nx = decomp.nx;
    const size_t ny = decomp.ny;
    const size_t local_nz = decomp.local_nz;
    const size_t local_nxy = decomp.local_nxy;
    const size_t owned_lo = decomp.owned_start_local();
    const size_t owned_hi = decomp.owned_end_local();

    for (size_t lz = owned_lo; lz < owned_hi; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = lz * local_nxy + y * nx + x;
                cnew[idx] = cold[idx] + dt * D *
                           computeLaplacianLocal(mu, nx, ny, local_nz, local_nxy, dx, dy, dz, x, y, lz);
            }
        }
    }
}

// Initialize concentration field on local domain
void initializeConcentrationLocal(std::vector<double>& c, const DomainDecomp& decomp) {
    const size_t vol = decomp.nx * decomp.ny * decomp.nz;
    const size_t nx = decomp.nx;
    const size_t ny = decomp.ny;

    for (size_t lz = 0; lz < decomp.local_nz; ++lz) {
        const size_t gz = decomp.local_to_global_z(lz);
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = lz * decomp.local_nxy + y * nx + x;
                const size_t linear_id = gz * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse the same)
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
            if (rank == 0) printUsage(argv[0]);
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
        printf("MPI processes: %d\n", nprocs);
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

    // Domain decomposition
    DomainDecomp decomp;
    decomp.compute(nx, ny, nz, rank, nprocs);

    size_t local_size = decomp.local_nz * decomp.local_nxy;

    // Allocate local arrays
    std::vector<double> cold(local_size);
    std::vector<double> cnew(local_size);
    std::vector<double> mu(local_size);

    // Initialize concentration field
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentrationLocal(cold, decomp);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // Exchange halos for cold
        {
            HaloRequest req;
            exchangeHalos(cold, decomp, req);
            waitHalos(req);
        }

        // Compute chemical potential on local domain
        computeChemicalPotentialLocal(cold, mu, decomp, dx, dy, dz, gamma, e_AA, e_BB, e_AB);

        // Exchange halos for mu
        {
            HaloRequest req;
            exchangeHalos(mu, decomp, req);
            waitHalos(req);
        }

        // Update concentration on local domain
        cahnHilliardUpdateLocal(cnew, cold, mu, decomp, D, dt, dx, dy, dz);

        // Swap buffers
        std::swap(cold, cnew);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Synchronize all ranks
    MPI_Barrier(MPI_COMM_WORLD);

    long local_duration = duration.count();
    long global_duration = local_duration;
    MPI_Reduce(&local_duration, &global_duration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Calculate performance (global)
    size_t gridSize = nx * ny * nz;
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (global_duration / 1000.0) / 1e6;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_duration);
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather full concentration field to rank 0 for results/validation
    std::vector<double> global_cold;
    if (printResults || validate) {
        // Gather sizes and displacements
        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);
        int offset = 0;
        for (int i = 0; i < nprocs; ++i) {
            DomainDecomp tmp;
            tmp.compute(nx, ny, nz, i, nprocs);
            int owned = static_cast<int>(tmp.nz_end - tmp.nz_start);
            recvcounts[i] = owned * static_cast<int>(tmp.local_nxy);
            displs[i] = offset;
            offset += recvcounts[i];
        }

        if (rank == 0) {
            global_cold.resize(gridSize);
        }

        // Gather only owned cells
        MPI_Gatherv(&cold[decomp.owned_start_local() * decomp.local_nxy],
                    static_cast<int>((decomp.nz_end - decomp.nz_start) * decomp.local_nxy),
                    MPI_DOUBLE,
                    global_cold.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        // Reassemble into proper global order on rank 0
        if (rank == 0) {
            std::vector<double> reordered(gridSize);
            for (int i = 0; i < nprocs; ++i) {
                DomainDecomp tmp;
                tmp.compute(nx, ny, nz, i, nprocs);
                const size_t owned_count = tmp.nz_end - tmp.nz_start;
                for (size_t local_oz = 0; local_oz < owned_count; ++local_oz) {
                    const size_t gz = tmp.nz_start + local_oz;
                    const size_t src_base = displs[i] + local_oz * tmp.local_nxy;
                    const size_t dst_base = gz * nx * ny;
                    for (size_t j = 0; j < tmp.local_nxy; ++j) {
                        reordered[dst_base + j] = global_cold[src_base + j];
                    }
                }
            }
            global_cold = std::move(reordered);
        }
    }

    // Print results for external validation (rank 0)
    if (printResults && rank == 0) {
        print_results(global_cold, "Concentration");
    }

    // Validation (rank 0 uses gathered data)
    if (validate && rank == 0) {
        double minVal = global_cold[0];
        double maxVal = global_cold[0];
        bool valid = true;
        for (const auto& val : global_cold) {
            if (std::isnan(val) || std::isinf(val)) {
                valid = false;
            }
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }
        printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
        if (maxVal > 10.0 || minVal < -10.0) {
            valid = false;
        }

        printf("Validating result...\n");
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
