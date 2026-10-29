#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include <mpi.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions
double computeLaplacian(const std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                        const double dx, const double dy, const double dz, const size_t x, const size_t y, const size_t z) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;
    
    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] - 
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);
    
    return cxx + cyy + czz;
}

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    for (size_t z = 1; z + 1 < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = c[idx];
                
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) 
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacian(c, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    for (size_t z = 1; z + 1 < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Exchange the two contiguous slab faces. Physical boundaries are clamped.
void exchangeHalos(std::vector<double>& field, size_t plane, size_t local_z,
                   int rank, int ranks, MPI_Comm comm) {
    const int prev = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    // Chunk large planes to respect MPI's int count limit.
    for (size_t offset = 0; offset < plane;) {
        const int count = static_cast<int>(std::min(plane - offset,
                                  static_cast<size_t>(std::numeric_limits<int>::max())));
        MPI_Sendrecv(field.data() + plane + offset, count, MPI_DOUBLE, prev, 0,
                     field.data() + (local_z + 1) * plane + offset, count, MPI_DOUBLE, next, 0,
                     comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(field.data() + local_z * plane + offset, count, MPI_DOUBLE, next, 1,
                     field.data() + offset, count, MPI_DOUBLE, prev, 1,
                     comm, MPI_STATUS_IGNORE);
        offset += count;
    }
    if (prev == MPI_PROC_NULL)
        std::copy_n(field.data() + plane, plane, field.data());
    if (next == MPI_PROC_NULL)
        std::copy_n(field.data() + local_z * plane, plane, field.data() + (local_z + 1) * plane);
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    if (nx == 0 || ny == 0 || nz == 0) {
        if (rank == 0) printf("Grid dimensions must be positive\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }

    const double dx = 1.0, dy = 1.0, dz = 1.0, dt = 0.01;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0);
    const double gamma = 0.5, D = 1.0;
    const size_t plane = nx * ny, gridSize = plane * nz;
    // Ranks beyond the number of planes participate in output and reductions,
    // but are excluded from the stencil communicator.
    const int active = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    MPI_Comm slab_comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &slab_comm);
    const size_t local_z = rank < active ? nz / active + (static_cast<size_t>(rank) < nz % active) : 0;
    const size_t z_start = rank < active ? static_cast<size_t>(rank) * (nz / active) +
                         std::min(static_cast<size_t>(rank), nz % active) : nz;
    std::vector<double> cold(local_z ? (local_z + 2) * plane : 0);
    std::vector<double> cnew(cold.size()), mu(cold.size());
    for (size_t z = 0; z < local_z; ++z) {
        for (size_t i = 0; i < plane; ++i) {
            const size_t linear_id = (z_start + z) * plane + i;
            const double pseudo = (((linear_id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
            cold[(z + 1) * plane + i] = -1.0 + 2.0 * pseudo;
        }
    }
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (local_z) {
        for (int t = 0; t < iterations; ++t) {
            exchangeHalos(cold, plane, local_z, rank, active, slab_comm);
            computeChemicalPotential(cold, mu, nx, ny, local_z + 2, dx, dy, dz,
                                     gamma, e_AA, e_BB, e_AB);
            exchangeHalos(mu, plane, local_z, rank, active, slab_comm);
            cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_z + 2, D, dt, dx, dy, dz);
            std::swap(cold, cnew);
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double max_elapsed = 0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long milliseconds = static_cast<long>(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / (milliseconds / 1000.0) / 1e6);
    }

    if (printResults) {
        std::vector<double> result;
        if (rank == 0) {
            result.resize(gridSize);
            std::copy_n(cold.data() + plane, local_z * plane, result.data());
            for (int source = 1; source < active; ++source) {
                const size_t count = (nz / active + (static_cast<size_t>(source) < nz % active)) * plane;
                const size_t start_z = static_cast<size_t>(source) * (nz / active) +
                                       std::min(static_cast<size_t>(source), nz % active);
                for (size_t offset = 0; offset < count;) {
                    const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(std::numeric_limits<int>::max())));
                    MPI_Recv(result.data() + start_z * plane + offset, chunk, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    offset += chunk;
                }
            }
            print_results(result, "Concentration");
        } else if (local_z) {
            const size_t count = local_z * plane;
            for (size_t offset = 0; offset < count;) {
                const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(std::numeric_limits<int>::max())));
                MPI_Send(cold.data() + plane + offset, chunk, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
                offset += chunk;
            }
        }
    }
    int valid = 1;
    if (validate) {
        int local_finite = 1;
        double local_min = std::numeric_limits<double>::infinity();
        double local_max = -std::numeric_limits<double>::infinity();
        for (size_t i = plane; i < (local_z + 1) * plane && local_z; ++i) {
            const double value = cold[i];
            if (!std::isfinite(value)) local_finite = 0;
            local_min = std::min(local_min, value);
            local_max = std::max(local_max, value);
        }
        int finite;
        double min_value, max_value;
        MPI_Allreduce(&local_finite, &finite, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Reduce(&local_min, &min_value, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &max_value, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        valid = finite;
        if (rank == 0) {
            printf("Validating result...\n");
            if (!finite) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", min_value, max_value);
                if (max_value > 10.0 || min_value < -10.0) {
                    printf("Validation failed: values out of expected range\n");
                    valid = 0;
                }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if (slab_comm != MPI_COMM_NULL) MPI_Comm_free(&slab_comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
