#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
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
                              const double gamma, const double e_AA, const double e_BB, const double e_AB,
                              const size_t begin, const size_t end) {
    for (size_t z = begin; z < end; ++z) {
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
                        const double D, const double dt, const double dx, const double dy, const double dz,
                        const size_t begin, const size_t end) {
    for (size_t z = begin; z < end; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
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
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

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
    
    if (nx == 0 || ny == 0 || nz == 0 || nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz || nx * ny > INT_MAX) {
        if (rank == 0) fprintf(stderr, "Invalid grid dimensions or MPI halo exceeds count limit\n");
        MPI_Finalize();
        return 1;
    }
    const int active = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
    if (rank >= active) { MPI_Finalize(); return 0; }
    const size_t plane = nx * ny;
    const size_t local_nz = nz / active + (static_cast<size_t>(rank) < nz % active);
    const size_t z_start = static_cast<size_t>(rank) * (nz / active) +
                           std::min(static_cast<size_t>(rank), nz % active);
    const size_t gridSize = plane * nz;
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
    std::vector<double> cold((local_nz + 2) * plane), cnew(cold.size()), mu(cold.size());
    for (size_t i = 0; i < local_nz * plane; ++i) {
        const size_t linear_id = z_start * plane + i;
        const double pseudo = (((linear_id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
        cold[plane + i] = -1.0 + 2.0 * pseudo;
    }
    const int prev = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank == active - 1 ? MPI_PROC_NULL : rank + 1;
    auto exchange = [&](std::vector<double>& field, MPI_Request* requests) {
        if (prev == MPI_PROC_NULL) std::copy_n(field.data() + plane, plane, field.data());
        if (next == MPI_PROC_NULL) std::copy_n(field.data() + local_nz * plane, plane, field.data() + (local_nz + 1) * plane);
        MPI_Irecv(field.data(), static_cast<int>(plane), MPI_DOUBLE, prev, 1, comm, &requests[0]);
        MPI_Irecv(field.data() + (local_nz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, next, 0, comm, &requests[1]);
        MPI_Isend(field.data() + plane, static_cast<int>(plane), MPI_DOUBLE, prev, 0, comm, &requests[2]);
        MPI_Isend(field.data() + local_nz * plane, static_cast<int>(plane), MPI_DOUBLE, next, 1, comm, &requests[3]);
    };
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations; ++t) {
        MPI_Request requests[4];
        exchange(cold, requests);
        if (local_nz > 2)
            computeChemicalPotential(cold, mu, nx, ny, local_nz + 2, dx, dy, dz, gamma, e_AA, e_BB, e_AB, 2, local_nz);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        computeChemicalPotential(cold, mu, nx, ny, local_nz + 2, dx, dy, dz, gamma, e_AA, e_BB, e_AB, 1, 2);
        if (local_nz > 1)
            computeChemicalPotential(cold, mu, nx, ny, local_nz + 2, dx, dy, dz, gamma, e_AA, e_BB, e_AB, local_nz, local_nz + 1);
        exchange(mu, requests);
        if (local_nz > 2)
            cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz + 2, D, dt, dx, dy, dz, 2, local_nz);
        MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz + 2, D, dt, dx, dy, dz, 1, 2);
        if (local_nz > 1)
            cahnHilliardUpdate(cnew, cold, mu, nx, ny, local_nz + 2, D, dt, dx, dy, dz, local_nz, local_nz + 1);
        std::swap(cold, cnew);
    }
    double elapsed = MPI_Wtime() - start, max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000));
        printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(gridSize) * iterations / max_elapsed / 1e6);
    }
    if (printResults) {
        std::vector<double> result;
        if (rank == 0) {
            result.resize(gridSize);
            std::copy_n(cold.data() + plane, local_nz * plane, result.data());
            size_t offset = local_nz * plane;
            for (int r = 1; r < active; ++r) {
                size_t count = (nz / active + (static_cast<size_t>(r) < nz % active)) * plane;
                while (count) {
                    int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(result.data() + offset, chunk, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
                    offset += chunk; count -= chunk;
                }
            }
            print_results(result, "Concentration");
        } else {
            size_t offset = plane, count = local_nz * plane;
            while (count) {
                int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
                MPI_Send(cold.data() + offset, chunk, MPI_DOUBLE, 0, 2, comm);
                offset += chunk; count -= chunk;
            }
        }
    }
    int valid = 1;
    if (validate) {
        double local_min = cold[plane], local_max = cold[plane];
        int finite = 1;
        for (size_t i = plane; i < (local_nz + 1) * plane; ++i) {
            if (!std::isfinite(cold[i])) finite = 0;
            local_min = std::min(local_min, cold[i]);
            local_max = std::max(local_max, cold[i]);
        }
        double global_min, global_max;
        int all_finite;
        MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&finite, &all_finite, 1, MPI_INT, MPI_MIN, 0, comm);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!all_finite) { printf("Validation failed: found NaN or Inf value\n"); valid = 0; }
            else {
                printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
                if (global_max > 10.0 || global_min < -10.0) { printf("Validation failed: values out of expected range\n"); valid = 0; }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}
