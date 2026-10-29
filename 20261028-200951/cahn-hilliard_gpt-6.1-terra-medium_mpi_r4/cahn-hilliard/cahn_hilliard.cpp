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
    for (size_t z = 0; z < nz; ++z) {
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
    for (size_t z = 0; z < nz; ++z) {
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

// Exchange a field's end planes while computing its interior planes.
template <typename Work>
void stencilStep(std::vector<double>& field, size_t plane, size_t localNz,
                 int rank, int ranks, Work work) {
    const int lower = rank ? rank - 1 : MPI_PROC_NULL;
    const int upper = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    if (lower == MPI_PROC_NULL)
        std::copy_n(field.data() + plane, plane, field.data());
    if (upper == MPI_PROC_NULL)
        std::copy_n(field.data() + localNz * plane, plane, field.data() + (localNz + 1) * plane);
    MPI_Request requests[4];
    MPI_Irecv(field.data(), static_cast<int>(plane), MPI_DOUBLE, lower, 1, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(field.data() + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, upper, 0, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(field.data() + plane, static_cast<int>(plane), MPI_DOUBLE, lower, 0, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(field.data() + localNz * plane, static_cast<int>(plane), MPI_DOUBLE, upper, 1, MPI_COMM_WORLD, &requests[3]);
    for (size_t z = 2; z < localNz; ++z) work(z);
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    work(1);
    if (localNz > 1) work(localNz);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (!nx || !ny || !nz || nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        if (rank == 0) fprintf(stderr, "Invalid or excessively large grid size\n");
        MPI_Finalize(); return 1;
    }
    const size_t plane = nx * ny, gridSize = plane * nz;
    // Ranks beyond the number of planes participate in collectives but own no cells.
    const int active = static_cast<int>(std::min(nz, static_cast<size_t>(ranks)));
    const size_t localNz = rank < active ? nz / active + (static_cast<size_t>(rank) < nz % active) : 0;
    const size_t zStart = rank < active ? static_cast<size_t>(rank) * (nz / active) +
                          std::min(static_cast<size_t>(rank), nz % active) : nz;
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\nRunning Cahn-Hilliard simulation...\n");
    }
    std::vector<double> cold((localNz + 2) * plane), cnew((localNz + 2) * plane), mu((localNz + 2) * plane);
    for (size_t z = 0; z < localNz; ++z)
        for (size_t i = 0; i < plane; ++i) {
            const size_t id = (zStart + z) * plane + i;
            const double pseudo = (((id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
            cold[(z + 1) * plane + i] = -1.0 + 2.0 * pseudo;
        }
    constexpr double dx = 1.0, dy = 1.0, dz = 1.0, dt = 0.01;
    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = (2.0 / 9.0), gamma = 0.5, D = 1.0;
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int t = 0; t < iterations && localNz; ++t) {
        stencilStep(cold, plane, localNz, rank, active, [&](size_t z) {
            for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = cold[idx];
                mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                        + 3.0 * cv + cv * cv * cv
                        - gamma * computeLaplacian(cold, nx, ny, localNz + 2, dx, dy, dz, x, y, z);
            }
        });
        stencilStep(mu, plane, localNz, rank, active, [&](size_t z) {
            for (size_t y = 0; y < ny; ++y) for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * computeLaplacian(mu, nx, ny, localNz + 2, dx, dy, dz, x, y, z);
            }
        });
        cold.swap(cnew);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long ms = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(gridSize) * iterations / maxElapsed / 1e6);
    }
    if (printResults) {
        // Send in chunks so global output is not limited by MPI's int counts.
        if (rank == 0) {
            std::vector<double> result(gridSize);
            std::copy_n(cold.data() + plane, localNz * plane, result.data());
            for (int r = 1; r < active; ++r) {
                const size_t count = (nz / active + (static_cast<size_t>(r) < nz % active)) * plane;
                const size_t offset = (static_cast<size_t>(r) * (nz / active) + std::min(static_cast<size_t>(r), nz % active)) * plane;
                for (size_t pos = 0; pos < count;) {
                    int n = static_cast<int>(std::min(count - pos, static_cast<size_t>(std::numeric_limits<int>::max())));
                    MPI_Recv(result.data() + offset + pos, n, MPI_DOUBLE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    pos += n;
                }
            }
            print_results(result, "Concentration");
        } else if (localNz) {
            const size_t count = localNz * plane;
            for (size_t pos = 0; pos < count;) {
                int n = static_cast<int>(std::min(count - pos, static_cast<size_t>(std::numeric_limits<int>::max())));
                MPI_Send(cold.data() + plane + pos, n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
                pos += n;
            }
        }
    }
    int valid = 1;
    if (validate) {
        int finite = 1, allFinite;
        double minVal = std::numeric_limits<double>::infinity(), maxVal = -minVal, globalMin, globalMax;
        for (size_t i = plane; i < (localNz + 1) * plane; ++i) {
            const double v = cold[i];
            if (!std::isfinite(v)) finite = 0;
            minVal = std::min(minVal, v); maxVal = std::max(maxVal, v);
        }
        MPI_Allreduce(&finite, &allFinite, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Reduce(&minVal, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&maxVal, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        valid = allFinite;
        if (rank == 0) {
            printf("Validating result...\n");
            if (!allFinite) printf("Validation failed: found NaN or Inf value\n");
            else {
                printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
                if (globalMax > 10.0 || globalMin < -10.0) {
                    printf("Validation failed: values out of expected range\n"); valid = 0;
                }
            }
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
