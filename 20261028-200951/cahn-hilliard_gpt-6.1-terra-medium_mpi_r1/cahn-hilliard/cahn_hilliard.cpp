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
                              const size_t zbegin, const size_t zend) {
    for (size_t z = zbegin; z < zend; ++z) {
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
                        const size_t zbegin, const size_t zend) {
    for (size_t z = zbegin; z < zend; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                cnew[idx] = cold[idx] + dt * D * 
                           computeLaplacian(mu, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Exchange contiguous boundary planes. Physical boundaries use a copy of the
// boundary plane, matching the original clamped stencil.
void exchangePlanes(std::vector<double>& field, size_t plane, size_t localZ,
                    int rank, int ranks, std::vector<MPI_Request>& requests) {
    requests.clear();
    if (rank == 0) std::copy_n(field.data() + plane, plane, field.data());
    if (rank == ranks - 1)
        std::copy_n(field.data() + localZ * plane, plane, field.data() + (localZ + 1) * plane);
    // Chunk messages so the MPI count is representable even for very large planes.
    for (size_t offset = 0; offset < plane;) {
        const int count = static_cast<int>(std::min(plane - offset, static_cast<size_t>(INT_MAX)));
        MPI_Request req;
        if (rank > 0) {
            MPI_Irecv(field.data() + offset, count, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
            MPI_Isend(field.data() + plane + offset, count, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
        if (rank + 1 < ranks) {
            MPI_Irecv(field.data() + (localZ + 1) * plane + offset, count, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &req);
            requests.push_back(req);
            MPI_Isend(field.data() + localZ * plane + offset, count, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);
        }
        offset += count;
    }
}

void waitPlanes(std::vector<MPI_Request>& requests) {
    if (!requests.empty()) MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
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
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
        if (rank == 0) fprintf(stderr, "Grid dimensions must be positive\n");
        MPI_Finalize();
        return 1;
    }
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
    
    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;
    const int activeRanks = static_cast<int>(std::min(nz, static_cast<size_t>(worldSize)));
    const size_t localZ = rank < activeRanks ? nz / activeRanks + (static_cast<size_t>(rank) < nz % activeRanks) : 0;
    const size_t firstZ = rank < activeRanks ? static_cast<size_t>(rank) * (nz / activeRanks) + std::min(static_cast<size_t>(rank), nz % activeRanks) : nz;
    std::vector<double> cold(localZ ? (localZ + 2) * plane : 0);
    std::vector<double> cnew(cold.size());
    std::vector<double> mu(cold.size());

    if (rank == 0) printf("Initializing concentration field...\n");
    for (size_t z = 0; z < localZ; ++z) {
        for (size_t i = 0; i < plane; ++i) {
            const size_t linear_id = (firstZ + z) * plane + i;
            const double pseudo = (((linear_id + 1) * 1299709) % gridSize) / static_cast<double>(gridSize);
            cold[(z + 1) * plane + i] = -1.0 + 2.0 * pseudo;
        }
    }

    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    std::vector<MPI_Request> requests;
    for (int t = 0; t < iterations && localZ; ++t) {
        exchangePlanes(cold, plane, localZ, rank, activeRanks, requests);
        computeChemicalPotential(cold, mu, nx, ny, localZ + 2, dx, dy, dz,
                                 gamma, e_AA, e_BB, e_AB, 2, localZ);
        waitPlanes(requests);
        computeChemicalPotential(cold, mu, nx, ny, localZ + 2, dx, dy, dz,
                                 gamma, e_AA, e_BB, e_AB, 1, 2);
        if (localZ > 1)
            computeChemicalPotential(cold, mu, nx, ny, localZ + 2, dx, dy, dz,
                                     gamma, e_AA, e_BB, e_AB, localZ, localZ + 1);
        exchangePlanes(mu, plane, localZ, rank, activeRanks, requests);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, localZ + 2, D, dt, dx, dy, dz, 2, localZ);
        waitPlanes(requests);
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, localZ + 2, D, dt, dx, dy, dz, 1, 2);
        if (localZ > 1)
            cahnHilliardUpdate(cnew, cold, mu, nx, ny, localZ + 2, D, dt, dx, dy, dz, localZ, localZ + 1);
        std::swap(cold, cnew);
    }
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double mcups = static_cast<double>(gridSize) * iterations / maxElapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    if (printResults) {
        std::vector<double> result;
        if (rank == 0) {
            result.resize(gridSize);
            std::copy_n(cold.data() + plane, localZ * plane, result.data());
            size_t pos = localZ * plane;
            for (int source = 1; source < activeRanks; ++source) {
                size_t remaining = (nz / activeRanks + (static_cast<size_t>(source) < nz % activeRanks)) * plane;
                while (remaining) {
                    int count = static_cast<int>(std::min(remaining, static_cast<size_t>(INT_MAX)));
                    MPI_Recv(result.data() + pos, count, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    pos += count;
                    remaining -= count;
                }
            }
            print_results(result, "Concentration");
        } else if (localZ) {
            size_t pos = plane, remaining = localZ * plane;
            while (remaining) {
                int count = static_cast<int>(std::min(remaining, static_cast<size_t>(INT_MAX)));
                MPI_Send(cold.data() + pos, count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
                pos += count;
                remaining -= count;
            }
        }
    }

    int valid = 1;
    if (validate) {
        double localMin = INFINITY, localMax = -INFINITY;
        int localFinite = 1;
        for (size_t i = plane; i < (localZ + 1) * plane && localZ; ++i) {
            const double value = cold[i];
            if (!std::isfinite(value)) localFinite = 0;
            localMin = std::min(localMin, value);
            localMax = std::max(localMax, value);
        }
        double minVal, maxVal;
        int finite;
        MPI_Reduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localFinite, &finite, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating result...\n");
            if (!finite) { printf("Validation failed: found NaN or Inf value\n"); valid = 0; }
            else {
                printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
                if (maxVal > 10.0 || minVal < -10.0) {
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
