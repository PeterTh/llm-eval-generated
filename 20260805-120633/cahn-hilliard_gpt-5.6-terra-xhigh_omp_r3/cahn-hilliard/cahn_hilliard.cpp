#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

// Evaluate a clamped-boundary stencil cell.  The interior of each row is
// handled directly in the kernels below; this is only used at x boundaries.
inline double clampedLaplacian(const double* const values, const size_t index,
                               const size_t xp, const size_t xn,
                               const size_t yp, const size_t yn,
                               const size_t zp, const size_t zn,
                               const double invDx2, const double invDy2,
                               const double invDz2) noexcept {
    const double center = values[index];
    const double cxx = (values[xp] + values[xn] - 2.0 * center) * invDx2;
    const double cyy = (values[yp] + values[yn] - 2.0 * center) * invDy2;
    const double czz = (values[zp] + values[zn] - 2.0 * center) * invDz2;
    return cxx + cyy + czz;
}

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t planeSize = nx * ny;
    const double invDx2 = 1.0 / (dx * dx);
    const double invDy2 = 1.0 / (dy * dy);
    const double invDz2 = 1.0 / (dz * dz);
    const double* const cValues = c.data();
    double* const muValues = mu.data();

    // This worksharing loop is called by every thread in the persistent team
    // in main.  Rows are the scheduling unit to retain contiguous x accesses.
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = z * planeSize + y * nx;
            const size_t previousRow = y == 0 ? row : row - nx;
            const size_t nextRow = y + 1 == ny ? row : row + nx;
            const size_t previousPlane = z == 0 ? row : row - planeSize;
            const size_t nextPlane = z + 1 == nz ? row : row + planeSize;

            if (nx == 0) {
                continue;
            }

            const auto computeCell = [&](const size_t index, const size_t xp, const size_t xn) {
                const double cv = cValues[index];
                const double laplacian = clampedLaplacian(cValues, index, xp, xn,
                                                           previousRow + (index - row), nextRow + (index - row),
                                                           previousPlane + (index - row), nextPlane + (index - row),
                                                           invDx2, invDy2, invDz2);
                muValues[index] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                + 3.0 * cv + cv * cv * cv - gamma * laplacian;
            };

            if (nx == 1) {
                computeCell(row, row, row);
                continue;
            }

            computeCell(row, row + 1, row);

            #pragma omp simd
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t index = row + x;
                const double cv = cValues[index];
                const double cxx = (cValues[index + 1] + cValues[index - 1] - 2.0 * cv) * invDx2;
                const double cyy = (cValues[nextRow + x] + cValues[previousRow + x] - 2.0 * cv) * invDy2;
                const double czz = (cValues[nextPlane + x] + cValues[previousPlane + x] - 2.0 * cv) * invDz2;
                muValues[index] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                + 3.0 * cv + cv * cv * cv - gamma * (cxx + cyy + czz);
            }

            computeCell(row + nx - 1, row + nx - 1, row + nx - 2);
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t planeSize = nx * ny;
    const double invDx2 = 1.0 / (dx * dx);
    const double invDy2 = 1.0 / (dy * dy);
    const double invDz2 = 1.0 / (dz * dz);
    const double dtD = dt * D;
    const double* const coldValues = cold.data();
    const double* const muValues = mu.data();
    double* const cnewValues = cnew.data();

    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = z * planeSize + y * nx;
            const size_t previousRow = y == 0 ? row : row - nx;
            const size_t nextRow = y + 1 == ny ? row : row + nx;
            const size_t previousPlane = z == 0 ? row : row - planeSize;
            const size_t nextPlane = z + 1 == nz ? row : row + planeSize;

            if (nx == 0) {
                continue;
            }

            const auto updateCell = [&](const size_t index, const size_t xp, const size_t xn) {
                const size_t x = index - row;
                cnewValues[index] = coldValues[index] + dtD * clampedLaplacian(
                    muValues, index, xp, xn, previousRow + x, nextRow + x,
                    previousPlane + x, nextPlane + x, invDx2, invDy2, invDz2);
            };

            if (nx == 1) {
                updateCell(row, row, row);
                continue;
            }

            updateCell(row, row + 1, row);

            #pragma omp simd
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t index = row + x;
                const double center = muValues[index];
                const double cxx = (muValues[index + 1] + muValues[index - 1] - 2.0 * center) * invDx2;
                const double cyy = (muValues[nextRow + x] + muValues[previousRow + x] - 2.0 * center) * invDy2;
                const double czz = (muValues[nextPlane + x] + muValues[previousPlane + x] - 2.0 * center) * invDz2;
                cnewValues[index] = coldValues[index] + dtD * (cxx + cyy + czz);
            }

            updateCell(row + nx - 1, row + nx - 1, row + nx - 2);
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    double* const values = c.data();

    #pragma omp parallel for schedule(static)
    for (size_t linearId = 0; linearId < vol; ++linearId) {
        // Generate pseudo-random value in [-1, 1].  This remains a pure
        // function of the cell index, so initialization is deterministic.
        const double pseudo = (((linearId + 1) * 1299709 % vol) / static_cast<double>(vol));
        values[linearId] = -1.0 + 2.0 * pseudo;
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
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
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate arrays
    std::vector<double> cold(gridSize);
    std::vector<double> cnew(gridSize);
    std::vector<double> mu(gridSize);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    // Keep one OpenMP team alive for the complete evolution.  The implicit
    // barriers on the worksharing loops preserve the two global stencil
    // phases, and the single region makes each buffer swap visible to all
    // threads before the next timestep starts.
    #pragma omp parallel
    {
        for (int t = 0; t < iterations; ++t) {
            computeChemicalPotential(cold, mu, nx, ny, nz, dx, dy, dz,
                                     gamma, e_AA, e_BB, e_AB);

            cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz, D, dt, dx, dy, dz);

            #pragma omp single
            std::swap(cold, cnew);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    if (printResults) {
        print_results(cold, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
