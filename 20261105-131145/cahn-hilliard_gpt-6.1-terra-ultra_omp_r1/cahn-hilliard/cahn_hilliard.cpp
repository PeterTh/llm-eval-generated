#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Evaluate a cell using precomputed offsets for the neighboring rows and planes.
inline double laplacian(const double* field, const size_t i, const size_t xm, const size_t xp,
                        const size_t ym, const size_t yp, const size_t zm, const size_t zp,
                        const double dx2, const double dy2, const double dz2) {
    const double center = field[i];
    const double cxx = (field[xp] + field[xm] - 2.0 * center) / dx2;
    const double cyy = (field[yp] + field[ym] - 2.0 * center) / dy2;
    const double czz = (field[zp] + field[zm] - 2.0 * center) / dz2;
    return cxx + cyy + czz;
}

// Called by every thread in the simulation's parallel region. The implicit
// barrier at the end makes the entire potential available to the update.
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const double* input = c.data();
    double* output = mu.data();
    const size_t plane = nx * ny;
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = z * plane + y * nx;
            const size_t ym = z * plane + (y > 0 ? y - 1 : y) * nx;
            const size_t yp = z * plane + (y + 1 < ny ? y + 1 : y) * nx;
            const size_t zm = (z > 0 ? z - 1 : z) * plane + y * nx;
            const size_t zp = (z + 1 < nz ? z + 1 : z) * plane + y * nx;
            const auto cell = [&](const size_t x, const size_t xm, const size_t xp) {
                const size_t i = row + x;
                const double cv = input[i];
                output[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                            + 3.0 * cv + cv * cv * cv
                            - gamma * laplacian(input, i, row + xm, row + xp,
                                                ym + x, yp + x, zm + x, zp + x, dx2, dy2, dz2);
            };
            if (nx != 0) {
                cell(0, 0, nx > 1 ? 1 : 0);
                #pragma omp simd
                for (size_t x = 1; x < nx - 1; ++x) {
                    cell(x, x - 1, x + 1);
                }
                if (nx > 1) cell(nx - 1, nx - 2, nx - 1);
            }
        }
    }
}

// The implicit barrier completes the new concentration before the next step.
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    double* output = cnew.data();
    const double* input = cold.data();
    const double* potential = mu.data();
    const size_t plane = nx * ny;
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = z * plane + y * nx;
            const size_t ym = z * plane + (y > 0 ? y - 1 : y) * nx;
            const size_t yp = z * plane + (y + 1 < ny ? y + 1 : y) * nx;
            const size_t zm = (z > 0 ? z - 1 : z) * plane + y * nx;
            const size_t zp = (z + 1 < nz ? z + 1 : z) * plane + y * nx;
            const auto cell = [&](const size_t x, const size_t xm, const size_t xp) {
                const size_t i = row + x;
                output[i] = input[i] + dt * D *
                    laplacian(potential, i, row + xm, row + xp,
                              ym + x, yp + x, zm + x, zp + x, dx2, dy2, dz2);
            };
            if (nx != 0) {
                cell(0, 0, nx > 1 ? 1 : 0);
                #pragma omp simd
                for (size_t x = 1; x < nx - 1; ++x) {
                    cell(x, x - 1, x + 1);
                }
                if (nx > 1) cell(nx - 1, nx - 2, nx - 1);
            }
        }
    }
}

void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < vol; ++i) {
        const double pseudo = (((i + 1) * 1299709) % vol) / static_cast<double>(vol);
        c[i] = -1.0 + 2.0 * pseudo;
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
    
    #pragma omp parallel
    {
        // Each thread swaps its own pointers after both worksharing barriers.
        auto* current = &cold;
        auto* next = &cnew;
        for (int t = 0; t < iterations; ++t) {
            computeChemicalPotential(*current, mu, nx, ny, nz, dx, dy, dz,
                                     gamma, e_AA, e_BB, e_AB);
            cahnHilliardUpdate(*next, *current, mu, nx, ny, nz, D, dt, dx, dy, dz);
            std::swap(current, next);
        }
    }
    if (iterations > 0 && iterations % 2 != 0) std::swap(cold, cnew);

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
