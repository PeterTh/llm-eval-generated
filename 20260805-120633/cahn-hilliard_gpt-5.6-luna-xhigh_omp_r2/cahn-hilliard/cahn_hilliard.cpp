#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t plane = nx * ny;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;
    const double* const cData = c.data();
    double* const muData = mu.data();

    // This worksharing loop is called by every thread in the persistent team
    // created in main. Static scheduling keeps each thread on contiguous rows.
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = z * plane + y * nx;
            const size_t yPrev = (y > 0) ? y - 1 : 0;
            const size_t yNext = (y + 1 < ny) ? y + 1 : y;
            const size_t zPrev = (z > 0) ? z - 1 : 0;
            const size_t zNext = (z + 1 < nz) ? z + 1 : z;

            const double* const cRow = cData + row;
            const double* const cYPrev = cData + z * plane + yPrev * nx;
            const double* const cYNext = cData + z * plane + yNext * nx;
            const double* const cZPrev = cData + zPrev * plane + y * nx;
            const double* const cZNext = cData + zNext * plane + y * nx;

            #pragma omp simd
            for (size_t x = 0; x < nx; ++x) {
                const size_t xPrev = (x > 0) ? x - 1 : 0;
                const size_t xNext = (x + 1 < nx) ? x + 1 : x;
                const double cv = cRow[x];
                const double cxx = (cRow[xNext] + cRow[xPrev] - 2.0 * cv) / dx2;
                const double cyy = (cYNext[x] + cYPrev[x] - 2.0 * cv) / dy2;
                const double czz = (cZNext[x] + cZPrev[x] - 2.0 * cv) / dz2;
                const double laplacian = cxx + cyy + czz;

                muData[row + x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                + 3.0 * cv + cv * cv * cv
                                - gamma * laplacian;
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t plane = nx * ny;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;
    const double* const coldData = cold.data();
    const double* const muData = mu.data();
    double* const cnewData = cnew.data();

    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = z * plane + y * nx;
            const size_t yPrev = (y > 0) ? y - 1 : 0;
            const size_t yNext = (y + 1 < ny) ? y + 1 : y;
            const size_t zPrev = (z > 0) ? z - 1 : 0;
            const size_t zNext = (z + 1 < nz) ? z + 1 : z;

            const double* const muRow = muData + row;
            const double* const muYPrev = muData + z * plane + yPrev * nx;
            const double* const muYNext = muData + z * plane + yNext * nx;
            const double* const muZPrev = muData + zPrev * plane + y * nx;
            const double* const muZNext = muData + zNext * plane + y * nx;

            #pragma omp simd
            for (size_t x = 0; x < nx; ++x) {
                const size_t xPrev = (x > 0) ? x - 1 : 0;
                const size_t xNext = (x + 1 < nx) ? x + 1 : x;
                const double muv = muRow[x];
                const double mxx = (muRow[xNext] + muRow[xPrev] - 2.0 * muv) / dx2;
                const double myy = (muYNext[x] + muYPrev[x] - 2.0 * muv) / dy2;
                const double mzz = (muZNext[x] + muZPrev[x] - 2.0 * muv) / dz2;

                cnewData[row + x] = coldData[row + x] + dt * D * (mxx + myy + mzz);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = z * (nx * ny) + y * nx;
            for (size_t x = 0; x < nx; ++x) {
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = row + x;
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

    // Keep the requested OpenMP team size; do not let the runtime reduce it
    // dynamically for any of the parallel regions below.
    omp_set_dynamic(0);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // Keep one OpenMP team alive for the whole time integration. Each stencil
    // has an implicit barrier, and the swap is performed after the update by
    // one thread with an implicit barrier before the next timestep.
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
