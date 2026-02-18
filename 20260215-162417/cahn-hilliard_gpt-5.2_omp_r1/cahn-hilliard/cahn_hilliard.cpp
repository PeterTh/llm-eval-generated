#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian with clamped boundary conditions (optimized for inner loops)
inline double computeLaplacianClampedIdx(const double* __restrict__ c,
                                        const size_t nx, const size_t ny, const size_t nz,
                                        const size_t sy, const size_t sz,
                                        const double invdx2, const double invdy2, const double invdz2,
                                        const size_t idx, const size_t x, const size_t y, const size_t z) noexcept {
    const size_t xp = (x + 1 < nx) ? (idx + 1) : idx;
    const size_t xn = (x > 0) ? (idx - 1) : idx;
    const size_t yp = (y + 1 < ny) ? (idx + sy) : idx;
    const size_t yn = (y > 0) ? (idx - sy) : idx;
    const size_t zp = (z + 1 < nz) ? (idx + sz) : idx;
    const size_t zn = (z > 0) ? (idx - sz) : idx;

    const double c0 = c[idx];
    return (c[xp] + c[xn] - 2.0 * c0) * invdx2 +
           (c[yp] + c[yn] - 2.0 * c0) * invdy2 +
           (c[zp] + c[zn] - 2.0 * c0) * invdz2;
}

// Compute chemical potential (expects to be called inside an OpenMP parallel region)
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);

    const double k1 = 4.5 * (e_AA + e_BB - 2.0 * e_AB) + 3.0;
    const double k0 = 4.5 * (e_AA - e_BB);

    const double* __restrict__ cp = c.data();
    double* __restrict__ mup = mu.data();

    const size_t sy = nx;
    const size_t sz = nx * ny;

    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = z * sz + y * sy;
            #pragma omp simd
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = base + x;
                const double cv = cp[idx];
                const double lap = computeLaplacianClampedIdx(cp, nx, ny, nz, sy, sz, invdx2, invdy2, invdz2, idx, x, y, z);
                mup[idx] = k0 + k1 * cv + cv * cv * cv - gamma * lap;
            }
        }
    }
}

// Cahn-Hilliard update step (expects to be called inside an OpenMP parallel region)
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);
    const double alpha = dt * D;

    const double* __restrict__ coldp = cold.data();
    const double* __restrict__ mup = mu.data();
    double* __restrict__ cnewp = cnew.data();

    const size_t sy = nx;
    const size_t sz = nx * ny;

    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = z * sz + y * sy;
            #pragma omp simd
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = base + x;
                cnewp[idx] = coldp[idx] + alpha *
                             computeLaplacianClampedIdx(mup, nx, ny, nz, sy, sz, invdx2, invdy2, invdz2, idx, x, y, z);
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    double* __restrict__ cp = c.data();

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = z * (nx * ny) + y * nx;
            #pragma omp simd
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = base + x;
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = idx;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                cp[idx] = -1.0 + 2.0 * pseudo;
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
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    #pragma omp parallel
    {
        #pragma omp single
        { start = std::chrono::high_resolution_clock::now(); }

        for (int t = 0; t < iterations; ++t) {
            // Compute chemical potential
            computeChemicalPotential(cold, mu, nx, ny, nz, dx, dy, dz,
                                    gamma, e_AA, e_BB, e_AB);

            // Update concentration
            cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz, D, dt, dx, dy, dz);

            // Swap buffers
            #pragma omp single
            { std::swap(cold, cnew); }
        }

        #pragma omp single
        { end = std::chrono::high_resolution_clock::now(); }
    }
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
