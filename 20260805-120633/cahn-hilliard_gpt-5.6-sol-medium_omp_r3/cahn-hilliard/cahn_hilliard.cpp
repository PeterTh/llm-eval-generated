#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// The neighbor indices are formed once per row in the stencil sweeps.  This
// avoids repeated 3-D index calculations while preserving clamped boundaries.
inline double computeLaplacian(const double* const c, const size_t center,
                               const size_t xn, const size_t xp,
                               const size_t yn, const size_t yp,
                               const size_t zn, const size_t zp,
                               const double dx2, const double dy2, const double dz2) noexcept {
    const double cxx = (c[xp] + c[xn] - 2.0 * c[center]) / dx2;
    const double cyy = (c[yp] + c[yn] - 2.0 * c[center]) / dy2;
    const double czz = (c[zp] + c[zn] - 2.0 * c[center]) / dz2;
    return cxx + cyy + czz;
}

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t plane = nx * ny;
    const size_t rows = ny * nz;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;
    const double* const input = c.data();
    double* const output = mu.data();

    #pragma omp for schedule(static)
    for (size_t row = 0; row < rows; ++row) {
        const size_t z = row / ny;
        const size_t y = row - z * ny;
        const size_t base = row * nx;
        const size_t ynBase = (y == 0) ? base : base - nx;
        const size_t ypBase = (y + 1 == ny) ? base : base + nx;
        const size_t znBase = (z == 0) ? base : base - plane;
        const size_t zpBase = (z + 1 == nz) ? base : base + plane;

        #pragma omp simd
        for (size_t x = 0; x < nx; ++x) {
            const size_t center = base + x;
            const size_t xn = (x == 0) ? center : center - 1;
            const size_t xp = (x + 1 == nx) ? center : center + 1;
            const double cv = input[center];
            const double laplacian = computeLaplacian(input, center, xn, xp,
                                                       ynBase + x, ypBase + x,
                                                       znBase + x, zpBase + x,
                                                       dx2, dy2, dz2);

            output[center] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                           + 3.0 * cv + cv * cv * cv - gamma * laplacian;
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t plane = nx * ny;
    const size_t rows = ny * nz;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;
    const double* const oldValues = cold.data();
    const double* const potential = mu.data();
    double* const newValues = cnew.data();

    #pragma omp for schedule(static)
    for (size_t row = 0; row < rows; ++row) {
        const size_t z = row / ny;
        const size_t y = row - z * ny;
        const size_t base = row * nx;
        const size_t ynBase = (y == 0) ? base : base - nx;
        const size_t ypBase = (y + 1 == ny) ? base : base + nx;
        const size_t znBase = (z == 0) ? base : base - plane;
        const size_t zpBase = (z + 1 == nz) ? base : base + plane;

        #pragma omp simd
        for (size_t x = 0; x < nx; ++x) {
            const size_t center = base + x;
            const size_t xn = (x == 0) ? center : center - 1;
            const size_t xp = (x + 1 == nx) ? center : center + 1;
            newValues[center] = oldValues[center] + dt * D *
                computeLaplacian(potential, center, xn, xp,
                                 ynBase + x, ypBase + x, znBase + x, zpBase + x,
                                 dx2, dy2, dz2);
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;

    #pragma omp parallel for schedule(static)
    for (size_t linearId = 0; linearId < vol; ++linearId) {
        // Generate the same deterministic pseudo-random value in [-1, 1].
        const double pseudo = (((linearId + 1) * 1299709) % vol) / static_cast<double>(vol);
        c[linearId] = -1.0 + 2.0 * pseudo;
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
    
    // Keep a single OpenMP team alive for all timesteps.  Alternating the
    // buffers by timestep parity avoids an additional synchronized swap;
    // only the two stencil dependency barriers remain in each timestep.
    #pragma omp parallel default(none) \
        shared(cold, cnew, mu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB, D, dt, iterations)
    {
        for (int t = 0; t < iterations; ++t) {
            if ((t & 1) == 0) {
                computeChemicalPotential(cold, mu, nx, ny, nz, dx, dy, dz,
                                         gamma, e_AA, e_BB, e_AB);
                cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz, D, dt, dx, dy, dz);
            } else {
                computeChemicalPotential(cnew, mu, nx, ny, nz, dx, dy, dz,
                                         gamma, e_AA, e_BB, e_AB);
                cahnHilliardUpdate(cold, cnew, mu, nx, ny, nz, D, dt, dx, dy, dz);
            }
        }
    }

    // Preserve the original convention that cold contains the final state.
    if (iterations > 0 && (iterations & 1) != 0) {
        std::swap(cold, cnew);
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
