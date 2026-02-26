#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
    const double* const cptr = c.data();
    double* const muptr = mu.data();

    const size_t sxy = nx * ny;
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);

    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t z0 = z * sxy;
            const size_t zp = (z + 1 < nz) ? (z + 1) * sxy : z0;
            const size_t zn = (z > 0) ? (z - 1) * sxy : z0;

            const size_t y0 = y * nx;
            const size_t yp = (y + 1 < ny) ? (y + 1) * nx : y0;
            const size_t yn = (y > 0) ? (y - 1) * nx : y0;

            const size_t base = z0 + y0;
            const size_t base_yp = z0 + yp;
            const size_t base_yn = z0 + yn;
            const size_t base_zp = zp + y0;
            const size_t base_zn = zn + y0;

            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = base + x;
                const double cv = cptr[idx];

                const size_t idx_xp = (x + 1 < nx) ? (idx + 1) : idx;
                const size_t idx_xn = (x > 0) ? (idx - 1) : idx;
                const size_t idx_yp = base_yp + x;
                const size_t idx_yn = base_yn + x;
                const size_t idx_zp = base_zp + x;
                const size_t idx_zn = base_zn + x;

                const double lap = (cptr[idx_xp] + cptr[idx_xn] - 2.0 * cv) * inv_dx2 +
                                   (cptr[idx_yp] + cptr[idx_yn] - 2.0 * cv) * inv_dy2 +
                                   (cptr[idx_zp] + cptr[idx_zn] - 2.0 * cv) * inv_dz2;

                muptr[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                           + 3.0 * cv + cv * cv * cv
                           - gamma * lap;
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const double* const coldptr = cold.data();
    const double* const muptr = mu.data();
    double* const cnewptr = cnew.data();

    const size_t sxy = nx * ny;
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    const double coef = dt * D;

    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t z0 = z * sxy;
            const size_t zp = (z + 1 < nz) ? (z + 1) * sxy : z0;
            const size_t zn = (z > 0) ? (z - 1) * sxy : z0;

            const size_t y0 = y * nx;
            const size_t yp = (y + 1 < ny) ? (y + 1) * nx : y0;
            const size_t yn = (y > 0) ? (y - 1) * nx : y0;

            const size_t base = z0 + y0;
            const size_t base_yp = z0 + yp;
            const size_t base_yn = z0 + yn;
            const size_t base_zp = zp + y0;
            const size_t base_zn = zn + y0;

            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = base + x;
                const double muv = muptr[idx];

                const size_t idx_xp = (x + 1 < nx) ? (idx + 1) : idx;
                const size_t idx_xn = (x > 0) ? (idx - 1) : idx;
                const size_t idx_yp = base_yp + x;
                const size_t idx_yn = base_yn + x;
                const size_t idx_zp = base_zp + x;
                const size_t idx_zn = base_zn + x;

                const double lap = (muptr[idx_xp] + muptr[idx_xn] - 2.0 * muv) * inv_dx2 +
                                   (muptr[idx_yp] + muptr[idx_yn] - 2.0 * muv) * inv_dy2 +
                                   (muptr[idx_zp] + muptr[idx_zn] - 2.0 * muv) * inv_dz2;

                cnewptr[idx] = coldptr[idx] + coef * lap;
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    double* const cptr = c.data();

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = z * (nx * ny) + y * nx;
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = base + x;
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = idx;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                cptr[idx] = -1.0 + 2.0 * pseudo;
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
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel
    {
        for (int t = 0; t < iterations; ++t) {
            // Compute chemical potential
            computeChemicalPotential(cold, mu, nx, ny, nz, dx, dy, dz,
                                    gamma, e_AA, e_BB, e_AB);

            // Update concentration
            cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz, D, dt, dx, dy, dz);

            // Swap buffers
            #pragma omp single
            {
                std::swap(cold, cnew);
            }
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
