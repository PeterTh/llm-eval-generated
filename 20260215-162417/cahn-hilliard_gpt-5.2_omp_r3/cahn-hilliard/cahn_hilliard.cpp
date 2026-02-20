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

// Compute Laplacian with clamped boundary conditions (optimized index math)
inline double computeLaplacianClamped(const double* __restrict__ a,
                                     const size_t nx, const size_t ny, const size_t nz,
                                     const double invdx2, const double invdy2, const double invdz2,
                                     const size_t x, const size_t y, const size_t z) {
    (void)nz;
    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zp = (z + 1 < nz) ? z + 1 : z;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const size_t xy = nx * ny;
    const size_t idx = z * xy + y * nx + x;

    const double center = a[idx];
    const double cxx = (a[z * xy + y * nx + xp] + a[z * xy + y * nx + xn] - 2.0 * center) * invdx2;
    const double cyy = (a[z * xy + yp * nx + x] + a[z * xy + yn * nx + x] - 2.0 * center) * invdy2;
    const double czz = (a[zp * xy + y * nx + x] + a[zn * xy + y * nx + x] - 2.0 * center) * invdz2;
    return cxx + cyy + czz;
}

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);
    const double* __restrict__ cp = c.data();
    double* __restrict__ mup = mu.data();

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double cv = cp[idx];
                const double lap = computeLaplacianClamped(cp, nx, ny, nz, invdx2, invdy2, invdz2, x, y, z);

                mup[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
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
    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);
    const double* __restrict__ mup = mu.data();
    const double* __restrict__ coldp = cold.data();
    double* __restrict__ cnewp = cnew.data();

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                const double lap = computeLaplacianClamped(mup, nx, ny, nz, invdx2, invdy2, invdz2, x, y, z);
                cnewp[idx] = coldp[idx] + dt * D * lap;
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
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                cp[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const double* __restrict__ cp = c.data();
    const size_t n = c.size();

    // Check for NaN or Inf
    int bad = 0;
    #pragma omp parallel for reduction(|:bad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const double v = cp[i];
        bad |= (std::isnan(v) || std::isinf(v)) ? 1 : 0;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = cp[0];
    double maxVal = cp[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const double v = cp[i];
        minVal = std::min(minVal, v);
        maxVal = std::max(maxVal, v);
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

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    std::chrono::high_resolution_clock::time_point start, end;

    const size_t xy = nx * ny;
    const size_t vol = gridSize;
    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double invdz2 = 1.0 / (dz * dz);

    #pragma omp parallel proc_bind(spread)
    {
        // First-touch initialization of concentration field
        double* __restrict__ coldp_init = cold.data();
        #pragma omp for collapse(2) schedule(static)
        for (size_t z = 0; z < nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                const size_t zoff = z * xy;
                const size_t yoff = zoff + y * nx;
                for (size_t x = 0; x < nx; ++x) {
                    const size_t idx = yoff + x;
                    // Generate pseudo-random value in [-1, 1]
                    const size_t linear_id = idx;
                    const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                    coldp_init[idx] = -1.0 + 2.0 * pseudo;
                }
            }
        }

        #pragma omp single
        start = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < iterations; ++t) {
            const double* __restrict__ cp = cold.data();
            double* __restrict__ mup = mu.data();

            // Compute chemical potential
            #pragma omp for collapse(2) schedule(static)
            for (size_t z = 0; z < nz; ++z) {
                for (size_t y = 0; y < ny; ++y) {
                    const size_t zoff = z * xy;
                    const size_t zoffp = (z + 1 < nz) ? (zoff + xy) : zoff;
                    const size_t zoffn = (z > 0) ? (zoff - xy) : zoff;
                    const size_t yz = y * nx;
                    const size_t yoff = zoff + yz;
                    const size_t yoffp = (y + 1 < ny) ? (yoff + nx) : yoff;
                    const size_t yoffn = (y > 0) ? (yoff - nx) : yoff;
                    const size_t yoff_zp = zoffp + yz;
                    const size_t yoff_zn = zoffn + yz;
                    for (size_t x = 0; x < nx; ++x) {
                        const size_t idx = yoff + x;
                        const double cv = cp[idx];
                        const size_t idx_xp = idx + ((x + 1 < nx) ? 1 : 0);
                        const size_t idx_xn = idx - ((x > 0) ? 1 : 0);

                        const double lap =
                            (cp[idx_xp] + cp[idx_xn] - 2.0 * cv) * invdx2 +
                            (cp[yoffp + x] + cp[yoffn + x] - 2.0 * cv) * invdy2 +
                            (cp[yoff_zp + x] + cp[yoff_zn + x] - 2.0 * cv) * invdz2;

                        mup[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                 + 3.0 * cv + cv * cv * cv
                                 - gamma * lap;
                    }
                }
            }

            const double* __restrict__ coldp = cold.data();
            const double* __restrict__ mup_r = mu.data();
            double* __restrict__ cnewp = cnew.data();

            // Update concentration
            #pragma omp for collapse(2) schedule(static)
            for (size_t z = 0; z < nz; ++z) {
                for (size_t y = 0; y < ny; ++y) {
                    const size_t zoff = z * xy;
                    const size_t zoffp = (z + 1 < nz) ? (zoff + xy) : zoff;
                    const size_t zoffn = (z > 0) ? (zoff - xy) : zoff;
                    const size_t yz = y * nx;
                    const size_t yoff = zoff + yz;
                    const size_t yoffp = (y + 1 < ny) ? (yoff + nx) : yoff;
                    const size_t yoffn = (y > 0) ? (yoff - nx) : yoff;
                    const size_t yoff_zp = zoffp + yz;
                    const size_t yoff_zn = zoffn + yz;
                    for (size_t x = 0; x < nx; ++x) {
                        const size_t idx = yoff + x;
                        const double center = mup_r[idx];
                        const size_t idx_xp = idx + ((x + 1 < nx) ? 1 : 0);
                        const size_t idx_xn = idx - ((x > 0) ? 1 : 0);

                        const double lap =
                            (mup_r[idx_xp] + mup_r[idx_xn] - 2.0 * center) * invdx2 +
                            (mup_r[yoffp + x] + mup_r[yoffn + x] - 2.0 * center) * invdy2 +
                            (mup_r[yoff_zp + x] + mup_r[yoff_zn + x] - 2.0 * center) * invdz2;

                        cnewp[idx] = coldp[idx] + dt * D * lap;
                    }
                }
            }

            // Swap buffers
            #pragma omp single
            std::swap(cold, cnew);
        }

        #pragma omp barrier
        #pragma omp single
        end = std::chrono::high_resolution_clock::now();
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
