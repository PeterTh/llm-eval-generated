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

// Compute Laplacian with clamped boundary conditions (Neumann/zero-gradient)
static inline double laplacian_clamped(const double* __restrict f,
                                      const size_t idx,
                                      const size_t x, const size_t y, const size_t z,
                                      const size_t nx, const size_t ny, const size_t nz,
                                      const size_t strideY, const size_t strideZ,
                                      const double inv_dx2, const double inv_dy2, const double inv_dz2) noexcept {
    const size_t idx_xp = idx + ((x + 1 < nx) ? 1 : 0);
    const size_t idx_xn = idx - ((x > 0) ? 1 : 0);
    const size_t idx_yp = idx + ((y + 1 < ny) ? strideY : 0);
    const size_t idx_yn = idx - ((y > 0) ? strideY : 0);
    const size_t idx_zp = idx + ((z + 1 < nz) ? strideZ : 0);
    const size_t idx_zn = idx - ((z > 0) ? strideZ : 0);

    const double c0 = f[idx];
    const double cxx = (f[idx_xp] + f[idx_xn] - 2.0 * c0) * inv_dx2;
    const double cyy = (f[idx_yp] + f[idx_yn] - 2.0 * c0) * inv_dy2;
    const double czz = (f[idx_zp] + f[idx_zn] - 2.0 * c0) * inv_dz2;
    return cxx + cyy + czz;
}

// Compute chemical potential
void computeChemicalPotential(const std::vector<double>& c, std::vector<double>& mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t strideY = nx;
    const size_t strideZ = nx * ny;
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);

    const double* __restrict cp = c.data();
    double* __restrict mup = mu.data();

    // Small grids: everything is boundary-like.
    if (nx < 3 || ny < 3 || nz < 3) {
#pragma omp parallel for collapse(3) schedule(static)
        for (size_t z = 0; z < nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t idx = z * strideZ + y * strideY + x;
                    const double cv = cp[idx];
                    const double lap = laplacian_clamped(cp, idx, x, y, z, nx, ny, nz, strideY, strideZ, inv_dx2, inv_dy2, inv_dz2);
                    mup[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv - gamma * lap;
                }
            }
        }
        return;
    }

#pragma omp parallel
    {
        // Interior: branch-free stencil, vectorizable.
#pragma omp for collapse(2) schedule(static) nowait
        for (size_t z = 1; z < nz - 1; ++z) {
            for (size_t y = 1; y < ny - 1; ++y) {
                const size_t base = z * strideZ + y * strideY;
#pragma omp simd
                for (size_t x = 1; x < nx - 1; ++x) {
                    const size_t idx = base + x;
                    const double cv = cp[idx];
                    const double c0 = cv;
                    const double lap = (cp[idx + 1] + cp[idx - 1] - 2.0 * c0) * inv_dx2 +
                                       (cp[idx + strideY] + cp[idx - strideY] - 2.0 * c0) * inv_dy2 +
                                       (cp[idx + strideZ] + cp[idx - strideZ] - 2.0 * c0) * inv_dz2;
                    mup[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv - gamma * lap;
                }
            }
        }

        // Boundary planes z=0 and z=nz-1
#pragma omp for collapse(2) schedule(static) nowait
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx0 = y * strideY + x;
                const double cv0 = cp[idx0];
                const double lap0 = laplacian_clamped(cp, idx0, x, y, 0, nx, ny, nz, strideY, strideZ, inv_dx2, inv_dy2, inv_dz2);
                mup[idx0] = 4.5 * ((cv0 + 1.0) * e_AA + (cv0 - 1.0) * e_BB - 2.0 * cv0 * e_AB) + 3.0 * cv0 + cv0 * cv0 * cv0 - gamma * lap0;

                if (nz > 1) {
                    const size_t z = nz - 1;
                    const size_t idx1 = z * strideZ + y * strideY + x;
                    const double cv1 = cp[idx1];
                    const double lap1 = laplacian_clamped(cp, idx1, x, y, z, nx, ny, nz, strideY, strideZ, inv_dx2, inv_dy2, inv_dz2);
                    mup[idx1] = 4.5 * ((cv1 + 1.0) * e_AA + (cv1 - 1.0) * e_BB - 2.0 * cv1 * e_AB) + 3.0 * cv1 + cv1 * cv1 * cv1 - gamma * lap1;
                }
            }
        }

        // Remaining boundary for z in (1..nz-2): y=0/ny-1 and x=0/nx-1
#pragma omp for collapse(2) schedule(static)
        for (size_t z = 1; z < nz - 1; ++z) {
            for (size_t x = 0; x < nx; ++x) {
                // y = 0
                {
                    const size_t y = 0;
                    const size_t idx = z * strideZ + x;
                    const double cv = cp[idx];
                    const double lap = laplacian_clamped(cp, idx, x, y, z, nx, ny, nz, strideY, strideZ, inv_dx2, inv_dy2, inv_dz2);
                    mup[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv - gamma * lap;
                }
                // y = ny - 1
                {
                    const size_t y = ny - 1;
                    const size_t idx = z * strideZ + y * strideY + x;
                    const double cv = cp[idx];
                    const double lap = laplacian_clamped(cp, idx, x, y, z, nx, ny, nz, strideY, strideZ, inv_dx2, inv_dy2, inv_dz2);
                    mup[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv - gamma * lap;
                }
            }
        }

#pragma omp for collapse(2) schedule(static)
        for (size_t z = 1; z < nz - 1; ++z) {
            for (size_t y = 1; y < ny - 1; ++y) {
                // x = 0
                {
                    const size_t x = 0;
                    const size_t idx = z * strideZ + y * strideY + x;
                    const double cv = cp[idx];
                    const double lap = laplacian_clamped(cp, idx, x, y, z, nx, ny, nz, strideY, strideZ, inv_dx2, inv_dy2, inv_dz2);
                    mup[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv - gamma * lap;
                }
                // x = nx - 1
                {
                    const size_t x = nx - 1;
                    const size_t idx = z * strideZ + y * strideY + x;
                    const double cv = cp[idx];
                    const double lap = laplacian_clamped(cp, idx, x, y, z, nx, ny, nz, strideY, strideZ, inv_dx2, inv_dy2, inv_dz2);
                    mup[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) + 3.0 * cv + cv * cv * cv - gamma * lap;
                }
            }
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(std::vector<double>& cnew, const std::vector<double>& cold,
                        const std::vector<double>& mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t strideY = nx;
    const size_t strideZ = nx * ny;
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    const double dtD = dt * D;

    const double* __restrict mup = mu.data();
    const double* __restrict coldp = cold.data();
    double* __restrict cnewp = cnew.data();

    if (nx < 3 || ny < 3 || nz < 3) {
#pragma omp parallel for collapse(3) schedule(static)
        for (size_t z = 0; z < nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t idx = z * strideZ + y * strideY + x;
                    const double lap = laplacian_clamped(mup, idx, x, y, z, nx, ny, nz, strideY, strideZ, inv_dx2, inv_dy2, inv_dz2);
                    cnewp[idx] = coldp[idx] + dtD * lap;
                }
            }
        }
        return;
    }

#pragma omp parallel
    {
#pragma omp for collapse(2) schedule(static) nowait
        for (size_t z = 1; z < nz - 1; ++z) {
            for (size_t y = 1; y < ny - 1; ++y) {
                const size_t base = z * strideZ + y * strideY;
#pragma omp simd
                for (size_t x = 1; x < nx - 1; ++x) {
                    const size_t idx = base + x;
                    const double m0 = mup[idx];
                    const double lap = (mup[idx + 1] + mup[idx - 1] - 2.0 * m0) * inv_dx2 +
                                       (mup[idx + strideY] + mup[idx - strideY] - 2.0 * m0) * inv_dy2 +
                                       (mup[idx + strideZ] + mup[idx - strideZ] - 2.0 * m0) * inv_dz2;
                    cnewp[idx] = coldp[idx] + dtD * lap;
                }
            }
        }

        // Boundary planes z=0 and z=nz-1
#pragma omp for collapse(2) schedule(static) nowait
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx0 = y * strideY + x;
                const double lap0 = laplacian_clamped(mup, idx0, x, y, 0, nx, ny, nz, strideY, strideZ, inv_dx2, inv_dy2, inv_dz2);
                cnewp[idx0] = coldp[idx0] + dtD * lap0;

                if (nz > 1) {
                    const size_t z = nz - 1;
                    const size_t idx1 = z * strideZ + y * strideY + x;
                    const double lap1 = laplacian_clamped(mup, idx1, x, y, z, nx, ny, nz, strideY, strideZ, inv_dx2, inv_dy2, inv_dz2);
                    cnewp[idx1] = coldp[idx1] + dtD * lap1;
                }
            }
        }

        // Remaining boundary for z in (1..nz-2): y=0/ny-1 and x=0/nx-1
#pragma omp for collapse(2) schedule(static)
        for (size_t z = 1; z < nz - 1; ++z) {
            for (size_t x = 0; x < nx; ++x) {
                // y = 0
                {
                    const size_t y = 0;
                    const size_t idx = z * strideZ + x;
                    const double lap = laplacian_clamped(mup, idx, x, y, z, nx, ny, nz, strideY, strideZ, inv_dx2, inv_dy2, inv_dz2);
                    cnewp[idx] = coldp[idx] + dtD * lap;
                }
                // y = ny - 1
                {
                    const size_t y = ny - 1;
                    const size_t idx = z * strideZ + y * strideY + x;
                    const double lap = laplacian_clamped(mup, idx, x, y, z, nx, ny, nz, strideY, strideZ, inv_dx2, inv_dy2, inv_dz2);
                    cnewp[idx] = coldp[idx] + dtD * lap;
                }
            }
        }

#pragma omp for collapse(2) schedule(static)
        for (size_t z = 1; z < nz - 1; ++z) {
            for (size_t y = 1; y < ny - 1; ++y) {
                // x = 0
                {
                    const size_t x = 0;
                    const size_t idx = z * strideZ + y * strideY + x;
                    const double lap = laplacian_clamped(mup, idx, x, y, z, nx, ny, nz, strideY, strideZ, inv_dx2, inv_dy2, inv_dz2);
                    cnewp[idx] = coldp[idx] + dtD * lap;
                }
                // x = nx - 1
                {
                    const size_t x = nx - 1;
                    const size_t idx = z * strideZ + y * strideY + x;
                    const double lap = laplacian_clamped(mup, idx, x, y, z, nx, ny, nz, strideY, strideZ, inv_dx2, inv_dy2, inv_dz2);
                    cnewp[idx] = coldp[idx] + dtD * lap;
                }
            }
        }
    }
}

// Initialize concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    const size_t strideY = nx;
    const size_t strideZ = nx * ny;
    double* __restrict cp = c.data();

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = z * strideZ + y * strideY;
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
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotential(cold, mu, nx, ny, nz, dx, dy, dz, 
                                gamma, e_AA, e_BB, e_AB);
        
        // Update concentration
        cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz, D, dt, dx, dy, dz);
        
        // Swap buffers
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
