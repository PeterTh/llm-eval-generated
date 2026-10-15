#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#include <omp.h>

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Laplacian at a single point with clamped boundary conditions
static inline double laplacianPoint(const double* __restrict c, const double* __restrict rYn, const double* __restrict rYp,
                                    const double* __restrict rZn, const double* __restrict rZp,
                                    const double* __restrict row, const size_t x, const size_t xn, const size_t xp,
                                    const double dx2, const double dy2, const double dz2) noexcept {
    (void)c;
    const double cc = row[x];
    const double cxx = (row[xp] + row[xn] - 2.0 * cc) / dx2;
    const double cyy = (rYp[x] + rYn[x] - 2.0 * cc) / dy2;
    const double czz = (rZp[x] + rZn[x] - 2.0 * cc) / dz2;
    return cxx + cyy + czz;
}

// Process one row (fixed y,z): out[x] = f(in[x], laplacian(lapSrc)[x]).
// MODE 0: chemical potential, MODE 1: concentration update.
template <int MODE>
static inline void processRow(double* __restrict out, const double* __restrict cold, const double* __restrict src,
                              const size_t nx, const size_t ny, const size_t nz,
                              const size_t y, const size_t z,
                              const double dx2, const double dy2, const double dz2,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB,
                              const double D, const double dt) noexcept {
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t zn = (z > 0) ? z - 1 : 0;
    const double* __restrict row = src + idx3(0, y, z, nx, ny);
    const double* __restrict rYn = src + idx3(0, yn, z, nx, ny);
    const double* __restrict rYp = src + idx3(0, yp, z, nx, ny);
    const double* __restrict rZn = src + idx3(0, y, zn, nx, ny);
    const double* __restrict rZp = src + idx3(0, y, zp, nx, ny);
    const double* __restrict cin = cold + idx3(0, y, z, nx, ny);
    double* __restrict o = out + idx3(0, y, z, nx, ny);

    auto apply = [&](const size_t x, const double lap) {
        if constexpr (MODE == 0) {
            const double cv = cin[x];
            o[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                   + 3.0 * cv + cv * cv * cv
                   - gamma * lap;
        } else {
            o[x] = cin[x] + dt * D * lap;
        }
    };

    // x = 0 boundary
    {
        const size_t xp = (nx > 1) ? 1 : 0;
        apply(0, laplacianPoint(src, rYn, rYp, rZn, rZp, row, 0, 0, xp, dx2, dy2, dz2));
    }
    // interior
const size_t xe = (nx > 1) ? nx - 1 : 1;
#pragma omp simd
    for (size_t x = 1; x < xe; ++x) {
        const double cc = row[x];
        const double cxx = (row[x + 1] + row[x - 1] - 2.0 * cc) / dx2;
        const double cyy = (rYp[x] + rYn[x] - 2.0 * cc) / dy2;
        const double czz = (rZp[x] + rZn[x] - 2.0 * cc) / dz2;
        apply(x, cxx + cyy + czz);
    }
    // x = nx-1 boundary
    if (nx > 1) {
        const size_t x = nx - 1;
        apply(x, laplacianPoint(src, rYn, rYp, rZn, rZp, row, x, x - 1, x, dx2, dy2, dz2));
    }
}

// Initialize concentration field (parallel first-touch)
static void initializeConcentration(double* c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
#pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const double pseudo = ((((idx + 1) * 1299709) % vol) / static_cast<double>(vol));
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
    
    // Allocate arrays (uninitialized; first-touch done in parallel for NUMA locality)
    double* bufA = static_cast<double*>(aligned_alloc(64, ((gridSize * sizeof(double) + 63) / 64) * 64));
    double* bufB = static_cast<double*>(aligned_alloc(64, ((gridSize * sizeof(double) + 63) / 64) * 64));
    double* muBuf = static_cast<double*>(aligned_alloc(64, ((gridSize * sizeof(double) + 63) / 64) * 64));
    if (!bufA || !bufB || !muBuf) {
        printf("Allocation failed\n");
        return 1;
    }

    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    double* finalPtr = bufA;
    double elapsedStart = 0.0, elapsedEnd = 0.0;

    printf("Initializing concentration field...\n");
    printf("Running Cahn-Hilliard simulation...\n");

    #pragma omp parallel default(shared)
    {
        // First touch the other buffers with the same static decomposition used by the kernels
#pragma omp for collapse(2) schedule(static)
        for (size_t z = 0; z < nz; ++z)
            for (size_t y = 0; y < ny; ++y) {
                const size_t o = idx3(0, y, z, nx, ny);
                for (size_t x = 0; x < nx; ++x) { bufB[o + x] = 0.0; muBuf[o + x] = 0.0; }
            }
        initializeConcentration(bufA, nx, ny, nz);

#pragma omp single
        elapsedStart = omp_get_wtime();

        double* cold = bufA;
        double* cnew = bufB;
        for (int t = 0; t < iterations; ++t) {
#pragma omp for collapse(2) schedule(static)
            for (size_t z = 0; z < nz; ++z)
                for (size_t y = 0; y < ny; ++y)
                    processRow<0>(muBuf, cold, cold, nx, ny, nz, y, z, dx2, dy2, dz2,
                                  gamma, e_AA, e_BB, e_AB, D, dt);
#pragma omp for collapse(2) schedule(static)
            for (size_t z = 0; z < nz; ++z)
                for (size_t y = 0; y < ny; ++y)
                    processRow<1>(cnew, cold, muBuf, nx, ny, nz, y, z, dx2, dy2, dz2,
                                  gamma, e_AA, e_BB, e_AB, D, dt);
            std::swap(cold, cnew);
        }
#pragma omp single nowait
        {
            finalPtr = cold;
        }
#pragma omp barrier
#pragma omp single
        elapsedEnd = omp_get_wtime();
    }

    std::vector<double> cold(finalPtr, finalPtr + gridSize);
    free(bufA);
    free(bufB);
    free(muBuf);

    auto duration = std::chrono::milliseconds(static_cast<long long>((elapsedEnd - elapsedStart) * 1000.0));
    
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
