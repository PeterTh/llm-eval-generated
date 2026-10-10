#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

#include "../common/results_output.hpp"

// Leave allocation untouched until the parallel initialization pass. Ordinary
// vector value-initialization would place all pages on the main thread's NUMA node.
template <typename T>
struct FieldAllocator : std::allocator<T> {
    template <typename U>
    struct rebind { using other = FieldAllocator<U>; };

    FieldAllocator() noexcept = default;
    template <typename U>
    FieldAllocator(const FieldAllocator<U>&) noexcept {}

    void construct(T* p) { ::new (static_cast<void*>(p)) T; }
};

using Field = std::vector<double, FieldAllocator<double>>;

// Contiguous X tiles retain SIMD locality and expose parallel work even when
// the grid has only a few Y/Z rows. Initialization uses the same partition.
inline constexpr size_t tileWidth = 256;

// Called collectively by the persistent OpenMP team. Each pass ends with
// a barrier so every neighboring value is ready for the following stencil.
template <typename Apply>
inline void applyLaplacian(const double* field, const size_t nx, const size_t ny, const size_t nz,
                           const double dx, const double dy, const double dz, const Apply& apply) {
    if (nx == 0) return;
    const size_t plane = nx * ny;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;

    #pragma omp for collapse(3) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t begin = 0; begin < nx; begin += tileWidth) {
                const size_t offset = z * plane + y * nx;
                const double* row = field + offset;
                const double* yp = row + ((y + 1 < ny) ? nx : 0);
                const double* yn = row - ((y > 0) ? nx : 0);
                const double* zp = row + ((z + 1 < nz) ? plane : 0);
                const double* zn = row - ((z > 0) ? plane : 0);

                const auto applyAt = [&](const size_t x, const size_t xp, const size_t xn) {
                    const double cxx = (row[xp] + row[xn] - 2.0 * row[x]) / dx2;
                    const double cyy = (yp[x] + yn[x] - 2.0 * row[x]) / dy2;
                    const double czz = (zp[x] + zn[x] - 2.0 * row[x]) / dz2;
                    apply(offset + x, cxx + cyy + czz);
                };

                // Keep clamped endpoints out of the vectorized interior.
                if (begin == 0) applyAt(0, (nx > 1) ? 1 : 0, 0);
                const size_t first = std::max(begin, size_t{1});
                const size_t end = std::min(begin + tileWidth, nx - 1);
                #pragma omp simd
                for (size_t x = first; x < end; ++x) {
                    applyAt(x, x + 1, x - 1);
                }
                if (nx > 1 && nx - begin <= tileWidth) applyAt(nx - 1, nx - 1, nx - 2);
            }
        }
    }
}

// Compute chemical potential
inline void computeChemicalPotential(const double* c, double* mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    applyLaplacian(c, nx, ny, nz, dx, dy, dz, [&](const size_t idx, const double laplacian) {
        const double cv = c[idx];
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                  + 3.0 * cv + cv * cv * cv - gamma * laplacian;
    });
}

// Cahn-Hilliard update step
inline void cahnHilliardUpdate(double* cnew, const double* cold, const double* mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    applyLaplacian(mu, nx, ny, nz, dx, dy, dz, [&](const size_t idx, const double laplacian) {
        cnew[idx] = cold[idx] + dt * D * laplacian;
    });
}

// Initialize concentration field
inline void initializeConcentration(Field& c, Field& cnew, Field& mu,
                                    const size_t nx, const size_t ny, const size_t nz) {
    const size_t plane = nx * ny;
    const size_t vol = plane * nz;

    // Match the stencil's partition so each thread first touches its own pages.
    #pragma omp for collapse(3) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t begin = 0; begin < nx; begin += tileWidth) {
                const size_t offset = z * plane + y * nx;
                const size_t end = std::min(begin + tileWidth, nx);
                for (size_t x = begin; x < end; ++x) {
                    const size_t idx = offset + x;
                    // The generator gives the same field for every thread count.
                    const double pseudo = (((idx + 1) * 1299709) % vol) / static_cast<double>(vol);
                    c[idx] = -1.0 + 2.0 * pseudo;
                    cnew[idx] = 0.0;
                    mu[idx] = 0.0;
                }
            }
        }
    }
}

bool validateResult(const Field& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
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
    Field cold(gridSize);
    Field cnew(gridSize);
    Field mu(gridSize);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    std::chrono::high_resolution_clock::time_point start;
    
    #pragma omp parallel default(none) shared(cold, cnew, mu, start) \
        firstprivate(nx, ny, nz, iterations, dx, dy, dz, dt, D, gamma, e_AA, e_BB, e_AB)
    {
        initializeConcentration(cold, cnew, mu, nx, ny, nz);
        #pragma omp single
        {
            printf("Running Cahn-Hilliard simulation...\n");
            start = std::chrono::high_resolution_clock::now();
        }

        double* current = cold.data();
        double* next = cnew.data();
        double* potential = mu.data();
        for (int t = 0; t < iterations; ++t) {
            computeChemicalPotential(current, potential, nx, ny, nz, dx, dy, dz,
                                     gamma, e_AA, e_BB, e_AB);
            cahnHilliardUpdate(next, current, potential, nx, ny, nz, D, dt, dx, dy, dz);

            // Pointers are private to each thread; the update barrier protects
            // the buffers, so swapping requires no single region or extra barrier.
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
