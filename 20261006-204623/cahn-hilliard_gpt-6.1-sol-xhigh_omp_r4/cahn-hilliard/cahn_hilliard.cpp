#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

#ifndef _OPENMP
#error "This benchmark requires OpenMP support."
#endif

// Row pointers keep boundary handling and index arithmetic out of the SIMD loop.
inline double computeLaplacian(const double* row, const double* yn, const double* yp,
                              const double* zn, const double* zp,
                              const size_t x, const size_t xn, const size_t xp,
                              const double dx2, const double dy2, const double dz2) noexcept {
    const double cv = row[x];
    const double cxx = (row[xp] + row[xn] - 2.0 * cv) / dx2;
    const double cyy = (yp[x] + yn[x] - 2.0 * cv) / dy2;
    const double czz = (zp[x] + zn[x] - 2.0 * cv) / dz2;
    return cxx + cyy + czz;
}

// Thread-private potential tiles are sized for cache reuse. Halos contain real
// neighboring cells, with clamping only at the physical domain boundary.
struct TileShape {
    size_t x, y, z;
};

inline size_t tileCount(const size_t extent, const size_t width) noexcept {
    return extent / width + (extent % width != 0);
}

TileShape chooseTileShape(const size_t nx, const size_t ny, const size_t nz) {
    TileShape shape{std::max(size_t{1}, std::min(nx, size_t{512})),
                    std::max(size_t{1}, std::min(ny, size_t{8})),
                    std::max(size_t{1}, std::min(nz, size_t{8}))};
    // Keep enough independent tiles for the requested team, including thin
    // grids. Prefer wide X rows for efficient SIMD and contiguous memory access.
    const size_t threads = static_cast<size_t>(omp_get_max_threads());
    while (tileCount(nx, shape.x) * tileCount(ny, shape.y) * tileCount(nz, shape.z) < threads) {
        if (shape.y >= shape.z && shape.y > 1) {
            shape.y = (shape.y + 1) / 2;
        } else if (shape.z > 1) {
            shape.z = (shape.z + 1) / 2;
        } else if (shape.x > 1) {
            shape.x = (shape.x + 1) / 2;
        } else {
            break;
        }
    }
    return shape;
}

// Apply a stencil to part of a row. Only physical X boundaries need scalar
// handling; both tile interiors and their halos use contiguous SIMD loads.
template <typename CellOperation>
void stencilRow(const double* row, const double* yn, const double* yp,
                const double* zn, const double* zp, double* __restrict__ result,
                const size_t xBegin, const size_t xEnd, const size_t nx,
                const size_t fieldXBegin,
                const double dx2, const double dy2, const double dz2,
                CellOperation operation) {
    const auto apply = [&](const size_t x, const size_t xn, const size_t xp) {
        const size_t localX = x - fieldXBegin;
        return operation(x, row[localX],
                         computeLaplacian(row, yn, yp, zn, zp, localX,
                                          xn - fieldXBegin, xp - fieldXBegin, dx2, dy2, dz2));
    };

    const size_t interiorBegin = std::max(xBegin, size_t{1});
    const size_t interiorEnd = std::min(xEnd, nx - 1);
    if (xBegin == 0) {
        result[0] = apply(0, 0, nx > 1 ? 1 : 0);
    }
    #pragma omp simd
    for (size_t x = interiorBegin; x < interiorEnd; ++x) {
        result[x - xBegin] = apply(x, x - 1, x + 1);
    }
    if (xEnd == nx && nx > 1) {
        result[nx - 1 - xBegin] = apply(nx - 1, nx - 2, nx - 1);
    }
}

// Compute a tile's chemical potential, including a one-cell halo, directly from
// the immutable old concentration. Then update just the tile's owned cells.
// Potentials are thread-private, so tiles need no intermediate global barrier.
void advanceTile(const double* __restrict__ cold, double* __restrict__ cnew,
                 double* __restrict__ mu,
                 const size_t nx, const size_t ny, const size_t nz,
                 const size_t xBegin, const size_t xEnd,
                 const size_t yBegin, const size_t yEnd,
                 const size_t zBegin, const size_t zEnd,
                 const double dx, const double dy, const double dz,
                 const double gamma, const double e_AA, const double e_BB, const double e_AB,
                 const double D, const double dt) {
    const size_t hxBegin = xBegin > 0 ? xBegin - 1 : 0;
    const size_t hxEnd = xEnd < nx ? xEnd + 1 : nx;
    const size_t hyBegin = yBegin > 0 ? yBegin - 1 : 0;
    const size_t hyEnd = yEnd < ny ? yEnd + 1 : ny;
    const size_t hzBegin = zBegin > 0 ? zBegin - 1 : 0;
    const size_t hzEnd = zEnd < nz ? zEnd + 1 : nz;
    const size_t muStride = hxEnd - hxBegin;
    const size_t muPlane = muStride * (hyEnd - hyBegin);
    const size_t plane = nx * ny;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;

    for (size_t z = hzBegin; z < hzEnd; ++z) {
        for (size_t y = hyBegin; y < hyEnd; ++y) {
            const double* row = cold + z * plane + y * nx;
            double* potentialRow = mu + (z - hzBegin) * muPlane + (y - hyBegin) * muStride;
            stencilRow(row, row - (y > 0 ? nx : 0), row + (y + 1 < ny ? nx : 0),
                       row - (z > 0 ? plane : 0), row + (z + 1 < nz ? plane : 0),
                       potentialRow, hxBegin, hxEnd, nx, 0, dx2, dy2, dz2,
                       [=](size_t, const double cv, const double laplacian) {
                           return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                  + 3.0 * cv + cv * cv * cv - gamma * laplacian;
                       });
        }
    }

    for (size_t z = zBegin; z < zEnd; ++z) {
        for (size_t y = yBegin; y < yEnd; ++y) {
            const double* oldRow = cold + z * plane + y * nx;
            const double* row = mu + (z - hzBegin) * muPlane + (y - hyBegin) * muStride;
            double* nextRow = cnew + z * plane + y * nx + xBegin;
            stencilRow(row, row - (y > 0 ? muStride : 0), row + (y + 1 < ny ? muStride : 0),
                       row - (z > 0 ? muPlane : 0), row + (z + 1 < nz ? muPlane : 0),
                       nextRow, xBegin, xEnd, nx, hxBegin, dx2, dy2, dz2,
                       [=](const size_t x, double, const double laplacian) {
                           return oldRow[x] + dt * D * laplacian;
                       });
        }
    }
}

// Called by every team member; initialization uses the same static tile
// assignment as the simulation to preserve NUMA locality for both buffers.
void initializeConcentration(double* c, double* cnew,
                             const size_t nx, const size_t ny, const size_t nz,
                             const TileShape shape) {
    const size_t vol = nx * ny * nz;
    const size_t tilesX = tileCount(nx, shape.x);
    const size_t tilesY = tileCount(ny, shape.y);
    const size_t tilesZ = tileCount(nz, shape.z);
    #pragma omp for collapse(3) schedule(static)
    for (size_t bz = 0; bz < tilesZ; ++bz) {
        for (size_t by = 0; by < tilesY; ++by) {
            for (size_t bx = 0; bx < tilesX; ++bx) {
                for (size_t z = bz * shape.z; z < std::min((bz + 1) * shape.z, nz); ++z) {
                    for (size_t y = by * shape.y; y < std::min((by + 1) * shape.y, ny); ++y) {
                        for (size_t x = bx * shape.x; x < std::min((bx + 1) * shape.x, nx); ++x) {
                            const size_t idx = z * (nx * ny) + y * nx + x;
                            const double pseudo = ((((idx + 1) * 1299709) % vol) / static_cast<double>(vol));
                            c[idx] = -1.0 + 2.0 * pseudo;
                            cnew[idx] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

bool validateResult(const double* c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    double minVal = c[0];
    double maxVal = c[0];
    int invalid = 0;
    #pragma omp parallel for simd schedule(static) reduction(min:minVal) reduction(max:maxVal) reduction(|:invalid)
    for (size_t idx = 0; idx < vol; ++idx) {
        const double val = c[idx];
        invalid |= std::isnan(val) || std::isinf(val);
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    if (invalid) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
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
    
    // Leave storage uninitialized so each thread first-touches its own tiles.
    auto cold = std::make_unique_for_overwrite<double[]>(gridSize);
    auto cnew = std::make_unique_for_overwrite<double[]>(gridSize);
    const TileShape shape = chooseTileShape(nx, ny, nz);
    const size_t tilesX = tileCount(nx, shape.x);
    const size_t tilesY = tileCount(ny, shape.y);
    const size_t tilesZ = tileCount(nz, shape.z);
    const size_t potentialSize = std::min(nx, shape.x + 2) * std::min(ny, shape.y + 2) * std::min(nz, shape.z + 2);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    std::chrono::high_resolution_clock::time_point start;
    #pragma omp parallel
    {
        auto potential = std::make_unique_for_overwrite<double[]>(potentialSize);
        std::fill_n(potential.get(), potentialSize, 0.0);
        double* current = cold.get();
        double* next = cnew.get();
        initializeConcentration(current, next, nx, ny, nz, shape);
        #pragma omp single
        {
            printf("Running Cahn-Hilliard simulation...\n");
            start = std::chrono::high_resolution_clock::now();
        }

        for (int t = 0; t < iterations; ++t) {
            #pragma omp for collapse(3) schedule(static)
            for (size_t bz = 0; bz < tilesZ; ++bz) {
                for (size_t by = 0; by < tilesY; ++by) {
                    for (size_t bx = 0; bx < tilesX; ++bx) {
                        advanceTile(current, next, potential.get(), nx, ny, nz,
                                    bx * shape.x, std::min((bx + 1) * shape.x, nx),
                                    by * shape.y, std::min((by + 1) * shape.y, ny),
                                    bz * shape.z, std::min((bz + 1) * shape.z, nz),
                                    dx, dy, dz, gamma, e_AA, e_BB, e_AB, D, dt);
                    }
                }
            }

            // Each thread swaps its own pointers after the update barrier. This
            // needs no single region or additional synchronization.
            std::swap(current, next);
        }
    }
    if (iterations > 0 && iterations % 2 != 0) {
        std::swap(cold, cnew);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / std::chrono::duration<double>(end - start).count() / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    if (printResults) {
        print_results(std::vector<double>(cold.get(), cold.get() + gridSize), "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold.get(), nx, ny, nz);
        
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
