#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

// Run all stencil sweeps inside one persistent OpenMP team. Rows are the unit
// of work so each thread receives contiguous memory, while the X dimension is
// left as the SIMD-friendly innermost loop.
void runSimulation(std::vector<double> &cold, std::vector<double> &cnew,
                   std::vector<double> &mu, const size_t nx, const size_t ny,
                   const size_t nz, const int iterations, const double D,
                   const double dt, const double dx, const double dy,
                   const double dz, const double gamma, const double e_AA,
                   const double e_BB, const double e_AB) {
  const size_t rows = ny * nz;
  const size_t planeSize = nx * ny;
  const double invDx2 = 1.0 / (dx * dx);
  const double invDy2 = 1.0 / (dy * dy);
  const double invDz2 = 1.0 / (dz * dz);
  const double dtD = dt * D;
  double *const initial = cold.data();
  double *const scratch = cnew.data();
  double *const muData = mu.data();

#pragma omp parallel
  {
    // These pointers are private to each thread. Swapping them locally
    // avoids a single-thread section and an additional barrier per step.
    double *current = initial;
    double *next = scratch;

    // A manual contiguous partition has the same load balance as a static
    // workshare and lets each thread derive Y/Z only once per sweep.
    const size_t threadId = static_cast<size_t>(omp_get_thread_num());
    const size_t threadCount = static_cast<size_t>(omp_get_num_threads());
    const size_t rowsPerThread = rows / threadCount;
    const size_t extraRows = rows % threadCount;
    const size_t rowBegin =
        threadId * rowsPerThread + std::min(threadId, extraRows);
    const size_t rowEnd =
        rowBegin + rowsPerThread + (threadId < extraRows ? 1 : 0);

    for (int t = 0; t < iterations; ++t) {
      if (nx != 0 && rowBegin < rowEnd) {
        size_t z = rowBegin / ny;
        size_t y = rowBegin - z * ny;

        for (size_t rowId = rowBegin; rowId < rowEnd; ++rowId) {
          const size_t base = rowId * nx;

          const double *const cRow = current + base;
          const double *const cYn = current + ((y > 0) ? base - nx : base);
          const double *const cYp = current + ((y + 1 < ny) ? base + nx : base);
          const double *const cZn =
              current + ((z > 0) ? base - planeSize : base);
          const double *const cZp =
              current + ((z + 1 < nz) ? base + planeSize : base);
          double *const muRow = muData + base;

          const auto computeBoundary = [&](const size_t x, const size_t xn,
                                           const size_t xp) noexcept {
            const double cv = cRow[x];
            const double cxx = (cRow[xp] + cRow[xn] - 2.0 * cv) * invDx2;
            const double cyy = (cYp[x] + cYn[x] - 2.0 * cv) * invDy2;
            const double czz = (cZp[x] + cZn[x] - 2.0 * cv) * invDz2;
            const double laplacian = cxx + cyy + czz;

            muRow[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB -
                              2.0 * cv * e_AB) +
                       3.0 * cv + cv * cv * cv - gamma * laplacian;
          };

          computeBoundary(0, 0, (nx > 1) ? 1 : 0);

#pragma omp simd
          for (size_t x = 1; x < nx - 1; ++x) {
            const double cv = cRow[x];
            const double cxx = (cRow[x + 1] + cRow[x - 1] - 2.0 * cv) * invDx2;
            const double cyy = (cYp[x] + cYn[x] - 2.0 * cv) * invDy2;
            const double czz = (cZp[x] + cZn[x] - 2.0 * cv) * invDz2;
            const double laplacian = cxx + cyy + czz;

            muRow[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB -
                              2.0 * cv * e_AB) +
                       3.0 * cv + cv * cv * cv - gamma * laplacian;
          }

          if (nx > 1) {
            computeBoundary(nx - 1, nx - 2, nx - 1);
          }

          if (++y == ny) {
            y = 0;
            ++z;
          }
        }
      }
#pragma omp barrier

      if (nx != 0 && rowBegin < rowEnd) {
        size_t z = rowBegin / ny;
        size_t y = rowBegin - z * ny;

        for (size_t rowId = rowBegin; rowId < rowEnd; ++rowId) {
          const size_t base = rowId * nx;

          const double *const muRow = muData + base;
          const double *const muYn = muData + ((y > 0) ? base - nx : base);
          const double *const muYp = muData + ((y + 1 < ny) ? base + nx : base);
          const double *const muZn =
              muData + ((z > 0) ? base - planeSize : base);
          const double *const muZp =
              muData + ((z + 1 < nz) ? base + planeSize : base);
          const double *const cRow = current + base;
          double *const nextRow = next + base;

          const auto updateBoundary = [&](const size_t x, const size_t xn,
                                          const size_t xp) noexcept {
            const double muv = muRow[x];
            const double muxx = (muRow[xp] + muRow[xn] - 2.0 * muv) * invDx2;
            const double muyy = (muYp[x] + muYn[x] - 2.0 * muv) * invDy2;
            const double muzz = (muZp[x] + muZn[x] - 2.0 * muv) * invDz2;

            nextRow[x] = cRow[x] + dtD * (muxx + muyy + muzz);
          };

          updateBoundary(0, 0, (nx > 1) ? 1 : 0);

#pragma omp simd
          for (size_t x = 1; x < nx - 1; ++x) {
            const double muv = muRow[x];
            const double muxx =
                (muRow[x + 1] + muRow[x - 1] - 2.0 * muv) * invDx2;
            const double muyy = (muYp[x] + muYn[x] - 2.0 * muv) * invDy2;
            const double muzz = (muZp[x] + muZn[x] - 2.0 * muv) * invDz2;

            nextRow[x] = cRow[x] + dtD * (muxx + muyy + muzz);
          }

          if (nx > 1) {
            updateBoundary(nx - 1, nx - 2, nx - 1);
          }

          if (++y == ny) {
            y = 0;
            ++z;
          }
        }
      }
#pragma omp barrier

      std::swap(current, next);
    }
  }

  // Match the original public buffer ownership after an odd step count.
  if (iterations > 0 && iterations % 2 != 0) {
    cold.swap(cnew);
  }
}

// Initialize concentration field
void initializeConcentration(std::vector<double> &c, const size_t nx,
                             const size_t ny, const size_t nz) {
  const size_t vol = nx * ny * nz;

#pragma omp parallel for schedule(static)
  for (size_t linearId = 0; linearId < vol; ++linearId) {
    // Generate pseudo-random value in [-1, 1]
    const double pseudo =
        ((((linearId + 1) * 1299709) % vol) / static_cast<double>(vol));
    c[linearId] = -1.0 + 2.0 * pseudo;
  }
}

bool validateResult(const std::vector<double> &c,
                    [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
  // Check for NaN or Inf
  for (const auto &val : c) {
    if (std::isnan(val) || std::isinf(val)) {
      printf("Validation failed: found NaN or Inf value\n");
      return false;
    }
  }

  // Check if values are in reasonable range for concentration field
  // After Cahn-Hilliard evolution, values should typically remain bounded
  double minVal = c[0];
  double maxVal = c[0];
  for (const auto &val : c) {
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

void printUsage(const char *progName) {
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

int main(int argc, char **argv) {
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

  if (ny == 0)
    ny = nx;
  if (nz == 0)
    nz = nx;

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

  runSimulation(cold, cnew, mu, nx, ny, nz, iterations, D, dt, dx, dy, dz,
                gamma, e_AA, e_BB, e_AB);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

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
