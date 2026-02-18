#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// 3D index calculation (host)
inline constexpr size_t idx3_host(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

struct ZDecomp {
    size_t start;
    size_t count;
};

static inline ZDecomp computeZDecomp(const size_t nz, const int rank, const int size) {
    const size_t size_sz = static_cast<size_t>(size);
    const size_t rank_sz = static_cast<size_t>(rank);
    const size_t base = nz / size_sz;
    const size_t rem = nz % size_sz;
    const size_t count = base + (rank_sz < rem ? 1 : 0);
    const size_t start = (rank_sz < rem)
        ? rank_sz * (base + 1)
        : rem * (base + 1) + (rank_sz - rem) * base;
    return {start, count};
}

static inline void checkCuda(const cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", context, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// 3D index calculation (device)
__device__ __forceinline__ size_t idx3_device(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                       const size_t nx, const size_t ny, const size_t local_nz,
                                       const double inv_dx2, const double inv_dy2, const double inv_dz2,
                                       const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (x >= nx || y >= ny || z > local_nz) {
        return;
    }

    const size_t idx = idx3_device(x, y, z, nx, ny);
    const double cv = c[idx];

    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const double cxx = (c[idx3_device(xp, y, z, nx, ny)] + c[idx3_device(xn, y, z, nx, ny)] - 2.0 * cv) * inv_dx2;
    const double cyy = (c[idx3_device(x, yp, z, nx, ny)] + c[idx3_device(x, yn, z, nx, ny)] - 2.0 * cv) * inv_dy2;
    const double czz = (c[idx3_device(x, y, z + 1, nx, ny)] + c[idx3_device(x, y, z - 1, nx, ny)] - 2.0 * cv) * inv_dz2;

    const double laplacian = cxx + cyy + czz;

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
            + 3.0 * cv + cv * cv * cv
            - gamma * laplacian;
}

__global__ void updateKernel(double* __restrict__ cnew, const double* __restrict__ cold, const double* __restrict__ mu,
                             const size_t nx, const size_t ny, const size_t local_nz,
                             const double inv_dx2, const double inv_dy2, const double inv_dz2,
                             const double D, const double dt) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (x >= nx || y >= ny || z > local_nz) {
        return;
    }

    const size_t idx = idx3_device(x, y, z, nx, ny);
    const double cv = mu[idx];

    const size_t xp = (x + 1 < nx) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y + 1 < ny) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const double cxx = (mu[idx3_device(xp, y, z, nx, ny)] + mu[idx3_device(xn, y, z, nx, ny)] - 2.0 * cv) * inv_dx2;
    const double cyy = (mu[idx3_device(x, yp, z, nx, ny)] + mu[idx3_device(x, yn, z, nx, ny)] - 2.0 * cv) * inv_dy2;
    const double czz = (mu[idx3_device(x, y, z + 1, nx, ny)] + mu[idx3_device(x, y, z - 1, nx, ny)] - 2.0 * cv) * inv_dz2;

    cnew[idx] = cold[idx] + dt * D * (cxx + cyy + czz);
}

void exchangeHalos(double* d_field,
                   std::vector<double>& send_lower,
                   std::vector<double>& send_upper,
                   std::vector<double>& recv_lower,
                   std::vector<double>& recv_upper,
                   const size_t planeSize,
                   const size_t local_nz,
                   const int rank,
                   const int size,
                   MPI_Comm comm) {
    const size_t planeBytes = planeSize * sizeof(double);
    double* d_lower = d_field + planeSize;
    double* d_upper = d_field + planeSize * local_nz;
    double* d_ghost_lower = d_field;
    double* d_ghost_upper = d_field + planeSize * (local_nz + 1);

    if (rank > 0) {
        checkCuda(cudaMemcpy(send_lower.data(), d_lower, planeBytes, cudaMemcpyDeviceToHost), "halo lower D2H");
    }
    if (rank < size - 1) {
        checkCuda(cudaMemcpy(send_upper.data(), d_upper, planeBytes, cudaMemcpyDeviceToHost), "halo upper D2H");
    }

    const int tag_down = 100;
    const int tag_up = 101;

    if (rank > 0) {
        MPI_Sendrecv(send_lower.data(), static_cast<int>(planeSize), MPI_DOUBLE, rank - 1, tag_down,
                     recv_lower.data(), static_cast<int>(planeSize), MPI_DOUBLE, rank - 1, tag_up,
                     comm, MPI_STATUS_IGNORE);
    } else {
        checkCuda(cudaMemcpy(d_ghost_lower, d_lower, planeBytes, cudaMemcpyDeviceToDevice), "mirror lower halo");
    }

    if (rank < size - 1) {
        MPI_Sendrecv(send_upper.data(), static_cast<int>(planeSize), MPI_DOUBLE, rank + 1, tag_up,
                     recv_upper.data(), static_cast<int>(planeSize), MPI_DOUBLE, rank + 1, tag_down,
                     comm, MPI_STATUS_IGNORE);
    } else {
        checkCuda(cudaMemcpy(d_ghost_upper, d_upper, planeBytes, cudaMemcpyDeviceToDevice), "mirror upper halo");
    }

    if (rank > 0) {
        checkCuda(cudaMemcpy(d_ghost_lower, recv_lower.data(), planeBytes, cudaMemcpyHostToDevice), "halo lower H2D");
    }
    if (rank < size - 1) {
        checkCuda(cudaMemcpy(d_ghost_upper, recv_upper.data(), planeBytes, cudaMemcpyHostToDevice), "halo upper H2D");
    }
}

void initializeConcentration(std::vector<double>& c,
                             const size_t nx, const size_t ny, const size_t nz,
                             const size_t z_start, const size_t local_nz) {
    const size_t planeSize = nx * ny;
    const size_t vol = nx * ny * nz;

    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t global_z = z_start + z;
                const size_t idx = idx3_host(x, y, z + 1, nx, ny);
                const size_t linear_id = global_z * planeSize + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    bool hasBad = false;
    double minVal = DBL_MAX;
    double maxVal = -DBL_MAX;

    #pragma omp parallel for reduction(||:hasBad) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < c.size(); ++i) {
        const double val = c[i];
        if (std::isnan(val) || std::isinf(val)) {
            hasBad = true;
        }
        if (val < minVal) {
            minVal = val;
        }
        if (val > maxVal) {
            maxVal = val;
        }
    }

    if (hasBad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseOk = true;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseOk = false;
        }
    }

    int parseOkInt = parseOk ? 1 : 0;
    int parseOkGlobal = 0;
    MPI_Allreduce(&parseOkInt, &parseOkGlobal, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!parseOkGlobal) {
        MPI_Finalize();
        return 1;
    }

    int showHelpInt = showHelp ? 1 : 0;
    int showHelpGlobal = 0;
    MPI_Allreduce(&showHelpInt, &showHelpGlobal, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (showHelpGlobal) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const ZDecomp decomp = computeZDecomp(nz, rank, size);
    const size_t local_nz = decomp.count;
    const size_t z_start = decomp.start;

    if (local_nz == 0) {
        if (rank == 0) {
            printf("Error: too many MPI ranks for nz=%zu\n", nz);
        }
        MPI_Finalize();
        return 1;
    }

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices found\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int deviceId = rank % deviceCount;
    checkCuda(cudaSetDevice(deviceId), "cudaSetDevice");

    const size_t planeSize = nx * ny;
    const size_t localWithGhost = (local_nz + 2) * planeSize;

    std::vector<double> host_cold(localWithGhost, 0.0);
    if (rank == 0) {
        printf("Initializing concentration field...\n");
    }
    initializeConcentration(host_cold, nx, ny, nz, z_start, local_nz);

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    const size_t bytes = localWithGhost * sizeof(double);

    checkCuda(cudaMalloc(&d_cold, bytes), "cudaMalloc cold");
    checkCuda(cudaMalloc(&d_cnew, bytes), "cudaMalloc cnew");
    checkCuda(cudaMalloc(&d_mu, bytes), "cudaMalloc mu");

    checkCuda(cudaMemcpy(d_cold, host_cold.data(), bytes, cudaMemcpyHostToDevice), "copy cold to device");

    std::vector<double> send_lower(planeSize);
    std::vector<double> send_upper(planeSize);
    std::vector<double> recv_lower(planeSize);
    std::vector<double> recv_upper(planeSize);

    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);

    const dim3 block(8, 8, 4);
    const dim3 grid((nx + block.x - 1) / block.x,
                    (ny + block.y - 1) / block.y,
                    (local_nz + block.z - 1) / block.z);

    if (rank == 0) {
        printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        exchangeHalos(d_cold, send_lower, send_upper, recv_lower, recv_upper,
                      planeSize, local_nz, rank, size, MPI_COMM_WORLD);

        chemicalPotentialKernel<<<grid, block>>>(d_cold, d_mu, nx, ny, local_nz,
                                                  inv_dx2, inv_dy2, inv_dz2,
                                                  gamma, e_AA, e_BB, e_AB);
        checkCuda(cudaGetLastError(), "chemicalPotentialKernel");

        exchangeHalos(d_mu, send_lower, send_upper, recv_lower, recv_upper,
                      planeSize, local_nz, rank, size, MPI_COMM_WORLD);

        updateKernel<<<grid, block>>>(d_cnew, d_cold, d_mu, nx, ny, local_nz,
                                      inv_dx2, inv_dy2, inv_dz2, D, dt);
        checkCuda(cudaGetLastError(), "updateKernel");

        std::swap(d_cold, d_cnew);
    }

    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    const double end = MPI_Wtime();
    const double elapsed = end - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
        const double cellUpdates = static_cast<double>(nx) * ny * nz * iterations;
        const double mcups = cellUpdates / maxElapsed / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<double> host_local;
    if (printResults || validate) {
        host_local.resize(local_nz * planeSize);
        checkCuda(cudaMemcpy(host_local.data(), d_cold + planeSize,
                             host_local.size() * sizeof(double), cudaMemcpyDeviceToHost),
                  "copy results to host");
    }

    std::vector<double> global_c;
    if (printResults || validate) {
        std::vector<int> counts(size);
        std::vector<int> displs(size);
        const size_t base = nz / static_cast<size_t>(size);
        const size_t rem = nz % static_cast<size_t>(size);
        size_t offset = 0;

        for (int r = 0; r < size; ++r) {
            const size_t r_nz = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            counts[r] = static_cast<int>(r_nz * planeSize);
            displs[r] = static_cast<int>(offset * planeSize);
            offset += r_nz;
        }

        if (rank == 0) {
            global_c.resize(nx * ny * nz);
        }

        MPI_Gatherv(host_local.data(), static_cast<int>(host_local.size()), MPI_DOUBLE,
                    rank == 0 ? global_c.data() : nullptr,
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (printResults && rank == 0) {
        print_results(global_c, "Concentration");
    }

    int validInt = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            const bool valid = validateResult(global_c, nx, ny, nz);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
            validInt = valid ? 1 : 0;
        }
        MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    checkCuda(cudaFree(d_cold), "cudaFree cold");
    checkCuda(cudaFree(d_cnew), "cudaFree cnew");
    checkCuda(cudaFree(d_mu), "cudaFree mu");

    MPI_Finalize();

    if (validate) {
        return validInt ? 0 : 1;
    }
    return 0;
}
