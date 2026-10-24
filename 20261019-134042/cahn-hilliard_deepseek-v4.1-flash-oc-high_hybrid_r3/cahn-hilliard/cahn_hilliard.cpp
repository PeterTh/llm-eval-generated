#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),        \
                    __FILE__, __LINE__);                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ c, double* __restrict__ mu,
    const int nx, const int ny, const int nzLocal, const int z0, const int nz,
    const double dx, const double dy, const double dz,
    const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int lzOwned = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || lzOwned >= nzLocal) return;

    const int lz = lzOwned + 1;
    const int gz = z0 + lzOwned;
    const size_t layer = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t idx = static_cast<size_t>(lz) * layer +
                       static_cast<size_t>(y) * static_cast<size_t>(nx) +
                       static_cast<size_t>(x);

    const int xp = (x < nx - 1) ? x + 1 : x;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int yn = (y > 0) ? y - 1 : 0;
    const int zp = (gz < nz - 1) ? lz + 1 : lz;
    const int zn = (gz > 0) ? lz - 1 : lz;

    const size_t bx = static_cast<size_t>(x);
    const size_t bxp = static_cast<size_t>(xp);
    const size_t bxn = static_cast<size_t>(xn);
    const size_t byx = static_cast<size_t>(y) * static_cast<size_t>(nx);
    const size_t bypx = static_cast<size_t>(yp) * static_cast<size_t>(nx);
    const size_t bynx = static_cast<size_t>(yn) * static_cast<size_t>(nx);
    const size_t base = static_cast<size_t>(lz) * layer;

    const double cv = c[idx];
    const double cxx = (c[base + byx + bxp] + c[base + byx + bxn] - 2.0 * cv) / (dx * dx);
    const double cyy = (c[base + bypx + bx] + c[base + bynx + bx] - 2.0 * cv) / (dy * dy);
    const double czz = (c[static_cast<size_t>(zp) * layer + byx + bx] +
                        c[static_cast<size_t>(zn) * layer + byx + bx] - 2.0 * cv) /
                       (dz * dz);

    mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) +
              3.0 * cv + cv * cv * cv - gamma * (cxx + cyy + czz);
}

__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ cnew, const double* __restrict__ cold,
    const double* __restrict__ mu,
    const int nx, const int ny, const int nzLocal, const int z0, const int nz,
    const double D, const double dt,
    const double dx, const double dy, const double dz) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int lzOwned = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || lzOwned >= nzLocal) return;

    const int lz = lzOwned + 1;
    const int gz = z0 + lzOwned;
    const size_t layer = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t idx = static_cast<size_t>(lz) * layer +
                       static_cast<size_t>(y) * static_cast<size_t>(nx) +
                       static_cast<size_t>(x);

    const int xp = (x < nx - 1) ? x + 1 : x;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int yn = (y > 0) ? y - 1 : 0;
    const int zp = (gz < nz - 1) ? lz + 1 : lz;
    const int zn = (gz > 0) ? lz - 1 : lz;

    const size_t bx = static_cast<size_t>(x);
    const size_t bxp = static_cast<size_t>(xp);
    const size_t bxn = static_cast<size_t>(xn);
    const size_t byx = static_cast<size_t>(y) * static_cast<size_t>(nx);
    const size_t bypx = static_cast<size_t>(yp) * static_cast<size_t>(nx);
    const size_t bynx = static_cast<size_t>(yn) * static_cast<size_t>(nx);
    const size_t base = static_cast<size_t>(lz) * layer;

    const double muv = mu[idx];
    const double muxx = (mu[base + byx + bxp] + mu[base + byx + bxn] - 2.0 * muv) / (dx * dx);
    const double muyy = (mu[base + bypx + bx] + mu[base + bynx + bx] - 2.0 * muv) / (dy * dy);
    const double muzz = (mu[static_cast<size_t>(zp) * layer + byx + bx] +
                         mu[static_cast<size_t>(zn) * layer + byx + bx] - 2.0 * muv) /
                        (dz * dz);

    cnew[idx] = cold[idx] + dt * D * (muxx + muyy + muzz);
}

void initializeConcentration(std::vector<double>& c, const size_t z0, const size_t nzLocal,
                             const size_t nx, const size_t ny, const size_t nz) {
    const size_t layer = nx * ny;
    const size_t vol = layer * nz;

#pragma omp parallel for collapse(2) schedule(static)
    for (long long lz = 0; lz < static_cast<long long>(nzLocal); ++lz) {
        for (long long y = 0; y < static_cast<long long>(ny); ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = static_cast<size_t>(lz + 1) * layer +
                                   static_cast<size_t>(y) * nx + x;
                const size_t linear_id = (z0 + static_cast<size_t>(lz)) * layer +
                                         static_cast<size_t>(y) * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) /
                                       static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

void exchangeHalo(double* dev, const size_t nzLocal, const size_t layer,
                  const bool hasPrev, const bool hasNext, const int prev, const int next,
                  const int mpiCount, MPI_Comm comm, const int tag,
                  double* sendLo, double* sendHi, double* recvLo, double* recvHi,
                  cudaStream_t stream) {
    if (nzLocal == 0) return;

    const size_t bytes = layer * sizeof(double);

    if (hasPrev) {
        CUDA_CHECK(cudaMemcpyAsync(sendLo, dev + layer, bytes, cudaMemcpyDeviceToHost, stream));
    }
    if (hasNext) {
        CUDA_CHECK(cudaMemcpyAsync(sendHi, dev + nzLocal * layer, bytes,
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (hasNext) {
        MPI_Sendrecv(sendHi, mpiCount, MPI_DOUBLE, next, tag,
                     recvHi, mpiCount, MPI_DOUBLE, next, tag + 1,
                     comm, MPI_STATUS_IGNORE);
    }
    if (hasPrev) {
        MPI_Sendrecv(sendLo, mpiCount, MPI_DOUBLE, prev, tag + 1,
                     recvLo, mpiCount, MPI_DOUBLE, prev, tag,
                     comm, MPI_STATUS_IGNORE);
    }

    if (hasPrev) {
        CUDA_CHECK(cudaMemcpyAsync(dev, recvLo, bytes, cudaMemcpyHostToDevice, stream));
    }
    if (hasNext) {
        CUDA_CHECK(cudaMemcpyAsync(dev + (nzLocal + 1) * layer, recvHi, bytes,
                                   cudaMemcpyHostToDevice, stream));
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const long long n = static_cast<long long>(c.size());
    int bad = 0;
    double minVal = c[0];
    double maxVal = c[0];

#pragma omp parallel for schedule(static) reduction(|| : bad) reduction(min : minVal) reduction(max : maxVal)
    for (long long i = 0; i < n; ++i) {
        const double val = c[static_cast<size_t>(i)];
        if (std::isnan(val) || std::isinf(val)) {
            bad = 1;
        }
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    if (bad) {
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

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

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
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    MPI_Comm_rank(nodeComm, &nodeRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (worldRank == 0) printf("CUDA device required but none was found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int deviceId = nodeRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(deviceId));
    CUDA_CHECK(cudaFree(nullptr));
    MPI_Comm_free(&nodeComm);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const size_t layer = nx * ny;
    const size_t baseCount = nz / static_cast<size_t>(worldSize);
    const size_t remainder = nz % static_cast<size_t>(worldSize);
    const size_t localNz = baseCount +
                           (static_cast<size_t>(worldRank) < remainder ? 1 : 0);
    const size_t z0 = static_cast<size_t>(worldRank) * baseCount +
                      std::min(static_cast<size_t>(worldRank), remainder);
    const int activeRanks = static_cast<int>(std::min(nz, static_cast<size_t>(worldSize)));
    const bool hasPrev = worldRank > 0 && worldRank < activeRanks;
    const bool hasNext = worldRank + 1 < activeRanks;
    const int prev = worldRank - 1;
    const int next = worldRank + 1;
    const int mpiCount = static_cast<int>(layer);
    const size_t localSize = (localNz + 2) * layer;
    const size_t ownedSize = localNz * layer;

    double* coldDev = nullptr;
    double* cnewDev = nullptr;
    double* muDev = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&coldDev), localSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&cnewDev), localSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&muDev), localSize * sizeof(double)));

    double* sendLo = nullptr;
    double* sendHi = nullptr;
    double* recvLo = nullptr;
    double* recvHi = nullptr;
    const size_t bytes = layer * sizeof(double);
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&sendLo), bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&sendHi), bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&recvLo), bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&recvHi), bytes, cudaHostAllocDefault));

    std::vector<double> hostCold(localSize, 0.0);
    if (worldRank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing concentration field...\n");
    }

    initializeConcentration(hostCold, z0, localNz, nx, ny, nz);
    CUDA_CHECK(cudaMemcpyAsync(coldDev, hostCold.data(), localSize * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    const dim3 block(32, 8, 1);
    const dim3 grid((static_cast<unsigned>(nx) + block.x - 1) / block.x,
                    (static_cast<unsigned>(ny) + block.y - 1) / block.y,
                    static_cast<unsigned>(localNz));

    if (worldRank == 0) printf("Running Cahn-Hilliard simulation...\n");

    if (localNz > 0) {
        computeChemicalPotentialKernel<<<grid, block, 0, stream>>>(
            coldDev, muDev, static_cast<int>(nx), static_cast<int>(ny),
            static_cast<int>(localNz), static_cast<int>(z0), static_cast<int>(nz),
            dx, dy, dz, gamma, e_AA, e_BB, e_AB);
        cahnHilliardUpdateKernel<<<grid, block, 0, stream>>>(
            cnewDev, coldDev, muDev, static_cast<int>(nx), static_cast<int>(ny),
            static_cast<int>(localNz), static_cast<int>(z0), static_cast<int>(nz),
            D, dt, dx, dy, dz);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        exchangeHalo(coldDev, localNz, layer, hasPrev, hasNext, prev, next,
                     mpiCount, MPI_COMM_WORLD, 100, sendLo, sendHi, recvLo, recvHi, stream);

        if (localNz > 0) {
            computeChemicalPotentialKernel<<<grid, block, 0, stream>>>(
                coldDev, muDev, static_cast<int>(nx), static_cast<int>(ny),
                static_cast<int>(localNz), static_cast<int>(z0), static_cast<int>(nz),
                dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            CUDA_CHECK(cudaGetLastError());
        }

        exchangeHalo(muDev, localNz, layer, hasPrev, hasNext, prev, next,
                     mpiCount, MPI_COMM_WORLD, 200, sendLo, sendHi, recvLo, recvHi, stream);

        if (localNz > 0) {
            cahnHilliardUpdateKernel<<<grid, block, 0, stream>>>(
                cnewDev, coldDev, muDev, static_cast<int>(nx), static_cast<int>(ny),
                static_cast<int>(localNz), static_cast<int>(z0), static_cast<int>(nz),
                D, dt, dx, dy, dz);
            CUDA_CHECK(cudaGetLastError());
        }

        std::swap(coldDev, cnewDev);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    const size_t gridSize = nx * ny * nz;

    if (worldRank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        const double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int result = 0;
    if (printResults || validate) {
        std::vector<double> ownedHost(ownedSize);
        if (ownedSize > 0) {
            CUDA_CHECK(cudaMemcpy(ownedHost.data(), coldDev + layer,
                                  ownedSize * sizeof(double), cudaMemcpyDeviceToHost));
        }

        std::vector<int> counts;
        std::vector<int> displs;
        if (worldRank == 0) {
            counts.resize(static_cast<size_t>(worldSize));
            displs.resize(static_cast<size_t>(worldSize));
        }
        for (int r = 0; r < worldSize; ++r) {
            const size_t lz = baseCount + (static_cast<size_t>(r) < remainder ? 1 : 0);
            if (worldRank == 0) {
                counts[static_cast<size_t>(r)] = static_cast<int>(lz * layer);
            }
        }
        if (worldRank == 0) {
            int offset = 0;
            for (int r = 0; r < worldSize; ++r) {
                displs[static_cast<size_t>(r)] = offset;
                offset += counts[static_cast<size_t>(r)];
            }
        }

        std::vector<double> global;
        if (worldRank == 0) global.resize(gridSize);

        MPI_Gatherv(ownedHost.data(), static_cast<int>(ownedSize), MPI_DOUBLE,
                    worldRank == 0 ? global.data() : nullptr,
                    worldRank == 0 ? counts.data() : nullptr,
                    worldRank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (worldRank == 0) {
            if (printResults) {
                print_results(global, "Concentration");
            }
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(global, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    result = 1;
                }
            }
        }
    }

    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(coldDev));
    CUDA_CHECK(cudaFree(cnewDev));
    CUDA_CHECK(cudaFree(muDev));
    CUDA_CHECK(cudaFreeHost(sendLo));
    CUDA_CHECK(cudaFreeHost(sendHi));
    CUDA_CHECK(cudaFreeHost(recvLo));
    CUDA_CHECK(cudaFreeHost(recvHi));

    MPI_Finalize();
    return result;
}
