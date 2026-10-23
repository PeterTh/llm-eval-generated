#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err__ = (call);                                                          \
        if (err__ != cudaSuccess) {                                                          \
            std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err__),      \
                         __FILE__, __LINE__);                                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                    \
        }                                                                                    \
    } while (0)

// Bodies are stored interleaved as [x, y, z] for both positions and velocities.
// Each thread owns one body and walks all other bodies in the original sequential
// order.  The fused operations below reproduce exactly the arithmetic emitted by
// the optimized scalar reference implementation (GCC contracts the distance
// expression and the force accumulation, but not the velocity update), so the
// simulation results are bit-identical to the original CPU code.
__global__ void computeForcesKernel(const double* __restrict__ allPos,
                                    double* __restrict__ localVel,
                                    const int n,
                                    const int start,
                                    const int count) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) {
        return;
    }

    const int i = start + t;
    const double xi = allPos[3 * i + 0];
    const double yi = allPos[3 * i + 1];
    const double zi = allPos[3 * i + 2];

    double Fx = 0.0;
    double Fy = 0.0;
    double Fz = 0.0;

#pragma unroll 4
    for (int j = 0; j < n; ++j) {
        const int b = 3 * j;
        const double dx = allPos[b + 0] - xi;
        const double dy = allPos[b + 1] - yi;
        const double dz = allPos[b + 2] - zi;

        const double dy2 = dy * dy;
        double distSqr = __fma_rn(dx, dx, dy2);
        distSqr = __fma_rn(dz, dz, distSqr);
        distSqr = distSqr + SOFTENING;

        const double invDist = 1.0 / sqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;

        Fx = __fma_rn(dx, invDist3, Fx);
        Fy = __fma_rn(dy, invDist3, Fy);
        Fz = __fma_rn(dz, invDist3, Fz);
    }

    const int o = 3 * t;
    localVel[o + 0] = __dadd_rn(localVel[o + 0], __dmul_rn(DT, Fx));
    localVel[o + 1] = __dadd_rn(localVel[o + 1], __dmul_rn(DT, Fy));
    localVel[o + 2] = __dadd_rn(localVel[o + 2], __dmul_rn(DT, Fz));
}

__global__ void integrateKernel(double* __restrict__ localPos,
                                const double* __restrict__ localVel,
                                const int count) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) {
        return;
    }

    const int o = 3 * t;
    localPos[o + 0] = __fma_rn(localVel[o + 0], DT, localPos[o + 0]);
    localPos[o + 1] = __fma_rn(localVel[o + 1], DT, localPos[o + 1]);
    localPos[o + 2] = __fma_rn(localVel[o + 2], DT, localPos[o + 2]);
}

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks energy conservation)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int commSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Assign one GPU per node-local MPI rank (round-robin when ranks outnumber GPUs).
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    MPI_Comm_rank(nodeComm, &nodeRank);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA device available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(nodeRank % deviceCount));

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const int n = numBodies;
    const int base = n / commSize;
    const int rem = n % commSize;

    std::vector<int> bodyCounts(commSize);
    std::vector<int> bodyDispls(commSize);
    std::vector<int> dataCounts(commSize);
    std::vector<int> dataDispls(commSize);

    int localN = base + (rank < rem ? 1 : 0);
    int localStart = rank * base + std::min(rank, rem);

    for (int r = 0; r < commSize; ++r) {
        bodyCounts[r] = base + (r < rem ? 1 : 0);
        bodyDispls[r] = r * base + std::min(r, rem);
        dataCounts[r] = 3 * bodyCounts[r];
        dataDispls[r] = 3 * bodyDispls[r];
    }

    // Rank 0 generates the complete initial state exactly as the reference code,
    // then scatters contiguous slices to the other ranks.
    std::vector<Body> allBodies;
    if (rank == 0) {
        allBodies.resize(n);
        randomizeBodies(allBodies);
    }

    std::vector<Body> localBodies(static_cast<size_t>(std::max(localN, 1)));
    std::vector<int> byteCounts(commSize);
    std::vector<int> byteDispls(commSize);
    for (int r = 0; r < commSize; ++r) {
        byteCounts[r] = bodyCounts[r] * static_cast<int>(sizeof(Body));
        byteDispls[r] = bodyDispls[r] * static_cast<int>(sizeof(Body));
    }

    MPI_Scatterv(rank == 0 ? static_cast<const void*>(allBodies.data()) : nullptr,
                 byteCounts.data(), byteDispls.data(), MPI_BYTE,
                 static_cast<void*>(localBodies.data()), localN * static_cast<int>(sizeof(Body)),
                 MPI_BYTE, 0, MPI_COMM_WORLD);
    allBodies.clear();
    allBodies.shrink_to_fit();

    const size_t localDoubles = static_cast<size_t>(std::max(3 * localN, 1));
    const size_t totalDoubles = static_cast<size_t>(std::max(3 * n, 1));

    double* hostLocalPos = nullptr;
    double* hostLocalVel = nullptr;
    double* hostAllPos = nullptr;
    double* hostAllVel = nullptr;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&hostLocalPos), localDoubles * sizeof(double),
                             cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&hostLocalVel), localDoubles * sizeof(double),
                             cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&hostAllPos), totalDoubles * sizeof(double),
                             cudaHostAllocDefault));
    if (printResults) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&hostAllVel), totalDoubles * sizeof(double),
                                 cudaHostAllocDefault));
    }

#pragma omp parallel for
    for (int t = 0; t < localN; ++t) {
        const Body& b = localBodies[t];
        const int o = 3 * t;
        hostLocalPos[o + 0] = b.pos.x;
        hostLocalPos[o + 1] = b.pos.y;
        hostLocalPos[o + 2] = b.pos.z;
        hostLocalVel[o + 0] = b.vel.x;
        hostLocalVel[o + 1] = b.vel.y;
        hostLocalVel[o + 2] = b.vel.z;
    }
    localBodies.clear();
    localBodies.shrink_to_fit();

    double* devAllPos = nullptr;
    double* devLocalPos = nullptr;
    double* devLocalVel = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devAllPos), totalDoubles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devLocalPos), localDoubles * sizeof(double)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devLocalVel), localDoubles * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(devLocalPos, hostLocalPos, 3 * localN * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devLocalVel, hostLocalVel, 3 * localN * sizeof(double),
                          cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int blockSize = 256;
    const int gridSize = (localN + blockSize - 1) / blockSize;

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Every rank needs every position to evaluate the forces on its own bodies.
        MPI_Allgatherv(hostLocalPos, 3 * localN, MPI_DOUBLE,
                       hostAllPos, dataCounts.data(), dataDispls.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpyAsync(devAllPos, hostAllPos, 3 * n * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        if (localN > 0) {
            computeForcesKernel<<<gridSize, blockSize, 0, stream>>>(
                devAllPos, devLocalVel, n, localStart, localN);
            integrateKernel<<<gridSize, blockSize, 0, stream>>>(devLocalPos, devLocalVel, localN);
        }
        CUDA_CHECK(cudaMemcpyAsync(hostLocalPos, devLocalPos, 3 * localN * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", static_cast<long>(duration.count()));
    }

    // Gather the final positions (and velocities when printing) for output and
    // for the global energy calculation.
    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(hostLocalVel, devLocalVel, 3 * localN * sizeof(double),
                              cudaMemcpyDeviceToHost));
        MPI_Allgatherv(hostLocalPos, 3 * localN, MPI_DOUBLE,
                       hostAllPos, dataCounts.data(), dataDispls.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
    }
    if (printResults) {
        MPI_Allgatherv(hostLocalVel, 3 * localN, MPI_DOUBLE,
                       hostAllVel, dataCounts.data(), dataDispls.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        if (rank == 0) {
            std::vector<double> bodyData(static_cast<size_t>(6 * n));
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                const int b = 3 * i;
                const int o = 6 * i;
                bodyData[o + 0] = hostAllPos[b + 0];
                bodyData[o + 1] = hostAllPos[b + 1];
                bodyData[o + 2] = hostAllPos[b + 2];
                bodyData[o + 3] = hostAllVel[b + 0];
                bodyData[o + 4] = hostAllVel[b + 1];
                bodyData[o + 5] = hostAllVel[b + 2];
            }
            print_results(bodyData, "Bodies");
        }
    }

    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating simulation results...\n");
        }

        int localError = 0;
#pragma omp parallel for
        for (int t = 0; t < localN; ++t) {
            const int o = 3 * t;
            const double px = hostLocalPos[o + 0];
            const double py = hostLocalPos[o + 1];
            const double pz = hostLocalPos[o + 2];
            const double vx = hostLocalVel[o + 0];
            const double vy = hostLocalVel[o + 1];
            const double vz = hostLocalVel[o + 2];

            int err = 0;
            if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz) ||
                !std::isfinite(vx) || !std::isfinite(vy) || !std::isfinite(vz)) {
                err = 1;
            } else if (std::abs(px) > 1e6 || std::abs(py) > 1e6 || std::abs(pz) > 1e6) {
                err = 2;
            } else if (std::abs(vx) > 1e6 || std::abs(vy) > 1e6 || std::abs(vz) > 1e6) {
                err = 3;
            }
            if (err != 0) {
#pragma omp critical
                {
                    if (localError == 0) {
                        localError = err;
                    }
                }
            }
        }

        int globalError = 0;
        MPI_Allreduce(&localError, &globalError, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        double finalEnergy = 0.0;
        if (globalError == 0) {
            double localKinetic = 0.0;
#pragma omp parallel for reduction(+ : localKinetic)
            for (int t = 0; t < localN; ++t) {
                const int o = 3 * t;
                const double vx = hostLocalVel[o + 0];
                const double vy = hostLocalVel[o + 1];
                const double vz = hostLocalVel[o + 2];
                localKinetic += 0.5 * (vx * vx + vy * vy + vz * vz);
            }

            double localPotential = 0.0;
#pragma omp parallel for reduction(+ : localPotential) schedule(dynamic, 64)
            for (int tt = 0; tt < localN; ++tt) {
                const int i = localStart + tt;
                const int bi = 3 * i;
                const double xi = hostAllPos[bi + 0];
                const double yi = hostAllPos[bi + 1];
                const double zi = hostAllPos[bi + 2];
                double contribution = 0.0;
                for (int j = i + 1; j < n; ++j) {
                    const int bj = 3 * j;
                    const double dx = hostAllPos[bj + 0] - xi;
                    const double dy = hostAllPos[bj + 1] - yi;
                    const double dz = hostAllPos[bj + 2] - zi;
                    const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                    contribution -= 1.0 / dist;
                }
                localPotential += contribution;
            }

            double globalKinetic = 0.0;
            double globalPotential = 0.0;
            MPI_Reduce(&localKinetic, &globalKinetic, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&localPotential, &globalPotential, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            finalEnergy = globalKinetic + globalPotential;
        }

        if (rank == 0) {
            if (globalError == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                if (globalError == 1) {
                    printf("Validation failed: found NaN or Inf value in body state\n");
                } else if (globalError == 2) {
                    printf("Validation failed: body position exceeds reasonable bounds\n");
                } else {
                    printf("Validation failed: body velocity exceeds reasonable bounds\n");
                }
                printf("Validation: FAILED\n");
            }
        }
        exitCode = (globalError == 0) ? 0 : 1;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(devAllPos));
    CUDA_CHECK(cudaFree(devLocalPos));
    CUDA_CHECK(cudaFree(devLocalVel));
    CUDA_CHECK(cudaFreeHost(hostLocalPos));
    CUDA_CHECK(cudaFreeHost(hostLocalVel));
    CUDA_CHECK(cudaFreeHost(hostAllPos));
    if (hostAllVel != nullptr) {
        CUDA_CHECK(cudaFreeHost(hostAllVel));
    }

    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return exitCode;
}
