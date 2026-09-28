#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                     \
        if (err__ != cudaSuccess) {                                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,            \
                    cudaGetErrorString(err__));                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                               \
        }                                                                                \
    } while (0)

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

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

// CUDA kernel: for each local body i (owned by this rank), sum the gravitational
// force contribution from every body j = 0..n-1, in strictly increasing j order,
// via shared-memory tiling. Preserving the increasing-j accumulation order keeps
// floating point results equivalent to the original sequential double-precision loop.
__global__ void computeForcesKernel(const double* __restrict__ px, const double* __restrict__ py,
                                     const double* __restrict__ pz, double* __restrict__ Fx,
                                     double* __restrict__ Fy, double* __restrict__ Fz, size_t n,
                                     size_t localStart, size_t localCount) {
    extern __shared__ double tile[];
    double* tpx = tile;
    double* tpy = tile + blockDim.x;
    double* tpz = tile + 2 * blockDim.x;

    const size_t localIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = localIdx < localCount;
    const size_t i = localStart + localIdx;

    double myx = 0.0, myy = 0.0, myz = 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    if (active) {
        myx = px[i];
        myy = py[i];
        myz = pz[i];
    }

    for (size_t tileStart = 0; tileStart < n; tileStart += blockDim.x) {
        const size_t j = tileStart + threadIdx.x;
        if (j < n) {
            tpx[threadIdx.x] = px[j];
            tpy[threadIdx.x] = py[j];
            tpz[threadIdx.x] = pz[j];
        }
        __syncthreads();

        const size_t tileLen = min(static_cast<size_t>(blockDim.x), n - tileStart);
        if (active) {
            for (size_t k = 0; k < tileLen; ++k) {
                const double dx = tpx[k] - myx;
                const double dy = tpy[k] - myy;
                const double dz = tpz[k] - myz;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        Fx[localIdx] = fx;
        Fy[localIdx] = fy;
        Fz[localIdx] = fz;
    }
}

// Distributes ownership of bodies [0, n) as evenly as possible across `numRanks` ranks.
static void computeRange(size_t n, int numRanks, int rank, size_t& start, size_t& count) {
    const size_t base = n / numRanks;
    const size_t rem = n % numRanks;
    start = static_cast<size_t>(rank) * base + std::min<size_t>(rank, rem);
    count = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
}

// Computes forces for the local slice of bodies owned by this rank, using this
// rank's assigned CUDA device against the globally replicated position array.
void computeForcesLocal(const double* px, const double* py, const double* pz, double* localFx,
                         double* localFy, double* localFz, size_t n, size_t localStart,
                         size_t localCount, double* d_px, double* d_py, double* d_pz, double* d_Fx,
                         double* d_Fy, double* d_Fz) {
    if (localCount == 0) return;

    CUDA_CHECK(cudaMemcpy(d_px, px, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, py, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pz, pz, n * sizeof(double), cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = static_cast<int>((localCount + blockSize - 1) / blockSize);
    const size_t sharedBytes = 3 * blockSize * sizeof(double);

    computeForcesKernel<<<gridSize, blockSize, sharedBytes>>>(d_px, d_py, d_pz, d_Fx, d_Fy, d_Fz, n,
                                                               localStart, localCount);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(localFx, d_Fx, localCount * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(localFy, d_Fy, localCount * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(localFz, d_Fz, localCount * sizeof(double), cudaMemcpyDeviceToHost));
}

void integrateBodiesLocal(std::vector<Body>& bodies, size_t localStart, size_t localCount) {
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < localCount; ++idx) {
        Body& body = bodies[localStart + idx];
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (assuming unit mass)
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const auto& body = bodies[i];
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies)
    #pragma omp parallel for reduction(+:energy) schedule(dynamic, 64)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
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
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Bind this rank to a CUDA device (round-robin across available GPUs).
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (worldRank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int myDevice = worldRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(myDevice));

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

    if (worldRank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, CUDA devices/node: %d, OpenMP threads/rank: %d\n", worldSize,
               deviceCount, omp_get_max_threads());
    }

    // Initialize bodies (identical on every rank, so the globally-replicated
    // position array stays consistent without needing a broadcast).
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Determine this rank's slice of bodies to own/update.
    size_t localStart = 0, localCount = 0;
    computeRange(static_cast<size_t>(numBodies), worldSize, worldRank, localStart, localCount);

    // Per-rank gather counts/displacements (in elements) for MPI_Allgatherv.
    std::vector<int> counts(worldSize), displs(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        size_t s, c;
        computeRange(static_cast<size_t>(numBodies), worldSize, r, s, c);
        counts[r] = static_cast<int>(c);
        displs[r] = static_cast<int>(s);
    }

    // Globally-replicated SoA position buffers (needed by every rank for the O(n^2) force sum).
    std::vector<double> px(numBodies), py(numBodies), pz(numBodies);
    for (int i = 0; i < numBodies; ++i) {
        px[i] = bodies[i].pos.x;
        py[i] = bodies[i].pos.y;
        pz[i] = bodies[i].pos.z;
    }

    std::vector<double> localFx(localCount), localFy(localCount), localFz(localCount);

    double *d_px = nullptr, *d_py = nullptr, *d_pz = nullptr;
    double *d_Fx = nullptr, *d_Fy = nullptr, *d_Fz = nullptr;
    CUDA_CHECK(cudaMalloc(&d_px, static_cast<size_t>(numBodies) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, static_cast<size_t>(numBodies) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pz, static_cast<size_t>(numBodies) * sizeof(double)));
    if (localCount > 0) {
        CUDA_CHECK(cudaMalloc(&d_Fx, localCount * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Fy, localCount * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Fz, localCount * sizeof(double)));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Compute forces on the locally-owned bodies against all n bodies on this rank's GPU.
        computeForcesLocal(px.data(), py.data(), pz.data(), localFx.data(), localFy.data(),
                            localFz.data(), static_cast<size_t>(numBodies), localStart, localCount,
                            d_px, d_py, d_pz, d_Fx, d_Fy, d_Fz);

        // Apply velocity update and integrate positions for the local slice (CPU, OpenMP).
        #pragma omp parallel for schedule(static)
        for (size_t idx = 0; idx < localCount; ++idx) {
            Body& body = bodies[localStart + idx];
            body.vel.x += DT * localFx[idx];
            body.vel.y += DT * localFy[idx];
            body.vel.z += DT * localFz[idx];
        }

        integrateBodiesLocal(bodies, localStart, localCount);

        // Publish updated local positions to every rank's replicated position array.
        std::vector<double> localPx(localCount), localPy(localCount), localPz(localCount);
        #pragma omp parallel for schedule(static)
        for (size_t idx = 0; idx < localCount; ++idx) {
            const Body& body = bodies[localStart + idx];
            localPx[idx] = body.pos.x;
            localPy[idx] = body.pos.y;
            localPz[idx] = body.pos.z;
        }

        MPI_Allgatherv(localPx.data(), static_cast<int>(localCount), MPI_DOUBLE, px.data(),
                       counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(localPy.data(), static_cast<int>(localCount), MPI_DOUBLE, py.data(),
                       counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(localPz.data(), static_cast<int>(localCount), MPI_DOUBLE, pz.data(),
                       counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather locally-owned velocities (positions are already globally replicated in px/py/pz)
    // so rank 0 has the full, up-to-date body state for reporting/validation.
    std::vector<double> localVx(localCount), localVy(localCount), localVz(localCount);
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < localCount; ++idx) {
        const Body& body = bodies[localStart + idx];
        localVx[idx] = body.vel.x;
        localVy[idx] = body.vel.y;
        localVz[idx] = body.vel.z;
    }
    std::vector<double> vx(numBodies), vy(numBodies), vz(numBodies);
    MPI_Allgatherv(localVx.data(), static_cast<int>(localCount), MPI_DOUBLE, vx.data(),
                   counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(localVy.data(), static_cast<int>(localCount), MPI_DOUBLE, vy.data(),
                   counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(localVz.data(), static_cast<int>(localCount), MPI_DOUBLE, vz.data(),
                   counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos.x = px[i];
        bodies[i].pos.y = py[i];
        bodies[i].pos.z = pz[i];
        bodies[i].vel.x = vx[i];
        bodies[i].vel.y = vy[i];
        bodies[i].vel.z = vz[i];
    }

    if (worldRank == 0) {
        printf("Simulation time: %ld ms\n", maxDuration);
    }

    int exitCode = 0;

    if (worldRank == 0) {
        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData;
            bodyData.reserve(numBodies * 6);
            for (const auto& body : bodies) {
                bodyData.push_back(body.pos.x);
                bodyData.push_back(body.pos.y);
                bodyData.push_back(body.pos.z);
                bodyData.push_back(body.vel.x);
                bodyData.push_back(body.vel.y);
                bodyData.push_back(body.vel.z);
            }
            print_results(bodyData, "Bodies");
        }

        // Validation: check that simulation produces finite, reasonable values
        if (validate) {
            printf("Validating simulation results...\n");

            if (validateSimulation(bodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    if (d_Fx) cudaFree(d_Fx);
    if (d_Fy) cudaFree(d_Fy);
    if (d_Fz) cudaFree(d_Fz);
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_pz);

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
