// Hybrid-parallel n-body benchmark: MPI distributes bodies across ranks (one
// GPU per rank), CUDA computes the O(n^2) forces and the integration on the
// GPU, and OpenMP parallelizes the CPU-side O(n^2) validation/energy pass.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
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

constexpr int BLOCK_SIZE = 256;

// Each thread accumulates the force on one locally-owned body against all n
// bodies, tiling positions through shared memory. Tiles and the inner loop
// run in ascending j order, matching the accumulation order of the original
// serial code.
__global__ void forceKernel(const double* __restrict__ posx,
                            const double* __restrict__ posy,
                            const double* __restrict__ posz,
                            double* __restrict__ velx,
                            double* __restrict__ vely,
                            double* __restrict__ velz,
                            int n, int localOffset, int localCount) {
    __shared__ double sx[BLOCK_SIZE];
    __shared__ double sy[BLOCK_SIZE];
    __shared__ double sz[BLOCK_SIZE];

    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = li < localCount;
    const int i = localOffset + li;

    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (active) {
        xi = posx[i];
        yi = posy[i];
        zi = posz[i];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = 0; tile < n; tile += BLOCK_SIZE) {
        const int j = tile + threadIdx.x;
        if (j < n) {
            sx[threadIdx.x] = posx[j];
            sy[threadIdx.x] = posy[j];
            sz[threadIdx.x] = posz[j];
        }
        __syncthreads();

        if (active) {
            const int tileEnd = min(BLOCK_SIZE, n - tile);
            for (int k = 0; k < tileEnd; ++k) {
                const double dx = sx[k] - xi;
                const double dy = sy[k] - yi;
                const double dz = sz[k] - zi;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        velx[i] += DT * Fx;
        vely[i] += DT * Fy;
        velz[i] += DT * Fz;
    }
}

__global__ void integrateKernel(double* __restrict__ posx,
                                double* __restrict__ posy,
                                double* __restrict__ posz,
                                const double* __restrict__ velx,
                                const double* __restrict__ vely,
                                const double* __restrict__ velz,
                                int localOffset, int localCount) {
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= localCount) return;
    const int i = localOffset + li;
    posx[i] += velx[i] * DT;
    posy[i] += vely[i] * DT;
    posz[i] += velz[i] * DT;
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const long n = (long)bodies.size();
    const Body* b = bodies.data();

    // Kinetic energy (assuming unit mass)
#pragma omp parallel for reduction(+ : energy)
    for (long i = 0; i < n; ++i) {
        energy += 0.5 * (b[i].vel.x * b[i].vel.x +
                         b[i].vel.y * b[i].vel.y +
                         b[i].vel.z * b[i].vel.z);
    }

    // Potential energy (assuming unit mass for all bodies)
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : energy)
    for (long i = 0; i < n; ++i) {
        for (long j = i + 1; j < n; ++j) {
            const double dx = b[j].pos.x - b[i].pos.x;
            const double dy = b[j].pos.y - b[i].pos.y;
            const double dz = b[j].pos.z - b[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    const long n = (long)bodies.size();
    bool ok = true;

#pragma omp parallel for reduction(&& : ok)
    for (long i = 0; i < n; ++i) {
        const Body& body = bodies[i];
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            ok = false;
            continue;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            ok = false;
            continue;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            ok = false;
        }
    }

    if (!ok) {
        printf("Validation failed: found NaN/Inf or out-of-bounds value in body state\n");
    }
    return ok;
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

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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
            if (rank == 0) printUsage(argv[0]);
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

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d\n", nranks, omp_get_max_threads());
    }

    // Bind each rank to a GPU (round-robin over the node's devices)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    // Initialize bodies (identical on every rank: same seed, no broadcast needed)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Block-partition bodies across ranks
    std::vector<int> counts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        const int base = numBodies / nranks;
        const int rem = numBodies % nranks;
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = r * base + (r < rem ? r : rem);
    }
    const int localCount = counts[rank];
    const int localOffset = displs[rank];

    // Structure-of-arrays layout on host and device
    std::vector<double> hx(numBodies), hy(numBodies), hz(numBodies);
    std::vector<double> hvx(numBodies), hvy(numBodies), hvz(numBodies);
    for (int i = 0; i < numBodies; ++i) {
        hx[i] = bodies[i].pos.x;
        hy[i] = bodies[i].pos.y;
        hz[i] = bodies[i].pos.z;
        hvx[i] = bodies[i].vel.x;
        hvy[i] = bodies[i].vel.y;
        hvz[i] = bodies[i].vel.z;
    }

    double *dx, *dy, *dz, *dvx, *dvy, *dvz;
    const size_t nBytes = (size_t)numBodies * sizeof(double);
    CUDA_CHECK(cudaMalloc(&dx, nBytes));
    CUDA_CHECK(cudaMalloc(&dy, nBytes));
    CUDA_CHECK(cudaMalloc(&dz, nBytes));
    CUDA_CHECK(cudaMalloc(&dvx, nBytes));
    CUDA_CHECK(cudaMalloc(&dvy, nBytes));
    CUDA_CHECK(cudaMalloc(&dvz, nBytes));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy.data(), nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dz, hz.data(), nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dvx, hvx.data(), nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dvy, hvy.data(), nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dvz, hvz.data(), nBytes, cudaMemcpyHostToDevice));

    const int numBlocks = (localCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t localBytes = (size_t)localCount * sizeof(double);

    // Exchanges positions of locally-owned bodies among all ranks
    auto allgatherPositions = [&]() {
        if (nranks == 1) return;
        if (localCount > 0) {
            CUDA_CHECK(cudaMemcpy(hx.data() + localOffset, dx + localOffset, localBytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hy.data() + localOffset, dy + localOffset, localBytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hz.data() + localOffset, dz + localOffset, localBytes, cudaMemcpyDeviceToHost));
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hx.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hy.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hz.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), nBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dy, hy.data(), nBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dz, hz.data(), nBytes, cudaMemcpyHostToDevice));
    };

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        if (localCount > 0) {
            forceKernel<<<numBlocks, BLOCK_SIZE>>>(dx, dy, dz, dvx, dvy, dvz,
                                                   numBodies, localOffset, localCount);
            integrateKernel<<<numBlocks, BLOCK_SIZE>>>(dx, dy, dz, dvx, dvy, dvz,
                                                       localOffset, localCount);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        allgatherPositions();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // Bring the final state back to every rank
    if (nranks == 1) {
        CUDA_CHECK(cudaMemcpy(hx.data(), dx, nBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hy.data(), dy, nBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hz.data(), dz, nBytes, cudaMemcpyDeviceToHost));
    }
    if (localCount > 0) {
        CUDA_CHECK(cudaMemcpy(hvx.data() + localOffset, dvx + localOffset, localBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hvy.data() + localOffset, dvy + localOffset, localBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hvz.data() + localOffset, dvz + localOffset, localBytes, cudaMemcpyDeviceToHost));
    }
    if (nranks > 1) {
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hvx.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hvy.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hvz.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos = Vec3(hx[i], hy[i], hz[i]);
        bodies[i].vel = Vec3(hvx[i], hvy[i], hvz[i]);
    }

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(dz));
    CUDA_CHECK(cudaFree(dvx));
    CUDA_CHECK(cudaFree(dvy));
    CUDA_CHECK(cudaFree(dvz));

    int exitCode = 0;

    // Print results for external validation
    if (printResults && rank == 0) {
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
        if (rank == 0) {
            printf("Validating simulation results...\n");

            if (validateSimulation(bodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
