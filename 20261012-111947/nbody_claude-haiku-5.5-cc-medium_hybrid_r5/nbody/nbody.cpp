// Hybrid MPI + OpenMP + CUDA n-body benchmark.
//   MPI    : bodies are block-partitioned across ranks; positions are
//            allgathered after every step so each rank holds the full state.
//   OpenMP : one host thread per local GPU; each thread drives its own device.
//   CUDA   : force, integration and energy kernels run on every GPU.

#include <algorithm>
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
constexpr int BLOCK_SIZE = 256;

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        const cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err_));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// Body state is stored as flat arrays holding x, y, z for each body.
void randomizeBodies(std::vector<double>& pos, std::vector<double>& vel, unsigned int seed = 42) {
    const size_t n = pos.size() / 3;
    for (size_t i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            pos[3 * i + c] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        }
        for (int c = 0; c < 3; ++c) {
            vel[3 * i + c] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        }
    }
}

// One thread per locally owned body. Each thread accumulates the interaction with
// j = 0..n-1 in ascending order, matching the serial summation order. Positions are
// staged through shared memory in tiles, which does not change that order.
__global__ void computeForcesKernel(const double* __restrict__ pos, int n,
                                    double* __restrict__ vel, int lo, int count) {
    __shared__ double tile[BLOCK_SIZE * 3];

    const int k = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool active = k < count;

    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (active) {
        const int i = lo + k;
        xi = pos[3 * i];
        yi = pos[3 * i + 1];
        zi = pos[3 * i + 2];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int base = 0; base < n; base += BLOCK_SIZE) {
        const int j = base + threadIdx.x;
        if (j < n) {
            tile[3 * threadIdx.x] = pos[3 * j];
            tile[3 * threadIdx.x + 1] = pos[3 * j + 1];
            tile[3 * threadIdx.x + 2] = pos[3 * j + 2];
        }
        __syncthreads();

        if (active) {
            const int tileCount = min(BLOCK_SIZE, n - base);
            for (int t = 0; t < tileCount; ++t) {
                const double dx = tile[3 * t] - xi;
                const double dy = tile[3 * t + 1] - yi;
                const double dz = tile[3 * t + 2] - zi;
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
        vel[3 * k] += DT * Fx;
        vel[3 * k + 1] += DT * Fy;
        vel[3 * k + 2] += DT * Fz;
    }
}

__global__ void integrateKernel(double* __restrict__ pos, const double* __restrict__ vel,
                                int lo, int count) {
    const int k = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (k < count) {
        double* p = pos + 3 * (lo + k);
        p[0] += vel[3 * k] * DT;
        p[1] += vel[3 * k + 1] * DT;
        p[2] += vel[3 * k + 2] * DT;
    }
}

// Per-body partial potential energy over pairs (i, j) with j > i.
__global__ void potentialKernel(const double* __restrict__ pos, int n, int lo, int count,
                                double* __restrict__ partial) {
    const int k = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (k >= count) {
        return;
    }

    const int i = lo + k;
    const double xi = pos[3 * i];
    const double yi = pos[3 * i + 1];
    const double zi = pos[3 * i + 2];

    double energy = 0.0;
    for (int j = i + 1; j < n; ++j) {
        const double dx = pos[3 * j] - xi;
        const double dy = pos[3 * j + 1] - yi;
        const double dz = pos[3 * j + 2] - zi;
        const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        energy -= 1.0 / dist;
    }
    partial[k] = energy;
}

// Validate that simulation produces finite, reasonable values.
// Returns nullptr if the state is valid, otherwise a description of the first failure.
const char* validateSimulation(const std::vector<double>& pos, const std::vector<double>& vel) {
    const size_t n = pos.size() / 3;
    const double maxPos = 1e6;
    const double maxVel = 1e6;
    for (size_t i = 0; i < n; ++i) {
        const double* p = &pos[3 * i];
        const double* v = &vel[3 * i];

        // Check for NaN or Inf values
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]) ||
            !std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) {
            return "Validation failed: found NaN or Inf value in body state";
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        if (std::abs(p[0]) > maxPos || std::abs(p[1]) > maxPos || std::abs(p[2]) > maxPos) {
            return "Validation failed: body position exceeds reasonable bounds";
        }
        if (std::abs(v[0]) > maxVel || std::abs(v[1]) > maxVel || std::abs(v[2]) > maxVel) {
            return "Validation failed: body velocity exceeds reasonable bounds";
        }
    }
    return nullptr;
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

// The slice of this rank's bodies handled by one GPU (and one OpenMP thread).
struct GpuSlice {
    int device = 0;
    int lo = 0;                  // first global body index of the slice
    int count = 0;               // number of bodies in the slice
    double* pos = nullptr;       // device copy of all positions (3 * numBodies)
    double* vel = nullptr;       // device velocities of the slice (3 * count)
    double* partial = nullptr;   // device per-body potential partial sums (count)
};

static int runBenchmark(int argc, char** argv, int rank, int numRanks) {
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (numBodies < 0) {
        if (rank == 0) {
            printf("Number of bodies must be non-negative\n");
        }
        return 1;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const int n = numBodies;
    const size_t stateSize = 3 * static_cast<size_t>(n);

    // Block partition of the bodies across ranks
    auto bodyStart = [n, numRanks](int r) {
        return static_cast<int>(static_cast<long long>(n) * r / numRanks);
    };
    const int rankLo = bodyStart(rank);
    const int rankCount = bodyStart(rank + 1) - rankLo;

    std::vector<int> recvCounts(numRanks);
    std::vector<int> displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        recvCounts[r] = 3 * (bodyStart(r + 1) - bodyStart(r));
        displs[r] = 3 * bodyStart(r);
    }

    // Every rank generates the identical full initial state (deterministic seed)
    std::vector<double> pos(stateSize), vel(stateSize);
    randomizeBodies(pos, vel);

    // Map this rank onto the GPUs of its node. Local ranks take devices round-robin,
    // and ranks beyond the device count share one device.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "No CUDA devices found on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<int> devices;
    for (int d = localRank; d < numDevices; d += localSize) {
        devices.push_back(d);
    }
    if (devices.empty()) {
        devices.push_back(localRank % numDevices);
    }

    // Split this rank's bodies across its GPUs and allocate device state
    const int numSlices = static_cast<int>(devices.size());
    std::vector<GpuSlice> slices(numSlices);
    for (int t = 0; t < numSlices; ++t) {
        GpuSlice& s = slices[t];
        const int begin = rankLo + static_cast<int>(static_cast<long long>(rankCount) * t / numSlices);
        const int end = rankLo + static_cast<int>(static_cast<long long>(rankCount) * (t + 1) / numSlices);
        s.device = devices[t];
        s.lo = begin;
        s.count = end - begin;

        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaMalloc(&s.pos, std::max<size_t>(stateSize, 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.vel, std::max<size_t>(3 * s.count, 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.partial, std::max<size_t>(s.count, 1) * sizeof(double)));
        if (s.count > 0) {
            CUDA_CHECK(cudaMemcpy(s.vel, vel.data() + 3 * s.lo, 3 * s.count * sizeof(double),
                                  cudaMemcpyHostToDevice));
        }
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(numSlices)
    {
        GpuSlice& s = slices[omp_get_thread_num()];
        CUDA_CHECK(cudaSetDevice(s.device));

        for (int step = 0; step < numSteps; ++step) {
            if (s.count > 0) {
                CUDA_CHECK(cudaMemcpy(s.pos, pos.data(), stateSize * sizeof(double),
                                      cudaMemcpyHostToDevice));
            }
            // All slices must finish reading the old positions before any slice writes its own back
            #pragma omp barrier

            if (s.count > 0) {
                const int blocks = (s.count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                computeForcesKernel<<<blocks, BLOCK_SIZE>>>(s.pos, n, s.vel, s.lo, s.count);
                integrateKernel<<<blocks, BLOCK_SIZE>>>(s.pos, s.vel, s.lo, s.count);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(pos.data() + 3 * s.lo, s.pos + 3 * s.lo,
                                      3 * s.count * sizeof(double), cudaMemcpyDeviceToHost));
            }
            #pragma omp barrier

            #pragma omp master
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pos.data(), recvCounts.data(),
                           displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            #pragma omp barrier
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long local_ms = duration.count();
    long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", max_ms);
    }

    // Velocities are only needed for result printing and validation
    const bool needState = printResults || validate;
    if (needState) {
        #pragma omp parallel num_threads(numSlices)
        {
            GpuSlice& s = slices[omp_get_thread_num()];
            CUDA_CHECK(cudaSetDevice(s.device));
            if (s.count > 0) {
                CUDA_CHECK(cudaMemcpy(vel.data() + 3 * s.lo, s.vel, 3 * s.count * sizeof(double),
                                      cudaMemcpyDeviceToHost));
            }
            #pragma omp barrier
            #pragma omp master
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, vel.data(), recvCounts.data(),
                           displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            #pragma omp barrier
        }
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(numBodies) * 6);
        for (int i = 0; i < n; ++i) {
            bodyData.push_back(pos[3 * i]);
            bodyData.push_back(pos[3 * i + 1]);
            bodyData.push_back(pos[3 * i + 2]);
            bodyData.push_back(vel[3 * i]);
            bodyData.push_back(vel[3 * i + 1]);
            bodyData.push_back(vel[3 * i + 2]);
        }
        print_results(bodyData, "Bodies");
    }

    int rc = 0;

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        if (rank == 0) {
            printf("Validating simulation results...\n");
        }

        const char* failure = validateSimulation(pos, vel);
        if (failure == nullptr) {
            // Kinetic energy (assuming unit mass); every rank holds the full state
            double kinetic = 0.0;
            for (int i = 0; i < n; ++i) {
                kinetic += 0.5 * (vel[3 * i] * vel[3 * i] +
                                  vel[3 * i + 1] * vel[3 * i + 1] +
                                  vel[3 * i + 2] * vel[3 * i + 2]);
            }

            // Potential energy (assuming unit mass for all bodies): computed per GPU
            // slice, summed on each rank, then reduced across ranks
            std::vector<double> partial(rankCount);
            #pragma omp parallel num_threads(numSlices)
            {
                GpuSlice& s = slices[omp_get_thread_num()];
                CUDA_CHECK(cudaSetDevice(s.device));
                if (s.count > 0) {
                    CUDA_CHECK(cudaMemcpy(s.pos, pos.data(), stateSize * sizeof(double),
                                          cudaMemcpyHostToDevice));
                    const int blocks = (s.count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    potentialKernel<<<blocks, BLOCK_SIZE>>>(s.pos, n, s.lo, s.count, s.partial);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaMemcpy(partial.data() + (s.lo - rankLo), s.partial,
                                          s.count * sizeof(double), cudaMemcpyDeviceToHost));
                }
            }

            double localPotential = 0.0;
            for (double p : partial) {
                localPotential += p;
            }
            double potential = 0.0;
            MPI_Allreduce(&localPotential, &potential, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            // Report final energy for reference
            if (rank == 0) {
                printf("Final energy: %.6f\n", kinetic + potential);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) {
                printf("%s\n", failure);
                printf("Validation: FAILED\n");
            }
            rc = 1;
        }
    }

    for (GpuSlice& s : slices) {
        CUDA_CHECK(cudaSetDevice(s.device));
        CUDA_CHECK(cudaFree(s.pos));
        CUDA_CHECK(cudaFree(s.vel));
        CUDA_CHECK(cudaFree(s.partial));
    }

    return rc;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    omp_set_dynamic(0);

    const int rc = runBenchmark(argc, argv, rank, numRanks);

    MPI_Finalize();
    return rc;
}
