// Hybrid MPI + OpenMP + CUDA N-body benchmark.
//
// Decomposition:
//   - MPI: bodies are block-partitioned across ranks. Each rank owns the
//     velocities and positions of its block and computes forces for it.
//     Positions are replicated on every rank with MPI_Allgatherv once per step.
//   - CUDA: each rank drives one GPU (selected by node-local rank). The force
//     kernel is tiled through shared memory; every body accumulates the
//     contributions of all other bodies in the same j-order as the serial code,
//     so results are independent of the number of ranks.
//   - OpenMP: host-side O(N^2) potential-energy evaluation used by validation.
//
// Semantics are preserved: the same integrator, the same per-body summation
// order, and the same output format as the original serial program.

#ifndef _OPENMP
#error "OpenMP is required"
#endif

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int BLOCK = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 must be tightly packed");

#define CUDA_CHECK(call)                                                                              \
    do {                                                                                              \
        cudaError_t err_ = (call);                                                                    \
        if (err_ != cudaSuccess) {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                             \
        }                                                                                             \
    } while (0)

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

// Force computation for the locally owned bodies [first, first + count).
// Each thread owns one body and loops over all n bodies in increasing j order,
// exactly as the serial reference does. Tiles of positions are staged in shared
// memory; the tile loop keeps the j-order intact.
// Forces are computed from the positions at the start of the step; positions are
// only modified by integrateKernel, which runs as a separate launch.
__global__ void computeForcesKernel(const Vec3* __restrict__ pos, Vec3* __restrict__ vel,
                                    int n, int first, int count) {
    __shared__ double sx[BLOCK], sy[BLOCK], sz[BLOCK];

    const int tid = threadIdx.x;
    const int li = blockIdx.x * BLOCK + tid;  // local index into owned range
    const bool active = li < count;

    double pix = 0.0, piy = 0.0, piz = 0.0;
    if (active) {
        const Vec3 p = pos[first + li];
        pix = p.x;
        piy = p.y;
        piz = p.z;
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int t = 0; t < n; t += BLOCK) {
        const int j = t + tid;
        if (j < n) {
            const Vec3 p = pos[j];
            sx[tid] = p.x;
            sy[tid] = p.y;
            sz[tid] = p.z;
        }
        __syncthreads();

        if (active) {
            const int lim = (n - t < BLOCK) ? (n - t) : BLOCK;
            for (int k = 0; k < lim; ++k) {
                const double dx = sx[k] - pix;
                const double dy = sy[k] - piy;
                const double dz = sz[k] - piz;
                // Same fused multiply-add contraction as the reference build, so the
                // device arithmetic reproduces the host results bit-for-bit.
                double r2 = dy * dy;
                r2 = fma(dx, dx, r2);
                r2 = fma(dz, dz, r2);
                const double distSqr = r2 + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx = fma(dx, invDist3, Fx);
                Fy = fma(dy, invDist3, Fy);
                Fz = fma(dz, invDist3, Fz);
            }
        }
        __syncthreads();
    }

    if (active) {
        vel[li].x += DT * Fx;
        vel[li].y += DT * Fy;
        vel[li].z += DT * Fz;
    }
}

// Position update for the locally owned bodies.
__global__ void integrateKernel(Vec3* __restrict__ pos, const Vec3* __restrict__ vel, int first, int count) {
    const int li = blockIdx.x * BLOCK + threadIdx.x;
    if (li < count) {
        const Vec3 v = vel[li];
        Vec3& p = pos[first + li];
        p.x = fma(v.x, DT, p.x);
        p.y = fma(v.y, DT, p.y);
        p.z = fma(v.z, DT, p.z);
    }
}

// Total energy with the same accumulation order as the serial reference.
// The O(N^2) pair terms are evaluated in parallel with OpenMP in blocks of rows,
// and then accumulated sequentially in reference order, so the result is
// bit-identical regardless of the thread count.
double computeTotalEnergy(const std::vector<Vec3>& pos, const std::vector<Vec3>& vel) {
    double energy = 0.0;
    const int n = static_cast<int>(pos.size());

    // Kinetic energy (assuming unit mass)
    for (const auto& v : vel) {
        energy += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }

    if (n < 2) return energy;

    // Potential energy (assuming unit mass for all bodies)
    constexpr size_t kMaxBlockTerms = size_t(1) << 22;
    const int rowsPerBlock = static_cast<int>(
        std::min<size_t>(static_cast<size_t>(n), std::max<size_t>(1, kMaxBlockTerms / static_cast<size_t>(n))));
    std::vector<double> terms(static_cast<size_t>(rowsPerBlock) * static_cast<size_t>(n));

    for (int i0 = 0; i0 < n; i0 += rowsPerBlock) {
        const int i1 = std::min(n, i0 + rowsPerBlock);

#pragma omp parallel for schedule(dynamic, 1)
        for (int i = i0; i < i1; ++i) {
            double* row = terms.data() + static_cast<size_t>(i - i0) * static_cast<size_t>(n);
            for (int j = i + 1; j < n; ++j) {
                const double dx = pos[j].x - pos[i].x;
                const double dy = pos[j].y - pos[i].y;
                const double dz = pos[j].z - pos[i].z;
                const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                row[j] = 1.0 / dist;
            }
        }

        for (int i = i0; i < i1; ++i) {
            const double* row = terms.data() + static_cast<size_t>(i - i0) * static_cast<size_t>(n);
            for (int j = i + 1; j < n; ++j) {
                energy -= row[j];
            }
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

static int runSimulation(int argc, char** argv, int rank, int size) {
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
        if (rank == 0) printf("Invalid number of bodies: %d\n", numBodies);
        return 1;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Bind each rank to a GPU based on its node-local rank
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        fprintf(stderr, "No CUDA devices available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devCount));

    const int n = numBodies;

    // Initial state (identical on every rank: same seed)
    std::vector<Body> bodies(n);
    randomizeBodies(bodies);
    std::vector<Vec3> h_pos(n), h_vel(n);
    for (int i = 0; i < n; ++i) {
        h_pos[i] = bodies[i].pos;
        h_vel[i] = bodies[i].vel;
    }
    std::vector<Body>().swap(bodies);

    // Block partition of bodies across ranks
    std::vector<int> counts(size), displs(size), cnt3(size), dsp3(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = n / size + (r < n % size ? 1 : 0);
        displs[r] = (r == 0) ? 0 : displs[r - 1] + counts[r - 1];
        cnt3[r] = counts[r] * 3;
        dsp3[r] = displs[r] * 3;
    }
    const int first = displs[rank];
    const int count = counts[rank];

    // Device allocations (outside the timed region)
    Vec3* d_pos = nullptr;
    Vec3* d_vel = nullptr;
    if (n > 0) CUDA_CHECK(cudaMalloc(&d_pos, n * sizeof(Vec3)));
    if (count > 0) {
        CUDA_CHECK(cudaMalloc(&d_vel, count * sizeof(Vec3)));
        CUDA_CHECK(cudaMemcpy(d_vel, h_vel.data() + first, count * sizeof(Vec3), cudaMemcpyHostToDevice));
    }
    const int gridBlocks = (count + BLOCK - 1) / BLOCK;

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        if (n > 0) {
            CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(), n * sizeof(Vec3), cudaMemcpyHostToDevice));
        }
        if (count > 0) {
            computeForcesKernel<<<gridBlocks, BLOCK>>>(d_pos, d_vel, n, first, count);
            integrateKernel<<<gridBlocks, BLOCK>>>(d_pos, d_vel, first, count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_pos.data() + first, d_pos + first, count * sizeof(Vec3),
                                  cudaMemcpyDeviceToHost));
        }
        // Replicate updated positions on all ranks for the next step
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, reinterpret_cast<double*>(h_pos.data()),
                       cnt3.data(), dsp3.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }

    auto end = std::chrono::high_resolution_clock::now();
    const long long localMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long maxMs = 0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Simulation time: %lld ms\n", maxMs);

    // Gather final velocities when needed for output or validation
    const bool needVel = printResults || validate;
    if (needVel) {
        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(h_vel.data() + first, d_vel, count * sizeof(Vec3), cudaMemcpyDeviceToHost));
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, reinterpret_cast<double*>(h_vel.data()),
                       cnt3.data(), dsp3.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }

    int rc = 0;

    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(static_cast<size_t>(n) * 6);
        for (int i = 0; i < n; ++i) {
            bodyData.push_back(h_pos[i].x);
            bodyData.push_back(h_pos[i].y);
            bodyData.push_back(h_pos[i].z);
            bodyData.push_back(h_vel[i].x);
            bodyData.push_back(h_vel[i].y);
            bodyData.push_back(h_vel[i].z);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation: check that simulation produces finite, reasonable values
    if (validate && rank == 0) {
        printf("Validating simulation results...\n");

        std::vector<Body> finalBodies(n);
        for (int i = 0; i < n; ++i) {
            finalBodies[i].pos = h_pos[i];
            finalBodies[i].vel = h_vel[i];
        }

        if (validateSimulation(finalBodies)) {
            const double finalEnergy = computeTotalEnergy(h_pos, h_vel);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            rc = 1;
        }
    }

    if (d_pos) CUDA_CHECK(cudaFree(d_pos));
    if (d_vel) CUDA_CHECK(cudaFree(d_vel));

    return rc;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rc = runSimulation(argc, argv, rank, size);

    MPI_Finalize();
    return rc;
}
