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

// ---------------------------------------------------------------------------
// Hybrid parallelization
//   MPI    : bodies are block-distributed across ranks (one rank per GPU);
//            updated positions are exchanged every step with MPI_Allgatherv.
//   CUDA   : each rank computes forces + integration for its own bodies on
//            its GPU using a shared-memory tiled all-pairs kernel.
//   OpenMP : host-side work (packing, validation, energy computation).
// Semantics match the sequential code: every velocity update in a step uses
// the positions from the start of that step, and the force on body i is
// accumulated over j = 0..n-1 in the original order.
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),      \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// Pairwise interaction with an explicitly pinned rounding sequence that
// reproduces the reference host build (GCC -O3 -march=native contracts
// dx*dx+dy*dy into fma(dx,dx,dy*dy) and the accumulations into FMAs), so the
// GPU results are bitwise identical to the original sequential program.
__device__ __forceinline__ void bodyInteraction(double dx, double dy, double dz, double& Fx, double& Fy,
                                                double& Fz) {
    const double distSqr = __dadd_rn(__fma_rn(dz, dz, __fma_rn(dx, dx, __dmul_rn(dy, dy))), SOFTENING);
    const double invDist = __ddiv_rn(1.0, __dsqrt_rn(distSqr));
    const double invDist3 = __dmul_rn(__dmul_rn(invDist, invDist), invDist);
    Fx = __fma_rn(dx, invDist3, Fx);
    Fy = __fma_rn(dy, invDist3, Fy);
    Fz = __fma_rn(dz, invDist3, Fz);
}

// pos    : positions of all n bodies (x,y,z interleaved), start of step
// vel    : velocities of the local bodies (x,y,z interleaved), updated in place
// newPos : updated positions of the local bodies (x,y,z interleaved)
template <int BS>
__global__ void __launch_bounds__(BS)
stepKernel(const double* __restrict__ pos, double* __restrict__ vel, double* __restrict__ newPos,
           int n, int offset, int nLocal) {
    __shared__ double sx[BS], sy[BS], sz[BS];

    const int li = blockIdx.x * BS + threadIdx.x;
    const bool active = li < nLocal;

    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (active) {
        const size_t gi = 3 * (size_t)(offset + li);
        xi = pos[gi];
        yi = pos[gi + 1];
        zi = pos[gi + 2];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = 0; tile < n; tile += BS) {
        const int j = tile + threadIdx.x;
        if (j < n) {
            sx[threadIdx.x] = pos[3 * (size_t)j];
            sy[threadIdx.x] = pos[3 * (size_t)j + 1];
            sz[threadIdx.x] = pos[3 * (size_t)j + 2];
        }
        __syncthreads();

        if (active) {
            const int m = min(BS, n - tile);
            if (m == BS) {
#pragma unroll 8
                for (int k = 0; k < BS; ++k) {
                    const double dx = sx[k] - xi;
                    const double dy = sy[k] - yi;
                    const double dz = sz[k] - zi;
                    bodyInteraction(dx, dy, dz, Fx, Fy, Fz);
                }
            } else {
                for (int k = 0; k < m; ++k) {
                    const double dx = sx[k] - xi;
                    const double dy = sy[k] - yi;
                    const double dz = sz[k] - zi;
                    bodyInteraction(dx, dy, dz, Fx, Fy, Fz);
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        const size_t l = 3 * (size_t)li;
        const double vx = __dadd_rn(vel[l], __dmul_rn(DT, Fx));
        const double vy = __dadd_rn(vel[l + 1], __dmul_rn(DT, Fy));
        const double vz = __dadd_rn(vel[l + 2], __dmul_rn(DT, Fz));
        vel[l] = vx;
        vel[l + 1] = vy;
        vel[l + 2] = vz;
        newPos[l] = __fma_rn(vx, DT, xi);
        newPos[l + 1] = __fma_rn(vy, DT, yi);
        newPos[l + 2] = __fma_rn(vz, DT, zi);
    }
}

static int g_blockSize = 128;

static void launchStep(const double* dPos, double* dVel, double* dNewPos, int n, int offset, int nLocal,
                       cudaStream_t stream) {
    if (nLocal <= 0) return;
    const int bs = g_blockSize;
    const int blocks = (nLocal + bs - 1) / bs;
    switch (bs) {
        case 64: stepKernel<64><<<blocks, 64, 0, stream>>>(dPos, dVel, dNewPos, n, offset, nLocal); break;
        case 128: stepKernel<128><<<blocks, 128, 0, stream>>>(dPos, dVel, dNewPos, n, offset, nLocal); break;
        default: stepKernel<256><<<blocks, 256, 0, stream>>>(dPos, dVel, dNewPos, n, offset, nLocal); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

// Distributed total energy. pos holds all n bodies, vel holds this rank's
// local bodies. Result is valid on rank 0.
double computeTotalEnergy(const double* pos, const double* vel, int n, int nLocal, int rank, int size) {
    double kinetic = 0.0;
    // Kinetic energy (assuming unit mass)
#pragma omp parallel for reduction(+ : kinetic) schedule(static)
    for (int i = 0; i < nLocal; ++i) {
        const double* v = vel + 3 * (size_t)i;
        kinetic += 0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }

    // Potential energy (assuming unit mass for all bodies); rows are
    // distributed cyclically across ranks to balance the triangular loop.
    double potential = 0.0;
#pragma omp parallel for reduction(+ : potential) schedule(dynamic, 16)
    for (int i = rank; i < n; i += size) {
        const double xi = pos[3 * (size_t)i], yi = pos[3 * (size_t)i + 1], zi = pos[3 * (size_t)i + 2];
        double local = 0.0;
        for (int j = i + 1; j < n; ++j) {
            const double dx = pos[3 * (size_t)j] - xi;
            const double dy = pos[3 * (size_t)j + 1] - yi;
            const double dz = pos[3 * (size_t)j + 2] - zi;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            local += 1.0 / dist;
        }
        potential += local;
    }

    double partial[2] = {kinetic, potential}, total[2] = {0.0, 0.0};
    MPI_Reduce(partial, total, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return total[0] - total[1];
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    const long n = (long)bodies.size();
    const double maxPos = 1e6;
    const double maxVel = 1e6;
    // Find the first offending body (same message as the sequential scan).
    long firstBad = n;
    int badKind = 0;
#pragma omp parallel for reduction(min : firstBad) schedule(static)
    for (long i = 0; i < n; ++i) {
        const Body& body = bodies[i];
        const bool bad =
            !std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z) ||
            std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos ||
            std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel;
        if (bad && i < firstBad) firstBad = i;
    }
    if (firstBad == n) return true;

    const Body& body = bodies[firstBad];
    if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
        !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
        badKind = 0;
    } else if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
        badKind = 1;
    } else {
        badKind = 2;
    }
    if (badKind == 0) printf("Validation failed: found NaN or Inf value in body state\n");
    else if (badKind == 1) printf("Validation failed: body position exceeds reasonable bounds\n");
    else printf("Validation failed: body velocity exceeds reasonable bounds\n");
    return false;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
    if (numBodies < 0) numBodies = 0;
    
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        fflush(stdout);
    }

    // Bind each rank to a GPU on its node.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices <= 0) {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % numDevices;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // Block distribution of bodies over ranks.
    std::vector<int> counts3(size), displs3(size);
    for (int r = 0; r < size; ++r) {
        const long long lo = (long long)numBodies * r / size;
        const long long hi = (long long)numBodies * (r + 1) / size;
        counts3[r] = (int)(3 * (hi - lo));
        displs3[r] = (int)(3 * lo);
    }
    const int offset = displs3[rank] / 3;
    const int nLocal = counts3[rank] / 3;

    // Pick a block size that keeps all SMs busy for this local problem size.
    {
        const int targetBlocks = 2 * prop.multiProcessorCount;
        if ((nLocal + 255) / 256 >= targetBlocks) g_blockSize = 256;
        else if ((nLocal + 127) / 128 >= targetBlocks) g_blockSize = 128;
        else g_blockSize = 64;
    }
    
    // Initialize bodies (deterministic, identical on every rank)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    const size_t n3 = 3 * (size_t)numBodies;
    const size_t l3 = 3 * (size_t)nLocal;
    double* hPos = nullptr;  // all positions (pinned)
    double* hVel = nullptr;  // local velocities (pinned)
    CUDA_CHECK(cudaMallocHost(&hPos, std::max<size_t>(n3, 1) * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&hVel, std::max<size_t>(l3, 1) * sizeof(double)));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        hPos[3 * (size_t)i] = bodies[i].pos.x;
        hPos[3 * (size_t)i + 1] = bodies[i].pos.y;
        hPos[3 * (size_t)i + 2] = bodies[i].pos.z;
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nLocal; ++i) {
        hVel[3 * (size_t)i] = bodies[offset + i].vel.x;
        hVel[3 * (size_t)i + 1] = bodies[offset + i].vel.y;
        hVel[3 * (size_t)i + 2] = bodies[offset + i].vel.z;
    }

    double *dPos = nullptr, *dVel = nullptr, *dNewPos = nullptr;
    CUDA_CHECK(cudaMalloc(&dPos, std::max<size_t>(n3, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dVel, std::max<size_t>(l3, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dNewPos, std::max<size_t>(l3, 1) * sizeof(double)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (n3) CUDA_CHECK(cudaMemcpy(dPos, hPos, n3 * sizeof(double), cudaMemcpyHostToDevice));
    if (l3) CUDA_CHECK(cudaMemcpy(dVel, hVel, l3 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        launchStep(dPos, dVel, dNewPos, numBodies, offset, nLocal, stream);
        if (size == 1) {
            if (l3) CUDA_CHECK(cudaMemcpyAsync(dPos, dNewPos, l3 * sizeof(double), cudaMemcpyDeviceToDevice, stream));
        } else {
            if (l3) CUDA_CHECK(cudaMemcpyAsync(hPos + 3 * (size_t)offset, dNewPos, l3 * sizeof(double),
                                               cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hPos, counts3.data(), displs3.data(), MPI_DOUBLE,
                           MPI_COMM_WORLD);
            if (n3) CUDA_CHECK(cudaMemcpyAsync(dPos, hPos, n3 * sizeof(double), cudaMemcpyHostToDevice, stream));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) printf("Simulation time: %ld ms\n", duration.count());

    // Bring final state back to the host.
    if (size == 1 && n3) CUDA_CHECK(cudaMemcpy(hPos, dPos, n3 * sizeof(double), cudaMemcpyDeviceToHost));
    if (l3) CUDA_CHECK(cudaMemcpy(hVel, dVel, l3 * sizeof(double), cudaMemcpyDeviceToHost));

    int exitCode = 0;
    if (printResults || validate) {
        // Gather all velocities on rank 0 and rebuild the body array there.
        std::vector<double> allVel(rank == 0 ? std::max<size_t>(n3, 1) : 1);
        MPI_Gatherv(hVel, (int)l3, MPI_DOUBLE, allVel.data(), counts3.data(), displs3.data(), MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        if (rank == 0) {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                bodies[i].pos = Vec3(hPos[3 * (size_t)i], hPos[3 * (size_t)i + 1], hPos[3 * (size_t)i + 2]);
                bodies[i].vel = Vec3(allVel[3 * (size_t)i], allVel[3 * (size_t)i + 1], allVel[3 * (size_t)i + 2]);
            }
        }
    }
    
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
        int ok = 1;
        if (rank == 0) {
            printf("Validating simulation results...\n");
            ok = validateSimulation(bodies) ? 1 : 0;
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (ok) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(hPos, hVel, numBodies, nLocal, rank, size);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            exitCode = 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(dPos));
    CUDA_CHECK(cudaFree(dVel));
    CUDA_CHECK(cudaFree(dNewPos));
    CUDA_CHECK(cudaFreeHost(hPos));
    CUDA_CHECK(cudaFreeHost(hVel));
    MPI_Finalize();
    return exitCode;
}
