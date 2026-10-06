#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
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
// Hybrid parallelization:
//   MPI     : bodies are block-decomposed across ranks; positions are
//             exchanged every step with MPI_Allgatherv.
//   OpenMP  : one host thread per GPU owned by a rank (a rank may drive
//             several GPUs), plus threaded CPU work (energy, packing).
//   CUDA    : all-pairs force + integration kernel on each GPU.
// Floating-point operations are written explicitly (with -fmad=false) so the
// per-body arithmetic and summation order match the original serial code.
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                           \
    do {                                                                           \
        cudaError_t err_ = (call);                                                 \
        if (err_ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),  \
                    __FILE__, __LINE__);                                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                          \
        }                                                                          \
    } while (0)

// pos: all bodies, packed xyz. vel/posOut: bodies [i0, i0+cnt), packed xyz.
// Each thread owns one body and sums over j in the original sequential order.
template <int BLOCK>
__global__ void __launch_bounds__(BLOCK)
stepKernel(const double* __restrict__ pos, int n, int i0, int cnt,
           double* __restrict__ vel, double* __restrict__ posOut) {
    __shared__ double sx[BLOCK], sy[BLOCK], sz[BLOCK];

    const int li = blockIdx.x * BLOCK + threadIdx.x;
    const bool active = li < cnt;
    const int gi = i0 + (active ? li : 0);
    const double px = pos[3 * gi + 0];
    const double py = pos[3 * gi + 1];
    const double pz = pos[3 * gi + 2];
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += BLOCK) {
        const int j = base + threadIdx.x;
        if (j < n) {
            sx[threadIdx.x] = pos[3 * j + 0];
            sy[threadIdx.x] = pos[3 * j + 1];
            sz[threadIdx.x] = pos[3 * j + 2];
        }
        __syncthreads();
        const int m = min(BLOCK, n - base);
        if (m == BLOCK) {
#pragma unroll 8
            for (int k = 0; k < BLOCK; ++k) {
                const double dx = __dsub_rn(sx[k], px);
                const double dy = __dsub_rn(sy[k], py);
                const double dz = __dsub_rn(sz[k], pz);
                const double d2 = __dadd_rn(__fma_rn(dz, dz, __fma_rn(dx, dx, __dmul_rn(dy, dy))), SOFTENING);
                const double inv = __ddiv_rn(1.0, __dsqrt_rn(d2));
                const double inv3 = __dmul_rn(__dmul_rn(inv, inv), inv);
                Fx = __fma_rn(dx, inv3, Fx);
                Fy = __fma_rn(dy, inv3, Fy);
                Fz = __fma_rn(dz, inv3, Fz);
            }
        } else {
            for (int k = 0; k < m; ++k) {
                const double dx = __dsub_rn(sx[k], px);
                const double dy = __dsub_rn(sy[k], py);
                const double dz = __dsub_rn(sz[k], pz);
                const double d2 = __dadd_rn(__fma_rn(dz, dz, __fma_rn(dx, dx, __dmul_rn(dy, dy))), SOFTENING);
                const double inv = __ddiv_rn(1.0, __dsqrt_rn(d2));
                const double inv3 = __dmul_rn(__dmul_rn(inv, inv), inv);
                Fx = __fma_rn(dx, inv3, Fx);
                Fy = __fma_rn(dy, inv3, Fy);
                Fz = __fma_rn(dz, inv3, Fz);
            }
        }
        __syncthreads();
    }

    if (active) {
        const double vx = __dadd_rn(__dmul_rn(Fx, DT), vel[3 * li + 0]);
        const double vy = __dadd_rn(__dmul_rn(Fy, DT), vel[3 * li + 1]);
        const double vz = __dadd_rn(__dmul_rn(Fz, DT), vel[3 * li + 2]);
        vel[3 * li + 0] = vx;
        vel[3 * li + 1] = vy;
        vel[3 * li + 2] = vz;
        posOut[3 * li + 0] = __fma_rn(vx, DT, px);
        posOut[3 * li + 1] = __fma_rn(vy, DT, py);
        posOut[3 * li + 2] = __fma_rn(vz, DT, pz);
    }
}

struct Device {
    int id = 0;
    int i0 = 0;   // global index of first owned body
    int cnt = 0;  // number of owned bodies
    double* dPos = nullptr;     // all positions
    double* dPosOut = nullptr;  // updated positions of owned bodies
    double* dVel = nullptr;     // velocities of owned bodies
    cudaStream_t stream = nullptr;
    int block = 32;             // autotuned thread-block size
};

// Launch the step kernel with a runtime-selected block size.
static void launchStep(int block, const double* pos, int n, int i0, int cnt, double* vel,
                       double* posOut, cudaStream_t s) {
    const int grid = std::max((cnt + block - 1) / block, 1);
    switch (block) {
        case 32: stepKernel<32><<<grid, 32, 0, s>>>(pos, n, i0, cnt, vel, posOut); break;
        case 64: stepKernel<64><<<grid, 64, 0, s>>>(pos, n, i0, cnt, vel, posOut); break;
        case 128: stepKernel<128><<<grid, 128, 0, s>>>(pos, n, i0, cnt, vel, posOut); break;
        default: stepKernel<256><<<grid, 256, 0, s>>>(pos, n, i0, cnt, vel, posOut); break;
    }
}

// Pick the fastest block size for this device's row count. Load balance across
// SMs depends on the number of rows, so a truncated j-range suffices. The
// block size does not affect results (per-row arithmetic is identical).
static int tuneBlock(Device& d, int n) {
    if (d.cnt <= 0) return 32;
    const int nTune = std::min(n, 8192);
    double* scratchVel = nullptr;
    double* scratchOut = nullptr;
    CUDA_CHECK(cudaMalloc(&scratchVel, sizeof(double) * 3 * d.cnt));
    CUDA_CHECK(cudaMalloc(&scratchOut, sizeof(double) * 3 * d.cnt));
    CUDA_CHECK(cudaMemsetAsync(scratchVel, 0, sizeof(double) * 3 * d.cnt, d.stream));
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    int best = 32;
    float bestMs = 1e30f;
    for (int b : {32, 64, 128, 256}) {
        launchStep(b, d.dPos, nTune, d.i0, d.cnt, scratchVel, scratchOut, d.stream);  // warm-up
        CUDA_CHECK(cudaEventRecord(e0, d.stream));
        launchStep(b, d.dPos, nTune, d.i0, d.cnt, scratchVel, scratchOut, d.stream);
        CUDA_CHECK(cudaEventRecord(e1, d.stream));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < bestMs * 0.98f) {
            bestMs = ms;
            best = b;
        }
    }
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(scratchVel));
    CUDA_CHECK(cudaFree(scratchOut));
    return best;
}

// Distributed total energy: kinetic over owned bodies, potential over pairs
// (i, j>i) with rows distributed cyclically over ranks and OpenMP threads.
double computeTotalEnergy(const std::vector<Body>& bodies, int rank, int nranks) {
    const long n = (long)bodies.size();
    double local = 0.0;

    if (rank == 0) {
#pragma omp parallel for reduction(+ : local) schedule(static)
        for (long i = 0; i < n; ++i) {
            const Body& b = bodies[i];
            local += 0.5 * (b.vel.x * b.vel.x + b.vel.y * b.vel.y + b.vel.z * b.vel.z);
        }
    }

#pragma omp parallel for reduction(- : local) schedule(dynamic, 16)
    for (long i = rank; i < n; i += nranks) {
        const Vec3 pi = bodies[i].pos;
        double row = 0.0;
        for (long j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - pi.x;
            const double dy = bodies[j].pos.y - pi.y;
            const double dz = bodies[j].pos.z - pi.z;
            row += 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
        local -= row;
    }

    double energy = 0.0;
    MPI_Allreduce(&local, &energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
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
        fflush(stdout);
    }

    // Initialize bodies (identically on every rank)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    const int n = numBodies;

    // --- GPU assignment: split node-local GPUs among node-local ranks ---
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    int numGpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numGpus));
    if (numGpus < 1) {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<Device> devs;
    if (localSize >= numGpus) {
        devs.resize(1);
        devs[0].id = localRank % numGpus;
    } else {
        for (int d = localRank; d < numGpus; d += localSize) {
            devs.emplace_back();
            devs.back().id = d;
        }
    }
    const int nd = (int)devs.size();

    // --- Domain decomposition: contiguous body ranges per GPU worker ---
    std::vector<int> workersPerRank(nranks);
    MPI_Allgather(&nd, 1, MPI_INT, workersPerRank.data(), 1, MPI_INT, MPI_COMM_WORLD);
    long totalWorkers = 0, myFirstWorker = 0;
    for (int r = 0; r < nranks; ++r) {
        if (r == rank) myFirstWorker = totalWorkers;
        totalWorkers += workersPerRank[r];
    }
    auto workerStart = [&](long w) { return (int)((w * (long)n) / totalWorkers); };

    std::vector<int> counts3(nranks), displs3(nranks);
    {
        long w = 0;
        for (int r = 0; r < nranks; ++r) {
            const int lo = workerStart(w);
            w += workersPerRank[r];
            const int hi = workerStart(w);
            counts3[r] = 3 * (hi - lo);
            displs3[r] = 3 * lo;
        }
    }
    const int rankLo = displs3[rank] / 3;
    const int rankCnt = counts3[rank] / 3;
    for (int t = 0; t < nd; ++t) {
        devs[t].i0 = workerStart(myFirstWorker + t);
        devs[t].cnt = workerStart(myFirstWorker + t + 1) - devs[t].i0;
    }

    // --- Host buffers (pinned) ---
    double* hPos = nullptr;  // all positions, packed xyz
    double* hNew = nullptr;  // this rank's updated positions / velocities
    CUDA_CHECK(cudaMallocHost(&hPos, sizeof(double) * 3 * std::max(n, 1)));
    CUDA_CHECK(cudaMallocHost(&hNew, sizeof(double) * 3 * std::max(rankCnt, 1)));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        hPos[3 * i + 0] = bodies[i].pos.x;
        hPos[3 * i + 1] = bodies[i].pos.y;
        hPos[3 * i + 2] = bodies[i].pos.z;
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < rankCnt; ++i) {
        hNew[3 * i + 0] = bodies[rankLo + i].vel.x;
        hNew[3 * i + 1] = bodies[rankLo + i].vel.y;
        hNew[3 * i + 2] = bodies[rankLo + i].vel.z;
    }

    // --- Device setup (one OpenMP thread per GPU) ---
#pragma omp parallel num_threads(nd)
    {
        Device& d = devs[omp_get_thread_num()];
        CUDA_CHECK(cudaSetDevice(d.id));
        CUDA_CHECK(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&d.dPos, sizeof(double) * 3 * std::max(n, 1)));
        CUDA_CHECK(cudaMalloc(&d.dPosOut, sizeof(double) * 3 * std::max(d.cnt, 1)));
        CUDA_CHECK(cudaMalloc(&d.dVel, sizeof(double) * 3 * std::max(d.cnt, 1)));
        if (d.cnt > 0)
            CUDA_CHECK(cudaMemcpy(d.dVel, hNew + 3 * (d.i0 - rankLo), sizeof(double) * 3 * d.cnt,
                                  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.dPos, hPos, sizeof(double) * 3 * n, cudaMemcpyHostToDevice));
        // Warm-up/autotune (excluded from timing); state is not modified.
        d.block = tuneBlock(d, n);
        CUDA_CHECK(cudaStreamSynchronize(d.stream));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(nd)
    {
        Device& d = devs[omp_get_thread_num()];
        CUDA_CHECK(cudaSetDevice(d.id));
        for (int step = 0; step < numSteps; ++step) {
            if (d.cnt > 0) {
                CUDA_CHECK(cudaMemcpyAsync(d.dPos, hPos, sizeof(double) * 3 * n,
                                           cudaMemcpyHostToDevice, d.stream));
                launchStep(d.block, d.dPos, n, d.i0, d.cnt, d.dVel, d.dPosOut, d.stream);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(hNew + 3 * (d.i0 - rankLo), d.dPosOut,
                                           sizeof(double) * 3 * d.cnt, cudaMemcpyDeviceToHost,
                                           d.stream));
                CUDA_CHECK(cudaStreamSynchronize(d.stream));
            }
#pragma omp barrier
#pragma omp master
            MPI_Allgatherv(hNew, counts3[rank], MPI_DOUBLE, hPos, counts3.data(), displs3.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
#pragma omp barrier
        }
        // Fetch final velocities of owned bodies
        if (d.cnt > 0)
            CUDA_CHECK(cudaMemcpy(hNew + 3 * (d.i0 - rankLo), d.dVel, sizeof(double) * 3 * d.cnt,
                                  cudaMemcpyDeviceToHost));
    }
    std::vector<double> allVel(3 * (size_t)std::max(n, 1));
    MPI_Allgatherv(hNew, counts3[rank], MPI_DOUBLE, allVel.data(), counts3.data(), displs3.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) printf("Simulation time: %ld ms\n", duration.count());

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        bodies[i].pos = Vec3(hPos[3 * i + 0], hPos[3 * i + 1], hPos[3 * i + 2]);
        bodies[i].vel = Vec3(allVel[3 * i + 0], allVel[3 * i + 1], allVel[3 * i + 2]);
    }

    for (auto& d : devs) {
        CUDA_CHECK(cudaSetDevice(d.id));
        CUDA_CHECK(cudaFree(d.dPos));
        CUDA_CHECK(cudaFree(d.dPosOut));
        CUDA_CHECK(cudaFree(d.dVel));
        CUDA_CHECK(cudaStreamDestroy(d.stream));
    }
    CUDA_CHECK(cudaFreeHost(hPos));
    CUDA_CHECK(cudaFreeHost(hNew));

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
    int exitCode = 0;
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");

        int ok = (rank == 0) ? (validateSimulation(bodies) ? 1 : 0) : 0;
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (ok) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies, rank, nranks);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    fflush(stdout);
    MPI_Finalize();
    return exitCode;
}
