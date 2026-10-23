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

constexpr int BLOCK_SIZE = 256;

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

// Force computation for a contiguous range of bodies.  Each body still
// visits every other body in the original ascending order, so the floating
// point summation is identical to the reference implementation.
__global__ void computeForcesKernel(const double* __restrict__ px,
                                    const double* __restrict__ py,
                                    const double* __restrict__ pz,
                                    double* __restrict__ vx,
                                    double* __restrict__ vy,
                                    double* __restrict__ vz,
                                    const int n,
                                    const int start,
                                    const int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const int gi = start + i;
    const double pix = px[gi];
    const double piy = py[gi];
    const double piz = pz[gi];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int j = 0; j < n; ++j) {
        const double dx = px[j] - pix;
        const double dy = py[j] - piy;
        const double dz = pz[j] - piz;
        const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        const double invDist = 1.0 / sqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    vx[i] += DT * Fx;
    vy[i] += DT * Fy;
    vz[i] += DT * Fz;
}

// Positions are advanced only after every force has been accumulated.
__global__ void integrateKernel(double* __restrict__ px,
                                double* __restrict__ py,
                                double* __restrict__ pz,
                                const double* __restrict__ vx,
                                const double* __restrict__ vy,
                                const double* __restrict__ vz,
                                const int start,
                                const int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const int gi = start + i;
        px[gi] += vx[i] * DT;
        py[gi] += vy[i] * DT;
        pz[gi] += vz[i] * DT;
    }
}

// Per-OpenMP-thread CUDA resources.  A thread owns one GPU and one contiguous
// range of the bodies assigned to this MPI rank.
struct GpuContext {
    int device = 0;
    double* dpx = nullptr;
    double* dpy = nullptr;
    double* dpz = nullptr;
    double* dvx = nullptr;
    double* dvy = nullptr;
    double* dvz = nullptr;
    cudaStream_t stream = nullptr;
};

static void checkCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    
    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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
    if (numSteps < 0) numSteps = 0;
    const int n = numBodies;

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine how many GPUs this MPI rank should drive.  Ranks sharing a
    // node are identified with a shared-memory communicator so that devices
    // are partitioned instead of being oversubscribed.
    int nodeRank = 0, nodeSize = 1;
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    if (nodeComm != MPI_COMM_NULL) {
        MPI_Comm_rank(nodeComm, &nodeRank);
        MPI_Comm_size(nodeComm, &nodeSize);
    }

    int deviceCount = 0;
    cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available: %s\n", rank,
                deviceStatus == cudaSuccess ? "device count is zero" : cudaGetErrorString(deviceStatus));
        if (nodeComm != MPI_COMM_NULL) MPI_Comm_free(&nodeComm);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int gpusPerRank = deviceCount / (nodeSize > 0 ? nodeSize : 1);
    if (gpusPerRank < 1) gpusPerRank = 1;
    int numThreads = gpusPerRank;
    std::vector<int> devices(numThreads);
    for (int t = 0; t < numThreads; ++t) {
        devices[t] = (nodeRank * gpusPerRank + t) % deviceCount;
    }

    // Global body range owned by this rank.
    const int localStart = static_cast<int>((static_cast<long long>(rank) * n) / worldSize);
    const int localEnd = static_cast<int>((static_cast<long long>(rank + 1) * n) / worldSize);
    const int localCount = localEnd - localStart;

    std::vector<int> bodyCounts(worldSize);
    std::vector<int> bodyDispls(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const int s = static_cast<int>((static_cast<long long>(r) * n) / worldSize);
        const int e = static_cast<int>((static_cast<long long>(r + 1) * n) / worldSize);
        bodyCounts[r] = e - s;
        bodyDispls[r] = s;
    }
    std::vector<int> posCounts(worldSize), posDispls(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        posCounts[r] = 3 * bodyCounts[r];
        posDispls[r] = 3 * bodyDispls[r];
    }

    // Body state on the host.  Every rank keeps a full copy of the positions
    // (needed by every force evaluation); velocities only need to be correct
    // for the local range during the simulation.
    std::vector<double> px(n), py(n), pz(n), vx(n), vy(n), vz(n);

    if (n > 0) {
        if (rank == 0) {
            std::vector<Body> initial(n);
            randomizeBodies(initial);
            for (int i = 0; i < n; ++i) {
                px[i] = initial[i].pos.x;
                py[i] = initial[i].pos.y;
                pz[i] = initial[i].pos.z;
                vx[i] = initial[i].vel.x;
                vy[i] = initial[i].vel.y;
                vz[i] = initial[i].vel.z;
            }
        }
        MPI_Bcast(px.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(py.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(pz.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(vx.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(vy.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(vz.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Split this rank's range across its OpenMP worker threads.
    std::vector<int> threadStart(numThreads), threadEnd(numThreads);
    for (int t = 0; t < numThreads; ++t) {
        threadStart[t] = localStart + static_cast<int>((static_cast<long long>(t) * localCount) / numThreads);
        threadEnd[t] = localStart + static_cast<int>((static_cast<long long>(t + 1) * localCount) / numThreads);
    }

    std::vector<GpuContext> contexts(numThreads);

    // Allocate persistent CUDA buffers before timing starts.
    omp_set_dynamic(0);
#pragma omp parallel num_threads(numThreads)
    {
        const int t = omp_get_thread_num();
        GpuContext& ctx = contexts[t];
        ctx.device = devices[t];
        checkCuda(cudaSetDevice(ctx.device), "cudaSetDevice");
        checkCuda(cudaStreamCreate(&ctx.stream), "cudaStreamCreate");
        if (n > 0) {
            checkCuda(cudaMalloc(&ctx.dpx, sizeof(double) * n), "cudaMalloc dpx");
            checkCuda(cudaMalloc(&ctx.dpy, sizeof(double) * n), "cudaMalloc dpy");
            checkCuda(cudaMalloc(&ctx.dpz, sizeof(double) * n), "cudaMalloc dpz");
        }
        const int cnt = threadEnd[t] - threadStart[t];
        if (cnt > 0) {
            checkCuda(cudaMalloc(&ctx.dvx, sizeof(double) * cnt), "cudaMalloc dvx");
            checkCuda(cudaMalloc(&ctx.dvy, sizeof(double) * cnt), "cudaMalloc dvy");
            checkCuda(cudaMalloc(&ctx.dvz, sizeof(double) * cnt), "cudaMalloc dvz");
        }
    }

    std::vector<double> posSend;
    std::vector<double> posRecv;
    if (n > 0) {
        posSend.resize(3 * localCount > 0 ? 3 * localCount : 1);
        posRecv.resize(3 * n);
    }

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        if (n > 0) {
#pragma omp parallel num_threads(numThreads)
            {
                const int t = omp_get_thread_num();
                GpuContext& ctx = contexts[t];
                checkCuda(cudaSetDevice(ctx.device), "cudaSetDevice");

                const int s = threadStart[t];
                const int cnt = threadEnd[t] - s;

                // Full positions are required by every GPU worker.
                checkCuda(cudaMemcpyAsync(ctx.dpx, px.data(), sizeof(double) * n,
                                          cudaMemcpyHostToDevice, ctx.stream), "copy px");
                checkCuda(cudaMemcpyAsync(ctx.dpy, py.data(), sizeof(double) * n,
                                          cudaMemcpyHostToDevice, ctx.stream), "copy py");
                checkCuda(cudaMemcpyAsync(ctx.dpz, pz.data(), sizeof(double) * n,
                                          cudaMemcpyHostToDevice, ctx.stream), "copy pz");

                if (cnt > 0) {
                    checkCuda(cudaMemcpyAsync(ctx.dvx, vx.data() + s, sizeof(double) * cnt,
                                              cudaMemcpyHostToDevice, ctx.stream), "copy vx");
                    checkCuda(cudaMemcpyAsync(ctx.dvy, vy.data() + s, sizeof(double) * cnt,
                                              cudaMemcpyHostToDevice, ctx.stream), "copy vy");
                    checkCuda(cudaMemcpyAsync(ctx.dvz, vz.data() + s, sizeof(double) * cnt,
                                              cudaMemcpyHostToDevice, ctx.stream), "copy vz");
                }

                // Make sure every worker has finished reading the shared host
                // position arrays before any worker writes updated positions
                // back into them.
                checkCuda(cudaStreamSynchronize(ctx.stream), "upload sync");
#pragma omp barrier

                if (cnt > 0) {
                    const int blocks = (cnt + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    computeForcesKernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                        ctx.dpx, ctx.dpy, ctx.dpz, ctx.dvx, ctx.dvy, ctx.dvz,
                        n, s, cnt);
                    checkCuda(cudaGetLastError(), "computeForcesKernel");
                    integrateKernel<<<blocks, BLOCK_SIZE, 0, ctx.stream>>>(
                        ctx.dpx, ctx.dpy, ctx.dpz, ctx.dvx, ctx.dvy, ctx.dvz,
                        s, cnt);
                    checkCuda(cudaGetLastError(), "integrateKernel");

                    checkCuda(cudaMemcpyAsync(vx.data() + s, ctx.dvx, sizeof(double) * cnt,
                                              cudaMemcpyDeviceToHost, ctx.stream), "copy back vx");
                    checkCuda(cudaMemcpyAsync(vy.data() + s, ctx.dvy, sizeof(double) * cnt,
                                              cudaMemcpyDeviceToHost, ctx.stream), "copy back vy");
                    checkCuda(cudaMemcpyAsync(vz.data() + s, ctx.dvz, sizeof(double) * cnt,
                                              cudaMemcpyDeviceToHost, ctx.stream), "copy back vz");
                    checkCuda(cudaMemcpyAsync(px.data() + s, ctx.dpx + s, sizeof(double) * cnt,
                                              cudaMemcpyDeviceToHost, ctx.stream), "copy back px");
                    checkCuda(cudaMemcpyAsync(py.data() + s, ctx.dpy + s, sizeof(double) * cnt,
                                              cudaMemcpyDeviceToHost, ctx.stream), "copy back py");
                    checkCuda(cudaMemcpyAsync(pz.data() + s, ctx.dpz + s, sizeof(double) * cnt,
                                              cudaMemcpyDeviceToHost, ctx.stream), "copy back pz");
                }
                checkCuda(cudaStreamSynchronize(ctx.stream), "stream sync");
            }

            // Publish the updated local positions to every rank.  Forces at
            // the next step need all positions, but velocities stay local.
            memcpy(posSend.data(), px.data() + localStart, sizeof(double) * localCount);
            memcpy(posSend.data() + localCount, py.data() + localStart, sizeof(double) * localCount);
            memcpy(posSend.data() + 2 * localCount, pz.data() + localStart, sizeof(double) * localCount);
            MPI_Allgatherv(posSend.data(), 3 * localCount, MPI_DOUBLE,
                           posRecv.data(), posCounts.data(), posDispls.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
            for (int r = 0; r < worldSize; ++r) {
                const int cnt = bodyCounts[r];
                const int off = bodyDispls[r];
                const double* base = posRecv.data() + 3 * off;
                memcpy(px.data() + off, base, sizeof(double) * cnt);
                memcpy(py.data() + off, base + cnt, sizeof(double) * cnt);
                memcpy(pz.data() + off, base + 2 * cnt, sizeof(double) * cnt);
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long maxDuration = 0;
    MPI_Reduce(&duration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", maxDuration);
    }

    // Bring all velocities together for result output and validation.
    std::vector<double> gatheredVel;
    if (n > 0 && rank == 0 && (printResults || validate)) {
        gatheredVel.resize(3 * n);
    }
    if (n > 0 && (printResults || validate)) {
        std::vector<double> velSend(3 * localCount > 0 ? 3 * localCount : 1);
        memcpy(velSend.data(), vx.data() + localStart, sizeof(double) * localCount);
        memcpy(velSend.data() + localCount, vy.data() + localStart, sizeof(double) * localCount);
        memcpy(velSend.data() + 2 * localCount, vz.data() + localStart, sizeof(double) * localCount);
        MPI_Gatherv(velSend.data(), 3 * localCount, MPI_DOUBLE,
                    gatheredVel.data(), posCounts.data(), posDispls.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (rank == 0) {
        if (printResults || validate) {
            for (int r = 0; r < worldSize; ++r) {
                const int cnt = bodyCounts[r];
                const int off = bodyDispls[r];
                const double* base = gatheredVel.data() + 3 * off;
                memcpy(vx.data() + off, base, sizeof(double) * cnt);
                memcpy(vy.data() + off, base + cnt, sizeof(double) * cnt);
                memcpy(vz.data() + off, base + 2 * cnt, sizeof(double) * cnt);
            }
        }

        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData;
            bodyData.reserve(static_cast<size_t>(n) * 6);
            for (int i = 0; i < n; ++i) {
                bodyData.push_back(px[i]);
                bodyData.push_back(py[i]);
                bodyData.push_back(pz[i]);
                bodyData.push_back(vx[i]);
                bodyData.push_back(vy[i]);
                bodyData.push_back(vz[i]);
            }
            print_results(bodyData, "Bodies");
        }
        
        // Validation: check that simulation produces finite, reasonable values
        if (validate) {
            printf("Validating simulation results...\n");

            std::vector<Body> finalBodies(n);
            for (int i = 0; i < n; ++i) {
                finalBodies[i].pos = Vec3(px[i], py[i], pz[i]);
                finalBodies[i].vel = Vec3(vx[i], vy[i], vz[i]);
            }

            if (validateSimulation(finalBodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(finalBodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    // Release CUDA resources.
#pragma omp parallel num_threads(numThreads)
    {
        const int t = omp_get_thread_num();
        GpuContext& ctx = contexts[t];
        cudaSetDevice(ctx.device);
        if (ctx.dpx) cudaFree(ctx.dpx);
        if (ctx.dpy) cudaFree(ctx.dpy);
        if (ctx.dpz) cudaFree(ctx.dpz);
        if (ctx.dvx) cudaFree(ctx.dvx);
        if (ctx.dvy) cudaFree(ctx.dvy);
        if (ctx.dvz) cudaFree(ctx.dvz);
        if (ctx.stream) cudaStreamDestroy(ctx.stream);
    }

    if (nodeComm != MPI_COMM_NULL) MPI_Comm_free(&nodeComm);

    int globalExitCode = 0;
    MPI_Allreduce(&exitCode, &globalExitCode, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return globalExitCode;
}
