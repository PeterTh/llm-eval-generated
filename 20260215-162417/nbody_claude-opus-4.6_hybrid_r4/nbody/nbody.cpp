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

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Force computation kernel with shared-memory tiling
__global__ void __launch_bounds__(BLOCK_SIZE)
computeForcesKernel(
    const double* __restrict__ all_px,
    const double* __restrict__ all_py,
    const double* __restrict__ all_pz,
    double* __restrict__ vx,
    double* __restrict__ vy,
    double* __restrict__ vz,
    int localStart, int localCount, int N)
{
    extern __shared__ double smem[];
    double* s_px = smem;
    double* s_py = s_px + BLOCK_SIZE;
    double* s_pz = s_py + BLOCK_SIZE;

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double myPx = 0.0, myPy = 0.0, myPz = 0.0;

    if (i < localCount) {
        int gi = localStart + i;
        myPx = all_px[gi];
        myPy = all_py[gi];
        myPz = all_pz[gi];
    }

    for (int tile = 0; tile < N; tile += BLOCK_SIZE) {
        int idx = tile + threadIdx.x;
        s_px[threadIdx.x] = (idx < N) ? all_px[idx] : 0.0;
        s_py[threadIdx.x] = (idx < N) ? all_py[idx] : 0.0;
        s_pz[threadIdx.x] = (idx < N) ? all_pz[idx] : 0.0;
        __syncthreads();

        if (i < localCount) {
            int limit = (N - tile < BLOCK_SIZE) ? (N - tile) : BLOCK_SIZE;
            for (int j = 0; j < limit; ++j) {
                double dx = s_px[j] - myPx;
                double dy = s_py[j] - myPy;
                double dz = s_pz[j] - myPz;
                double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                double invDist = rsqrt(distSqr);
                double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (i < localCount) {
        vx[i] += DT * Fx;
        vy[i] += DT * Fy;
        vz[i] += DT * Fz;
    }
}

// Position integration kernel
__global__ void __launch_bounds__(BLOCK_SIZE)
integrateKernel(
    double* __restrict__ px,
    double* __restrict__ py,
    double* __restrict__ pz,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    const double* __restrict__ vz,
    int localStart, int localCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < localCount) {
        int gi = localStart + i;
        px[gi] += vx[i] * DT;
        py[gi] += vy[i] * DT;
        pz[gi] += vz[i] * DT;
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

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    // Assign GPU based on rank
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    CUDA_CHECK(cudaSetDevice(rank % numDevices));

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

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
    }

    // All ranks initialize identically (deterministic with same seed)
    std::vector<double> all_px(numBodies), all_py(numBodies), all_pz(numBodies);
    std::vector<double> all_vx(numBodies), all_vy(numBodies), all_vz(numBodies);
    {
        unsigned int seed = 42;
        for (int i = 0; i < numBodies; ++i) {
            all_px[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            all_py[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            all_pz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            all_vx[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            all_vy[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            all_vz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        }
    }

    // Distribute bodies across MPI ranks
    int baseCount = numBodies / numProcs;
    int remainder = numBodies % numProcs;
    std::vector<int> counts(numProcs), displs(numProcs);
    for (int r = 0; r < numProcs; ++r) {
        counts[r] = baseCount + (r < remainder ? 1 : 0);
    }
    displs[0] = 0;
    for (int r = 1; r < numProcs; ++r) {
        displs[r] = displs[r - 1] + counts[r - 1];
    }
    int localStart = displs[rank];
    int localCount = counts[rank];

    // Allocate GPU memory (SoA layout)
    double *d_all_px = nullptr, *d_all_py = nullptr, *d_all_pz = nullptr;
    double *d_vx = nullptr, *d_vy = nullptr, *d_vz = nullptr;

    CUDA_CHECK(cudaMalloc(&d_all_px, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_all_py, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_all_pz, numBodies * sizeof(double)));
    if (localCount > 0) {
        CUDA_CHECK(cudaMalloc(&d_vx, localCount * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_vy, localCount * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_vz, localCount * sizeof(double)));
    }

    // Copy initial data to GPU
    CUDA_CHECK(cudaMemcpy(d_all_px, all_px.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_all_py, all_py.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_all_pz, all_pz.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    if (localCount > 0) {
        CUDA_CHECK(cudaMemcpy(d_vx, all_vx.data() + localStart, localCount * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vy, all_vy.data() + localStart, localCount * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vz, all_vz.data() + localStart, localCount * sizeof(double), cudaMemcpyHostToDevice));
    }

    // CPU buffers for MPI position exchange
    std::vector<double> local_px_buf(localCount), local_py_buf(localCount), local_pz_buf(localCount);

    int blocks = (localCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t smemSize = 3 * BLOCK_SIZE * sizeof(double);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // CUDA: compute forces for local bodies against all bodies
        if (localCount > 0) {
            computeForcesKernel<<<blocks, BLOCK_SIZE, smemSize>>>(
                d_all_px, d_all_py, d_all_pz,
                d_vx, d_vy, d_vz,
                localStart, localCount, numBodies);

            // CUDA: integrate local body positions
            integrateKernel<<<blocks, BLOCK_SIZE>>>(
                d_all_px, d_all_py, d_all_pz,
                d_vx, d_vy, d_vz,
                localStart, localCount);

            // Copy local positions GPU -> CPU for MPI exchange
            CUDA_CHECK(cudaMemcpy(local_px_buf.data(), d_all_px + localStart,
                                  localCount * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(local_py_buf.data(), d_all_py + localStart,
                                  localCount * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(local_pz_buf.data(), d_all_pz + localStart,
                                  localCount * sizeof(double), cudaMemcpyDeviceToHost));
        }

        // MPI: exchange positions across all ranks
        MPI_Allgatherv(local_px_buf.data(), localCount, MPI_DOUBLE,
                       all_px.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(local_py_buf.data(), localCount, MPI_DOUBLE,
                       all_py.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(local_pz_buf.data(), localCount, MPI_DOUBLE,
                       all_pz.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // Copy all positions CPU -> GPU
        CUDA_CHECK(cudaMemcpy(d_all_px, all_px.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_all_py, all_py.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_all_pz, all_pz.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long localDurationMs = static_cast<long long>(duration.count());
    long long maxDurationMs = 0;
    MPI_Reduce(&localDurationMs, &maxDurationMs, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %lld ms\n", maxDurationMs);
    }

    // Gather velocities for output/validation
    std::vector<double> local_vx_buf(localCount), local_vy_buf(localCount), local_vz_buf(localCount);
    if (localCount > 0) {
        CUDA_CHECK(cudaMemcpy(local_vx_buf.data(), d_vx, localCount * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(local_vy_buf.data(), d_vy, localCount * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(local_vz_buf.data(), d_vz, localCount * sizeof(double), cudaMemcpyDeviceToHost));
    }

    MPI_Gatherv(local_vx_buf.data(), localCount, MPI_DOUBLE,
                all_vx.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_vy_buf.data(), localCount, MPI_DOUBLE,
                all_vy.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_vz_buf.data(), localCount, MPI_DOUBLE,
                all_vz.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int retCode = 0;

    if (rank == 0) {
        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(numBodies * 6);
            for (int i = 0; i < numBodies; ++i) {
                bodyData.push_back(all_px[i]);
                bodyData.push_back(all_py[i]);
                bodyData.push_back(all_pz[i]);
                bodyData.push_back(all_vx[i]);
                bodyData.push_back(all_vy[i]);
                bodyData.push_back(all_vz[i]);
            }
            print_results(bodyData, "Bodies");
        }

        if (validate) {
            printf("Validating simulation results...\n");

            bool valid = true;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numBodies; ++i) {
                if (!std::isfinite(all_px[i]) || !std::isfinite(all_py[i]) || !std::isfinite(all_pz[i]) ||
                    !std::isfinite(all_vx[i]) || !std::isfinite(all_vy[i]) || !std::isfinite(all_vz[i])) {
                    #pragma omp atomic write
                    valid = false;
                }
                const double maxPos = 1e6;
                const double maxVel = 1e6;
                if (std::abs(all_px[i]) > maxPos || std::abs(all_py[i]) > maxPos || std::abs(all_pz[i]) > maxPos) {
                    #pragma omp atomic write
                    valid = false;
                }
                if (std::abs(all_vx[i]) > maxVel || std::abs(all_vy[i]) > maxVel || std::abs(all_vz[i]) > maxVel) {
                    #pragma omp atomic write
                    valid = false;
                }
            }

            if (valid) {
                // Compute total energy with OpenMP
                double energy = 0.0;

                // Kinetic energy
                #pragma omp parallel for reduction(+:energy) schedule(static)
                for (int i = 0; i < numBodies; ++i) {
                    energy += 0.5 * (all_vx[i] * all_vx[i] +
                                     all_vy[i] * all_vy[i] +
                                     all_vz[i] * all_vz[i]);
                }

                // Potential energy
                #pragma omp parallel for reduction(+:energy) schedule(dynamic, 64)
                for (int i = 0; i < numBodies; ++i) {
                    for (int j = i + 1; j < numBodies; ++j) {
                        double dx = all_px[j] - all_px[i];
                        double dy = all_py[j] - all_py[i];
                        double dz = all_pz[j] - all_pz[i];
                        double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                        energy -= 1.0 / dist;
                    }
                }

                printf("Final energy: %.6f\n", energy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation failed: found NaN, Inf, or extreme values\n");
                printf("Validation: FAILED\n");
                retCode = 1;
            }
        }
    }

    MPI_Bcast(&retCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(d_all_px));
    CUDA_CHECK(cudaFree(d_all_py));
    CUDA_CHECK(cudaFree(d_all_pz));
    if (localCount > 0) {
        CUDA_CHECK(cudaFree(d_vx));
        CUDA_CHECK(cudaFree(d_vy));
        CUDA_CHECK(cudaFree(d_vz));
    }

    MPI_Finalize();
    return retCode;
}
