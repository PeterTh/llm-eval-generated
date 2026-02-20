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
constexpr int BLOCK_SIZE = 256;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Shared-memory tiled force computation on GPU
__global__ void computeForcesKernel(
    const double* __restrict__ px, const double* __restrict__ py, const double* __restrict__ pz,
    double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
    int n, int local_start, int local_count)
{
    extern __shared__ double shm[];
    double* spx = shm;
    double* spy = spx + blockDim.x;
    double* spz = spy + blockDim.x;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = local_start + idx;

    double pxi = 0.0, pyi = 0.0, pzi = 0.0;
    if (idx < local_count) {
        pxi = px[i]; pyi = py[i]; pzi = pz[i];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = 0; tile < n; tile += static_cast<int>(blockDim.x)) {
        int j = tile + threadIdx.x;
        spx[threadIdx.x] = (j < n) ? px[j] : 0.0;
        spy[threadIdx.x] = (j < n) ? py[j] : 0.0;
        spz[threadIdx.x] = (j < n) ? pz[j] : 0.0;
        __syncthreads();

        if (idx < local_count) {
            int tile_end = min(static_cast<int>(blockDim.x), n - tile);
            for (int k = 0; k < tile_end; ++k) {
                double dx = spx[k] - pxi;
                double dy = spy[k] - pyi;
                double dz = spz[k] - pzi;
                double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                double invDist = 1.0 / sqrt(distSqr);
                double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (idx < local_count) {
        vx[i] += DT * Fx;
        vy[i] += DT * Fy;
        vz[i] += DT * Fz;
    }
}

__global__ void integrateKernel(
    double* __restrict__ px, double* __restrict__ py, double* __restrict__ pz,
    const double* __restrict__ vx, const double* __restrict__ vy, const double* __restrict__ vz,
    int local_start, int local_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= local_count) return;
    int i = local_start + idx;
    px[i] += vx[i] * DT;
    py[i] += vy[i] * DT;
    pz[i] += vz[i] * DT;
}

// Initialize bodies in SoA layout with same rand_r sequence as original AoS code
void randomizeBodies(double* px, double* py, double* pz,
                     double* vx, double* vy, double* vz,
                     int n, unsigned int seed = 42) {
    for (int i = 0; i < n; ++i) {
        px[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        py[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vx[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vy[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

double computeTotalEnergy(const double* px, const double* py, const double* pz,
                          const double* vx, const double* vy, const double* vz, int n) {
    double energy = 0.0;

    // Kinetic energy (OpenMP parallelized)
    #pragma omp parallel for reduction(+:energy)
    for (int i = 0; i < n; ++i) {
        energy += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    }

    // Potential energy (OpenMP parallelized, dynamic schedule for load balance)
    #pragma omp parallel for reduction(+:energy) schedule(dynamic, 64)
    for (int i = 0; i < n; ++i) {
        double local_e = 0.0;
        for (int j = i + 1; j < n; ++j) {
            double dx = px[j] - px[i];
            double dy = py[j] - py[i];
            double dz = pz[j] - pz[i];
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            local_e -= 1.0 / dist;
        }
        energy += local_e;
    }

    return energy;
}

bool validateSimulation(const double* px, const double* py, const double* pz,
                        const double* vx, const double* vy, const double* vz, int n) {
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(px[i]) || !std::isfinite(py[i]) || !std::isfinite(pz[i]) ||
            !std::isfinite(vx[i]) || !std::isfinite(vy[i]) || !std::isfinite(vz[i])) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(px[i]) > maxPos || std::abs(py[i]) > maxPos || std::abs(pz[i]) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(vx[i]) > maxVel || std::abs(vy[i]) > maxVel || std::abs(vz[i]) > maxVel) {
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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Assign GPU based on local rank
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    CUDA_CHECK(cudaSetDevice(rank % num_devices));

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResultsFlag = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResultsFlag = true;
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

    const int N = numBodies;

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", N);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // SoA host arrays — all ranks hold all positions; velocities are distributed
    std::vector<double> h_px(N), h_py(N), h_pz(N);
    std::vector<double> h_vx(N), h_vy(N), h_vz(N);

    // Deterministic initialization identical on every rank
    randomizeBodies(h_px.data(), h_py.data(), h_pz.data(),
                    h_vx.data(), h_vy.data(), h_vz.data(), N);

    // MPI work distribution
    int local_count = N / size + (rank < N % size ? 1 : 0);
    int local_start = rank * (N / size) + std::min(rank, N % size);

    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = N / size + (r < N % size ? 1 : 0);
        displs[r] = r * (N / size) + std::min(r, N % size);
    }

    // Allocate device memory (full arrays for positions, only need local for velocities
    // but allocating full size simplifies indexing)
    double *d_px, *d_py, *d_pz, *d_vx, *d_vy, *d_vz;
    CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pz, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vx, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vy, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vz, N * sizeof(double)));

    // Copy initial state to GPU
    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pz, h_pz.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, h_vz.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    int numBlocks = (local_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shmSize = 3 * BLOCK_SIZE * sizeof(double);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // GPU: compute forces for this rank's bodies against all bodies
        if (local_count > 0) {
            computeForcesKernel<<<numBlocks, BLOCK_SIZE, shmSize>>>(
                d_px, d_py, d_pz, d_vx, d_vy, d_vz,
                N, local_start, local_count);

            // GPU: integrate positions for this rank's bodies
            integrateKernel<<<numBlocks, BLOCK_SIZE>>>(
                d_px, d_py, d_pz, d_vx, d_vy, d_vz,
                local_start, local_count);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy local updated positions back to host for MPI exchange
        if (local_count > 0) {
            CUDA_CHECK(cudaMemcpy(h_px.data() + local_start, d_px + local_start,
                                  local_count * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_py.data() + local_start, d_py + local_start,
                                  local_count * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_pz.data() + local_start, d_pz + local_start,
                                  local_count * sizeof(double), cudaMemcpyDeviceToHost));
        }

        // MPI: exchange positions among all ranks
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       h_px.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       h_py.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       h_pz.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // Copy gathered positions back to GPU for next step
        CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pz, h_pz.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // Retrieve local velocities from GPU
    if (local_count > 0) {
        CUDA_CHECK(cudaMemcpy(h_vx.data() + local_start, d_vx + local_start,
                              local_count * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vy.data() + local_start, d_vy + local_start,
                              local_count * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vz.data() + local_start, d_vz + local_start,
                              local_count * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // Gather all velocities to rank 0 for output / validation
    if (printResultsFlag || validate) {
        if (rank == 0) {
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       h_vx.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       h_vy.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       h_vz.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(h_vx.data() + local_start, local_count, MPI_DOUBLE,
                       nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(h_vy.data() + local_start, local_count, MPI_DOUBLE,
                       nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gatherv(h_vz.data() + local_start, local_count, MPI_DOUBLE,
                       nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    }

    int ret = 0;
    if (rank == 0) {
        if (printResultsFlag) {
            std::vector<double> bodyData;
            bodyData.reserve(N * 6);
            for (int i = 0; i < N; ++i) {
                bodyData.push_back(h_px[i]);
                bodyData.push_back(h_py[i]);
                bodyData.push_back(h_pz[i]);
                bodyData.push_back(h_vx[i]);
                bodyData.push_back(h_vy[i]);
                bodyData.push_back(h_vz[i]);
            }
            print_results(bodyData, "Bodies");
        }

        if (validate) {
            printf("Validating simulation results...\n");
            if (validateSimulation(h_px.data(), h_py.data(), h_pz.data(),
                                   h_vx.data(), h_vy.data(), h_vz.data(), N)) {
                double finalEnergy = computeTotalEnergy(h_px.data(), h_py.data(), h_pz.data(),
                                                        h_vx.data(), h_vy.data(), h_vz.data(), N);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
    }

    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py)); CUDA_CHECK(cudaFree(d_pz));
    CUDA_CHECK(cudaFree(d_vx)); CUDA_CHECK(cudaFree(d_vy)); CUDA_CHECK(cudaFree(d_vz));

    MPI_Finalize();
    return ret;
}
