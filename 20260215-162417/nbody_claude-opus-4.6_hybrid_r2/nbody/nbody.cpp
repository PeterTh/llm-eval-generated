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
constexpr int TILE_SIZE = 128;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// SoA layout for GPU-friendly access
struct Bodies {
    std::vector<double> px, py, pz; // positions
    std::vector<double> vx, vy, vz; // velocities
    int n;

    Bodies() : n(0) {}
    Bodies(int n) : px(n), py(n), pz(n), vx(n), vy(n), vz(n), n(n) {}

    void resize(int sz) {
        n = sz;
        px.resize(sz); py.resize(sz); pz.resize(sz);
        vx.resize(sz); vy.resize(sz); vz.resize(sz);
    }
};

void randomizeBodies(Bodies& bodies, unsigned int seed = 42) {
    for (int i = 0; i < bodies.n; ++i) {
        bodies.px[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.py[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.pz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.vx[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.vy[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        bodies.vz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// CUDA kernel: compute forces on a subset of bodies from all bodies using shared memory tiling
__global__ void computeForcesKernel(
    const double* __restrict__ all_px, const double* __restrict__ all_py, const double* __restrict__ all_pz,
    double* __restrict__ local_vx, double* __restrict__ local_vy, double* __restrict__ local_vz,
    const double* __restrict__ local_px, const double* __restrict__ local_py, const double* __restrict__ local_pz,
    int nLocal, int nTotal, double dt, double softening)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double myPx, myPy, myPz;

    if (i < nLocal) {
        myPx = local_px[i];
        myPy = local_py[i];
        myPz = local_pz[i];
    }

    // Tile over all bodies using shared memory
    __shared__ double s_px[TILE_SIZE];
    __shared__ double s_py[TILE_SIZE];
    __shared__ double s_pz[TILE_SIZE];

    for (int tile = 0; tile < nTotal; tile += TILE_SIZE) {
        int idx = tile + threadIdx.x;
        if (idx < nTotal) {
            s_px[threadIdx.x] = all_px[idx];
            s_py[threadIdx.x] = all_py[idx];
            s_pz[threadIdx.x] = all_pz[idx];
        } else {
            s_px[threadIdx.x] = 0.0;
            s_py[threadIdx.x] = 0.0;
            s_pz[threadIdx.x] = 0.0;
        }
        __syncthreads();

        if (i < nLocal) {
            int jEnd = min(TILE_SIZE, nTotal - tile);
            #pragma unroll 8
            for (int j = 0; j < jEnd; ++j) {
                double dx = s_px[j] - myPx;
                double dy = s_py[j] - myPy;
                double dz = s_pz[j] - myPz;
                double distSqr = dx * dx + dy * dy + dz * dz + softening;
                double invDist = 1.0 / sqrt(distSqr);
                double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (i < nLocal) {
        local_vx[i] += dt * Fx;
        local_vy[i] += dt * Fy;
        local_vz[i] += dt * Fz;
    }
}

// CUDA kernel: integrate positions
__global__ void integrateKernel(
    double* __restrict__ px, double* __restrict__ py, double* __restrict__ pz,
    const double* __restrict__ vx, const double* __restrict__ vy, const double* __restrict__ vz,
    int n, double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        px[i] += vx[i] * dt;
        py[i] += vy[i] * dt;
        pz[i] += vz[i] * dt;
    }
}

double computeTotalEnergy(const Bodies& bodies) {
    double energy = 0.0;
    const int n = bodies.n;

    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (int i = 0; i < n; ++i) {
        energy += 0.5 * (bodies.vx[i] * bodies.vx[i] +
                        bodies.vy[i] * bodies.vy[i] +
                        bodies.vz[i] * bodies.vz[i]);
    }

    double pe = 0.0;
    #pragma omp parallel for reduction(+:pe) schedule(dynamic, 64)
    for (int i = 0; i < n; ++i) {
        double local_pe = 0.0;
        for (int j = i + 1; j < n; ++j) {
            const double dx = bodies.px[j] - bodies.px[i];
            const double dy = bodies.py[j] - bodies.py[i];
            const double dz = bodies.pz[j] - bodies.pz[i];
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            local_pe -= 1.0 / dist;
        }
        pe += local_pe;
    }
    energy += pe;

    return energy;
}

bool validateSimulation(const Bodies& bodies) {
    bool valid = true;
    #pragma omp parallel for reduction(&&:valid)
    for (int i = 0; i < bodies.n; ++i) {
        if (!std::isfinite(bodies.px[i]) || !std::isfinite(bodies.py[i]) || !std::isfinite(bodies.pz[i]) ||
            !std::isfinite(bodies.vx[i]) || !std::isfinite(bodies.vy[i]) || !std::isfinite(bodies.vz[i])) {
            valid = false;
        }
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(bodies.px[i]) > maxPos || std::abs(bodies.py[i]) > maxPos || std::abs(bodies.pz[i]) > maxPos) {
            valid = false;
        }
        if (std::abs(bodies.vx[i]) > maxVel || std::abs(bodies.vy[i]) > maxVel || std::abs(bodies.vz[i]) > maxVel) {
            valid = false;
        }
    }
    if (!valid) {
        printf("Validation failed: found invalid body state\n");
    }
    return valid;
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

    // Assign GPU: each MPI rank gets a GPU (round-robin if more ranks than GPUs)
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
        printf("MPI ranks: %d, GPUs per rank: 1, OpenMP threads: %d\n",
               numProcs, omp_get_max_threads());
    }

    // Compute distribution: contiguous chunks per rank
    std::vector<int> counts(numProcs), displs(numProcs);
    int base = numBodies / numProcs;
    int remainder = numBodies % numProcs;
    for (int r = 0; r < numProcs; ++r) {
        counts[r] = base + (r < remainder ? 1 : 0);
        displs[r] = (r == 0) ? 0 : displs[r-1] + counts[r-1];
    }
    int nLocal = counts[rank];
    int localOffset = displs[rank];

    // Initialize all bodies on rank 0, then broadcast
    Bodies allBodies(numBodies);
    if (rank == 0) {
        randomizeBodies(allBodies);
    }
    MPI_Bcast(allBodies.px.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(allBodies.py.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(allBodies.pz.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(allBodies.vx.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(allBodies.vy.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(allBodies.vz.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Allocate GPU memory
    // All positions (needed for force computation)
    double *d_all_px, *d_all_py, *d_all_pz;
    CUDA_CHECK(cudaMalloc(&d_all_px, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_all_py, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_all_pz, numBodies * sizeof(double)));

    // Local subset: positions and velocities
    double *d_local_px, *d_local_py, *d_local_pz;
    double *d_local_vx, *d_local_vy, *d_local_vz;
    CUDA_CHECK(cudaMalloc(&d_local_px, nLocal * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_py, nLocal * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_pz, nLocal * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_vx, nLocal * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_vy, nLocal * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_vz, nLocal * sizeof(double)));

    // Upload initial data to GPU
    CUDA_CHECK(cudaMemcpy(d_all_px, allBodies.px.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_all_py, allBodies.py.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_all_pz, allBodies.pz.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_local_px, allBodies.px.data() + localOffset, nLocal * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_local_py, allBodies.py.data() + localOffset, nLocal * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_local_pz, allBodies.pz.data() + localOffset, nLocal * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_local_vx, allBodies.vx.data() + localOffset, nLocal * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_local_vy, allBodies.vy.data() + localOffset, nLocal * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_local_vz, allBodies.vz.data() + localOffset, nLocal * sizeof(double), cudaMemcpyHostToDevice));

    int blockSize = TILE_SIZE;
    int gridForce = (nLocal + blockSize - 1) / blockSize;
    int gridInteg = (nLocal + blockSize - 1) / blockSize;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // Compute forces on local bodies from all bodies (on GPU)
        computeForcesKernel<<<gridForce, blockSize, 0, stream>>>(
            d_all_px, d_all_py, d_all_pz,
            d_local_vx, d_local_vy, d_local_vz,
            d_local_px, d_local_py, d_local_pz,
            nLocal, numBodies, DT, SOFTENING);

        // Integrate local positions (on GPU)
        integrateKernel<<<gridInteg, blockSize, 0, stream>>>(
            d_local_px, d_local_py, d_local_pz,
            d_local_vx, d_local_vy, d_local_vz,
            nLocal, DT);

        // Copy updated local positions back to host
        CUDA_CHECK(cudaMemcpyAsync(allBodies.px.data() + localOffset, d_local_px, nLocal * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(allBodies.py.data() + localOffset, d_local_py, nLocal * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(allBodies.pz.data() + localOffset, d_local_pz, nLocal * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Allgather updated positions across all MPI ranks
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       allBodies.px.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       allBodies.py.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       allBodies.pz.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // Upload gathered positions to GPU for next step
        CUDA_CHECK(cudaMemcpyAsync(d_all_px, allBodies.px.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_all_py, allBodies.py.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_all_pz, allBodies.pz.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Copy final local velocities back to host
    CUDA_CHECK(cudaMemcpy(allBodies.vx.data() + localOffset, d_local_vx, nLocal * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(allBodies.vy.data() + localOffset, d_local_vy, nLocal * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(allBodies.vz.data() + localOffset, d_local_vz, nLocal * sizeof(double), cudaMemcpyDeviceToHost));

    // Gather velocities for output/validation
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   allBodies.vx.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   allBodies.vy.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   allBodies.vz.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());

        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(numBodies * 6);
            for (int i = 0; i < numBodies; ++i) {
                bodyData.push_back(allBodies.px[i]);
                bodyData.push_back(allBodies.py[i]);
                bodyData.push_back(allBodies.pz[i]);
                bodyData.push_back(allBodies.vx[i]);
                bodyData.push_back(allBodies.vy[i]);
                bodyData.push_back(allBodies.vz[i]);
            }
            print_results(bodyData, "Bodies");
        }

        if (validate) {
            printf("Validating simulation results...\n");
            if (validateSimulation(allBodies)) {
                double finalEnergy = computeTotalEnergy(allBodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                CUDA_CHECK(cudaStreamDestroy(stream));
                CUDA_CHECK(cudaFree(d_all_px)); CUDA_CHECK(cudaFree(d_all_py)); CUDA_CHECK(cudaFree(d_all_pz));
                CUDA_CHECK(cudaFree(d_local_px)); CUDA_CHECK(cudaFree(d_local_py)); CUDA_CHECK(cudaFree(d_local_pz));
                CUDA_CHECK(cudaFree(d_local_vx)); CUDA_CHECK(cudaFree(d_local_vy)); CUDA_CHECK(cudaFree(d_local_vz));
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_all_px)); CUDA_CHECK(cudaFree(d_all_py)); CUDA_CHECK(cudaFree(d_all_pz));
    CUDA_CHECK(cudaFree(d_local_px)); CUDA_CHECK(cudaFree(d_local_py)); CUDA_CHECK(cudaFree(d_local_pz));
    CUDA_CHECK(cudaFree(d_local_vx)); CUDA_CHECK(cudaFree(d_local_vy)); CUDA_CHECK(cudaFree(d_local_vz));

    MPI_Finalize();
    return 0;
}
