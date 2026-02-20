#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include <cuda_runtime.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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

static int getLocalRank() {
    const char* vars[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "MPI_LOCALRANKID",
    };
    for (const char* v : vars) {
        if (const char* s = std::getenv(v)) return std::atoi(s);
    }
    return 0;
}

#define CUDA_CHECK(call)                                                                                 \
    do {                                                                                                 \
        cudaError_t _e = (call);                                                                          \
        if (_e != cudaSuccess) {                                                                          \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                 \
        }                                                                                                \
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

__global__ void computeForcesKernel(const int n,
                                   const int local_n,
                                   const double* __restrict__ gpx,
                                   const double* __restrict__ gpy,
                                   const double* __restrict__ gpz,
                                   const double* __restrict__ lpx,
                                   const double* __restrict__ lpy,
                                   const double* __restrict__ lpz,
                                   double* __restrict__ lvx,
                                   double* __restrict__ lvy,
                                   double* __restrict__ lvz) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (i < local_n);

    const double xi = active ? lpx[i] : 0.0;
    const double yi = active ? lpy[i] : 0.0;
    const double zi = active ? lpz[i] : 0.0;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    extern __shared__ double sh[];
    double* shx = sh;
    double* shy = sh + blockDim.x;
    double* shz = sh + 2 * blockDim.x;

    for (int tile = 0; tile < n; tile += blockDim.x) {
        const int j = tile + threadIdx.x;
        if (j < n) {
            shx[threadIdx.x] = gpx[j];
            shy[threadIdx.x] = gpy[j];
            shz[threadIdx.x] = gpz[j];
        }
        __syncthreads();

        if (active) {
            const int tileSize = (tile + blockDim.x <= n) ? blockDim.x : (n - tile);
            for (int k = 0; k < tileSize; ++k) {
                const double dx = shx[k] - xi;
                const double dy = shy[k] - yi;
                const double dz = shz[k] - zi;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / __dsqrt_rn(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        lvx[i] += DT * Fx;
        lvy[i] += DT * Fy;
        lvz[i] += DT * Fz;
    }
}

__global__ void integrateKernel(const int local_n,
                               double* __restrict__ lpx,
                               double* __restrict__ lpy,
                               double* __restrict__ lpz,
                               const double* __restrict__ lvx,
                               const double* __restrict__ lvy,
                               const double* __restrict__ lvz) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= local_n) return;
    lpx[i] += lvx[i] * DT;
    lpy[i] += lvy[i] * DT;
    lpz[i] += lvz[i] * DT;
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
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

static void computeCountsDispls(const int n, const int worldSize, std::vector<int>& counts, std::vector<int>& displs) {
    counts.resize(worldSize);
    displs.resize(worldSize);
    const int base = n / worldSize;
    const int rem = n % worldSize;
    int off = 0;
    for (int r = 0; r < worldSize; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = off;
        off += counts[r];
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

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

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // CUDA setup (one GPU per local rank)
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount <= 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int dev = getLocalRank() % devCount;
    CUDA_CHECK(cudaSetDevice(dev));

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // MPI decomposition
    std::vector<int> counts, displs;
    computeCountsDispls(numBodies, worldSize, counts, displs);
    const int local_n = counts[rank];

    // Host buffers
    std::vector<double> lpx(local_n), lpy(local_n), lpz(local_n);
    std::vector<double> lvx(local_n), lvy(local_n), lvz(local_n);

    std::vector<double> gpx(numBodies), gpy(numBodies), gpz(numBodies);

    // Rank0 initializes exactly like the original and scatters.
    std::vector<double> init_px, init_py, init_pz, init_vx, init_vy, init_vz;
    if (rank == 0) {
        std::vector<Body> bodies(numBodies);
        randomizeBodies(bodies);
        init_px.resize(numBodies);
        init_py.resize(numBodies);
        init_pz.resize(numBodies);
        init_vx.resize(numBodies);
        init_vy.resize(numBodies);
        init_vz.resize(numBodies);
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < numBodies; ++i) {
            init_px[i] = bodies[i].pos.x;
            init_py[i] = bodies[i].pos.y;
            init_pz[i] = bodies[i].pos.z;
            init_vx[i] = bodies[i].vel.x;
            init_vy[i] = bodies[i].vel.y;
            init_vz[i] = bodies[i].vel.z;
        }
    }

    MPI_Scatterv(rank == 0 ? init_px.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, lpx.data(), local_n,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? init_py.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, lpy.data(), local_n,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? init_pz.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, lpz.data(), local_n,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? init_vx.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, lvx.data(), local_n,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? init_vy.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, lvy.data(), local_n,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? init_vz.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE, lvz.data(), local_n,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Initial global position replication.
    MPI_Allgatherv(lpx.data(), local_n, MPI_DOUBLE, gpx.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(lpy.data(), local_n, MPI_DOUBLE, gpy.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(lpz.data(), local_n, MPI_DOUBLE, gpz.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // Device buffers
    double *d_gpx = nullptr, *d_gpy = nullptr, *d_gpz = nullptr;
    double *d_lpx = nullptr, *d_lpy = nullptr, *d_lpz = nullptr;
    double *d_lvx = nullptr, *d_lvy = nullptr, *d_lvz = nullptr;

    CUDA_CHECK(cudaMalloc(&d_gpx, sizeof(double) * numBodies));
    CUDA_CHECK(cudaMalloc(&d_gpy, sizeof(double) * numBodies));
    CUDA_CHECK(cudaMalloc(&d_gpz, sizeof(double) * numBodies));

    CUDA_CHECK(cudaMalloc(&d_lpx, sizeof(double) * local_n));
    CUDA_CHECK(cudaMalloc(&d_lpy, sizeof(double) * local_n));
    CUDA_CHECK(cudaMalloc(&d_lpz, sizeof(double) * local_n));

    CUDA_CHECK(cudaMalloc(&d_lvx, sizeof(double) * local_n));
    CUDA_CHECK(cudaMalloc(&d_lvy, sizeof(double) * local_n));
    CUDA_CHECK(cudaMalloc(&d_lvz, sizeof(double) * local_n));

    CUDA_CHECK(cudaMemcpyAsync(d_lpx, lpx.data(), sizeof(double) * local_n, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_lpy, lpy.data(), sizeof(double) * local_n, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_lpz, lpz.data(), sizeof(double) * local_n, cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_lvx, lvx.data(), sizeof(double) * local_n, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_lvy, lvy.data(), sizeof(double) * local_n, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_lvz, lvz.data(), sizeof(double) * local_n, cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const int block = 256;
    const int grid = (local_n + block - 1) / block;
    const size_t shmem = sizeof(double) * 3 * block;

    for (int step = 0; step < numSteps; ++step) {
        CUDA_CHECK(cudaMemcpyAsync(d_gpx, gpx.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_gpy, gpy.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_gpz, gpz.data(), sizeof(double) * numBodies, cudaMemcpyHostToDevice, stream));

        computeForcesKernel<<<grid, block, shmem, stream>>>(numBodies, local_n, d_gpx, d_gpy, d_gpz, d_lpx, d_lpy, d_lpz,
                                                           d_lvx, d_lvy, d_lvz);
        CUDA_CHECK(cudaGetLastError());

        integrateKernel<<<grid, block, 0, stream>>>(local_n, d_lpx, d_lpy, d_lpz, d_lvx, d_lvy, d_lvz);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(lpx.data(), d_lpx, sizeof(double) * local_n, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(lpy.data(), d_lpy, sizeof(double) * local_n, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(lpz.data(), d_lpz, sizeof(double) * local_n, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        MPI_Allgatherv(lpx.data(), local_n, MPI_DOUBLE, gpx.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(lpy.data(), local_n, MPI_DOUBLE, gpy.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(lpz.data(), local_n, MPI_DOUBLE, gpz.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather final state to rank0 for output/validation
    CUDA_CHECK(cudaMemcpyAsync(lvx.data(), d_lvx, sizeof(double) * local_n, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(lvy.data(), d_lvy, sizeof(double) * local_n, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(lvz.data(), d_lvz, sizeof(double) * local_n, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> full_px, full_py, full_pz, full_vx, full_vy, full_vz;
    if (rank == 0) {
        full_px.resize(numBodies);
        full_py.resize(numBodies);
        full_pz.resize(numBodies);
        full_vx.resize(numBodies);
        full_vy.resize(numBodies);
        full_vz.resize(numBodies);
    }

    MPI_Gatherv(lpx.data(), local_n, MPI_DOUBLE, rank == 0 ? full_px.data() : nullptr, counts.data(), displs.data(),
               MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(lpy.data(), local_n, MPI_DOUBLE, rank == 0 ? full_py.data() : nullptr, counts.data(), displs.data(),
               MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(lpz.data(), local_n, MPI_DOUBLE, rank == 0 ? full_pz.data() : nullptr, counts.data(), displs.data(),
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Gatherv(lvx.data(), local_n, MPI_DOUBLE, rank == 0 ? full_vx.data() : nullptr, counts.data(), displs.data(),
               MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(lvy.data(), local_n, MPI_DOUBLE, rank == 0 ? full_vy.data() : nullptr, counts.data(), displs.data(),
               MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(lvz.data(), local_n, MPI_DOUBLE, rank == 0 ? full_vz.data() : nullptr, counts.data(), displs.data(),
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int validation_ok = 1;

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());

        // Print results for external validation (keep identical element order)
        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve((size_t)numBodies * 6);
            for (int i = 0; i < numBodies; ++i) {
                bodyData.push_back(full_px[i]);
                bodyData.push_back(full_py[i]);
                bodyData.push_back(full_pz[i]);
                bodyData.push_back(full_vx[i]);
                bodyData.push_back(full_vy[i]);
                bodyData.push_back(full_vz[i]);
            }
            print_results(bodyData, "Bodies");
        }

        if (validate) {
            printf("Validating simulation results...\n");
            std::vector<Body> bodies(numBodies);
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (int i = 0; i < numBodies; ++i) {
                bodies[i].pos.x = full_px[i];
                bodies[i].pos.y = full_py[i];
                bodies[i].pos.z = full_pz[i];
                bodies[i].vel.x = full_vx[i];
                bodies[i].vel.y = full_vy[i];
                bodies[i].vel.z = full_vz[i];
            }

            if (validateSimulation(bodies)) {
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
                validation_ok = 1;
            } else {
                printf("Validation: FAILED\n");
                validation_ok = 0;
            }
        }
    }

    if (validate) MPI_Bcast(&validation_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(d_gpx));
    CUDA_CHECK(cudaFree(d_gpy));
    CUDA_CHECK(cudaFree(d_gpz));

    CUDA_CHECK(cudaFree(d_lpx));
    CUDA_CHECK(cudaFree(d_lpy));
    CUDA_CHECK(cudaFree(d_lpz));

    CUDA_CHECK(cudaFree(d_lvx));
    CUDA_CHECK(cudaFree(d_lvy));
    CUDA_CHECK(cudaFree(d_lvz));

    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Finalize();

    return (validate && !validation_ok) ? 1 : 0;
}
