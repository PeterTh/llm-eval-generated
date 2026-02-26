#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#define OMPI_SKIP_MPICXX 1
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

static inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

static inline void computeRange(int n, int rank, int size, int& start, int& count) {
    const int base = n / size;
    const int rem = n % size;
    count = base + (rank < rem ? 1 : 0);
    start = rank * base + (rank < rem ? rank : rem);
}

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

// CUDA kernels: each rank updates only its local [start, start+count) bodies.
// All ranks keep a full copy of positions/velocities (via MPI_Allgatherv each step).
template <int TILE>
__global__ void computeForcesKernel(const double* __restrict__ posx,
                                   const double* __restrict__ posy,
                                   const double* __restrict__ posz,
                                   double* __restrict__ velx,
                                   double* __restrict__ vely,
                                   double* __restrict__ velz,
                                   int n, int start, int count) {
    const int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (local_i < count);

    // All threads in the block must participate in the shared-memory tile syncs.
    const int i = start + local_i;
    const double xi = active ? posx[i] : 0.0;
    const double yi = active ? posy[i] : 0.0;
    const double zi = active ? posz[i] : 0.0;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    __shared__ double shx[TILE];
    __shared__ double shy[TILE];
    __shared__ double shz[TILE];

    for (int tile = 0; tile < n; tile += TILE) {
        const int j = tile + threadIdx.x;
        if (j < n) {
            shx[threadIdx.x] = posx[j];
            shy[threadIdx.x] = posy[j];
            shz[threadIdx.x] = posz[j];
        }
        __syncthreads();

        const int tileSize = (tile + TILE <= n) ? TILE : (n - tile);
        #pragma unroll 4
        for (int k = 0; k < tileSize; ++k) {
            const double dx = shx[k] - xi;
            const double dy = shy[k] - yi;
            const double dz = shz[k] - zi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
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
                                int start, int count) {
    const int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_i >= count) return;
    const int i = start + local_i;
    posx[i] += velx[i] * DT;
    posy[i] += vely[i] * DT;
    posz[i] += velz[i] * DT;
}

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of bodies (default: 1024)\n");
    printf("  -s <num>     Number of simulation steps (default: 10)\n");
    printf("  -v           Enable validation (checks energy conservation)\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Validate that simulation produces finite, reasonable values
static bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
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

static double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy
    #pragma omp parallel for reduction(+:energy)
    for (size_t i = 0; i < n; ++i) {
        const auto& body = bodies[i];
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }

    // Potential energy
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double e = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            e -= 1.0 / dist;
        }
        energy += e;
    }

    return energy;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI does not provide required thread support (FUNNELED).\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

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

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) std::fprintf(stderr, "No CUDA devices found; this benchmark requires CUDA.\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    const int device = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    if (rank == 0) {
        printf("N-Body Simulation (MPI + OpenMP + CUDA)\n");
        printf("MPI ranks: %d\n", size);
        printf("Using CUDA device %d / %d per rank (rank 0 device shown)\n", device, deviceCount);
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate host SoA state (full copy on each rank).
    const int n = numBodies;
    std::vector<double> h_posx(n), h_posy(n), h_posz(n);
    std::vector<double> h_velx(n), h_vely(n), h_velz(n);

    // Initialize bodies on rank 0 and broadcast state.
    if (rank == 0) {
        std::vector<Body> bodies(n);
        randomizeBodies(bodies);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            h_posx[i] = bodies[i].pos.x;
            h_posy[i] = bodies[i].pos.y;
            h_posz[i] = bodies[i].pos.z;
            h_velx[i] = bodies[i].vel.x;
            h_vely[i] = bodies[i].vel.y;
            h_velz[i] = bodies[i].vel.z;
        }
    }

    MPI_Bcast(h_posx.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_posy.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_posz.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_velx.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_vely.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_velz.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int start = 0, count = 0;
    computeRange(n, rank, size, start, count);

    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        int s = 0, c = 0;
        computeRange(n, r, size, s, c);
        counts[r] = c;
        displs[r] = s;
    }

    // Allocate device state (full copy) and copy initial data.
    double *d_posx = nullptr, *d_posy = nullptr, *d_posz = nullptr;
    double *d_velx = nullptr, *d_vely = nullptr, *d_velz = nullptr;

    CUDA_CHECK(cudaMalloc(&d_posx, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_posy, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_posz, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_velx, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vely, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_velz, n * sizeof(double)));

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpyAsync(d_posx, h_posx.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_posy, h_posy.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_posz, h_posz.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_velx, h_velx.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vely, h_vely.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_velz, h_velz.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);
    const auto wallStart = std::chrono::high_resolution_clock::now();

    constexpr int BLOCK = 256;
    const int grid = (count + BLOCK - 1) / BLOCK;

    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<BLOCK><<<grid, BLOCK, 0, stream>>>(d_posx, d_posy, d_posz, d_velx, d_vely, d_velz, n, start, count);
        integrateKernel<<<grid, BLOCK, 0, stream>>>(d_posx, d_posy, d_posz, d_velx, d_vely, d_velz, start, count);
        CUDA_CHECK(cudaGetLastError());

        // Pull back only the locally-updated segment, then allgather to refresh the full state.
        CUDA_CHECK(cudaMemcpyAsync(h_posx.data() + start, d_posx + start, count * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_posy.data() + start, d_posy + start, count * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_posz.data() + start, d_posz + start, count * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_velx.data() + start, d_velx + start, count * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_vely.data() + start, d_vely + start, count * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_velz.data() + start, d_velz + start, count * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // In-place allgather: local segment is already in the correct offset within the receive buffer.
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, h_posx.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, h_posy.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, h_posz.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, h_velx.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, h_vely.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, h_velz.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // Refresh device full state for next iteration.
        CUDA_CHECK(cudaMemcpyAsync(d_posx, h_posx.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_posy, h_posy.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_posz, h_posz.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_velx, h_velx.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_vely, h_vely.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_velz, h_velz.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto wallEnd = std::chrono::high_resolution_clock::now();
    const auto localMs = std::chrono::duration_cast<std::chrono::milliseconds>(wallEnd - wallStart).count();
    long maxMs = 0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", maxMs);
    }

    // Output/validation only on rank 0.
    if (rank == 0) {
        std::vector<Body> bodies(n);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            bodies[i].pos.x = h_posx[i];
            bodies[i].pos.y = h_posy[i];
            bodies[i].pos.z = h_posz[i];
            bodies[i].vel.x = h_velx[i];
            bodies[i].vel.y = h_vely[i];
            bodies[i].vel.z = h_velz[i];
        }

        if (printResults) {
            std::vector<double> bodyData;
            bodyData.resize(static_cast<size_t>(n) * 6);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                const size_t off = static_cast<size_t>(i) * 6;
                bodyData[off + 0] = bodies[i].pos.x;
                bodyData[off + 1] = bodies[i].pos.y;
                bodyData[off + 2] = bodies[i].pos.z;
                bodyData[off + 3] = bodies[i].vel.x;
                bodyData[off + 4] = bodies[i].vel.y;
                bodyData[off + 5] = bodies[i].vel.z;
            }
            print_results(bodyData, "Bodies");
        }

        if (validate) {
            printf("Validating simulation results...\n");
            if (validateSimulation(bodies)) {
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                CUDA_CHECK(cudaStreamDestroy(stream));
                CUDA_CHECK(cudaFree(d_posx));
                CUDA_CHECK(cudaFree(d_posy));
                CUDA_CHECK(cudaFree(d_posz));
                CUDA_CHECK(cudaFree(d_velx));
                CUDA_CHECK(cudaFree(d_vely));
                CUDA_CHECK(cudaFree(d_velz));
                MPI_Finalize();
                return 1;
            }
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_posx));
    CUDA_CHECK(cudaFree(d_posy));
    CUDA_CHECK(cudaFree(d_posz));
    CUDA_CHECK(cudaFree(d_velx));
    CUDA_CHECK(cudaFree(d_vely));
    CUDA_CHECK(cudaFree(d_velz));

    MPI_Finalize();
    return 0;
}
