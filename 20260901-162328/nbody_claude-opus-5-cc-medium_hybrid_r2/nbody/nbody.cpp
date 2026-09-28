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

// One CUDA thread per body, one MPI rank per GPU, OpenMP for the host-side work.
// A warp-sized block (which is also the shared-memory tile size) measured fastest: the force
// evaluation is FP64-throughput bound and small blocks spread the work evenly over all SMs.
constexpr int BLOCK_SIZE = 32;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call)                                                                                     \
    do {                                                                                                     \
        const cudaError_t err_ = (call);                                                                      \
        if (err_ != cudaSuccess) {                                                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__);        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                     \
        }                                                                                                    \
    } while (0)

// Structure-of-arrays state. Positions are replicated on every rank (needed by every interaction),
// velocities only exist for the bodies a rank owns.
struct DeviceState {
    double *px = nullptr, *py = nullptr, *pz = nullptr;  // n entries (all bodies)
    double *vx = nullptr, *vy = nullptr, *vz = nullptr;  // count entries (owned bodies)
    double* partial = nullptr;                           // n entries, potential-energy partials
};

// The reference build lets the host compiler contract these expressions into FMAs; the device code
// spells the same contractions out explicitly (and is compiled with -fmad=false) so that both
// produce bit-identical results.
__device__ __forceinline__ double distanceSquared(const double dx, const double dy, const double dz) {
    return fma(dz, dz, fma(dx, dx, dy * dy)) + SOFTENING;
}

// Force accumulation for the locally owned bodies. The inner loop walks j in ascending order,
// exactly like the sequential reference, so the summation order (and thus the result) is preserved.
__global__ void computeForcesKernel(const double* __restrict__ px, const double* __restrict__ py,
                                    const double* __restrict__ pz, double* __restrict__ vx, double* __restrict__ vy,
                                    double* __restrict__ vz, const int n, const int start, const int count) {
    __shared__ double sx[BLOCK_SIZE], sy[BLOCK_SIZE], sz[BLOCK_SIZE];

    const int local = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool active = local < count;
    const int i = start + local;

    const double xi = active ? px[i] : 0.0;
    const double yi = active ? py[i] : 0.0;
    const double zi = active ? pz[i] : 0.0;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    const int fullTiles = n / BLOCK_SIZE;
    for (int tile = 0; tile < fullTiles; ++tile) {
        const int j = tile * BLOCK_SIZE + threadIdx.x;
        sx[threadIdx.x] = px[j];
        sy[threadIdx.x] = py[j];
        sz[threadIdx.x] = pz[j];
        __syncthreads();

#pragma unroll 8
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            const double dx = sx[k] - xi;
            const double dy = sy[k] - yi;
            const double dz = sz[k] - zi;
            const double distSqr = distanceSquared(dx, dy, dz);
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx = fma(dx, invDist3, Fx);
            Fy = fma(dy, invDist3, Fy);
            Fz = fma(dz, invDist3, Fz);
        }
        __syncthreads();
    }

    // Remainder tile
    const int tailStart = fullTiles * BLOCK_SIZE;
    const int tail = n - tailStart;
    if (tail > 0) {
        if (threadIdx.x < tail) {
            const int j = tailStart + threadIdx.x;
            sx[threadIdx.x] = px[j];
            sy[threadIdx.x] = py[j];
            sz[threadIdx.x] = pz[j];
        }
        __syncthreads();

        for (int k = 0; k < tail; ++k) {
            const double dx = sx[k] - xi;
            const double dy = sy[k] - yi;
            const double dz = sz[k] - zi;
            const double distSqr = distanceSquared(dx, dy, dz);
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx = fma(dx, invDist3, Fx);
            Fy = fma(dy, invDist3, Fy);
            Fz = fma(dz, invDist3, Fz);
        }
    }

    if (active) {
        vx[local] += DT * Fx;
        vy[local] += DT * Fy;
        vz[local] += DT * Fz;
    }
}

__global__ void integrateKernel(double* __restrict__ px, double* __restrict__ py, double* __restrict__ pz,
                                const double* __restrict__ vx, const double* __restrict__ vy,
                                const double* __restrict__ vz, const int start, const int count) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;
    const int i = start + local;
    px[i] = fma(vx[local], DT, px[i]);
    py[i] = fma(vy[local], DT, py[i]);
    pz[i] = fma(vz[local], DT, pz[i]);
}

// Potential-energy partial sums: partial[i] = sum_{j>i} 1/dist(i,j). Bodies are distributed
// cyclically across ranks because the work per i shrinks with i.
__global__ void potentialKernel(const double* __restrict__ px, const double* __restrict__ py,
                                const double* __restrict__ pz, double* __restrict__ partial, const int n,
                                const int rank, const int numRanks) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = rank + idx * numRanks;
    if (i >= n) return;

    const double xi = px[i], yi = py[i], zi = pz[i];
    double sum = 0.0;
    for (int j = i + 1; j < n; ++j) {
        const double dx = px[j] - xi;
        const double dy = py[j] - yi;
        const double dz = pz[j] - zi;
        const double dist = sqrt(distanceSquared(dx, dy, dz));
        sum -= 1.0 / dist;
    }
    partial[i] = sum;
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

// Kinetic energy on the host (O(n), kept in the original order), potential energy from the
// distributed GPU partials.
double computeTotalEnergy(const std::vector<Body>& bodies, const std::vector<double>& potentialPartials) {
    double energy = 0.0;

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies)
    for (const double partial : potentialPartials) {
        energy += partial;
    }

    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    // 0 = ok, 1 = non-finite, 2 = position out of bounds, 3 = velocity out of bounds
    std::vector<unsigned char> status(n, 0);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const Body& body = bodies[i];

        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            status[i] = 1;
            continue;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            status[i] = 2;
        } else if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            status[i] = 3;
        }
    }

    // Report the first offending body, matching the sequential order of the original check
    for (size_t i = 0; i < n; ++i) {
        switch (status[i]) {
            case 0: continue;
            case 1: printf("Validation failed: found NaN or Inf value in body state\n"); return false;
            case 2: printf("Validation failed: body position exceeds reasonable bounds\n"); return false;
            default: printf("Validation failed: body velocity exceeds reasonable bounds\n"); return false;
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

// Pick the GPU for this rank based on its position within the node.
static void selectDevice(MPI_Comm comm, int rank) {
    MPI_Comm nodeComm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(nodeRank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr));  // force context creation outside the timed region
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    const bool isRoot = (rank == 0);

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
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    selectDevice(MPI_COMM_WORLD, rank);

    if (isRoot) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const int n = numBodies;
    if (n <= 0) {
        if (isRoot) {
            printf("Simulation time: 0 ms\n");
            if (printResults) print_results(std::vector<double>{}, "Bodies");
            if (validate) {
                printf("Validating simulation results...\n");
                printf("Final energy: %.6f\n", 0.0);
                printf("Validation: PASSED\n");
            }
        }
        MPI_Finalize();
        return 0;
    }

    // Block distribution of bodies over ranks
    std::vector<int> counts(numRanks), displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        counts[r] = n / numRanks + (r < n % numRanks ? 1 : 0);
        displs[r] = (r == 0) ? 0 : displs[r - 1] + counts[r - 1];
    }
    const int start = displs[rank];
    const int count = counts[rank];

    // Initialize bodies (identical on every rank; the RNG stream is inherently sequential)
    std::vector<Body> bodies(n);
    randomizeBodies(bodies);

    // Pinned host staging buffers for the per-step position exchange
    double *hx = nullptr, *hy = nullptr, *hz = nullptr;
    CUDA_CHECK(cudaMallocHost(&hx, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&hy, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&hz, n * sizeof(double)));

    std::vector<double> vxHost(count), vyHost(count), vzHost(count);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        hx[i] = bodies[i].pos.x;
        hy[i] = bodies[i].pos.y;
        hz[i] = bodies[i].pos.z;
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) {
        vxHost[i] = bodies[start + i].vel.x;
        vyHost[i] = bodies[start + i].vel.y;
        vzHost[i] = bodies[start + i].vel.z;
    }

    DeviceState d;
    CUDA_CHECK(cudaMalloc(&d.px, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.py, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.pz, n * sizeof(double)));
    const size_t velBytes = (count > 0 ? count : 1) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d.vx, velBytes));
    CUDA_CHECK(cudaMalloc(&d.vy, velBytes));
    CUDA_CHECK(cudaMalloc(&d.vz, velBytes));

    CUDA_CHECK(cudaMemcpy(d.px, hx, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.py, hy, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.pz, hz, n * sizeof(double), cudaMemcpyHostToDevice));
    if (count > 0) {
        CUDA_CHECK(cudaMemcpy(d.vx, vxHost.data(), count * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.vy, vyHost.data(), count * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.vz, vzHost.data(), count * sizeof(double), cudaMemcpyHostToDevice));
    }

    const int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        if (count > 0) {
            computeForcesKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(d.px, d.py, d.pz, d.vx, d.vy, d.vz, n, start, count);
            integrateKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(d.px, d.py, d.pz, d.vx, d.vy, d.vz, start, count);
            CUDA_CHECK(cudaMemcpyAsync(hx + start, d.px + start, count * sizeof(double), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(hy + start, d.py + start, count * sizeof(double), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(hz + start, d.pz + start, count * sizeof(double), cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (numRanks > 1) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hx, counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hy, counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hz, counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

            CUDA_CHECK(cudaMemcpyAsync(d.px, hx, n * sizeof(double), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d.py, hy, n * sizeof(double), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d.pz, hz, n * sizeof(double), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (isRoot) printf("Simulation time: %ld ms\n", maxDuration);

    // Collect the final state on the root rank
    if (count > 0) {
        CUDA_CHECK(cudaMemcpy(vxHost.data(), d.vx, count * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vyHost.data(), d.vy, count * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vzHost.data(), d.vz, count * sizeof(double), cudaMemcpyDeviceToHost));
    }

    std::vector<double> allVx(isRoot ? n : 0), allVy(isRoot ? n : 0), allVz(isRoot ? n : 0);
    MPI_Gatherv(vxHost.data(), count, MPI_DOUBLE, allVx.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vyHost.data(), count, MPI_DOUBLE, allVy.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vzHost.data(), count, MPI_DOUBLE, allVz.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // hx/hy/hz already hold the full, up-to-date position array on every rank
    if (isRoot) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            bodies[i].pos = Vec3(hx[i], hy[i], hz[i]);
            bodies[i].vel = Vec3(allVx[i], allVy[i], allVz[i]);
        }
    }

    // Print results for external validation
    if (printResults && isRoot) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData(static_cast<size_t>(n) * 6);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double* out = &bodyData[static_cast<size_t>(i) * 6];
            out[0] = bodies[i].pos.x;
            out[1] = bodies[i].pos.y;
            out[2] = bodies[i].pos.z;
            out[3] = bodies[i].vel.x;
            out[4] = bodies[i].vel.y;
            out[5] = bodies[i].vel.z;
        }
        print_results(bodyData, "Bodies");
    }

    // Validation: check that simulation produces finite, reasonable values
    int status = 0;
    if (validate) {
        // Potential energy partials are computed on all GPUs, then combined on the root
        std::vector<double> partials(n, 0.0);
        std::vector<double> localPartials(n, 0.0);
        CUDA_CHECK(cudaMalloc(&d.partial, n * sizeof(double)));
        CUDA_CHECK(cudaMemset(d.partial, 0, n * sizeof(double)));
        const int myRows = (n - rank + numRanks - 1) / numRanks;
        if (myRows > 0) {
            const int pblocks = (myRows + BLOCK_SIZE - 1) / BLOCK_SIZE;
            potentialKernel<<<pblocks, BLOCK_SIZE>>>(d.px, d.py, d.pz, d.partial, n, rank, numRanks);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaMemcpy(localPartials.data(), d.partial, n * sizeof(double), cudaMemcpyDeviceToHost));
        // Each rank contributes zeros outside its own rows, so the sum is exact
        MPI_Reduce(localPartials.data(), partials.data(), n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (isRoot) {
            printf("Validating simulation results...\n");

            if (validateSimulation(bodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(bodies, partials);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        CUDA_CHECK(cudaFree(d.partial));
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d.px));
    CUDA_CHECK(cudaFree(d.py));
    CUDA_CHECK(cudaFree(d.pz));
    CUDA_CHECK(cudaFree(d.vx));
    CUDA_CHECK(cudaFree(d.vy));
    CUDA_CHECK(cudaFree(d.vz));
    CUDA_CHECK(cudaFreeHost(hx));
    CUDA_CHECK(cudaFreeHost(hy));
    CUDA_CHECK(cudaFreeHost(hz));

    MPI_Finalize();
    return status;
}
