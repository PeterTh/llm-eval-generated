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

// Number of bodies staged in shared memory per tile (also the CUDA block size)
constexpr int TILE = 64;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call)                                                                                    \
    do {                                                                                                    \
        const cudaError_t err__ = (call);                                                                   \
        if (err__ != cudaSuccess) {                                                                         \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__);      \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                   \
        }                                                                                                   \
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

// Each thread owns one local body i and accumulates over all j in ascending order,
// matching the summation order of the original serial code.
__global__ void computeForcesKernel(const double* __restrict__ px, const double* __restrict__ py,
                                    const double* __restrict__ pz, double* __restrict__ vx, double* __restrict__ vy,
                                    double* __restrict__ vz, const int n, const int offset, const int localN) {
    __shared__ double sx[TILE];
    __shared__ double sy[TILE];
    __shared__ double sz[TILE];

    const int li = blockIdx.x * TILE + threadIdx.x;
    const bool active = li < localN;
    const int gi = offset + li;

    const double xi = active ? px[gi] : 0.0;
    const double yi = active ? py[gi] : 0.0;
    const double zi = active ? pz[gi] : 0.0;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = 0; tile < n; tile += TILE) {
        const int idx = tile + threadIdx.x;
        if (idx < n) {
            sx[threadIdx.x] = px[idx];
            sy[threadIdx.x] = py[idx];
            sz[threadIdx.x] = pz[idx];
        }
        __syncthreads();

        const int m = min(TILE, n - tile);
#pragma unroll 8
        for (int k = 0; k < m; ++k) {
            const double dx = sx[k] - xi;
            const double dy = sy[k] - yi;
            const double dz = sz[k] - zi;
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
        vx[li] += DT * Fx;
        vy[li] += DT * Fy;
        vz[li] += DT * Fz;
    }
}

// Advances the locally owned positions in place and packs them for the all-gather.
__global__ void integrateKernel(double* __restrict__ px, double* __restrict__ py, double* __restrict__ pz,
                                const double* __restrict__ vx, const double* __restrict__ vy,
                                const double* __restrict__ vz, double* __restrict__ sendBuf, const int offset,
                                const int localN) {
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= localN) return;

    const int gi = offset + li;
    const double x = px[gi] + vx[li] * DT;
    const double y = py[gi] + vy[li] * DT;
    const double z = pz[gi] + vz[li] * DT;

    px[gi] = x;
    py[gi] = y;
    pz[gi] = z;

    sendBuf[3 * li + 0] = x;
    sendBuf[3 * li + 1] = y;
    sendBuf[3 * li + 2] = z;
}

// Scatters the gathered interleaved positions back into the SoA device arrays.
__global__ void unpackKernel(const double* __restrict__ recvBuf, double* __restrict__ px, double* __restrict__ py,
                             double* __restrict__ pz, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    px[i] = recvBuf[3 * i + 0];
    py[i] = recvBuf[3 * i + 1];
    pz[i] = recvBuf[3 * i + 2];
}

double computeTotalEnergy(const std::vector<Body>& bodies, const int rank, const int size) {
    const int n = static_cast<int>(bodies.size());

    double kinetic = 0.0;
    // Kinetic energy (assuming unit mass); each rank sums a strided slice.
#pragma omp parallel for reduction(+ : kinetic) schedule(static)
    for (int i = rank; i < n; i += size) {
        const auto& body = bodies[i];
        kinetic += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }

    double potential = 0.0;
    // Potential energy (assuming unit mass for all bodies); strided i keeps the
    // triangular workload balanced across ranks and threads.
#pragma omp parallel for reduction(+ : potential) schedule(dynamic, 8)
    for (int i = rank; i < n; i += size) {
        double local = 0.0;
        for (int j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            local -= 1.0 / dist;
        }
        potential += local;
    }

    double energy = kinetic + potential;
    double total = 0.0;
    MPI_Allreduce(&energy, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return total;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    const int n = static_cast<int>(bodies.size());
    int badValue = 0, badPos = 0, badVel = 0;

#pragma omp parallel for schedule(static) reduction(| : badValue, badPos, badVel)
    for (int i = 0; i < n; ++i) {
        const auto& body = bodies[i];
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            badValue = 1;
            continue;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            badPos = 1;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            badVel = 1;
        }
    }

    if (badValue) {
        printf("Validation failed: found NaN or Inf value in body state\n");
        return false;
    }
    if (badPos) {
        printf("Validation failed: body position exceeds reasonable bounds\n");
        return false;
    }
    if (badVel) {
        printf("Validation failed: body velocity exceeds reasonable bounds\n");
        return false;
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
    (void)provided;

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

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Bind one GPU per rank, round-robin within each node
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    // Split the node's cores between the ranks sharing it (unless the user asked otherwise)
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        const int cores = omp_get_num_procs() / localSize;
        omp_set_num_threads(cores > 0 ? cores : 1);
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // Initialize bodies (deterministic, replicated on every rank)
    std::vector<Body> bodies(numBodies > 0 ? numBodies : 0);
    randomizeBodies(bodies);

    const int n = numBodies > 0 ? numBodies : 0;

    // Block distribution of bodies over ranks
    std::vector<int> counts(size), displs(size), posCounts(size), posDispls(size);
    for (int r = 0; r < size; ++r) {
        const int base = n / size;
        const int rem = n % size;
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = r == 0 ? 0 : displs[r - 1] + counts[r - 1];
        posCounts[r] = 3 * counts[r];
        posDispls[r] = 3 * displs[r];
    }
    const int localN = counts[rank];
    const int offset = displs[rank];

    // Host staging buffers (SoA positions, local velocities, gather buffer)
    std::vector<double> hpx(n), hpy(n), hpz(n);
    std::vector<double> hvx(localN), hvy(localN), hvz(localN);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        hpx[i] = bodies[i].pos.x;
        hpy[i] = bodies[i].pos.y;
        hpz[i] = bodies[i].pos.z;
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < localN; ++i) {
        hvx[i] = bodies[offset + i].vel.x;
        hvy[i] = bodies[offset + i].vel.y;
        hvz[i] = bodies[offset + i].vel.z;
    }

    double *dpx = nullptr, *dpy = nullptr, *dpz = nullptr;
    double *dvx = nullptr, *dvy = nullptr, *dvz = nullptr;
    double *dsend = nullptr, *drecv = nullptr;
    const size_t nBytes = static_cast<size_t>(n) * sizeof(double);
    const size_t lBytes = static_cast<size_t>(localN > 0 ? localN : 1) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&dpx, nBytes ? nBytes : sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dpy, nBytes ? nBytes : sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dpz, nBytes ? nBytes : sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dvx, lBytes));
    CUDA_CHECK(cudaMalloc(&dvy, lBytes));
    CUDA_CHECK(cudaMalloc(&dvz, lBytes));
    CUDA_CHECK(cudaMalloc(&dsend, 3 * lBytes));
    CUDA_CHECK(cudaMalloc(&drecv, 3 * (nBytes ? nBytes : sizeof(double))));

    double* hsend = nullptr;
    double* hrecv = nullptr;
    CUDA_CHECK(cudaMallocHost(&hsend, 3 * lBytes));
    CUDA_CHECK(cudaMallocHost(&hrecv, 3 * (nBytes ? nBytes : sizeof(double))));

    if (n > 0) {
        CUDA_CHECK(cudaMemcpy(dpx, hpx.data(), nBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dpy, hpy.data(), nBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dpz, hpz.data(), nBytes, cudaMemcpyHostToDevice));
    }
    if (localN > 0) {
        CUDA_CHECK(cudaMemcpy(dvx, hvx.data(), localN * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dvy, hvy.data(), localN * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dvz, hvz.data(), localN * sizeof(double), cudaMemcpyHostToDevice));
    }

    const int forceBlocks = (localN + TILE - 1) / TILE;
    const int intBlocks = (localN + 255) / 256;
    const int unpackBlocks = (n + 255) / 256;

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps && n > 0; ++step) {
        if (forceBlocks > 0) {
            computeForcesKernel<<<forceBlocks, TILE>>>(dpx, dpy, dpz, dvx, dvy, dvz, n, offset, localN);
            integrateKernel<<<intBlocks, 256>>>(dpx, dpy, dpz, dvx, dvy, dvz, dsend, offset, localN);
        }

        if (size > 1) {
            if (localN > 0) {
                CUDA_CHECK(cudaMemcpy(hsend, dsend, 3 * localN * sizeof(double), cudaMemcpyDeviceToHost));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            MPI_Allgatherv(hsend, 3 * localN, MPI_DOUBLE, hrecv, posCounts.data(), posDispls.data(), MPI_DOUBLE,
                           MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(drecv, hrecv, 3 * nBytes, cudaMemcpyHostToDevice));
            unpackKernel<<<unpackBlocks, 256>>>(drecv, dpx, dpy, dpz, n);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // Collect the final state on every rank
    if (n > 0) {
        CUDA_CHECK(cudaMemcpy(hpx.data(), dpx, nBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hpy.data(), dpy, nBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hpz.data(), dpz, nBytes, cudaMemcpyDeviceToHost));
    }
    if (localN > 0) {
        CUDA_CHECK(cudaMemcpy(hsend, dvx, localN * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hsend + localN, dvy, localN * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hsend + 2 * localN, dvz, localN * sizeof(double), cudaMemcpyDeviceToHost));
#pragma omp parallel for schedule(static)
        for (int i = 0; i < localN; ++i) {
            hvx[i] = hsend[i];
            hvy[i] = hsend[localN + i];
            hvz[i] = hsend[2 * localN + i];
        }
    }

    std::vector<double> gvx(n), gvy(n), gvz(n);
    MPI_Allgatherv(hvx.data(), localN, MPI_DOUBLE, gvx.data(), counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(hvy.data(), localN, MPI_DOUBLE, gvy.data(), counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(hvz.data(), localN, MPI_DOUBLE, gvz.data(), counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        bodies[i].pos.x = hpx[i];
        bodies[i].pos.y = hpy[i];
        bodies[i].pos.z = hpz[i];
        bodies[i].vel.x = gvx[i];
        bodies[i].vel.y = gvy[i];
        bodies[i].vel.z = gvz[i];
    }

    CUDA_CHECK(cudaFree(dpx));
    CUDA_CHECK(cudaFree(dpy));
    CUDA_CHECK(cudaFree(dpz));
    CUDA_CHECK(cudaFree(dvx));
    CUDA_CHECK(cudaFree(dvy));
    CUDA_CHECK(cudaFree(dvz));
    CUDA_CHECK(cudaFree(dsend));
    CUDA_CHECK(cudaFree(drecv));
    CUDA_CHECK(cudaFreeHost(hsend));
    CUDA_CHECK(cudaFreeHost(hrecv));

    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData(static_cast<size_t>(n) * 6);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            bodyData[6 * i + 0] = bodies[i].pos.x;
            bodyData[6 * i + 1] = bodies[i].pos.y;
            bodyData[6 * i + 2] = bodies[i].pos.z;
            bodyData[6 * i + 3] = bodies[i].vel.x;
            bodyData[6 * i + 4] = bodies[i].vel.y;
            bodyData[6 * i + 5] = bodies[i].vel.z;
        }
        print_results(bodyData, "Bodies");
    }

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");

        // The state is replicated, so one rank checks it and shares the verdict.
        int allOk = rank == 0 ? (validateSimulation(bodies) ? 1 : 0) : 0;
        MPI_Bcast(&allOk, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (allOk) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies, rank, size);
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
