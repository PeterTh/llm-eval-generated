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

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

constexpr int BLOCK = 128;

// Forces on this rank's slice [offset, offset+count) from all n bodies; updates velocity.
// Summation over j in ascending order, same as the serial code.
__global__ void forcesKernel(const double4* __restrict__ allPos, double4* __restrict__ vel,
                             int n, int offset, int count) {
    __shared__ double4 tile[BLOCK];
    const int li = blockIdx.x * BLOCK + threadIdx.x;
    const int i = offset + li;
    double4 pi = make_double4(0, 0, 0, 0);
    if (li < count) pi = allPos[i];
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += BLOCK) {
        const int j = base + threadIdx.x;
        tile[threadIdx.x] = (j < n) ? allPos[j] : make_double4(0, 0, 0, 0);
        __syncthreads();
        const int lim = min(BLOCK, n - base);
#pragma unroll 8
        for (int k = 0; k < lim; ++k) {
            const double dx = tile[k].x - pi.x;
            const double dy = tile[k].y - pi.y;
            const double dz = tile[k].z - pi.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }
    if (li < count) {
        double4 v = vel[li];
        v.x += DT * Fx;
        v.y += DT * Fy;
        v.z += DT * Fz;
        vel[li] = v;
    }
}

// Advance positions of local slice; writes into the full position array and a compact send buffer.
__global__ void integrateKernel(double4* __restrict__ allPos, const double4* __restrict__ vel,
                                double4* __restrict__ sendBuf, int offset, int count) {
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= count) return;
    double4 p = allPos[offset + li];
    const double4 v = vel[li];
    p.x += v.x * DT;
    p.y += v.y * DT;
    p.z += v.z * DT;
    allPos[offset + li] = p;
    sendBuf[li] = p;
}

// Per-body potential energy sum (over all j != i) for the local slice.
__global__ void energyKernel(const double4* __restrict__ allPos, double* __restrict__ out,
                             int n, int offset, int count) {
    __shared__ double4 tile[BLOCK];
    const int li = blockIdx.x * BLOCK + threadIdx.x;
    const int i = offset + li;
    double4 pi = make_double4(0, 0, 0, 0);
    if (li < count) pi = allPos[i];
    double acc = 0.0;
    for (int base = 0; base < n; base += BLOCK) {
        const int j = base + threadIdx.x;
        tile[threadIdx.x] = (j < n) ? allPos[j] : make_double4(0, 0, 0, 0);
        __syncthreads();
        const int lim = min(BLOCK, n - base);
        for (int k = 0; k < lim; ++k) {
            if (base + k == i) continue;
            const double dx = tile[k].x - pi.x;
            const double dy = tile[k].y - pi.y;
            const double dz = tile[k].z - pi.z;
            acc += 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
        __syncthreads();
    }
    if (li < count) out[li] = acc;
}

// Validate that simulation produces finite, reasonable values (local slice)
bool validateSimulation(const std::vector<Body>& bodies) {
    int bad = 0;
    const long n = (long)bodies.size();
#pragma omp parallel for reduction(| : bad) schedule(static)
    for (long i = 0; i < n; ++i) {
        const auto& body = bodies[i];
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            bad |= 1;
            continue;
        }
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos)
            bad |= 2;
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel)
            bad |= 4;
    }
    int all = 0;
    MPI_Allreduce(&bad, &all, 1, MPI_INT, MPI_BOR, MPI_COMM_WORLD);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        if (all & 1) printf("Validation failed: found NaN or Inf value in body state\n");
        if (all & 2) printf("Validation failed: body position exceeds reasonable bounds\n");
        if (all & 4) printf("Validation failed: body velocity exceeds reasonable bounds\n");
    }
    return all == 0;
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
    if (numBodies < 0) numBodies = 0;

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Select a GPU per rank (round-robin over ranks sharing a node)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lrank = 0, ndev = 0;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev == 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

    // Block partition of bodies across ranks
    std::vector<int> counts(nranks), displs(nranks), counts4(nranks), displs4(nranks);
    for (int r = 0; r < nranks; ++r) {
        const int base = numBodies / nranks, rem = numBodies % nranks;
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = r * base + (r < rem ? r : rem);
        counts4[r] = counts[r] * 4;
        displs4[r] = displs[r] * 4;
    }
    const int offset = displs[rank];
    const int count = counts[rank];

    // Initialize bodies (deterministic; every rank generates the same full set)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    double4 *h_pos = nullptr, *h_vel = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pos, sizeof(double4) * (numBodies + 1)));
    CUDA_CHECK(cudaMallocHost(&h_vel, sizeof(double4) * (numBodies + 1)));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) {
        h_pos[i] = make_double4(bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z, 0.0);
        h_vel[i] = make_double4(bodies[i].vel.x, bodies[i].vel.y, bodies[i].vel.z, 0.0);
    }

    double4 *d_pos, *d_vel, *d_send;
    double* d_en;
    CUDA_CHECK(cudaMalloc(&d_pos, sizeof(double4) * (numBodies + 1)));
    CUDA_CHECK(cudaMalloc(&d_vel, sizeof(double4) * (count + 1)));
    CUDA_CHECK(cudaMalloc(&d_send, sizeof(double4) * (count + 1)));
    CUDA_CHECK(cudaMalloc(&d_en, sizeof(double) * (count + 1)));
    CUDA_CHECK(cudaMemcpy(d_pos, h_pos, sizeof(double4) * numBodies, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, h_vel + offset, sizeof(double4) * count, cudaMemcpyHostToDevice));

    const int grid = (count + BLOCK - 1) / BLOCK;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        if (grid > 0) {
            forcesKernel<<<grid, BLOCK, 0, stream>>>(d_pos, d_vel, numBodies, offset, count);
            integrateKernel<<<grid, BLOCK, 0, stream>>>(d_pos, d_vel, d_send, offset, count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(h_pos + offset, d_send, sizeof(double4) * count,
                                       cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (nranks > 1) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_pos, counts4.data(), displs4.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
            // Upload only the remote portions (own slice is already current on the device)
            if (offset > 0)
                CUDA_CHECK(cudaMemcpyAsync(d_pos, h_pos, sizeof(double4) * offset,
                                           cudaMemcpyHostToDevice, stream));
            if (offset + count < numBodies)
                CUDA_CHECK(cudaMemcpyAsync(d_pos + offset + count, h_pos + offset + count,
                                           sizeof(double4) * (numBodies - offset - count),
                                           cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    // Bring back final state: positions are already global on host; gather velocities
    if (count > 0)
        CUDA_CHECK(cudaMemcpy(h_vel + offset, d_vel, sizeof(double4) * count, cudaMemcpyDeviceToHost));
    if (nranks > 1)
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_vel, counts4.data(), displs4.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) printf("Simulation time: %ld ms\n", duration.count());

    if (printResults && rank == 0) {
        std::vector<double> bodyData(numBodies * 6);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            bodyData[6 * i + 0] = h_pos[i].x;
            bodyData[6 * i + 1] = h_pos[i].y;
            bodyData[6 * i + 2] = h_pos[i].z;
            bodyData[6 * i + 3] = h_vel[i].x;
            bodyData[6 * i + 4] = h_vel[i].y;
            bodyData[6 * i + 5] = h_vel[i].z;
        }
        print_results(bodyData, "Bodies");
    }

    int rc = 0;
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");

        // Each rank validates its own slice
        std::vector<Body> local(count);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < count; ++i) {
            const double4 p = h_pos[offset + i], v = h_vel[offset + i];
            local[i].pos = Vec3(p.x, p.y, p.z);
            local[i].vel = Vec3(v.x, v.y, v.z);
        }
        if (validateSimulation(local)) {
            // Kinetic energy (unit mass)
            double kin = 0.0;
#pragma omp parallel for reduction(+ : kin) schedule(static)
            for (int i = 0; i < count; ++i) {
                const double4 v = h_vel[offset + i];
                kin += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
            }
            // Potential energy: each pair counted twice in the per-body sums
            double pot = 0.0;
            if (grid > 0) {
                std::vector<double> en(count);
                energyKernel<<<grid, BLOCK>>>(d_pos, d_en, numBodies, offset, count);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(en.data(), d_en, sizeof(double) * count, cudaMemcpyDeviceToHost));
#pragma omp parallel for reduction(+ : pot) schedule(static)
                for (int i = 0; i < count; ++i) pot += en[i];
            }
            double loc = kin - 0.5 * pot, energy = 0.0;
            MPI_Reduce(&loc, &energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Final energy: %.6f\n", energy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            rc = 1;
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    cudaFree(d_pos);
    cudaFree(d_vel);
    cudaFree(d_send);
    cudaFree(d_en);
    cudaFreeHost(h_pos);
    cudaFreeHost(h_vel);
    MPI_Finalize();
    return rc;
}
