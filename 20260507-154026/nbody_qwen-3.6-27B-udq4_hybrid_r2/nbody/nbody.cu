#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

/* ------------------------------------------------------------------ */
/*  CUDA kernel – O(N^2) force computation on the GPU                 */
/*  Each thread computes the net force on one local body from all N   */
/*  bodies.  Velocities are updated in-place on the device.            */
/* ------------------------------------------------------------------ */
__global__ void computeForcesKernel(
        const Body* __restrict__ all_bodies, int n,
        Body* __restrict__        local_bodies, int local_n)
{
    int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
            + static_cast<int>(threadIdx.x);
    if (idx >= local_n) return;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    const double px = local_bodies[idx].pos.x;
    const double py = local_bodies[idx].pos.y;
    const double pz = local_bodies[idx].pos.z;

    for (int j = 0; j < n; ++j) {
        const double dx = all_bodies[j].pos.x - px;
        const double dy = all_bodies[j].pos.y - py;
        const double dz = all_bodies[j].pos.z - pz;
        const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        const double invDist = 1.0 / sqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    local_bodies[idx].vel.x += DT * Fx;
    local_bodies[idx].vel.y += DT * Fy;
    local_bodies[idx].vel.z += DT * Fz;
}

/* ------------------------------------------------------------------ */
/*  Host helpers                                                       */
/* ------------------------------------------------------------------ */

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

bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) ||
            !std::isfinite(body.pos.z) || !std::isfinite(body.vel.x) ||
            !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z))
        {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        const double maxPos = 1e6, maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos ||
            std::abs(body.pos.z) > maxPos)
        {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel ||
            std::abs(body.vel.z) > maxVel)
        {
            printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
}

/*  Total energy – OpenMP-parallelised reduction.                     */
double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy
    #pragma omp parallel for reduction(+:energy)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (bodies[i].vel.x * bodies[i].vel.x +
                         bodies[i].vel.y * bodies[i].vel.y +
                         bodies[i].vel.z * bodies[i].vel.z);
    }

    // Potential energy (upper-triangle pair loop)
    #pragma omp parallel for reduction(+:energy)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx*dx + dy*dy + dz*dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }
    return energy;
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

/* ------------------------------------------------------------------ */
/*  main – hybrid MPI / OpenMP / CUDA driver                           */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv) {
    int numBodies = 1024;
    int numSteps  = 10;
    bool validate     = false;
    bool printResults = false;

    // Parse CLI (before MPI_Init so -h works standalone)
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numBodies = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) numSteps  = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0)                  validate     = true;
        else if (strcmp(argv[i], "-r") == 0)                  printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); return 0; }
        else { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }

    MPI_Init(&argc, &argv);
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    /* ---- GPU selection (round-robin across ranks) ---------------- */
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0 && rank == 0) {
        fprintf(stderr, "Error: no CUDA devices available\n");
        MPI_Finalize(); return 1;
    }
    cudaSetDevice(rank % deviceCount);

    /* ---- Domain decomposition ------------------------------------ */
    const int base_n    = numBodies / numRanks;
    const int remainder = numBodies % numRanks;
    const int local_n   = base_n + (rank < remainder ? 1 : 0);
    const int local_start = base_n * rank + std::min(rank, remainder);

    /* Pre-compute Allgatherv counts / displacements (in bytes). ----- */
    const int body_sz = static_cast<int>(sizeof(Body));
    std::vector<int> recv_counts(numRanks), displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        int rn = base_n + (r < remainder ? 1 : 0);
        recv_counts[r] = rn * body_sz;
        displs[r] = (base_n * r + std::min(r, remainder)) * body_sz;
    }

    /* ---- Initialise bodies on rank 0, broadcast to all ----------- */
    std::vector<Body> all_bodies(numBodies);
    if (rank == 0) randomizeBodies(all_bodies);
    MPI_Bcast(all_bodies.data(), numBodies * body_sz, MPI_BYTE, 0, MPI_COMM_WORLD);

    /* Local chunk extracted from the global array. ------------------ */
    std::vector<Body> local_bodies(local_n);
    for (int i = 0; i < local_n; ++i)
        local_bodies[i] = all_bodies[local_start + i];

    /* ---- GPU memory ---------------------------------------------- */
    Body *d_all   = nullptr, *d_local = nullptr;
    cudaMalloc(&d_all,   numBodies * body_sz);
    if (local_n > 0) cudaMalloc(&d_local, local_n * body_sz);

    /* ---- Print info (rank 0 only) -------------------------------- */
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d | GPUs: %d | OpenMP threads: %d\n",
               numRanks, deviceCount, omp_get_max_threads());
    }

    /* ---- Kernel launch config ------------------------------------ */
    const int blockSize = 256;
    const int numBlocks = local_n > 0 ? (local_n + blockSize - 1) / blockSize : 0;

    /* ---- Simulation loop ----------------------------------------- */
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        // 1. MPI – gather every rank's bodies so each rank has the full set
        MPI_Allgatherv(local_bodies.data(), local_n * body_sz, MPI_BYTE,
                       all_bodies.data(), recv_counts.data(), displs.data(),
                       MPI_BYTE, MPI_COMM_WORLD);

        if (local_n > 0) {
            // 2. Host → Device transfers
            cudaMemcpy(d_all,   all_bodies.data(),   numBodies * body_sz, cudaMemcpyHostToDevice);
            cudaMemcpy(d_local, local_bodies.data(), local_n   * body_sz, cudaMemcpyHostToDevice);

            // 3. CUDA – force computation (the O(N^2) hot path)
            computeForcesKernel<<<numBlocks, blockSize>>>(d_all, numBodies, d_local, local_n);
            cudaDeviceSynchronize();

            // 4. Device → Host – retrieve updated velocities
            cudaMemcpy(local_bodies.data(), d_local, local_n * body_sz, cudaMemcpyDeviceToHost);
        }

        // 5. OpenMP – integrate positions on CPU
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < local_n; ++i) {
            local_bodies[i].pos.x += local_bodies[i].vel.x * DT;
            local_bodies[i].pos.y += local_bodies[i].vel.y * DT;
            local_bodies[i].pos.z += local_bodies[i].vel.z * DT;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

    /* ---- Gather final state to rank 0 ---------------------------- */
    MPI_Gatherv(local_bodies.data(), local_n * body_sz, MPI_BYTE,
                all_bodies.data(),    recv_counts.data(), displs.data(),
                MPI_BYTE, 0, MPI_COMM_WORLD);

    /* ---- Output (rank 0 only) ------------------------------------ */
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());

        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(static_cast<size_t>(numBodies) * 6);
            for (const auto& b : all_bodies) {
                bodyData.push_back(b.pos.x); bodyData.push_back(b.pos.y);
                bodyData.push_back(b.pos.z); bodyData.push_back(b.vel.x);
                bodyData.push_back(b.vel.y); bodyData.push_back(b.vel.z);
            }
            print_results(bodyData, "Bodies");
        }

        if (validate) {
            printf("Validating simulation results...\n");
            if (validateSimulation(all_bodies)) {
                printf("Final energy: %.6f\n", computeTotalEnergy(all_bodies));
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    /* ---- Cleanup ------------------------------------------------- */
    cudaFree(d_all);
    if (local_n > 0) cudaFree(d_local);
    MPI_Finalize();

    return 0;
}
