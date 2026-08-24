#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept
        : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// ---------------------------------------------------------------------------
// CUDA kernel: compute gravitational forces on local bodies against all
// bodies (all-pairs).  Each thread handles one local body index `li`;
// the corresponding global index is `gid[li]`.  Forces accumulate into
// `d_force` (SoA, 3 * n_local).
//
//   d_all_pos  : SoA positions of ALL bodies  (3 * n_global)
//   d_gid      : global indices of local bodies (n_local)
//   d_force    : output force accumulator       (3 * n_local)
// ---------------------------------------------------------------------------
__global__ void computeForcesKernel(const double* d_all_pos,
                                    const int*     d_gid,
                                    double*        d_force,
                                    const int n_local, const int n_global,
                                    const double softening) {
    const int li = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + threadIdx.x;
    if (li >= n_local) return;

    const int gi = d_gid[li];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    const double pi_x = d_all_pos[gi * 3 + 0];
    const double pi_y = d_all_pos[gi * 3 + 1];
    const double pi_z = d_all_pos[gi * 3 + 2];

    for (int j = 0; j < n_global; ++j) {
        const double dx = d_all_pos[j * 3 + 0] - pi_x;
        const double dy = d_all_pos[j * 3 + 1] - pi_y;
        const double dz = d_all_pos[j * 3 + 2] - pi_z;
        const double distSqr = dx * dx + dy * dy + dz * dz + softening;
        const double invDist = 1.0 / sqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    d_force[li * 3 + 0] = Fx;
    d_force[li * 3 + 1] = Fy;
    d_force[li * 3 + 2] = Fz;
}

// ---------------------------------------------------------------------------
// CUDA kernel: integrate positions from velocities for local bodies
// Writes to local index `li` (d_pos is sized n_local * 3).
// ---------------------------------------------------------------------------
__global__ void integrateKernel(double* d_pos, const double* d_vel,
                                const int n_local, const double dt) {
    const int li = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + threadIdx.x;
    if (li >= n_local) return;

    d_pos[li * 3 + 0] += d_vel[li * 3 + 0] * dt;
    d_pos[li * 3 + 1] += d_vel[li * 3 + 1] * dt;
    d_pos[li * 3 + 2] += d_vel[li * 3 + 2] * dt;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

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

// Flatten bodies to SoA arrays (size 3*n each).
inline void flattenBodies(const std::vector<Body>& bodies,
                          std::vector<double>& pos, std::vector<double>& vel) {
    const int n = static_cast<int>(bodies.size());
    pos.resize(n * 3);
    vel.resize(n * 3);
    for (int i = 0; i < n; ++i) {
        pos[i * 3 + 0] = bodies[i].pos.x;
        pos[i * 3 + 1] = bodies[i].pos.y;
        pos[i * 3 + 2] = bodies[i].pos.z;
        vel[i * 3 + 0] = bodies[i].vel.x;
        vel[i * 3 + 1] = bodies[i].vel.y;
        vel[i * 3 + 2] = bodies[i].vel.z;
    }
}

inline void unflattenBodies(const std::vector<double>& pos,
                            const std::vector<double>& vel,
                            std::vector<Body>& bodies) {
    const int n = static_cast<int>(bodies.size());
    for (int i = 0; i < n; ++i) {
        bodies[i].pos.x = pos[i * 3 + 0];
        bodies[i].pos.y = pos[i * 3 + 1];
        bodies[i].pos.z = pos[i * 3 + 2];
        bodies[i].vel.x = vel[i * 3 + 0];
        bodies[i].vel.y = vel[i * 3 + 1];
        bodies[i].vel.z = vel[i * 3 + 2];
    }
}

// ---------------------------------------------------------------------------
// Compute total energy: OpenMP for kinetic, OpenMP for potential
// ---------------------------------------------------------------------------
double computeTotalEnergy(const std::vector<Body>& bodies) {
    const int n = static_cast<int>(bodies.size());
    double energy = 0.0;

    // Kinetic energy (unit mass)
    #pragma omp parallel for reduction(+ : energy) schedule(static)
    for (int i = 0; i < n; ++i) {
        energy += 0.5 * (bodies[i].vel.x * bodies[i].vel.x +
                         bodies[i].vel.y * bodies[i].vel.y +
                         bodies[i].vel.z * bodies[i].vel.z);
    }

    // Potential energy (unit mass for all bodies)
    double pot = 0.0;
    #pragma omp parallel for reduction(+ : pot) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            pot -= 1.0 / dist;
        }
    }
    energy += pot;

    return energy;
}

// ---------------------------------------------------------------------------
// Validate that simulation produces finite, reasonable values (OpenMP)
// ---------------------------------------------------------------------------
bool validateSimulation(const std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    bool ok = true;

    #pragma omp parallel for
    for (size_t idx = 0; idx < n; ++idx) {
        const auto& body = bodies[idx];
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) ||
            !std::isfinite(body.pos.z) || !std::isfinite(body.vel.x) ||
            !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            #pragma omp critical
            {
                printf("Validation failed: found NaN or Inf value in body state\n");
            }
            ok = false;
        }
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos ||
            std::abs(body.pos.z) > maxPos) {
            #pragma omp critical
            {
                printf("Validation failed: body position exceeds reasonable bounds\n");
            }
            ok = false;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel ||
            std::abs(body.vel.z) > maxVel) {
            #pragma omp critical
            {
                printf("Validation failed: body velocity exceeds reasonable bounds\n");
            }
            ok = false;
        }
    }
    return ok;
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
    // ------------------------------------------------------------------
    // MPI initialisation
    // ------------------------------------------------------------------
    int mpi_size, mpi_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // ------------------------------------------------------------------
    // Parse command line arguments (every rank parses the same args)
    // ------------------------------------------------------------------
    int numBodies = 1024;
    int numSteps  = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numBodies = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            numSteps  = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // ------------------------------------------------------------------
    // Print banner (rank 0 only)
    // ------------------------------------------------------------------
    if (mpi_rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d  |  OpenMP threads: %d\n",
               mpi_size, omp_get_max_threads());
    }

    // ------------------------------------------------------------------
    // Domain decomposition: each MPI rank owns a contiguous chunk of
    // bodies.  n_local is the number of bodies owned by this rank.
    // ------------------------------------------------------------------
    const int n_global = numBodies;
    const int n_local  = (n_global + mpi_size - 1) / mpi_size;
    const int local_start = mpi_rank * n_local;
    const int local_end   = std::min(local_start + n_local, n_global);
    const int n_owned = local_end - local_start;

    // Global arrays (only rank 0 initialises bodies).
    std::vector<double> all_pos(n_global * 3, 0.0);
    std::vector<double> all_vel(n_global * 3, 0.0);

    // Initialise bodies on rank 0, then broadcast.
    if (mpi_rank == 0) {
        std::vector<Body> bodies(n_global);
        randomizeBodies(bodies);
        flattenBodies(bodies, all_pos, all_vel);
    }
    MPI_Bcast(all_pos.data(), n_global * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(all_vel.data(), n_global * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Build global-index map for local bodies on this rank.
    std::vector<int> gid(n_owned);
    for (int i = 0; i < n_owned; ++i) gid[i] = local_start + i;

    // Local copies of positions / velocities for this rank's bodies.
    std::vector<double> local_pos(n_owned * 3);
    std::vector<double> local_vel(n_owned * 3);
    for (int i = 0; i < n_owned; ++i) {
        local_pos[i * 3 + 0] = all_pos[(local_start + i) * 3 + 0];
        local_pos[i * 3 + 1] = all_pos[(local_start + i) * 3 + 1];
        local_pos[i * 3 + 2] = all_pos[(local_start + i) * 3 + 2];
        local_vel[i * 3 + 0] = all_vel[(local_start + i) * 3 + 0];
        local_vel[i * 3 + 1] = all_vel[(local_start + i) * 3 + 1];
        local_vel[i * 3 + 2] = all_vel[(local_start + i) * 3 + 2];
    }

    // ------------------------------------------------------------------
    // Device allocations
    // ------------------------------------------------------------------
    // d_all_pos : full positions (read-only during kernel)
    double* d_all_pos = nullptr;
    cudaMalloc(&d_all_pos, n_global * 3 * sizeof(double));
    cudaMemcpy(d_all_pos, all_pos.data(), n_global * 3 * sizeof(double),
               cudaMemcpyHostToDevice);

    // d_gid : global index map
    int* d_gid = nullptr;
    cudaMalloc(&d_gid, n_owned * sizeof(int));
    cudaMemcpy(d_gid, gid.data(), n_owned * sizeof(int),
               cudaMemcpyHostToDevice);

    // d_local_pos : local positions (updated by integrate kernel)
    double* d_local_pos = nullptr;
    cudaMalloc(&d_local_pos, n_owned * 3 * sizeof(double));
    cudaMemcpy(d_local_pos, local_pos.data(), n_owned * 3 * sizeof(double),
               cudaMemcpyHostToDevice);

    // d_local_vel : local velocities (updated by force kernel)
    double* d_local_vel = nullptr;
    cudaMalloc(&d_local_vel, n_owned * 3 * sizeof(double));
    cudaMemcpy(d_local_vel, local_vel.data(), n_owned * 3 * sizeof(double),
               cudaMemcpyHostToDevice);

    // d_force : temporary force accumulator
    double* d_force = nullptr;
    cudaMalloc(&d_force, n_owned * 3 * sizeof(double));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ------------------------------------------------------------------
    // Simulation loop
    // ------------------------------------------------------------------
    auto start = std::chrono::high_resolution_clock::now();

    // Pre-allocate host force buffer.
    std::vector<double> h_force(n_owned * 3);
    // Pre-compute MPI Allgatherv parameters (displacements in units of MPI_DOUBLE).
    std::vector<int> recvcounts(mpi_size, n_local * 3);
    std::vector<int> displs(mpi_size);
    for (int r = 0; r < mpi_size; ++r) {
        displs[r] = std::min(r * n_local * 3, n_global * 3);
    }

    for (int step = 0; step < numSteps; ++step) {
        // 1. Allgather positions: every rank sends its local positions
        //    so that every rank has the complete position state.
        MPI_Allgatherv(local_pos.data(), n_owned * 3, MPI_DOUBLE,
                       all_pos.data(), recvcounts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // 2. Upload full positions to GPU.
        cudaMemcpyAsync(d_all_pos, all_pos.data(), n_global * 3 * sizeof(double),
                        cudaMemcpyHostToDevice, stream);

        // 3. CUDA: compute forces on local bodies against all bodies.
        const int threads = 256;
        const int blocks  = (n_owned + threads - 1) / threads;
        computeForcesKernel<<<blocks, threads, 0, stream>>>(
            d_all_pos, d_gid, d_force, n_owned, n_global, SOFTENING);

        // 4. Download forces and update velocities on host with OpenMP.
        cudaMemcpyAsync(h_force.data(), d_force, n_owned * 3 * sizeof(double),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n_owned * 3; ++i) {
            local_vel[i] += DT * h_force[i];
        }

        // 5. Upload updated velocities back to GPU.
        cudaMemcpyAsync(d_local_vel, local_vel.data(),
                        n_owned * 3 * sizeof(double),
                        cudaMemcpyHostToDevice, stream);

        // 6. CUDA: integrate positions for local bodies.
        integrateKernel<<<blocks, threads, 0, stream>>>(
            d_local_pos, d_local_vel, n_owned, DT);

        // 7. Download updated local positions.
        cudaMemcpyAsync(local_pos.data(), d_local_pos,
                        n_owned * 3 * sizeof(double),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = static_cast<long long>(duration.count());
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG_INT,
               MPI_MAX, 0, MPI_COMM_WORLD);

    // ------------------------------------------------------------------
    // Gather final state: collect local positions and velocities into
    // the global arrays on every rank.
    // ------------------------------------------------------------------
    MPI_Allgatherv(local_pos.data(), n_owned * 3, MPI_DOUBLE,
                   all_pos.data(), recvcounts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(local_vel.data(), n_owned * 3, MPI_DOUBLE,
                   all_vel.data(), recvcounts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    // Reconstruct body vector from global arrays.
    std::vector<Body> bodies(n_global);
    unflattenBodies(all_pos, all_vel, bodies);

    // ------------------------------------------------------------------
    // Results output (rank 0 only)
    // ------------------------------------------------------------------
    if (mpi_rank == 0) {
        printf("Simulation time: %lld ms\n", max_duration_ms);

        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(n_global * 6);
            for (const auto& body : bodies) {
                bodyData.push_back(body.pos.x);
                bodyData.push_back(body.pos.y);
                bodyData.push_back(body.pos.z);
                bodyData.push_back(body.vel.x);
                bodyData.push_back(body.vel.y);
                bodyData.push_back(body.vel.z);
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
            }
        }
    }

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    cudaStreamDestroy(stream);
    cudaFree(d_all_pos);
    cudaFree(d_gid);
    cudaFree(d_local_pos);
    cudaFree(d_local_vel);
    cudaFree(d_force);
    MPI_Finalize();

    return 0;
}
