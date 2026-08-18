#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static void checkCuda(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA %s failed: %s\n", rank, operation,
                     cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
    }
}

// Each rank updates a disjoint range of targets.  The source positions are
// replicated, so this preserves the original direct all-pairs algorithm.
__global__ void advanceBodies(double* __restrict__ x,
                              double* __restrict__ y,
                              double* __restrict__ z,
                              double* __restrict__ vx,
                              double* __restrict__ vy,
                              double* __restrict__ vz,
                              const int first, const int localCount,
                              const int n) {
    extern __shared__ double tile[];
    double* const tx = tile;
    double* const ty = tx + BLOCK_SIZE;
    double* const tz = ty + BLOCK_SIZE;

    const int localI = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = first + localI;
    const bool active = localI < localCount;
    const double xi = active ? x[i] : 0.0;
    const double yi = active ? y[i] : 0.0;
    const double zi = active ? z[i] : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int base = 0; base < n; base += BLOCK_SIZE) {
        const int source = base + threadIdx.x;
        if (source < n) {
            tx[threadIdx.x] = x[source];
            ty[threadIdx.x] = y[source];
            tz[threadIdx.x] = z[source];
        }
        __syncthreads();

        const int count = min(BLOCK_SIZE, n - base);
        if (active) {
            // The loop order is ascending j, as in the serial reference.
            #pragma unroll 4
            for (int k = 0; k < count; ++k) {
                const double dx = tx[k] - xi;
                const double dy = ty[k] - yi;
                const double dz = tz[k] - zi;
                const double invDist = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        const double newVx = vx[i] + DT * fx;
        const double newVy = vy[i] + DT * fy;
        const double newVz = vz[i] + DT * fz;
        vx[i] = newVx;
        vy[i] = newVy;
        vz[i] = newVz;
        // Combining force and integration is safe: every target is independent.
        x[i] = xi + DT * newVx;
        y[i] = yi + DT * newVy;
        z[i] = zi + DT * newVz;
    }
}

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 1.0;
    }
}

double computeTotalEnergy(const std::vector<double>& x, const std::vector<double>& y,
                          const std::vector<double>& z, const std::vector<double>& vx,
                          const std::vector<double>& vy, const std::vector<double>& vz) {
    const int n = static_cast<int>(x.size());
    double energy = 0.0;
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (int i = 0; i < n; ++i) {
        energy += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
        for (int j = i + 1; j < n; ++j) {
            const double dx = x[j] - x[i], dy = y[j] - y[i], dz = z[j] - z[i];
            energy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    return energy;
}

bool validateSimulation(const std::vector<double>& x, const std::vector<double>& y,
                        const std::vector<double>& z, const std::vector<double>& vx,
                        const std::vector<double>& vy, const std::vector<double>& vz) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        const double values[] = {x[i], y[i], z[i], vx[i], vy[i], vz[i]};
        for (double value : values) valid &= std::isfinite(value) && std::abs(value) <= 1e6;
    }
    return valid != 0;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show help\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);

    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
    int parseOK = 1;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) numBodies = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) numSteps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else parseOK = 0;
    }
    if (numBodies <= 0 || numSteps < 0) parseOK = 0;
    if (!parseOK) {
        if (rank == 0) { std::fprintf(stderr, "Invalid command line arguments\n"); printUsage(argv[0]); }
        MPI_Finalize(); return 1;
    }

    int devices = 0;
    checkCuda(cudaGetDeviceCount(&devices), "device discovery", rank);
    if (devices == 0) { if (rank == 0) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    // Use the node-local rank: global ranks are not suitable for multi-node GPU mapping.
    checkCuda(cudaSetDevice(localRank % devices), "device selection", rank);

    const int base = numBodies / ranks, remainder = numBodies % ranks;
    const int localCount = base + (rank < remainder ? 1 : 0);
    const int first = rank * base + (rank < remainder ? rank : remainder);
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) { counts[r] = base + (r < remainder ? 1 : 0); offsets[r] = r * base + (r < remainder ? r : remainder); }

    if (rank == 0) {
        std::printf("N-Body Simulation (MPI ranks: %d, OpenMP threads/rank: %d, CUDA: enabled)\n", ranks, omp_get_max_threads());
        std::printf("Number of bodies: %d\nNumber of steps: %d\nValidation: %s\n", numBodies, numSteps, validate ? "enabled" : "disabled");
    }

    std::vector<Body> initial(numBodies);
    randomizeBodies(initial); // deliberately serial: matches the reference seed sequence
    std::vector<double> x(numBodies), y(numBodies), z(numBodies), vx(numBodies), vy(numBodies), vz(numBodies);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numBodies; ++i) { x[i] = initial[i].pos.x; y[i] = initial[i].pos.y; z[i] = initial[i].pos.z; vx[i] = initial[i].vel.x; vy[i] = initial[i].vel.y; vz[i] = initial[i].vel.z; }

    double *dx, *dy, *dz, *dvx, *dvy, *dvz;
    const size_t bytes = static_cast<size_t>(numBodies) * sizeof(double);
    checkCuda(cudaMalloc(&dx, bytes), "allocation", rank); checkCuda(cudaMalloc(&dy, bytes), "allocation", rank); checkCuda(cudaMalloc(&dz, bytes), "allocation", rank);
    checkCuda(cudaMalloc(&dvx, bytes), "allocation", rank); checkCuda(cudaMalloc(&dvy, bytes), "allocation", rank); checkCuda(cudaMalloc(&dvz, bytes), "allocation", rank);
    auto uploadPositions = [&] { checkCuda(cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice), "position upload", rank); checkCuda(cudaMemcpy(dy, y.data(), bytes, cudaMemcpyHostToDevice), "position upload", rank); checkCuda(cudaMemcpy(dz, z.data(), bytes, cudaMemcpyHostToDevice), "position upload", rank); };
    uploadPositions();
    checkCuda(cudaMemcpy(dvx, vx.data(), bytes, cudaMemcpyHostToDevice), "velocity upload", rank);
    checkCuda(cudaMemcpy(dvy, vy.data(), bytes, cudaMemcpyHostToDevice), "velocity upload", rank);
    checkCuda(cudaMemcpy(dvz, vz.data(), bytes, cudaMemcpyHostToDevice), "velocity upload", rank);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    const int blocks = (localCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int step = 0; step < numSteps; ++step) {
        if (localCount != 0) {
            advanceBodies<<<blocks, BLOCK_SIZE, 3 * BLOCK_SIZE * sizeof(double)>>>(dx, dy, dz, dvx, dvy, dvz, first, localCount, numBodies);
            checkCuda(cudaGetLastError(), "kernel launch", rank);
            checkCuda(cudaDeviceSynchronize(), "kernel completion", rank);
        }
        const size_t localBytes = static_cast<size_t>(localCount) * sizeof(double);
        checkCuda(cudaMemcpy(x.data() + first, dx + first, localBytes, cudaMemcpyDeviceToHost), "state download", rank);
        checkCuda(cudaMemcpy(y.data() + first, dy + first, localBytes, cudaMemcpyDeviceToHost), "state download", rank);
        checkCuda(cudaMemcpy(z.data() + first, dz + first, localBytes, cudaMemcpyDeviceToHost), "state download", rank);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, y.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, z.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        if (step + 1 < numSteps) uploadPositions();
    }
    const auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0) std::printf("Simulation time: %ld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    // Velocities are target-local state and are only needed globally after the final step.
    const size_t localBytes = static_cast<size_t>(localCount) * sizeof(double);
    checkCuda(cudaMemcpy(vx.data() + first, dvx + first, localBytes, cudaMemcpyDeviceToHost), "final velocity download", rank);
    checkCuda(cudaMemcpy(vy.data() + first, dvy + first, localBytes, cudaMemcpyDeviceToHost), "final velocity download", rank);
    checkCuda(cudaMemcpy(vz.data() + first, dvz + first, localBytes, cudaMemcpyDeviceToHost), "final velocity download", rank);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, vx.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, vy.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, vz.data(), counts.data(), offsets.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    if (rank == 0 && printResults) {
        std::vector<double> bodyData; bodyData.reserve(static_cast<size_t>(numBodies) * 6);
        for (int i = 0; i < numBodies; ++i) { bodyData.push_back(x[i]); bodyData.push_back(y[i]); bodyData.push_back(z[i]); bodyData.push_back(vx[i]); bodyData.push_back(vy[i]); bodyData.push_back(vz[i]); }
        print_results(bodyData, "Bodies");
    }
    int returnCode = 0;
    if (rank == 0 && validate) {
        std::printf("Validating simulation results...\n");
        if (validateSimulation(x, y, z, vx, vy, vz)) { std::printf("Final energy: %.6f\nValidation: PASSED\n", computeTotalEnergy(x, y, z, vx, vy, vz)); }
        else { std::printf("Validation: FAILED\n"); returnCode = 1; }
    }
    MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cudaFree(dx); cudaFree(dy); cudaFree(dz); cudaFree(dvx); cudaFree(dvy); cudaFree(dvz);
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return returnCode;
}
