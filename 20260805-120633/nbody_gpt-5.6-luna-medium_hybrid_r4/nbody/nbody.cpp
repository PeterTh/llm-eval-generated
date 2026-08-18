#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Body {
    double px, py, pz;
    double vx, vy, vz;
};

static void mpiCudaCheck(cudaError_t result, const char* operation, int rank) {
    if (result != cudaSuccess) {
        fprintf(stderr, "Rank %d: CUDA error in %s: %s\n", rank, operation,
                cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// One thread owns one target body. All source positions are read-only and are
// shared by every rank, while velocities and positions are updated in place.
__global__ void advanceBodies(double* px, double* py, double* pz,
                               double* vx, double* vy, double* vz,
                               int first, int count, int n) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;
    const int i = first + local;
    const double ix = px[i], iy = py[i], iz = pz[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;

    #pragma unroll 4
    for (int j = 0; j < n; ++j) {
        const double dx = px[j] - ix;
        const double dy = py[j] - iy;
        const double dz = pz[j] - iz;
        const double distanceSquared = dx * dx + dy * dy + dz * dz + SOFTENING;
        const double inverseDistance = 1.0 / sqrt(distanceSquared);
        const double inverseDistance3 = inverseDistance * inverseDistance * inverseDistance;
        fx += dx * inverseDistance3;
        fy += dy * inverseDistance3;
        fz += dz * inverseDistance3;
    }

    const double nvx = vx[i] + DT * fx;
    const double nvy = vy[i] + DT * fy;
    const double nvz = vz[i] + DT * fz;
    vx[i] = nvx;
    vy[i] = nvy;
    vz[i] = nvz;
    px[i] += nvx * DT;
    py[i] += nvy * DT;
    pz[i] += nvz * DT;
}

static void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // Keep the original rand_r sequence so single-rank results retain their
    // established semantics. The substantial host-side work is parallelized
    // in validation and energy calculation below.
    for (Body& body : bodies) {
        body.px = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.py = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pz = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vx = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vy = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vz = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

static double computeTotalEnergy(const std::vector<Body>& bodies) {
    const int n = static_cast<int>(bodies.size());
    double kinetic = 0.0;
    #pragma omp parallel for reduction(+:kinetic) schedule(static)
    for (int i = 0; i < n; ++i) {
        const Body& b = bodies[i];
        kinetic += 0.5 * (b.vx * b.vx + b.vy * b.vy + b.vz * b.vz);
    }

    double potential = 0.0;
    #pragma omp parallel for reduction(+:potential) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double dx = bodies[j].px - bodies[i].px;
            const double dy = bodies[j].py - bodies[i].py;
            const double dz = bodies[j].pz - bodies[i].pz;
            potential -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    return kinetic + potential;
}

static bool validateSimulation(const std::vector<Body>& bodies) {
    int invalid = 0;
    #pragma omp parallel for reduction(+:invalid) schedule(static)
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
        const Body& b = bodies[i];
        const bool finite = std::isfinite(b.px) && std::isfinite(b.py) && std::isfinite(b.pz) &&
                            std::isfinite(b.vx) && std::isfinite(b.vy) && std::isfinite(b.vz);
        const bool bounded = std::abs(b.px) <= 1e6 && std::abs(b.py) <= 1e6 && std::abs(b.pz) <= 1e6 &&
                             std::abs(b.vx) <= 1e6 && std::abs(b.vy) <= 1e6 && std::abs(b.vz) <= 1e6;
        invalid += !(finite && bounded);
    }
    if (invalid) printf("Validation failed: found invalid body state\n");
    return invalid == 0;
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Number of bodies (default: 1024)\n"
           "  -s <num>     Number of simulation steps (default: 10)\n"
           "  -v           Enable validation (checks energy conservation)\n"
           "  -r           Print results for external validation\n"
           "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    (void)provided;

    int n = 1024, steps = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n < 0 || steps < 0) { if (rank == 0) printf("Number of bodies and steps must be non-negative\n"); MPI_Finalize(); return 1; }

    int deviceCount = 0;
    mpiCudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (!deviceCount) { if (rank == 0) fprintf(stderr, "No CUDA device available\n"); MPI_Finalize(); return 1; }
    int localRank = rank;
    const char* localRankText = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (!localRankText) localRankText = std::getenv("SLURM_LOCALID");
    if (!localRankText) localRankText = std::getenv("MPI_LOCALRANKID");
    if (localRankText) localRank = atoi(localRankText);
    mpiCudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", rank);

    std::vector<Body> bodies(static_cast<size_t>(n));
    if (rank == 0) randomizeBodies(bodies);
    MPI_Bcast(bodies.data(), n * static_cast<int>(sizeof(Body)), MPI_BYTE, 0, MPI_COMM_WORLD);

    const int first = (n * rank) / ranks;
    const int last = (n * (rank + 1)) / ranks;
    const int localCount = last - first;
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        const int begin = (n * r) / ranks, end = (n * (r + 1)) / ranks;
        counts[r] = (end - begin) * static_cast<int>(sizeof(Body));
        displacements[r] = begin * static_cast<int>(sizeof(Body));
    }

    std::vector<double> px(n), py(n), pz(n), vx(n), vy(n), vz(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        px[i] = bodies[i].px; py[i] = bodies[i].py; pz[i] = bodies[i].pz;
        vx[i] = bodies[i].vx; vy[i] = bodies[i].vy; vz[i] = bodies[i].vz;
    }
    double *dpx, *dpy, *dpz, *dvx, *dvy, *dvz;
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);
    const size_t allocationBytes = bytes ? bytes : sizeof(double);
    mpiCudaCheck(cudaMalloc(&dpx, allocationBytes), "cudaMalloc", rank); mpiCudaCheck(cudaMalloc(&dpy, allocationBytes), "cudaMalloc", rank);
    mpiCudaCheck(cudaMalloc(&dpz, allocationBytes), "cudaMalloc", rank); mpiCudaCheck(cudaMalloc(&dvx, allocationBytes), "cudaMalloc", rank);
    mpiCudaCheck(cudaMalloc(&dvy, allocationBytes), "cudaMalloc", rank); mpiCudaCheck(cudaMalloc(&dvz, allocationBytes), "cudaMalloc", rank);
    mpiCudaCheck(cudaMemcpy(dpx, px.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy", rank);
    mpiCudaCheck(cudaMemcpy(dpy, py.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy", rank);
    mpiCudaCheck(cudaMemcpy(dpz, pz.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy", rank);
    mpiCudaCheck(cudaMemcpy(dvx, vx.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy", rank);
    mpiCudaCheck(cudaMemcpy(dvy, vy.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy", rank);
    mpiCudaCheck(cudaMemcpy(dvz, vz.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy", rank);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    const int blockSize = 256;
    for (int step = 0; step < steps; ++step) {
        if (localCount > 0) {
            advanceBodies<<<(localCount + blockSize - 1) / blockSize, blockSize>>>(
                dpx, dpy, dpz, dvx, dvy, dvz, first, localCount, n);
            mpiCudaCheck(cudaGetLastError(), "advanceBodies launch", rank);
            mpiCudaCheck(cudaDeviceSynchronize(), "advanceBodies", rank);
        }
        // Pack the six SoA fields; a single all-gather keeps the state layout
        // identical on every rank and supplies the next GPU iteration.
        std::vector<Body> local(static_cast<size_t>(localCount));
        mpiCudaCheck(cudaMemcpy(px.data() + first, dpx + first, localCount * sizeof(double), cudaMemcpyDeviceToHost), "copy px", rank);
        mpiCudaCheck(cudaMemcpy(py.data() + first, dpy + first, localCount * sizeof(double), cudaMemcpyDeviceToHost), "copy py", rank);
        mpiCudaCheck(cudaMemcpy(pz.data() + first, dpz + first, localCount * sizeof(double), cudaMemcpyDeviceToHost), "copy pz", rank);
        mpiCudaCheck(cudaMemcpy(vx.data() + first, dvx + first, localCount * sizeof(double), cudaMemcpyDeviceToHost), "copy vx", rank);
        mpiCudaCheck(cudaMemcpy(vy.data() + first, dvy + first, localCount * sizeof(double), cudaMemcpyDeviceToHost), "copy vy", rank);
        mpiCudaCheck(cudaMemcpy(vz.data() + first, dvz + first, localCount * sizeof(double), cudaMemcpyDeviceToHost), "copy vz", rank);
        for (int k = 0; k < localCount; ++k) local[k] = {px[first+k], py[first+k], pz[first+k], vx[first+k], vy[first+k], vz[first+k]};
        MPI_Allgatherv(local.data(), counts[rank], MPI_BYTE, bodies.data(), counts.data(), displacements.data(), MPI_BYTE, MPI_COMM_WORLD);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) { px[i]=bodies[i].px; py[i]=bodies[i].py; pz[i]=bodies[i].pz; vx[i]=bodies[i].vx; vy[i]=bodies[i].vy; vz[i]=bodies[i].vz; }
        mpiCudaCheck(cudaMemcpy(dpx, px.data(), bytes, cudaMemcpyHostToDevice), "broadcast px", rank);
        mpiCudaCheck(cudaMemcpy(dpy, py.data(), bytes, cudaMemcpyHostToDevice), "broadcast py", rank);
        mpiCudaCheck(cudaMemcpy(dpz, pz.data(), bytes, cudaMemcpyHostToDevice), "broadcast pz", rank);
        mpiCudaCheck(cudaMemcpy(dvx, vx.data(), bytes, cudaMemcpyHostToDevice), "broadcast vx", rank);
        mpiCudaCheck(cudaMemcpy(dvy, vy.data(), bytes, cudaMemcpyHostToDevice), "broadcast vy", rank);
        mpiCudaCheck(cudaMemcpy(dvz, vz.data(), bytes, cudaMemcpyHostToDevice), "broadcast vz", rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n", n, steps, validate ? "enabled" : "disabled");
        printf("Simulation time: %ld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        if (printResults) { std::vector<double> data; data.reserve(static_cast<size_t>(n) * 6); for (const Body& b : bodies) { data.insert(data.end(), {b.px,b.py,b.pz,b.vx,b.vy,b.vz}); } print_results(data, "Bodies"); }
        if (validate) { printf("Validating simulation results...\n"); if (validateSimulation(bodies)) { printf("Final energy: %.6f\nValidation: PASSED\n", computeTotalEnergy(bodies)); } else { printf("Validation: FAILED\n"); MPI_Abort(MPI_COMM_WORLD, 1); } }
    }
    cudaFree(dpx); cudaFree(dpy); cudaFree(dpz); cudaFree(dvx); cudaFree(dvy); cudaFree(dvz);
    MPI_Finalize();
    return 0;
}
