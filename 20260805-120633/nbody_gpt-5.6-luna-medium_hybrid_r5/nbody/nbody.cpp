#include <mpi.h>
#include <omp.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos, vel; };

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void advanceKernel(double* px, double* py, double* pz,
                              double* vx, double* vy, double* vz,
                              const double* allX, const double* allY, const double* allZ,
                              int localCount, int globalCount) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= localCount) return;
    const double ix = px[i], iy = py[i], iz = pz[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    // Keep the source loop and accumulation order equivalent to the reference solver.
    for (int j = 0; j < globalCount; ++j) {
        const double dx = allX[j] - ix;
        const double dy = allY[j] - iy;
        const double dz = allZ[j] - iz;
        const double inv = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        const double inv3 = inv * inv * inv;
        fx += dx * inv3;
        fy += dy * inv3;
        fz += dz * inv3;
    }
    vx[i] += DT * fx;
    vy[i] += DT * fy;
    vz[i] += DT * fz;
    px[i] += vx[i] * DT;
    py[i] += vy[i] * DT;
    pz[i] += vz[i] * DT;
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

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n", name);
    std::printf("  -n <num>     Number of bodies (default: 1024)\n");
    std::printf("  -s <num>     Number of simulation steps (default: 10)\n");
    std::printf("  -v           Enable validation\n  -r           Print results\n  -h           Show this help\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int n = 1024, steps = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n < 1 || steps < 0) { if (rank == 0) std::fprintf(stderr, "Invalid size or step count\n"); MPI_Finalize(); return 1; }

    std::vector<Body> bodies(n);
    if (rank == 0) randomizeBodies(bodies);
    MPI_Bcast(bodies.data(), n * 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const int begin = (n * rank) / ranks;
    const int end = (n * (rank + 1)) / ranks;
    const int localN = end - begin;
    const int allocN = std::max(1, localN); // CUDA does not accept zero-byte allocations.
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = (n * (r + 1)) / ranks - (n * r) / ranks;
        displacements[r] = (n * r) / ranks;
    }

    std::vector<double> posX(n), posY(n), posZ(n), velX(localN), velY(localN), velZ(localN);
    std::vector<double> localX(localN), localY(localN), localZ(localN);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) { posX[i] = bodies[i].pos.x; posY[i] = bodies[i].pos.y; posZ[i] = bodies[i].pos.z; }
    #pragma omp parallel for
    for (int i = 0; i < localN; ++i) {
        velX[i] = bodies[begin + i].vel.x; velY[i] = bodies[begin + i].vel.y; velZ[i] = bodies[begin + i].vel.z;
        localX[i] = posX[begin + i]; localY[i] = posY[begin + i]; localZ[i] = posZ[begin + i];
    }

    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) { std::fprintf(stderr, "No CUDA device available on MPI rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0; MPI_Comm_rank(localComm, &localRank); MPI_Comm_free(&localComm);
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");

    double *dX, *dY, *dZ, *dVX, *dVY, *dVZ, *dAllX, *dAllY, *dAllZ;
    cudaCheck(cudaMalloc(&dX, allocN * sizeof(double)), "cudaMalloc");
    cudaCheck(cudaMalloc(&dY, allocN * sizeof(double)), "cudaMalloc");
    cudaCheck(cudaMalloc(&dZ, allocN * sizeof(double)), "cudaMalloc");
    cudaCheck(cudaMalloc(&dVX, allocN * sizeof(double)), "cudaMalloc");
    cudaCheck(cudaMalloc(&dVY, allocN * sizeof(double)), "cudaMalloc");
    cudaCheck(cudaMalloc(&dVZ, allocN * sizeof(double)), "cudaMalloc");
    cudaCheck(cudaMalloc(&dAllX, n * sizeof(double)), "cudaMalloc");
    cudaCheck(cudaMalloc(&dAllY, n * sizeof(double)), "cudaMalloc");
    cudaCheck(cudaMalloc(&dAllZ, n * sizeof(double)), "cudaMalloc");
    cudaCheck(cudaMemcpy(dX, localX.data(), localN*sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy");
    cudaCheck(cudaMemcpy(dY, localY.data(), localN*sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy");
    cudaCheck(cudaMemcpy(dZ, localZ.data(), localN*sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy");
    cudaCheck(cudaMemcpy(dVX, velX.data(), localN*sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy");
    cudaCheck(cudaMemcpy(dVY, velY.data(), localN*sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy");
    cudaCheck(cudaMemcpy(dVZ, velZ.data(), localN*sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy");

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < steps; ++step) {
        // Exchange each coordinate plane separately so the GPU can consume contiguous SoA arrays.
        MPI_Allgatherv(localY.data(), localN, MPI_DOUBLE, posY.data(), counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(localZ.data(), localN, MPI_DOUBLE, posZ.data(), counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(dAllX, posX.data(), n*sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy");
        cudaCheck(cudaMemcpy(dAllY, posY.data(), n*sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy");
        cudaCheck(cudaMemcpy(dAllZ, posZ.data(), n*sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy");
        advanceKernel<<<(localN + 255) / 256, 256>>>(dX,dY,dZ,dVX,dVY,dVZ,dAllX,dAllY,dAllZ,localN,n);
        cudaCheck(cudaGetLastError(), "advanceKernel");
        cudaCheck(cudaDeviceSynchronize(), "advanceKernel synchronize");
        cudaCheck(cudaMemcpy(localX.data(), dX, localN*sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
        cudaCheck(cudaMemcpy(localY.data(), dY, localN*sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
        cudaCheck(cudaMemcpy(localZ.data(), dZ, localN*sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
    }
    MPI_Allgatherv(localX.data(), localN, MPI_DOUBLE, posX.data(), counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(localY.data(), localN, MPI_DOUBLE, posY.data(), counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(localZ.data(), localN, MPI_DOUBLE, posZ.data(), counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    cudaCheck(cudaMemcpy(velX.data(), dVX, localN*sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
    cudaCheck(cudaMemcpy(velY.data(), dVY, localN*sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
    cudaCheck(cudaMemcpy(velZ.data(), dVZ, localN*sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy");
    const auto endTime = std::chrono::high_resolution_clock::now();
    if (rank == 0) std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\nSimulation time: %ld ms\n", n, steps, validate ? "enabled" : "disabled", (long)std::chrono::duration_cast<std::chrono::milliseconds>(endTime-start).count());

    std::vector<double> allVX(n), allVY(n), allVZ(n);
    MPI_Allgatherv(velX.data(), localN, MPI_DOUBLE, allVX.data(), counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(velY.data(), localN, MPI_DOUBLE, allVY.data(), counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(velZ.data(), localN, MPI_DOUBLE, allVZ.data(), counts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    if (rank == 0) {
        std::vector<double> result; result.reserve(6*n);
        for (int i=0;i<n;++i) { result.insert(result.end(), {posX[i],posY[i],posZ[i],allVX[i],allVY[i],allVZ[i]}); }
        if (printResults) print_results(result, "Bodies");
        if (validate) {
            bool ok = true; double energy = 0.0;
            #pragma omp parallel for reduction(+:energy)
            for (int i=0;i<n;++i) energy += 0.5*(allVX[i]*allVX[i]+allVY[i]*allVY[i]+allVZ[i]*allVZ[i]);
            for (int i=0;i<n;++i) for (int j=i+1;j<n;++j) { const double dx=posX[j]-posX[i],dy=posY[j]-posY[i],dz=posZ[j]-posZ[i]; energy -= 1.0/std::sqrt(dx*dx+dy*dy+dz*dz+SOFTENING); }
            for (int i=0;i<n;++i) if (!std::isfinite(posX[i])||!std::isfinite(posY[i])||!std::isfinite(posZ[i])||!std::isfinite(allVX[i])||!std::isfinite(allVY[i])||!std::isfinite(allVZ[i])||std::abs(posX[i])>1e6||std::abs(posY[i])>1e6||std::abs(posZ[i])>1e6||std::abs(allVX[i])>1e6||std::abs(allVY[i])>1e6||std::abs(allVZ[i])>1e6) ok=false;
            std::printf("Final energy: %.6f\nValidation: %s\n", energy, ok ? "PASSED" : "FAILED");
            if (!ok) { cudaFree(dX); MPI_Finalize(); return 1; }
        }
    }
    cudaFree(dX); cudaFree(dY); cudaFree(dZ); cudaFree(dVX); cudaFree(dVY); cudaFree(dVZ); cudaFree(dAllX); cudaFree(dAllY); cudaFree(dAllZ);
    MPI_Finalize(); return 0;
}
