#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos; Vec3 vel; };

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

__global__ void computeForcesKernel(const double* px, const double* py, const double* pz,
                                    double* vx, double* vy, double* vz,
                                    int n, int start, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int i = start + idx;
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double pix = px[i], piy = py[i], piz = pz[i];
    for (int j = 0; j < n; ++j) {
        double dx = px[j] - pix;
        double dy = py[j] - piy;
        double dz = pz[j] - piz;
        double distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        double invDist = 1.0 / sqrt(distSqr);
        double invDist3 = invDist * invDist * invDist;
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }
    vx[i] += DT * Fx;
    vy[i] += DT * Fy;
    vz[i] += DT * Fz;
}

void computeForces(std::vector<Body>& bodies, int rank, int size) {
    const size_t n = bodies.size();
    int base = n / size;
    int rem = n % size;
    int start = rank * base + (rank < rem ? rank : rem);
    int count = base + (rank < rem ? 1 : 0);

    std::vector<double> px(n), py(n), pz(n), vx(n), vy(n), vz(n);
    for (size_t i = 0; i < n; ++i) { px[i]=bodies[i].pos.x; py[i]=bodies[i].pos.y; pz[i]=bodies[i].pos.z; vx[i]=bodies[i].vel.x; vy[i]=bodies[i].vel.y; vz[i]=bodies[i].vel.z; }

    double *d_px,*d_py,*d_pz,*d_vx,*d_vy,*d_vz;
    cudaMalloc(&d_px, n*sizeof(double)); cudaMalloc(&d_py, n*sizeof(double)); cudaMalloc(&d_pz, n*sizeof(double));
    cudaMalloc(&d_vx, n*sizeof(double)); cudaMalloc(&d_vy, n*sizeof(double)); cudaMalloc(&d_vz, n*sizeof(double));
    cudaMemcpy(d_px, px.data(), n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, py.data(), n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pz, pz.data(), n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, vx.data(), n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, vy.data(), n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, vz.data(), n*sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    computeForcesKernel<<<blocks, threads>>>(d_px,d_py,d_pz,d_vx,d_vy,d_vz,(int)n,start,count);
    cudaDeviceSynchronize();

    cudaMemcpy(vx.data(), d_vx, n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vy.data(), d_vy, n*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vz.data(), d_vz, n*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz); cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);

    // write back owned velocities
    for (int k = 0; k < count; ++k) {
        int i = start + k;
        bodies[i].vel.x = vx[i];
        bodies[i].vel.y = vy[i];
        bodies[i].vel.z = vz[i];
    }

    std::vector<int> cnt(size), dsp(size);
    int off = 0;
    for (int r = 0; r < size; ++r) {
        int c = base + (r < rem ? 1 : 0);
        cnt[r] = c * (int)sizeof(Body);
        dsp[r] = off * (int)sizeof(Body);
        off += c;
    }
    MPI_Allgatherv(bodies.data() + start, count * (int)sizeof(Body), MPI_BYTE,
                   bodies.data(), cnt.data(), dsp.data(), MPI_BYTE, MPI_COMM_WORLD);
}

void integrateBodies(std::vector<Body>& bodies) {
    #pragma omp parallel for
    for (size_t i = 0; i < bodies.size(); ++i) {
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();
    #pragma omp parallel for reduction(+:energy)
    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (bodies[i].vel.x*bodies[i].vel.x + bodies[i].vel.y*bodies[i].vel.y + bodies[i].vel.z*bodies[i].vel.z);
    }
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i+1; j < n; ++j) {
            double dx = bodies[j].pos.x - bodies[i].pos.x;
            double dy = bodies[j].pos.y - bodies[i].pos.y;
            double dz = bodies[j].pos.z - bodies[i].pos.z;
            double dist = std::sqrt(dx*dx+dy*dy+dz*dz+SOFTENING);
            energy -= 1.0 / dist;
        }
    return energy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        const double maxPos = 1e6, maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) { printf("Validation failed: body position exceeds reasonable bounds\n"); return false; }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) { printf("Validation failed: body velocity exceeds reasonable bounds\n"); return false; }
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
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numBodies = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) numSteps = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank==0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, rank, size);
        integrateBodies(bodies);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (rank == 0) printf("Simulation time: %ld ms\n", duration.count());

    // MPI gather already ensures all ranks have full bodies; but ensure broadcast
    MPI_Bcast(bodies.data(), numBodies * (int)sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0 && printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (const auto& body : bodies) {
            bodyData.push_back(body.pos.x); bodyData.push_back(body.pos.y); bodyData.push_back(body.pos.z);
            bodyData.push_back(body.vel.x); bodyData.push_back(body.vel.y); bodyData.push_back(body.vel.z);
        }
        print_results(bodyData, "Bodies");
    }
    if (rank == 0 && validate) {
        printf("Validating simulation results...\n");
        if (validateSimulation(bodies)) {
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    MPI_Finalize();
    return 0;
}
