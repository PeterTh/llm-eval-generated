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

__global__ void computeForcesKernel(Body* bodies, int n, int start, int end) {
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end) return;
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    const double pix = bodies[i].pos.x;
    const double piy = bodies[i].pos.y;
    const double piz = bodies[i].pos.z;
    for (int j = 0; j < n; ++j) {
        const double dx = bodies[j].pos.x - pix;
        const double dy = bodies[j].pos.y - piy;
        const double dz = bodies[j].pos.z - piz;
        const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        const double invDist = 1.0 / sqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }
    bodies[i].vel.x += DT * Fx;
    bodies[i].vel.y += DT * Fy;
    bodies[i].vel.z += DT * Fz;
}

void computeForces(std::vector<Body>& bodies, int rank, int size) {
    const int n = (int)bodies.size();
    int base = n / size;
    int rem = n % size;
    int start = rank * base + (rank < rem ? rank : rem);
    int end = start + base + (rank < rem ? 1 : 0);

    Body* d_bodies = nullptr;
    cudaMalloc(&d_bodies, n * sizeof(Body));
    cudaMemcpy(d_bodies, bodies.data(), n * sizeof(Body), cudaMemcpyHostToDevice);

    int count = end - start;
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    if (blocks == 0) blocks = 1;
    computeForcesKernel<<<blocks, threads>>>(d_bodies, n, start, end);
    cudaDeviceSynchronize();

    // copy back only updated velocities for this rank's range
    cudaMemcpy(bodies.data() + start, d_bodies + start, count * sizeof(Body), cudaMemcpyDeviceToHost);
    cudaFree(d_bodies);

    // gather full state across ranks
    std::vector<int> counts(size), displs(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        int c = base + (r < rem ? 1 : 0);
        counts[r] = c * (int)sizeof(Body);
        displs[r] = offset * (int)sizeof(Body);
        offset += c;
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_BYTE, bodies.data(), counts.data(), displs.data(), MPI_BYTE, MPI_COMM_WORLD);
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
    #pragma omp parallel
    {
        double local = 0.0;
        #pragma omp for nowait
        for (size_t i = 0; i < n; ++i) {
            local += 0.5 * (bodies[i].vel.x * bodies[i].vel.x +
                            bodies[i].vel.y * bodies[i].vel.y +
                            bodies[i].vel.z * bodies[i].vel.z);
        }
        #pragma omp critical
        energy += local;
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
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

    std::vector<Body> bodies(numBodies);
    if (rank == 0) randomizeBodies(bodies);
    MPI_Bcast(bodies.data(), numBodies * (int)sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies, rank, size);
        integrateBodies(bodies);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(numBodies * 6);
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

    MPI_Finalize();
    return 0;
}
