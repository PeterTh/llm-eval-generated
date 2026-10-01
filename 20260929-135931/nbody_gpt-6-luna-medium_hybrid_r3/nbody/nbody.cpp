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
static_assert(sizeof(Body) == 6 * sizeof(double));

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

__global__ void stepKernel(const Body* in, Body* out, int n, int begin, int end) {
    int i = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end) return;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    const Body a = in[i];
    for (int j = 0; j < n; ++j) {
        const double dx = in[j].pos.x - a.pos.x;
        const double dy = in[j].pos.y - a.pos.y;
        const double dz = in[j].pos.z - a.pos.z;
        const double d2 = dx * dx + dy * dy + dz * dz + SOFTENING;
        const double inv = 1.0 / sqrt(d2);
        const double inv3 = inv * inv * inv;
        fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
    }
    Body b;
    b.vel = {a.vel.x + DT * fx, a.vel.y + DT * fy, a.vel.z + DT * fz};
    b.pos = {a.pos.x + b.vel.x * DT, a.pos.y + b.vel.y * DT, a.pos.z + b.vel.z * DT};
    out[i] = b;
}

void randomizeBodies(std::vector<Body>& bodies) {
    unsigned int seed = 42;
    for (auto& b : bodies) {
        b.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        b.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

double computeTotalEnergy(const std::vector<Body>& b) {
    const int n = static_cast<int>(b.size());
    double kinetic = 0.0, potential = 0.0;
    #pragma omp parallel for reduction(+:kinetic) schedule(static)
    for (int i = 0; i < n; ++i)
        kinetic += 0.5 * (b[i].vel.x*b[i].vel.x + b[i].vel.y*b[i].vel.y + b[i].vel.z*b[i].vel.z);
    #pragma omp parallel for reduction(+:potential) schedule(static)
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) {
        const double dx=b[j].pos.x-b[i].pos.x, dy=b[j].pos.y-b[i].pos.y, dz=b[j].pos.z-b[i].pos.z;
        potential -= 1.0 / sqrt(dx*dx+dy*dy+dz*dz+SOFTENING);
    }
    return kinetic + potential;
}

bool validateSimulation(const std::vector<Body>& b) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (int i = 0; i < static_cast<int>(b.size()); ++i) {
        const Body& x=b[i];
        valid &= std::isfinite(x.pos.x) && std::isfinite(x.pos.y) && std::isfinite(x.pos.z) &&
                 std::isfinite(x.vel.x) && std::isfinite(x.vel.y) && std::isfinite(x.vel.z) &&
                 fabs(x.pos.x)<=1e6 && fabs(x.pos.y)<=1e6 && fabs(x.pos.z)<=1e6 &&
                 fabs(x.vel.x)<=1e6 && fabs(x.vel.y)<=1e6 && fabs(x.vel.z)<=1e6;
    }
    if (!valid) printf("Validation failed: body state contains non-finite or extreme values\n");
    return valid;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int devices=0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    CUDA_CHECK(cudaSetDevice(rank % devices));

    int numBodies=1024, numSteps=10; bool validate=false, printResults=false;
    for (int i=1;i<argc;++i) {
        if (!strcmp(argv[i],"-n") && i+1<argc) numBodies=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-s") && i+1<argc) numSteps=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-v")) validate=true;
        else if (!strcmp(argv[i],"-r")) printResults=true;
        else if (!strcmp(argv[i],"-h")) { if(!rank) printf("Usage: %s [-n num] [-s num] [-v] [-r] [-h]\n",argv[0]); MPI_Finalize(); return 0; }
        else { if(!rank) { printf("Unknown option: %s\n",argv[i]); printf("Usage: %s [-n num] [-s num] [-v] [-r] [-h]\n",argv[0]); } MPI_Abort(MPI_COMM_WORLD,1); }
    }
    if (numBodies < 0 || numSteps < 0) MPI_Abort(MPI_COMM_WORLD,1);
    if (!rank) { printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",numBodies,numSteps,validate?"enabled":"disabled"); }
    std::vector<Body> bodies(numBodies), local(numBodies);
    if (!rank) randomizeBodies(bodies);
    MPI_Bcast(bodies.data(), numBodies*6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    const int begin=(static_cast<long long>(numBodies)*rank)/ranks;
    const int end=(static_cast<long long>(numBodies)*(rank+1))/ranks;
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r=0;r<ranks;++r) {
        const int first=static_cast<int>((static_cast<long long>(numBodies)*r)/ranks);
        const int last=static_cast<int>((static_cast<long long>(numBodies)*(r+1))/ranks);
        counts[r]=(last-first)*6; offsets[r]=first*6;
    }
    Body *din=nullptr,*dout=nullptr;
    CUDA_CHECK(cudaMalloc(&din, size_t(numBodies)*sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&dout, size_t(numBodies)*sizeof(Body)));
    MPI_Barrier(MPI_COMM_WORLD);
    auto start=std::chrono::high_resolution_clock::now();
    const int threads=256;
    for (int step=0;step<numSteps;++step) {
        CUDA_CHECK(cudaMemcpy(din,bodies.data(),size_t(numBodies)*sizeof(Body),cudaMemcpyHostToDevice));
        if (end>begin) stepKernel<<<(end-begin+threads-1)/threads,threads>>>(din,dout,numBodies,begin,end);
        CUDA_CHECK(cudaGetLastError());
        const int owned=end-begin;
        if (owned) CUDA_CHECK(cudaMemcpy(local.data()+begin,dout+begin,size_t(owned)*sizeof(Body),cudaMemcpyDeviceToHost));
        MPI_Allgatherv(owned ? local.data()+begin : local.data(),owned*6,MPI_DOUBLE,
                       bodies.data(),counts.data(),offsets.data(),MPI_DOUBLE,MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto stop=std::chrono::high_resolution_clock::now();
    if (!rank) printf("Simulation time: %ld ms\n",(long)std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count());
    if (rank==0 && printResults) {
        std::vector<double> data; data.reserve(size_t(numBodies)*6);
        for (const auto& b:bodies) { data.insert(data.end(),{b.pos.x,b.pos.y,b.pos.z,b.vel.x,b.vel.y,b.vel.z}); }
        print_results(data,"Bodies");
    }
    if (rank==0 && validate) {
        printf("Validating simulation results...\n");
        if (validateSimulation(bodies)) { printf("Final energy: %.6f\n",computeTotalEnergy(bodies)); printf("Validation: PASSED\n"); }
        else { printf("Validation: FAILED\n"); MPI_Finalize(); return 1; }
    }
    CUDA_CHECK(cudaFree(din)); CUDA_CHECK(cudaFree(dout));
    MPI_Finalize(); return 0;
}
