#include <mpi.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <omp.h>

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

__global__ void forceKernel(const double* px, const double* py, const double* pz,
                            const double* vx, const double* vy, const double* vz,
                            double* outVx, double* outVy, double* outVz,
                            int n, int begin, int count) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;
    const int i = begin + local;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    const double ix = px[i], iy = py[i], iz = pz[i];
    for (int j = 0; j < n; ++j) {
        const double dx = px[j] - ix, dy = py[j] - iy, dz = pz[j] - iz;
        const double inv = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        const double inv3 = inv * inv * inv;
        fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
    }
    outVx[i] = vx[i] + DT * fx;
    outVy[i] = vy[i] + DT * fy;
    outVz[i] = vz[i] + DT * fz;
}

__global__ void integrateKernel(double* px, double* py, double* pz,
                                const double* vx, const double* vy, const double* vz, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        px[i] += vx[i] * DT; py[i] += vy[i] * DT; pz[i] += vz[i] * DT;
    }
}

static void cudaCheck(cudaError_t err, const char* what, int rank) {
    if (err != cudaSuccess) {
        if (rank == 0) std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

static double computeTotalEnergy(const std::vector<Body>& b) {
    const size_t n = b.size();
    double kinetic = 0.0, potential = 0.0;
#pragma omp parallel for reduction(+:kinetic) schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        const auto& v = b[static_cast<size_t>(i)].vel;
        kinetic += 0.5 * (v.x*v.x + v.y*v.y + v.z*v.z);
    }
#pragma omp parallel for reduction(+:potential) schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j) {
            const double dx=b[j].pos.x-b[i].pos.x, dy=b[j].pos.y-b[i].pos.y, dz=b[j].pos.z-b[i].pos.z;
            potential -= 1.0 / std::sqrt(dx*dx + dy*dy + dz*dz + SOFTENING);
        }
    }
    return kinetic + potential;
}

static bool validateSimulation(const std::vector<Body>& b) {
    int invalid = 0;
#pragma omp parallel for reduction(|:invalid) schedule(static)
    for (long long i = 0; i < static_cast<long long>(b.size()); ++i) {
        const auto& v = b[static_cast<size_t>(i)];
        if (!std::isfinite(v.pos.x) || !std::isfinite(v.pos.y) || !std::isfinite(v.pos.z) ||
            !std::isfinite(v.vel.x) || !std::isfinite(v.vel.y) || !std::isfinite(v.vel.z) ||
            std::abs(v.pos.x)>1e6 || std::abs(v.pos.y)>1e6 || std::abs(v.pos.z)>1e6 ||
            std::abs(v.vel.x)>1e6 || std::abs(v.vel.y)>1e6 || std::abs(v.vel.z)>1e6) invalid = 1;
    }
    if (invalid) std::printf("Validation failed: found invalid body state\n");
    return !invalid;
}

static void printUsage(const char* p) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num>     Number of bodies (default: 1024)\n  -s <num>     Number of simulation steps (default: 10)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int numBodies=1024, numSteps=10;
    bool validate=false, printResults=false;
    for (int i=1; i<argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i+1<argc) numBodies=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-s") && i+1<argc) numSteps=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate=true;
        else if (!std::strcmp(argv[i], "-r")) printResults=true;
        else if (!std::strcmp(argv[i], "-h")) { if(rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    int bad = (numBodies < 0 || numSteps < 0);
    if (bad) { if(rank==0) std::fprintf(stderr,"Body and step counts must be nonnegative\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (rank==0) {
        std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n", numBodies, numSteps, validate?"enabled":"disabled");
    }

    // Bind each MPI rank to its local accelerator, including oversubscribed jobs.
    int devices=0;
    cudaError_t deviceStatus=cudaGetDeviceCount(&devices);
    if (deviceStatus != cudaSuccess || devices == 0) {
        if (rank==0) std::fprintf(stderr,"This nbody build requires a CUDA device on every MPI rank\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    cudaCheck(cudaSetDevice(rank % devices), "select device", rank);

    std::vector<Body> bodies(static_cast<size_t>(numBodies));
    randomizeBodies(bodies);
    const int begin = static_cast<int>((static_cast<long long>(numBodies) * rank) / ranks);
    const int end = static_cast<int>((static_cast<long long>(numBodies) * (rank+1)) / ranks);
    const int count=end-begin;
    std::vector<double> px(numBodies), py(numBodies), pz(numBodies), vx(numBodies), vy(numBodies), vz(numBodies);
#pragma omp parallel for schedule(static)
    for (int i=0;i<numBodies;++i) {
        const Body& b=bodies[static_cast<size_t>(i)];
        px[i]=b.pos.x; py[i]=b.pos.y; pz[i]=b.pos.z; vx[i]=b.vel.x; vy[i]=b.vel.y; vz[i]=b.vel.z;
    }
    double *dpx=nullptr,*dpy=nullptr,*dpz=nullptr,*dvx=nullptr,*dvy=nullptr,*dvz=nullptr;
    const size_t bytes=sizeof(double)*static_cast<size_t>(std::max(numBodies,1));
    cudaCheck(cudaMalloc(&dpx, bytes), "allocate positions", rank);
    cudaCheck(cudaMalloc(&dpy, bytes), "allocate positions", rank);
    cudaCheck(cudaMalloc(&dpz, bytes), "allocate positions", rank);
    cudaCheck(cudaMalloc(&dvx, bytes), "allocate velocities", rank);
    cudaCheck(cudaMalloc(&dvy, bytes), "allocate velocities", rank);
    cudaCheck(cudaMalloc(&dvz, bytes), "allocate velocities", rank);
    if(numBodies) {
        cudaCheck(cudaMemcpy(dpx,px.data(),sizeof(double)*numBodies,cudaMemcpyHostToDevice),"copy positions",rank);
        cudaCheck(cudaMemcpy(dpy,py.data(),sizeof(double)*numBodies,cudaMemcpyHostToDevice),"copy positions",rank);
        cudaCheck(cudaMemcpy(dpz,pz.data(),sizeof(double)*numBodies,cudaMemcpyHostToDevice),"copy positions",rank);
        cudaCheck(cudaMemcpy(dvx,vx.data(),sizeof(double)*numBodies,cudaMemcpyHostToDevice),"copy velocities",rank);
        cudaCheck(cudaMemcpy(dvy,vy.data(),sizeof(double)*numBodies,cudaMemcpyHostToDevice),"copy velocities",rank);
        cudaCheck(cudaMemcpy(dvz,vz.data(),sizeof(double)*numBodies,cudaMemcpyHostToDevice),"copy velocities",rank);
    }
    int* recvCounts=new int[ranks]; int* displs=new int[ranks];
    for(int r=0;r<ranks;++r){ const int lo=static_cast<int>((static_cast<long long>(numBodies)*r)/ranks); const int hi=static_cast<int>((static_cast<long long>(numBodies)*(r+1))/ranks); recvCounts[r]=hi-lo; displs[r]=lo; }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start=std::chrono::high_resolution_clock::now();
    for(int step=0;step<numSteps;++step) {
        if(count) forceKernel<<<(count+255)/256,256>>>(dpx,dpy,dpz,dvx,dvy,dvz,dvx,dvy,dvz,numBodies,begin,count);
        cudaCheck(cudaGetLastError(),"force kernel",rank);
        cudaCheck(cudaDeviceSynchronize(),"force kernel sync",rank);
        cudaCheck(cudaMemcpy(vx.data()+begin,dvx+begin,sizeof(double)*count,cudaMemcpyDeviceToHost),"read local velocities",rank);
        cudaCheck(cudaMemcpy(vy.data()+begin,dvy+begin,sizeof(double)*count,cudaMemcpyDeviceToHost),"read local velocities",rank);
        cudaCheck(cudaMemcpy(vz.data()+begin,dvz+begin,sizeof(double)*count,cudaMemcpyDeviceToHost),"read local velocities",rank);
        MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DOUBLE,vx.data(),recvCounts,displs,MPI_DOUBLE,MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DOUBLE,vy.data(),recvCounts,displs,MPI_DOUBLE,MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DOUBLE,vz.data(),recvCounts,displs,MPI_DOUBLE,MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(dvx,vx.data(),sizeof(double)*numBodies,cudaMemcpyHostToDevice),"broadcast velocities to device",rank);
        cudaCheck(cudaMemcpy(dvy,vy.data(),sizeof(double)*numBodies,cudaMemcpyHostToDevice),"broadcast velocities to device",rank);
        cudaCheck(cudaMemcpy(dvz,vz.data(),sizeof(double)*numBodies,cudaMemcpyHostToDevice),"broadcast velocities to device",rank);
        if(numBodies) integrateKernel<<<(numBodies+255)/256,256>>>(dpx,dpy,dpz,dvx,dvy,dvz,numBodies);
        cudaCheck(cudaGetLastError(),"integration kernel",rank);
        cudaCheck(cudaDeviceSynchronize(),"integration kernel sync",rank);
    }
    if(numBodies) {
        cudaCheck(cudaMemcpy(px.data(),dpx,sizeof(double)*numBodies,cudaMemcpyDeviceToHost),"read positions",rank);
        cudaCheck(cudaMemcpy(py.data(),dpy,sizeof(double)*numBodies,cudaMemcpyDeviceToHost),"read positions",rank);
        cudaCheck(cudaMemcpy(pz.data(),dpz,sizeof(double)*numBodies,cudaMemcpyDeviceToHost),"read positions",rank);
        cudaCheck(cudaMemcpy(vx.data(),dvx,sizeof(double)*numBodies,cudaMemcpyDeviceToHost),"read velocities",rank);
        cudaCheck(cudaMemcpy(vy.data(),dvy,sizeof(double)*numBodies,cudaMemcpyDeviceToHost),"read velocities",rank);
        cudaCheck(cudaMemcpy(vz.data(),dvz,sizeof(double)*numBodies,cudaMemcpyDeviceToHost),"read velocities",rank);
    }
    const auto endTime=std::chrono::high_resolution_clock::now();
    double elapsed=std::chrono::duration<double,std::milli>(endTime-start).count(), maxElapsed=0;
    MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
#pragma omp parallel for schedule(static)
    for(int i=0;i<numBodies;++i){ Body& b=bodies[static_cast<size_t>(i)]; b.pos={px[i],py[i],pz[i]}; b.vel={vx[i],vy[i],vz[i]}; }
    int failed=0;
    if(rank==0){
        std::printf("Simulation time: %ld ms\n",static_cast<long>(maxElapsed));
        if(printResults){ std::vector<double> data; data.reserve(static_cast<size_t>(numBodies)*6); for(const auto& b:bodies){data.insert(data.end(),{b.pos.x,b.pos.y,b.pos.z,b.vel.x,b.vel.y,b.vel.z});} print_results(data,"Bodies"); }
        if(validate){ std::printf("Validating simulation results...\n"); failed=!validateSimulation(bodies); if(!failed){ std::printf("Final energy: %.6f\n",computeTotalEnergy(bodies)); std::printf("Validation: PASSED\n"); } else std::printf("Validation: FAILED\n"); }
    }
    delete[] recvCounts; delete[] displs;
    cudaFree(dpx); cudaFree(dpy); cudaFree(dpz); cudaFree(dvx); cudaFree(dvy); cudaFree(dvz);
    MPI_Bcast(&failed,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Finalize();
    return failed;
}
