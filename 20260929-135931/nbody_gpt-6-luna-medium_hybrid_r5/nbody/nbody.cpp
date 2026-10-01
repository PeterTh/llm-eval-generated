#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos; Vec3 vel; };

__global__ void advanceBodies(Body* bodies, int n, int first, int count) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;
    const int i = first + local;
    const Body self = bodies[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < n; ++j) {
        const double dx = bodies[j].pos.x - self.pos.x;
        const double dy = bodies[j].pos.y - self.pos.y;
        const double dz = bodies[j].pos.z - self.pos.z;
        const double r2 = dx*dx + dy*dy + dz*dz + SOFTENING;
        const double inv = 1.0 / sqrt(r2);
        const double inv3 = inv * inv * inv;
        fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
    }
    Body out = self;
    out.vel.x += DT * fx; out.vel.y += DT * fy; out.vel.z += DT * fz;
    out.pos.x += out.vel.x * DT; out.pos.y += out.vel.y * DT; out.pos.z += out.vel.z * DT;
    bodies[i] = out;
}

static void cudaCheck(cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

void randomizeBodies(std::vector<Body>& b, unsigned int seed = 42) {
    for (auto& a : b) {
        a.pos.x=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; a.pos.y=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
        a.pos.z=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; a.vel.x=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
        a.vel.y=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; a.vel.z=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
    }
}

double computeTotalEnergy(const std::vector<Body>& b) {
    const int n=(int)b.size(); double energy=0.0;
    #pragma omp parallel for reduction(+:energy) schedule(static)
    for (int i=0;i<n;++i) energy += 0.5*(b[i].vel.x*b[i].vel.x+b[i].vel.y*b[i].vel.y+b[i].vel.z*b[i].vel.z);
    double potential=0.0;
    #pragma omp parallel for reduction(+:potential) schedule(static)
    for(int i=0;i<n;++i) for(int j=i+1;j<n;++j) {
        const double dx=b[j].pos.x-b[i].pos.x, dy=b[j].pos.y-b[i].pos.y, dz=b[j].pos.z-b[i].pos.z;
        potential -= 1.0/std::sqrt(dx*dx+dy*dy+dz*dz+SOFTENING);
    }
    return energy+potential;
}

bool validateSimulation(const std::vector<Body>& b) {
    int bad=0;
    #pragma omp parallel for reduction(|:bad) schedule(static)
    for(int i=0;i<(int)b.size();++i) {
        const Body& a=b[i];
        if(!std::isfinite(a.pos.x)||!std::isfinite(a.pos.y)||!std::isfinite(a.pos.z)||!std::isfinite(a.vel.x)||!std::isfinite(a.vel.y)||!std::isfinite(a.vel.z)||std::abs(a.pos.x)>1e6||std::abs(a.pos.y)>1e6||std::abs(a.pos.z)>1e6||std::abs(a.vel.x)>1e6||std::abs(a.vel.y)>1e6||std::abs(a.vel.z)>1e6) bad=1;
    }
    if(bad) printf("Validation failed: body state contains invalid or extreme values\n");
    return !bad;
}

void printUsage(const char* p) { printf("Usage: %s [-n num] [-s num] [-v] [-r] [-h]\n",p); }

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    int rank=0,ranks=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    int n=1024,steps=10; bool validate=false,printResults=false; int parseError=0;
    for(int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"-n")&&i+1<argc)n=atoi(argv[++i]); else if(!strcmp(argv[i],"-s")&&i+1<argc)steps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-v"))validate=true; else if(!strcmp(argv[i],"-r"))printResults=true;
        else if(!strcmp(argv[i],"-h")){if(rank==0)printUsage(argv[0]); MPI_Finalize();return 0;} else {if(rank==0)printf("Unknown option: %s\n",argv[i]);parseError=1;}
    }
    if(n<0||steps<0)parseError=1;
    if(parseError){MPI_Finalize();return 1;}
    int devices=0; cudaCheck(cudaGetDeviceCount(&devices));
    if(devices<1){if(rank==0)fprintf(stderr,"This benchmark requires a CUDA GPU on every MPI rank.\n");MPI_Abort(MPI_COMM_WORLD,2);}
    int localRank=0; MPI_Comm local; MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&local); MPI_Comm_rank(local,&localRank);
    cudaCheck(cudaSetDevice(localRank%devices)); MPI_Comm_free(&local);
    std::vector<Body> bodies((size_t)n); randomizeBodies(bodies);
    const int base=n/ranks, rem=n%ranks, first=rank*base+std::min(rank,rem), count=base+(rank<rem);
    std::vector<int> counts(ranks),displs(ranks);
    for(int r=0;r<ranks;++r){int c=base+(r<rem);counts[r]=c*(int)sizeof(Body);displs[r]=(r*base+std::min(r,rem))*(int)sizeof(Body);}
    Body* deviceBodies=nullptr; cudaCheck(cudaMalloc(&deviceBodies,std::max<size_t>(1,(size_t)n)*sizeof(Body)));
    if(rank==0){printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",n,steps,validate?"enabled":"disabled");}
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
    for(int step=0;step<steps;++step) {
        cudaCheck(cudaMemcpy(deviceBodies,bodies.data(),(size_t)n*sizeof(Body),cudaMemcpyHostToDevice));
        if(count){advanceBodies<<<(count+127)/128,128>>>(deviceBodies,n,first,count);cudaCheck(cudaGetLastError());}
        cudaCheck(cudaDeviceSynchronize());
        std::vector<Body> localBodies((size_t)count);
        if(count)cudaCheck(cudaMemcpy(localBodies.data(),deviceBodies+first,(size_t)count*sizeof(Body),cudaMemcpyDeviceToHost));
        MPI_Allgatherv(localBodies.data(),counts[rank],MPI_BYTE,bodies.data(),counts.data(),displs.data(),MPI_BYTE,MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD); auto end=std::chrono::high_resolution_clock::now();
    if(rank==0)printf("Simulation time: %ld ms\n",(long)std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count());
    cudaFree(deviceBodies);
    if(rank==0&&printResults){std::vector<double> data;data.reserve((size_t)n*6);for(const auto& b:bodies){data.insert(data.end(),{b.pos.x,b.pos.y,b.pos.z,b.vel.x,b.vel.y,b.vel.z});}print_results(data,"Bodies");}
    int valid=1;
    if(validate&&rank==0){printf("Validating simulation results...\n");valid=validateSimulation(bodies);if(valid){printf("Final energy: %.6f\n",computeTotalEnergy(bodies));printf("Validation: PASSED\n");}else printf("Validation: FAILED\n");}
    MPI_Bcast(&valid,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Finalize(); return valid?0:1;
}
