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

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos; Vec3 vel; };

static void cuda_check(cudaError_t e, const char *what) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

__global__ static void advance_kernel(const Body *b, Body *out, int n, int first, int count) {
    int i = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= first + count) return;
    const Body bi = b[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < n; ++j) {
        double dx = b[j].pos.x - bi.pos.x, dy = b[j].pos.y - bi.pos.y, dz = b[j].pos.z - bi.pos.z;
        double inv = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        double inv3 = inv * inv * inv;
        fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
    }
    Body r = bi;
    r.vel.x += DT * fx; r.vel.y += DT * fy; r.vel.z += DT * fz;
    r.pos.x += r.vel.x * DT; r.pos.y += r.vel.y * DT; r.pos.z += r.vel.z * DT;
    out[i] = r;
}

static void randomizeBodies(std::vector<Body>& b) {
    unsigned int seed = 42;
    for (auto &x : b) { x.pos.x=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; x.pos.y=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; x.pos.z=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; x.vel.x=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; x.vel.y=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; x.vel.z=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; }
}

static double energy(const std::vector<Body>& b) {
    double e=0; const int n=(int)b.size();
    #pragma omp parallel for reduction(+:e)
    for(int i=0;i<n;++i) e += .5*(b[i].vel.x*b[i].vel.x+b[i].vel.y*b[i].vel.y+b[i].vel.z*b[i].vel.z);
    #pragma omp parallel for reduction(+:e)
    for(int i=0;i<n;++i) for(int j=i+1;j<n;++j) { double x=b[j].pos.x-b[i].pos.x,y=b[j].pos.y-b[i].pos.y,z=b[j].pos.z-b[i].pos.z; e-=1.0/sqrt(x*x+y*y+z*z+SOFTENING); }
    return e;
}
static bool valid(const std::vector<Body>& b) { for(const auto& x:b) for(double v:{x.pos.x,x.pos.y,x.pos.z,x.vel.x,x.vel.y,x.vel.z}) if(!std::isfinite(v)||std::abs(v)>1e6) return false; return true; }
static void usage(const char* p) { printf("Usage: %s [-n num] [-s num] [-v] [-r] [-h]\n",p); }

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv); int rank=0,size=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    int n=1024, steps=10; bool validateFlag=false, results=false;
    for(int i=1;i<argc;++i) { if(!strcmp(argv[i],"-n")&&i+1<argc)n=atoi(argv[++i]); else if(!strcmp(argv[i],"-s")&&i+1<argc)steps=atoi(argv[++i]); else if(!strcmp(argv[i],"-v"))validateFlag=true; else if(!strcmp(argv[i],"-r"))results=true; else if(!strcmp(argv[i],"-h")){if(!rank)usage(argv[0]); MPI_Finalize(); return 0;} else {if(!rank)usage(argv[0]); MPI_Finalize(); return 1;} }
    if(n<0||steps<0){MPI_Finalize();return 1;}
    std::vector<Body> bodies(n), next(n); randomizeBodies(bodies);
    std::vector<int> bodyCounts(size), bodyDispls(size), counts(size), displs(size); for(int r=0;r<size;++r){int a=n*r/size,z=n*(r+1)/size;bodyCounts[r]=z-a;bodyDispls[r]=a;counts[r]=bodyCounts[r]*(int)sizeof(Body);displs[r]=bodyDispls[r]*(int)sizeof(Body);}
    int devices=0; cuda_check(cudaGetDeviceCount(&devices),"cudaGetDeviceCount"); if(!devices){fprintf(stderr,"No CUDA device available\n");MPI_Abort(MPI_COMM_WORLD,2);} cuda_check(cudaSetDevice(rank%devices),"cudaSetDevice");
    int first=bodyDispls[rank], local=bodyCounts[rank];
    Body *d_in=nullptr,*d_out=nullptr; cuda_check(cudaMalloc(&d_in,n*sizeof(Body)),"cudaMalloc"); cuda_check(cudaMalloc(&d_out,n*sizeof(Body)),"cudaMalloc");
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
    for(int s=0;s<steps;++s){ cuda_check(cudaMemcpy(d_in,bodies.data(),n*sizeof(Body),cudaMemcpyHostToDevice),"H2D"); advance_kernel<<<(local+255)/256,256>>>(d_in,d_out,n,first,local); cuda_check(cudaGetLastError(),"kernel"); cuda_check(cudaMemcpy(next.data()+first,d_out+first,local*sizeof(Body),cudaMemcpyDeviceToHost),"D2H"); MPI_Allgatherv(next.data()+first,counts[rank],MPI_BYTE,bodies.data(),counts.data(),displs.data(),MPI_BYTE,MPI_COMM_WORLD); }
    cuda_check(cudaDeviceSynchronize(),"final synchronize"); cudaFree(d_in); cudaFree(d_out); MPI_Barrier(MPI_COMM_WORLD); auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start).count();
    if(!rank){printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\nSimulation time: %ld ms\n",n,steps,validateFlag?"enabled":"disabled",(long)ms); if(results){std::vector<double>d;d.reserve(6*n);for(auto&x:bodies){d.insert(d.end(),{x.pos.x,x.pos.y,x.pos.z,x.vel.x,x.vel.y,x.vel.z});}print_results(d,"Bodies");} if(validateFlag){bool ok=valid(bodies);printf("Final energy: %.6f\nValidation: %s\n",energy(bodies),ok?"PASSED":"FAILED"); MPI_Finalize(); return ok?0:1;}}
    MPI_Finalize(); return 0;
}
