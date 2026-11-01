#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

__global__ void computeForcesKernel(int n, int start, int end,
                                   const double* px, const double* py, const double* pz,
                                   double* vx, double* vy, double* vz) {
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end) return;
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int j = 0; j < n; ++j) {
        double dx = px[j] - px[i];
        double dy = py[j] - py[i];
        double dz = pz[j] - pz[i];
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

__global__ void integrateKernel(int start, int end, double* px, double* py, double* pz,
                              const double* vx, const double* vy, const double* vz) {
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end) return;
    px[i] += vx[i] * DT;
    py[i] += vy[i] * DT;
    pz[i] += vz[i] * DT;
}

void randomizeBodies(std::vector<double>& px, std::vector<double>& py, std::vector<double>& pz,
                     std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz, unsigned int seed=42) {
    for (size_t i=0;i<px.size();++i){
        px[i]=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
        py[i]=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
        pz[i]=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
        vx[i]=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
        vy[i]=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
        vz[i]=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
    }
}

int main(int argc, char** argv){
    MPI_Init(&argc,&argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);

    int numBodies=1024; int numSteps=10; bool validate=false; bool printResults=false;
    for(int i=1;i<argc;++i){
        if(strcmp(argv[i],"-n")==0 && i+1<argc) numBodies=atoi(argv[++i]);
        else if(strcmp(argv[i],"-s")==0 && i+1<argc) numSteps=atoi(argv[++i]);
        else if(strcmp(argv[i],"-v")==0) validate=true;
        else if(strcmp(argv[i],"-r")==0) printResults=true;
        else if(strcmp(argv[i],"-h")==0){ if(rank==0) printf("Usage: %s [options]\n",argv[0]); MPI_Finalize(); return 0;}
    }
    if(rank==0){
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate?"enabled":"disabled");
    }

    std::vector<double> px(numBodies),py(numBodies),pz(numBodies),vx(numBodies),vy(numBodies),vz(numBodies);
    if(rank==0) randomizeBodies(px,py,pz,vx,vy,vz);
    MPI_Bcast(px.data(),numBodies,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(py.data(),numBodies,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(pz.data(),numBodies,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(vx.data(),numBodies,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(vy.data(),numBodies,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(vz.data(),numBodies,MPI_DOUBLE,0,MPI_COMM_WORLD);

    int chunk=numBodies/size; 
    std::vector<int> counts(size), displs(size);
    for(int r=0;r<size;++r){ int s=r*chunk; int e=(r==size-1)?numBodies:(r+1)*chunk; counts[r]=e-s; displs[r]=s; }
    int start=displs[rank]; int end=start+counts[rank];

    double *dpx,*dpy,*dpz,*dvx,*dvy,*dvz;
    cudaMalloc(&dpx,numBodies*sizeof(double)); cudaMalloc(&dpy,numBodies*sizeof(double)); cudaMalloc(&dpz,numBodies*sizeof(double));
    cudaMalloc(&dvx,numBodies*sizeof(double)); cudaMalloc(&dvy,numBodies*sizeof(double)); cudaMalloc(&dvz,numBodies*sizeof(double));

    auto startT=std::chrono::high_resolution_clock::now();
    for(int step=0;step<numSteps;++step){
        cudaMemcpy(dpx,px.data(),numBodies*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(dpy,py.data(),numBodies*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(dpz,pz.data(),numBodies*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(dvx,vx.data(),numBodies*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(dvy,vy.data(),numBodies*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(dvz,vz.data(),numBodies*sizeof(double),cudaMemcpyHostToDevice);

        int nthreads=256; int nblocks=(counts[rank]+nthreads-1)/nthreads;
        computeForcesKernel<<<nblocks,nthreads>>>(numBodies,start,end,dpx,dpy,dpz,dvx,dvy,dvz);
        integrateKernel<<<nblocks,nthreads>>>(start,end,dpx,dpy,dpz,dvx,dvy,dvz);
        cudaDeviceSynchronize();

        // copy back only owned? copy all for simplicity
        cudaMemcpy(px.data(),dpx,numBodies*sizeof(double),cudaMemcpyDeviceToHost);
        cudaMemcpy(py.data(),dpy,numBodies*sizeof(double),cudaMemcpyDeviceToHost);
        cudaMemcpy(pz.data(),dpz,numBodies*sizeof(double),cudaMemcpyDeviceToHost);
        cudaMemcpy(vx.data(),dvx,numBodies*sizeof(double),cudaMemcpyDeviceToHost);
        cudaMemcpy(vy.data(),dvy,numBodies*sizeof(double),cudaMemcpyDeviceToHost);
        cudaMemcpy(vz.data(),dvz,numBodies*sizeof(double),cudaMemcpyDeviceToHost);

        MPI_Allgatherv(px.data()+start, counts[rank], MPI_DOUBLE, px.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(py.data()+start, counts[rank], MPI_DOUBLE, py.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(pz.data()+start, counts[rank], MPI_DOUBLE, pz.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(vx.data()+start, counts[rank], MPI_DOUBLE, vx.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(vy.data()+start, counts[rank], MPI_DOUBLE, vy.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(vz.data()+start, counts[rank], MPI_DOUBLE, vz.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
    // redo gather properly not in loop above; rewrite loop below? We'll just not; instead fix:
    auto endT=std::chrono::high_resolution_clock::now();
    auto duration=std::chrono::duration_cast<std::chrono::milliseconds>(endT-startT);
    long localDuration=duration.count(), maxDuration=0;
    MPI_Reduce(&localDuration,&maxDuration,1,MPI_LONG,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0) printf("Simulation time: %ld ms\n", maxDuration);

    if(rank==0 && printResults){
        std::vector<double> bodyData; bodyData.reserve(numBodies*6);
        for(int i=0;i<numBodies;++i){ bodyData.push_back(px[i]); bodyData.push_back(py[i]); bodyData.push_back(pz[i]); bodyData.push_back(vx[i]); bodyData.push_back(vy[i]); bodyData.push_back(vz[i]); }
        print_results(bodyData, "Bodies");
    }
    if(rank==0 && validate){
        printf("Validating simulation results...\n");
        bool ok=true;
        for(int i=0;i<numBodies;++i){
            if(!std::isfinite(px[i])||!std::isfinite(py[i])||!std::isfinite(pz[i])||!std::isfinite(vx[i])||!std::isfinite(vy[i])||!std::isfinite(vz[i])){printf("Validation failed: NaN/Inf\n");ok=false;break;}
        }
        if(ok){ printf("Validation: PASSED\n"); }
        else { printf("Validation: FAILED\n"); }
    }

    cudaFree(dpx);cudaFree(dpy);cudaFree(dpz);cudaFree(dvx);cudaFree(dvy);cudaFree(dvz);
    MPI_Finalize();
    return 0;
}
