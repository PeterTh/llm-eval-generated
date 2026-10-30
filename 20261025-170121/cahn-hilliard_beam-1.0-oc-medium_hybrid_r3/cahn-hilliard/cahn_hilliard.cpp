#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

// 3D index calculation for local slab with halo in z: dims (nx,ny,nz_l+2)
inline __host__ __device__ size_t idx3l(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

__global__ void computeChemicalPotentialKernel(const double* c, double* mu,
    size_t nx, size_t ny, size_t nz_l,
    double dx, double dy, double dz, double gamma,
    double e_AA, double e_BB, double e_AB) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nx * ny * nz_l;
    if (i >= total) return;
    size_t x = i % nx;
    size_t y = (i / nx) % ny;
    size_t zl = i / (nx * ny); // 0..nz_l-1 local real, stored at zl+1
    size_t z = zl + 1;
    size_t idx = idx3l(x,y,z,nx,ny);
    double cv = c[idx];
    size_t xp = (x < nx-1)? x+1 : x;
    size_t xn = (x > 0)? x-1 : 0;
    size_t yp = (y < ny-1)? y+1 : y;
    size_t yn = (y > 0)? y-1 : 0;
    size_t zp = z+1; // halo handles clamp? need clamp at global boundary
    // global z boundary clamping: if zl==nz_l-1 (last local) and last global rank, zp=z
    // We'll handle by padding halo with clamped values during exchange instead.
    size_t zn = z-1;
    double cxx = (c[idx3l(xp,y,z,nx,ny)] + c[idx3l(xn,y,z,nx,ny)] - 2.0*c[idx])/(dx*dx);
    double cyy = (c[idx3l(x,yp,z,nx,ny)] + c[idx3l(x,yn,z,nx,ny)] - 2.0*c[idx])/(dy*dy);
    double czz = (c[idx3l(x,y,zp,nx,ny)] + c[idx3l(x,y,zn,nx,ny)] - 2.0*c[idx])/(dz*dz);
    double lap = cxx+cyy+czz;
    mu[idx] = 4.5*((cv+1.0)*e_AA + (cv-1.0)*e_BB - 2.0*cv*e_AB)
             + 3.0*cv + cv*cv*cv - gamma*lap;
}

__global__ void updateKernel(double* cnew, const double* cold, const double* mu,
    size_t nx, size_t ny, size_t nz_l, double D, double dt,
    double dx, double dy, double dz) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nx * ny * nz_l;
    if (i >= total) return;
    size_t x = i % nx;
    size_t y = (i / nx) % ny;
    size_t z = i / (nx*ny) + 1;
    size_t idx = idx3l(x,y,z,nx,ny);
    size_t xp = (x < nx-1)? x+1 : x;
    size_t xn = (x > 0)? x-1 : 0;
    size_t yp = (y < ny-1)? y+1 : y;
    size_t yn = (y > 0)? y-1 : 0;
    size_t zp = z+1;
    size_t zn = z-1;
    double lap = (mu[idx3l(xp,y,z,nx,ny)] + mu[idx3l(xn,y,z,nx,ny)] - 2.0*mu[idx])/(dx*dx)
               + (mu[idx3l(x,yp,z,nx,ny)] + mu[idx3l(x,yn,z,nx,ny)] - 2.0*mu[idx])/(dy*dy)
               + (mu[idx3l(x,y,zp,nx,ny)] + mu[idx3l(x,y,zn,nx,ny)] - 2.0*mu[idx])/(dz*dz);
    cnew[idx] = cold[idx] + dt*D*lap;
}


bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) { printf("Validation failed: found NaN or Inf value\n"); return false; }
    }
    double minVal=c[0], maxVal=c[0];
    for (const auto& val : c){ minVal=std::min(minVal,val); maxVal=std::max(maxVal,val); }
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 10.0 || minVal < -10.0){ printf("Validation failed: values out of expected range\n"); return false; }
    return true;
}
void printUsage(const char* progName){
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t nx=64, ny=0, nz=0;
    int iterations=20;
    bool validate=false, printResults=false;
    for (int i=1;i<argc;++i){
        if(!strcmp(argv[i],"-x")&&i+1<argc) nx=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-y")&&i+1<argc) ny=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-z")&&i+1<argc) nz=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-i")&&i+1<argc) iterations=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-v")) validate=true;
        else if(!strcmp(argv[i],"-r")) printResults=true;
        else if(!strcmp(argv[i],"-h")){ if(rank==0) printUsage(argv[0]); MPI_Finalize(); return 0;}
    }
    if(ny==0) ny=nx;
    if(nz==0) nz=nx;

    if(rank==0){
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate?"enabled":"disabled");
    }

    double dx=1.0,dy=1.0,dz=1.0,dt=0.01;
    double e_AA=-(2.0/9.0), e_BB=-(2.0/9.0), e_AB=(2.0/9.0), gamma=0.5, D=1.0;

    // partition z
    size_t base = nz / size;
    size_t rem = nz % size;
    size_t nz_l = base + (rank < (int)rem ? 1 : 0);
    size_t z_start = rank*base + (rank < (int)rem ? rank : rem);

    size_t plane = nx*ny;
    size_t localN = plane*(nz_l+2);
    std::vector<double> cold(localN,0), cnew(localN,0), mu(localN,0);

    // init
    #pragma omp parallel for collapse(3)
    for(size_t zl=0; zl<nz_l; ++zl){
        for(size_t y=0; y<ny; ++y){
            for(size_t x=0; x<nx; ++x){
                size_t gz = z_start + zl;
                size_t idx = idx3l(x,y,zl+1,nx,ny);
                size_t linear_id = gz*(nx*ny) + y*nx + x;
                size_t vol = nx*ny*nz;
                double pseudo = ((((linear_id+1)*1299709)%vol)/static_cast<double>(vol));
                cold[idx] = -1.0 + 2.0*pseudo;
            }
        }
    }

    // halo init clamped
    // first halo plane = copy of first real if z_start==0 else will exchange; set self for boundaries
    if (nz_l>0){
        // z=0 halo (below first) clamp
        for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x){
            cold[idx3l(x,y,0,nx,ny)] = cold[idx3l(x,y,1,nx,ny)];
            cold[idx3l(x,y,nz_l+1,nx,ny)] = cold[idx3l(x,y,nz_l,nx,ny)];
        }
    }

    double *d_cold,*d_cnew,*d_mu;
    cudaMalloc(&d_cold, localN*sizeof(double));
    cudaMalloc(&d_cnew, localN*sizeof(double));
    cudaMalloc(&d_mu, localN*sizeof(double));

    auto exchangeHalo = [&](){
        // send first real plane to prev, receive into halo 0
        if (size>1){
            // prev
            if (rank>0){
                MPI_Sendrecv(cold.data()+plane*1, plane, MPI_DOUBLE, rank-1, 0,
                             cold.data()+plane*0, plane, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                // clamp: copy first real to halo 0
                for(size_t i=0;i<plane;++i) cold[i]=cold[plane+i];
            }
            if (rank<size-1){
                MPI_Sendrecv(cold.data()+plane*nz_l, plane, MPI_DOUBLE, rank+1, 1,
                             cold.data()+plane*(nz_l+1), plane, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                for(size_t i=0;i<plane;++i) cold[plane*(nz_l+1)+i]=cold[plane*nz_l+i];
            }
        } else {
            for(size_t i=0;i<plane;++i){ cold[i]=cold[plane+i]; cold[plane*(nz_l+1)+i]=cold[plane*nz_l+i]; }
        }
    };

    auto start = std::chrono::high_resolution_clock::now();
    for(int t=0;t<iterations;++t){
        exchangeHalo();
        cudaMemcpy(d_cold, cold.data(), localN*sizeof(double), cudaMemcpyHostToDevice);
        int bs=256;
        int ng=(nx*ny*nz_l+bs-1)/bs;
        computeChemicalPotentialKernel<<<ng,bs>>>(d_cold,d_mu,nx,ny,nz_l,dx,dy,dz,gamma,e_AA,e_BB,e_AB);
        cudaDeviceSynchronize();
        // mu halo? mu computed for real cells only; need clamp boundaries for update laplacian of mu at boundaries.
        // Fill mu halos similarly (clamped) by copying
        cudaMemcpy(mu.data(), d_mu, localN*sizeof(double), cudaMemcpyDeviceToHost);
        // clamp mu halos
        if(size>1){
            if(rank==0){ for(size_t i=0;i<plane;++i) mu[i]=mu[plane+i]; }
            if(rank==size-1){ for(size_t i=0;i<plane;++i) mu[plane*(nz_l+1)+i]=mu[plane*nz_l+i]; }
        } else {
            for(size_t i=0;i<plane;++i){ mu[i]=mu[plane+i]; mu[plane*(nz_l+1)+i]=mu[plane*nz_l+i]; }
        }
        // exchange mu halos between ranks
        if(size>1){
            if(rank>0) MPI_Sendrecv(mu.data()+plane*1, plane, MPI_DOUBLE, rank-1,2, mu.data()+plane*0, plane, MPI_DOUBLE, rank-1,3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            else for(size_t i=0;i<plane;++i) mu[i]=mu[plane+i];
            if(rank<size-1) MPI_Sendrecv(mu.data()+plane*nz_l, plane, MPI_DOUBLE, rank+1,3, mu.data()+plane*(nz_l+1), plane, MPI_DOUBLE, rank+1,2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            else for(size_t i=0;i<plane;++i) mu[plane*(nz_l+1)+i]=mu[plane*nz_l+i];
        }
        cudaMemcpy(d_mu, mu.data(), localN*sizeof(double), cudaMemcpyHostToDevice);
        updateKernel<<<ng,bs>>>(d_cnew,d_cold,d_mu,nx,ny,nz_l,D,dt,dx,dy,dz);
        cudaDeviceSynchronize();
        cudaMemcpy(cnew.data(), d_cnew, localN*sizeof(double), cudaMemcpyDeviceToHost);
        // swap
        cold.swap(cnew);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);

    // gather
    std::vector<int> counts(size), displs(size);
    // counts in doubles
    for(int r=0;r<size;++r){
        size_t b = nz/size, rem2=nz%size;
        size_t nl = b + (r<(int)rem2?1:0);
        counts[r]=nl*plane;
    }
    displs[0]=0;
    for(int r=1;r<size;++r) displs[r]=displs[r-1]+counts[r-1];
    std::vector<double> globalC;
    if(rank==0) globalC.resize(nx*ny*nz);
    // prepare send buffer: real planes
    std::vector<double> sendbuf(nz_l*plane);
    for(size_t zl=0;zl<nz_l;++zl) for(size_t i=0;i<plane;++i) sendbuf[zl*plane+i]=cold[(zl+1)*plane+i];
    MPI_Gatherv(sendbuf.data(), nz_l*plane, MPI_DOUBLE, globalC.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank==0){
        printf("Computation time: %ld ms\n", duration.count());
        double cellUpdates=(double)(nx*ny*nz)*iterations;
        double mcups=cellUpdates/(duration.count()/1000.0)/1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
        if(printResults) print_results(globalC, "Concentration");
        if(validate){
            printf("Validating result...\n");
            bool valid=validateResult(globalC,nx,ny,nz);
            if(valid){ printf("Validation: PASSED\n"); }
            else { printf("Validation: FAILED\n"); }
        }
    }

    cudaFree(d_cold); cudaFree(d_cnew); cudaFree(d_mu);
    MPI_Finalize();
    return 0;
}
