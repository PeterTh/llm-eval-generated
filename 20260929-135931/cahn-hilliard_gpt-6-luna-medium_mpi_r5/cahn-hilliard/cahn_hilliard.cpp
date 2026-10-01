#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) {
    return (z * ny + y) * nx + x;
}

// Each rank stores one ghost plane on either side of its local z slab.
void exchangeHalos(std::vector<double>& a, size_t nx, size_t ny, size_t nz,
                   int rank, int ranks) {
    const size_t plane = nx * ny;
    const int below = rank ? rank - 1 : MPI_PROC_NULL;
    const int above = rank + 1 < ranks ? rank + 1 : MPI_PROC_NULL;
    MPI_Sendrecv(a.data() + plane, static_cast<int>(plane), MPI_DOUBLE, below, 0,
                 a.data() + (nz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, above, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(a.data() + nz * plane, static_cast<int>(plane), MPI_DOUBLE, above, 1,
                 a.data(), static_cast<int>(plane), MPI_DOUBLE, below, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank == 0) std::copy_n(a.data() + plane, plane, a.data());
    if (rank == ranks - 1) std::copy_n(a.data() + nz * plane, plane, a.data() + (nz + 1) * plane);
}

double laplacian(const std::vector<double>& a, size_t nx, size_t ny, size_t zcount,
                 size_t x, size_t y, size_t z, double dx, double dy, double dz) {
    const size_t i = idx3(x, y, z, nx, ny);
    const size_t xp = x + (x + 1 < nx), xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny), yn = y - (y > 0);
    const double center = a[i];
    const double xx = (a[idx3(xp,y,z,nx,ny)] + a[idx3(xn,y,z,nx,ny)] - 2.0*center)/(dx*dx);
    const double yy = (a[idx3(x,yp,z,nx,ny)] + a[idx3(x,yn,z,nx,ny)] - 2.0*center)/(dy*dy);
    const double zz = (a[idx3(x,y,z+1,nx,ny)] + a[idx3(x,y,z-1,nx,ny)] - 2.0*center)/(dz*dz);
    (void)zcount;
    return xx + yy + zz;
}

bool validateResult(const std::vector<double>& c) {
    double lo = c.empty() ? 0.0 : c[0], hi = lo;
    for (double v : c) {
        if (!std::isfinite(v)) { std::printf("Validation failed: found NaN or Inf value\n"); return false; }
        lo = std::min(lo, v); hi = std::max(hi, v);
    }
    std::printf("Concentration range: [%.6f, %.6f]\n", lo, hi);
    if (hi > 10.0 || lo < -10.0) { std::printf("Validation failed: values out of expected range\n"); return false; }
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx=64, ny=0, nz=0;
    int iterations=20;
    bool validate=false, printResults=false, help=false, bad=false;
    for (int i=1;i<argc;++i) {
        if (!std::strcmp(argv[i],"-x") && i+1<argc) nx=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-y") && i+1<argc) ny=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-z") && i+1<argc) nz=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-i") && i+1<argc) iterations=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"-v")) validate=true;
        else if (!std::strcmp(argv[i],"-r")) printResults=true;
        else if (!std::strcmp(argv[i],"-h")) help=true;
        else bad=true;
    }
    if (!ny) ny=nx;
    if (!nz) nz=nx;
    if (help || bad || nx==0 || ny==0 || nz < static_cast<size_t>(ranks)) {
        if (rank==0) {
            if (bad || nx==0 || ny==0 || nz < static_cast<size_t>(ranks)) std::printf("Invalid arguments or too many MPI ranks for z dimension\n");
            std::printf("Usage: %s [-x num] [-y num] [-z num] [-i num] [-v] [-r] [-h]\n",argv[0]);
        }
        MPI_Finalize(); return bad ? 1 : 0;
    }
    const size_t base=nz/ranks, rem=nz%ranks;
    const size_t localNz=base+(static_cast<size_t>(rank)<rem), z0=rank*base+std::min(static_cast<size_t>(rank),rem);
    const size_t plane=nx*ny, localCells=plane*localNz;
    std::vector<double> cold((localNz+2)*plane), cnew((localNz+2)*plane), mu((localNz+2)*plane);
    const size_t volume=nx*ny*nz;
    for(size_t z=0;z<localNz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
        const size_t gid=(z+z0)*plane+y*nx+x;
        cold[idx3(x,y,z+1,nx,ny)]=-1.0+2.0*((((gid+1)*1299709)%volume)/static_cast<double>(volume));
    }
    constexpr double dx=1,dy=1,dz=1,dt=.01,eAA=-(2.0/9),eBB=-(2.0/9),eAB=2.0/9,gamma=.5,D=1;
    if(rank==0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",nx,ny,nz,iterations,validate?"enabled":"disabled");
        std::printf("Initializing concentration field...\nRunning Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start=MPI_Wtime();
    for(int t=0;t<iterations;++t) {
        exchangeHalos(cold,nx,ny,localNz,rank,ranks);
        for(size_t z=1;z<=localNz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
            const size_t i=idx3(x,y,z,nx,ny); const double cv=cold[i];
            mu[i]=4.5*((cv+1)*eAA+(cv-1)*eBB-2*cv*eAB)+3*cv+cv*cv*cv-
                  gamma*laplacian(cold,nx,ny,localNz,x,y,z,dx,dy,dz);
        }
        exchangeHalos(mu,nx,ny,localNz,rank,ranks);
        for(size_t z=1;z<=localNz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
            const size_t i=idx3(x,y,z,nx,ny);
            cnew[i]=cold[i]+dt*D*laplacian(mu,nx,ny,localNz,x,y,z,dx,dy,dz);
        }
        cold.swap(cnew);
    }
    const double elapsed=MPI_Wtime()-start;
    double maxElapsed; MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0) {
        const long ms=static_cast<long>(maxElapsed*1000);
        std::printf("Computation time: %ld ms\n",ms);
        const double mcups=static_cast<double>(volume)*iterations/(maxElapsed*1e6);
        std::printf("Performance: %.3f MCellUpdates/s\n",mcups);
    }
    std::vector<double> result;
    if(printResults || validate) {
        std::vector<int> counts(ranks),displs(ranks);
        for(int r=0;r<ranks;++r) { const size_t rn=base+(static_cast<size_t>(r)<rem); counts[r]=static_cast<int>(rn*plane); displs[r]=static_cast<int>((r*base+std::min(static_cast<size_t>(r),rem))*plane); }
        if(rank==0) result.resize(volume);
        MPI_Gatherv(cold.data()+plane,static_cast<int>(localCells),MPI_DOUBLE,rank==0?result.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
    int status=0;
    if(rank==0 && printResults) print_results(result,"Concentration");
    if(rank==0 && validate) { std::printf("Validating result...\n"); status=validateResult(result)?0:1; std::printf("Validation: %s\n",status?"FAILED":"PASSED"); }
    MPI_Bcast(&status,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
