#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

static void exchangeHalos(std::vector<double>& a, size_t plane, size_t localNz, int rank, int ranks) {
    const int prev = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank == ranks - 1 ? MPI_PROC_NULL : rank + 1;
    MPI_Sendrecv(a.data() + plane, static_cast<int>(plane), MPI_DOUBLE, prev, 0,
                 a.data() + (localNz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, next, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(a.data() + localNz * plane, static_cast<int>(plane), MPI_DOUBLE, next, 1,
                 a.data(), static_cast<int>(plane), MPI_DOUBLE, prev, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank == 0) std::copy_n(a.data() + plane, plane, a.data());
    if (rank == ranks - 1) std::copy_n(a.data() + localNz * plane, plane, a.data() + (localNz + 1) * plane);
}

static inline double lap(const std::vector<double>& a, size_t x, size_t y, size_t lz,
                         size_t nx, size_t ny, double dx, double dy, double dz) {
    const size_t p = idx3(x, y, lz, nx, ny);
    const size_t xp = x + 1 < nx ? x + 1 : x, xm = x ? x - 1 : 0;
    const size_t yp = y + 1 < ny ? y + 1 : y, ym = y ? y - 1 : 0;
    return (a[idx3(xp,y,lz,nx,ny)] + a[idx3(xm,y,lz,nx,ny)] - 2*a[p])/(dx*dx)
         + (a[idx3(x,yp,lz,nx,ny)] + a[idx3(x,ym,lz,nx,ny)] - 2*a[p])/(dy*dy)
         + (a[idx3(x,y,lz+1,nx,ny)] + a[idx3(x,y,lz-1,nx,ny)] - 2*a[p])/(dz*dz);
}

static bool validateResult(const std::vector<double>& c) {
    double lo = c[0], hi = c[0];
    for (double v : c) {
        if (!std::isfinite(v)) { printf("Validation failed: found NaN or Inf value\n"); return false; }
        lo = std::min(lo, v); hi = std::max(hi, v);
    }
    printf("Concentration range: [%.6f, %.6f]\n", lo, hi);
    if (hi > 10.0 || lo < -10.0) { printf("Validation failed: values out of expected range\n"); return false; }
    return true;
}

static void printUsage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -x <num>     Grid size in X dimension (default: 64)\n  -y <num>     Grid size in Y dimension (default: same as X)\n  -z <num>     Grid size in Z dimension (default: same as X)\n  -i <num>     Number of time steps (default: 20)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx=64, ny=0, nz=0;
    int iterations=20; bool validate=false, printResults=false;
    for (int i=1; i<argc; ++i) {
        if (!strcmp(argv[i], "-x") && i+1<argc) nx=std::strtoull(argv[++i],nullptr,10);
        else if (!strcmp(argv[i], "-y") && i+1<argc) ny=std::strtoull(argv[++i],nullptr,10);
        else if (!strcmp(argv[i], "-z") && i+1<argc) nz=std::strtoull(argv[++i],nullptr,10);
        else if (!strcmp(argv[i], "-i") && i+1<argc) iterations=std::atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate=true;
        else if (!strcmp(argv[i], "-r")) printResults=true;
        else if (!strcmp(argv[i], "-h")) { if(rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) { printf("Unknown option: %s\n",argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!ny) ny=nx;
    if (!nz) nz=nx;
    if (nx==0 || ny==0 || nz==0 || nz < static_cast<size_t>(ranks) || nx*ny > static_cast<size_t>(INT_MAX) || nx*ny*nz > static_cast<size_t>(INT_MAX)) {
        if(rank==0) fprintf(stderr,"Grid dimensions must be positive, Z must be at least the MPI rank count, and grid counts must fit MPI int counts.\n");
        MPI_Finalize(); return 1;
    }
    if(rank==0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",nx,ny,nz,iterations,validate?"enabled":"disabled");
        printf("Initializing concentration field...\nRunning Cahn-Hilliard simulation...\n");
    }
    const size_t plane=nx*ny;
    const size_t base=nz/static_cast<size_t>(ranks), rem=nz%static_cast<size_t>(ranks);
    const size_t localNz=base+(static_cast<size_t>(rank)<rem);
    const size_t zStart=static_cast<size_t>(rank)*base+std::min(static_cast<size_t>(rank),rem);
    std::vector<double> c((localNz+2)*plane), cnew((localNz+2)*plane), mu((localNz+2)*plane);
    for(size_t lz=1;lz<=localNz;++lz) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
        size_t gid=idx3(x,y,zStart+lz-1,nx,ny);
        size_t vol=nx*ny*nz;
        double pseudo=(((gid+1)*1299709)%vol)/static_cast<double>(vol);
        c[idx3(x,y,lz,nx,ny)]=-1.0+2.0*pseudo;
    }
    constexpr double dx=1,dy=1,dz=1,dt=.01,eAA=-(2.0/9.0),eBB=-(2.0/9.0),eAB=2.0/9.0,gamma=.5,D=1;
    MPI_Barrier(MPI_COMM_WORLD);
    auto start=std::chrono::steady_clock::now();
    for(int t=0;t<iterations;++t) {
        exchangeHalos(c,plane,localNz,rank,ranks);
        for(size_t z=1;z<=localNz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
            size_t p=idx3(x,y,z,nx,ny); double cv=c[p];
            mu[p]=4.5*((cv+1)*eAA+(cv-1)*eBB-2*cv*eAB)+3*cv+cv*cv*cv-gamma*lap(c,x,y,z,nx,ny,dx,dy,dz);
        }
        exchangeHalos(mu,plane,localNz,rank,ranks);
        for(size_t z=1;z<=localNz;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
            size_t p=idx3(x,y,z,nx,ny); cnew[p]=c[p]+dt*D*lap(mu,x,y,z,nx,ny,dx,dy,dz);
        }
        c.swap(cnew);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end=std::chrono::steady_clock::now();
    double elapsed=std::chrono::duration<double>(end-start).count(), maxElapsed=0;
    MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0) { double seconds=std::max(maxElapsed,1e-12); printf("Computation time: %ld ms\n",static_cast<long>(seconds*1000)); printf("Performance: %.3f MCellUpdates/s\n",(double(nx)*ny*nz*iterations)/seconds/1e6); }
    std::vector<double> global;
    if(printResults || validate) {
        std::vector<int> counts(ranks), displs(ranks); int offset=0;
        for(int r=0;r<ranks;++r) { size_t rz=base+(static_cast<size_t>(r)<rem); counts[r]=static_cast<int>(rz*plane); displs[r]=offset; offset+=counts[r]; }
        if(rank==0) global.resize(nx*ny*nz);
        MPI_Gatherv(c.data()+plane,static_cast<int>(localNz*plane),MPI_DOUBLE,rank==0?global.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
    int ok=1;
    if(rank==0) {
        if(printResults) print_results(global,"Concentration");
        if(validate) { printf("Validating result...\n"); ok=validateResult(global)?1:0; printf("Validation: %s\n",ok?"PASSED":"FAILED"); }
    }
    MPI_Bcast(&ok,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Finalize();
    return validate && !ok ? 1 : 0;
}
