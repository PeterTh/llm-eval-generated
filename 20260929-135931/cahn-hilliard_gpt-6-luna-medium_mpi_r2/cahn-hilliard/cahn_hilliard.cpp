#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return z * nx * ny + y * nx + x;
}

// Exchange the edge planes and fill physical z boundaries by replication.
static void exchange_halo(std::vector<double>& a, size_t plane, size_t local_z,
                          int rank, int size, MPI_Comm comm) {
    const int prev = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int next = rank + 1 == size ? MPI_PROC_NULL : rank + 1;
    MPI_Sendrecv(a.data() + plane, static_cast<int>(plane), MPI_DOUBLE, prev, 10,
                 a.data() + (local_z + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, next, 10,
                 comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(a.data() + local_z * plane, static_cast<int>(plane), MPI_DOUBLE, next, 11,
                 a.data(), static_cast<int>(plane), MPI_DOUBLE, prev, 11,
                 comm, MPI_STATUS_IGNORE);
    if (rank == 0) std::copy_n(a.data() + plane, plane, a.data());
    if (rank + 1 == size) std::copy_n(a.data() + local_z * plane, plane, a.data() + (local_z + 1) * plane);
}

static double lap(const std::vector<double>& a, size_t x, size_t y, size_t z,
                  size_t nx, size_t ny, size_t plane) {
    const size_t p = idx3(x, y, z, nx, ny);
    const size_t xp = x + 1 < nx ? x + 1 : x, xm = x ? x - 1 : 0;
    const size_t yp = y + 1 < ny ? y + 1 : y, ym = y ? y - 1 : 0;
    return a[idx3(xp,y,z,nx,ny)] + a[idx3(xm,y,z,nx,ny)] +
           a[idx3(x,yp,z,nx,ny)] + a[idx3(x,ym,z,nx,ny)] +
           a[p-plane] + a[p+plane] - 6.0*a[p];
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t nx=64, ny=0, nz=0;
    int iterations=20;
    bool validate=false, printResults=false;
    for (int i=1; i<argc; ++i) {
        if (!strcmp(argv[i],"-x") && i+1<argc) nx=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-y") && i+1<argc) ny=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-z") && i+1<argc) nz=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-i") && i+1<argc) iterations=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-v")) validate=true;
        else if (!strcmp(argv[i],"-r")) printResults=true;
        else if (!strcmp(argv[i],"-h")) {
            if(rank==0) printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r] [-h]\n",argv[0]);
            MPI_Finalize(); return 0;
        } else { if(rank==0) fprintf(stderr,"Unknown option: %s\n",argv[i]); MPI_Abort(MPI_COMM_WORLD,1); }
    }
    if (!ny) ny=nx;
    if (!nz) nz=nx;
    if (ranks > static_cast<int>(nz) || nx==0 || ny==0 || nz==0) {
        if(rank==0) fprintf(stderr,"Grid must be nonempty and MPI ranks must not exceed Z planes.\n");
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    const size_t base=nz/ranks, rem=nz%ranks;
    const size_t local_z=base+(static_cast<size_t>(rank)<rem);
    const size_t z0=static_cast<size_t>(rank)*base+std::min(static_cast<size_t>(rank),rem);
    const size_t plane=nx*ny, cells=plane*local_z;
    const size_t vol=plane*nz;
    std::vector<double> c0((local_z+2)*plane), c1((local_z+2)*plane), mu((local_z+2)*plane);
    for(size_t z=1; z<=local_z; ++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
        const size_t id=(z0+z-1)*plane+y*nx+x;
        c0[idx3(x,y,z,nx,ny)]=-1.0+2.0*((((id+1)*1299709)%vol)/static_cast<double>(vol));
    }
    const double dtD=0.01;
    if(rank==0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",nx,ny,nz,iterations,validate?"enabled":"disabled");
        printf("Initializing concentration field...\nRunning Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start=MPI_Wtime();
    for(int t=0;t<iterations;++t) {
        exchange_halo(c0,plane,local_z,rank,ranks,MPI_COMM_WORLD);
        for(size_t z=1;z<=local_z;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
            const size_t p=idx3(x,y,z,nx,ny); const double cv=c0[p];
            mu[p]=4.5*((cv+1.0)*(-2.0/9.0)+(cv-1.0)*(-2.0/9.0)-2.0*cv*(2.0/9.0))+3.0*cv+cv*cv*cv-0.5*lap(c0,x,y,z,nx,ny,plane);
        }
        exchange_halo(mu,plane,local_z,rank,ranks,MPI_COMM_WORLD);
        for(size_t z=1;z<=local_z;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
            const size_t p=idx3(x,y,z,nx,ny);
            c1[p]=c0[p]+dtD*lap(mu,x,y,z,nx,ny,plane);
        }
        c0.swap(c1);
    }
    double elapsed=MPI_Wtime()-start, maxElapsed;
    MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0) printf("Computation time: %.0f ms\nPerformance: %.3f MCellUpdates/s\n",maxElapsed*1000,vol*static_cast<double>(iterations)/maxElapsed/1e6);

    std::vector<int> counts, displs;
    std::vector<double> global;
    if(rank==0) { counts.resize(ranks); displs.resize(ranks); global.resize(vol); }
    const int sendcount=static_cast<int>(cells);
    if(rank==0) for(int r=0;r<ranks;++r) {
        const size_t rz=nz/ranks+(static_cast<size_t>(r)<nz%ranks);
        const size_t rz0=static_cast<size_t>(r)*(nz/ranks)+std::min(static_cast<size_t>(r),nz%ranks);
        counts[r]=static_cast<int>(rz*plane); displs[r]=static_cast<int>(rz0*plane);
    }
    MPI_Gatherv(c0.data()+plane,sendcount,MPI_DOUBLE,rank==0?global.data():nullptr,
                rank==0?counts.data():nullptr,rank==0?displs.data():nullptr,MPI_DOUBLE,0,MPI_COMM_WORLD);
    int result=0;
    if(rank==0) {
        if(printResults) print_results(global,"Concentration");
        if(validate) {
            bool finite=true; double mn=global[0], mx=global[0];
            for(double v:global) { finite &= std::isfinite(v); mn=std::min(mn,v); mx=std::max(mx,v); }
            printf("Validating result...\nConcentration range: [%.6f, %.6f]\n",mn,mx);
            result=finite && mx<=10.0 && mn>=-10.0 ? 0:1;
            printf("Validation: %s\n",result?"FAILED":"PASSED");
        }
    }
    MPI_Bcast(&result,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
