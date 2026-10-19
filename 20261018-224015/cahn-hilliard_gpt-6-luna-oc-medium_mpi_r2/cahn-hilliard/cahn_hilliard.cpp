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
    return z * nx * ny + y * nx + x;
}

static void exchangeHalos(std::vector<double>& a, size_t plane, int localZ, int rank, int size) {
    if (rank > 0)
        MPI_Sendrecv(a.data() + plane, static_cast<int>(plane), MPI_DOUBLE, rank - 1, 0,
                     a.data(), static_cast<int>(plane), MPI_DOUBLE, rank - 1, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank + 1 < size)
        MPI_Sendrecv(a.data() + static_cast<size_t>(localZ) * plane, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 1,
                     a.data() + static_cast<size_t>(localZ + 1) * plane, static_cast<int>(plane), MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

static double laplacian(const std::vector<double>& a, size_t nx, size_t ny,
                        size_t x, size_t y, size_t z, size_t globalZ, size_t nz) {
    const size_t xp = x + 1 < nx ? x + 1 : x, xm = x ? x - 1 : 0;
    const size_t yp = y + 1 < ny ? y + 1 : y, ym = y ? y - 1 : 0;
    const size_t zp = globalZ + 1 < nz ? z + 1 : z, zm = globalZ ? z - 1 : z;
    const size_t i = idx3(x,y,z,nx,ny);
    return a[idx3(xp,y,z,nx,ny)] + a[idx3(xm,y,z,nx,ny)] - 2*a[i]
         + a[idx3(x,yp,z,nx,ny)] + a[idx3(x,ym,z,nx,ny)] - 2*a[i]
         + a[idx3(x,y,zp,nx,ny)] + a[idx3(x,y,zm,nx,ny)] - 2*a[i];
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
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
            if (rank==0) printf("Usage: %s [-x num] [-y num] [-z num] [-i num] [-v] [-r] [-h]\n",argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if(rank==0) fprintf(stderr,"Unknown option: %s\n",argv[i]);
            MPI_Finalize(); return 1;
        }
    }
    if (!ny) ny=nx;
    if (!nz) nz=nx;
    if (!nx || !ny || !nz || size > static_cast<int>(nz)) {
        if(rank==0) fprintf(stderr,"Grid dimensions must be positive and MPI ranks must not exceed Z size\n");
        MPI_Finalize(); return 1;
    }
    const size_t plane=nx*ny;
    const size_t base=nz/size, rem=nz%size;
    const size_t localZ=base+(static_cast<size_t>(rank)<rem);
    const size_t startZ=static_cast<size_t>(rank)*base+std::min(static_cast<size_t>(rank),rem);
    const size_t count=localZ*plane;
    // Two halo planes are included; owned planes occupy [1, localZ].
    std::vector<double> c((localZ+2)*plane), next(c.size()), mu(c.size());
    for(size_t z=0; z<localZ; ++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
        size_t linear=(startZ+z)*plane+y*nx+x;
        size_t pseudo=((linear+1)*1299709ULL)%(nx*ny*nz);
        c[idx3(x,y,z+1,nx,ny)]=-1.0+2.0*pseudo/static_cast<double>(nx*ny*nz);
    }
    if(rank==0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",nx,ny,nz,iterations,validate?"enabled":"disabled");
        printf("Initializing concentration field...\nRunning Cahn-Hilliard simulation...\n");
    }
    const double dt=.01,gamma=.5,D=1,eAA=-2.0/9,eBB=-2.0/9,eAB=2.0/9;
    MPI_Barrier(MPI_COMM_WORLD);
    auto begin=std::chrono::high_resolution_clock::now();
    for(int t=0;t<iterations;++t) {
        exchangeHalos(c,plane,static_cast<int>(localZ),rank,size);
        for(size_t z=1;z<=localZ;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
            size_t i=idx3(x,y,z,nx,ny); double v=c[i];
            mu[i]=4.5*((v+1)*eAA+(v-1)*eBB-2*v*eAB)+3*v+v*v*v-gamma*laplacian(c,nx,ny,x,y,z,startZ+z-1,nz);
        }
        exchangeHalos(mu,plane,static_cast<int>(localZ),rank,size);
        for(size_t z=1;z<=localZ;++z) for(size_t y=0;y<ny;++y) for(size_t x=0;x<nx;++x) {
            size_t i=idx3(x,y,z,nx,ny);
            next[i]=c[i]+dt*D*laplacian(mu,nx,ny,x,y,z,startZ+z-1,nz);
        }
        c.swap(next);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end=std::chrono::high_resolution_clock::now();
    double localSeconds=std::chrono::duration<double>(end-begin).count(), seconds;
    MPI_Reduce(&localSeconds,&seconds,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    int n=static_cast<int>(count);
    std::vector<int> counts, offsets;
    if(rank==0) { counts.resize(size); offsets.resize(size); }
    int ni=n;
    MPI_Gather(&ni,1,MPI_INT,rank==0?counts.data():nullptr,1,MPI_INT,0,MPI_COMM_WORLD);
    if(rank==0) { int off=0; for(int r=0;r<size;++r){ offsets[r]=off; off+=counts[r]; } }
    std::vector<double> result;
    if(rank==0) result.resize(nx*ny*nz);
    MPI_Gatherv(c.data()+plane,n,MPI_DOUBLE,rank==0?result.data():nullptr,
                rank==0?counts.data():nullptr,rank==0?offsets.data():nullptr,MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(rank==0) {
        printf("Computation time: %ld ms\n",static_cast<long>(seconds*1000));
        printf("Performance: %.3f MCellUpdates/s\n",seconds>0?static_cast<double>(nx*ny*nz)*iterations/seconds/1e6:0.0);
        if(printResults) print_results(result,"Concentration");
        if(validate) {
            double lo=*std::min_element(result.begin(),result.end()), hi=*std::max_element(result.begin(),result.end());
            bool valid=std::all_of(result.begin(),result.end(),[](double v){return std::isfinite(v);})&&lo>=-10&&hi<=10;
            printf("Concentration range: [%.6f, %.6f]\nValidation: %s\n",lo,hi,valid?"PASSED":"FAILED");
            MPI_Finalize(); return valid?0:1;
        }
    }
    MPI_Finalize();
    return 0;
}
