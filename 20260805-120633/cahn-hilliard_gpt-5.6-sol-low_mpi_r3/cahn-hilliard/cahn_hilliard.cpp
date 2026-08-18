#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

inline size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return (z * ny + y) * nx + x;
}

struct Domain {
    size_t nx, ny, nz, local_nz, z0, plane;
    int rank, size;
};

static void exchange_z(std::vector<double>& a, const Domain& d) {
    const int lower = d.rank == 0 ? MPI_PROC_NULL : d.rank - 1;
    const int upper = d.rank + 1 == d.size ? MPI_PROC_NULL : d.rank + 1;
    const int count = static_cast<int>(d.plane);

    MPI_Sendrecv(a.data() + d.plane, count, MPI_DOUBLE, lower, 10,
                 a.data() + (d.local_nz + 1) * d.plane, count, MPI_DOUBLE, upper, 10,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(a.data() + d.local_nz * d.plane, count, MPI_DOUBLE, upper, 11,
                 a.data(), count, MPI_DOUBLE, lower, 11,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // MPI_PROC_NULL leaves the receive buffer untouched: implement the original
    // clamped (zero normal derivative) boundary by copying the boundary plane.
    if (lower == MPI_PROC_NULL)
        std::copy_n(a.data() + d.plane, d.plane, a.data());
    if (upper == MPI_PROC_NULL)
        std::copy_n(a.data() + d.local_nz * d.plane, d.plane,
                    a.data() + (d.local_nz + 1) * d.plane);
}

static void chemical_potential(const std::vector<double>& c, std::vector<double>& mu,
                               const Domain& d, double gamma, double eAA,
                               double eBB, double eAB) {
    const size_t nx = d.nx, ny = d.ny, p = d.plane;
    for (size_t z = 1; z <= d.local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t yn = y ? y - 1 : y;
            const size_t yp = y + 1 < ny ? y + 1 : y;
            for (size_t x = 0; x < nx; ++x) {
                const size_t xn = x ? x - 1 : x;
                const size_t xp = x + 1 < nx ? x + 1 : x;
                const size_t i = idx3(x, y, z, nx, ny);
                const double v = c[i];
                const double lap = c[idx3(xp,y,z,nx,ny)] + c[idx3(xn,y,z,nx,ny)] - 2.0*v
                                 + c[idx3(x,yp,z,nx,ny)] + c[idx3(x,yn,z,nx,ny)] - 2.0*v
                                 + c[i+p] + c[i-p] - 2.0*v;
                mu[i] = 4.5 * ((v + 1.0)*eAA + (v - 1.0)*eBB - 2.0*v*eAB)
                      + 3.0*v + v*v*v - gamma*lap;
            }
        }
    }
}

static void update(std::vector<double>& out, const std::vector<double>& old,
                   const std::vector<double>& mu, const Domain& d, double dtD) {
    const size_t nx = d.nx, ny = d.ny, p = d.plane;
    for (size_t z = 1; z <= d.local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t yn = y ? y - 1 : y;
            const size_t yp = y + 1 < ny ? y + 1 : y;
            for (size_t x = 0; x < nx; ++x) {
                const size_t xn = x ? x - 1 : x;
                const size_t xp = x + 1 < nx ? x + 1 : x;
                const size_t i = idx3(x,y,z,nx,ny);
                const double v = mu[i];
                const double lap = mu[idx3(xp,y,z,nx,ny)] + mu[idx3(xn,y,z,nx,ny)] - 2.0*v
                                 + mu[idx3(x,yp,z,nx,ny)] + mu[idx3(x,yn,z,nx,ny)] - 2.0*v
                                 + mu[i+p] + mu[i-p] - 2.0*v;
                out[i] = old[i] + dtD*lap;
            }
        }
    }
}

static void initialize(std::vector<double>& c, const Domain& d) {
    const size_t volume = d.nx*d.ny*d.nz;
    for (size_t z = 0; z < d.local_nz; ++z)
        for (size_t y = 0; y < d.ny; ++y)
            for (size_t x = 0; x < d.nx; ++x) {
                const size_t global = (d.z0 + z)*d.plane + y*d.nx + x;
                const double pseudo = (((global + 1)*size_t{1299709}) % volume)
                                    / static_cast<double>(volume);
                c[idx3(x,y,z+1,d.nx,d.ny)] = -1.0 + 2.0*pseudo;
            }
}

static void usage(const char* p) {
    std::printf("Usage: %s [options]\n  -x <num>  Grid X (default 64)\n"
                "  -y <num>  Grid Y (default X)\n  -z <num>  Grid Z (default X)\n"
                "  -i <num>  Time steps (default 20)\n  -v  Validate\n"
                "  -r  Print results\n  -h  Show help\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t nx=64, ny=0, nz=0;
    int iterations=20;
    bool validate=false, results=false, help=false, bad=false;
    for (int i=1; i<argc; ++i) {
        if (!std::strcmp(argv[i],"-x") && i+1<argc) nx=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-y") && i+1<argc) ny=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-z") && i+1<argc) nz=std::strtoull(argv[++i],nullptr,10);
        else if (!std::strcmp(argv[i],"-i") && i+1<argc) iterations=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"-v")) validate=true;
        else if (!std::strcmp(argv[i],"-r")) results=true;
        else if (!std::strcmp(argv[i],"-h")) help=true;
        else { if (rank==0) std::printf("Unknown option: %s\n",argv[i]); bad=true; }
    }
    if (!ny) ny=nx;
    if (!nz) nz=nx;
    if (help || bad) {
        if (rank==0) usage(argv[0]);
        MPI_Finalize(); return bad ? 1 : 0;
    }
    if (!nx || !ny || !nz || iterations < 0 || static_cast<size_t>(nranks)>nz ||
        nx > static_cast<size_t>(std::numeric_limits<int>::max()) / ny) {
        if (rank==0) std::fprintf(stderr,"Invalid grid/iteration count, or MPI ranks exceed Z planes\n");
        MPI_Finalize(); return 1;
    }

    const size_t base=nz/nranks, rem=nz%nranks;
    const size_t local=base+(static_cast<size_t>(rank)<rem);
    const size_t z0=static_cast<size_t>(rank)*base+std::min(static_cast<size_t>(rank),rem);
    Domain d{nx,ny,nz,local,z0,nx*ny,rank,nranks};
    std::vector<double> cold((local+2)*d.plane), cnew((local+2)*d.plane), mu((local+2)*d.plane);

    if (rank==0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n",nx,ny,nz);
        std::printf("Time steps: %d\nValidation: %s\nMPI processes: %d\n",iterations,validate?"enabled":"disabled",nranks);
        std::printf("Initializing concentration field...\n");
    }
    initialize(cold,d);
    if (rank==0) std::printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start=MPI_Wtime();
    for (int t=0; t<iterations; ++t) {
        exchange_z(cold,d);
        chemical_potential(cold,mu,d,0.5,-2.0/9.0,-2.0/9.0,2.0/9.0);
        exchange_z(mu,d);
        update(cnew,cold,mu,d,0.01);
        cold.swap(cnew);
    }
    const double local_time=MPI_Wtime()-start;
    double elapsed=0;
    MPI_Reduce(&local_time,&elapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if (rank==0) {
        const long ms=static_cast<long>(elapsed*1000.0);
        std::printf("Computation time: %ld ms\n",ms);
        const double mcups = elapsed>0 ? (static_cast<double>(nx)*ny*nz*iterations/elapsed/1e6) : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n",mcups);
    }

    if (results) {
        std::vector<int> counts(nranks), displs(nranks);
        for (int r=0; r<nranks; ++r) {
            const size_t rn=base+(static_cast<size_t>(r)<rem);
            const size_t rz=static_cast<size_t>(r)*base+std::min(static_cast<size_t>(r),rem);
            counts[r]=static_cast<int>(rn*d.plane); displs[r]=static_cast<int>(rz*d.plane);
        }
        std::vector<double> global(rank==0 ? nx*ny*nz : 0);
        MPI_Gatherv(cold.data()+d.plane,counts[rank],MPI_DOUBLE,global.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
        if (rank==0) print_results(global,"Concentration");
    }
    int valid=1;
    if (validate) {
        double local_min=std::numeric_limits<double>::infinity(), local_max=-local_min;
        for (size_t i=d.plane; i<(local+1)*d.plane; ++i) {
            if (!std::isfinite(cold[i])) valid=0;
            local_min=std::min(local_min,cold[i]); local_max=std::max(local_max,cold[i]);
        }
        int all_valid=0; double minv,maxv;
        MPI_Reduce(&valid,&all_valid,1,MPI_INT,MPI_LAND,0,MPI_COMM_WORLD);
        MPI_Reduce(&local_min,&minv,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
        MPI_Reduce(&local_max,&maxv,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        if (rank==0) {
            std::printf("Validating result...\nConcentration range: [%.6f, %.6f]\n",minv,maxv);
            valid=all_valid && minv>=-10.0 && maxv<=10.0;
            std::printf("Validation: %s\n",valid?"PASSED":"FAILED");
        }
        MPI_Bcast(&valid,1,MPI_INT,0,MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return valid ? 0 : 1;
}
