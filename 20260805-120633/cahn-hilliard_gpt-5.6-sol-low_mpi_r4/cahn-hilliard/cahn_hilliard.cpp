#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

inline size_t at(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return (z * ny + y) * nx + x;
}

static void exchange_planes(std::vector<double>& a, size_t local_nz, size_t plane,
                            int rank, int ranks, MPI_Comm comm) {
    const int lo = rank == 0 ? MPI_PROC_NULL : rank - 1;
    const int hi = rank + 1 == ranks ? MPI_PROC_NULL : rank + 1;
    MPI_Request req[4];
    MPI_Irecv(a.data(), static_cast<int>(plane), MPI_DOUBLE, lo, 11, comm, &req[0]);
    MPI_Irecv(a.data() + (local_nz + 1) * plane, static_cast<int>(plane), MPI_DOUBLE,
              hi, 10, comm, &req[1]);
    MPI_Isend(a.data() + plane, static_cast<int>(plane), MPI_DOUBLE, lo, 10, comm, &req[2]);
    MPI_Isend(a.data() + local_nz * plane, static_cast<int>(plane), MPI_DOUBLE, hi, 11, comm, &req[3]);
    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

    // MPI_PROC_NULL leaves the receive buffer untouched: copy the boundary plane
    // to reproduce the original clamped (zero normal derivative) boundary.
    if (lo == MPI_PROC_NULL)
        std::copy_n(a.data() + plane, plane, a.data());
    if (hi == MPI_PROC_NULL)
        std::copy_n(a.data() + local_nz * plane, plane,
                    a.data() + (local_nz + 1) * plane);
}

static void chemical_potential(const std::vector<double>& c, std::vector<double>& mu,
                               size_t nx, size_t ny, size_t local_nz,
                               double gamma, double e_AA, double e_BB, double e_AB) {
    const size_t plane = nx * ny;
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t ym = y ? y - 1 : y;
            const size_t yp = y + 1 < ny ? y + 1 : y;
            for (size_t x = 0; x < nx; ++x) {
                const size_t xm = x ? x - 1 : x;
                const size_t xp = x + 1 < nx ? x + 1 : x;
                const size_t i = at(x, y, z, nx, ny);
                const double v = c[i];
                const double lap = c[at(xm,y,z,nx,ny)] + c[at(xp,y,z,nx,ny)]
                                 + c[at(x,ym,z,nx,ny)] + c[at(x,yp,z,nx,ny)]
                                 + c[i-plane] + c[i+plane] - 6.0*v;
                mu[i] = 4.5 * ((v + 1.0)*e_AA + (v - 1.0)*e_BB - 2.0*v*e_AB)
                        + 3.0*v + v*v*v - gamma*lap;
            }
        }
    }
}

static void update(std::vector<double>& out, const std::vector<double>& in,
                   const std::vector<double>& mu, size_t nx, size_t ny,
                   size_t local_nz, double dtD) {
    const size_t plane = nx * ny;
    for (size_t z = 1; z <= local_nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t ym = y ? y - 1 : y;
            const size_t yp = y + 1 < ny ? y + 1 : y;
            for (size_t x = 0; x < nx; ++x) {
                const size_t xm = x ? x - 1 : x;
                const size_t xp = x + 1 < nx ? x + 1 : x;
                const size_t i = at(x,y,z,nx,ny);
                const double lap = mu[at(xm,y,z,nx,ny)] + mu[at(xp,y,z,nx,ny)]
                                 + mu[at(x,ym,z,nx,ny)] + mu[at(x,yp,z,nx,ny)]
                                 + mu[i-plane] + mu[i+plane] - 6.0*mu[i];
                out[i] = in[i] + dtD*lap;
            }
        }
    }
}

static void usage(const char* p) {
    std::printf("Usage: %s [options]\n  -x <num>  Grid X (default 64)\n"
                "  -y <num>  Grid Y (default X)\n  -z <num>  Grid Z (default X)\n"
                "  -i <num>  Time steps (default 20)\n  -v  Validate\n"
                "  -r  Print results for external validation\n  -h  Help\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t nx=64, ny=0, nz=0; int iterations=20;
    bool validate=false, printResults=false, help=false, bad=false;
    for (int i=1; i<argc; ++i) {
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
    const bool dimensions_bad = !nx || !ny || !nz || nx > size_t(INT_MAX)
        || ny > size_t(INT_MAX) / nx || nx * ny > size_t(INT_MAX)
        || nz > size_t(INT_MAX);
    if (help || bad || dimensions_bad || iterations < 0) {
        if (world_rank==0) { if (bad) std::printf("Invalid option\n"); usage(argv[0]); }
        MPI_Finalize(); return bad ? 1 : 0;
    }

    // At most nz ranks have useful slabs. Extra launched ranks leave cleanly.
    const int active_size = std::min<int>(world_size, static_cast<int>(nz));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, world_rank < active_size ? 0 : MPI_UNDEFINED,
                   world_rank, &comm);
    if (comm == MPI_COMM_NULL) { MPI_Finalize(); return 0; }
    const int rank=world_rank, ranks=active_size;
    const size_t base=nz/ranks, rem=nz%ranks;
    const size_t local_nz=base + (size_t(rank)<rem);
    const size_t z0=size_t(rank)*base + std::min<size_t>(rank,rem);
    const size_t plane=nx*ny, local_count=local_nz*plane;
    if (local_count > size_t(INT_MAX)) {
        if (rank==0) std::fprintf(stderr,"Local MPI message exceeds INT_MAX elements\n");
        MPI_Abort(comm, 2);
    }

    if (rank==0) {
        std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n",nx,ny,nz);
        std::printf("Time steps: %d\nValidation: %s\nMPI ranks: %d\nInitializing concentration field...\n",
                    iterations,validate?"enabled":"disabled",ranks);
    }
    std::vector<double> cold((local_nz+2)*plane), cnew((local_nz+2)*plane), mu((local_nz+2)*plane);
    const size_t volume=nx*ny*nz;
    for (size_t lz=1; lz<=local_nz; ++lz)
        for (size_t q=0; q<plane; ++q) {
            const size_t gid=(z0+lz-1)*plane+q;
            const double pseudo=(((gid+1)*size_t(1299709))%volume)/double(volume);
            cold[lz*plane+q]=-1.0+2.0*pseudo;
        }

    if (rank==0) std::printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(comm); const double start=MPI_Wtime();
    constexpr double e_AA=-(2.0/9.0), e_BB=-(2.0/9.0), e_AB=2.0/9.0;
    for (int t=0; t<iterations; ++t) {
        exchange_planes(cold,local_nz,plane,rank,ranks,comm);
        chemical_potential(cold,mu,nx,ny,local_nz,0.5,e_AA,e_BB,e_AB);
        exchange_planes(mu,local_nz,plane,rank,ranks,comm);
        update(cnew,cold,mu,nx,ny,local_nz,0.01);
        cold.swap(cnew);
    }
    const double elapsed=MPI_Wtime()-start, seconds=[](double local, MPI_Comm c) {
        double global=0.0; MPI_Reduce(&local,&global,1,MPI_DOUBLE,MPI_MAX,0,c); return global;
    }(elapsed,comm);
    if (rank==0) {
        std::printf("Computation time: %ld ms\n",long(seconds*1000.0));
        const double mcups=seconds>0 ? double(volume)*iterations/seconds/1e6 : 0.0;
        std::printf("Performance: %.3f MCellUpdates/s\n",mcups);
    }

    if (printResults) {
        std::vector<int> counts, displs; std::vector<double> global;
        if (rank==0) { counts.resize(ranks); displs.resize(ranks); global.resize(volume); }
        const int n=static_cast<int>(local_count);
        MPI_Gather(&n,1,MPI_INT,rank==0?counts.data():nullptr,1,MPI_INT,0,comm);
        if (rank==0) { displs[0]=0; for (int r=1;r<ranks;++r) displs[r]=displs[r-1]+counts[r-1]; }
        MPI_Gatherv(cold.data()+plane,n,MPI_DOUBLE,rank==0?global.data():nullptr,
                    rank==0?counts.data():nullptr,rank==0?displs.data():nullptr,MPI_DOUBLE,0,comm);
        if (rank==0) print_results(global,"Concentration");
    }

    int result=0;
    if (validate) {
        bool finite=true; double lmin=std::numeric_limits<double>::infinity(), lmax=-lmin;
        for (size_t i=plane;i<plane+local_count;++i) {
            finite &= std::isfinite(cold[i]); lmin=std::min(lmin,cold[i]); lmax=std::max(lmax,cold[i]);
        }
        int lok=finite, gok; double gmin,gmax;
        MPI_Reduce(&lok,&gok,1,MPI_INT,MPI_LAND,0,comm);
        MPI_Reduce(&lmin,&gmin,1,MPI_DOUBLE,MPI_MIN,0,comm);
        MPI_Reduce(&lmax,&gmax,1,MPI_DOUBLE,MPI_MAX,0,comm);
        if (rank==0) {
            std::printf("Validating result...\nConcentration range: [%.6f, %.6f]\n",gmin,gmax);
            result = (!gok || gmax>10.0 || gmin< -10.0);
            std::printf("Validation: %s\n",result?"FAILED":"PASSED");
        }
        MPI_Bcast(&result,1,MPI_INT,0,comm);
    }
    MPI_Comm_free(&comm); MPI_Finalize(); return result;
}
