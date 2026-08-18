#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

static inline size_t index3(size_t x, size_t y, size_t z, size_t sx, size_t sy) {
    return (z * sy + y) * sx + x;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

// Divide n elements evenly; the first n%p ranks own one extra element.
static void block(size_t n, int p, int c, size_t& begin, size_t& count) {
    const size_t q = n / static_cast<size_t>(p), r = n % static_cast<size_t>(p);
    count = q + (static_cast<size_t>(c) < r);
    begin = static_cast<size_t>(c) * q + std::min(static_cast<size_t>(c), r);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); parseStatus = 2; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } parseStatus = 1; }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (parseStatus) { MPI_Finalize(); return parseStatus == 2 ? 0 : 1; }
    if (nx < 2 || ny < 2 || nz < 2 || iterations < 0 ||
        nx > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        ny > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        nz > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::fprintf(stderr, "Grid dimensions must be in [2, INT_MAX] and iterations nonnegative.\n");
        MPI_Finalize(); return 1;
    }

    int dims[3] = {0, 0, 0};
    MPI_Dims_create(nranks, 3, dims);
    // Dims_create is shape-agnostic. Pair large process dimensions with large grid dimensions.
    size_t ext[3] = {nx, ny, nz};
    for (int i = 0; i < 3; ++i) for (int j = i + 1; j < 3; ++j)
        if ((ext[i] < ext[j]) != (dims[i] < dims[j])) std::swap(dims[i], dims[j]);
    if (static_cast<size_t>(dims[0]) > nx || static_cast<size_t>(dims[1]) > ny || static_cast<size_t>(dims[2]) > nz) {
        if (rank == 0) std::fprintf(stderr, "Too many MPI ranks for this grid decomposition.\n");
        MPI_Finalize(); return 1;
    }

    MPI_Comm cart;
    int periods[3] = {0, 0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart);
    int coord[3]; MPI_Cart_coords(cart, rank, 3, coord);
    size_t ox, oy, oz, lx, ly, lz;
    block(nx, dims[0], coord[0], ox, lx);
    block(ny, dims[1], coord[1], oy, ly);
    block(nz, dims[2], coord[2], oz, lz);
    const size_t sx = lx + 2, sy = ly + 2;
    std::vector<Real> a(sx * sy * (lz + 2)), b(a.size());
    for (size_t z = 1; z <= lz; ++z) for (size_t y = 1; y <= ly; ++y)
        for (size_t x = 1; x <= lx; ++x) {
            const size_t global = ((oz + z - 1) * ny + oy + y - 1) * nx + ox + x - 1;
            a[index3(x, y, z, sx, sy)] = static_cast<Real>(global % 19);
        }

    int lo[3], hi[3];
    for (int d = 0; d < 3; ++d) MPI_Cart_shift(cart, d, 1, &lo[d], &hi[d]);
    int sizes[3] = {static_cast<int>(lz + 2), static_cast<int>(ly + 2), static_cast<int>(lx + 2)};
    MPI_Datatype face[3];
    for (int d = 0; d < 3; ++d) {
        int subs[3] = {static_cast<int>(lz), static_cast<int>(ly), static_cast<int>(lx)};
        subs[2 - d] = 1;
        // The buffer passed to MPI already points at the first face element.
        int starts[3] = {0, 0, 0};
        MPI_Type_create_subarray(3, sizes, subs, starts, MPI_ORDER_C, MPI_DOUBLE, &face[d]);
        MPI_Type_commit(&face[d]);
    }

    if (rank == 0) {
        std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\n", nx, ny, nz, iterations);
        std::printf("Validation: %s\nInitializing grid...\nRunning stencil computation...\n", validate ? "enabled" : "disabled");
    }
    MPI_Barrier(cart);
    const double start = MPI_Wtime();
    Real* in = a.data(); Real* out = b.data();
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Request req[12]; int nreq = 0;
        const size_t sendLo[3] = {index3(1,1,1,sx,sy), index3(1,1,1,sx,sy), index3(1,1,1,sx,sy)};
        const size_t sendHi[3] = {index3(lx,1,1,sx,sy), index3(1,ly,1,sx,sy), index3(1,1,lz,sx,sy)};
        const size_t recvLo[3] = {index3(0,1,1,sx,sy), index3(1,0,1,sx,sy), index3(1,1,0,sx,sy)};
        const size_t recvHi[3] = {index3(lx+1,1,1,sx,sy), index3(1,ly+1,1,sx,sy), index3(1,1,lz+1,sx,sy)};
        for (int d = 0; d < 3; ++d) {
            MPI_Irecv(in + recvLo[d], 1, face[d], lo[d], 2*d+1, cart, &req[nreq++]);
            MPI_Irecv(in + recvHi[d], 1, face[d], hi[d], 2*d, cart, &req[nreq++]);
            MPI_Isend(in + sendLo[d], 1, face[d], lo[d], 2*d, cart, &req[nreq++]);
            MPI_Isend(in + sendHi[d], 1, face[d], hi[d], 2*d+1, cart, &req[nreq++]);
        }
        auto update = [&](size_t x, size_t y, size_t z) {
            const size_t gx=ox+x-1, gy=oy+y-1, gz=oz+z-1, q=index3(x,y,z,sx,sy);
            if (gx==0 || gx+1==nx || gy==0 || gy+1==ny || gz==0 || gz+1==nz) out[q]=in[q];
            else out[q]=(in[q]+in[q-1]+in[q+1]+in[q-sx]+in[q+sx]+in[q-sx*sy]+in[q+sx*sy])*(1.0/7.0);
        };
        // The strict block interior has no dependency on incoming halo data.
        for (size_t z=2; z<lz; ++z) for (size_t y=2; y<ly; ++y) for (size_t x=2; x<lx; ++x) update(x,y,z);
        MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
        for (size_t z=1; z<=lz; ++z) for (size_t y=1; y<=ly; ++y) for (size_t x=1; x<=lx; ++x)
            if (!(x>1 && x<lx && y>1 && y<ly && z>1 && z<lz)) update(x,y,z);
        std::swap(in, out);
    }
    const double localTime = MPI_Wtime() - start;
    double elapsed; MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        const long ms = static_cast<long>(elapsed * 1000.0);
        const double updates = static_cast<double>((nx-2)*(ny-2)*(nz-2))*iterations;
        std::printf("Computation time: %ld ms\nPerformance: %.3f MCellUpdates/s\n", ms, elapsed > 0 ? updates/elapsed/1e6 : 0.0);
    }

    if (printResults) {
        std::vector<Real> packed(lx*ly*lz); size_t p=0;
        for(size_t z=1;z<=lz;++z) for(size_t y=1;y<=ly;++y) for(size_t x=1;x<=lx;++x) packed[p++]=in[index3(x,y,z,sx,sy)];
        int mine=static_cast<int>(packed.size());
        std::vector<int> counts(rank==0?nranks:0), displs(rank==0?nranks:0);
        MPI_Gather(&mine,1,MPI_INT,rank==0?counts.data():nullptr,1,MPI_INT,0,cart);
        std::vector<Real> gathered;
        if(rank==0){ for(int r=1;r<nranks;++r) displs[r]=displs[r-1]+counts[r-1]; gathered.resize(nx*ny*nz); }
        MPI_Gatherv(packed.data(),mine,MPI_DOUBLE,rank==0?gathered.data():nullptr,rank==0?counts.data():nullptr,rank==0?displs.data():nullptr,MPI_DOUBLE,0,cart);
        if(rank==0){
            std::vector<Real> global(nx*ny*nz); size_t base=0;
            for(int r=0;r<nranks;++r){ int c[3]; MPI_Cart_coords(cart,r,3,c); size_t bx,by,bz,cx,cy,cz; block(nx,dims[0],c[0],bx,cx); block(ny,dims[1],c[1],by,cy); block(nz,dims[2],c[2],bz,cz); size_t q=base;
                for(size_t z=0;z<cz;++z) for(size_t y=0;y<cy;++y) for(size_t x=0;x<cx;++x)
                    global[((bz+z)*ny+by+y)*nx+bx+x]=gathered[q++];
                base+=counts[r];
            }
            print_results(global,"Grid");
        }
    }

    int localBad=0; Real localMin=std::numeric_limits<Real>::infinity(), localMax=-localMin;
    if(validate) for(size_t z=1;z<=lz;++z) for(size_t y=1;y<=ly;++y) for(size_t x=1;x<=lx;++x){ Real v=in[index3(x,y,z,sx,sy)]; localBad |= !std::isfinite(v); localMin=std::min(localMin,v); localMax=std::max(localMax,v); }
    int bad=0; Real minv=0,maxv=0;
    if(validate){ MPI_Reduce(&localBad,&bad,1,MPI_INT,MPI_LOR,0,cart); MPI_Reduce(&localMin,&minv,1,MPI_DOUBLE,MPI_MIN,0,cart); MPI_Reduce(&localMax,&maxv,1,MPI_DOUBLE,MPI_MAX,0,cart); }
    int rc=0;
    if(rank==0 && validate){ std::printf("Validating result...\nValue range: [%.6f, %.6f]\n",minv,maxv); if(bad) std::printf("Validation failed: found NaN or Inf value\n"); if(maxv>1e6||minv< -1e6) { std::printf("Validation failed: values out of expected range\n"); bad=1; } std::printf("Validation: %s\n",bad?"FAILED":"PASSED"); rc=bad; }
    MPI_Bcast(&rc,1,MPI_INT,0,cart);
    for(auto& t:face) MPI_Type_free(&t);
    MPI_Comm_free(&cart); MPI_Finalize(); return rc;
}
