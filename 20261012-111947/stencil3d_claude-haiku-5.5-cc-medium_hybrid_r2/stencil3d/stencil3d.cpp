#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error: %s at %s:%d: %s\n", #call, __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Local block index: owned cells are 1..ln on each axis, 0 and ln+1 are halo cells
__host__ __device__ inline size_t localIdx(const int x, const int y, const int z, const int ax, const int ay) noexcept {
    return (static_cast<size_t>(z) * ay + y) * ax + x;
}

// Global problem geometry and the part of it owned by one rank
struct Domain {
    long gnx, gny, gnz;     // global grid size
    long offX, offY, offZ;  // global coordinates of owned local cell (1,1,1)
    int lnx, lny, lnz;      // owned cells per axis
    int ax, ay, az;         // allocated extents including one halo layer per side
};

// Half-open range of local owned indices [x0,x1) x [y0,y1) x [z0,z1)
struct Box {
    int x0, x1, y0, y1, z0, z1;
};

// Faces of the local block: 0/1 = -x/+x, 2/3 = -y/+y, 4/5 = -z/+z
struct Face {
    size_t sendBase;          // local index of the first owned cell of the boundary plane sent out
    size_t recvBase;          // local index of the first halo cell of the plane written by the neighbour
    size_t uStride, vStride;  // strides of the two in-plane axes
    size_t nu, nv;            // in-plane extents
    size_t offset;            // position of this face inside the packed halo buffers
    size_t count;             // nu * nv, or 0 when there is no neighbour on this side
};

struct FaceTable {
    Face f[6];
};

// 7-point stencil over a box of owned cells, one thread per cell
__global__ void stencilKernel(const Real* __restrict__ in, Real* __restrict__ out, const Box box, const Domain dom) {
    const int x = box.x0 + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = box.y0 + static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int z = box.z0 + static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
    if (x >= box.x1 || y >= box.y1 || z >= box.z1) return;

    const size_t c = localIdx(x, y, z, dom.ax, dom.ay);
    const long gx = dom.offX + x - 1;
    const long gy = dom.offY + y - 1;
    const long gz = dom.offZ + z - 1;

    // Global boundary cells are copied unchanged
    if (gx == 0 || gx == dom.gnx - 1 || gy == 0 || gy == dom.gny - 1 || gz == 0 || gz == dom.gnz - 1) {
        out[c] = in[c];
        return;
    }

    const size_t sy = dom.ax;
    const size_t sz = static_cast<size_t>(dom.ax) * dom.ay;
    const Real center = in[c];
    const Real left = in[c - 1];
    const Real right = in[c + 1];
    const Real front = in[c - sy];
    const Real back = in[c + sy];
    const Real bottom = in[c - sz];
    const Real top = in[c + sz];
    // Same evaluation order as the reference; no fast-math so the division stays IEEE-exact
    out[c] = (center + left + right + front + back + bottom + top) / 7.0;
}

// Packs (Unpack = false) or unpacks (Unpack = true) all active faces in one launch
template <bool Unpack>
__global__ void haloCopyKernel(Real* __restrict__ field, Real* __restrict__ buf, const FaceTable tab, const size_t total) {
    const size_t t = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (t >= total) return;

    int face = 0;
    while (face < 5 && t >= tab.f[face + 1].offset) ++face;
    const Face& f = tab.f[face];

    const size_t rel = t - f.offset;
    const size_t off = (rel % f.nu) * f.uStride + (rel / f.nu) * f.vStride;
    if constexpr (Unpack) {
        field[f.recvBase + off] = buf[t];
    } else {
        buf[t] = field[f.sendBase + off];
    }
}

// Block partition of n cells over p ranks: coordinate c owns [start, start + len)
void blockRange(const int c, const int p, const long n, long& start, long& len) {
    start = c * n / p;
    len = (c + 1) * n / p - start;
}

// Splits the owned cells into an inner box that needs no halo data and six shell boxes that do
void partitionBlock(const Domain& d, Box& inner, Box shell[6]) {
    const int xi = std::max(2, d.lnx);
    const int yi = std::max(2, d.lny);
    const int zi = std::max(2, d.lnz);
    inner = Box{2, xi, 2, yi, 2, zi};
    shell[0] = Box{1, 2, 1, d.lny + 1, 1, d.lnz + 1};
    shell[1] = Box{xi, d.lnx + 1, 1, d.lny + 1, 1, d.lnz + 1};
    shell[2] = Box{2, xi, 1, 2, 1, d.lnz + 1};
    shell[3] = Box{2, xi, yi, d.lny + 1, 1, d.lnz + 1};
    shell[4] = Box{2, xi, 2, yi, 1, 2};
    shell[5] = Box{2, xi, 2, yi, zi, d.lnz + 1};
}

// One rank's share of the stencil: owned block on the GPU, halo exchange through pinned host buffers
class StencilSolver {
  public:
    StencilSolver(MPI_Comm cart, int rank, const int dims[3], const int coords[3], const long n[3], const int nbr[6]);
    ~StencilSolver();
    StencilSolver(const StencilSolver&) = delete;
    StencilSolver& operator=(const StencilSolver&) = delete;

    void initialize();
    void run(int iterations);
    Real* finalBuffer(int iterations) const { return iterations % 2 == 0 ? grid_[0] : grid_[1]; }
    // Assembles the global grid on rank 0 (empty vector on other ranks)
    std::vector<Real> gather(const Real* field) const;

  private:
    void exchangeHalos(Real* field);
    void launchStencil(const Box& b, const Real* in, Real* out, cudaStream_t s) const;

    MPI_Comm cart_;
    int rank_;
    int dims_[3];
    int coords_[3];
    int nbr_[6];
    Domain dom_;
    Box inner_;
    Box shell_[6];
    FaceTable faces_;
    size_t halo_ = 0;  // total number of values in the packed halo buffers

    Real* grid_[2] = {nullptr, nullptr};
    Real* sendDev_ = nullptr;
    Real* recvDev_ = nullptr;
    Real* sendHost_ = nullptr;
    Real* recvHost_ = nullptr;

    cudaStream_t compute_ = nullptr;
    cudaStream_t comm_ = nullptr;
    cudaEvent_t shellDone_ = nullptr;   // boundary planes of the output are final
    cudaEvent_t haloReady_ = nullptr;   // halo cells of the output have been unpacked
};

StencilSolver::StencilSolver(MPI_Comm cart, int rank, const int dims[3], const int coords[3], const long n[3], const int nbr[6])
    : cart_(cart), rank_(rank) {
    long off[3];
    int ln[3];
    for (int a = 0; a < 3; ++a) {
        dims_[a] = dims[a];
        coords_[a] = coords[a];
        long start = 0, len = 0;
        blockRange(coords[a], dims[a], n[a], start, len);
        off[a] = start;
        ln[a] = static_cast<int>(len);
    }
    for (int f = 0; f < 6; ++f) nbr_[f] = nbr[f];

    dom_ = Domain{n[0], n[1], n[2], off[0], off[1], off[2], ln[0], ln[1], ln[2], ln[0] + 2, ln[1] + 2, ln[2] + 2};
    partitionBlock(dom_, inner_, shell_);

    // Halo faces, packed back to back; a face without neighbour is empty
    const size_t extent[3] = {static_cast<size_t>(ln[0]), static_cast<size_t>(ln[1]), static_cast<size_t>(ln[2])};
    const size_t stride[3] = {1, static_cast<size_t>(dom_.ax), static_cast<size_t>(dom_.ax) * dom_.ay};
    size_t offset = 0;
    for (int f = 0; f < 6; ++f) {
        const int a = f / 2;
        const int s = f % 2;
        const int u = (a + 1) % 3;
        const int v = (a + 2) % 3;
        const size_t sendPlane = (s == 0) ? 1 : extent[a];
        const size_t ghostPlane = (s == 0) ? 0 : extent[a] + 1;
        Face& face = faces_.f[f];
        face.sendBase = sendPlane * stride[a] + stride[u] + stride[v];
        face.recvBase = ghostPlane * stride[a] + stride[u] + stride[v];
        face.uStride = stride[u];
        face.vStride = stride[v];
        face.nu = extent[u];
        face.nv = extent[v];
        face.count = (nbr_[f] == MPI_PROC_NULL) ? 0 : face.nu * face.nv;
        face.offset = offset;
        offset += face.count;
    }
    halo_ = offset;

    const size_t total = static_cast<size_t>(dom_.ax) * dom_.ay * dom_.az;
    const size_t haloAlloc = std::max<size_t>(halo_, 1);
    CUDA_CHECK(cudaMalloc(&grid_[0], total * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&grid_[1], total * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&sendDev_, haloAlloc * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&recvDev_, haloAlloc * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&sendHost_, haloAlloc * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&recvHost_, haloAlloc * sizeof(Real)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&comm_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&shellDone_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&haloReady_, cudaEventDisableTiming));
}

StencilSolver::~StencilSolver() {
    cudaFree(grid_[0]);
    cudaFree(grid_[1]);
    cudaFree(sendDev_);
    cudaFree(recvDev_);
    cudaFreeHost(sendHost_);
    cudaFreeHost(recvHost_);
    cudaStreamDestroy(compute_);
    cudaStreamDestroy(comm_);
    cudaEventDestroy(shellDone_);
    cudaEventDestroy(haloReady_);
}

// Fills the owned cells and halo cells from the global formula, so no exchange is needed for step 0.
// The second grid starts at zero, as in the reference.
void StencilSolver::initialize() {
    const size_t total = static_cast<size_t>(dom_.ax) * dom_.ay * dom_.az;
    std::vector<Real> host(total, 0.0);

    #pragma omp parallel for collapse(2)
    for (int z = 0; z < dom_.az; ++z) {
        for (int y = 0; y < dom_.ay; ++y) {
            for (int x = 0; x < dom_.ax; ++x) {
                const long gx = dom_.offX + x - 1;
                const long gy = dom_.offY + y - 1;
                const long gz = dom_.offZ + z - 1;
                const bool inside = gx >= 0 && gx < dom_.gnx && gy >= 0 && gy < dom_.gny && gz >= 0 && gz < dom_.gnz;
                if (inside) {
                    const size_t g = idx3(static_cast<size_t>(gx), static_cast<size_t>(gy), static_cast<size_t>(gz),
                                          static_cast<size_t>(dom_.gnx), static_cast<size_t>(dom_.gny));
                    host[localIdx(x, y, z, dom_.ax, dom_.ay)] = (g % 19) * 1.0;
                }
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(grid_[0], host.data(), total * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(grid_[1], 0, total * sizeof(Real)));
}

void StencilSolver::launchStencil(const Box& b, const Real* in, Real* out, cudaStream_t s) const {
    const int ex = b.x1 - b.x0;
    const int ey = b.y1 - b.y0;
    const int ez = b.z1 - b.z0;
    if (ex <= 0 || ey <= 0 || ez <= 0) return;

    const dim3 threads(32, 4, 4);
    const dim3 blocks((ex + 31) / 32, (ey + 3) / 4, (ez + 3) / 4);
    stencilKernel<<<blocks, threads, 0, s>>>(in, out, b, dom_);
    CUDA_CHECK(cudaGetLastError());
}

// Sends the boundary planes of `field` to the neighbours and receives their planes into the halo.
// Runs on comm_ while compute_ works on the inner cells.
void StencilSolver::exchangeHalos(Real* field) {
    if (halo_ == 0) return;

    CUDA_CHECK(cudaStreamWaitEvent(comm_, shellDone_, 0));

    const size_t bytes = halo_ * sizeof(Real);
    const int threads = 256;
    const unsigned blocks = static_cast<unsigned>((halo_ + threads - 1) / threads);

    haloCopyKernel<false><<<blocks, threads, 0, comm_>>>(field, sendDev_, faces_, halo_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(sendHost_, sendDev_, bytes, cudaMemcpyDeviceToHost, comm_));
    CUDA_CHECK(cudaStreamSynchronize(comm_));

    // Face f is sent with tag f and received with tag f^1, which is the tag its neighbour uses for the opposite face
    MPI_Request req[12];
    int nreq = 0;
    for (int f = 0; f < 6; ++f) {
        const Face& face = faces_.f[f];
        MPI_Irecv(recvHost_ + face.offset, static_cast<int>(face.count), MPI_DOUBLE, nbr_[f], f ^ 1, cart_, &req[nreq++]);
        MPI_Isend(sendHost_ + face.offset, static_cast<int>(face.count), MPI_DOUBLE, nbr_[f], f, cart_, &req[nreq++]);
    }
    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);

    CUDA_CHECK(cudaMemcpyAsync(recvDev_, recvHost_, bytes, cudaMemcpyHostToDevice, comm_));
    haloCopyKernel<true><<<blocks, threads, 0, comm_>>>(field, recvDev_, faces_, halo_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(haloReady_, comm_));
}

void StencilSolver::run(int iterations) {
    for (int iter = 0; iter < iterations; ++iter) {
        Real* in = grid_[iter % 2];
        Real* out = grid_[(iter + 1) % 2];

        // Shell first, so its boundary planes can be exchanged while the inner cells are computed
        if (iter > 0 && halo_ > 0) CUDA_CHECK(cudaStreamWaitEvent(compute_, haloReady_, 0));
        for (const Box& b : shell_) launchStencil(b, in, out, compute_);
        CUDA_CHECK(cudaEventRecord(shellDone_, compute_));
        launchStencil(inner_, in, out, compute_);

        if (iter + 1 < iterations) exchangeHalos(out);
    }
    CUDA_CHECK(cudaStreamSynchronize(compute_));
    CUDA_CHECK(cudaStreamSynchronize(comm_));
}

std::vector<Real> StencilSolver::gather(const Real* field) const {
    const size_t total = static_cast<size_t>(dom_.ax) * dom_.ay * dom_.az;
    std::vector<Real> local(total);
    CUDA_CHECK(cudaMemcpy(local.data(), field, total * sizeof(Real), cudaMemcpyDeviceToHost));

    // Owned block of this rank inside its local array (halo excluded)
    const int sizes[3] = {dom_.az, dom_.ay, dom_.ax};
    const int subsizes[3] = {dom_.lnz, dom_.lny, dom_.lnx};
    const int starts[3] = {1, 1, 1};

    if (rank_ != 0) {
        MPI_Datatype block;
        MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Send(local.data(), 1, block, 0, 0, cart_);
        MPI_Type_free(&block);
        return {};
    }

    const size_t gnx = static_cast<size_t>(dom_.gnx);
    const size_t gny = static_cast<size_t>(dom_.gny);
    std::vector<Real> global(static_cast<size_t>(dom_.gnx) * dom_.gny * dom_.gnz);

    // Rank 0 copies its own block directly
    #pragma omp parallel for collapse(2)
    for (int z = 0; z < dom_.lnz; ++z) {
        for (int y = 0; y < dom_.lny; ++y) {
            const size_t src = localIdx(1, y + 1, z + 1, dom_.ax, dom_.ay);
            const size_t dst = idx3(static_cast<size_t>(dom_.offX), static_cast<size_t>(dom_.offY + y),
                                    static_cast<size_t>(dom_.offZ + z), gnx, gny);
            std::copy_n(local.begin() + src, dom_.lnx, global.begin() + dst);
        }
    }

    // Other ranks send their block; the subarray type scatters it into the global layout
    const int gsizes[3] = {static_cast<int>(dom_.gnz), static_cast<int>(dom_.gny), static_cast<int>(dom_.gnx)};
    const int nranks = dims_[0] * dims_[1] * dims_[2];
    for (int r = 1; r < nranks; ++r) {
        int c[3];
        MPI_Cart_coords(cart_, r, 3, c);
        long start[3], len[3];
        const long n[3] = {dom_.gnx, dom_.gny, dom_.gnz};
        for (int a = 0; a < 3; ++a) blockRange(c[a], dims_[a], n[a], start[a], len[a]);

        const int rsub[3] = {static_cast<int>(len[2]), static_cast<int>(len[1]), static_cast<int>(len[0])};
        const int rstart[3] = {static_cast<int>(start[2]), static_cast<int>(start[1]), static_cast<int>(start[0])};
        MPI_Datatype block;
        MPI_Type_create_subarray(3, gsizes, rsub, rstart, MPI_ORDER_C, MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Recv(global.data(), 1, block, r, 0, cart_, MPI_STATUS_IGNORE);
        MPI_Type_free(&block);
    }
    return global;
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for reduction(min : minVal) reduction(max : maxVal)
    for (size_t i = 0; i < grid.size(); ++i) {
        minVal = std::min(minVal, grid[i]);
        maxVal = std::max(maxVal, grid[i]);
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = rank == 0;

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0) {
        if (root) printf("Grid dimensions must be positive\n");
        MPI_Finalize();
        return 1;
    }

    // One GPU per rank: ranks on a node are spread over the node's devices
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    // 3D Cartesian decomposition with non-periodic boundaries
    const long n[3] = {static_cast<long>(nx), static_cast<long>(ny), static_cast<long>(nz)};
    int dims[3] = {0, 0, 0};
    MPI_Dims_create(size, 3, dims);
    for (int a = 0; a < 3; ++a) {
        if (dims[a] > n[a]) {
            if (root) printf("Too many ranks (%d) for the grid: each rank needs at least one cell per axis\n", size);
            MPI_Finalize();
            return 1;
        }
    }
    const int periods[3] = {0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart);
    int coords[3];
    MPI_Cart_coords(cart, rank, 3, coords);
    int nbr[6];
    for (int a = 0; a < 3; ++a) {
        MPI_Cart_shift(cart, a, 1, &nbr[2 * a], &nbr[2 * a + 1]);
    }

    if (root) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    int status = 0;
    {
        StencilSolver solver(cart, rank, dims, coords, n, nbr);
        solver.initialize();

        if (root) printf("Running stencil computation...\n");
        MPI_Barrier(cart);
        const double t0 = MPI_Wtime();
        solver.run(iterations);
        const double localMs = (MPI_Wtime() - t0) * 1e3;
        double computeMs = 0.0;
        MPI_Reduce(&localMs, &computeMs, 1, MPI_DOUBLE, MPI_MAX, 0, cart);

        if (root) {
            const long ms = static_cast<long>(computeMs);
            printf("Computation time: %ld ms\n", ms);

            // Calculate performance metrics
            double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
            double mcups = cellUpdates / (ms / 1000.0) / 1e6;  // Million cell updates per second
            printf("Performance: %.3f MCellUpdates/s\n", mcups);
        }

        // Gather the final grid when it is needed for output or validation
        const bool needGlobal = printResults || validate;
        std::vector<Real> finalGrid;
        if (needGlobal) {
            finalGrid = solver.gather(solver.finalBuffer(iterations));
        }

        if (root) {
            if (printResults) {
                print_results(finalGrid, "Grid");
            }

            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(finalGrid, nx, ny, nz);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }
    }

    MPI_Comm_free(&cart);
    MPI_Finalize();
    return status;
}
