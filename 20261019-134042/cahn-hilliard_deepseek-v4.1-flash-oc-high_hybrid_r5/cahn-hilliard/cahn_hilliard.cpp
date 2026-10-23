#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Error checking helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        cudaError_t err__ = (call);                                                             \
        if (err__ != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                    \
                    cudaGetErrorString(err__));                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)

#define MPI_CHECK(call)                                                                         \
    do {                                                                                        \
        int err__ = (call);                                                                     \
        if (err__ != MPI_SUCCESS) {                                                             \
            fprintf(stderr, "MPI error at %s:%d\n", __FILE__, __LINE__);                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)

namespace {

// ---------------------------------------------------------------------------
// Domain decomposition description
// ---------------------------------------------------------------------------
enum Face : int { XM = 0, XP = 1, YM = 2, YP = 3, ZM = 4, ZP = 5 };

struct Domain {
    int nx, ny, nz;             // global grid
    int lx, ly, lz;             // local interior grid
    int sx, sy, sz;             // local padded grid (one ghost layer on each side)
    int px, py, pz;             // process grid
    int cx, cy, cz;             // this rank's process coordinates
    int x0, y0, z0;             // global origin of the local interior
    int rank, nprocs;
    bool hxm, hxp, hym, hyp, hzm, hzp; // true if the corresponding neighbour exists
    int nbr[6];                 // neighbouring ranks (MPI_PROC_NULL at global boundaries)
    MPI_Comm cart;
};

struct BlockInfo {
    int x0, y0, z0;
    int lx, ly, lz;
};

inline int part_start(int n, int p, int c) {
    return static_cast<int>((static_cast<long long>(n) * c) / p);
}

inline int part_len(int n, int p, int c) {
    return part_start(n, p, c + 1) - part_start(n, p, c);
}

inline int part_max_len(int n, int p) {
    return (n + p - 1) / p;
}

// Pick a Cartesian process grid whose product equals nprocs and which fits the
// global grid.  The score prefers cube-shaped sub-domains to minimise the
// communication surface.
void choose_dims(int nprocs, int nx, int ny, int nz, int& px, int& py, int& pz) {
    px = py = pz = 1;
    if (nprocs == 1) return;

    int bestx = 0, besty = 0, bestz = 0;
    double best_score = 1.0e300;

    for (int a = 1; a <= std::min(nx, nprocs); ++a) {
        if (nprocs % a != 0) continue;
        const int rem = nprocs / a;
        for (int b = 1; b <= std::min(ny, rem); ++b) {
            if (rem % b != 0) continue;
            const int c = rem / b;
            if (c > nz) continue;

            const double lx = part_max_len(nx, a);
            const double ly = part_max_len(ny, b);
            const double lz = part_max_len(nz, c);
            const double score = (lx * ly + ly * lz + lz * lx) / (lx * ly * lz);

            if (score < best_score) {
                best_score = score;
                bestx = a;
                besty = b;
                bestz = c;
            }
        }
    }

    if (bestx == 0) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            fprintf(stderr,
                    "Unable to decompose a %d x %d x %d grid over %d MPI ranks\n",
                    nx, ny, nz, nprocs);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    px = bestx;
    py = besty;
    pz = bestz;
}

inline size_t padded_index(const Domain& d, int x, int y, int z) {
    return (static_cast<size_t>(z + 1) * d.sy + (y + 1)) * d.sx + (x + 1);
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------
struct KernelParams {
    int lx, ly, lz;
    int sx, sy, sz;
    int hxm, hxp, hym, hyp, hzm, hzp;
    double e_AA, e_BB, e_AB;
    double gamma;
    double dtD;
};

__device__ __forceinline__ double laplacian_padded(const double* __restrict__ c,
                                                   const KernelParams& p,
                                                   int x, int y, int z) {
    const int px = x + 1;
    const int py = y + 1;
    const int pz = z + 1;

    const int xm = (x > 0) ? (px - 1) : (p.hxm ? 0 : px);
    const int xp = (x < p.lx - 1) ? (px + 1) : (p.hxp ? p.sx - 1 : px);
    const int ym = (y > 0) ? (py - 1) : (p.hym ? 0 : py);
    const int yp = (y < p.ly - 1) ? (py + 1) : (p.hyp ? p.sy - 1 : py);
    const int zm = (z > 0) ? (pz - 1) : (p.hzm ? 0 : pz);
    const int zp = (z < p.lz - 1) ? (pz + 1) : (p.hzp ? p.sz - 1 : pz);

    const size_t slice = static_cast<size_t>(p.sx) * p.sy;
    const size_t center = static_cast<size_t>(pz) * slice + static_cast<size_t>(py) * p.sx + px;
    const size_t row = static_cast<size_t>(py) * p.sx;

    const double c0 = c[center];
    const double cxx = c[static_cast<size_t>(pz) * slice + row + xp] +
                       c[static_cast<size_t>(pz) * slice + row + xm] - 2.0 * c0;
    const double cyy = c[static_cast<size_t>(pz) * slice + static_cast<size_t>(yp) * p.sx + px] +
                       c[static_cast<size_t>(pz) * slice + static_cast<size_t>(ym) * p.sx + px] -
                       2.0 * c0;
    const double czz = c[static_cast<size_t>(zp) * slice + row + px] +
                       c[static_cast<size_t>(zm) * slice + row + px] - 2.0 * c0;
    return cxx + cyy + czz;
}

__global__ void compute_mu_kernel(const double* __restrict__ c,
                                  double* __restrict__ mu,
                                  const KernelParams p) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;

    if (x >= p.lx || y >= p.ly || z >= p.lz) return;

    const size_t idx = (static_cast<size_t>(z + 1) * p.sy + (y + 1)) * p.sx + (x + 1);
    const double cv = c[idx];
    const double lap = laplacian_padded(c, p, x, y, z);

    mu[idx] = 4.5 * ((cv + 1.0) * p.e_AA + (cv - 1.0) * p.e_BB - 2.0 * cv * p.e_AB) +
              3.0 * cv + cv * cv * cv - p.gamma * lap;
}

__global__ void update_kernel(double* __restrict__ cnew,
                              const double* __restrict__ cold,
                              const double* __restrict__ mu,
                              const KernelParams p) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;

    if (x >= p.lx || y >= p.ly || z >= p.lz) return;

    const size_t idx = (static_cast<size_t>(z + 1) * p.sy + (y + 1)) * p.sx + (x + 1);
    cnew[idx] = cold[idx] + p.dtD * laplacian_padded(mu, p, x, y, z);
}

// ---------------------------------------------------------------------------
// Halo exchange
// ---------------------------------------------------------------------------
struct HaloBuffers {
    double* send[6]{};
    double* recv[6]{};

    void allocate(size_t max_face) {
        if (max_face == 0) max_face = 1;
        for (int f = 0; f < 6; ++f) {
            CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&send[f]), max_face * sizeof(double),
                                     cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&recv[f]), max_face * sizeof(double),
                                     cudaHostAllocDefault));
        }
    }

    void release() {
        for (int f = 0; f < 6; ++f) {
            if (send[f]) cudaFreeHost(send[f]);
            if (recv[f]) cudaFreeHost(recv[f]);
        }
    }
};

inline size_t face_count(const Domain& d, int face) {
    switch (face) {
        case XM:
        case XP:
            return static_cast<size_t>(d.ly) * d.lz;
        case YM:
        case YP:
            return static_cast<size_t>(d.lx) * d.lz;
        default:
            return static_cast<size_t>(d.lx) * d.ly;
    }
}

void copy_face(const Domain& d, double* dev, double* host, int face, bool pack) {
    int ex, ey, ez, ox, oy, oz;
    switch (face) {
        case XM:
            ex = 1;    ey = d.ly; ez = d.lz;
            ox = pack ? 1 : 0;          oy = 1;          oz = 1;          break;
        case XP:
            ex = 1;    ey = d.ly; ez = d.lz;
            ox = pack ? d.lx : d.sx - 1; oy = 1;         oz = 1;          break;
        case YM:
            ex = d.lx; ey = 1;    ez = d.lz;
            ox = 1;    oy = pack ? 1 : 0;                oz = 1;          break;
        case YP:
            ex = d.lx; ey = 1;    ez = d.lz;
            ox = 1;    oy = pack ? d.ly : d.sy - 1;      oz = 1;          break;
        case ZM:
            ex = d.lx; ey = d.ly; ez = 1;
            ox = 1;    oy = 1;    oz = pack ? 1 : 0;                      break;
        default:
            ex = d.lx; ey = d.ly; ez = 1;
            ox = 1;    oy = 1;    oz = pack ? d.lz : d.sz - 1;            break;
    }

    const size_t dev_off = (static_cast<size_t>(oz) * d.sy + oy) * d.sx + ox;

    cudaPitchedPtr dev_ptr = make_cudaPitchedPtr(dev + dev_off,
                                                 static_cast<size_t>(d.sx) * sizeof(double),
                                                 d.sx, d.sy);
    cudaPitchedPtr host_ptr = make_cudaPitchedPtr(host,
                                                  static_cast<size_t>(ex) * sizeof(double),
                                                  ex, ey);
    cudaExtent extent = make_cudaExtent(static_cast<size_t>(ex) * sizeof(double), ey, ez);

    cudaMemcpy3DParms params{};
    params.extent = extent;
    if (pack) {
        params.srcPtr = dev_ptr;
        params.dstPtr = host_ptr;
        params.kind = cudaMemcpyDeviceToHost;
    } else {
        params.srcPtr = host_ptr;
        params.dstPtr = dev_ptr;
        params.kind = cudaMemcpyHostToDevice;
    }
    CUDA_CHECK(cudaMemcpy3D(&params));
}

void exchange_halos(const Domain& d, double* field, HaloBuffers& hb, int tag_base) {
    MPI_Request reqs[12];
    int nreq = 0;

    // Post all receives first so that packing overlaps with network progress.
    for (int f = 0; f < 6; ++f) {
        if (d.nbr[f] == MPI_PROC_NULL) continue;
        MPI_CHECK(MPI_Irecv(hb.recv[f], static_cast<int>(face_count(d, f)), MPI_DOUBLE,
                            d.nbr[f], tag_base + f / 2, d.cart, &reqs[nreq++]));
    }

    for (int f = 0; f < 6; ++f) {
        if (d.nbr[f] == MPI_PROC_NULL) continue;
        copy_face(d, field, hb.send[f], f, true);
        MPI_CHECK(MPI_Isend(hb.send[f], static_cast<int>(face_count(d, f)), MPI_DOUBLE,
                            d.nbr[f], tag_base + f / 2, d.cart, &reqs[nreq++]));
    }

    if (nreq > 0) {
        MPI_CHECK(MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE));
    }

    for (int f = 0; f < 6; ++f) {
        if (d.nbr[f] == MPI_PROC_NULL) continue;
        copy_face(d, field, hb.recv[f], f, false);
    }
}

// ---------------------------------------------------------------------------
// Initialisation and result handling
// ---------------------------------------------------------------------------
void initialize_concentration(std::vector<double>& local, double* device, const Domain& d) {
    const size_t local_cells = static_cast<size_t>(d.lx) * d.ly * d.lz;
    local.resize(local_cells);

    const size_t vol = static_cast<size_t>(d.nx) * d.ny * d.nz;
    const size_t nxny = static_cast<size_t>(d.nx) * d.ny;

#pragma omp parallel for collapse(3) schedule(static)
    for (int z = 0; z < d.lz; ++z) {
        for (int y = 0; y < d.ly; ++y) {
            for (int x = 0; x < d.lx; ++x) {
                const size_t gx = static_cast<size_t>(d.x0 + x);
                const size_t gy = static_cast<size_t>(d.y0 + y);
                const size_t gz = static_cast<size_t>(d.z0 + z);
                const size_t linear_id = gz * nxny + gy * d.nx + gx;
                const double pseudo =
                    static_cast<double>(((linear_id + 1) * 1299709) % vol) /
                    static_cast<double>(vol);
                local[(static_cast<size_t>(z) * d.ly + y) * d.lx + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }

    cudaPitchedPtr host_ptr = make_cudaPitchedPtr(local.data(),
                                                  static_cast<size_t>(d.lx) * sizeof(double),
                                                  d.lx, d.ly);
    cudaPitchedPtr dev_ptr = make_cudaPitchedPtr(device + padded_index(d, 0, 0, 0),
                                                 static_cast<size_t>(d.sx) * sizeof(double),
                                                 d.sx, d.sy);
    cudaExtent extent =
        make_cudaExtent(static_cast<size_t>(d.lx) * sizeof(double), d.ly, d.lz);
    cudaMemcpy3DParms params{};
    params.srcPtr = host_ptr;
    params.dstPtr = dev_ptr;
    params.extent = extent;
    params.kind = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&params));
}

void copy_interior_to_host(std::vector<double>& local, const double* device, const Domain& d) {
    local.resize(static_cast<size_t>(d.lx) * d.ly * d.lz);

    cudaPitchedPtr host_ptr = make_cudaPitchedPtr(local.data(),
                                                  static_cast<size_t>(d.lx) * sizeof(double),
                                                  d.lx, d.ly);
    cudaPitchedPtr dev_ptr = make_cudaPitchedPtr(const_cast<double*>(device) +
                                                     padded_index(d, 0, 0, 0),
                                                 static_cast<size_t>(d.sx) * sizeof(double),
                                                 d.sx, d.sy);
    cudaExtent extent =
        make_cudaExtent(static_cast<size_t>(d.lx) * sizeof(double), d.ly, d.lz);
    cudaMemcpy3DParms params{};
    params.srcPtr = dev_ptr;
    params.dstPtr = host_ptr;
    params.extent = extent;
    params.kind = cudaMemcpyDeviceToHost;
    CUDA_CHECK(cudaMemcpy3D(&params));
}

BlockInfo block_for_rank(const Domain& d, int r) {
    int crd[3];
    MPI_Cart_coords(d.cart, r, 3, crd);
    BlockInfo b;
    b.x0 = part_start(d.nx, d.px, crd[0]);
    b.lx = part_len(d.nx, d.px, crd[0]);
    b.y0 = part_start(d.ny, d.py, crd[1]);
    b.ly = part_len(d.ny, d.py, crd[1]);
    b.z0 = part_start(d.nz, d.pz, crd[2]);
    b.lz = part_len(d.nz, d.pz, crd[2]);
    return b;
}

void gather_result(const Domain& d, const std::vector<double>& local, std::vector<double>& global) {
    const int local_cells = static_cast<int>(local.size());
    std::vector<int> counts, displs;
    std::vector<BlockInfo> blocks;
    std::vector<double> packed;

    if (d.rank == 0) {
        counts.resize(d.nprocs);
        displs.resize(d.nprocs);
        blocks.resize(d.nprocs);
        int offset = 0;
        for (int r = 0; r < d.nprocs; ++r) {
            blocks[r] = block_for_rank(d, r);
            counts[r] = blocks[r].lx * blocks[r].ly * blocks[r].lz;
            displs[r] = offset;
            offset += counts[r];
        }
        packed.resize(offset);
    }

    MPI_CHECK(MPI_Gatherv(local.data(), local_cells, MPI_DOUBLE,
                          d.rank == 0 ? packed.data() : nullptr,
                          d.rank == 0 ? counts.data() : nullptr,
                          d.rank == 0 ? displs.data() : nullptr,
                          MPI_DOUBLE, 0, d.cart));

    if (d.rank == 0) {
        const size_t nxny = static_cast<size_t>(d.nx) * d.ny;
#pragma omp parallel for schedule(dynamic)
        for (int r = 0; r < d.nprocs; ++r) {
            const BlockInfo& b = blocks[r];
            const double* src = packed.data() + displs[r];
            for (int z = 0; z < b.lz; ++z) {
                for (int y = 0; y < b.ly; ++y) {
                    double* dst = global.data() +
                                  (static_cast<size_t>(b.z0 + z) * nxny +
                                   static_cast<size_t>(b.y0 + y) * d.nx + b.x0);
                    const double* src_row = src + (static_cast<size_t>(z) * b.ly + y) * b.lx;
                    for (int x = 0; x < b.lx; ++x) {
                        dst[x] = src_row[x];
                    }
                }
            }
        }
    }
}

bool validate_global(const Domain& d, const std::vector<double>& local) {
    double local_min = HUGE_VAL;
    double local_max = -HUGE_VAL;
    int local_nan = 0;

#pragma omp parallel
    {
        double mn = HUGE_VAL;
        double mx = -HUGE_VAL;
        int nan = 0;
#pragma omp for nowait
        for (long long i = 0; i < static_cast<long long>(local.size()); ++i) {
            const double v = local[static_cast<size_t>(i)];
            if (!std::isfinite(v)) {
                nan = 1;
            } else {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
        }
#pragma omp critical
        {
            local_min = std::min(local_min, mn);
            local_max = std::max(local_max, mx);
            local_nan |= nan;
        }
    }

    double global_min = 0.0;
    double global_max = 0.0;
    int global_nan = 0;
    MPI_CHECK(MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, d.cart));
    MPI_CHECK(MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, d.cart));
    MPI_CHECK(MPI_Allreduce(&local_nan, &global_nan, 1, MPI_INT, MPI_LOR, d.cart));

    bool valid = true;
    if (d.rank == 0) {
        if (global_nan) {
            printf("Validation failed: found NaN or Inf value\n");
            valid = false;
        } else {
            printf("Concentration range: [%.6f, %.6f]\n", global_min, global_max);
            if (global_max > 10.0 || global_min < -10.0) {
                printf("Validation failed: values out of expected range\n");
                valid = false;
            }
        }
    }
    int v = valid ? 1 : 0;
    MPI_CHECK(MPI_Bcast(&v, 1, MPI_INT, 0, d.cart));
    return v != 0;
}

void printUsage(const char* progName) {
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

} // namespace

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int world_rank = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    bool show_help = false;
    bool bad_option = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            show_help = true;
        } else {
            bad_option = true;
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
        }
    }

    if (show_help) {
        if (world_rank == 0) printUsage(argv[0]);
        MPI_CHECK(MPI_Finalize());
        return 0;
    }
    if (bad_option) {
        if (world_rank == 0) printUsage(argv[0]);
        MPI_CHECK(MPI_Finalize());
        return 1;
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // Determine node-local rank so that each rank can claim a distinct GPU.
    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                                  &node_comm));
    int local_rank = 0;
    int local_size = 1;
    MPI_CHECK(MPI_Comm_rank(node_comm, &local_rank));
    MPI_CHECK(MPI_Comm_size(node_comm, &local_size));
    MPI_CHECK(MPI_Comm_free(&node_comm));

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        if (world_rank == 0) fprintf(stderr, "No CUDA device found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));

    // Give each MPI rank a fair share of the host threads for the non-GPU work.
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        int threads = omp_get_num_procs() / std::max(1, local_size);
        omp_set_num_threads(std::max(1, threads));
    }

    Domain d{};
    d.nx = static_cast<int>(nx);
    d.ny = static_cast<int>(ny);
    d.nz = static_cast<int>(nz);

    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &d.nprocs));
    choose_dims(d.nprocs, d.nx, d.ny, d.nz, d.px, d.py, d.pz);

    int dims[3] = {d.px, d.py, d.pz};
    int periods[3] = {0, 0, 0};
    MPI_CHECK(MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &d.cart));
    MPI_CHECK(MPI_Comm_rank(d.cart, &d.rank));

    int coords[3];
    MPI_CHECK(MPI_Cart_coords(d.cart, d.rank, 3, coords));
    d.cx = coords[0];
    d.cy = coords[1];
    d.cz = coords[2];

    d.x0 = part_start(d.nx, d.px, d.cx);
    d.lx = part_len(d.nx, d.px, d.cx);
    d.y0 = part_start(d.ny, d.py, d.cy);
    d.ly = part_len(d.ny, d.py, d.cy);
    d.z0 = part_start(d.nz, d.pz, d.cz);
    d.lz = part_len(d.nz, d.pz, d.cz);

    d.sx = d.lx + 2;
    d.sy = d.ly + 2;
    d.sz = d.lz + 2;

    // MPI_Cart_shift with displacement +1 returns the lower neighbour as the
    // source and the upper neighbour as the destination.
    MPI_CHECK(MPI_Cart_shift(d.cart, 0, 1, &d.nbr[XM], &d.nbr[XP]));
    MPI_CHECK(MPI_Cart_shift(d.cart, 1, 1, &d.nbr[YM], &d.nbr[YP]));
    MPI_CHECK(MPI_Cart_shift(d.cart, 2, 1, &d.nbr[ZM], &d.nbr[ZP]));

    d.hxm = d.nbr[XM] != MPI_PROC_NULL;
    d.hxp = d.nbr[XP] != MPI_PROC_NULL;
    d.hym = d.nbr[YM] != MPI_PROC_NULL;
    d.hyp = d.nbr[YP] != MPI_PROC_NULL;
    d.hzm = d.nbr[ZM] != MPI_PROC_NULL;
    d.hzp = d.nbr[ZP] != MPI_PROC_NULL;

    if (world_rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;
    (void)dx;
    (void)dy;
    (void)dz;

    const size_t local_alloc = static_cast<size_t>(d.sx) * d.sy * d.sz;
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cold, local_alloc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cnew, local_alloc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, local_alloc * sizeof(double)));

    HaloBuffers hb;
    const size_t max_face =
        std::max({static_cast<size_t>(d.lx) * d.ly, static_cast<size_t>(d.lx) * d.lz,
                  static_cast<size_t>(d.ly) * d.lz});
    hb.allocate(max_face);

    if (world_rank == 0) printf("Initializing concentration field...\n");
    std::vector<double> local_host;
    initialize_concentration(local_host, d_cold, d);
    std::vector<double>().swap(local_host);

    KernelParams kp{};
    kp.lx = d.lx;
    kp.ly = d.ly;
    kp.lz = d.lz;
    kp.sx = d.sx;
    kp.sy = d.sy;
    kp.sz = d.sz;
    kp.hxm = d.hxm ? 1 : 0;
    kp.hxp = d.hxp ? 1 : 0;
    kp.hym = d.hym ? 1 : 0;
    kp.hyp = d.hyp ? 1 : 0;
    kp.hzm = d.hzm ? 1 : 0;
    kp.hzp = d.hzp ? 1 : 0;
    kp.e_AA = e_AA;
    kp.e_BB = e_BB;
    kp.e_AB = e_AB;
    kp.gamma = gamma;
    kp.dtD = dt * D;

    int bx = (d.lx >= 32) ? 32 : ((d.lx >= 16) ? 16 : ((d.lx >= 8) ? 8 : 1));
    int by = (d.ly >= 8) ? 8 : ((d.ly >= 4) ? 4 : 1);
    dim3 block(bx, by, 1);
    dim3 grid((d.lx + bx - 1) / bx, (d.ly + by - 1) / by, d.lz);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDA_CHECK(cudaDeviceSynchronize());

    if (world_rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        exchange_halos(d, d_cold, hb, 100);

        compute_mu_kernel<<<grid, block>>>(d_cold, d_mu, kp);
        CUDA_CHECK(cudaGetLastError());

        exchange_halos(d, d_mu, hb, 200);

        update_kernel<<<grid, block>>>(d_cnew, d_cold, d_mu, kp);
        CUDA_CHECK(cudaGetLastError());

        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (world_rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        const double gridSize = static_cast<double>(nx) * static_cast<double>(ny) *
                                static_cast<double>(nz);
        const double cellUpdates = gridSize * iterations;
        const double seconds = duration.count() / 1000.0;
        const double mcups = seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    std::vector<double> global_result;
    if (printResults || validate) {
        local_host.clear();
        copy_interior_to_host(local_host, d_cold, d);
    }

    if (printResults) {
        if (world_rank == 0) {
            global_result.assign(static_cast<size_t>(nx) * ny * nz, 0.0);
        }
        gather_result(d, local_host, global_result);
        if (world_rank == 0) {
            print_results(global_result, "Concentration");
        }
    }

    int exit_code = 0;
    if (validate) {
        if (world_rank == 0) printf("Validating result...\n");
        const bool valid = validate_global(d, local_host);
        if (world_rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        exit_code = valid ? 0 : 1;
    }

    hb.release();
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));
    MPI_CHECK(MPI_Comm_free(&d.cart));
    MPI_CHECK(MPI_Finalize());

    return exit_code;
}
