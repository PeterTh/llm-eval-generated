/**
 * Room Response Simulation Benchmark (CUDA version)
 *
 * This is a GPU-parallel implementation of room impulse response simulation
 * using radiosity-based wave propagation. It models how sound/light waves
 * propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization strategy (CUDA):
 *  - Mesh generation and the spatial acceleration structure (a uniform grid in
 *    which triangles are registered with the same triangle/box overlap test the
 *    sequential octree build uses) are built on the host and uploaded.
 *  - Form factors: one GPU thread per (triangle pair, ray). Visibility walks the
 *    segment through the grid with a 3D-DDA and tests the triangles of the cells
 *    it crosses (any-hit, same ray/triangle test as the sequential version).
 *    The random point samples reproduce the exact std::mt19937 /
 *    uniform_real_distribution<float> stream of the sequential version: the
 *    Mersenne Twister stream is generated on the GPU (each 624-word twist in
 *    three data-parallel stages), split into segments whose start states are
 *    obtained by polynomial jump-ahead (characteristic polynomial computed on the
 *    host with Berlekamp-Massey, x^J mod p and the Horner evaluation on the GPU),
 *    and pipelined on a separate stream with the form-factor kernel.
 *  - Time delays: one thread per pair.
 *  - Wave propagation: one kernel launch per timestep, one warp per receiving
 *    triangle; the terms are added in the sequential order via shuffles, so the
 *    result is bit-identical to the sequential summation.
 *  - Distances: one thread per (triangle, lag) for the cross-correlation, then one
 *    thread per triangle for the arg-max.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

// Random numbers consumed per (non-culled) triangle pair: 2 points x 2 coords x NUM_RAYS
constexpr int RANDS_PER_PAIR = 4 * NUM_RAYS;

// ============================================================================
// CUDA error handling
// ============================================================================

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,        \
                    __LINE__, cudaGetErrorString(err__));                           \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

// ============================================================================
// Vector and Triangle Types (host side)
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    constexpr Vec3() : x(0), y(0), z(0) {}
    constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }

    val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    val_t squaredNorm() const { return x * x + y * y + z * z; }
    val_t norm() const { return std::sqrt(squaredNorm()); }
    Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    Vec3 center() const { return (a + b + c) / 3.0f; }
    Vec3 normal() const { return _normal; }

    val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
};

// ============================================================================
// Triangle-Box Overlap Test (for grid construction)
// ============================================================================

// Separating axis test for triangle-box overlap
bool triangleBoxOverlap(const Vec3& boxCenter, const Vec3& boxHalfSize, const Triangle& tri) {
    // Translate triangle to box center
    Vec3 v0 = tri.a - boxCenter;
    Vec3 v1 = tri.b - boxCenter;
    Vec3 v2 = tri.c - boxCenter;

    // Triangle edges
    Vec3 e0 = v1 - v0;
    Vec3 e1 = v2 - v1;
    Vec3 e2 = v0 - v2;

    // Test box normals (AABB axes)
    auto minMax3 = [](val_t a, val_t b, val_t c) {
        return std::make_pair(std::min({a, b, c}), std::max({a, b, c}));
    };

    auto [minX, maxX] = minMax3(v0.x, v1.x, v2.x);
    if (minX > boxHalfSize.x || maxX < -boxHalfSize.x) return false;

    auto [minY, maxY] = minMax3(v0.y, v1.y, v2.y);
    if (minY > boxHalfSize.y || maxY < -boxHalfSize.y) return false;

    auto [minZ, maxZ] = minMax3(v0.z, v1.z, v2.z);
    if (minZ > boxHalfSize.z || maxZ < -boxHalfSize.z) return false;

    // Test triangle normal
    Vec3 triNormal = e0.cross(e1);
    val_t d = triNormal.dot(v0);
    val_t r = boxHalfSize.x * std::abs(triNormal.x) +
              boxHalfSize.y * std::abs(triNormal.y) +
              boxHalfSize.z * std::abs(triNormal.z);
    if (std::abs(d) > r) return false;

    // Test 9 edge cross products
    auto testAxis = [&](const Vec3& axis) {
        val_t p0 = axis.dot(v0);
        val_t p1 = axis.dot(v1);
        val_t p2 = axis.dot(v2);
        val_t r = boxHalfSize.x * std::abs(axis.x) +
                  boxHalfSize.y * std::abs(axis.y) +
                  boxHalfSize.z * std::abs(axis.z);
        auto [minP, maxP] = minMax3(p0, p1, p2);
        return !(minP > r || maxP < -r);
    };

    Vec3 axes[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
    Vec3 edges[3] = {e0, e1, e2};
    for (const auto& axis : axes) {
        for (const auto& edge : edges) {
            Vec3 crossAxis = axis.cross(edge);
            if (crossAxis.squaredNorm() > EPSILON) {
                if (!testAxis(crossAxis)) return false;
            }
        }
    }

    return true;
}

// ============================================================================
// Uniform Grid for Spatial Acceleration
//
// The visibility test needs, for a segment between two triangles, every
// triangle the segment could intersect. Triangles are registered in all cells
// of a uniform grid whose (slightly enlarged) box they overlap, using the same
// separating-axis test the sequential octree build uses; a segment is then walked
// through the grid with a 3D-DDA and tested against the triangles of every cell
// it passes through. A triangle hit by a segment lies in a cell the segment
// passes through, so the visibility outcome is that of testing all triangles.
// ============================================================================

constexpr val_t GRID_REGISTER_SLACK = 1e-4f;         // enlargement of cells for registration

struct DeviceGridParams {
    float3 origin;      // minimum corner of the grid
    float3 cellSize;    // cell extent per axis
    float3 invCellSize;
    int res;            // cells per axis
};

// Grid resolution: the cell count per axis scales with the square root of the
// triangle count (i.e. with the surface mesh density), so that cells hold a
// handful of triangles each while segments cross few cells. Tuned on an RTX 3090:
// 32 cells per axis for 5120 triangles, 64 for 20480.
int gridResolutionFor(size_t numTriangles) {
    double res = 32.0 * std::sqrt(static_cast<double>(numTriangles) / 5120.0);
    int r = static_cast<int>(std::lround(res));
    return std::min(std::max(r, 4), 128);
}

class UniformGrid {
public:
    DeviceGridParams params;
    std::vector<int> cellStart;   // res^3 + 1 (CSR offsets)
    std::vector<int> cellTris;    // triangle indices, grouped by cell

    void build(const std::vector<Triangle>& triangles) {
        const int GRID_RES = gridResolutionFor(triangles.size());
        const int GRID_CELLS = GRID_RES * GRID_RES * GRID_RES;
        params.res = GRID_RES;
        // Bounding box of all vertices, enlarged slightly
        Vec3 minBound = triangles[0].a;
        Vec3 maxBound = triangles[0].a;
        for (const auto& tri : triangles) {
            for (const auto* v : {&tri.a, &tri.b, &tri.c}) {
                minBound.x = std::min(minBound.x, v->x);
                minBound.y = std::min(minBound.y, v->y);
                minBound.z = std::min(minBound.z, v->z);
                maxBound.x = std::max(maxBound.x, v->x);
                maxBound.y = std::max(maxBound.y, v->y);
                maxBound.z = std::max(maxBound.z, v->z);
            }
        }
        Vec3 extent = maxBound - minBound;
        val_t margin = 1e-3f * std::max({extent.x, extent.y, extent.z}) + 1e-3f;
        minBound = minBound - Vec3(margin);
        maxBound = maxBound + Vec3(margin);
        extent = maxBound - minBound;

        Vec3 cell = extent / static_cast<val_t>(GRID_RES);
        params.origin = make_float3(minBound.x, minBound.y, minBound.z);
        params.cellSize = make_float3(cell.x, cell.y, cell.z);
        params.invCellSize = make_float3(1.0f / cell.x, 1.0f / cell.y, 1.0f / cell.z);
        Vec3 halfSize = cell * 0.5f + Vec3(GRID_REGISTER_SLACK);

        // Register every triangle in all cells its box overlaps
        std::vector<std::vector<int>> perCell(GRID_CELLS);
        auto cellCoord = [&](val_t v, val_t lo, val_t inv) {
            int c = static_cast<int>(std::floor((v - lo) * inv));
            return std::min(std::max(c, 0), GRID_RES - 1);
        };
        for (size_t t = 0; t < triangles.size(); ++t) {
            const Triangle& tri = triangles[t];
            Vec3 tmin = tri.a, tmax = tri.a;
            for (const auto* v : {&tri.b, &tri.c}) {
                tmin.x = std::min(tmin.x, v->x); tmax.x = std::max(tmax.x, v->x);
                tmin.y = std::min(tmin.y, v->y); tmax.y = std::max(tmax.y, v->y);
                tmin.z = std::min(tmin.z, v->z); tmax.z = std::max(tmax.z, v->z);
            }
            tmin = tmin - Vec3(GRID_REGISTER_SLACK);
            tmax = tmax + Vec3(GRID_REGISTER_SLACK);
            int x0 = cellCoord(tmin.x, minBound.x, 1.0f / cell.x), x1 = cellCoord(tmax.x, minBound.x, 1.0f / cell.x);
            int y0 = cellCoord(tmin.y, minBound.y, 1.0f / cell.y), y1 = cellCoord(tmax.y, minBound.y, 1.0f / cell.y);
            int z0 = cellCoord(tmin.z, minBound.z, 1.0f / cell.z), z1 = cellCoord(tmax.z, minBound.z, 1.0f / cell.z);
            for (int z = z0; z <= z1; ++z) {
                for (int y = y0; y <= y1; ++y) {
                    for (int x = x0; x <= x1; ++x) {
                        Vec3 center = minBound + Vec3((x + 0.5f) * cell.x, (y + 0.5f) * cell.y, (z + 0.5f) * cell.z);
                        if (triangleBoxOverlap(center, halfSize, tri)) {
                            perCell[(static_cast<size_t>(z) * GRID_RES + y) * GRID_RES + x].push_back(static_cast<int>(t));
                        }
                    }
                }
            }
        }

        cellStart.assign(GRID_CELLS + 1, 0);
        for (int c = 0; c < GRID_CELLS; ++c) cellStart[c + 1] = cellStart[c] + static_cast<int>(perCell[c].size());
        cellTris.resize(static_cast<size_t>(cellStart[GRID_CELLS]));
        for (int c = 0; c < GRID_CELLS; ++c) {
            std::copy(perCell[c].begin(), perCell[c].end(), cellTris.begin() + cellStart[c]);
        }
    }
};

// ============================================================================
// Mesh Generation: Icosphere
// ============================================================================

// Generate an icosphere by subdividing an icosahedron
class IcosphereMesh {
public:
    std::vector<Triangle> triangles;

    IcosphereMesh(int subdivisions, val_t radius) {
        // Initial icosahedron vertices
        const val_t t = (1.0f + std::sqrt(5.0f)) / 2.0f;

        std::vector<Vec3> vertices = {
            Vec3(-1,  t,  0).normalized() * radius,
            Vec3( 1,  t,  0).normalized() * radius,
            Vec3(-1, -t,  0).normalized() * radius,
            Vec3( 1, -t,  0).normalized() * radius,
            Vec3( 0, -1,  t).normalized() * radius,
            Vec3( 0,  1,  t).normalized() * radius,
            Vec3( 0, -1, -t).normalized() * radius,
            Vec3( 0,  1, -t).normalized() * radius,
            Vec3( t,  0, -1).normalized() * radius,
            Vec3( t,  0,  1).normalized() * radius,
            Vec3(-t,  0, -1).normalized() * radius,
            Vec3(-t,  0,  1).normalized() * radius
        };

        // Initial icosahedron faces (20 triangles)
        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };

        // Subdivide
        for (int i = 0; i < subdivisions; ++i) {
            std::vector<std::array<idx_t, 3>> newFaces;
            std::map<std::pair<idx_t, idx_t>, idx_t> midpointCache;

            auto getMidpoint = [&](idx_t i1, idx_t i2) -> idx_t {
                auto key = std::make_pair(std::min(i1, i2), std::max(i1, i2));
                auto it = midpointCache.find(key);
                if (it != midpointCache.end()) return it->second;

                Vec3 mid = (vertices[i1] + vertices[i2]) / 2.0f;
                mid = mid.normalized() * radius;
                idx_t idx = static_cast<idx_t>(vertices.size());
                vertices.push_back(mid);
                midpointCache[key] = idx;
                return idx;
            };

            for (const auto& face : faces) {
                idx_t a = getMidpoint(face[0], face[1]);
                idx_t b = getMidpoint(face[1], face[2]);
                idx_t c = getMidpoint(face[2], face[0]);

                newFaces.push_back({face[0], a, c});
                newFaces.push_back({face[1], b, a});
                newFaces.push_back({face[2], c, b});
                newFaces.push_back({a, b, c});
            }
            faces = std::move(newFaces);
        }

        // Build triangles (normals pointing inward for a "room")
        triangles.reserve(faces.size());
        for (const auto& face : faces) {
            // Reverse winding to make normals point inward
            triangles.emplace_back(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Random Number Generation: Mersenne Twister (MT19937) on the GPU
//
// Reproduces the exact output stream of std::mt19937 seeded with 42 as consumed
// through std::uniform_real_distribution<float>(0, 1) in the sequential version.
//
// The 624-word state update has a dependency distance of 227 words, so one
// thread block evaluates each twist in three data-parallel stages. To obtain
// GPU-wide parallelism the stream is split into MT_SEGMENTS contiguous segments,
// each generated by its own block. The starting state of segment k is obtained
// by jump-ahead: with p(x) the characteristic polynomial of the MT19937 state
// transition T (computed once on the host with Berlekamp-Massey), the jumped
// state is g(T) * state where g(x) = x^J mod p(x). The polynomial arithmetic
// (squaring mod p) and the Horner evaluation of g(T) run on the GPU.
// ============================================================================

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr int MT_STAGE = MT_N - MT_M;       // 227 words can be twisted in parallel
constexpr uint32_t MT_MATRIX_A = 0x9908b0dfu;
constexpr uint32_t MT_UPPER_MASK = 0x80000000u;
constexpr uint32_t MT_LOWER_MASK = 0x7fffffffu;
constexpr int MT_BLOCK_THREADS = 640;       // >= MT_N, multiple of 32
constexpr int MT_DEGREE = 19937;            // degree of the characteristic polynomial
constexpr int MT_POLY_WORDS = 624;          // 19968 bits >= MT_DEGREE + 1
constexpr int MT_POLY_BITS = MT_POLY_WORDS * 32;
constexpr int MT_HORNER_GROUPS = (MT_DEGREE + MT_STAGE - 1) / MT_STAGE;  // 88

// Description of how a batch of pairs is striped over the generator segments:
// segment k owns pair ranks [k * segPairs, min((k+1) * segPairs, totalPairs));
// batch b of segment k covers ranks [k*segPairs + b*slicePairs, +slicePairs) and is
// stored at list index k * slicePairs + local.
struct BatchLayout {
    unsigned long long firstPair;    // first pair rank covered by this layout
    unsigned long long totalPairs;   // one past the last pair rank covered
    unsigned long long segPairs;
    unsigned long long slicePairs;
    unsigned long long batch;
    int segments;
};

__host__ __device__ inline void sliceRange(const BatchLayout& L, int k,
                                           unsigned long long& r0, unsigned long long& r1) {
    unsigned long long segBegin = L.firstPair + static_cast<unsigned long long>(k) * L.segPairs;
    unsigned long long segEnd = segBegin + L.segPairs;
    if (segEnd > L.totalPairs) segEnd = L.totalPairs;
    r0 = segBegin + L.batch * L.slicePairs;
    r1 = r0 + L.slicePairs;
    if (r1 > segEnd) r1 = segEnd;
    if (r0 > r1) r0 = r1;
}

struct MtState {
    uint32_t mt[MT_N];
    int mti;
};

__device__ __forceinline__ uint32_t mtTemper(uint32_t y) {
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

__device__ __forceinline__ uint32_t mtTwistWord(uint32_t xk, uint32_t xk1, uint32_t xkm) {
    uint32_t y = (xk & MT_UPPER_MASK) | (xk1 & MT_LOWER_MASK);
    return xkm ^ (y >> 1) ^ ((y & 1u) ? MT_MATRIX_A : 0u);
}

__device__ __forceinline__ uint32_t polyBit(const uint32_t* poly, int i) {
    return (i < MT_POLY_BITS) ? ((poly[i >> 5] >> (i & 31)) & 1u) : 0u;
}

// Generate the words of segment `blockIdx.x` for the batch described by `L`
// (RANDS_PER_PAIR tempered 32-bit words per pair) into
// out + blockIdx.x * slicePairs * RANDS_PER_PAIR, continuing from states[blockIdx.x].
// Launch with MT_BLOCK_THREADS threads per block.
__global__ void __launch_bounds__(MT_BLOCK_THREADS)
mtGenerateKernel(MtState* states, uint32_t* __restrict__ out, BatchLayout L) {
    __shared__ uint32_t bufA[MT_N];
    __shared__ uint32_t bufB[MT_N];
    const int tid = threadIdx.x;
    MtState* state = states + blockIdx.x;
    unsigned long long r0, r1;
    sliceRange(L, blockIdx.x, r0, r1);
    const size_t count = static_cast<size_t>(r1 - r0) * RANDS_PER_PAIR;
    out += static_cast<size_t>(blockIdx.x) * L.slicePairs * RANDS_PER_PAIR;
    if (count == 0) return;

    if (tid < MT_N) bufA[tid] = state->mt[tid];
    int mti = state->mti;
    __syncthreads();

    uint32_t* cur = bufA;
    uint32_t* nxt = bufB;
    size_t emitted = 0;

    while (emitted < count) {
        if (mti >= MT_N) {
            // Stage 1: k in [0, 227): all inputs from the old state
            if (tid < MT_STAGE) {
                nxt[tid] = mtTwistWord(cur[tid], cur[tid + 1], cur[tid + MT_M]);
            }
            __syncthreads();
            // Stage 2: k in [227, 454): x[k+397-624] = new[k-227] from stage 1
            if (tid < MT_STAGE) {
                int k = tid + MT_STAGE;
                nxt[k] = mtTwistWord(cur[k], cur[k + 1], nxt[k - MT_STAGE]);
            }
            __syncthreads();
            // Stage 3: k in [454, 624): x[k+1] wraps to new[0] for k = 623
            if (tid < MT_N - 2 * MT_STAGE) {
                int k = tid + 2 * MT_STAGE;
                uint32_t xk1 = (k + 1 < MT_N) ? cur[k + 1] : nxt[0];
                nxt[k] = mtTwistWord(cur[k], xk1, nxt[k - MT_STAGE]);
            }
            __syncthreads();
            uint32_t* tmp = cur; cur = nxt; nxt = tmp;
            mti = 0;
        }

        size_t remaining = count - emitted;
        int n = static_cast<int>(remaining < static_cast<size_t>(MT_N - mti)
                                     ? remaining : static_cast<size_t>(MT_N - mti));
        if (tid < n) {
            out[emitted + static_cast<size_t>(tid)] = mtTemper(cur[mti + tid]);
        }
        mti += n;
        emitted += static_cast<size_t>(n);
    }

    __syncthreads();
    if (tid < MT_N) state->mt[tid] = cur[tid];
    if (tid == 0) state->mti = mti;
}

// Host-side seeding, identical to std::mt19937(seed)
void mtSeedHost(MtState& s, uint32_t seed) {
    s.mt[0] = seed;
    for (int i = 1; i < MT_N; ++i) {
        s.mt[i] = 1812433253u * (s.mt[i - 1] ^ (s.mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
    }
    s.mti = MT_N;
}

// Convert a raw 32-bit word to a float in [0,1) exactly like
// std::generate_canonical<float, 24>(std::mt19937&) does.
__device__ __forceinline__ float mtToUnitFloat(uint32_t x) {
    float r = __uint2float_rn(x) * (1.0f / 4294967296.0f);
    if (r >= 1.0f) r = nextafterf(1.0f, 0.0f);
    return r;
}

// ---- Polynomial arithmetic over GF(2) modulo the characteristic polynomial ----

// Reduction table: row j = x^(MT_DEGREE + j) mod p, j in [0, MT_DEGREE).
// Phase 1 (single block): rows 0..31 by bit-serial shift-and-reduce, then the
// chain rows 32g = rows 32(g-1) * x^32 mod p using rows 0..31 for the reduction.
__global__ void __launch_bounds__(MT_BLOCK_THREADS)
mtReductionTableChainKernel(const uint32_t* __restrict__ p, uint32_t* __restrict__ table) {
    __shared__ uint32_t bufA[MT_POLY_WORDS];
    __shared__ uint32_t bufB[MT_POLY_WORDS];
    __shared__ uint32_t pS[MT_POLY_WORDS];
    const int w = threadIdx.x;
    const int topWordIdx = MT_DEGREE >> 5;
    const int topBitIdx = MT_DEGREE & 31;
    if (w < MT_POLY_WORDS) {
        pS[w] = p[w];
        uint32_t v = pS[w];
        if (w == topWordIdx) v &= ~(1u << topBitIdx);   // x^L mod p = p - x^L
        bufA[w] = v;
    }
    __syncthreads();
    uint32_t* cur = bufA;
    uint32_t* nxt = bufB;

    // Rows 0..31: multiply by x and reduce by p when bit L appears
    for (int j = 0; j < 32; ++j) {
        if (w < MT_POLY_WORDS) {
            table[static_cast<size_t>(j) * MT_POLY_WORDS + w] = cur[w];
            uint32_t v = (cur[w] << 1) | (w > 0 ? (cur[w - 1] >> 31) : 0u);
            uint32_t topWord = (cur[topWordIdx] << 1) | (cur[topWordIdx - 1] >> 31);
            if ((topWord >> topBitIdx) & 1u) v ^= pS[w];
            nxt[w] = v;
        }
        __syncthreads();
        uint32_t* tmp = cur; cur = nxt; nxt = tmp;
    }

    // Chain: cur = row 32 now; rows 32g for g = 1 .. (MT_DEGREE-1)/32
    const int lastGroup = (MT_DEGREE - 1) / 32;
    for (int g = 1; g <= lastGroup; ++g) {
        if (w < MT_POLY_WORDS) {
            table[static_cast<size_t>(g) * 32 * MT_POLY_WORDS + w] = cur[w];
            // cur * x^32: shift by one word, bits >= L reduced via rows 0..31
            uint32_t v = (w > 0) ? cur[w - 1] : 0u;
            if (w == topWordIdx) v &= (1u << topBitIdx) - 1u;
            if (w > topWordIdx) v = 0u;
            for (int b = 0; b < 32; ++b) {
                if (polyBit(cur, MT_DEGREE - 32 + b)) {
                    v ^= __ldg(&table[static_cast<size_t>(b) * MT_POLY_WORDS + w]);
                }
            }
            nxt[w] = v;
        }
        __syncthreads();
        uint32_t* tmp = cur; cur = nxt; nxt = tmp;
    }
}

// Phase 2 (block g): rows 32g + i, i = 1..31, from row 32g and rows 0..31.
__global__ void __launch_bounds__(MT_BLOCK_THREADS)
mtReductionTableFillKernel(uint32_t* __restrict__ table) {
    __shared__ uint32_t row[MT_POLY_WORDS];
    const int w = threadIdx.x;
    const int g = blockIdx.x;
    const int topWordIdx = MT_DEGREE >> 5;
    const int topBitIdx = MT_DEGREE & 31;
    if (w < MT_POLY_WORDS) row[w] = table[static_cast<size_t>(g) * 32 * MT_POLY_WORDS + w];
    __syncthreads();
    if (w >= MT_POLY_WORDS) return;
    for (int i = 1; i < 32; ++i) {
        const int j = g * 32 + i;
        if (j >= MT_DEGREE) break;
        uint32_t v = (row[w] << i) | (w > 0 ? (row[w - 1] >> (32 - i)) : 0u);
        if (w == topWordIdx) v &= (1u << topBitIdx) - 1u;
        if (w > topWordIdx) v = 0u;
        for (int b = 0; b < i; ++b) {
            if (polyBit(row, MT_DEGREE - i + b)) {
                v ^= __ldg(&table[static_cast<size_t>(b) * MT_POLY_WORDS + w]);
            }
        }
        table[static_cast<size_t>(j) * MT_POLY_WORDS + w] = v;
    }
}

// out ^= (a^2 mod p) restricted to the rows handled by this block; `out` must be
// zero-initialised. Grid: ceil(MT_DEGREE / rowsPerBlock) blocks of MT_BLOCK_THREADS.
constexpr int MT_SQUARE_ROWS_PER_BLOCK = 64;
__global__ void __launch_bounds__(MT_BLOCK_THREADS)
mtPolySquareKernel(const uint32_t* __restrict__ a, const uint32_t* __restrict__ table,
                   uint32_t* __restrict__ out) {
    __shared__ uint32_t aS[MT_POLY_WORDS];
    const int w = threadIdx.x;
    if (w < MT_POLY_WORDS) aS[w] = a[w];
    __syncthreads();
    if (w >= MT_POLY_WORDS) return;

    // a^2 = sum a_i x^(2i). High part: bit j of (a^2 >> L) = a_((L+j)/2) if L+j even.
    uint32_t acc = 0;
    const int jBegin = blockIdx.x * MT_SQUARE_ROWS_PER_BLOCK;
    const int jEnd = min(jBegin + MT_SQUARE_ROWS_PER_BLOCK, MT_DEGREE - 1);
    for (int j = jBegin; j < jEnd; ++j) {
        const int q = MT_DEGREE + j;
        if (((q & 1) == 0) && polyBit(aS, q >> 1)) {
            acc ^= table[static_cast<size_t>(j) * MT_POLY_WORDS + w];
        }
    }
    if (blockIdx.x == 0) {
        // Low part: bits q < L of a^2
        uint32_t lo = 0;
        for (int t = 0; t < 32; ++t) {
            const int q = w * 32 + t;
            if (q < MT_DEGREE && ((q & 1) == 0) && polyBit(aS, q >> 1)) lo |= (1u << t);
        }
        acc ^= lo;
    }
    if (acc) atomicXor(&out[w], acc);
}

// a = a * x mod p (in place). Single block.
__global__ void __launch_bounds__(MT_BLOCK_THREADS)
mtPolyMulXKernel(uint32_t* a, const uint32_t* __restrict__ p) {
    const int w = threadIdx.x;
    uint32_t v = 0, pv = 0, top = 0;
    if (w < MT_POLY_WORDS) {
        v = (a[w] << 1) | (w > 0 ? (a[w - 1] >> 31) : 0u);
        pv = p[w];
        uint32_t topWord = (a[MT_DEGREE >> 5] << 1) |
                           ((MT_DEGREE >> 5) > 0 ? (a[(MT_DEGREE >> 5) - 1] >> 31) : 0u);
        top = (topWord >> (MT_DEGREE & 31)) & 1u;
    }
    __syncthreads();
    if (w < MT_POLY_WORDS) a[w] = top ? (v ^ pv) : v;
}

// Jump-ahead: for every block k with bit `bit` of k set, replace states[k] by
// g(T) * states[k], where g (degree < MT_DEGREE) is given as MT_POLY_WORDS words.
// Uses Horner's rule in groups of MT_STAGE powers of T (one twist stage each).
__global__ void __launch_bounds__(MT_BLOCK_THREADS)
mtJumpKernel(MtState* states, const uint32_t* __restrict__ g, int bit) {
    const int k = blockIdx.x;
    if (((k >> bit) & 1) == 0) return;

    __shared__ uint32_t ext[MT_N + MT_STAGE];   // window extended by one stage
    __shared__ uint32_t ring[1024];             // circular buffer for the Horner state
    __shared__ uint32_t gS[MT_POLY_WORDS];
    const int tid = threadIdx.x;
    MtState* state = states + k;

    if (tid < MT_N) ext[tid] = state->mt[tid];
    if (tid < MT_POLY_WORDS) gS[tid] = g[tid];
    ring[tid] = 0;
    if (tid + MT_BLOCK_THREADS < 1024) ring[tid + MT_BLOCK_THREADS] = 0;
    __syncthreads();
    if (tid < MT_STAGE) {
        ext[MT_N + tid] = mtTwistWord(ext[tid], ext[tid + 1], ext[tid + MT_M]);
    }
    __syncthreads();

    int base = 0;
    for (int u = MT_HORNER_GROUPS - 1; u >= 0; --u) {
        // S_u[k] = sum_v g_(227u+v) * ext[v + k]
        uint32_t s = 0;
        if (tid < MT_N) {
            const int i0 = u * MT_STAGE;
            for (int v0 = 0; v0 < MT_STAGE; v0 += 32) {
                // 32 coefficient bits g_(i0+v0) .. g_(i0+v0+31), possibly straddling two words
                const int i = i0 + v0;
                const int w = i >> 5, sh = i & 31;
                const uint32_t lo = (w < MT_POLY_WORDS) ? gS[w] : 0u;
                const uint32_t hi = (w + 1 < MT_POLY_WORDS) ? gS[w + 1] : 0u;
                uint32_t cw = sh ? ((lo >> sh) | (hi << (32 - sh))) : lo;
                if (v0 + 32 > MT_STAGE) cw &= (1u << (MT_STAGE - v0)) - 1u;
#pragma unroll
                for (int b = 0; b < 32; ++b) {
                    if ((cw >> b) & 1u) s ^= ext[v0 + b + tid];
                }
            }
        }
        // R = T^227 R
        if (tid < MT_STAGE) {
            uint32_t xk = ring[(base + tid) & 1023];
            uint32_t xk1 = ring[(base + tid + 1) & 1023];
            uint32_t xkm = ring[(base + tid + MT_M) & 1023];
            ring[(base + MT_N + tid) & 1023] = mtTwistWord(xk, xk1, xkm);
        }
        __syncthreads();
        base += MT_STAGE;
        // R ^= S_u
        if (tid < MT_N) ring[(base + tid) & 1023] ^= s;
        __syncthreads();
    }

    if (tid < MT_N) state->mt[tid] = ring[(base + tid) & 1023];
    if (tid == 0) state->mti = MT_N;
}

// Host: characteristic polynomial of the MT19937 transition via Berlekamp-Massey
// on the sequence of the live bit (bit 31) of the first state word.
// Returns p as MT_POLY_WORDS words with bit MT_DEGREE set.
std::vector<uint32_t> mtCharacteristicPolynomial() {
    const int L = MT_DEGREE;
    const int M = 2 * L + 256;              // sequence length (>= 2L to determine p)
    const int extra = 64;                   // extra terms to verify the recurrence

    // Generate the raw (untempered) state sequence x[0 .. M + extra + L + 64)
    std::vector<uint32_t> x(static_cast<size_t>(M + extra + L + 64));
    MtState seed;
    mtSeedHost(seed, 42);
    for (int i = 0; i < MT_N; ++i) x[i] = seed.mt[i];
    for (size_t n = MT_N; n < x.size(); ++n) {
        uint32_t y = (x[n - MT_N] & MT_UPPER_MASK) | (x[n - MT_N + 1] & MT_LOWER_MASK);
        x[n] = x[n - MT_N + MT_M] ^ (y >> 1) ^ ((y & 1u) ? MT_MATRIX_A : 0u);
    }
    auto seqBit = [&](size_t n) -> uint64_t { return x[n] >> 31; };

    // Reversed bit sequence: bit (M-1-n) holds s[n], so s[n-i] is bit (M-1-n+i).
    // 64 pre-shifted copies make the discrepancy a plain AND/XOR over words.
    const int W = (M + 63) / 64 + 2;
    std::vector<uint64_t> srev(W + 1, 0);
    for (int n = 0; n < M; ++n) {
        if (seqBit(n)) srev[(M - 1 - n) / 64] |= (1ull << ((M - 1 - n) % 64));
    }
    std::vector<uint64_t> shifted(static_cast<size_t>(64) * (W + 1), 0);
    for (int sh = 0; sh < 64; ++sh) {
        uint64_t* dst = shifted.data() + static_cast<size_t>(sh) * (W + 1);
        for (int t = 0; t < W; ++t) {
            dst[t] = (srev[t] >> sh) | (sh ? (srev[t + 1] << (64 - sh)) : 0ull);
        }
    }

    std::vector<uint64_t> C(W, 0), B(W, 0), T(W, 0);
    C[0] = 1;
    B[0] = 1;
    int Lc = 0;
    int m = 1;
    for (int n = 0; n < M; ++n) {
        // discrepancy d = sum_{i=0}^{Lc} C_i s[n-i]
        const int off = M - 1 - n;
        const uint64_t* sv = shifted.data() + static_cast<size_t>(off % 64) * (W + 1) + off / 64;
        const int nw = Lc / 64 + 1;
        uint64_t par = 0;
        for (int t = 0; t < nw; ++t) par ^= C[t] & sv[t];
        if (__builtin_parityll(par) == 0) {
            ++m;
            continue;
        }
        // C ^= B << m  (B has degree <= n, so only words up to (n + m) / 64 are touched)
        auto xorShift = [&](uint64_t* dst, const uint64_t* srcp, int s, int tmax) {
            const int ws = s / 64, bs = s % 64;
            if (bs == 0) {
                for (int t = ws; t <= tmax; ++t) dst[t] ^= srcp[t - ws];
            } else {
                dst[ws] ^= srcp[0] << bs;
                for (int t = ws + 1; t <= tmax; ++t) {
                    dst[t] ^= (srcp[t - ws] << bs) | (srcp[t - ws - 1] >> (64 - bs));
                }
            }
        };
        const int tmax = std::min(W - 1, (n + m) / 64 + 1);
        if (2 * Lc <= n) {
            std::copy(C.begin(), C.begin() + tmax + 1, T.begin());
            xorShift(C.data(), B.data(), m, tmax);
            Lc = n + 1 - Lc;
            std::swap(B, T);
            m = 1;
        } else {
            xorShift(C.data(), B.data(), m, tmax);
            ++m;
        }
    }

    if (Lc != L) {
        fprintf(stderr, "Berlekamp-Massey returned degree %d, expected %d\n", Lc, L);
        exit(1);
    }

    // p(x) = x^L C(1/x): p_j = C_(L-j)
    std::vector<uint32_t> p(MT_POLY_WORDS, 0);
    for (int j = 0; j <= L; ++j) {
        const int i = L - j;
        if ((C[i / 64] >> (i % 64)) & 1ull) p[j >> 5] |= (1u << (j & 31));
    }

    // Verify s[n+L] = sum_{j<L} p_j s[n+j] on terms beyond those used by Berlekamp-Massey
    const int PW = MT_POLY_WORDS / 2;      // p as 64-bit words
    std::vector<uint64_t> p64(PW);
    for (int t = 0; t < PW; ++t) p64[t] = p[2 * t] | (static_cast<uint64_t>(p[2 * t + 1]) << 32);
    const size_t FW = x.size() / 64 + 2;
    std::vector<uint64_t> sfwd(FW, 0);
    for (size_t n = 0; n < x.size(); ++n) {
        if (seqBit(n)) sfwd[n / 64] |= (1ull << (n % 64));
    }
    for (int n = M - L; n < M - L + extra; ++n) {
        const int base = n / 64, sh = n % 64;
        uint64_t par = 0;
        for (int t = 0; t < PW; ++t) {
            uint64_t v = (sfwd[base + t] >> sh) | (sh ? (sfwd[base + t + 1] << (64 - sh)) : 0ull);
            par ^= p64[t] & v;
        }
        if (__builtin_parityll(par) != 0) {
            fprintf(stderr, "Characteristic polynomial verification failed\n");
            exit(1);
        }
    }
    return p;
}

// Host helper: launch a^2 mod p into out (out is overwritten)
void mtPolySquare(const uint32_t* a, const uint32_t* table, uint32_t* out, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(out, 0, MT_POLY_WORDS * sizeof(uint32_t), stream));
    const int blocks = (MT_DEGREE - 1 + MT_SQUARE_ROWS_PER_BLOCK - 1) / MT_SQUARE_ROWS_PER_BLOCK;
    mtPolySquareKernel<<<blocks, MT_BLOCK_THREADS, 0, stream>>>(a, table, out);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Device geometry helpers
// ============================================================================

__device__ __forceinline__ float3 f3sub(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float3 f3neg(float3 a) { return make_float3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ float f3dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float3 f3cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ float3 f3of4(float4 v) { return make_float3(v.x, v.y, v.z); }

// Generate a point inside a triangle using barycentric coordinates
__device__ __forceinline__ float3 pointInTriangle(float3 a, float3 b, float3 c, float u, float v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    float3 ab = f3sub(b, a);
    float3 ac = f3sub(c, a);
    return make_float3(a.x + ab.x * u + ac.x * v,
                       a.y + ab.y * u + ac.y * v,
                       a.z + ab.z * u + ac.z * v);
}

// Ray-Triangle Intersection (Möller-Trumbore algorithm)
__device__ __forceinline__ float rayTriangleIntersect(float3 orig, float3 dir,
                                                      float3 v0, float3 v1, float3 v2) {
    float3 e1 = f3sub(v1, v0);
    float3 e2 = f3sub(v2, v0);
    float3 pvec = f3cross(dir, e2);
    float det = f3dot(e1, pvec);

    const float inf = 3.402823466e+38f;  // numeric_limits<float>::max()
    if (fabsf(det) < EPSILON) return inf;

    float invDet = 1.0f / det;
    float3 tvec = f3sub(orig, v0);
    float u = f3dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return inf;

    float3 qvec = f3cross(tvec, e1);
    float v = f3dot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return inf;

    return f3dot(e2, qvec) * invDet;
}

struct DeviceMesh {
    const float4* __restrict__ a;
    const float4* __restrict__ b;
    const float4* __restrict__ c;
    const float4* __restrict__ n;
    const int* __restrict__ cellStart;   // res^3 + 1
    const int* __restrict__ cellTris;
    DeviceGridParams grid;
};

__device__ __forceinline__ int gridCoord(float v, float origin, float inv, int res) {
    int c = __float2int_rd((v - origin) * inv);
    return min(max(c, 0), res - 1);
}

// Check if a ray between two triangles is blocked by any other triangle.
// Walks the segment through the grid (3D-DDA) and tests the triangles of every
// cell it passes through (any-hit).
__device__ bool isRayBlocked(float3 from, float3 to, const DeviceMesh& mesh,
                             int srcTriIdx, int dstTriIdx) {
    float3 dir = f3sub(to, from);
    float rayLen = sqrtf(f3dot(dir, dir));
    if (rayLen < EPSILON) return true;
    float3 dirNorm = make_float3(dir.x / rayLen, dir.y / rayLen, dir.z / rayLen);

    const DeviceGridParams& g = mesh.grid;
    const int res = g.res;
    int cx = gridCoord(from.x, g.origin.x, g.invCellSize.x, res);
    int cy = gridCoord(from.y, g.origin.y, g.invCellSize.y, res);
    int cz = gridCoord(from.z, g.origin.z, g.invCellSize.z, res);
    const int ex = gridCoord(to.x, g.origin.x, g.invCellSize.x, res);
    const int ey = gridCoord(to.y, g.origin.y, g.invCellSize.y, res);
    const int ez = gridCoord(to.z, g.origin.z, g.invCellSize.z, res);

    // DDA setup: parametric distance (in units of the segment, t in [0,1]) to the
    // next cell boundary per axis, and per-cell increments.
    const float inf = 3.402823466e+38f;
    const int sx = dir.x > 0.0f ? 1 : (dir.x < 0.0f ? -1 : 0);
    const int sy = dir.y > 0.0f ? 1 : (dir.y < 0.0f ? -1 : 0);
    const int sz = dir.z > 0.0f ? 1 : (dir.z < 0.0f ? -1 : 0);
    const float dtx = sx != 0 ? fabsf(g.cellSize.x / dir.x) : inf;
    const float dty = sy != 0 ? fabsf(g.cellSize.y / dir.y) : inf;
    const float dtz = sz != 0 ? fabsf(g.cellSize.z / dir.z) : inf;
    float tx = sx != 0 ? ((g.origin.x + (cx + (sx > 0 ? 1 : 0)) * g.cellSize.x) - from.x) / dir.x : inf;
    float ty = sy != 0 ? ((g.origin.y + (cy + (sy > 0 ? 1 : 0)) * g.cellSize.y) - from.y) / dir.y : inf;
    float tz = sz != 0 ? ((g.origin.z + (cz + (sz > 0 ? 1 : 0)) * g.cellSize.z) - from.z) / dir.z : inf;

    for (int steps = 0; steps < 3 * res + 3; ++steps) {
        const int cell = (cz * res + cy) * res + cx;
        const int begin = __ldg(&mesh.cellStart[cell]);
        const int endIdx = __ldg(&mesh.cellStart[cell + 1]);
        for (int t = begin; t < endIdx; ++t) {
            int idx = __ldg(&mesh.cellTris[t]);
            if (idx == srcTriIdx || idx == dstTriIdx) continue;

            float3 v0 = f3of4(__ldg(&mesh.a[idx]));
            float3 v1 = f3of4(__ldg(&mesh.b[idx]));
            float3 v2 = f3of4(__ldg(&mesh.c[idx]));
            float dist = rayTriangleIntersect(from, dirNorm, v0, v1, v2);
            if (dist > EPSILON && dist < rayLen - EPSILON) {
                return true;  // Ray is blocked
            }
        }

        if (cx == ex && cy == ey && cz == ez) break;
        // Advance to the next cell along the segment
        if (tx <= ty && tx <= tz) {
            if (tx > 1.0f) break;
            cx += sx; tx += dtx;
            if (cx < 0 || cx >= res) break;
        } else if (ty <= tz) {
            if (ty > 1.0f) break;
            cy += sy; ty += dty;
            if (cy < 0 || cy >= res) break;
        } else {
            if (tz > 1.0f) break;
            cz += sz; tz += dtz;
            if (cz < 0 || cz >= res) break;
        }
    }
    return false;
}

// Compute the cosine of angle between vector and triangle normal
__device__ __forceinline__ float cosPhi(float3 v, float3 normal) {
    float vNorm = sqrtf(f3dot(v, v));
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, f3dot(v, normal) / vNorm);
}

// Pair (i, j) participates in form factor computation (and consumes random numbers)
// unless i == j or the triangles face the same direction.
__device__ __forceinline__ bool pairActive(int i, int j, const float4* __restrict__ normals) {
    if (i == j) return false;
    float3 nI = f3of4(__ldg(&normals[i]));
    float3 nJ = f3of4(__ldg(&normals[j]));
    return !(f3dot(nI, nJ) > 0.99f);
}

// ============================================================================
// Form Factor Kernels (Kij)
// ============================================================================

// Count active pairs per row (block per row)
__global__ void countActivePairsKernel(int N, const float4* __restrict__ normals,
                                       unsigned long long* __restrict__ rowCounts) {
    __shared__ int warpSums[32];
    const int i = blockIdx.x;
    int cnt = 0;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        cnt += pairActive(i, j, normals) ? 1 : 0;
    }
    for (int off = 16; off > 0; off >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, off);
    if ((threadIdx.x & 31) == 0) warpSums[threadIdx.x >> 5] = cnt;
    __syncthreads();
    if (threadIdx.x < 32) {
        int nw = (blockDim.x + 31) >> 5;
        int v = (threadIdx.x < nw) ? warpSums[threadIdx.x] : 0;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (threadIdx.x == 0) rowCounts[i] = static_cast<unsigned long long>(v);
    }
}

// Compact the active pairs with ranks in a slice into (pairI, pairJ) lists.
// Grid: (rows per slice, segments); block (x, k) handles row (firstRow(k) + x).
__global__ void __launch_bounds__(1024)
compactPairsKernel(int N, BatchLayout L, const float4* __restrict__ normals,
                   const unsigned long long* __restrict__ rowStart,
                   int* __restrict__ pairI, int* __restrict__ pairJ) {
    __shared__ int warpTotals[32];
    __shared__ int chunkTotal;
    __shared__ int rowLo;
    const int k = blockIdx.y;
    unsigned long long r0, r1;
    sliceRange(L, k, r0, r1);
    if (r0 >= r1) return;

    if (threadIdx.x == 0) {
        // first row whose range contains r0: largest i with rowStart[i] <= r0
        int lo = 0, hi = N - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (rowStart[mid] <= r0) lo = mid; else hi = mid - 1;
        }
        rowLo = lo;
    }
    __syncthreads();
    const int i = rowLo + blockIdx.x;
    if (i >= N || rowStart[i] >= r1) return;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nWarps = blockDim.x >> 5;
    const unsigned long long rowBase = rowStart[i];
    const unsigned long long listBase = static_cast<unsigned long long>(k) * L.slicePairs;
    int running = 0;

    for (int j0 = 0; j0 < N; j0 += blockDim.x) {
        int j = j0 + threadIdx.x;
        int flag = (j < N && pairActive(i, j, normals)) ? 1 : 0;

        // Inclusive warp scan
        int incl = flag;
        for (int off = 1; off < 32; off <<= 1) {
            int v = __shfl_up_sync(0xffffffffu, incl, off);
            if (lane >= off) incl += v;
        }
        if (lane == 31) warpTotals[warp] = incl;
        __syncthreads();
        if (warp == 0) {
            int v = (lane < nWarps) ? warpTotals[lane] : 0;
            int s = v;
            for (int off = 1; off < 32; off <<= 1) {
                int w = __shfl_up_sync(0xffffffffu, s, off);
                if (lane >= off) s += w;
            }
            if (lane < nWarps) warpTotals[lane] = s - v;  // exclusive warp offsets
            if (lane == 31) chunkTotal = s;
        }
        __syncthreads();

        if (flag) {
            unsigned long long rank = rowBase + static_cast<unsigned long long>(
                running + warpTotals[warp] + (incl - 1));
            if (rank >= r0 && rank < r1) {
                size_t pos = static_cast<size_t>(listBase + (rank - r0));
                pairI[pos] = i;
                pairJ[pos] = j;
            }
        }
        running += chunkTotal;
        __syncthreads();
    }
}

// Form factors: one thread per (pair, ray). NUM_RAYS consecutive threads share a pair.
// Writes kij[i * N + j].
constexpr int FF_THREADS_PER_PAIR = NUM_RAYS;
__global__ void __launch_bounds__(256)
formFactorKernel(int N, BatchLayout L, DeviceMesh mesh,
                 const int* __restrict__ pairI, const int* __restrict__ pairJ,
                 const uint4* __restrict__ rands, float* __restrict__ kij) {
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t pair = gid / NUM_RAYS;   // position in the batch's pair list
    const int ray = static_cast<int>(gid % NUM_RAYS);
    // Lanes of this pair within the warp (validity below is uniform per pair)
    const unsigned pairMask = 0xffffu << (threadIdx.x & (32 - NUM_RAYS));
    const int k = static_cast<int>(pair / L.slicePairs);
    if (k >= L.segments) return;
    unsigned long long r0, r1;
    sliceRange(L, k, r0, r1);
    if (pair % L.slicePairs >= r1 - r0) return;

    const int i = __ldg(&pairI[pair]);
    const int j = __ldg(&pairJ[pair]);

    const float3 aI = f3of4(__ldg(&mesh.a[i]));
    const float3 bI = f3of4(__ldg(&mesh.b[i]));
    const float3 cI = f3of4(__ldg(&mesh.c[i]));
    const float3 nI = f3of4(__ldg(&mesh.n[i]));
    const float3 aJ = f3of4(__ldg(&mesh.a[j]));
    const float3 bJ = f3of4(__ldg(&mesh.b[j]));
    const float3 cJ = f3of4(__ldg(&mesh.c[j]));
    const float3 nJ = f3of4(__ldg(&mesh.n[j]));

    // Each ray consumes 4 consecutive random numbers: uI, vI, uJ, vJ
    const uint4 rw = __ldg(&rands[pair * NUM_RAYS + ray]);
    const float uI = mtToUnitFloat(rw.x);
    const float vI = mtToUnitFloat(rw.y);
    const float uJ = mtToUnitFloat(rw.z);
    const float vJ = mtToUnitFloat(rw.w);

    float3 pI = pointInTriangle(aI, bI, cI, uI, vI);
    float3 pJ = pointInTriangle(aJ, bJ, cJ, uJ, vJ);

    float contrib = ZERO;
    if (!isRayBlocked(pI, pJ, mesh, i, j)) {
        float3 v = f3sub(pJ, pI);
        float distSqr = f3dot(v, v);
        if (!(distSqr < EPSILON)) {
            float cosPhiI = cosPhi(v, nI);
            float cosPhiJ = cosPhi(f3neg(v), nJ);
            if (!(cosPhiI <= ZERO || cosPhiJ <= ZERO)) {
                contrib = (cosPhiI * cosPhiJ) / (PI * distSqr);
            }
        }
    }

    // Sum the NUM_RAYS contributions in the original sequential order
    float sum = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        sum += __shfl_sync(pairMask, contrib, r, NUM_RAYS);
    }

    if (ray == 0) {
        kij[static_cast<size_t>(i) * N + j] = sum * INV_NUM_RAYS;
    }
}

// ============================================================================
// Tau (time delay) Kernel
// ============================================================================

__device__ __forceinline__ float3 triCenter(float3 a, float3 b, float3 c) {
    return make_float3((a.x + b.x + c.x) / 3.0f, (a.y + b.y + c.y) / 3.0f, (a.z + b.z + c.z) / 3.0f);
}

// Time delays are stored as 8-bit values (the room has diameter 20 and the wave
// travels 0.5 per step, so delays are at most 40); larger values are reported.
__global__ void timeDelayKernel(int N, const float4* __restrict__ ta, const float4* __restrict__ tb,
                                const float4* __restrict__ tc, unsigned char* __restrict__ tau,
                                int* __restrict__ overflow) {
    const size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(N) * N;
    if (gid >= total) return;
    const int i = static_cast<int>(gid / N);
    const int j = static_cast<int>(gid % N);
    if (i == j) {
        tau[gid] = 0;
        return;
    }
    float3 ci = triCenter(f3of4(__ldg(&ta[i])), f3of4(__ldg(&tb[i])), f3of4(__ldg(&tc[i])));
    float3 cj = triCenter(f3of4(__ldg(&ta[j])), f3of4(__ldg(&tb[j])), f3of4(__ldg(&tc[j])));
    float3 d = f3sub(ci, cj);
    float dist = sqrtf(f3dot(d, d));
    const int t = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
    if (t < 0 || t > 255) atomicOr(overflow, 1);
    tau[gid] = static_cast<unsigned char>(t);
}

// ============================================================================
// Simulation Kernel (Wave Propagation), one launch per timestep
// ============================================================================

// One warp per receiving triangle i. The lanes load 32 consecutive emitters j
// at a time (coalesced) and compute their terms; the terms are then added in
// index order on every lane via shuffles, so the result is bit-identical to the
// sequential summation (terms the sequential version skips are exactly zero and
// are left out, which does not change a sum of non-negative values).
__global__ void __launch_bounds__(256)
simulationStepKernel(int N, int t, const float* __restrict__ kij,
                     const unsigned char* __restrict__ tau, const float* __restrict__ areas,
                     const float* __restrict__ rho, const float* __restrict__ radE,
                     float* radB) {
    const int warpsPerBlock = blockDim.x >> 5;
    const int i = blockIdx.x * warpsPerBlock + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (i >= N) return;

    const size_t row = static_cast<size_t>(i) * N;
    float sumB = ZERO;
    for (int j0 = 0; j0 < N; j0 += 32) {
        const int j = j0 + lane;
        float term = ZERO;
        if (j < N && j != i) {
            const int tauij = static_cast<int>(__ldg(&tau[row + j]));
            const float k = __ldg(&kij[row + j]);
            const int srcTime = t - tauij;
            if (srcTime >= 0 && k > ZERO) {
                // Rows < t were written by earlier launches: read-only here
                const float radJ = __ldg(&radB[static_cast<size_t>(srcTime) * N + j]);
                if (radJ > ZERO) term = fminf(k * __ldg(&areas[j]), ONE) * radJ;
            }
        }
        // Add the non-zero terms of this chunk in increasing j on every lane
        unsigned mask = __ballot_sync(0xffffffffu, term != ZERO);
        while (mask) {
            const int src = __ffs(mask) - 1;
            mask &= mask - 1;
            sumB += __shfl_sync(0xffffffffu, term, src);
        }
    }

    if (lane == 0) {
        radB[static_cast<size_t>(t) * N + i] = __ldg(&rho[i]) * sumB + __ldg(&radE[static_cast<size_t>(t) * N + i]);
    }
}

// ============================================================================
// Distance Kernels (Cross-Correlation)
// ============================================================================

// corr[t * N + i] = sum_{tt >= t} radB[tt][i] * radB[tt - t][src]
__global__ void crossCorrelationKernel(int N, int T, int src, const float* __restrict__ radB,
                                       float* __restrict__ corr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= N || t >= T) return;

    float sum = ZERO;
    for (int tt = t; tt < T; ++tt) {
        float pB = __ldg(&radB[static_cast<size_t>(tt) * N + i]);
        float pS = __ldg(&radB[static_cast<size_t>(tt - t) * N + src]);
        sum += pS * pB;
    }
    corr[static_cast<size_t>(t) * N + i] = sum;
}

__global__ void argmaxDistanceKernel(int N, int T, const float* __restrict__ corr,
                                     float* __restrict__ distances) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float maxCorr = ZERO;
    int bestT = 0;
    for (int t = 0; t < T; ++t) {
        float sum = __ldg(&corr[static_cast<size_t>(t) * N + i]);
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }
    distances[i] = WAVE_SPEED * static_cast<float>(bestT);
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Form factors (N x N matrix, row-major; downloaded for validation)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix; downloaded for validation)
    std::vector<val_t> distances;   // Computed distances from source

    UniformGrid grid;               // Spatial acceleration structure

    size_t sourceIndex;

    // Device data
    float4* dTriA = nullptr;
    float4* dTriB = nullptr;
    float4* dTriC = nullptr;
    float4* dTriN = nullptr;
    int* dCellStart = nullptr;
    int* dCellTris = nullptr;
    float* dAreas = nullptr;
    float* dRho = nullptr;
    float* dKij = nullptr;          // Form factors (N x N, row-major)
    unsigned char* dTau = nullptr;  // Time delays (N x N, row-major; values fit in 8 bits)
    int* dTauOverflow = nullptr;    // Set if a time delay does not fit in 8 bits
    float* dRadE = nullptr;
    float* dRadB = nullptr;
    float* dDistances = nullptr;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    const size_t N = state.numTriangles;

    printf("Generated icosphere mesh with %zu triangles\n", N);

    // Build the acceleration grid for visibility tests
    printf("Building acceleration grid...\n");
    state.grid.build(state.triangles);

    // Initialize areas
    state.areas.resize(N);
    for (size_t i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(N, reflectivity);

    // Initialize host matrices that are needed on the host
    state.radE.resize(timesteps * N, ZERO);
    state.distances.resize(N, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Upload geometry and allocate device matrices
    std::vector<float4> ha(N), hb(N), hc(N), hn(N);
    for (size_t i = 0; i < N; ++i) {
        const Triangle& tr = state.triangles[i];
        ha[i] = make_float4(tr.a.x, tr.a.y, tr.a.z, 0.0f);
        hb[i] = make_float4(tr.b.x, tr.b.y, tr.b.z, 0.0f);
        hc[i] = make_float4(tr.c.x, tr.c.y, tr.c.z, 0.0f);
        hn[i] = make_float4(tr._normal.x, tr._normal.y, tr._normal.z, 0.0f);
    }

    CUDA_CHECK(cudaMalloc(&state.dTriA, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.dTriB, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.dTriC, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.dTriN, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.dCellStart, state.grid.cellStart.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.dCellTris, std::max<size_t>(state.grid.cellTris.size(), 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.dAreas, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dRho, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dKij, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dTau, N * N * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&state.dTauOverflow, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.dRadE, timesteps * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dRadB, timesteps * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.dDistances, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(state.dTriA, ha.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dTriB, hb.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dTriC, hc.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dTriN, hn.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dCellStart, state.grid.cellStart.data(),
                          state.grid.cellStart.size() * sizeof(int), cudaMemcpyHostToDevice));
    if (!state.grid.cellTris.empty()) {
        CUDA_CHECK(cudaMemcpy(state.dCellTris, state.grid.cellTris.data(),
                              state.grid.cellTris.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(state.dAreas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRho, state.rho.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRadE, state.radE.data(), timesteps * N * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.dKij, 0, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.dTau, 0, N * N * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemset(state.dTauOverflow, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, timesteps * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(state.dDistances, 0, N * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void releaseDeviceMemory(SimulationState& state) {
    cudaFree(state.dTriA);
    cudaFree(state.dTriB);
    cudaFree(state.dTriC);
    cudaFree(state.dTriN);
    cudaFree(state.dCellStart);
    cudaFree(state.dCellTris);
    cudaFree(state.dAreas);
    cudaFree(state.dRho);
    cudaFree(state.dKij);
    cudaFree(state.dTau);
    cudaFree(state.dTauOverflow);
    cudaFree(state.dRadE);
    cudaFree(state.dRadB);
    cudaFree(state.dDistances);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const int N = static_cast<int>(state.numTriangles);

    DeviceMesh mesh;
    mesh.a = state.dTriA;
    mesh.b = state.dTriB;
    mesh.c = state.dTriC;
    mesh.n = state.dTriN;
    mesh.cellStart = state.dCellStart;
    mesh.cellTris = state.dCellTris;
    mesh.grid = state.grid.params;

    cudaStream_t genStream, kijStream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&genStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&kijStream, cudaStreamNonBlocking));

    // 1. Count active pairs per row and build row offsets (exclusive scan)
    unsigned long long* dRowCounts = nullptr;
    unsigned long long* dRowStart = nullptr;
    CUDA_CHECK(cudaMalloc(&dRowCounts, N * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&dRowStart, (N + 1) * sizeof(unsigned long long)));
    countActivePairsKernel<<<N, 256, 0, kijStream>>>(N, state.dTriN, dRowCounts);
    CUDA_CHECK(cudaGetLastError());

    std::vector<unsigned long long> rowCounts(N), rowStart(N + 1);
    CUDA_CHECK(cudaMemcpyAsync(rowCounts.data(), dRowCounts, N * sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost, kijStream));
    CUDA_CHECK(cudaStreamSynchronize(kijStream));
    rowStart[0] = 0;
    for (int i = 0; i < N; ++i) rowStart[i + 1] = rowStart[i] + rowCounts[i];
    CUDA_CHECK(cudaMemcpyAsync(dRowStart, rowStart.data(), (N + 1) * sizeof(unsigned long long),
                               cudaMemcpyHostToDevice, kijStream));
    const unsigned long long totalPairs = rowStart[N];

    // 2. Batch layout. The random stream is split into `segments` contiguous
    //    segments (one generator block each); a batch takes one slice of every
    //    segment. Segment boundaries are multiples of 624 words so that jumped
    //    states are twist-aligned. Small problems use a single segment and need
    //    no jump-ahead. Large problems first process a "prologue" batch with a
    //    single generator block while the host computes the characteristic
    //    polynomial needed for the jumps.
    const unsigned long long maxBatchPairs = 1ull << 18;     // 64 MB of random words per batch
    const unsigned long long parallelThreshold = 1ull << 19; // pairs (~33M words)
    const int maxSegments = 64;
    const bool parallelGen = totalPairs >= parallelThreshold;

    BatchLayout prologue{};
    BatchLayout mainLayout{};
    if (parallelGen) {
        prologue.firstPair = 0;
        // Sized so that its single-block generation (~0.7 ns/word) roughly covers
        // the host-side characteristic polynomial computation (a few ms).
        prologue.totalPairs = (maxBatchPairs / 2 / 39) * 39;   // 64 * pairs multiple of 624
        prologue.segPairs = prologue.totalPairs;
        prologue.slicePairs = prologue.totalPairs;
        prologue.segments = 1;

        mainLayout.firstPair = prologue.totalPairs;
        mainLayout.totalPairs = totalPairs;
        mainLayout.segments = maxSegments;
        unsigned long long seg = (totalPairs - mainLayout.firstPair + maxSegments - 1) / maxSegments;
        mainLayout.segPairs = ((seg + 38) / 39) * 39;
        mainLayout.slicePairs = maxBatchPairs / maxSegments;
    } else {
        mainLayout.firstPair = 0;
        mainLayout.totalPairs = totalPairs;
        mainLayout.segments = 1;
        mainLayout.segPairs = std::max<unsigned long long>(totalPairs, 1);
        mainLayout.slicePairs = std::min<unsigned long long>(maxBatchPairs, mainLayout.segPairs);
    }
    const size_t batchPairsCap = std::max<size_t>(
        static_cast<size_t>(mainLayout.slicePairs) * mainLayout.segments,
        parallelGen ? static_cast<size_t>(prologue.slicePairs) : 0);
    const size_t batchWordsCap = batchPairsCap * RANDS_PER_PAIR;

    // 3. Generator states (segment 0 is the seeded generator) and batch buffers
    std::vector<MtState> hostStates(mainLayout.segments);
    for (int k = 0; k < mainLayout.segments; ++k) mtSeedHost(hostStates[k], 42);
    MtState* dStates = nullptr;
    CUDA_CHECK(cudaMalloc(&dStates, mainLayout.segments * sizeof(MtState)));
    CUDA_CHECK(cudaMemcpyAsync(dStates, hostStates.data(), mainLayout.segments * sizeof(MtState),
                               cudaMemcpyHostToDevice, genStream));

    constexpr int NBUF = 2;
    uint32_t* dRands[NBUF];
    int* dPairI[NBUF];
    int* dPairJ[NBUF];
    cudaEvent_t genDone[NBUF], kijDone[NBUF];
    for (int s = 0; s < NBUF; ++s) {
        CUDA_CHECK(cudaMalloc(&dRands[s], batchWordsCap * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dPairI[s], batchPairsCap * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dPairJ[s], batchPairsCap * sizeof(int)));
        CUDA_CHECK(cudaEventCreateWithFlags(&genDone[s], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&kijDone[s], cudaEventDisableTiming));
    }

    unsigned long long pairsEnqueued = 0;
    int progressRows = 0;
    int bufferIndex = 0;

    // Enqueue one batch: random words on genStream, compaction + form factors on
    // kijStream, double-buffered so generation of the next batch overlaps.
    auto enqueueBatch = [&](const BatchLayout& layout) {
        const int s = bufferIndex % NBUF;
        ++bufferIndex;

        int maxRows = 0;
        unsigned long long batchPairs = 0;
        for (int k = 0; k < layout.segments; ++k) {
            unsigned long long r0, r1;
            sliceRange(layout, k, r0, r1);
            batchPairs += r1 - r0;
            if (r1 > r0) {
                int rowLo = static_cast<int>(std::upper_bound(rowStart.begin(), rowStart.end(), r0) -
                                             rowStart.begin()) - 1;
                int rowHi = static_cast<int>(std::upper_bound(rowStart.begin(), rowStart.end(), r1 - 1) -
                                             rowStart.begin()) - 1;
                maxRows = std::max(maxRows, rowHi - rowLo + 1);
            }
        }
        if (batchPairs == 0) return;

        // Generate the random words for this batch (after buffer s is free)
        CUDA_CHECK(cudaStreamWaitEvent(genStream, kijDone[s], 0));
        mtGenerateKernel<<<layout.segments, MT_BLOCK_THREADS, 0, genStream>>>(dStates, dRands[s], layout);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(genDone[s], genStream));

        // Compact pairs and compute form factors
        CUDA_CHECK(cudaStreamWaitEvent(kijStream, genDone[s], 0));
        dim3 cgrid(maxRows, layout.segments);
        compactPairsKernel<<<cgrid, 1024, 0, kijStream>>>(N, layout, state.dTriN, dRowStart,
                                                           dPairI[s], dPairJ[s]);
        CUDA_CHECK(cudaGetLastError());

        const size_t threads = static_cast<size_t>(layout.slicePairs) * layout.segments * FF_THREADS_PER_PAIR;
        const unsigned blocks = static_cast<unsigned>((threads + 255) / 256);
        formFactorKernel<<<blocks, 256, 0, kijStream>>>(
            N, layout, mesh, dPairI[s], dPairJ[s],
            reinterpret_cast<const uint4*>(dRands[s]), state.dKij);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(kijDone[s], kijStream));

        // Progress report (in units of equivalent rows)
        pairsEnqueued += batchPairs;
        int rowsDone = static_cast<int>((pairsEnqueued * static_cast<unsigned long long>(N)) /
                                        std::max<unsigned long long>(totalPairs, 1));
        for (int i = progressRows; i < rowsDone; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == N) {
                printf("  Progress: %d/%d triangles\n", i + 1, N);
            }
        }
        progressRows = rowsDone;
    };

    // 4. Prologue batch (single generator block) overlapping the host-side
    //    polynomial computation, then jump-ahead for the remaining segments.
    uint32_t* dTable = nullptr;
    uint32_t* dPolys = nullptr;   // [0] = p, [1] = scratch, [2 + b] = x^(2^b * J) mod p
    if (parallelGen) {
        prologue.batch = 0;
        enqueueBatch(prologue);

        std::vector<uint32_t> p = mtCharacteristicPolynomial();

        // All segments continue from the state reached after the prologue ...
        for (int k = 1; k < mainLayout.segments; ++k) {
            CUDA_CHECK(cudaMemcpyAsync(dStates + k, dStates, sizeof(MtState),
                                       cudaMemcpyDeviceToDevice, genStream));
        }

        int jumpBits = 0;
        while ((1 << jumpBits) < mainLayout.segments) ++jumpBits;
        const int numPolys = 2 + jumpBits;
        CUDA_CHECK(cudaMalloc(&dTable, static_cast<size_t>(MT_DEGREE) * MT_POLY_WORDS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dPolys, static_cast<size_t>(numPolys) * MT_POLY_WORDS * sizeof(uint32_t)));
        uint32_t* dP = dPolys;
        uint32_t* dScratch = dPolys + MT_POLY_WORDS;
        auto dJump = [&](int b) { return dPolys + static_cast<size_t>(2 + b) * MT_POLY_WORDS; };
        CUDA_CHECK(cudaMemcpyAsync(dP, p.data(), MT_POLY_WORDS * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, genStream));
        mtReductionTableChainKernel<<<1, MT_BLOCK_THREADS, 0, genStream>>>(dP, dTable);
        CUDA_CHECK(cudaGetLastError());
        mtReductionTableFillKernel<<<(MT_DEGREE - 1) / 32 + 1, MT_BLOCK_THREADS, 0, genStream>>>(dTable);
        CUDA_CHECK(cudaGetLastError());

        // ... and segment k is jumped ahead by k * J words, J = 64 * segPairs:
        // g = x^J mod p by square-and-multiply, then x^(2^b J) by repeated squaring.
        const unsigned long long J = mainLayout.segPairs * RANDS_PER_PAIR;
        std::vector<uint32_t> one(MT_POLY_WORDS, 0);
        one[0] = 1u;
        uint32_t* r = dJump(0);
        uint32_t* tmp = dScratch;
        CUDA_CHECK(cudaMemcpyAsync(r, one.data(), MT_POLY_WORDS * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, genStream));
        int topBit = 63;
        while (topBit > 0 && !((J >> topBit) & 1ull)) --topBit;
        for (int b = topBit; b >= 0; --b) {
            mtPolySquare(r, dTable, tmp, genStream);
            std::swap(r, tmp);
            if ((J >> b) & 1ull) {
                mtPolyMulXKernel<<<1, MT_BLOCK_THREADS, 0, genStream>>>(r, dP);
                CUDA_CHECK(cudaGetLastError());
            }
        }
        if (r != dJump(0)) {
            CUDA_CHECK(cudaMemcpyAsync(dJump(0), r, MT_POLY_WORDS * sizeof(uint32_t),
                                       cudaMemcpyDeviceToDevice, genStream));
        }
        for (int b = 1; b < jumpBits; ++b) {
            mtPolySquare(dJump(b - 1), dTable, dJump(b), genStream);
        }
        for (int b = 0; b < jumpBits; ++b) {
            mtJumpKernel<<<mainLayout.segments, MT_BLOCK_THREADS, 0, genStream>>>(dStates, dJump(b), b);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // 5. Main batches
    const unsigned long long numBatches =
        (mainLayout.segPairs + mainLayout.slicePairs - 1) / mainLayout.slicePairs;
    for (unsigned long long b = 0; b < numBatches; ++b) {
        mainLayout.batch = b;
        enqueueBatch(mainLayout);
    }
    if (progressRows < N) printf("  Progress: %d/%d triangles\n", N, N);

    CUDA_CHECK(cudaStreamSynchronize(kijStream));
    CUDA_CHECK(cudaStreamSynchronize(genStream));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int s = 0; s < NBUF; ++s) {
        cudaFree(dRands[s]);
        cudaFree(dPairI[s]);
        cudaFree(dPairJ[s]);
        cudaEventDestroy(genDone[s]);
        cudaEventDestroy(kijDone[s]);
    }
    cudaStreamDestroy(genStream);
    cudaStreamDestroy(kijStream);
    cudaFree(dStates);
    cudaFree(dTable);
    cudaFree(dPolys);
    cudaFree(dRowCounts);
    cudaFree(dRowStart);
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    const int N = static_cast<int>(state.numTriangles);
    const size_t total = static_cast<size_t>(N) * N;
    const unsigned blocks = static_cast<unsigned>((total + 255) / 256);
    timeDelayKernel<<<blocks, 256>>>(N, state.dTriA, state.dTriB, state.dTriC, state.dTau,
                                     state.dTauOverflow);
    CUDA_CHECK(cudaGetLastError());
    int overflow = 0;
    CUDA_CHECK(cudaMemcpy(&overflow, state.dTauOverflow, sizeof(int), cudaMemcpyDeviceToHost));
    if (overflow) {
        fprintf(stderr, "Time delay exceeds the 8-bit device storage\n");
        exit(1);
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);
    const int threads = 256;                      // 8 warps = 8 receiving triangles per block
    const int blocks = (N + 7) / 8;

    for (int t = 0; t < T; ++t) {
        simulationStepKernel<<<blocks, threads>>>(N, t, state.dKij, state.dTau, state.dAreas,
                                                  state.dRho, state.dRadE, state.dRadB);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            printf("  Timestep %d/%d\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);

    float* dCorr = nullptr;
    CUDA_CHECK(cudaMalloc(&dCorr, std::max<size_t>(1, static_cast<size_t>(T) * N) * sizeof(float)));

    if (T > 0) {
        const int threads = 128;
        dim3 grid((N + threads - 1) / threads, T);
        crossCorrelationKernel<<<grid, threads>>>(N, T, static_cast<int>(state.sourceIndex),
                                                  state.dRadB, dCorr);
        CUDA_CHECK(cudaGetLastError());
    }
    argmaxDistanceKernel<<<(N + 127) / 128, 128>>>(N, T, dCorr, state.dDistances);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.dDistances, N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dCorr));
}

// Download the form factor and radiosity matrices for host-side validation
void downloadForValidation(SimulationState& state) {
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    state.kij.resize(N * N);
    CUDA_CHECK(cudaMemcpy(state.kij.data(), state.dKij, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    state.radB.resize(T * N);
    if (T * N > 0) {
        CUDA_CHECK(cudaMemcpy(state.radB.data(), state.dRadB, T * N * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    printf("\nValidation:\n");

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < state.numTriangles; ++i) {
        val_t d = state.distances[i];
        if (d < 0) {
            allNonNegative = false;
            printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    int receivedEnergy = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, i)] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    int nonZeroKij = 0;
    for (size_t i = 0; i < state.numTriangles * state.numTriangles; ++i) {
        if (state.kij[i] > EPSILON) nonZeroKij++;
    }
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
           nonZeroKij, state.numTriangles * state.numTriangles,
           100.0f * nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));

    if (nonZeroKij == 0) {
        printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) {
        return false;
    }

    printf("  Validation: PASSED\n");
    return true;
}

// ============================================================================
// Hash for Verification
// ============================================================================

uint64_t computeHash(const SimulationState& state) {
    uint64_t hash = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&state.distances[i]);
        hash ^= (static_cast<uint64_t>(*ptr) + i) * 0x9e3779b97f4a7c15ULL;
    }
    return hash;
}

// ============================================================================
// Helper: Compute subdivision level from target triangle count
// ============================================================================

int getSubdivisionsForTriangleCount(int targetTriangles) {
    // Icosphere: 20 triangles initially, 4x per subdivision
    // subdivisions: 0->20, 1->80, 2->320, 3->1280, 4->5120, 5->20480
    int subdivisions = 0;
    int triangles = 20;
    while (triangles < targetTriangles && subdivisions < 6) {
        subdivisions++;
        triangles *= 4;
    }
    return subdivisions;
}

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Target number of triangles (default: 320)\n");
    printf("               Actual count will be rounded to nearest icosphere level:\n");
    printf("               20, 80, 320, 1280, 5120, 20480\n");
    printf("  -t <num>     Number of timesteps (default: 50)\n");
    printf("  -s <num>     Source triangle index (default: 0)\n");
    printf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    printf("  -v           Enable validation\n");
    printf("  -o           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            targetTriangles = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            timesteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sourceIdx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            reflectivity = static_cast<val_t>(atof(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-o") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    printf("Room Response Simulation Benchmark\n");
    printf("===================================\n");
    printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    printf("Timesteps: %d\n", timesteps);
    printf("Source triangle: %d\n", sourceIdx);
    printf("Reflectivity: %.2f\n", reflectivity);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");

    // Initialize the CUDA context up front so it is not attributed to a phase
    CUDA_CHECK(cudaFree(0));

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    printf("Distance computation time: %ld ms\n", distDuration);
    printf("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    printf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    printf("\nPerformance:\n");
    printf("  Triangles: %zu\n", n);
    printf("  Timesteps: %zu\n", t);
    printf("  Form factor computations: %.2e\n", kijOps);
    printf("  Simulation operations: %.2e\n", simOps);
    printf("  Distance computations: %.2e\n", distOps);
    printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

    // Memory usage
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");

    // Print results for external validation
    if (printResults) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    int rc = 0;
    if (validate) {
        downloadForValidation(state);
        if (!validateResults(state)) {
            rc = 1;
        }
    }

    releaseDeviceMemory(state);
    return rc;
}
