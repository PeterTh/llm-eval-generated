/**
 * Room Response Simulation Benchmark  --  hybrid MPI + OpenMP + CUDA version
 *
 * This implements room impulse response simulation using radiosity-based wave
 * propagation. It models how sound/light waves propagate between surfaces in a
 * room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization
 * ---------------
 *  * MPI     distributes the receiver triangles (matrix rows) over the ranks in
 *            block-cyclic row groups, one group set per GPU.  Ranks exchange
 *            tauMin radiosity rows per collective, plus one final gather.
 *  * OpenMP  drives all host side pre-/post-processing (mesh analysis, octree
 *            construction, pair list compaction, packing/unpacking) and, when a
 *            rank owns more than one GPU, one OpenMP thread drives each GPU.
 *  * CUDA    executes every O(N^2) / O(N^2 T) / O(N T^2) kernel: form factors
 *            (ray traced through a flattened octree), time delays, the wave
 *            propagation update, the random stream regeneration and the cross
 *            correlation.
 *
 * Bit-level reproducibility of the original sequential code is preserved by
 *  (a) reproducing the exact std::mt19937 / std::uniform_real_distribution<float>
 *      stream: every non-culled triangle pair consumes exactly 64 draws, so the
 *      stream position of every pair is known in advance.  A dedicated host
 *      thread per GPU records Mersenne-Twister checkpoint states, from which the
 *      GPU regenerates the stream fully in parallel; block-cyclic work
 *      assignment keeps every GPU busy while the (inherently sequential) walk
 *      progresses.
 *  (b) keeping every floating point reduction in its original sequential order:
 *      one accumulator per receiver triangle, folded in increasing emitter
 *      order, with skipped emitters contributing an exact +0.0f.
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

#define HD __host__ __device__

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;
// std::numeric_limits<val_t>::max(), usable from device code
constexpr val_t VAL_MAX = FLT_MAX;

// Every non-culled (i, j) pair consumes exactly this many random draws:
// NUM_RAYS * 2 points * 2 barycentric coordinates.
constexpr int DRAWS_PER_PAIR = NUM_RAYS * 4;

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t _e = (call);                                                               \
        if (_e != cudaSuccess) {                                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__,      \
                    __LINE__);                                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                      \
        }                                                                                      \
    } while (0)

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    HD constexpr Vec3() : x(0), y(0), z(0) {}
    HD constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    HD explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    HD Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    HD Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    HD Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    HD Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    HD Vec3 operator-() const { return {-x, -y, -z}; }

    HD val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    HD Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    HD val_t squaredNorm() const { return x * x + y * y + z * z; }
    HD val_t norm() const { return sqrtf(squaredNorm()); }
    HD Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    HD bool operator==(const Vec3& o) const {
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON && fabsf(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    HD Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    HD Vec3 center() const { return (a + b + c) / 3.0f; }
    HD Vec3 normal() const { return _normal; }

    HD val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    HD bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
};

// ============================================================================
// Triangle-Box Overlap Test (for octree construction)
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
// Octree for Spatial Acceleration
// ============================================================================

constexpr size_t MAX_OCTREE_TRIS = 8;
constexpr val_t MAX_OCTREE_LEAF_SIZE = 0.5f;

class Octree {
public:
    Vec3 minBound, maxBound;
    Vec3 halfExtent, center;
    std::unique_ptr<Octree> children[8];
    std::vector<size_t> triangleIndices;  // Indices into the global triangle list
    const std::vector<Triangle>* allTriangles;  // Pointer to all triangles

    Octree() : allTriangles(nullptr) {}

    void build(const std::vector<Triangle>& triangles) {
        allTriangles = &triangles;
        if (triangles.empty()) return;

        // Compute bounding box (OpenMP reduction over all vertices)
        val_t lox = triangles[0].a.x, loy = triangles[0].a.y, loz = triangles[0].a.z;
        val_t hix = lox, hiy = loy, hiz = loz;
        const size_t nt = triangles.size();
#pragma omp parallel for schedule(static) reduction(min : lox, loy, loz)                          \
    reduction(max : hix, hiy, hiz)
        for (size_t t = 0; t < nt; ++t) {
            const Triangle& tri = triangles[t];
            for (const auto* v : {&tri.a, &tri.b, &tri.c}) {
                lox = std::min(lox, v->x);
                loy = std::min(loy, v->y);
                loz = std::min(loz, v->z);
                hix = std::max(hix, v->x);
                hiy = std::max(hiy, v->y);
                hiz = std::max(hiz, v->z);
            }
        }
        minBound = Vec3(lox, loy, loz);
        maxBound = Vec3(hix, hiy, hiz);

        // Collect all indices
        std::vector<size_t> allIndices(triangles.size());
        for (size_t i = 0; i < triangles.size(); ++i) allIndices[i] = i;

        Vec3 rootMin = minBound, rootMax = maxBound;
#pragma omp parallel
#pragma omp single
        buildNode(allIndices, rootMin, rootMax);
    }

private:
    void buildNode(const std::vector<size_t>& indices, const Vec3& nodeMin, const Vec3& nodeMax) {
        minBound = nodeMin;
        maxBound = nodeMax;
        halfExtent = (maxBound - minBound) * 0.5f;
        center = (minBound + maxBound) * 0.5f;

        // If few enough triangles or too small, make this a leaf
        if (indices.size() <= MAX_OCTREE_TRIS ||
            (maxBound - minBound).norm() < MAX_OCTREE_LEAF_SIZE) {
            triangleIndices = indices;
            return;
        }

        // Subdivide into 8 children
        Vec3 childHalfSize = halfExtent * 0.5f;
        std::vector<size_t> childIndices[8];

        for (size_t idx : indices) {
            const Triangle& tri = (*allTriangles)[idx];

            // Check which children this triangle overlaps
            for (int i = 0; i < 8; ++i) {
                Vec3 childCenter = center;
                childCenter.x += (i & 1) ? childHalfSize.x : -childHalfSize.x;
                childCenter.y += (i & 2) ? childHalfSize.y : -childHalfSize.y;
                childCenter.z += (i & 4) ? childHalfSize.z : -childHalfSize.z;

                if (triangleBoxOverlap(childCenter, childHalfSize, tri)) {
                    childIndices[i].push_back(idx);
                }
            }
        }

        // Check if we can't split further (all triangles in one child)
        bool canSplit = false;
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty() && childIndices[i].size() < indices.size()) {
                canSplit = true;
                break;
            }
        }

        if (!canSplit) {
            triangleIndices = indices;
            return;
        }

        // Build children (independent subtrees -> OpenMP tasks)
        const bool spawn = indices.size() > 512;
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty()) {
                Vec3 childMin = center;
                Vec3 childMax = center;
                childMin.x = (i & 1) ? center.x : minBound.x;
                childMax.x = (i & 1) ? maxBound.x : center.x;
                childMin.y = (i & 2) ? center.y : minBound.y;
                childMax.y = (i & 2) ? maxBound.y : center.y;
                childMin.z = (i & 4) ? center.z : minBound.z;
                childMax.z = (i & 4) ? maxBound.z : center.z;

                children[i] = std::make_unique<Octree>();
                children[i]->allTriangles = allTriangles;
                Octree* child = children[i].get();
                if (spawn) {
#pragma omp task firstprivate(child, i, childMin, childMax) shared(childIndices)
                    child->buildNode(childIndices[i], childMin, childMax);
                } else {
                    child->buildNode(childIndices[i], childMin, childMax);
                }
            }
        }
        if (spawn) {
#pragma omp taskwait
        }
    }
};

// ============================================================================
// Flattened (GPU resident) octree
// ============================================================================

struct GNode {
    Vec3 center;
    Vec3 halfExtent;
    int child[8];
    int triStart;
    int triCount;
};

static int flattenOctree(const Octree* node, std::vector<GNode>& nodes, std::vector<int>& triRefs) {
    const int id = static_cast<int>(nodes.size());
    nodes.emplace_back();

    GNode g;
    g.center = node->center;
    g.halfExtent = node->halfExtent;
    for (int i = 0; i < 8; ++i) g.child[i] = -1;
    g.triStart = 0;
    g.triCount = 0;

    if (!node->triangleIndices.empty()) {
        // Leaf: applyToTris tests the triangle list and never descends.
        g.triStart = static_cast<int>(triRefs.size());
        g.triCount = static_cast<int>(node->triangleIndices.size());
        for (size_t idx : node->triangleIndices) triRefs.push_back(static_cast<int>(idx));
        nodes[id] = g;
        return id;
    }

    for (int i = 0; i < 8; ++i) {
        if (node->children[i]) {
            const int cid = flattenOctree(node->children[i].get(), nodes, triRefs);
            g.child[i] = cid;
        }
    }
    nodes[id] = g;
    return id;
}

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
        triangles.resize(faces.size());
#pragma omp parallel for schedule(static)
        for (size_t f = 0; f < faces.size(); ++f) {
            const auto& face = faces[f];
            // Reverse winding to make normals point inward
            triangles[f] = Triangle(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Mersenne Twister (bit-exact std::mt19937 replacement)
// ============================================================================
//
// std::uniform_real_distribution<float>(0,1) on libstdc++ evaluates
// generate_canonical<float,24>, which consumes exactly one 32 bit mt19937 word
// and returns float(word) * 2^-32 (clamped below 1).  Reproducing the raw word
// stream therefore reproduces every random value bit-for-bit.

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr uint32_t MT_MATRIX_A = 0x9908b0dfu;
constexpr uint32_t MT_UPPER = 0x80000000u;
constexpr uint32_t MT_LOWER = 0x7fffffffu;

// Number of twists (blocks of MT_N words) covered by one checkpoint chunk.
constexpr int TWISTS_PER_CHUNK = 128;
constexpr long long CHUNK_DRAWS = static_cast<long long>(MT_N) * TWISTS_PER_CHUNK;

static void mtSeed(uint32_t* mt, uint32_t seed) {
    mt[0] = seed;
    for (int i = 1; i < MT_N; ++i) {
        mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
    }
}

// Advance the state by MT_N words (auto-vectorized: every loop below has a
// dependence distance of at least 227 elements).
static inline void mtTwist(uint32_t* __restrict__ mt) {
    uint32_t y;
    for (int k = 0; k < MT_N - MT_M; ++k) {
        y = (mt[k] & MT_UPPER) | (mt[k + 1] & MT_LOWER);
        mt[k] = mt[k + MT_M] ^ (y >> 1) ^ (static_cast<uint32_t>(-static_cast<int32_t>(y & 1u)) & MT_MATRIX_A);
    }
    for (int k = MT_N - MT_M; k < MT_N - 1; ++k) {
        y = (mt[k] & MT_UPPER) | (mt[k + 1] & MT_LOWER);
        mt[k] = mt[k + (MT_M - MT_N)] ^ (y >> 1) ^ (static_cast<uint32_t>(-static_cast<int32_t>(y & 1u)) & MT_MATRIX_A);
    }
    y = (mt[MT_N - 1] & MT_UPPER) | (mt[0] & MT_LOWER);
    mt[MT_N - 1] = mt[MT_M - 1] ^ (y >> 1) ^ (static_cast<uint32_t>(-static_cast<int32_t>(y & 1u)) & MT_MATRIX_A);
}

__device__ __forceinline__ uint32_t mtTemper(uint32_t y) {
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

// float(word) * 2^-32, matching libstdc++'s generate_canonical<float, 24>.
__device__ __forceinline__ val_t mtCanonical(uint32_t w) {
    val_t r = static_cast<val_t>(w) * 0x1p-32f;
    if (r >= 1.0f) r = 0x1.fffffep-1f;  // nextafterf(1, 0)
    return r;
}

// Regenerate the raw (tempered) word stream: one block per checkpoint chunk.
__global__ __launch_bounds__(256) void mtGenerateKernel(const uint32_t* __restrict__ chkStates,
                                                        uint32_t* __restrict__ out,
                                                        long long totalWords) {
    __shared__ uint32_t mt[MT_N];
    const int tid = threadIdx.x;
    const long long base = static_cast<long long>(blockIdx.x) * CHUNK_DRAWS;

    for (int k = tid; k < MT_N; k += blockDim.x) {
        mt[k] = chkStates[static_cast<size_t>(blockIdx.x) * MT_N + k];
    }
    __syncthreads();

    for (int rep = 0; rep < TWISTS_PER_CHUNK; ++rep) {
        // Phase A: k in [0, 227)   -- all inputs are old state
        uint32_t nv = 0;
        if (tid < MT_N - MT_M) {
            const int k = tid;
            const uint32_t y = (mt[k] & MT_UPPER) | (mt[k + 1] & MT_LOWER);
            nv = mt[k + MT_M] ^ (y >> 1) ^ ((y & 1u) ? MT_MATRIX_A : 0u);
        }
        __syncthreads();
        if (tid < MT_N - MT_M) mt[tid] = nv;
        __syncthreads();

        // Phase B: k in [227, 454) -- mt[k-227] already updated by phase A
        if (tid < MT_N - MT_M) {
            const int k = (MT_N - MT_M) + tid;
            const uint32_t y = (mt[k] & MT_UPPER) | (mt[k + 1] & MT_LOWER);
            nv = mt[k - (MT_N - MT_M)] ^ (y >> 1) ^ ((y & 1u) ? MT_MATRIX_A : 0u);
        }
        __syncthreads();
        if (tid < MT_N - MT_M) mt[(MT_N - MT_M) + tid] = nv;
        __syncthreads();

        // Phase C: k in [454, 623) -- mt[k-227] already updated by phase B
        if (tid < 169) {
            const int k = 454 + tid;
            const uint32_t y = (mt[k] & MT_UPPER) | (mt[k + 1] & MT_LOWER);
            nv = mt[k - (MT_N - MT_M)] ^ (y >> 1) ^ ((y & 1u) ? MT_MATRIX_A : 0u);
        }
        __syncthreads();
        if (tid < 169) mt[454 + tid] = nv;
        __syncthreads();

        // Phase D: wrap-around element
        if (tid == 0) {
            const uint32_t y = (mt[MT_N - 1] & MT_UPPER) | (mt[0] & MT_LOWER);
            mt[MT_N - 1] = mt[MT_M - 1] ^ (y >> 1) ^ ((y & 1u) ? MT_MATRIX_A : 0u);
        }
        __syncthreads();

        const long long obase = base + static_cast<long long>(rep) * MT_N;
        for (int k = tid; k < MT_N; k += blockDim.x) {
            const long long o = obase + k;
            if (o < totalWords) out[o] = mtTemper(mt[k]);
        }
        __syncthreads();
    }
}

// ============================================================================
// Device geometry kernels
// ============================================================================

struct GTri {
    Vec3 a, b, c, n;
};

__device__ __forceinline__ Vec3 randomPointInTriangle(const GTri& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    const Vec3 ab = t.b - t.a;
    const Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// Ray-Triangle Intersection (Moeller-Trumbore algorithm)
__device__ __forceinline__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                                      const Vec3& v0, const Vec3& v1,
                                                      const Vec3& v2) {
    const Vec3 e1 = v1 - v0;
    const Vec3 e2 = v2 - v0;
    const Vec3 pvec = dir.cross(e2);
    const val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return VAL_MAX;

    const val_t invDet = 1.0f / det;
    const Vec3 tvec = orig - v0;
    const val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return VAL_MAX;

    const Vec3 qvec = tvec.cross(e1);
    const val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return VAL_MAX;

    return e2.dot(qvec) * invDet;
}

__device__ __forceinline__ bool rayIntersectsBox(const Vec3& p1, const Vec3& p2, const GNode& n) {
    const Vec3 d = (p2 - p1) * 0.5f;
    const Vec3 c = p1 + d - n.center;
    const Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > n.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > n.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > n.halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > n.halfExtent.y * ad.z + n.halfExtent.z * ad.y + EPSILON)
        return false;
    if (fabsf(d.z * c.x - d.x * c.z) > n.halfExtent.z * ad.x + n.halfExtent.x * ad.z + EPSILON)
        return false;
    if (fabsf(d.x * c.y - d.y * c.x) > n.halfExtent.x * ad.y + n.halfExtent.y * ad.x + EPSILON)
        return false;

    return true;
}

constexpr int TRAVERSAL_STACK = 64;

// Check if the segment from -> to is blocked by any triangle other than src/dst
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to, const GNode* __restrict__ nodes,
                             const int* __restrict__ triRefs, const GTri* __restrict__ tris,
                             int srcTriIdx, int dstTriIdx) {
    const Vec3 dir = to - from;
    const val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    const Vec3 dirNorm = dir / rayLen;

    int stack[TRAVERSAL_STACK];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const GNode& node = nodes[stack[--sp]];
        if (node.triCount > 0) {
            for (int k = 0; k < node.triCount; ++k) {
                const int idx = triRefs[node.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const GTri& tri = tris[idx];
                const val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int i = 0; i < 8; ++i) {
                const int c = node.child[i];
                if (c >= 0 && rayIntersectsBox(from, to, nodes[c])) {
                    if (sp < TRAVERSAL_STACK) stack[sp++] = c;
                }
            }
        }
    }
    return false;
}

__device__ __forceinline__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    const val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    const val_t c = v.dot(normal) / vNorm;
    return c > ZERO ? c : ZERO;
}

// One thread per (non-culled) triangle pair.
__global__ __launch_bounds__(128) void kijKernel(const uint32_t* __restrict__ pairs,
                                                 long long pairBase, long long numPairs,
                                                 long long firstGlobalPair, long long rndBase,
                                                 const uint32_t* __restrict__ rnd,
                                                 const GTri* __restrict__ tris,
                                                 const GNode* __restrict__ nodes,
                                                 const int* __restrict__ triRefs, int N,
                                                 const int* __restrict__ rowLocal,
                                                 val_t* __restrict__ kijR) {
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= numPairs) return;

    const uint32_t code = pairs[pairBase + t];
    const int i = static_cast<int>(code / static_cast<uint32_t>(N));
    const int j = static_cast<int>(code % static_cast<uint32_t>(N));

    const GTri triI = tris[i];
    const GTri triJ = tris[j];

    long long off = (firstGlobalPair + t) * DRAWS_PER_PAIR - rndBase;

    val_t kij = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        const val_t u0 = mtCanonical(rnd[off + 0]);
        const val_t v0 = mtCanonical(rnd[off + 1]);
        const val_t u1 = mtCanonical(rnd[off + 2]);
        const val_t v1 = mtCanonical(rnd[off + 3]);
        off += 4;

        const Vec3 pI = randomPointInTriangle(triI, u0, v0);
        const Vec3 pJ = randomPointInTriangle(triJ, u1, v1);

        if (isRayBlocked(pI, pJ, nodes, triRefs, tris, i, j)) continue;

        const Vec3 v = pJ - pI;
        const val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        const val_t cosPhiI = cosPhi(v, triI.n);
        const val_t cosPhiJ = cosPhi(-v, triJ.n);

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kijR[static_cast<size_t>(rowLocal[i]) * N + j] = kij * INV_NUM_RAYS;
}

// Time delays, row-major over the locally owned receiver triangles.
__global__ void tauKernel(const GTri* __restrict__ tris, int N, const int* __restrict__ ownedRows,
                          int nRows, int* __restrict__ tauR) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int li = blockIdx.y * blockDim.y + threadIdx.y;
    if (li >= nRows || j >= N) return;
    const int i = ownedRows[li];
    int tau = 0;
    if (i != j) {
        const GTri& a = tris[i];
        const GTri& b = tris[j];
        const Vec3 ca = (a.a + a.b + a.c) / 3.0f;
        const Vec3 cb = (b.a + b.b + b.c) / 3.0f;
        const val_t dist = (ca - cb).norm();
        tau = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
    }
    tauR[static_cast<size_t>(li) * N + j] = tau;
}

// Every radiosity update reads history that is at least tauMin timesteps old, so
// tauMin consecutive timesteps are independent and can be produced by a single
// kernel launch - which also amortizes the Kij/Tau matrix traffic over them.
constexpr int MAX_TBATCH = 8;

// One warp per receiver triangle: the 32 lanes evaluate 32 emitter contributions
// in parallel (fully coalesced), but they are folded into the accumulator strictly
// in increasing j order, so the floating point result is identical to the original
// sequential loop (skipped emitters contribute an exact +0.0f).
__global__ __launch_bounds__(128) void simKernel(int tStart, int tCount,
                                                 const val_t* __restrict__ kijR,
                                                 const int* __restrict__ tauR,
                                                 const val_t* __restrict__ areas,
                                                 const val_t* __restrict__ rho,
                                                 val_t* __restrict__ radB, int N,
                                                 const int* __restrict__ ownedRows, int nRows,
                                                 int sourceIndex, int timeOff,
                                                 val_t* __restrict__ packOut) {
    const int li = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (li >= nRows) return;
    const int i = ownedRows[li];

    const val_t* __restrict__ kijRow = kijR + static_cast<size_t>(li) * N;
    const int* __restrict__ tauRow = tauR + static_cast<size_t>(li) * N;

    val_t sumB[MAX_TBATCH];
#pragma unroll
    for (int k = 0; k < MAX_TBATCH; ++k) sumB[k] = ZERO;

    for (int base = 0; base < N; base += 32) {
        const int j = base + lane;
        bool active = (j < N && j != i);
        int tauij = 0;
        val_t w = ZERO;
        if (active) {
            tauij = tauRow[j];
            const val_t kij = kijRow[j];
            if (kij > ZERO) {
                w = fminf(kij * areas[j], ONE);
            } else {
                active = false;
            }
        }

#pragma unroll
        for (int k = 0; k < MAX_TBATCH; ++k) {
            if (k >= tCount) break;
            const int t = tStart + k;
            val_t term = ZERO;
            if (active && t >= tauij) {
                const val_t radJ = radB[static_cast<size_t>(t - tauij) * N + j];
                if (radJ > ZERO) term = w * radJ;
            }
#pragma unroll
            for (int q = 0; q < 32; ++q) sumB[k] += __shfl_sync(0xffffffffu, term, q);
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int k = 0; k < MAX_TBATCH; ++k) {
            if (k >= tCount) break;
            const int t = tStart + k;
            const val_t emission = (i == sourceIndex && t < timeOff) ? 1.0f : ZERO;
            const val_t v = rho[i] * sumB[k] + emission;
            radB[static_cast<size_t>(t) * N + i] = v;
            if (packOut) packOut[static_cast<size_t>(k) * nRows + li] = v;
        }
    }
}

// Smallest non-zero propagation delay over the locally owned rows
__global__ void minTauKernel(const int* __restrict__ tauR, size_t n, int* __restrict__ out) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    int local = INT_MAX;
    for (; i < n; i += stride) {
        const int v = tauR[i];
        if (v > 0 && v < local) local = v;
    }
    atomicMin(out, local);
}

// Cross correlation: one block per receiver, one thread per lag.
__global__ void corrKernel(const val_t* __restrict__ radB, val_t* __restrict__ corr, int N, int T,
                           const int* __restrict__ ownedRows, int sourceIndex) {
    extern __shared__ val_t sh[];
    val_t* sSrc = sh;
    val_t* sB = sh + T;

    const int li = blockIdx.x;
    const int i = ownedRows[li];

    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        sSrc[t] = radB[static_cast<size_t>(t) * N + sourceIndex];
        sB[t] = radB[static_cast<size_t>(t) * N + i];
    }
    __syncthreads();

    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        val_t sum = ZERO;
        for (int tt = t; tt < T; ++tt) sum += sSrc[tt - t] * sB[tt];
        corr[static_cast<size_t>(li) * T + t] = sum;
    }
}

// Fallback for very large T (shared memory would not fit).
__global__ void corrKernelGlobal(const val_t* __restrict__ radB, val_t* __restrict__ corr, int N,
                                 int T, const int* __restrict__ ownedRows, int sourceIndex) {
    const int li = blockIdx.x;
    const int i = ownedRows[li];
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        val_t sum = ZERO;
        for (int tt = t; tt < T; ++tt) {
            const val_t pB = radB[static_cast<size_t>(tt) * N + i];
            const val_t pS = radB[static_cast<size_t>(tt - t) * N + sourceIndex];
            sum += pS * pB;
        }
        corr[static_cast<size_t>(li) * T + t] = sum;
    }
}

__global__ void argmaxKernel(const val_t* __restrict__ corr, val_t* __restrict__ distances, int T,
                             int nRows) {
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= nRows) return;
    val_t maxCorr = ZERO;
    int bestT = 0;
    for (int t = 0; t < T; ++t) {
        const val_t s = corr[static_cast<size_t>(li) * T + t];
        if (s > maxCorr) {
            maxCorr = s;
            bestT = t;
        }
    }
    distances[li] = WAVE_SPEED * static_cast<val_t>(bestT);
}

__global__ void countPositiveKernel(const val_t* __restrict__ data, size_t n,
                                    unsigned long long* __restrict__ out) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned long long local = 0;
    for (; i < n; i += stride) {
        if (data[i] > EPSILON) ++local;
    }
    atomicAdd(out, local);
}

// ============================================================================
// Simulation state (host side)
// ============================================================================

struct SimulationState {
    size_t numTriangles = 0;
    size_t numTimesteps = 0;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix), rank 0 only
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex = 0;

    // Flattened octree + GPU friendly triangle array
    std::vector<GNode> nodes;
    std::vector<int> triRefs;
    std::vector<GTri> gtris;

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// One contiguous block of receiver triangles handed to a single GPU
struct RowGroup {
    int rowLo = 0, rowHi = 0;          // global receiver triangle range
    long long pairLo = 0, pairHi = 0;  // range in the globally compacted pair list
    long long chunkLo = 0, chunkHi = 0;  // covered mt19937 checkpoint chunks
    long long chkBase = 0;             // offset into DeviceCtx::chkStates (in chunks)
    long long listBase = 0;            // offset into DeviceCtx::pairsHost
    int localBase = 0;                 // offset into the GPU's local row numbering
};

// Per-GPU context
struct DeviceCtx {
    int dev = 0;
    std::vector<RowGroup> groups;
    std::vector<int> ownedRows;        // local row index -> global receiver triangle
    std::vector<uint32_t> pairsHost;   // compacted (i,j) pairs, group by group
    int nLocal = 0;

    GTri* d_tris = nullptr;
    GNode* d_nodes = nullptr;
    int* d_triRefs = nullptr;
    uint32_t* d_pairs = nullptr;
    uint32_t* d_rnd[2] = {nullptr, nullptr};
    uint32_t* d_chk[2] = {nullptr, nullptr};
    cudaEvent_t rndDone[2] = {nullptr, nullptr};
    val_t* d_kijR = nullptr;
    int* d_tauR = nullptr;
    val_t* d_areas = nullptr;
    val_t* d_rho = nullptr;
    val_t* d_radB = nullptr;
    val_t* d_corr = nullptr;
    val_t* d_dist = nullptr;
    val_t* d_pack = nullptr;
    int* d_ownedRows = nullptr;
    int* d_rowLocal = nullptr;
    cudaStream_t stream = nullptr;
    long long maxChunks = 0;

    // Mersenne-Twister checkpoints covering this GPU's slices of the draw stream
    std::vector<uint32_t> chkStates;

    int nRows() const { return nLocal; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity, bool verbose) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    if (verbose) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if (verbose) printf("Building octree...\n");
    state.octree.build(state.triangles);

    state.nodes.reserve(state.numTriangles);
    state.triRefs.reserve(state.numTriangles * 4);
    flattenOctree(&state.octree, state.nodes, state.triRefs);

    // Pack triangles for the device
    state.gtris.resize(state.numTriangles);
    state.areas.resize(state.numTriangles);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        const Triangle& t = state.triangles[i];
        state.gtris[i] = GTri{t.a, t.b, t.c, t._normal};
        state.areas[i] = t.area();
    }

    // Initialize reflectivity
    state.rho.assign(state.numTriangles, reflectivity);
    state.distances.assign(state.numTriangles, ZERO);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, unsigned long long nonZeroKij) {
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
    const size_t total = state.numTriangles * state.numTriangles;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n", nonZeroKij, total,
           100.0f * static_cast<val_t>(nonZeroKij) / static_cast<val_t>(total));

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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    (void)provided;  // all MPI calls are made from the master thread

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const bool isRoot = (rank == 0);

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
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (isRoot) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // ------------------------------------------------------------------
    // GPU assignment: split the node-local GPUs over the node-local ranks
    // ------------------------------------------------------------------
    MPI_Comm shmComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shmComm);
    int lrank = 0, lsize = 1;
    MPI_Comm_rank(shmComm, &lrank);
    MPI_Comm_size(shmComm, &lsize);

    // Host-side OpenMP team: a moderate number of threads is plenty for the
    // remaining host work and leaves cores free for the GPU driver threads and
    // the sequential random-stream walker.
    {
        const int maxT = omp_get_max_threads();
        omp_set_num_threads(std::max(1, std::min(maxT, 32)));
    }

    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        if (isRoot) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<int> myDevs;
    if (devCount >= lsize) {
        for (int d = lrank * devCount / lsize; d < (lrank + 1) * devCount / lsize; ++d)
            myDevs.push_back(d);
    } else {
        myDevs.push_back(lrank % devCount);
    }
    const int nLocalDev = static_cast<int>(myDevs.size());

    // Global part enumeration: part = one GPU worth of rows
    int partBase = 0;
    MPI_Exscan(&nLocalDev, &partBase, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) partBase = 0;
    int nParts = 0;
    MPI_Allreduce(&nLocalDev, &nParts, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // ------------------------------------------------------------------
    // Initialize (replicated on every rank; cheap compared to the kernels)
    // ------------------------------------------------------------------
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity, isRoot);

    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);
    const int srcIndex = static_cast<int>(state.sourceIndex);
    const int timeOff = T / 2;

    if (isRoot) printf("\n");

    // ------------------------------------------------------------------
    // Determine which pairs consume random numbers, and where in the
    // mt19937 stream each of them starts.
    // ------------------------------------------------------------------
    std::vector<long long> rowOff(N + 1, 0);
    {
        std::vector<long long> rowCnt(N, 0);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            const Vec3 nI = state.gtris[i].n;
            long long c = 0;
            for (int j = 0; j < N; ++j) {
                if (i == j) continue;
                const Vec3 nJ = state.gtris[j].n;
                if (nI.x * nJ.x + nI.y * nJ.y + nI.z * nJ.z > 0.99f) continue;
                ++c;
            }
            rowCnt[i] = c;
        }
        for (int i = 0; i < N; ++i) rowOff[i + 1] = rowOff[i] + rowCnt[i];
    }
    const long long totalPairs = rowOff[N];

    // Split the receiver triangles into contiguous row groups of equal ray-traced
    // pair count and hand the groups to the GPUs round-robin.  Because the random
    // stream can only be advanced sequentially, every GPU must be given work near
    // the beginning of the stream; a block-cyclic mapping guarantees that while
    // still balancing the load.
    // Small groups improve the pipelining but must stay large enough to fill a GPU.
    constexpr long long MIN_PAIRS_PER_GROUP = 128 * 1024;
    int groupsPerPart = static_cast<int>(totalPairs / (nParts * MIN_PAIRS_PER_GROUP));
    groupsPerPart = std::max(1, std::min(groupsPerPart, 16));
    int nGroups = nParts * groupsPerPart;
    if (nGroups > N) nGroups = std::max(nParts, N);
    std::vector<int> groupRow(nGroups + 1, N);
    groupRow[0] = 0;
    for (int g = 1; g < nGroups; ++g) {
        const long long target = totalPairs * g / nGroups;
        const auto it = std::lower_bound(rowOff.begin(), rowOff.end(), target);
        int r = static_cast<int>(it - rowOff.begin());
        if (r < groupRow[g - 1]) r = groupRow[g - 1];
        if (r > N) r = N;
        groupRow[g] = r;
    }
    groupRow[nGroups] = N;

    // Row counts of every rank and the global gather order (needed for the
    // per-timestep radiosity exchange and the final distance gather)
    std::vector<int> rankDevCount(nranks, 1);
    MPI_Allgather(&nLocalDev, 1, MPI_INT, rankDevCount.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> rankRowCounts(nranks, 0), rankRowDispls(nranks, 0);
    std::vector<int> gatherIndex;  // packed position -> global row
    gatherIndex.reserve(N);
    {
        int base = 0;
        for (int r = 0; r < nranks; ++r) {
            rankRowDispls[r] = static_cast<int>(gatherIndex.size());
            for (int d = 0; d < rankDevCount[r]; ++d) {
                for (int g = base + d; g < nGroups; g += nParts) {
                    for (int i = groupRow[g]; i < groupRow[g + 1]; ++i) gatherIndex.push_back(i);
                }
            }
            rankRowCounts[r] = static_cast<int>(gatherIndex.size()) - rankRowDispls[r];
            base += rankDevCount[r];
        }
    }
    const int rankRowCount = rankRowCounts[rank];
    const int rankRowDispl = rankRowDispls[rank];

    // ------------------------------------------------------------------
    // Per-GPU setup
    // ------------------------------------------------------------------
    std::vector<DeviceCtx> ctxs(nLocalDev);
    const long long maxRndWords = 96LL * 1024 * 1024;  // 384 MB staging buffer per GPU
    const long long maxChunks = std::max<long long>(1, maxRndWords / CHUNK_DRAWS);

    const int ompDev = std::max(1, nLocalDev);
#pragma omp parallel for num_threads(ompDev) schedule(static, 1)
    for (int d = 0; d < nLocalDev; ++d) {
        DeviceCtx& c = ctxs[d];
        c.dev = myDevs[d];

        // Collect the round-robin row groups belonging to this GPU
        long long chkChunks = 0, listPairs = 0;
        for (int g = partBase + d; g < nGroups; g += nParts) {
            if (groupRow[g + 1] <= groupRow[g]) continue;
            RowGroup rg;
            rg.rowLo = groupRow[g];
            rg.rowHi = groupRow[g + 1];
            rg.pairLo = rowOff[rg.rowLo];
            rg.pairHi = rowOff[rg.rowHi];
            rg.chunkLo = (rg.pairLo * DRAWS_PER_PAIR) / CHUNK_DRAWS;
            rg.chunkHi = (rg.pairHi * DRAWS_PER_PAIR + CHUNK_DRAWS - 1) / CHUNK_DRAWS;
            if (rg.pairHi <= rg.pairLo) rg.chunkHi = rg.chunkLo;
            rg.chkBase = chkChunks;
            rg.listBase = listPairs;
            rg.localBase = c.nLocal;
            chkChunks += rg.chunkHi - rg.chunkLo;
            listPairs += rg.pairHi - rg.pairLo;
            c.nLocal += rg.rowHi - rg.rowLo;
            for (int i = rg.rowLo; i < rg.rowHi; ++i) c.ownedRows.push_back(i);
            c.groups.push_back(rg);
        }
        c.chkStates.resize(static_cast<size_t>(chkChunks) * MT_N);
        c.pairsHost.resize(static_cast<size_t>(listPairs));
        const int nr = c.nLocal;

        CUDA_CHECK(cudaSetDevice(c.dev));
        CUDA_CHECK(cudaStreamCreate(&c.stream));
        CUDA_CHECK(cudaMalloc(&c.d_tris, sizeof(GTri) * N));
        CUDA_CHECK(cudaMemcpy(c.d_tris, state.gtris.data(), sizeof(GTri) * N,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&c.d_nodes, sizeof(GNode) * state.nodes.size()));
        CUDA_CHECK(cudaMemcpy(c.d_nodes, state.nodes.data(), sizeof(GNode) * state.nodes.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&c.d_triRefs, sizeof(int) * std::max<size_t>(1, state.triRefs.size())));
        if (!state.triRefs.empty()) {
            CUDA_CHECK(cudaMemcpy(c.d_triRefs, state.triRefs.data(),
                                  sizeof(int) * state.triRefs.size(), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMalloc(&c.d_areas, sizeof(val_t) * N));
        CUDA_CHECK(cudaMemcpy(c.d_areas, state.areas.data(), sizeof(val_t) * N,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&c.d_rho, sizeof(val_t) * N));
        CUDA_CHECK(cudaMemcpy(c.d_rho, state.rho.data(), sizeof(val_t) * N,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&c.d_radB, sizeof(val_t) * static_cast<size_t>(T) * N));
        CUDA_CHECK(cudaMemset(c.d_radB, 0, sizeof(val_t) * static_cast<size_t>(T) * N));

        if (nr > 0) {
            const size_t mat = static_cast<size_t>(N) * nr;
            CUDA_CHECK(cudaMalloc(&c.d_kijR, sizeof(val_t) * mat));
            CUDA_CHECK(cudaMemset(c.d_kijR, 0, sizeof(val_t) * mat));
            CUDA_CHECK(cudaMalloc(&c.d_tauR, sizeof(int) * mat));
            CUDA_CHECK(cudaMalloc(&c.d_corr, sizeof(val_t) * static_cast<size_t>(nr) * T));
            CUDA_CHECK(cudaMalloc(&c.d_dist, sizeof(val_t) * nr));

            CUDA_CHECK(cudaMalloc(&c.d_ownedRows, sizeof(int) * nr));
            CUDA_CHECK(cudaMemcpy(c.d_ownedRows, c.ownedRows.data(), sizeof(int) * nr,
                                  cudaMemcpyHostToDevice));
            std::vector<int> rowLocal(N, -1);
            for (int li = 0; li < nr; ++li) rowLocal[c.ownedRows[li]] = li;
            CUDA_CHECK(cudaMalloc(&c.d_rowLocal, sizeof(int) * N));
            CUDA_CHECK(cudaMemcpy(c.d_rowLocal, rowLocal.data(), sizeof(int) * N,
                                  cudaMemcpyHostToDevice));

            if (!c.pairsHost.empty()) {
                CUDA_CHECK(cudaMalloc(&c.d_pairs, sizeof(uint32_t) * c.pairsHost.size()));
                long long maxGroupChunks = 1;
                for (const RowGroup& g : c.groups)
                    maxGroupChunks = std::max(maxGroupChunks, g.chunkHi - g.chunkLo);
                c.maxChunks = std::max<long long>(1, std::min(maxChunks, maxGroupChunks));
                for (int b = 0; b < 2; ++b) {
                    CUDA_CHECK(cudaMalloc(&c.d_rnd[b], sizeof(uint32_t) * c.maxChunks * CHUNK_DRAWS));
                    CUDA_CHECK(cudaMalloc(&c.d_chk[b], sizeof(uint32_t) * c.maxChunks * MT_N));
                    CUDA_CHECK(cudaEventCreateWithFlags(&c.rndDone[b], cudaEventDisableTiming));
                    CUDA_CHECK(cudaEventRecord(c.rndDone[b], c.stream));
                }
            }
        }
    }

    // Warm up the OpenMP team so its creation cost is not attributed to a phase
    {
        int warm = 0;
#pragma omp parallel reduction(+ : warm)
        warm = warm + 1;
        if (warm < 0) printf(" ");  // keep the reduction alive
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ==================================================================
    // Precomputation: time delays + form factors
    // ==================================================================
    auto startPre = std::chrono::high_resolution_clock::now();

    // Advancing the Mersenne-Twister is inherently sequential, so every GPU gets
    // its own walker thread that fast-forwards to the GPU's slice of the draw
    // stream and then publishes checkpoint states as they are produced.  The
    // walkers run concurrently with each other and with the ray tracing kernels.
    std::vector<std::atomic<long long>> chkReady(std::max(1, nLocalDev));
    std::vector<std::thread> mtWalkers;
    for (int d = 0; d < nLocalDev; ++d) {
        chkReady[d].store(0, std::memory_order_relaxed);
        mtWalkers.emplace_back([&, d]() {
            DeviceCtx& c = ctxs[d];
            const int nGrp = static_cast<int>(c.groups.size());
            if (nGrp == 0) return;
            alignas(64) uint32_t mt[MT_N];
            mtSeed(mt, 42);
            int gi = 0;
            for (long long ch = 0; gi < nGrp; ++ch) {
                // A chunk can be shared by several consecutive groups
                for (int k = gi; k < nGrp && c.groups[k].chunkLo <= ch; ++k) {
                    const RowGroup& g = c.groups[k];
                    if (ch < g.chunkHi) {
                        memcpy(&c.chkStates[static_cast<size_t>(g.chkBase + (ch - g.chunkLo)) * MT_N],
                               mt, MT_N * sizeof(uint32_t));
                    }
                }
                chkReady[d].store(ch + 1, std::memory_order_release);
                while (gi < nGrp && c.groups[gi].chunkHi <= ch + 1) ++gi;
                if (gi >= nGrp) break;
                for (int rep = 0; rep < TWISTS_PER_CHUNK; ++rep) mtTwist(mt);
            }
            chkReady[d].store(std::numeric_limits<long long>::max(), std::memory_order_release);
        });
    }

    if (isRoot) printf("Computing time delays (Tau)...\n");

    // Compacted list of the pairs that are actually ray traced (self pairs and
    // back-to-back facing pairs consume neither rays nor random numbers).
    for (int d = 0; d < nLocalDev; ++d) {
        DeviceCtx& c = ctxs[d];
#pragma omp parallel for schedule(static)
        for (int li = 0; li < c.nLocal; ++li) {
            const int i = c.ownedRows[li];
            const Vec3 nI = state.gtris[i].n;
            // Locate the group this row belongs to for the write offset
            size_t w = 0;
            for (const RowGroup& g : c.groups) {
                if (i >= g.rowLo && i < g.rowHi) {
                    w = static_cast<size_t>(g.listBase + (rowOff[i] - g.pairLo));
                    break;
                }
            }
            for (int j = 0; j < N; ++j) {
                if (i == j) continue;
                const Vec3 nJ = state.gtris[j].n;
                if (nI.x * nJ.x + nI.y * nJ.y + nI.z * nJ.z > 0.99f) continue;
                c.pairsHost[w++] = static_cast<uint32_t>(i) * static_cast<uint32_t>(N) +
                                   static_cast<uint32_t>(j);
            }
        }
    }

    if (isRoot) printf("Computing form factors (Kij)...\n");
#pragma omp parallel for num_threads(ompDev) schedule(static, 1)
    for (int d = 0; d < nLocalDev; ++d) {
        DeviceCtx& c = ctxs[d];
        if (c.nLocal <= 0) continue;
        CUDA_CHECK(cudaSetDevice(c.dev));

        const dim3 blk(64, 4);
        const dim3 grd((N + blk.x - 1) / blk.x, (c.nLocal + blk.y - 1) / blk.y);
        tauKernel<<<grd, blk, 0, c.stream>>>(c.d_tris, N, c.d_ownedRows, c.nLocal, c.d_tauR);
        CUDA_CHECK(cudaGetLastError());

        if (c.pairsHost.empty()) continue;
        CUDA_CHECK(cudaMemcpyAsync(c.d_pairs, c.pairsHost.data(),
                                   sizeof(uint32_t) * c.pairsHost.size(), cudaMemcpyHostToDevice,
                                   c.stream));

        int buf = 0;
        for (const RowGroup& g : c.groups) {
            long long p = g.pairLo;
            while (p < g.pairHi) {
                const long long firstChunk = (p * DRAWS_PER_PAIR) / CHUNK_DRAWS;
                // How many pairs fit into maxChunks chunks starting at firstChunk?
                const long long lastDraw = (firstChunk + c.maxChunks) * CHUNK_DRAWS;
                long long pEnd = std::min(g.pairHi, lastDraw / DRAWS_PER_PAIR);
                if (pEnd <= p) pEnd = p + 1;  // safety (cannot happen: chunk >> 64)
                const long long nch =
                    (pEnd * DRAWS_PER_PAIR + CHUNK_DRAWS - 1) / CHUNK_DRAWS - firstChunk;

                while (chkReady[d].load(std::memory_order_acquire) < firstChunk + nch) {
                    std::this_thread::yield();
                }

                // Wait until the ray tracing that still reads this staging buffer is done
                CUDA_CHECK(cudaEventSynchronize(c.rndDone[buf]));

                CUDA_CHECK(cudaMemcpyAsync(
                    c.d_chk[buf],
                    c.chkStates.data() + static_cast<size_t>(g.chkBase + (firstChunk - g.chunkLo)) * MT_N,
                    sizeof(uint32_t) * static_cast<size_t>(nch) * MT_N, cudaMemcpyHostToDevice,
                    c.stream));

                const long long rndBase = firstChunk * CHUNK_DRAWS;
                const long long words = pEnd * DRAWS_PER_PAIR - rndBase;
                mtGenerateKernel<<<static_cast<int>(nch), 256, 0, c.stream>>>(c.d_chk[buf],
                                                                             c.d_rnd[buf], words);
                CUDA_CHECK(cudaGetLastError());

                const long long npairs = pEnd - p;
                const int blkSize = 128;
                const long long grid = (npairs + blkSize - 1) / blkSize;
                kijKernel<<<static_cast<int>(grid), blkSize, 0, c.stream>>>(
                    c.d_pairs, g.listBase + (p - g.pairLo), npairs, p, rndBase, c.d_rnd[buf],
                    c.d_tris, c.d_nodes, c.d_triRefs, N, c.d_rowLocal, c.d_kijR);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(c.rndDone[buf], c.stream));
                buf ^= 1;

                p = pEnd;
            }
        }
    }

    for (auto& w : mtWalkers) w.join();

#pragma omp parallel for num_threads(ompDev) schedule(static, 1)
    for (int d = 0; d < nLocalDev; ++d) {
        CUDA_CHECK(cudaSetDevice(ctxs[d].dev));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (isRoot) {
        for (int i = 0; i < N; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == N) {
                printf("  Progress: %d/%d triangles\n", i + 1, N);
            }
        }
    }

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();
    long maxPreDuration = 0;
    MPI_Reduce(&preDuration, &maxPreDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (isRoot) {
        printf("Precomputation time: %ld ms\n", maxPreDuration);
        printf("\n");
    }

    // ==================================================================
    // Simulation Phase (Wave Propagation)
    // ==================================================================
    auto startSim = std::chrono::high_resolution_clock::now();

    if (isRoot) printf("Running wave propagation simulation...\n");

    const bool exchange = (nParts > 1);

    // Offset of every local GPU inside the rank's packed row buffer
    std::vector<int> localOffset(nLocalDev + 1, 0);
    for (int d = 0; d < nLocalDev; ++d) localOffset[d + 1] = localOffset[d] + ctxs[d].nLocal;

    // Smallest propagation delay: that many timesteps can be produced at once
    int tauMin = INT_MAX;
    for (int d = 0; d < nLocalDev; ++d) {
        DeviceCtx& c = ctxs[d];
        if (c.nLocal <= 0) continue;
        CUDA_CHECK(cudaSetDevice(c.dev));
        int* d_min = nullptr;
        CUDA_CHECK(cudaMalloc(&d_min, sizeof(int)));
        const int init = INT_MAX;
        CUDA_CHECK(cudaMemcpy(d_min, &init, sizeof(int), cudaMemcpyHostToDevice));
        minTauKernel<<<1024, 256>>>(c.d_tauR, static_cast<size_t>(N) * c.nLocal, d_min);
        CUDA_CHECK(cudaGetLastError());
        int h = INT_MAX;
        CUDA_CHECK(cudaMemcpy(&h, d_min, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_min));
        tauMin = std::min(tauMin, h);
    }
    if (nranks > 1) MPI_Allreduce(MPI_IN_PLACE, &tauMin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    const int tBatch = std::max(1, std::min(tauMin, MAX_TBATCH));

    auto launchSim = [&](DeviceCtx& c, int t, int k, val_t* packOut) {
        const int blk = 128;
        const int warpsPerBlk = blk / 32;
        const int grid = (c.nRows() + warpsPerBlk - 1) / warpsPerBlk;
        simKernel<<<grid, blk, 0, c.stream>>>(t, k, c.d_kijR, c.d_tauR, c.d_areas, c.d_rho,
                                              c.d_radB, N, c.d_ownedRows, c.nRows(), srcIndex,
                                              timeOff, packOut);
        CUDA_CHECK(cudaGetLastError());
    };
    auto printSteps = [&](int t, int k) {
        if (!isRoot) return;
        for (int s = t; s < t + k; ++s) {
            if ((s + 1) % 10 == 0 || s + 1 == T) printf("  Timestep %d/%d\n", s + 1, T);
        }
    };

    if (!exchange) {
        // Single partition: the whole radiosity row is produced on one GPU, so no
        // communication is needed at all - just enqueue every timestep back to back.
        DeviceCtx& c = ctxs[0];
        CUDA_CHECK(cudaSetDevice(c.dev));
        for (int t = 0; t < T; t += tBatch) {
            const int k = std::min(tBatch, T - t);
            launchSim(c, t, k, nullptr);
            printSteps(t, k);
        }
    } else {
        val_t* rowPacked = nullptr;
        val_t* rowFull = nullptr;
        CUDA_CHECK(cudaSetDevice(ctxs[0].dev));
        CUDA_CHECK(cudaHostAlloc(&rowPacked, sizeof(val_t) * tBatch * N, cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(&rowFull, sizeof(val_t) * tBatch * N, cudaHostAllocPortable));
        std::vector<int> allCounts(nranks), allDispls(nranks);
        for (int d = 0; d < nLocalDev; ++d) {
            if (ctxs[d].nLocal > 0) {
                CUDA_CHECK(cudaSetDevice(ctxs[d].dev));
                CUDA_CHECK(cudaMalloc(&ctxs[d].d_pack, sizeof(val_t) * tBatch * ctxs[d].nLocal));
            }
        }

#pragma omp parallel num_threads(ompDev)
        {
            const int d = omp_get_thread_num();
            DeviceCtx* c = (d < nLocalDev) ? &ctxs[d] : nullptr;
            if (c) CUDA_CHECK(cudaSetDevice(c->dev));

            for (int t = 0; t < T; t += tBatch) {
                const int nb = std::min(tBatch, T - t);
                if (c && c->nRows() > 0) {
                    launchSim(*c, t, nb, c->d_pack);
                    for (int k = 0; k < nb; ++k) {
                        CUDA_CHECK(cudaMemcpyAsync(
                            rowPacked + static_cast<size_t>(nb) * rankRowDispl + k * rankRowCount +
                                localOffset[d],
                            c->d_pack + static_cast<size_t>(k) * c->nRows(),
                            sizeof(val_t) * c->nRows(), cudaMemcpyDeviceToHost, c->stream));
                    }
                    CUDA_CHECK(cudaStreamSynchronize(c->stream));
                }
#pragma omp barrier
#pragma omp master
                {
                    if (nranks > 1) {
                        for (int r = 0; r < nranks; ++r) {
                            allCounts[r] = nb * rankRowCounts[r];
                            allDispls[r] = nb * rankRowDispls[r];
                        }
                        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, rowPacked, allCounts.data(),
                                       allDispls.data(), MPI_FLOAT, MPI_COMM_WORLD);
                    }
                    for (int r = 0; r < nranks; ++r) {
                        const int cnt = rankRowCounts[r];
                        const int base = nb * rankRowDispls[r];
                        for (int k = 0; k < nb; ++k) {
                            for (int li = 0; li < cnt; ++li) {
                                rowFull[static_cast<size_t>(k) * N +
                                        gatherIndex[rankRowDispls[r] + li]] =
                                    rowPacked[base + k * cnt + li];
                            }
                        }
                    }
                    printSteps(t, nb);
                }
#pragma omp barrier
                if (c) {
                    CUDA_CHECK(cudaMemcpyAsync(c->d_radB + static_cast<size_t>(t) * N, rowFull,
                                               sizeof(val_t) * nb * N, cudaMemcpyHostToDevice,
                                               c->stream));
                    CUDA_CHECK(cudaStreamSynchronize(c->stream));
                }
#pragma omp barrier
            }
        }
        CUDA_CHECK(cudaFreeHost(rowPacked));
        CUDA_CHECK(cudaFreeHost(rowFull));
    }

#pragma omp parallel for num_threads(ompDev) schedule(static, 1)
    for (int d = 0; d < nLocalDev; ++d) {
        CUDA_CHECK(cudaSetDevice(ctxs[d].dev));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    MPI_Barrier(MPI_COMM_WORLD);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();
    long maxSimDuration = 0;
    MPI_Reduce(&simDuration, &maxSimDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (isRoot) {
        printf("Simulation time: %ld ms\n", maxSimDuration);
        printf("\n");
    }

    // ==================================================================
    // Distance Computation (Cross-Correlation)
    // ==================================================================
    auto startDist = std::chrono::high_resolution_clock::now();

    if (isRoot) printf("Computing distances via cross-correlation...\n");

    std::vector<val_t> distPacked(N, ZERO);

#pragma omp parallel for num_threads(ompDev) schedule(static, 1)
    for (int d = 0; d < nLocalDev; ++d) {
        DeviceCtx& c = ctxs[d];
        if (c.nRows() <= 0) continue;
        CUDA_CHECK(cudaSetDevice(c.dev));
        const size_t shmem = sizeof(val_t) * 2 * static_cast<size_t>(T);
        const int threads = std::min(1024, std::max(32, ((T + 31) / 32) * 32));
        if (shmem <= 47 * 1024) {
            corrKernel<<<c.nRows(), threads, shmem, c.stream>>>(c.d_radB, c.d_corr, N, T,
                                                                c.d_ownedRows, srcIndex);
        } else {
            corrKernelGlobal<<<c.nRows(), threads, 0, c.stream>>>(c.d_radB, c.d_corr, N, T,
                                                                  c.d_ownedRows, srcIndex);
        }
        CUDA_CHECK(cudaGetLastError());
        const int blk = 128;
        argmaxKernel<<<(c.nRows() + blk - 1) / blk, blk, 0, c.stream>>>(c.d_corr, c.d_dist, T,
                                                                       c.nRows());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(distPacked.data() + rankRowDispl + localOffset[d], c.d_dist,
                                   sizeof(val_t) * c.nRows(), cudaMemcpyDeviceToHost, c.stream));
        CUDA_CHECK(cudaStreamSynchronize(c.stream));
    }

    if (nranks > 1) {
        if (isRoot) {
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_FLOAT, distPacked.data(), rankRowCounts.data(),
                        rankRowDispls.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(distPacked.data() + rankRowDispl, rankRowCount, MPI_FLOAT, nullptr, nullptr,
                        nullptr, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
    }
    if (isRoot) {
        for (int k = 0; k < N; ++k) state.distances[gatherIndex[k]] = distPacked[k];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();
    long maxDistDuration = 0;
    MPI_Reduce(&distDuration, &maxDistDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (isRoot) {
        printf("Distance computation time: %ld ms\n", maxDistDuration);
        printf("\n");
    }

    // Total time
    long totalTime = maxPreDuration + maxSimDuration + maxDistDuration;

    if (isRoot) {
        printf("Total computation time: %ld ms\n", totalTime);

        // Performance metrics
        size_t n = state.numTriangles;
        size_t tt = state.numTimesteps;
        double kijOps = static_cast<double>(n * n);
        double simOps = static_cast<double>(n * n * tt);
        double distOps = static_cast<double>(n * tt * tt);

        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", tt);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        // Memory usage
        size_t memKij = n * n * sizeof(val_t);
        size_t memTau = n * n * sizeof(int);
        size_t memRad = 2 * tt * n * sizeof(val_t);
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
    }

    int status = 0;
    if (validate) {
        // Non-zero form factors: reduce over all GPUs / ranks
        unsigned long long localCount = 0;
        for (int d = 0; d < nLocalDev; ++d) {
            DeviceCtx& c = ctxs[d];
            if (c.nRows() <= 0) continue;
            CUDA_CHECK(cudaSetDevice(c.dev));
            unsigned long long* d_cnt = nullptr;
            CUDA_CHECK(cudaMalloc(&d_cnt, sizeof(unsigned long long)));
            CUDA_CHECK(cudaMemset(d_cnt, 0, sizeof(unsigned long long)));
            const size_t mat = static_cast<size_t>(N) * c.nRows();
            countPositiveKernel<<<1024, 256>>>(c.d_kijR, mat, d_cnt);
            CUDA_CHECK(cudaGetLastError());
            unsigned long long h = 0;
            CUDA_CHECK(cudaMemcpy(&h, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_cnt));
            localCount += h;
        }
        unsigned long long globalCount = localCount;
        if (nranks > 1) {
            MPI_Allreduce(&localCount, &globalCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                          MPI_COMM_WORLD);
        }

        if (isRoot) {
            // Pull the complete radiosity history back for the energy check.
            state.radB.resize(static_cast<size_t>(T) * N);
            const DeviceCtx& c = ctxs[0];
            CUDA_CHECK(cudaSetDevice(c.dev));
            CUDA_CHECK(cudaMemcpy(state.radB.data(), c.d_radB,
                                  sizeof(val_t) * static_cast<size_t>(T) * N,
                                  cudaMemcpyDeviceToHost));
            if (!validateResults(state, globalCount)) status = 1;
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Cleanup
    for (int d = 0; d < nLocalDev; ++d) {
        DeviceCtx& c = ctxs[d];
        cudaSetDevice(c.dev);
        cudaFree(c.d_tris);
        cudaFree(c.d_nodes);
        cudaFree(c.d_triRefs);
        cudaFree(c.d_pairs);
        for (int b = 0; b < 2; ++b) {
            cudaFree(c.d_rnd[b]);
            cudaFree(c.d_chk[b]);
            if (c.rndDone[b]) cudaEventDestroy(c.rndDone[b]);
        }
        cudaFree(c.d_kijR);
        cudaFree(c.d_tauR);
        cudaFree(c.d_areas);
        cudaFree(c.d_rho);
        cudaFree(c.d_radB);
        cudaFree(c.d_corr);
        cudaFree(c.d_dist);
        cudaFree(c.d_pack);
        cudaFree(c.d_ownedRows);
        cudaFree(c.d_rowLocal);
        if (c.stream) cudaStreamDestroy(c.stream);
    }

    MPI_Comm_free(&shmComm);
    MPI_Finalize();
    return status;
}
