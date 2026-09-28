/**
 * Room Response Simulation Benchmark
 *
 * Radiosity-based room impulse response simulation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization strategy (hybrid MPI + OpenMP + CUDA):
 *  - MPI decomposes the triangle index space into contiguous row blocks, one
 *    block per rank. Each rank owns exactly one GPU (round-robin over the
 *    GPUs visible on its node) and computes only its rows of the Kij/Tau
 *    matrices and radiosity/distance fields.
 *  - CUDA kernels execute the three O(N^2) / O(N*T^2) hot loops (form factor
 *    Monte Carlo sampling, wave propagation timestep update, cross-correlation
 *    distance search) on the GPU, including a flattened (array based) octree
 *    for device-side visibility queries.
 *  - Because every triangle pair's radiosity can, in principle, influence any
 *    other triangle, the per-timestep radiosity row is exchanged across ranks
 *    with MPI_Allgatherv (bulk-synchronous parallel style) so every rank's
 *    device holds a fully consistent radiosity history for the next step.
 *  - OpenMP parallelizes the remaining host-side work: the Tau (time delay)
 *    precomputation and the reduction loops used for validation/hashing.
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
#include <memory>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t cudaCheckErr__ = (call);                                             \
        if (cudaCheckErr__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(cudaCheckErr__));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    __host__ __device__ explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }

    __host__ __device__ val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    __host__ __device__ val_t squaredNorm() const { return x * x + y * y + z * z; }
    __host__ __device__ val_t norm() const { return std::sqrt(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
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
    __host__ __device__ Vec3 normal() const { return _normal; }

    val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
};

// ============================================================================
// Triangle-Box Overlap Test (for octree construction, host-only, one-time cost)
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
// Octree for Spatial Acceleration (built redundantly, identically, on every
// rank -- construction is O(N log N) and cheap relative to the O(N^2) work
// below, so there is no benefit to distributing it). It is flattened into
// device-friendly arrays (see GpuOctreeNode) for GPU traversal.
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

        // Compute bounding box
        minBound = triangles[0].a;
        maxBound = triangles[0].a;
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

        // Collect all indices
        std::vector<size_t> allIndices(triangles.size());
        for (size_t i = 0; i < triangles.size(); ++i) allIndices[i] = i;

        buildNode(allIndices, minBound, maxBound);
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

        // Build children
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
                children[i]->buildNode(childIndices[i], childMin, childMax);
            }
        }
    }
};

// ============================================================================
// Flattened (array-based) octree for GPU traversal
// ============================================================================

struct GpuOctreeNode {
    Vec3 center, halfExtent;
    int children[8];   // index into node array, -1 if absent
    int isLeaf;        // 1 if leaf
    int triStart;      // offset into flattened triangle index array (leaf only)
    int triCount;      // number of triangles at this leaf
};

int flattenOctree(const Octree& node, std::vector<GpuOctreeNode>& nodes, std::vector<int>& triIdxFlat) {
    bool leaf = true;
    for (int c = 0; c < 8; ++c) {
        if (node.children[c]) { leaf = false; break; }
    }

    GpuOctreeNode g{};
    g.center = node.center;
    g.halfExtent = node.halfExtent;
    for (int c = 0; c < 8; ++c) g.children[c] = -1;
    g.isLeaf = leaf ? 1 : 0;
    if (leaf) {
        g.triStart = static_cast<int>(triIdxFlat.size());
        g.triCount = static_cast<int>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) triIdxFlat.push_back(static_cast<int>(idx));
    } else {
        g.triStart = 0;
        g.triCount = 0;
    }

    int myIndex = static_cast<int>(nodes.size());
    nodes.push_back(g);

    if (!leaf) {
        for (int c = 0; c < 8; ++c) {
            if (node.children[c]) {
                int childIndex = flattenOctree(*node.children[c], nodes, triIdxFlat);
                nodes[myIndex].children[c] = childIndex;
            }
        }
    }
    return myIndex;
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
        triangles.reserve(faces.size());
        for (const auto& face : faces) {
            // Reverse winding to make normals point inward
            triangles.emplace_back(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Deterministic, parallel-safe random number generation
//
// The original sequential implementation drew samples from a single shared
// mt19937 stream in (i, j, ray) order. That is inherently serial. To keep the
// Monte Carlo form-factor estimate parallel (across GPU threads, MPI ranks,
// and independent of scheduling order) each (i, j) triangle pair is given its
// own independent, reproducible pseudo-random stream derived from a fixed
// global seed and the pair indices (splitmix64 to seed a PCG32 generator).
// This preserves the algorithm's semantics (a NUM_RAYS-sample Monte Carlo
// estimate of the visibility-weighted form factor) while making every sample
// stream independent of execution/thread order.
// ============================================================================

__host__ __device__ inline uint64_t splitmix64(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__host__ __device__ inline uint32_t pcg32_next(uint64_t& state, uint64_t inc) {
    uint64_t oldstate = state;
    state = oldstate * 6364136223846793005ULL + inc;
    uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

__host__ __device__ inline float pcg32_nextFloat(uint64_t& state, uint64_t inc) {
    uint32_t r = pcg32_next(state, inc);
    return static_cast<float>(r >> 8) * (1.0f / 16777216.0f);
}

// Generate a random point inside a triangle using barycentric coordinates
__host__ __device__ inline Vec3 randomPointInTriangle(const Triangle& t, float u, float v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

__host__ __device__ inline val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                                        const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (std::abs(det) < EPSILON) return std::numeric_limits<val_t>::max();

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return std::numeric_limits<val_t>::max();

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return std::numeric_limits<val_t>::max();

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Device-side octree traversal / visibility test
// ============================================================================

__host__ __device__ inline bool rayIntersectsBoxDevice(const Vec3& center, const Vec3& halfExtent,
                                                          const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

__device__ inline bool isRayBlockedDevice(const Vec3& from, const Vec3& to, const Triangle* tris,
                                           const GpuOctreeNode* nodes, const int* triIdx,
                                           int srcIdx, int dstIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[128];
    int sp = 0;
    stack[sp++] = 0;  // root

    while (sp > 0) {
        int ni = stack[--sp];
        const GpuOctreeNode& node = nodes[ni];
        if (!rayIntersectsBoxDevice(node.center, node.halfExtent, from, to)) continue;

        if (node.isLeaf) {
            for (int k = 0; k < node.triCount; ++k) {
                int tidx = triIdx[node.triStart + k];
                if (tidx == srcIdx || tidx == dstIdx) continue;
                const Triangle& tri = tris[tidx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                int ci = node.children[c];
                if (ci >= 0) stack[sp++] = ci;
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij) helpers
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__host__ __device__ inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return v.dot(normal) / vNorm > ZERO ? v.dot(normal) / vNorm : ZERO;
}

// ============================================================================
// Tau (time delay) Computation -- cheap O(N^2) arithmetic, kept on host
// ============================================================================

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// GPU reduction helpers
// ============================================================================

__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

__device__ inline float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    int numWarps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < numWarps) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// One thread per (local row, column) pair. Computes the Monte Carlo estimate
// of the form factor Kij using an independent per-pair RNG stream.
__global__ void kernComputeKij(const Triangle* __restrict__ tris, int N, int rowStart, int rowCount,
                                const GpuOctreeNode* __restrict__ nodes, const int* __restrict__ triIdx,
                                float* __restrict__ kijLocal, uint64_t seed) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(rowCount) * static_cast<size_t>(N);
    if (idx >= total) return;

    int li = static_cast<int>(idx / N);
    int j = static_cast<int>(idx % N);
    int i = rowStart + li;

    if (i == j) { kijLocal[idx] = 0.0f; return; }

    const Triangle& triI = tris[i];
    const Triangle& triJ = tris[j];

    if (triI.normal().dot(triJ.normal()) > 0.99f) { kijLocal[idx] = 0.0f; return; }

    uint64_t mix = seed ^ (static_cast<uint64_t>(static_cast<uint32_t>(i)) << 32) ^
                   static_cast<uint64_t>(static_cast<uint32_t>(j));
    uint64_t rngState = splitmix64(mix);
    uint64_t rngInc = splitmix64(mix) | 1ULL;

    float kij = 0.0f;
    for (int r = 0; r < NUM_RAYS; ++r) {
        float u1 = pcg32_nextFloat(rngState, rngInc);
        float v1 = pcg32_nextFloat(rngState, rngInc);
        float u2 = pcg32_nextFloat(rngState, rngInc);
        float v2 = pcg32_nextFloat(rngState, rngInc);

        Vec3 pI = randomPointInTriangle(triI, u1, v1);
        Vec3 pJ = randomPointInTriangle(triJ, u2, v2);

        if (isRayBlockedDevice(pI, pJ, tris, nodes, triIdx, i, j)) continue;

        Vec3 v = pJ - pI;
        float distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        float cosPhiI = cosPhi(v, triI.normal());
        float cosPhiJ = cosPhi(-v, triJ.normal());
        if (cosPhiI <= 0.0f || cosPhiJ <= 0.0f) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kijLocal[idx] = kij * INV_NUM_RAYS;
}

__global__ void kernCountNonzero(const float* __restrict__ kij, size_t n, unsigned long long* counter) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n && kij[idx] > EPSILON) {
        atomicAdd(counter, 1ULL);
    }
}

// One block per local row. Computes the radiosity update for timestep t.
__global__ void kernComputeSimStep(int N, int rowStart, int rowCount, int t,
                                    const float* __restrict__ kijLocal, const int* __restrict__ tauLocal,
                                    const float* __restrict__ areas, const float* __restrict__ rho,
                                    const float* __restrict__ radEFull, const float* __restrict__ radBFull,
                                    float* __restrict__ radBRowLocal) {
    int li = blockIdx.x;
    if (li >= rowCount) return;
    int i = rowStart + li;

    float partial = 0.0f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        if (j == i) continue;
        int tauij = tauLocal[static_cast<size_t>(li) * N + j];
        if (t < tauij) continue;
        float kij = kijLocal[static_cast<size_t>(li) * N + j];
        if (kij <= 0.0f) continue;
        int srcTime = t - tauij;
        float radJ = radBFull[static_cast<size_t>(srcTime) * N + j];
        if (radJ <= 0.0f) continue;
        float contrib = kij * areas[j];
        if (contrib > 1.0f) contrib = 1.0f;
        partial += contrib * radJ;
    }

    float sumB = blockReduceSum(partial);
    if (threadIdx.x == 0) {
        radBRowLocal[li] = rho[i] * sumB + radEFull[static_cast<size_t>(t) * N + i];
    }
}

// One block per local row. Cross-correlates against the source triangle's
// radiosity history to find the peak-correlation time delay.
__global__ void kernComputeDistances(int N, int rowStart, int rowCount, int T, int sourceIndex,
                                      const float* __restrict__ radBFull, float* __restrict__ distLocal) {
    int li = blockIdx.x;
    if (li >= rowCount) return;
    int i = rowStart + li;

    __shared__ float sMaxCorr;
    __shared__ int sBestT;
    if (threadIdx.x == 0) {
        sMaxCorr = 0.0f;
        sBestT = 0;
    }
    __syncthreads();

    for (int t = 0; t < T; ++t) {
        float partial = 0.0f;
        for (int tt = t + threadIdx.x; tt < T; tt += blockDim.x) {
            float pB = radBFull[static_cast<size_t>(tt) * N + i];
            float pS = radBFull[static_cast<size_t>(tt - t) * N + sourceIndex];
            partial += pS * pB;
        }
        float sum = blockReduceSum(partial);
        if (threadIdx.x == 0 && sum > sMaxCorr) {
            sMaxCorr = sum;
            sBestT = t;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        distLocal[li] = WAVE_SPEED * static_cast<float>(sBestT);
    }
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle (full, replicated)
    std::vector<val_t> rho;         // Reflectivity (full, replicated)
    std::vector<int> tau;           // Time delays, LOCAL rows: rowCount x numTriangles
    std::vector<val_t> radE;        // Emission radiosity, FULL replicated: T x N
    std::vector<val_t> radB;        // Reflected radiosity, FULL replicated: T x N
    std::vector<val_t> distances;   // Computed distances, FULL replicated: N

    Octree octree;                  // Spatial acceleration structure (built redundantly on every rank)

    size_t sourceIndex;
    size_t rowStart = 0, rowCount = 0;  // This rank's row-block of the triangle index space

    size_t idxLocal(size_t li, size_t j) const { return li * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// Contiguous block row decomposition of [0, N) across `size` ranks.
void computeRowRange(size_t N, int rank, int size, size_t& start, size_t& count) {
    size_t chunk = N / static_cast<size_t>(size);
    size_t rem = N % static_cast<size_t>(size);
    if (static_cast<size_t>(rank) < rem) {
        count = chunk + 1;
        start = static_cast<size_t>(rank) * (chunk + 1);
    } else {
        count = chunk;
        start = rem * (chunk + 1) + (static_cast<size_t>(rank) - rem) * chunk;
    }
}

// ============================================================================
// GPU resident buffers (one set per rank/GPU)
// ============================================================================

struct GpuBuffers {
    Triangle* triangles = nullptr;      // N
    GpuOctreeNode* nodes = nullptr;     // flattened octree nodes
    int* triIdx = nullptr;              // flattened octree triangle index list
    float* areas = nullptr;             // N
    float* rho = nullptr;               // N
    float* kijLocal = nullptr;          // rowCount x N
    int* tauLocal = nullptr;            // rowCount x N
    float* radE = nullptr;              // T x N
    float* radB = nullptr;              // T x N
    float* rowBufLocal = nullptr;       // rowCount (scratch, reused for sim step / distances)
};

void freeGpuBuffers(GpuBuffers& gpu) {
    if (gpu.triangles) cudaFree(gpu.triangles);
    if (gpu.nodes) cudaFree(gpu.nodes);
    if (gpu.triIdx) cudaFree(gpu.triIdx);
    if (gpu.areas) cudaFree(gpu.areas);
    if (gpu.rho) cudaFree(gpu.rho);
    if (gpu.kijLocal) cudaFree(gpu.kijLocal);
    if (gpu.tauLocal) cudaFree(gpu.tauLocal);
    if (gpu.radE) cudaFree(gpu.radE);
    if (gpu.radB) cudaFree(gpu.radB);
    if (gpu.rowBufLocal) cudaFree(gpu.rowBufLocal);
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity, int rank, int mpiSize) {
    // Generate mesh (identical, deterministic, redundant on every rank)
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    computeRowRange(state.numTriangles, rank, mpiSize, state.rowStart, state.rowCount);

    if (rank == 0) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if (rank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices (tau is local-rows-only; kij lives purely on device)
    state.tau.resize(state.rowCount * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
}

// ============================================================================
// GPU setup: flatten octree, copy static data to device, allocate work buffers
// ============================================================================

void setupGpuBuffers(SimulationState& state, GpuBuffers& gpu) {
    size_t N = state.numTriangles;

    CUDA_CHECK(cudaMalloc(&gpu.triangles, N * sizeof(Triangle)));
    CUDA_CHECK(cudaMemcpy(gpu.triangles, state.triangles.data(), N * sizeof(Triangle), cudaMemcpyHostToDevice));

    std::vector<GpuOctreeNode> nodes;
    std::vector<int> triIdxFlat;
    flattenOctree(state.octree, nodes, triIdxFlat);

    CUDA_CHECK(cudaMalloc(&gpu.nodes, nodes.size() * sizeof(GpuOctreeNode)));
    CUDA_CHECK(cudaMemcpy(gpu.nodes, nodes.data(), nodes.size() * sizeof(GpuOctreeNode), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&gpu.triIdx, std::max<size_t>(triIdxFlat.size(), 1) * sizeof(int)));
    if (!triIdxFlat.empty()) {
        CUDA_CHECK(cudaMemcpy(gpu.triIdx, triIdxFlat.data(), triIdxFlat.size() * sizeof(int), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&gpu.areas, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(gpu.areas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&gpu.rho, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(gpu.rho, state.rho.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    size_t T = state.numTimesteps;
    CUDA_CHECK(cudaMalloc(&gpu.radE, T * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(gpu.radE, state.radE.data(), T * N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&gpu.radB, T * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(gpu.radB, 0, T * N * sizeof(float)));

    size_t rowCount = state.rowCount;
    if (rowCount > 0) {
        CUDA_CHECK(cudaMalloc(&gpu.kijLocal, rowCount * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu.tauLocal, rowCount * N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&gpu.rowBufLocal, rowCount * sizeof(float)));
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state, GpuBuffers& gpu, int rank) {
    if (rank == 0) printf("Computing form factors (Kij) on GPU...\n");

    size_t N = state.numTriangles;
    size_t rowCount = state.rowCount, rowStart = state.rowStart;
    if (rowCount > 0) {
        size_t total = rowCount * N;
        int threads = 256;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        const uint64_t seed = 42ULL;
        kernComputeKij<<<blocks, threads>>>(gpu.triangles, static_cast<int>(N), static_cast<int>(rowStart),
                                             static_cast<int>(rowCount), gpu.nodes, gpu.triIdx, gpu.kijLocal, seed);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

unsigned long long countNonzeroKijGPU(GpuBuffers& gpu, size_t rowCount, size_t N) {
    if (rowCount == 0) return 0ULL;
    unsigned long long* d_counter;
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(unsigned long long)));

    size_t total = rowCount * N;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    kernCountNonzero<<<blocks, threads>>>(gpu.kijLocal, total, d_counter);
    CUDA_CHECK(cudaGetLastError());

    unsigned long long h = 0;
    CUDA_CHECK(cudaMemcpy(&h, d_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    cudaFree(d_counter);
    return h;
}

void computeTimeDelays(SimulationState& state, GpuBuffers& gpu, int rank) {
    if (rank == 0) printf("Computing time delays (Tau)...\n");

    size_t N = state.numTriangles;
    size_t rowCount = state.rowCount, rowStart = state.rowStart;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t li = 0; li < rowCount; ++li) {
        for (size_t j = 0; j < N; ++j) {
            size_t i = rowStart + li;
            state.tau[state.idxLocal(li, j)] = (i == j) ? 0 : computeTau(state.triangles[i], state.triangles[j]);
        }
    }

    if (rowCount > 0) {
        CUDA_CHECK(cudaMemcpy(gpu.tauLocal, state.tau.data(), rowCount * N * sizeof(int), cudaMemcpyHostToDevice));
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
//
// Each timestep is bulk-synchronous: every rank computes its local rows of
// radB for time t on its GPU, then an MPI_Allgatherv combines the local
// segments into the full radB row (needed by every rank, since Kij couples
// every triangle to every other triangle), which is then pushed back to
// every rank's device for use by timestep t+1.
// ============================================================================

void runSimulation(SimulationState& state, GpuBuffers& gpu, int rank,
                    const std::vector<int>& recvCounts, const std::vector<int>& displs) {
    if (rank == 0) printf("Running wave propagation simulation...\n");

    size_t N = state.numTriangles;
    size_t rowCount = state.rowCount, rowStart = state.rowStart;
    std::vector<float> localRow(rowCount);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (rowCount > 0) {
            int threads = 128;
            kernComputeSimStep<<<static_cast<int>(rowCount), threads>>>(
                static_cast<int>(N), static_cast<int>(rowStart), static_cast<int>(rowCount), static_cast<int>(t),
                gpu.kijLocal, gpu.tauLocal, gpu.areas, gpu.rho, gpu.radE, gpu.radB, gpu.rowBufLocal);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(localRow.data(), gpu.rowBufLocal, rowCount * sizeof(float), cudaMemcpyDeviceToHost));
        }

        MPI_Allgatherv(rowCount > 0 ? localRow.data() : nullptr, static_cast<int>(rowCount), MPI_FLOAT,
                        state.radB.data() + t * N, recvCounts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpy(gpu.radB + t * N, state.radB.data() + t * N, N * sizeof(float), cudaMemcpyHostToDevice));

        if (rank == 0 && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state, GpuBuffers& gpu, int rank,
                       const std::vector<int>& recvCounts, const std::vector<int>& displs) {
    if (rank == 0) printf("Computing distances via cross-correlation...\n");

    size_t N = state.numTriangles;
    size_t rowCount = state.rowCount, rowStart = state.rowStart;
    std::vector<float> localDist(rowCount);

    if (rowCount > 0) {
        int threads = 128;
        kernComputeDistances<<<static_cast<int>(rowCount), threads>>>(
            static_cast<int>(N), static_cast<int>(rowStart), static_cast<int>(rowCount),
            static_cast<int>(state.numTimesteps), static_cast<int>(state.sourceIndex), gpu.radB, gpu.rowBufLocal);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(localDist.data(), gpu.rowBufLocal, rowCount * sizeof(float), cudaMemcpyDeviceToHost));
    }

    MPI_Allgatherv(rowCount > 0 ? localDist.data() : nullptr, static_cast<int>(rowCount), MPI_FLOAT,
                    state.distances.data(), recvCounts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);
}

// ============================================================================
// Validation (rank 0 only; distances/radB are fully replicated, but the Kij
// nonzero count is a global reduction across the distributed local matrices)
// ============================================================================

bool validateResults(const SimulationState& state, unsigned long long globalNonZeroKij, size_t globalKijElements) {
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
    #pragma omp parallel for reduction(+:receivedEnergy) schedule(dynamic)
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

    // Check Kij matrix (should have some non-zero entries), reduced globally across ranks
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           globalNonZeroKij, globalKijElements,
           100.0 * static_cast<double>(globalNonZeroKij) / static_cast<double>(globalKijElements));

    if (globalNonZeroKij == 0) {
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
    #pragma omp parallel for reduction(^:hash) schedule(static)
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
    MPI_Init(&argc, &argv);
    int rank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // Assign one GPU per rank, round-robin over the GPUs visible on this node.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices visible to rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical argv on every rank under mpirun)
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("MPI ranks: %d, GPUs visible per node: %d\n", mpiSize, deviceCount);
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity, rank, mpiSize);

    // Row-block gather metadata (row decomposition of the triangle index space)
    std::vector<int> recvCounts(mpiSize), displs(mpiSize);
    for (int r = 0; r < mpiSize; ++r) {
        size_t s, c;
        computeRowRange(state.numTriangles, r, mpiSize, s, c);
        recvCounts[r] = static_cast<int>(c);
        displs[r] = static_cast<int>(s);
    }

    GpuBuffers gpu;
    setupGpuBuffers(state, gpu);

    if (rank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    double startPre = MPI_Wtime();

    computeTimeDelays(state, gpu, rank);
    computeFormFactors(state, gpu, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double preDuration = (MPI_Wtime() - startPre) * 1000.0;
    double globalPreDuration = 0.0;
    MPI_Reduce(&preDuration, &globalPreDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    unsigned long long localNonZeroKij = countNonzeroKijGPU(gpu, state.rowCount, state.numTriangles);
    unsigned long long globalNonZeroKij = 0;
    MPI_Reduce(&localNonZeroKij, &globalNonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Precomputation time: %.3f ms\n", globalPreDuration);
        printf("\n");
    }

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    double startSim = MPI_Wtime();

    runSimulation(state, gpu, rank, recvCounts, displs);

    MPI_Barrier(MPI_COMM_WORLD);
    double simDuration = (MPI_Wtime() - startSim) * 1000.0;
    double globalSimDuration = 0.0;
    MPI_Reduce(&simDuration, &globalSimDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %.3f ms\n", globalSimDuration);
        printf("\n");
    }

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    double startDist = MPI_Wtime();

    computeDistances(state, gpu, rank, recvCounts, displs);

    MPI_Barrier(MPI_COMM_WORLD);
    double distDuration = (MPI_Wtime() - startDist) * 1000.0;
    double globalDistDuration = 0.0;
    MPI_Reduce(&distDuration, &globalDistDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Distance computation time: %.3f ms\n", globalDistDuration);
        printf("\n");
    }

    double totalTime = globalPreDuration + globalSimDuration + globalDistDuration;

    int validationResult = 0;
    if (rank == 0) {
        printf("Total computation time: %.3f ms\n", totalTime);

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
        printf("  Total time per triangle: %.4f ms\n", totalTime / static_cast<double>(n));

        // Memory usage (conceptual global problem size, independent of distribution)
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
            std::vector<double> distData(state.distances.begin(), state.distances.end());
            print_results(distData, "Distances");
        }

        // Validation
        if (validate) {
            validationResult = validateResults(state, globalNonZeroKij, n * n) ? 0 : 1;
        }
    }

    MPI_Bcast(&validationResult, 1, MPI_INT, 0, MPI_COMM_WORLD);

    freeGpuBuffers(gpu);
    MPI_Finalize();
    return validationResult;
}
