/**
 * Room Response Simulation Benchmark (CUDA-parallel implementation)
 *
 * GPU-parallel implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light waves propagate
 * between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization strategy (CUDA):
 *  - Form factors: one GPU thread per triangle pair; visibility rays traverse a
 *    flattened (array-based) octree with an explicit stack. Monte Carlo samples
 *    come from a counter-based RNG seeded per (pair, ray) so all pairs are
 *    independent and deterministic.
 *  - Time delays: one GPU thread per triangle pair.
 *  - Wave propagation: one kernel launch per timestep (timesteps are inherently
 *    sequential since tau >= 1); one block per receiving triangle with a
 *    block-wide reduction over emitting triangles.
 *  - Distance estimation: one block per triangle; threads compute correlations
 *    for different lags and reduce to the argmax (ties -> smallest lag, matching
 *    the sequential scan semantics).
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err_));                        \
            exit(1);                                                            \
        }                                                                       \
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
    __host__ __device__ val_t norm() const { return sqrtf(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    __host__ __device__ bool operator==(const Vec3& o) const {
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON && fabsf(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
};

// ============================================================================
// Triangle-Box Overlap Test (for octree construction, host-side)
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
// Octree for Spatial Acceleration (built on host, flattened for the GPU)
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

// ----------------------------------------------------------------------------
// Flattened (array-based) octree representation for GPU traversal
// ----------------------------------------------------------------------------

struct FlatNode {
    Vec3 center;
    Vec3 halfExtent;
    int child[8];    // Node index of each child, -1 if absent
    int triStart;    // Offset into flat triangle-index array (leaves only)
    int triCount;    // Number of triangles; > 0 identifies a leaf
};

static int flattenOctree(const Octree& node, std::vector<FlatNode>& nodes,
                         std::vector<int>& triIndices) {
    int self = static_cast<int>(nodes.size());
    nodes.emplace_back();

    FlatNode fn;
    fn.center = node.center;
    fn.halfExtent = node.halfExtent;
    fn.triStart = 0;
    fn.triCount = 0;
    for (int i = 0; i < 8; ++i) fn.child[i] = -1;

    if (!node.triangleIndices.empty()) {
        fn.triStart = static_cast<int>(triIndices.size());
        fn.triCount = static_cast<int>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) triIndices.push_back(static_cast<int>(idx));
    } else {
        for (int i = 0; i < 8; ++i) {
            if (node.children[i]) {
                fn.child[i] = flattenOctree(*node.children[i], nodes, triIndices);
            }
        }
    }

    nodes[self] = fn;
    return self;
}

// ============================================================================
// Mesh Generation: Icosphere (host-side)
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
// Random Number Generation (counter-based, GPU)
// ============================================================================

// Counter-based RNG (splitmix64 finalizer): every (pair, ray, draw) gets an
// independent, deterministic uniform sample in [0, 1). This makes the Monte
// Carlo form-factor sampling order-independent and thus fully parallel.
__device__ __forceinline__ val_t rnd01(unsigned long long counter) {
    unsigned long long z = counter + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z = z ^ (z >> 31);
    return static_cast<val_t>(z >> 40) * (1.0f / 16777216.0f);
}

// Generate a random point inside a triangle using barycentric coordinates
__device__ __forceinline__ Vec3 randomPointInTriangle(const Triangle& t,
                                                      unsigned long long counter) {
    val_t u = rnd01(counter);
    val_t v = rnd01(counter + 1);
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

__device__ __forceinline__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                                      const Vec3& v0, const Vec3& v1,
                                                      const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (flattened-octree traversal on the GPU)
// ============================================================================

// Check if a ray (as a segment) intersects a node's bounding box
__device__ __forceinline__ bool rayIntersectsBox(const Vec3& p1, const Vec3& p2,
                                                 const FlatNode& node) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3& h = node.halfExtent;

    if (fabsf(c.x) > h.x + ad.x) return false;
    if (fabsf(c.y) > h.y + ad.y) return false;
    if (fabsf(c.z) > h.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle.
// Iterative octree traversal with an explicit stack (no recursion on GPU).
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const FlatNode* __restrict__ nodes,
                             const int* __restrict__ triIndices,
                             const Triangle* __restrict__ triangles,
                             int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[96];
    int sp = 0;
    stack[sp++] = 0;  // The root's box is not tested, matching applyToTris()

    while (sp > 0) {
        const FlatNode& node = nodes[stack[--sp]];

        if (node.triCount > 0) {
            // Leaf: test contained triangles
            for (int k = 0; k < node.triCount; ++k) {
                int idx = triIndices[node.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const Triangle& tri = triangles[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;  // Ray is blocked
                }
            }
        } else {
            // Internal node: descend into children intersecting the ray
            for (int i = 0; i < 8; ++i) {
                int c = node.child[i];
                if (c >= 0 && rayIntersectsBox(from, to, nodes[c])) {
                    stack[sp++] = c;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij) — one thread per triangle pair
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__device__ __forceinline__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// One group of NUM_RAYS threads per triangle pair: each thread traces a single
// visibility ray, and the group's contributions are combined with warp shuffles.
// Rays of the same pair traverse similar octree paths, keeping warps coherent.
constexpr int KIJ_BLOCK = 128;
constexpr int PAIRS_PER_BLOCK = KIJ_BLOCK / NUM_RAYS;

__global__ void kijKernel(int numTriangles,
                          unsigned long long pairOffset, unsigned long long pairCount,
                          const Triangle* __restrict__ triangles,
                          const FlatNode* __restrict__ nodes,
                          const int* __restrict__ triIndices,
                          val_t* __restrict__ kijOut) {
    int ray = threadIdx.x % NUM_RAYS;
    unsigned long long localPair =
        static_cast<unsigned long long>(blockIdx.x) * PAIRS_PER_BLOCK +
        threadIdx.x / NUM_RAYS;
    if (localPair >= pairCount) return;
    unsigned long long pair = pairOffset + localPair;

    int i = static_cast<int>(pair / numTriangles);
    int j = static_cast<int>(pair % numTriangles);

    val_t contrib = ZERO;

    // Cull the diagonal and triangles facing the same direction
    if (i != j && triangles[i].normal().dot(triangles[j].normal()) <= 0.99f) {
        const Triangle& triI = triangles[i];
        const Triangle& triJ = triangles[j];

        // 4 independent draws per ray (2 per sampled point)
        unsigned long long base = (pair * NUM_RAYS + ray) * 4ull;
        Vec3 pI = randomPointInTriangle(triI, base);
        Vec3 pJ = randomPointInTriangle(triJ, base + 2);

        if (!isRayBlocked(pI, pJ, nodes, triIndices, triangles, i, j)) {
            Vec3 v = pJ - pI;
            val_t distSqr = v.squaredNorm();
            if (distSqr >= EPSILON) {
                val_t cosPhiI = cosPhi(v, triI.normal());
                val_t cosPhiJ = cosPhi(-v, triJ.normal());
                if (cosPhiI > ZERO && cosPhiJ > ZERO) {
                    contrib = (cosPhiI * cosPhiJ) / (PI * distSqr);
                }
            }
        }
    }

    // Reduce the NUM_RAYS contributions of this pair within the warp. The mask
    // covers only this pair's 16-lane group: the warp's other group may have
    // exited at the pair-range boundary.
    unsigned groupMask = 0xffffu << ((threadIdx.x & 31u) & ~(NUM_RAYS - 1u));
    for (int offset = NUM_RAYS / 2; offset > 0; offset >>= 1)
        contrib += __shfl_down_sync(groupMask, contrib, offset, NUM_RAYS);

    if (ray == 0) kijOut[localPair] = contrib * INV_NUM_RAYS;
}

// ============================================================================
// Tau (time delay) Computation — one thread per triangle pair
// ============================================================================

// Delays fit comfortably in 8 bits (room diameter 20 units / 0.5 per step),
// which halves the per-pair memory traffic in the propagation kernel.
__global__ void tauKernel(int numTriangles,
                          const Triangle* __restrict__ triangles,
                          uint8_t* __restrict__ tauOut) {
    unsigned long long pair =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    unsigned long long numPairs =
        static_cast<unsigned long long>(numTriangles) * numTriangles;
    if (pair >= numPairs) return;

    int i = static_cast<int>(pair / numTriangles);
    int j = static_cast<int>(pair % numTriangles);

    if (i == j) {
        tauOut[pair] = 0;
        return;
    }

    val_t dist = (triangles[i].center() - triangles[j].center()).norm();
    tauOut[pair] = static_cast<uint8_t>(static_cast<int>(ceilf(dist * INV_WAVE_SPEED)));
}

// ============================================================================
// Simulation Kernel (one timestep; block per receiving triangle)
// ============================================================================

constexpr int SIM_BLOCK = 256;

__global__ void simStepKernel(int t, int numTriangles,
                              const val_t* __restrict__ kij,
                              const uint8_t* __restrict__ tau,
                              const val_t* __restrict__ areas,
                              const val_t* __restrict__ rho,
                              const val_t* __restrict__ radE,
                              val_t* __restrict__ radB) {
    int i = blockIdx.x;
    size_t rowOff = static_cast<size_t>(i) * numTriangles;

    val_t sumB = ZERO;
    for (int j = threadIdx.x; j < numTriangles; j += blockDim.x) {
        if (j == i) continue;

        val_t k = kij[rowOff + j];
        if (k <= ZERO) continue;

        int tauij = tau[rowOff + j];

        // Skip if wave hasn't yet propagated from j to i
        if (t < tauij) continue;

        // Get radiosity from source triangle at time when emission occurred
        val_t radJ = radB[static_cast<size_t>(t - tauij) * numTriangles + j];
        if (radJ <= ZERO) continue;

        // Accumulate contribution: form factor * area * source radiosity
        sumB += fminf(k * areas[j], ONE) * radJ;
    }

    // Block-wide reduction
    __shared__ val_t shSum[SIM_BLOCK / 32];
    for (int offset = 16; offset > 0; offset >>= 1)
        sumB += __shfl_down_sync(0xffffffffu, sumB, offset);
    if ((threadIdx.x & 31) == 0) shSum[threadIdx.x >> 5] = sumB;
    __syncthreads();

    if (threadIdx.x < 32) {
        int numWarps = (blockDim.x + 31) >> 5;
        val_t s = (threadIdx.x < numWarps) ? shSum[threadIdx.x] : ZERO;
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down_sync(0xffffffffu, s, offset);
        if (threadIdx.x == 0) {
            size_t ti = static_cast<size_t>(t) * numTriangles + i;
            // Update radiosity: reflection + emission
            radB[ti] = rho[i] * s + radE[ti];
        }
    }
}

// ============================================================================
// Distance Kernel (cross-correlation; block per triangle)
// ============================================================================

constexpr int DIST_BLOCK = 256;

__global__ void distKernel(int numTriangles, int numTimesteps, int sourceIndex,
                           const val_t* __restrict__ radB,
                           val_t* __restrict__ distances) {
    int i = blockIdx.x;

    // Each thread scans its lags in ascending order with a strict ">" update,
    // so ties resolve to the smallest lag, matching the sequential scan.
    val_t maxCorr = ZERO;
    int bestT = 0;

    for (int t = threadIdx.x; t < numTimesteps; t += blockDim.x) {
        val_t sum = ZERO;
        for (int tt = t; tt < numTimesteps; ++tt) {
            val_t pB = radB[static_cast<size_t>(tt) * numTriangles + i];
            val_t pS = radB[static_cast<size_t>(tt - t) * numTriangles + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    // Block-wide argmax reduction (ties -> smaller lag)
    __shared__ val_t shCorr[DIST_BLOCK];
    __shared__ int shT[DIST_BLOCK];
    shCorr[threadIdx.x] = maxCorr;
    shT[threadIdx.x] = bestT;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            val_t otherCorr = shCorr[threadIdx.x + stride];
            int otherT = shT[threadIdx.x + stride];
            if (otherCorr > shCorr[threadIdx.x] ||
                (otherCorr == shCorr[threadIdx.x] && otherT < shT[threadIdx.x])) {
                shCorr[threadIdx.x] = otherCorr;
                shT[threadIdx.x] = otherT;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        distances[i] = WAVE_SPEED * static_cast<val_t>(shT[0]);
    }
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
    std::vector<val_t> kij;         // Form factors (copied back only for validation)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure (host)

    size_t sourceIndex;

    // Device buffers (primary GPU, device 0)
    Triangle* dTriangles = nullptr;
    FlatNode* dNodes = nullptr;
    int* dTriIndices = nullptr;
    val_t* dKij = nullptr;
    uint8_t* dTau = nullptr;
    val_t* dAreas = nullptr;
    val_t* dRho = nullptr;
    val_t* dRadE = nullptr;
    val_t* dRadB = nullptr;
    val_t* dDistances = nullptr;

    // Additional GPUs assist with the dominant form-factor phase: each holds a
    // copy of the geometry and computes a contiguous slice of triangle pairs,
    // which is then gathered into dKij on device 0.
    struct PeerCtx {
        int device;
        unsigned long long pairStart, pairCount;
        Triangle* triangles = nullptr;
        FlatNode* nodes = nullptr;
        int* triIndices = nullptr;
        val_t* kijChunk = nullptr;
    };
    std::vector<PeerCtx> peers;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }

    ~SimulationState() {
        for (auto& p : peers) {
            cudaSetDevice(p.device);
            cudaFree(p.triangles);
            cudaFree(p.nodes);
            cudaFree(p.triIndices);
            cudaFree(p.kijChunk);
        }
        cudaSetDevice(0);
        cudaFree(dTriangles);
        cudaFree(dNodes);
        cudaFree(dTriIndices);
        cudaFree(dKij);
        cudaFree(dTau);
        cudaFree(dAreas);
        cudaFree(dRho);
        cudaFree(dRadE);
        cudaFree(dRadB);
        cudaFree(dDistances);
    }
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

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Flatten the octree for GPU traversal
    std::vector<FlatNode> flatNodes;
    std::vector<int> flatTriIndices;
    flattenOctree(state.octree, flatNodes, flatTriIndices);

    size_t n = state.numTriangles;

    // Initialize areas
    state.areas.resize(n);
    for (size_t i = 0; i < n; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(n, reflectivity);

    // Initialize radiosity matrices (host copies for output/validation)
    state.radE.resize(timesteps * n, ZERO);
    state.radB.resize(timesteps * n, ZERO);
    state.distances.resize(n, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Allocate and populate device buffers
    printf("Uploading geometry to GPU...\n");
    CUDA_CHECK(cudaMalloc(&state.dTriangles, n * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&state.dNodes, flatNodes.size() * sizeof(FlatNode)));
    CUDA_CHECK(cudaMalloc(&state.dTriIndices, flatTriIndices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.dKij, n * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dTau, n * n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&state.dAreas, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dRho, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dRadE, timesteps * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dRadB, timesteps * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dDistances, n * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(state.dTriangles, state.triangles.data(),
                          n * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dNodes, flatNodes.data(),
                          flatNodes.size() * sizeof(FlatNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dTriIndices, flatTriIndices.data(),
                          flatTriIndices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dAreas, state.areas.data(),
                          n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRho, state.rho.data(),
                          n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRadE, state.radE.data(),
                          timesteps * n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, timesteps * n * sizeof(val_t)));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Replicate the geometry on any additional GPUs so they can help with the
    // form-factor computation (each gets a contiguous slice of the pair range)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount > 1) {
        unsigned long long numPairs = static_cast<unsigned long long>(n) * n;
        unsigned long long perDev = numPairs / deviceCount;
        printf("Using %d GPUs for form-factor computation\n", deviceCount);

        for (int d = 1; d < deviceCount; ++d) {
            SimulationState::PeerCtx p;
            p.device = d;
            p.pairStart = perDev * d;
            p.pairCount = (d == deviceCount - 1) ? numPairs - p.pairStart : perDev;

            CUDA_CHECK(cudaSetDevice(d));
            CUDA_CHECK(cudaMalloc(&p.triangles, n * sizeof(Triangle)));
            CUDA_CHECK(cudaMalloc(&p.nodes, flatNodes.size() * sizeof(FlatNode)));
            CUDA_CHECK(cudaMalloc(&p.triIndices, flatTriIndices.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&p.kijChunk, p.pairCount * sizeof(val_t)));

            CUDA_CHECK(cudaMemcpy(p.triangles, state.triangles.data(),
                                  n * sizeof(Triangle), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(p.nodes, flatNodes.data(),
                                  flatNodes.size() * sizeof(FlatNode), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(p.triIndices, flatTriIndices.data(),
                                  flatTriIndices.size() * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
            state.peers.push_back(p);
        }
        CUDA_CHECK(cudaSetDevice(0));
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij) on GPU...\n");

    int n = static_cast<int>(state.numTriangles);
    unsigned long long numPairs = static_cast<unsigned long long>(n) * n;
    unsigned long long dev0Pairs =
        state.peers.empty() ? numPairs : state.peers.front().pairStart;

    // Launch the slices on all GPUs (kernel launches are asynchronous, so all
    // devices work concurrently)
    for (const auto& p : state.peers) {
        CUDA_CHECK(cudaSetDevice(p.device));
        unsigned long long grid = (p.pairCount + PAIRS_PER_BLOCK - 1) / PAIRS_PER_BLOCK;
        kijKernel<<<static_cast<unsigned int>(grid), KIJ_BLOCK>>>(
            n, p.pairStart, p.pairCount, p.triangles, p.nodes, p.triIndices, p.kijChunk);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaSetDevice(0));
    unsigned long long grid = (dev0Pairs + PAIRS_PER_BLOCK - 1) / PAIRS_PER_BLOCK;
    kijKernel<<<static_cast<unsigned int>(grid), KIJ_BLOCK>>>(
        n, 0ull, dev0Pairs, state.dTriangles, state.dNodes, state.dTriIndices, state.dKij);
    CUDA_CHECK(cudaGetLastError());

    // Gather the peer slices into the full Kij matrix on device 0
    for (const auto& p : state.peers) {
        CUDA_CHECK(cudaSetDevice(p.device));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpyPeer(state.dKij + p.pairStart, 0, p.kijChunk, p.device,
                                  p.pairCount * sizeof(val_t)));
    }
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("  Progress: %zu/%zu triangles\n", state.numTriangles, state.numTriangles);
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau) on GPU...\n");

    int n = static_cast<int>(state.numTriangles);
    unsigned long long numPairs = static_cast<unsigned long long>(n) * n;
    int block = 256;
    unsigned long long grid = (numPairs + block - 1) / block;

    tauKernel<<<static_cast<unsigned int>(grid), block>>>(n, state.dTriangles, state.dTau);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation on GPU...\n");

    int n = static_cast<int>(state.numTriangles);

    // Timesteps are inherently sequential (radB at time t depends on earlier
    // timesteps, tau >= 1), so launch one fully-parallel kernel per timestep.
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        simStepKernel<<<n, SIM_BLOCK>>>(static_cast<int>(t), n, state.dKij, state.dTau,
                                        state.dAreas, state.dRho, state.dRadE, state.dRadB);

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Keep a host copy of the radiosity history for validation
    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.dRadB,
                          state.numTimesteps * state.numTriangles * sizeof(val_t),
                          cudaMemcpyDeviceToHost));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation on GPU...\n");

    int n = static_cast<int>(state.numTriangles);

    distKernel<<<n, DIST_BLOCK>>>(n, static_cast<int>(state.numTimesteps),
                                  static_cast<int>(state.sourceIndex),
                                  state.dRadB, state.dDistances);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.dDistances,
                          n * sizeof(val_t), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(SimulationState& state) {
    printf("\nValidation:\n");

    // Fetch form factors from the GPU (only needed for validation)
    state.kij.resize(state.numTriangles * state.numTriangles);
    CUDA_CHECK(cudaMemcpy(state.kij.data(), state.dKij,
                          state.kij.size() * sizeof(val_t), cudaMemcpyDeviceToHost));

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

    // Initialize CUDA device up front so context creation isn't timed
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));

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
    size_t memTau = n * n * sizeof(uint8_t);
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
    if (validate) {
        if (!validateResults(state)) {
            return 1;
        }
    }

    return 0;
}
