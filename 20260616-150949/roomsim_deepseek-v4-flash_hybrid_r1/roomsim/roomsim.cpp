/**
 * Room Response Simulation Benchmark
 * 
 * This is a simplified, purely sequential implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
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
#include <random>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA error checking helper
#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                       \
    }                                                                  \
} while(0)

// Device math helpers (available on both host and device)
__host__ __device__ inline float dev_sqrt(float v) {
#ifdef __CUDA_ARCH__
    return sqrtf(v);
#else
    return std::sqrt(v);
#endif
}

__host__ __device__ inline float dev_abs(float v) {
#ifdef __CUDA_ARCH__
    return fabsf(v);
#else
    return std::abs(v);
#endif
}

__host__ __device__ inline float dev_min(float a, float b) {
#ifdef __CUDA_ARCH__
    return fminf(a, b);
#else
    return std::min(a, b);
#endif
}

__host__ __device__ inline float dev_max(float a, float b) {
#ifdef __CUDA_ARCH__
    return fmaxf(a, b);
#else
    return std::max(a, b);
#endif
}

__host__ __device__ inline int dev_ceil(float v) {
#ifdef __CUDA_ARCH__
    return (int)ceilf(v);
#else
    return (int)std::ceil(v);
#endif
}

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

// ============================================================================
// Vector and Triangle Types
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
// GPU-Compatible Types (POD structs for device-side computation)
// ============================================================================

struct GPUVec3 { float x, y, z; };

struct GPUTriangle {
    GPUVec3 a, b, c;
    GPUVec3 normal;
};

// Flattened octree node for GPU traversal
struct GPUNode {
    float cx, cy, cz;       // center
    float hx, hy, hz;       // half-extent
    int firstChild;          // index of first of 8 children (-1 if leaf)
    int triStart;            // start index in leafTriLists (-1 if internal)
    int triCount;            // number of triangles in leaf (0 if internal)
    int _pad;
};

// ============================================================================
// GPU Context (device-side state)
// ============================================================================

struct GPUContext {
    GPUTriangle* d_triangles;  // Device triangle data
    GPUNode* d_nodes;          // Flattened octree nodes
    int* d_leafTriLists;       // Triangle indices for octree leaves
    int numNodes;
    int leafTriListSize;
    float* d_kij;              // Form factor matrix (device)
    int* d_tau;                // Time delay matrix (device)
    float* d_radB;             // Radiosity B matrix (device) T x N
    float* d_radE;             // Emission radiosity (device) T x N
    float* d_areas;            // Triangle areas (device)
    int kijPitch;              // Pitch of kij matrix (in floats)
    int tauPitch;              // Pitch of tau matrix (in ints)
};

// ============================================================================
// CUDA Kernel Declarations
// ============================================================================

// Compute Kij form factors for rows [iStart, iEnd) of the Kij matrix
__global__ void computeKijKernel(
    const GPUTriangle* d_triangles, int N,
    const GPUNode* d_nodes, int numNodes,
    const int* d_leafTriLists,
    float* d_kij, int kijPitch,
    int iStart, int iEnd, unsigned long long seed);

// Compute radB[t][i] for given timestep t and triangles
// d_radB_curr points to d_radB + t * N (current timestep output)
__global__ void computeRadBKernel(
    const float* d_kij, int kijPitch,
    const int* d_tau, int tauPitch,
    const float* d_radB_hist, const float* d_radE,
    float* d_radB_curr,
    int N, int t, int iStart, int iEnd, float rho,
    const float* d_areas);

// Compute distances via cross-correlation
__global__ void computeDistancesKernel(
    const float* d_radB, int radPitch,
    float* d_distances,
    int N, int T, int iStart, int iEnd, int sourceIdx);

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

public:
    // Check if a ray intersects this node's bounding box
    bool rayIntersectsBox(const Vec3& p1, const Vec3& p2) const {
        Vec3 d = (p2 - p1) * 0.5f;
        Vec3 c = p1 + d - center;
        Vec3 ad = {std::abs(d.x), std::abs(d.y), std::abs(d.z)};

        if (std::abs(c.x) > halfExtent.x + ad.x) return false;
        if (std::abs(c.y) > halfExtent.y + ad.y) return false;
        if (std::abs(c.z) > halfExtent.z + ad.z) return false;

        if (std::abs(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
        if (std::abs(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
        if (std::abs(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

        return true;
    }

    // Apply a function to all triangles potentially intersecting the ray
    // Returns true if the function returns true for any triangle
    template<typename Func>
    bool applyToTris(const Vec3& p1, const Vec3& p2, Func&& func) const {
        // If leaf node, check triangles directly
        if (!triangleIndices.empty()) {
            for (size_t idx : triangleIndices) {
                if (func(idx, (*allTriangles)[idx])) return true;
            }
            return false;
        }

        // Otherwise, descend to children
        for (int i = 0; i < 8; ++i) {
            if (children[i] && children[i]->rayIntersectsBox(p1, p2)) {
                if (children[i]->applyToTris(p1, p2, func)) return true;
            }
        }
        return false;
    }

    // Flatten octree into GPU-friendly arrays
    void flatten(std::vector<GPUNode>& nodes, std::vector<int>& leafTriLists) const {
        nodes.clear();
        leafTriLists.clear();
        flattenNode(nodes, leafTriLists);
    }

private:
    int flattenNode(std::vector<GPUNode>& nodes, std::vector<int>& leafTriLists) const {
        int idx = static_cast<int>(nodes.size());
        GPUNode n;
        n.cx = center.x; n.cy = center.y; n.cz = center.z;
        n.hx = halfExtent.x; n.hy = halfExtent.y; n.hz = halfExtent.z;
        n._pad = 0;

        if (!triangleIndices.empty()) {
            // Leaf node
            n.firstChild = -1;
            n.triStart = static_cast<int>(leafTriLists.size());
            n.triCount = static_cast<int>(triangleIndices.size());
            leafTriLists.insert(leafTriLists.end(), triangleIndices.begin(), triangleIndices.end());
            nodes.push_back(n);
            return idx;
        }

        // Internal node: push placeholder, reserve child slots
        n.firstChild = -1;
        n.triStart = -1;
        n.triCount = 0;
        nodes.push_back(n);

        int childBase = static_cast<int>(nodes.size());
        nodes[idx].firstChild = childBase;

        // Pre-allocate 8 child slots
        GPUNode dummy;
        dummy.firstChild = -1; dummy.triStart = -1; dummy.triCount = 0;
        dummy.cx = dummy.cy = dummy.cz = 0;
        dummy.hx = dummy.hy = dummy.hz = 0;
        for (int i = 0; i < 8; ++i) nodes.push_back(dummy);

        // Recursively fill children
        for (int i = 0; i < 8; ++i) {
            if (children[i]) {
                children[i]->flattenNode(nodes, leafTriLists);
            }
        }
        return idx;
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
// Random Number Generation
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
};

// Generate a random point inside a triangle using barycentric coordinates
Vec3 randomPointInTriangle(const Triangle& t, RandomGenerator& rng) {
    val_t u = rng.rand();
    val_t v = rng.rand();
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

val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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
// Visibility Testing (Octree-accelerated)
// ============================================================================

// Check if a ray between two triangles is blocked by any other triangle
// Uses octree for O(log N) instead of O(N) search
bool isRayBlocked(const Vec3& from, const Vec3& to,
                  const Octree& octree,
                  size_t srcTriIdx, size_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    return octree.applyToTris(from, to, [&](size_t idx, const Triangle& tri) {
        if (idx == srcTriIdx || idx == dstTriIdx) return false;

        val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
        if (dist > EPSILON && dist < rayLen - EPSILON) {
            return true;  // Ray is blocked
        }
        return false;
    });
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return std::max(ZERO, v.dot(normal) / vNorm);
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle(triI, rng);
        Vec3 pJ = randomPointInTriangle(triJ, rng);

        if (isRayBlocked(pI, pJ, octree, idxI, idxJ)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.normal());
        val_t cosPhiJ = cosPhi(-v, triJ.normal());

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    return kij * INV_NUM_RAYS;
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
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
    std::vector<val_t> kij;         // Form factors (N x N matrix, row-major)
    std::vector<int> tau;           // Time delays (N x N matrix, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // GPU context for parallelized computation
    GPUContext gpu;

    // MPI work distribution
    int mpiRank;
    int mpiSize;
    size_t localStart;   // Start index for this rank's triangles
    size_t localEnd;     // End index (exclusive)
    size_t localCount;   // Number of triangles assigned to this rank

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }

    void distributeWork() {
        size_t base = (numTriangles + mpiSize - 1) / mpiSize;
        localStart = mpiRank * base;
        if (localStart >= numTriangles) { localStart = localEnd = 0; localCount = 0; return; }
        localEnd = std::min(localStart + base, numTriangles);
        localCount = localEnd - localStart;
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

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
    state.tau.resize(state.numTriangles * state.numTriangles, 0);
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
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    int rank = state.mpiRank;
    int size = state.mpiSize;

    if (rank == 0) printf("Computing form factors (Kij) using MPI+OpenMP+CUDA...\n");

    size_t N = state.numTriangles;
    size_t localStart = state.localStart;
    size_t localEnd = state.localEnd;
    size_t localCount = state.localCount;

    // All ranks launch GPU kernel for their row range
    if (localCount > 0) {
        int iStartGPU = (int)localStart;
        int iEndGPU = (int)localEnd;
        int totalI = (int)localCount;

        dim3 blockDim(128);
        dim3 gridDim((totalI + 127) / 128);

        computeKijKernel<<<gridDim, blockDim>>>(
            state.gpu.d_triangles, (int)N,
            state.gpu.d_nodes, state.gpu.numNodes,
            state.gpu.d_leafTriLists,
            state.gpu.d_kij, state.gpu.kijPitch,
            iStartGPU, iEndGPU, 42ULL);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy results back (kernel writes at global row positions)
        size_t rowSizeBytes = N * sizeof(float);
        size_t srcPitch = (size_t)state.gpu.kijPitch * sizeof(float);
        float* srcPtr = state.gpu.d_kij + (size_t)localStart * state.gpu.kijPitch;
        float* dstPtr = &state.kij[localStart * N];
        CUDA_CHECK(cudaMemcpy2D(dstPtr, rowSizeBytes, srcPtr, srcPitch,
                                rowSizeBytes, localCount, cudaMemcpyDeviceToHost));
    }

    // Gather all Kij data on rank 0, then broadcast to all
    if (size > 1) {
        // Use MPI_Gatherv to collect all pieces on rank 0
        std::vector<int> recvCounts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            size_t rBase = ((size_t)N + size - 1) / size;
            size_t rStart = (size_t)r * rBase;
            size_t rEnd = std::min(rStart + rBase, N);
            int rCount = (rEnd > rStart) ? (int)((rEnd - rStart) * N) : 0;
            recvCounts[r] = rCount;
            displs[r] = (int)(rStart * N);
        }

        if (rank == 0) {
            for (int r = 1; r < size; ++r) {
                if (recvCounts[r] > 0) {
                    MPI_Recv(&state.kij[displs[r]], recvCounts[r], MPI_FLOAT, r, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        } else if (localCount > 0) {
            MPI_Send(&state.kij[localStart * N], (int)(localCount * N),
                     MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }

        // Broadcast to all ranks
        MPI_Bcast(state.kij.data(), (int)(N * N), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(state.tau.data(), (int)(N * N), MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Copy complete matrices to GPU for simulation phase
    CUDA_CHECK(cudaMemcpy2D(
        state.gpu.d_kij, (size_t)state.gpu.kijPitch * sizeof(float),
        state.kij.data(), N * sizeof(float),
        N * sizeof(float), N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(
        state.gpu.d_tau, (size_t)state.gpu.tauPitch * sizeof(int),
        state.tau.data(), N * sizeof(int),
        N * sizeof(int), N, cudaMemcpyHostToDevice));

    if (rank == 0) {
        int nonZero = 0;
        for (size_t i = 0; i < N * N; ++i) {
            if (state.kij[i] > EPSILON) nonZero++;
        }
        printf("  Non-zero form factors: %d/%zu (%.1f%%)\n",
               nonZero, N * N, 100.0 * nonZero / (N * N));
    }
}

void computeTimeDelays(SimulationState& state) {
    if (state.mpiRank == 0) printf("Computing time delays (Tau)...\n");

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.tau[state.idx2d(i, j)] = computeTau(
                state.triangles[i], state.triangles[j]);
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    int rank = state.mpiRank;
    int size = state.mpiSize;
    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;

    if (rank == 0) printf("Running wave propagation simulation using MPI+OpenMP+CUDA...\n");

    // Copy radB, radE and areas to GPU
    CUDA_CHECK(cudaMemset(state.gpu.d_radB, 0, T * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(state.gpu.d_radE, state.radE.data(),
                          T * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.gpu.d_areas, state.areas.data(),
                          N * sizeof(float), cudaMemcpyHostToDevice));

    for (size_t t = 0; t < T; ++t) {
        size_t localCount = state.localCount;
        size_t localStart = state.localStart;

        // Launch GPU kernel for this rank's triangle range
        if (localCount > 0) {
            dim3 blockDim(256);
            dim3 gridDim((unsigned int)((localCount + 255) / 256));

            float* d_radB_curr = state.gpu.d_radB + (size_t)t * state.numTriangles;
            computeRadBKernel<<<gridDim, blockDim>>>(
                state.gpu.d_kij, state.gpu.kijPitch,
                state.gpu.d_tau, state.gpu.tauPitch,
                state.gpu.d_radB,  // historical radB (full T x N buffer)
                state.gpu.d_radE,
                d_radB_curr,       // output at current timestep t
                (int)N, (int)t, (int)localStart, (int)(localStart + localCount),
                state.rho[0], state.gpu.d_areas);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Share radB[t] across all MPI ranks via Allgather
        // radB is a 2D array: T x N, with stride N
        if (size > 1) {
            std::vector<float> localRadB;
            if (localCount > 0) {
                localRadB.resize(localCount);
                CUDA_CHECK(cudaMemcpy(localRadB.data(),
                                      state.gpu.d_radB + (size_t)t * N + localStart,
                                      localCount * sizeof(float),
                                      cudaMemcpyDeviceToHost));
            }

            // Compute recv counts and displacements
            std::vector<int> recvCounts(size);
            std::vector<int> displs(size);
            for (int r = 0; r < size; ++r) {
                size_t rBase = ((size_t)N + size - 1) / size;
                size_t rStart = rBase * r;
                size_t rEnd = std::min(rStart + rBase, N);
                recvCounts[r] = (int)(rEnd - rStart);
                displs[r] = (int)rStart;
            }

            std::vector<float> allRadB(N);
            MPI_Allgatherv(localRadB.data(), (int)localCount, MPI_FLOAT,
                           allRadB.data(), recvCounts.data(), displs.data(),
                           MPI_FLOAT, MPI_COMM_WORLD);

            // Copy back to GPU so all ranks have radB[t][*] for next timestep
            CUDA_CHECK(cudaMemcpy(state.gpu.d_radB + (size_t)t * N,
                                  allRadB.data(), N * sizeof(float),
                                  cudaMemcpyHostToDevice));
        }

        if (rank == 0 && ((t + 1) % 10 == 0 || t + 1 == T)) {
            printf("  Timestep %zu/%zu\n", t + 1, T);
        }
    }

    // Copy final radB back to host
    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.gpu.d_radB,
                          T * N * sizeof(float), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    int rank = state.mpiRank;
    int size = state.mpiSize;
    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;

    if (rank == 0) printf("Computing distances via cross-correlation using MPI+CUDA...\n");

    // RadB is already on GPU from the simulation phase
    // Allocate GPU distances buffer
    float* d_distances = nullptr;
    CUDA_CHECK(cudaMalloc(&d_distances, N * sizeof(float)));

    size_t localCount = state.localCount;
    size_t localStart = state.localStart;

    if (localCount > 0) {
        dim3 blockDim(128);
        dim3 gridDim((unsigned int)((localCount + 127) / 128));

        computeDistancesKernel<<<gridDim, blockDim>>>(
            state.gpu.d_radB, (int)N,
            d_distances,
            (int)N, (int)T, (int)localStart, (int)(localStart + localCount),
            (int)state.sourceIndex);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy local distances back
        CUDA_CHECK(cudaMemcpy(&state.distances[localStart],
                              d_distances + localStart,
                              localCount * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_distances));

    // Gather distances on rank 0
    if (size > 1) {
        // Compute recv counts and displacements for root
        std::vector<int> recvCounts(size, 0);
        std::vector<int> displs(size, 0);
        for (int r = 0; r < size; ++r) {
            size_t rBase = ((size_t)N + size - 1) / size;
            size_t rStart = rBase * r;
            size_t rEnd = std::min(rStart + rBase, N);
            recvCounts[r] = (int)(rEnd - rStart);
            displs[r] = (int)rStart;
        }

        // Root receives from all non-root ranks (its own data already in place)
        if (rank == 0) {
            for (int r = 1; r < size; ++r) {
                if (recvCounts[r] > 0) {
                    MPI_Recv(&state.distances[displs[r]], recvCounts[r],
                             MPI_FLOAT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        } else if (localCount > 0) {
            MPI_Send(&state.distances[localStart], (int)localCount,
                     MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }

        // Broadcast to all ranks for validation
        MPI_Bcast(state.distances.data(), (int)N, MPI_FLOAT, 0, MPI_COMM_WORLD);
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
// CUDA Device Functions
// ============================================================================

// Inline cross/dot for CUDA float3 (avoids dependence on CUDA math headers)
__device__ inline float3 d_cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ inline float d_dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float d_rayTriangleIntersect(
    float3 orig, float3 dir,
    float3 v0, float3 v1, float3 v2)
{
    float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    float3 pvec = d_cross(dir, e2);
    float det = d_dot(e1, pvec);
    if (fabsf(det) < 1e-6f) return 1e30f;
    float invDet = 1.0f / det;
    float3 tvec = make_float3(orig.x - v0.x, orig.y - v0.y, orig.z - v0.z);
    float u = d_dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return 1e30f;
    float3 qvec = d_cross(tvec, e1);
    float v = d_dot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 1e30f;
    return d_dot(e2, qvec) * invDet;
}

__device__ bool d_rayBoxIntersect(float3 p1, float3 p2, const GPUNode& node) {
    float3 d = make_float3((p2.x - p1.x) * 0.5f, (p2.y - p1.y) * 0.5f, (p2.z - p1.z) * 0.5f);
    float3 c = make_float3(p1.x + d.x - node.cx, p1.y + d.y - node.cy, p1.z + d.z - node.cz);
    float adx = fabsf(d.x), ady = fabsf(d.y), adz = fabsf(d.z);
    if (fabsf(c.x) > node.hx + adx) return false;
    if (fabsf(c.y) > node.hy + ady) return false;
    if (fabsf(c.z) > node.hz + adz) return false;
    if (fabsf(d.y * c.z - d.z * c.y) > node.hy * adz + node.hz * ady + 1e-6f) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > node.hz * adx + node.hx * adz + 1e-6f) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > node.hx * ady + node.hy * adx + 1e-6f) return false;
    return true;
}

__device__ bool d_isRayBlocked(
    float3 from, float3 to,
    const GPUTriangle* tris, int N,
    const GPUNode* nodes,
    const int* leafTriLists,
    int srcIdx, int dstIdx)
{
    float3 dir = make_float3(to.x - from.x, to.y - from.y, to.z - from.z);
    float rayLen = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    if (rayLen < 1e-6f) return true;
    dir.x /= rayLen; dir.y /= rayLen; dir.z /= rayLen;

    // Iterative octree traversal with explicit stack
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int nIdx = stack[--sp];
        const GPUNode& node = nodes[nIdx];
        if (!d_rayBoxIntersect(from, to, node)) continue;

        if (node.firstChild < 0) {
            // Leaf node
            for (int t = 0; t < node.triCount; ++t) {
                int triIdx = leafTriLists[node.triStart + t];
                if (triIdx == srcIdx || triIdx == dstIdx) continue;
                const GPUTriangle& tri = tris[triIdx];
                float hit = d_rayTriangleIntersect(
                    from, dir,
                    make_float3(tri.a.x, tri.a.y, tri.a.z),
                    make_float3(tri.b.x, tri.b.y, tri.b.z),
                    make_float3(tri.c.x, tri.c.y, tri.c.z));
                if (hit > 1e-6f && hit < rayLen - 1e-6f) return true;
            }
        } else {
            // Internal node: push children in reverse order
            for (int c = 7; c >= 0; --c) {
                stack[sp++] = node.firstChild + c;
            }
        }
    }
    return false;
}

// Simple xorshift RNG for GPU
__device__ inline unsigned int d_xorshift(unsigned int& state) {
    unsigned int x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

__device__ inline float d_randf(unsigned int& state) {
    return (float)d_xorshift(state) / (float)0xFFFFFFFFu;
}

__device__ inline float d_cosPhi(float3 v, float3 normal) {
    float vn = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if (vn <= 1e-6f) return 0.0f;
    float d = (v.x*normal.x + v.y*normal.y + v.z*normal.z) / vn;
    return fmaxf(0.0f, d);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// Compute a block of form factors Kij[i][j] for i in [iStart, iEnd)
// Each thread handles one i, all NUM_RAYS samples sequentially for all j
// Writes to global row positions using globalI * kijPitch + j
__global__ void computeKijKernel(
    const GPUTriangle* d_triangles, int N,
    const GPUNode* d_nodes, int numNodes,
    const int* d_leafTriLists,
    float* d_kij, int kijPitch,
    int iStart, int iEnd, unsigned long long seed)
{
    int totalI = iEnd - iStart;
    int localI = blockIdx.x * blockDim.x + threadIdx.x;
    if (localI >= totalI) return;
    int globalI = iStart + localI;

    for (int j = 0; j < N; ++j) {
        if (globalI == j) {
            d_kij[(size_t)globalI * kijPitch + j] = 0.0f;
            continue;
        }

        const GPUTriangle& tI = d_triangles[globalI];
        const GPUTriangle& tJ = d_triangles[j];

        // Cull same-facing triangles
        float3 nI = make_float3(tI.normal.x, tI.normal.y, tI.normal.z);
        float3 nJ = make_float3(tJ.normal.x, tJ.normal.y, tJ.normal.z);
        float nd = nI.x * nJ.x + nI.y * nJ.y + nI.z * nJ.z;
        if (nd > 0.99f) {
            d_kij[(size_t)globalI * kijPitch + j] = 0.0f;
            continue;
        }

        unsigned int rngState = (unsigned int)(seed + (unsigned long long)globalI * N + j);

        float kij = 0.0f;
        for (int r = 0; r < NUM_RAYS; ++r) {
            // Random point in triangle I (barycentric)
            float u = d_randf(rngState);
            float v = d_randf(rngState);
            if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
            float3 pI;
            pI.x = tI.a.x + (tI.b.x - tI.a.x) * u + (tI.c.x - tI.a.x) * v;
            pI.y = tI.a.y + (tI.b.y - tI.a.y) * u + (tI.c.y - tI.a.y) * v;
            pI.z = tI.a.z + (tI.b.z - tI.a.z) * u + (tI.c.z - tI.a.z) * v;

            // Random point in triangle J
            u = d_randf(rngState);
            v = d_randf(rngState);
            if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
            float3 pJ;
            pJ.x = tJ.a.x + (tJ.b.x - tJ.a.x) * u + (tJ.c.x - tJ.a.x) * v;
            pJ.y = tJ.a.y + (tJ.b.y - tJ.a.y) * u + (tJ.c.y - tJ.a.y) * v;
            pJ.z = tJ.a.z + (tJ.b.z - tJ.a.z) * u + (tJ.c.z - tJ.a.z) * v;

            if (d_isRayBlocked(pI, pJ, d_triangles, N, d_nodes, d_leafTriLists, globalI, j))
                continue;

            float3 dir = make_float3(pJ.x - pI.x, pJ.y - pI.y, pJ.z - pI.z);
            float distSq = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;
            if (distSq < 1e-6f) continue;

            float cpI = d_cosPhi(dir, make_float3(tI.normal.x, tI.normal.y, tI.normal.z));
            float3 negDir = make_float3(-dir.x, -dir.y, -dir.z);
            float cpJ = d_cosPhi(negDir, make_float3(tJ.normal.x, tJ.normal.y, tJ.normal.z));
            if (cpI <= 0.0f || cpJ <= 0.0f) continue;

            kij += (cpI * cpJ) / (3.14159265f * distSq);
        }

        d_kij[(size_t)globalI * kijPitch + j] = kij * INV_NUM_RAYS;
    }
}

// Compute radB[t][i] for given timestep t and triangle range [iStart, iEnd)
// Each thread handles one i, reduces over j
// d_radB_curr should point to d_radB + t * N (output for current timestep)
__global__ void computeRadBKernel(
    const float* d_kij, int kijPitch,
    const int* d_tau, int tauPitch,
    const float* d_radB_hist,  // all historical radB data (T x N)
    const float* d_radE,
    float* d_radB_curr,        // output at timestep t (N elements)
    int N, int t, int iStart, int iEnd, float rho,
    const float* d_areas)
{
    int localI = blockIdx.x * blockDim.x + threadIdx.x;
    if (localI >= (iEnd - iStart)) return;
    int globalI = iStart + localI;

    float sumB = 0.0f;
    for (int j = 0; j < N; ++j) {
        if (globalI == j) continue;
        int tauij = d_tau[(size_t)globalI * tauPitch + j];
        if (t < tauij) continue;
        float kij = d_kij[(size_t)globalI * kijPitch + j];
        if (kij <= 0.0f) continue;
        int srcTime = t - tauij;
        float radJ = d_radB_hist[(size_t)srcTime * N + j];
        if (radJ <= 0.0f) continue;
        // Include area weighting as in original code: min(kij * area, 1) * radJ
        sumB += fminf(kij * d_areas[j], 1.0f) * radJ;
    }

    d_radB_curr[globalI] = rho * sumB + d_radE[(size_t)t * N + globalI];
}

// Compute distances via cross-correlation for triangles [iStart, iEnd)
__global__ void computeDistancesKernel(
    const float* d_radB, int radPitch,
    float* d_distances,
    int N, int T, int iStart, int iEnd, int sourceIdx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (iEnd - iStart)) return;
    int globalI = iStart + i;

    float maxCorr = 0.0f;
    int bestT = 0;

    for (int tau = 0; tau < T; ++tau) {
        float sum = 0.0f;
        for (int tt = tau; tt < T; ++tt) {
            float pB = d_radB[(size_t)tt * radPitch + globalI];
            float pS = d_radB[(size_t)(tt - tau) * radPitch + sourceIdx];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = tau;
        }
    }

    d_distances[globalI] = 0.5f * (float)bestT;
}

// ============================================================================
// GPU Management Functions
// ============================================================================

void initGPUContext(GPUContext& gpu, const Octree& octree,
                    const std::vector<Triangle>& triangles,
                    int N, int T)
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(0));

    // Convert triangles to GPU format
    std::vector<GPUTriangle> hostTris(N);
    for (int i = 0; i < N; ++i) {
        hostTris[i].a = {triangles[i].a.x, triangles[i].a.y, triangles[i].a.z};
        hostTris[i].b = {triangles[i].b.x, triangles[i].b.y, triangles[i].b.z};
        hostTris[i].c = {triangles[i].c.x, triangles[i].c.y, triangles[i].c.z};
        hostTris[i].normal = {triangles[i].normal().x, triangles[i].normal().y, triangles[i].normal().z};
    }

    std::vector<GPUNode> hostNodes;
    std::vector<int> hostLeafTriLists;
    octree.flatten(hostNodes, hostLeafTriLists);
    gpu.numNodes = (int)hostNodes.size();
    gpu.leafTriListSize = (int)hostLeafTriLists.size();

    CUDA_CHECK(cudaMalloc(&gpu.d_triangles, N * sizeof(GPUTriangle)));
    CUDA_CHECK(cudaMalloc(&gpu.d_nodes, hostNodes.size() * sizeof(GPUNode)));
    CUDA_CHECK(cudaMalloc(&gpu.d_leafTriLists, hostLeafTriLists.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gpu.d_triangles, hostTris.data(), N * sizeof(GPUTriangle),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_nodes, hostNodes.data(), hostNodes.size() * sizeof(GPUNode),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_leafTriLists, hostLeafTriLists.data(),
                          hostLeafTriLists.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate Kij matrix with pitch for alignment
    size_t kijPitchBytes;
    CUDA_CHECK(cudaMallocPitch(&gpu.d_kij, &kijPitchBytes, N * sizeof(float), N));
    gpu.kijPitch = (int)(kijPitchBytes / sizeof(float));

    // Allocate tau matrix
    size_t tauPitchBytes;
    CUDA_CHECK(cudaMallocPitch(&gpu.d_tau, &tauPitchBytes, N * sizeof(int), N));
    gpu.tauPitch = (int)(tauPitchBytes / sizeof(int));

    // Allocate radB and radE
    CUDA_CHECK(cudaMalloc(&gpu.d_radB, (size_t)T * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu.d_radE, (size_t)T * N * sizeof(float)));

    // Allocate areas array
    CUDA_CHECK(cudaMalloc(&gpu.d_areas, N * sizeof(float)));
}

void destroyGPUContext(GPUContext& gpu) {
    CUDA_CHECK(cudaFree(gpu.d_triangles));
    CUDA_CHECK(cudaFree(gpu.d_nodes));
    CUDA_CHECK(cudaFree(gpu.d_leafTriLists));
    CUDA_CHECK(cudaFree(gpu.d_kij));
    CUDA_CHECK(cudaFree(gpu.d_tau));
    CUDA_CHECK(cudaFree(gpu.d_radB));
    CUDA_CHECK(cudaFree(gpu.d_radE));
    CUDA_CHECK(cudaFree(gpu.d_areas));
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

    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks parse independently)
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
            if (mpiRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpiRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (mpiRank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("MPI ranks: %d\n", mpiSize);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    // Initialize (rank 0 builds the mesh, then broadcasts to all)
    SimulationState state;
    state.mpiRank = mpiRank;
    state.mpiSize = mpiSize;

    if (mpiRank == 0) {
        initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                             static_cast<size_t>(sourceIdx), reflectivity);
    }

    // Broadcast initialization parameters
    MPI_Bcast(&state.numTriangles, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&state.numTimesteps, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&state.sourceIndex, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&reflectivity, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Non-root ranks allocate state data
    if (mpiRank != 0) {
        state.triangles.resize(state.numTriangles);
        state.areas.resize(state.numTriangles, ZERO);
        state.rho.resize(state.numTriangles, reflectivity);
        state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
        state.tau.resize(state.numTriangles * state.numTriangles, 0);
        state.radE.resize(state.numTimesteps * state.numTriangles, ZERO);
        state.radB.resize(state.numTimesteps * state.numTriangles, ZERO);
        state.distances.resize(state.numTriangles, ZERO);
    }

    // Broadcast triangle data
    MPI_Bcast(&state.triangles[0], (int)(state.numTriangles * sizeof(Triangle)),
              MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&state.rho[0], (int)state.numTriangles, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&state.radE[0], (int)(state.numTimesteps * state.numTriangles),
              MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Non-root ranks need to build octree
    if (mpiRank != 0) {
        state.octree.build(state.triangles);
        for (size_t i = 0; i < state.numTriangles; ++i) {
            state.areas[i] = state.triangles[i].area();
        }
    }

    // Distribute work across ranks
    state.distributeWork();

    // Initialize GPU context on each rank
    initGPUContext(state.gpu, state.octree, state.triangles,
                   (int)state.numTriangles, (int)state.numTimesteps);

    if (mpiRank == 0) printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (mpiRank == 0) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (mpiRank == 0) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (mpiRank == 0) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    // Cleanup GPU resources
    destroyGPUContext(state.gpu);

    // Rank 0: report results
    if (mpiRank == 0) {
        long totalTime = preDuration + simDuration + distDuration;
        printf("Total computation time: %ld ms\n", totalTime);

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

        size_t memKij = n * n * sizeof(val_t);
        size_t memTau = n * n * sizeof(int);
        size_t memRad = 2 * t * n * sizeof(val_t);
        size_t totalMem = memKij + memTau + memRad;
        printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

        uint64_t hash = computeHash(state);
        printf("  Result hash: %016lX\n", hash);
        printf("\n");

        if (printResults) {
            std::vector<double> distData(state.distances.begin(), state.distances.end());
            print_results(distData, "Distances");
        }

        if (validate) {
            if (!validateResults(state)) {
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
