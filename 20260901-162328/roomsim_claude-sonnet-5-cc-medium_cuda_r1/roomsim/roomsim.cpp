/**
 * Room Response Simulation Benchmark
 *
 * This is a CUDA-accelerated implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 * The heavy O(N^2) and O(N^2*T) / O(N*T^2) computations (form factor
 * estimation, time-delay computation, wave propagation and cross-correlation
 * distance estimation) are all executed as CUDA kernels on the GPU. Mesh
 * generation and octree construction remain on the host (they are cheap
 * relative to the rest of the pipeline) and the resulting octree is flattened
 * into GPU-friendly arrays for device-side visibility traversal.
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

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err__));                                 \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

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

constexpr int GPU_BLOCK_SIZE = 256;

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
// Octree for Spatial Acceleration (built on host)
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
// Flattened Octree (GPU-friendly representation)
// ============================================================================

struct GPUOctreeNode {
    Vec3 center;
    Vec3 halfExtent;
    int children[8];  // -1 if absent
    int leafStart;     // >= 0 if this is a leaf (index into leafTriIndices)
    int leafCount;
};

struct FlatOctree {
    std::vector<GPUOctreeNode> nodes;
    std::vector<int> leafTriIndices;
};

// Recursively flatten the host-built octree into a flat array representation
// suitable for iterative traversal on the GPU. Returns the index of the node.
int flattenOctreeNode(const Octree& node, FlatOctree& flat) {
    int myIndex = static_cast<int>(flat.nodes.size());
    flat.nodes.emplace_back();

    GPUOctreeNode gnode;
    gnode.center = node.center;
    gnode.halfExtent = node.halfExtent;
    for (int i = 0; i < 8; ++i) gnode.children[i] = -1;

    if (!node.triangleIndices.empty()) {
        gnode.leafStart = static_cast<int>(flat.leafTriIndices.size());
        gnode.leafCount = static_cast<int>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) {
            flat.leafTriIndices.push_back(static_cast<int>(idx));
        }
    } else {
        gnode.leafStart = -1;
        gnode.leafCount = 0;
        for (int i = 0; i < 8; ++i) {
            if (node.children[i]) {
                gnode.children[i] = flattenOctreeNode(*node.children[i], flat);
            }
        }
    }

    flat.nodes[myIndex] = gnode;
    return myIndex;
}

// ============================================================================
// Mesh Generation: Icosphere (host)
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
// GPU-side deterministic RNG (splitmix64-based, independent per (i,j) pair)
//
// The original sequential implementation drew samples from a single shared
// std::mt19937 stream. Reproducing that exact global ordering is fundamentally
// incompatible with data-parallel execution (each (i,j) form-factor pair must
// be computable independently, on any thread, in any order). Instead each
// pair gets its own well-mixed, deterministic stream derived from (i, j) and
// a fixed global seed, so the Monte Carlo estimator remains a faithful,
// reproducible implementation of the same algorithm.
// ============================================================================

__device__ __forceinline__ uint64_t splitmix64Next(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ val_t randFloat01(uint64_t& state) {
    uint64_t x = splitmix64Next(state);
    // 24 bits of mantissa precision mapped to [0, 1)
    return static_cast<val_t>((x >> 40) * (1.0 / (1ULL << 24)));
}

__device__ __forceinline__ uint64_t pairSeed(uint64_t globalSeed, size_t i, size_t j) {
    uint64_t state = globalSeed ^ (static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ULL) ^
                      (static_cast<uint64_t>(j) * 0xC2B2AE3D27D4EB4FULL);
    return splitmix64Next(state);
}

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

__host__ __device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                                const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return std::numeric_limits<val_t>::max();

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
// Form Factor helpers
// ============================================================================

__host__ __device__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// ============================================================================
// GPU Visibility Testing (iterative octree traversal)
// ============================================================================

__device__ bool rayIntersectsBoxGPU(const Vec3& center, const Vec3& halfExtent,
                                     const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - center;
    Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));

    if (fabsf(c.x) > halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle,
// using an iterative (stack-based) traversal of the flattened octree.
__device__ bool isRayBlockedGPU(const Vec3& from, const Vec3& to,
                                 const GPUOctreeNode* nodes, const int* leafTriIndices,
                                 const Vec3* triA, const Vec3* triB, const Vec3* triC,
                                 size_t srcTriIdx, size_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    constexpr int MAX_STACK = 64;
    int stack[MAX_STACK];
    int sp = 0;
    stack[sp++] = 0;  // root node index

    while (sp > 0) {
        int ni = stack[--sp];
        const GPUOctreeNode& node = nodes[ni];

        if (!rayIntersectsBoxGPU(node.center, node.halfExtent, from, to)) continue;

        if (node.leafStart >= 0) {
            for (int k = 0; k < node.leafCount; ++k) {
                size_t idx = static_cast<size_t>(leafTriIndices[node.leafStart + k]);
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                val_t dist = rayTriangleIntersect(from, dirNorm, triA[idx], triB[idx], triC[idx]);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;
                }
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                if (node.children[c] >= 0 && sp < MAX_STACK) {
                    stack[sp++] = node.children[c];
                }
            }
        }
    }
    return false;
}

// ============================================================================
// GPU Kernels
// ============================================================================

// Compute form factor Kij for every ordered pair (i, j), i != j.
__global__ void computeFormFactorsKernel(size_t numTriangles, uint64_t seed,
                                          const Vec3* triA, const Vec3* triB, const Vec3* triC,
                                          const Vec3* triNormal,
                                          const GPUOctreeNode* nodes, const int* leafTriIndices,
                                          val_t* kij) {
    size_t pairIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = numTriangles * numTriangles;
    if (pairIdx >= total) return;

    size_t i = pairIdx / numTriangles;
    size_t j = pairIdx % numTriangles;
    if (i == j) {
        kij[pairIdx] = ZERO;
        return;
    }

    Vec3 nI = triNormal[i];
    Vec3 nJ = triNormal[j];
    if (nI.dot(nJ) > 0.99f) {
        kij[pairIdx] = ZERO;
        return;
    }

    Vec3 aI = triA[i], bI = triB[i], cI = triC[i];
    Vec3 aJ = triA[j], bJ = triB[j], cJ = triC[j];

    uint64_t state = pairSeed(seed, i, j);

    val_t kijVal = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        val_t u1 = randFloat01(state);
        val_t v1 = randFloat01(state);
        if (u1 + v1 > 1.0f) {
            u1 = 1.0f - u1;
            v1 = 1.0f - v1;
        }
        Vec3 pI = aI + (bI - aI) * u1 + (cI - aI) * v1;

        val_t u2 = randFloat01(state);
        val_t v2 = randFloat01(state);
        if (u2 + v2 > 1.0f) {
            u2 = 1.0f - u2;
            v2 = 1.0f - v2;
        }
        Vec3 pJ = aJ + (bJ - aJ) * u2 + (cJ - aJ) * v2;

        if (isRayBlockedGPU(pI, pJ, nodes, leafTriIndices, triA, triB, triC, i, j)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, nI);
        val_t cosPhiJ = cosPhi(-v, nJ);
        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kijVal += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kij[pairIdx] = kijVal * INV_NUM_RAYS;
}

// Compute time delay Tau for every ordered pair (i, j), i != j.
__global__ void computeTauKernel(size_t numTriangles, const Vec3* triCenter, int* tau) {
    size_t pairIdx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = numTriangles * numTriangles;
    if (pairIdx >= total) return;

    size_t i = pairIdx / numTriangles;
    size_t j = pairIdx % numTriangles;
    if (i == j) {
        tau[pairIdx] = 0;
        return;
    }

    val_t dist = (triCenter[i] - triCenter[j]).norm();
    tau[pairIdx] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// One wave-propagation timestep: each thread owns one receiving triangle i
// and accumulates contributions from all emitting triangles j, exactly as
// the sequential reference does (same summation order per i).
__global__ void simulationStepKernel(size_t numTriangles, size_t t,
                                      const val_t* kij, const int* tau,
                                      const val_t* areas, const val_t* rho,
                                      const val_t* radE, val_t* radB) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numTriangles) return;

    val_t sumB = ZERO;
    for (size_t j = 0; j < numTriangles; ++j) {
        if (i == j) continue;

        int tauij = tau[i * numTriangles + j];
        if (static_cast<long long>(t) < tauij) continue;

        val_t kijVal = kij[i * numTriangles + j];
        if (kijVal <= ZERO) continue;

        size_t srcTime = t - static_cast<size_t>(tauij);
        val_t radJ = radB[srcTime * numTriangles + j];
        if (radJ <= ZERO) continue;

        sumB += fminf(kijVal * areas[j], ONE) * radJ;
    }

    radB[t * numTriangles + i] = rho[i] * sumB + radE[t * numTriangles + i];
}

// Discrete cross-correlation distance estimation: each thread owns one
// triangle i and performs the same nested (t, tt) reduction as the reference.
__global__ void computeDistancesKernel(size_t numTriangles, size_t numTimesteps,
                                        size_t sourceIndex, const val_t* radB,
                                        val_t* distances) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numTriangles) return;

    val_t maxCorr = ZERO;
    int bestT = 0;

    for (size_t t = 0; t < numTimesteps; ++t) {
        val_t sum = ZERO;
        for (size_t tt = t; tt < numTimesteps; ++tt) {
            val_t pB = radB[tt * numTriangles + i];
            val_t pS = radB[(tt - t) * numTriangles + sourceIndex];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }

    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
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
    std::vector<val_t> kij;         // Form factors (N x N matrix, row-major) - host copy for validation
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix) - host copy for validation
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure (host)

    size_t sourceIndex;

    // Persistent device buffers (allocated once, reused across phases)
    Vec3* d_triA = nullptr;
    Vec3* d_triB = nullptr;
    Vec3* d_triC = nullptr;
    Vec3* d_triNormal = nullptr;
    Vec3* d_triCenter = nullptr;
    val_t* d_areas = nullptr;
    val_t* d_rho = nullptr;
    GPUOctreeNode* d_octreeNodes = nullptr;
    int* d_leafTriIndices = nullptr;
    val_t* d_kij = nullptr;
    int* d_tau = nullptr;
    val_t* d_radE = nullptr;
    val_t* d_radB = nullptr;
    val_t* d_distances = nullptr;

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

    // Initialize host-side matrices (only populated from GPU when validation is requested)
    state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
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
// GPU Buffer Setup / Teardown
// ============================================================================

void initGPUBuffers(SimulationState& state) {
    const size_t n = state.numTriangles;

    std::vector<Vec3> hA(n), hB(n), hC(n), hNormal(n), hCenter(n);
    for (size_t i = 0; i < n; ++i) {
        hA[i] = state.triangles[i].a;
        hB[i] = state.triangles[i].b;
        hC[i] = state.triangles[i].c;
        hNormal[i] = state.triangles[i].normal();
        hCenter[i] = state.triangles[i].center();
    }

    CUDA_CHECK(cudaMalloc(&state.d_triA, n * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&state.d_triB, n * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&state.d_triC, n * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&state.d_triNormal, n * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&state.d_triCenter, n * sizeof(Vec3)));
    CUDA_CHECK(cudaMemcpy(state.d_triA, hA.data(), n * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_triB, hB.data(), n * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_triC, hC.data(), n * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_triNormal, hNormal.data(), n * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_triCenter, hCenter.data(), n * sizeof(Vec3), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_areas, n * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_rho, n * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));

    // Flatten and upload octree
    FlatOctree flat;
    flattenOctreeNode(state.octree, flat);

    CUDA_CHECK(cudaMalloc(&state.d_octreeNodes, flat.nodes.size() * sizeof(GPUOctreeNode)));
    CUDA_CHECK(cudaMemcpy(state.d_octreeNodes, flat.nodes.data(),
                           flat.nodes.size() * sizeof(GPUOctreeNode), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_leafTriIndices, flat.leafTriIndices.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(state.d_leafTriIndices, flat.leafTriIndices.data(),
                           flat.leafTriIndices.size() * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_kij, n * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_tau, n * n * sizeof(int)));

    const size_t radElems = state.numTimesteps * n;
    CUDA_CHECK(cudaMalloc(&state.d_radE, radElems * sizeof(val_t)));
    CUDA_CHECK(cudaMemcpy(state.d_radE, state.radE.data(), radElems * sizeof(val_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&state.d_radB, radElems * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, radElems * sizeof(val_t)));

    CUDA_CHECK(cudaMalloc(&state.d_distances, n * sizeof(val_t)));
}

void freeGPUBuffers(SimulationState& state) {
    cudaFree(state.d_triA);
    cudaFree(state.d_triB);
    cudaFree(state.d_triC);
    cudaFree(state.d_triNormal);
    cudaFree(state.d_triCenter);
    cudaFree(state.d_areas);
    cudaFree(state.d_rho);
    cudaFree(state.d_octreeNodes);
    cudaFree(state.d_leafTriIndices);
    cudaFree(state.d_kij);
    cudaFree(state.d_tau);
    cudaFree(state.d_radE);
    cudaFree(state.d_radB);
    cudaFree(state.d_distances);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state, bool keepHostCopy) {
    printf("Computing form factors (Kij) on GPU...\n");
    const size_t n = state.numTriangles;
    const size_t total = n * n;
    const size_t numBlocks = (total + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;

    constexpr uint64_t GLOBAL_SEED = 42;

    computeFormFactorsKernel<<<numBlocks, GPU_BLOCK_SIZE>>>(
        n, GLOBAL_SEED, state.d_triA, state.d_triB, state.d_triC, state.d_triNormal,
        state.d_octreeNodes, state.d_leafTriIndices, state.d_kij);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (keepHostCopy) {
        CUDA_CHECK(cudaMemcpy(state.kij.data(), state.d_kij, total * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau) on GPU...\n");
    const size_t n = state.numTriangles;
    const size_t total = n * n;
    const size_t numBlocks = (total + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;

    computeTauKernel<<<numBlocks, GPU_BLOCK_SIZE>>>(n, state.d_triCenter, state.d_tau);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state, bool keepHostCopy) {
    printf("Running wave propagation simulation on GPU...\n");
    const size_t n = state.numTriangles;
    const size_t numBlocks = (n + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        simulationStepKernel<<<numBlocks, GPU_BLOCK_SIZE>>>(
            n, t, state.d_kij, state.d_tau, state.d_areas, state.d_rho, state.d_radE, state.d_radB);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (keepHostCopy) {
        CUDA_CHECK(cudaMemcpy(state.radB.data(), state.d_radB,
                               state.numTimesteps * n * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation on GPU...\n");
    const size_t n = state.numTriangles;
    const size_t numBlocks = (n + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;

    computeDistancesKernel<<<numBlocks, GPU_BLOCK_SIZE>>>(
        n, state.numTimesteps, state.sourceIndex, state.d_radB, state.d_distances);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.d_distances, n * sizeof(val_t), cudaMemcpyDeviceToHost));
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

    printf("Room Response Simulation Benchmark (CUDA)\n");
    printf("===========================================\n");
    printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    printf("Timesteps: %d\n", timesteps);
    printf("Source triangle: %d\n", sourceIdx);
    printf("Reflectivity: %.2f\n", reflectivity);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable device found\n");
        return 1;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU device: %s\n", prop.name);
    printf("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    initGPUBuffers(state);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state, validate);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state, validate);

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
    bool validationFailed = false;
    if (validate) {
        if (!validateResults(state)) {
            validationFailed = true;
        }
    }

    freeGPUBuffers(state);

    if (validationFailed) {
        return 1;
    }

    return 0;
}
