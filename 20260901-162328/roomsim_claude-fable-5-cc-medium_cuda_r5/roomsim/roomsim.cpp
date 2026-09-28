/**
 * Room Response Simulation Benchmark (CUDA-parallelized)
 *
 * GPU implementation of room impulse response simulation using radiosity-based
 * wave propagation. It models how sound/light waves propagate between surfaces
 * in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization:
 *  - Kij / Tau: one GPU thread per triangle pair; visibility rays traverse a
 *    flattened octree with an explicit stack. Monte Carlo sampling uses a
 *    counter-based per-pair RNG so all pairs are independent.
 *  - Wave propagation: one kernel launch per timestep (timesteps are causally
 *    dependent), one thread per receiving triangle.
 *  - Distances: one block per triangle; threads evaluate correlation lags and
 *    reduce to the arg-max (ties resolved to the smallest lag, as in the
 *    sequential code).
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
    __host__ __device__ Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    __host__ __device__ val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    __host__ __device__ bool operator==(const Triangle& o) const {
        return a == o.a && b == o.b && c == o.c;
    }
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
// Octree for Spatial Acceleration (built on host, flattened for GPU traversal)
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

// Flattened octree node for GPU traversal. Leaf nodes have triCount > 0;
// internal nodes reference children by index (-1 for absent children).
struct FlatOctreeNode {
    Vec3 center;
    Vec3 halfExtent;
    int children[8];
    int triStart;
    int triCount;
};

struct FlatOctree {
    std::vector<FlatOctreeNode> nodes;
    std::vector<int> triIndices;

    void build(const Octree& root) {
        nodes.clear();
        triIndices.clear();
        flatten(root);
    }

private:
    int flatten(const Octree& node) {
        int myIdx = static_cast<int>(nodes.size());
        nodes.push_back({});
        FlatOctreeNode flat;
        flat.center = node.center;
        flat.halfExtent = node.halfExtent;
        flat.triStart = static_cast<int>(triIndices.size());
        flat.triCount = static_cast<int>(node.triangleIndices.size());
        for (size_t idx : node.triangleIndices) {
            triIndices.push_back(static_cast<int>(idx));
        }
        for (int i = 0; i < 8; ++i) flat.children[i] = -1;
        nodes[myIdx] = flat;
        for (int i = 0; i < 8; ++i) {
            if (node.children[i]) {
                nodes[myIdx].children[i] = flatten(*node.children[i]);
            }
        }
        return myIdx;
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
// Random Number Generation (device, counter-based per triangle pair)
// ============================================================================

__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Small, fast xorshift64* generator; each triangle pair gets an independent
// stream seeded from the pair index, so all pairs can be sampled in parallel.
struct DeviceRandomGenerator {
    uint64_t state;

    __device__ explicit DeviceRandomGenerator(uint64_t seed) {
        state = splitmix64(seed);
        if (state == 0) state = 0x9e3779b97f4a7c15ULL;
    }

    __device__ val_t rand() {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        uint32_t bits = static_cast<uint32_t>((state * 0x2545F4914F6CDD1DULL) >> 40);
        return static_cast<val_t>(bits) * (1.0f / 16777216.0f);  // [0, 1)
    }
};

// Generate a random point inside a triangle using barycentric coordinates
__device__ Vec3 randomPointInTriangle(const Triangle& t, DeviceRandomGenerator& rng) {
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

__device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                      const Vec3& v0, const Vec3& v1, const Vec3& v2) {
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

constexpr int OCTREE_STACK_SIZE = 64;

// Check if a ray segment intersects a node's bounding box
__device__ bool rayIntersectsBox(const FlatOctreeNode& node, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3& he = node.halfExtent;

    if (fabsf(c.x) > he.x + ad.x) return false;
    if (fabsf(c.y) > he.y + ad.y) return false;
    if (fabsf(c.z) > he.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > he.y * ad.z + he.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > he.z * ad.x + he.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > he.x * ad.y + he.y * ad.x + EPSILON) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const FlatOctreeNode* nodes, const int* triIndices,
                             const Triangle* triangles,
                             int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[OCTREE_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;  // root (root box is not tested, matching the recursive version)

    while (sp > 0) {
        const FlatOctreeNode& node = nodes[stack[--sp]];

        if (node.triCount > 0) {  // leaf: test triangles directly
            for (int k = 0; k < node.triCount; ++k) {
                int idx = triIndices[node.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const Triangle& tri = triangles[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;  // Ray is blocked
                }
            }
        } else {  // internal: descend into children whose boxes the ray hits
            for (int i = 0; i < 8; ++i) {
                int child = node.children[i];
                if (child >= 0 && rayIntersectsBox(nodes[child], from, to)) {
                    stack[sp++] = child;
                }
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor (Kij) and Tau Kernels
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
__device__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// One thread per (i, j) triangle pair
__global__ void formFactorKernel(const Triangle* triangles,
                                 const FlatOctreeNode* nodes, const int* triIndices,
                                 val_t* kij, long long numPairs, int n) {
    long long p = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= numPairs) return;

    int i = static_cast<int>(p / n);
    int j = static_cast<int>(p % n);
    if (i == j) return;  // kij is pre-zeroed

    const Triangle triI = triangles[i];
    const Triangle triJ = triangles[j];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) {
        kij[p] = ZERO;
        return;
    }

    DeviceRandomGenerator rng(0x2A5C3D96ULL + static_cast<uint64_t>(p));

    val_t k = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle(triI, rng);
        Vec3 pJ = randomPointInTriangle(triJ, rng);

        if (isRayBlocked(pI, pJ, nodes, triIndices, triangles, i, j)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.normal());
        val_t cosPhiJ = cosPhi(-v, triJ.normal());

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        k += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kij[p] = k * INV_NUM_RAYS;
}

// One thread per (i, j) triangle pair
__global__ void tauKernel(const Triangle* triangles, int* tau, long long numPairs, int n) {
    long long p = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= numPairs) return;

    int i = static_cast<int>(p / n);
    int j = static_cast<int>(p % n);
    if (i == j) return;  // tau is pre-zeroed

    val_t dist = (triangles[i].center() - triangles[j].center()).norm();
    tau[p] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Simulation Kernel (Wave Propagation, one launch per timestep)
// ============================================================================

__global__ void simulationKernel(const val_t* kij, const int* tau,
                                 const val_t* areas, const val_t* rho,
                                 const val_t* radE, val_t* radB, int n, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const val_t* kijRow = kij + static_cast<long long>(i) * n;
    const int* tauRow = tau + static_cast<long long>(i) * n;

    val_t sumB = ZERO;
    for (int j = 0; j < n; ++j) {
        if (i == j) continue;

        int tauij = tauRow[j];

        // Skip if wave hasn't yet propagated from j to i
        if (t < tauij) continue;

        val_t k = kijRow[j];
        if (k <= ZERO) continue;

        // Get radiosity from source triangle at time when emission occurred
        val_t radJ = radB[static_cast<long long>(t - tauij) * n + j];
        if (radJ <= ZERO) continue;

        // Accumulate contribution: form factor * area * source radiosity
        sumB += fminf(k * areas[j], ONE) * radJ;
    }

    // Update radiosity: reflection + emission
    long long ti = static_cast<long long>(t) * n + i;
    radB[ti] = rho[i] * sumB + radE[ti];
}

// ============================================================================
// Distance Kernel (Cross-Correlation, one block per triangle)
// ============================================================================

constexpr int DIST_BLOCK_SIZE = 256;

__global__ void distanceKernel(const val_t* radB, val_t* distances,
                               int n, int numTimesteps, int sourceIndex) {
    int i = blockIdx.x;
    int tid = threadIdx.x;

    // Each thread evaluates a strided subset of lags; sums run sequentially
    // over tt so per-lag results match the sequential implementation exactly.
    val_t bestCorr = ZERO;
    int bestT = 0;

    for (int t = tid; t < numTimesteps; t += blockDim.x) {
        val_t sum = ZERO;
        for (int tt = t; tt < numTimesteps; ++tt) {
            val_t pB = radB[static_cast<long long>(tt) * n + i];
            val_t pS = radB[static_cast<long long>(tt - t) * n + sourceIndex];
            sum += pS * pB;
        }
        // Within a thread, t increases, so ">" keeps the smallest tied lag
        if (sum > bestCorr) {
            bestCorr = sum;
            bestT = t;
        }
    }

    __shared__ val_t sCorr[DIST_BLOCK_SIZE];
    __shared__ int sT[DIST_BLOCK_SIZE];
    sCorr[tid] = bestCorr;
    sT[tid] = bestT;
    __syncthreads();

    // Reduce to max correlation; ties resolve to the smallest lag, matching
    // the sequential scan's "first strictly greater" update rule.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sCorr[tid + s] > sCorr[tid] ||
                (sCorr[tid + s] == sCorr[tid] && sT[tid + s] < sT[tid])) {
                sCorr[tid] = sCorr[tid + s];
                sT[tid] = sT[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        distances[i] = WAVE_SPEED * static_cast<val_t>(sT[0]);
    }
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct DeviceState {
    Triangle* triangles = nullptr;
    FlatOctreeNode* octreeNodes = nullptr;
    int* octreeTriIndices = nullptr;
    val_t* areas = nullptr;
    val_t* rho = nullptr;
    val_t* kij = nullptr;
    int* tau = nullptr;
    val_t* radE = nullptr;
    val_t* radB = nullptr;
    val_t* distances = nullptr;
};

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
    FlatOctree flatOctree;          // GPU-traversable octree

    DeviceState dev;                // Device-side buffers

    size_t sourceIndex;

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
    state.flatOctree.build(state.octree);

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

    // Upload static data to the GPU
    size_t n = state.numTriangles;
    DeviceState& d = state.dev;
    CUDA_CHECK(cudaMalloc(&d.triangles, n * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&d.octreeNodes, state.flatOctree.nodes.size() * sizeof(FlatOctreeNode)));
    CUDA_CHECK(cudaMalloc(&d.octreeTriIndices,
                          std::max<size_t>(1, state.flatOctree.triIndices.size()) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d.areas, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d.rho, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d.kij, n * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d.tau, n * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d.radE, timesteps * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d.radB, timesteps * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&d.distances, n * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(d.triangles, state.triangles.data(),
                          n * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.octreeNodes, state.flatOctree.nodes.data(),
                          state.flatOctree.nodes.size() * sizeof(FlatOctreeNode),
                          cudaMemcpyHostToDevice));
    if (!state.flatOctree.triIndices.empty()) {
        CUDA_CHECK(cudaMemcpy(d.octreeTriIndices, state.flatOctree.triIndices.data(),
                              state.flatOctree.triIndices.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d.areas, state.areas.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.rho, state.rho.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d.kij, 0, n * n * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(d.tau, 0, n * n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d.radE, state.radE.data(),
                          timesteps * n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d.radB, 0, timesteps * n * sizeof(val_t)));
}

void freeDeviceState(DeviceState& d) {
    cudaFree(d.triangles);
    cudaFree(d.octreeNodes);
    cudaFree(d.octreeTriIndices);
    cudaFree(d.areas);
    cudaFree(d.rho);
    cudaFree(d.kij);
    cudaFree(d.tau);
    cudaFree(d.radE);
    cudaFree(d.radB);
    cudaFree(d.distances);
    d = DeviceState{};
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");

    int n = static_cast<int>(state.numTriangles);
    long long numPairs = static_cast<long long>(n) * n;
    int blockSize = 128;
    long long numBlocks = (numPairs + blockSize - 1) / blockSize;

    formFactorKernel<<<static_cast<unsigned int>(numBlocks), blockSize>>>(
        state.dev.triangles, state.dev.octreeNodes, state.dev.octreeTriIndices,
        state.dev.kij, numPairs, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back for host-side validation
    CUDA_CHECK(cudaMemcpy(state.kij.data(), state.dev.kij,
                          numPairs * sizeof(val_t), cudaMemcpyDeviceToHost));
    printf("  Progress: %zu/%zu triangles\n", state.numTriangles, state.numTriangles);
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    int n = static_cast<int>(state.numTriangles);
    long long numPairs = static_cast<long long>(n) * n;
    int blockSize = 256;
    long long numBlocks = (numPairs + blockSize - 1) / blockSize;

    tauKernel<<<static_cast<unsigned int>(numBlocks), blockSize>>>(
        state.dev.triangles, state.dev.tau, numPairs, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.tau.data(), state.dev.tau,
                          numPairs * sizeof(int), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    int n = static_cast<int>(state.numTriangles);
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Timesteps are causally dependent; each launch parallelizes over triangles
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        simulationKernel<<<numBlocks, blockSize>>>(
            state.dev.kij, state.dev.tau, state.dev.areas, state.dev.rho,
            state.dev.radE, state.dev.radB, n, static_cast<int>(t));
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back for host-side validation
    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.dev.radB,
                          state.numTimesteps * state.numTriangles * sizeof(val_t),
                          cudaMemcpyDeviceToHost));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    int n = static_cast<int>(state.numTriangles);
    distanceKernel<<<n, DIST_BLOCK_SIZE>>>(
        state.dev.radB, state.dev.distances, n,
        static_cast<int>(state.numTimesteps), static_cast<int>(state.sourceIndex));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.dev.distances,
                          n * sizeof(val_t), cudaMemcpyDeviceToHost));
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

    // Initialize CUDA before timing (context creation is one-time setup)
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
    bool ok = true;
    if (validate) {
        ok = validateResults(state);
    }

    freeDeviceState(state.dev);

    return ok ? 0 : 1;
}
