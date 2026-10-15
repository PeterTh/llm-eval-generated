/**
 * Room Response Simulation Benchmark
 * 
 * Hybrid MPI + OpenMP + CUDA implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization:
 *  - MPI: triangle rows are distributed cyclically over ranks (one GPU per rank).
 *    The radiosity row of each timestep is exchanged with MPI_Allgather.
 *  - CUDA: form factors (octree ray casting), the propagation step and the
 *    cross-correlation run on the GPU.
 *  - OpenMP: host-side mesh preprocessing, octree construction and reductions.
 *
 * Form factor sampling uses a counter-based RNG keyed on the (i, j) pair, so the
 * result is independent of the number of ranks/threads/GPUs.
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
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

#define HD __host__ __device__

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

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
    HD val_t norm() const { return std::sqrt(squaredNorm()); }
    HD Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    HD bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
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

        #pragma omp parallel
        #pragma omp single
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

        // Classify triangles per child (independent per child -> OpenMP tasks)
        const bool big = indices.size() > 512;
        for (int i = 0; i < 8; ++i) {
            #pragma omp task default(shared) firstprivate(i) if(big)
            {
                Vec3 childCenter = center;
                childCenter.x += (i & 1) ? childHalfSize.x : -childHalfSize.x;
                childCenter.y += (i & 2) ? childHalfSize.y : -childHalfSize.y;
                childCenter.z += (i & 4) ? childHalfSize.z : -childHalfSize.z;

                for (size_t idx : indices) {
                    if (triangleBoxOverlap(childCenter, childHalfSize, (*allTriangles)[idx])) {
                        childIndices[i].push_back(idx);
                    }
                }
            }
        }
        #pragma omp taskwait

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
                #pragma omp task default(shared) firstprivate(i, childMin, childMax) if(big)
                children[i]->buildNode(childIndices[i], childMin, childMax);
            }
        }
        #pragma omp taskwait
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
// Random Number Generation (counter-based, one independent stream per pair)
// ============================================================================

HD inline uint64_t splitmix64(uint64_t& s) {
    s += 0x9e3779b97f4a7c15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

class RandomGenerator {
    uint64_t state;
    uint32_t cached;
    bool hasCached;
public:
    // Stream is keyed by (seed, stream id) so results don't depend on the decomposition
    HD RandomGenerator(uint32_t seed, uint64_t streamId) : cached(0), hasCached(false) {
        state = (streamId + 1) * 0xd1342543de82ef95ULL ^ (static_cast<uint64_t>(seed) << 32);
        splitmix64(state);
    }
    // Uniform in [0, 1) with 24 random bits
    HD val_t rand() {
        if (!hasCached) {
            uint64_t z = splitmix64(state);
            cached = static_cast<uint32_t>(z >> 40);
            hasCached = true;
            return static_cast<val_t>(static_cast<uint32_t>(z) & 0xFFFFFFu) * (1.0f / 16777216.0f);
        }
        hasCached = false;
        return static_cast<val_t>(cached) * (1.0f / 16777216.0f);
    }
};

// Generate a random point inside a triangle using barycentric coordinates
HD Vec3 randomPointInTriangle(const Triangle& t, RandomGenerator& rng) {
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

HD val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                              const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    const val_t NO_HIT = 3.402823466e+38f;
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (std::abs(det) < EPSILON) return NO_HIT;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return NO_HIT;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return NO_HIT;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Flattened Octree for the GPU
// ============================================================================

struct DevOctNode {
    val_t cx, cy, cz;       // box center
    val_t hx, hy, hz;       // box half extent
    int triStart, triCount; // leaf: range in leafTris (triCount > 0); inner: triCount == 0
    int child[8];           // child node indices or -1
};

constexpr int MAX_STACK = 160;
constexpr int MAX_OCTREE_DEPTH = (MAX_STACK - 1) / 7;

static int flattenOctree(const Octree& n, std::vector<DevOctNode>& nodes,
                         std::vector<int>& leafTris, int depth, int& maxDepth) {
    int me = static_cast<int>(nodes.size());
    nodes.emplace_back();
    maxDepth = std::max(maxDepth, depth);
    DevOctNode d;
    d.cx = n.center.x; d.cy = n.center.y; d.cz = n.center.z;
    d.hx = n.halfExtent.x; d.hy = n.halfExtent.y; d.hz = n.halfExtent.z;
    d.triStart = static_cast<int>(leafTris.size());
    d.triCount = static_cast<int>(n.triangleIndices.size());
    for (size_t idx : n.triangleIndices) leafTris.push_back(static_cast<int>(idx));
    for (int i = 0; i < 8; ++i) d.child[i] = -1;
    if (d.triCount == 0) {
        for (int i = 0; i < 8; ++i) {
            if (n.children[i]) d.child[i] = flattenOctree(*n.children[i], nodes, leafTris, depth + 1, maxDepth);
        }
    }
    nodes[me] = d;
    return me;
}

__device__ __forceinline__ bool rayIntersectsBoxDev(const DevOctNode& nd, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - Vec3(nd.cx, nd.cy, nd.cz);
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > nd.hx + ad.x) return false;
    if (fabsf(c.y) > nd.hy + ad.y) return false;
    if (fabsf(c.z) > nd.hz + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > nd.hy * ad.z + nd.hz * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > nd.hz * ad.x + nd.hx * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > nd.hx * ad.y + nd.hy * ad.x + EPSILON) return false;

    return true;
}

// ============================================================================
// Visibility Testing (Octree-accelerated, device)
// ============================================================================

__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const DevOctNode* nodes, const int* leafTris,
                             const Triangle* tris, int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[MAX_STACK];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const DevOctNode& nd = nodes[stack[--sp]];
        if (nd.triCount > 0) {
            for (int k = 0; k < nd.triCount; ++k) {
                int idx = leafTris[nd.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const Triangle& tri = tris[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int i = 7; i >= 0; --i) {
                int c = nd.child[i];
                if (c >= 0 && rayIntersectsBoxDev(nodes[c], from, to)) stack[sp++] = c;
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
HD val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
__device__ val_t computeKij(int idxI, int idxJ, int N,
                            const Triangle* triangles,
                            const DevOctNode* nodes, const int* leafTris) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    RandomGenerator rng(42, static_cast<uint64_t>(idxI) * static_cast<uint64_t>(N) + idxJ);
    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle(triI, rng);
        Vec3 pJ = randomPointInTriangle(triJ, rng);

        if (isRayBlocked(pI, pJ, nodes, leafTris, triangles, idxI, idxJ)) continue;

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

__device__ __forceinline__ int computeTau(const Vec3& ci, const Vec3& cj) {
    val_t dist = (ci - cj).norm();
    return static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

// Combined propagation table entry: weight = min(Kij * area_j, 1) (0 if Kij <= 0), and delay
struct SimEntry {
    val_t w;
    int tau;
};

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;
    size_t sourceIndex;

    std::vector<Triangle> triangles;
    std::vector<Vec3> centers;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> distances;   // Computed distances from source (all triangles)

    Octree octree;                  // Spatial acceleration structure (host build)

    // MPI decomposition: rank r owns rows i = k * nranks + r
    int rank = 0;
    int nranks = 1;
    size_t nLocal = 0;              // rows owned by this rank
    size_t maxLocal = 0;            // max rows owned by any rank

    // Validation statistics (globally reduced)
    long long nonZeroKij = 0;
    long long receivedEnergy = 0;

    // Device data
    Triangle* d_tris = nullptr;
    Vec3* d_centers = nullptr;
    val_t* d_areas = nullptr;
    val_t* d_rho = nullptr;
    DevOctNode* d_nodes = nullptr;
    int* d_leafTris = nullptr;
    SimEntry* d_table = nullptr;    // nLocal x N
    val_t* d_radB = nullptr;        // T x N (replicated on every rank)

    size_t globalRow(size_t k) const { return k * static_cast<size_t>(nranks) + static_cast<size_t>(rank); }
};

template <typename T>
static T* deviceAlloc(size_t count) {
    T* p = nullptr;
    if (count == 0) count = 1;
    CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
    return p;
}

template <typename T>
static T* deviceUpload(const std::vector<T>& v) {
    T* p = deviceAlloc<T>(v.size());
    if (!v.empty()) CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

static void releaseDevice(SimulationState& s) {
    cudaFree(s.d_tris); cudaFree(s.d_centers); cudaFree(s.d_areas); cudaFree(s.d_rho);
    cudaFree(s.d_nodes); cudaFree(s.d_leafTris); cudaFree(s.d_table); cudaFree(s.d_radB);
}

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

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas / centers
    state.areas.resize(N);
    state.centers.resize(N);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
        state.centers[i] = state.triangles[i].center();
    }

    // Initialize reflectivity
    state.rho.assign(N, reflectivity);
    state.distances.assign(N, ZERO);

    // Row decomposition
    const size_t P = static_cast<size_t>(state.nranks);
    const size_t r = static_cast<size_t>(state.rank);
    state.nLocal = (N > r) ? (N - r + P - 1) / P : 0;
    state.maxLocal = (N + P - 1) / P;

    // Flatten octree and upload everything to the GPU
    std::vector<DevOctNode> nodes;
    std::vector<int> leafTris;
    int maxDepth = 0;
    flattenOctree(state.octree, nodes, leafTris, 0, maxDepth);
    if (maxDepth >= MAX_OCTREE_DEPTH) {
        fprintf(stderr, "Octree too deep for device traversal stack (%d)\n", maxDepth);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    state.d_tris = deviceUpload(state.triangles);
    state.d_centers = deviceUpload(state.centers);
    state.d_areas = deviceUpload(state.areas);
    state.d_rho = deviceUpload(state.rho);
    state.d_nodes = deviceUpload(nodes);
    state.d_leafTris = deviceUpload(leafTris);

    size_t tableBytes = state.nLocal * N * sizeof(SimEntry);
    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    if (tableBytes + timesteps * N * sizeof(val_t) > freeB) {
        fprintf(stderr, "Rank %d: not enough GPU memory (need %.2f GB, free %.2f GB); use more ranks\n",
                state.rank, (tableBytes + timesteps * N * sizeof(val_t)) / 1e9, freeB / 1e9);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    state.d_table = deviceAlloc<SimEntry>(state.nLocal * N);
    state.d_radB = deviceAlloc<val_t>(timesteps * N);
    CUDA_CHECK(cudaMemset(state.d_radB, 0, std::max<size_t>(timesteps * N, 1) * sizeof(val_t)));
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// One thread per (local row, column) pair; computes Kij and Tau and stores the
// propagation weight and delay.
__global__ void formFactorKernel(const Triangle* tris, const Vec3* centers, const val_t* areas,
                                 const DevOctNode* nodes, const int* leafTris,
                                 int N, int P, int rank, size_t rowBegin, size_t rowEnd,
                                 SimEntry* table, unsigned long long* nonZeroCount) {
    __shared__ unsigned int blockNz;
    if (threadIdx.x == 0) blockNz = 0;
    __syncthreads();

    size_t pairs = (rowEnd - rowBegin) * static_cast<size_t>(N);
    size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g < pairs) {
        size_t k = rowBegin + g / N;
        int j = static_cast<int>(g % N);
        int i = static_cast<int>(k * P + rank);

        SimEntry e;
        e.w = ZERO;
        e.tau = 0;
        if (i != j) {
            e.tau = computeTau(centers[i], centers[j]);
            val_t kij = computeKij(i, j, N, tris, nodes, leafTris);
            if (kij > EPSILON) atomicAdd(&blockNz, 1u);
            if (kij > ZERO) e.w = fminf(kij * areas[j], ONE);
        }
        table[k * N + j] = e;
    }
    __syncthreads();
    if (threadIdx.x == 0 && blockNz) atomicAdd(nonZeroCount, static_cast<unsigned long long>(blockNz));
}

void computeFormFactors(SimulationState& state) {
    printf("Computing time delays (Tau) and form factors (Kij)...\n");
    const size_t N = state.numTriangles;

    unsigned long long* d_nz = deviceAlloc<unsigned long long>(1);
    CUDA_CHECK(cudaMemset(d_nz, 0, sizeof(unsigned long long)));

    const int threads = 128;
    const size_t rowsPerChunk = std::max<size_t>(1, (size_t(1) << 22) / N);
    for (size_t r0 = 0; r0 < state.nLocal; r0 += rowsPerChunk) {
        size_t r1 = std::min(state.nLocal, r0 + rowsPerChunk);
        size_t pairs = (r1 - r0) * N;
        size_t blocks = (pairs + threads - 1) / threads;
        formFactorKernel<<<static_cast<unsigned int>(blocks), threads>>>(
            state.d_tris, state.d_centers, state.d_areas, state.d_nodes, state.d_leafTris,
            static_cast<int>(N), state.nranks, state.rank, r0, r1, state.d_table, d_nz);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long localNz = 0;
    CUDA_CHECK(cudaMemcpy(&localNz, d_nz, sizeof(localNz), cudaMemcpyDeviceToHost));
    cudaFree(d_nz);
    long long gl = static_cast<long long>(localNz);
    MPI_Allreduce(&gl, &state.nonZeroKij, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    printf("  Progress: %zu/%zu triangles\n", N, N);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// One warp per owned row; lanes stride over the columns j.
__global__ void propagateKernel(const SimEntry* table, const val_t* radB, const val_t* rho,
                                val_t* out, int N, int nLocal, int P, int rank,
                                int t, int sourceIdx, bool emitting) {
    int warp = static_cast<int>((static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5);
    int lane = threadIdx.x & 31;
    if (warp >= nLocal) return;

    const SimEntry* row = table + static_cast<size_t>(warp) * N;
    val_t sumB = ZERO;
    for (int j = lane; j < N; j += 32) {
        SimEntry e = row[j];
        if (e.w <= ZERO || t < e.tau) continue;
        val_t radJ = radB[static_cast<size_t>(t - e.tau) * N + j];
        if (radJ <= ZERO) continue;
        sumB += e.w * radJ;
    }
    for (int off = 16; off > 0; off >>= 1) sumB += __shfl_down_sync(0xffffffffu, sumB, off);

    if (lane == 0) {
        int i = warp * P + rank;
        val_t emit = (emitting && i == sourceIdx) ? ONE : ZERO;
        out[warp] = rho[i] * sumB + emit;
    }
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const size_t N = state.numTriangles;
    const size_t P = static_cast<size_t>(state.nranks);
    const size_t timeOff = state.numTimesteps / 2;

    val_t* d_out = deviceAlloc<val_t>(state.maxLocal);
    std::vector<val_t> sendBuf(state.maxLocal, ZERO);
    std::vector<val_t> recvBuf(state.maxLocal * P, ZERO);
    std::vector<val_t> row(N, ZERO);

    const int threads = 128;
    const int warpsPerBlock = threads / 32;
    unsigned int blocks = static_cast<unsigned int>((state.nLocal + warpsPerBlock - 1) / warpsPerBlock);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (blocks > 0) {
            propagateKernel<<<blocks, threads>>>(state.d_table, state.d_radB, state.d_rho, d_out,
                                                 static_cast<int>(N), static_cast<int>(state.nLocal),
                                                 state.nranks, state.rank, static_cast<int>(t),
                                                 static_cast<int>(state.sourceIndex), t < timeOff);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(sendBuf.data(), d_out, state.nLocal * sizeof(val_t), cudaMemcpyDeviceToHost));
        }

        // Exchange this timestep's radiosity row between all ranks
        MPI_Allgather(sendBuf.data(), static_cast<int>(state.maxLocal), MPI_FLOAT,
                      recvBuf.data(), static_cast<int>(state.maxLocal), MPI_FLOAT, MPI_COMM_WORLD);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            row[i] = recvBuf[(i % P) * state.maxLocal + i / P];
        }
        CUDA_CHECK(cudaMemcpy(state.d_radB + t * N, row.data(), N * sizeof(val_t), cudaMemcpyHostToDevice));

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    cudaFree(d_out);
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

// Gather the time series (column) of each owned triangle and of the source.
__global__ void extractColumnsKernel(const val_t* radB, val_t* cols, val_t* srcCol,
                                     int N, int T, int nLocal, int P, int rank, int sourceIdx) {
    size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(nLocal) * T;
    if (g < total) {
        int t = static_cast<int>(g % T);
        int k = static_cast<int>(g / T);
        cols[g] = radB[static_cast<size_t>(t) * N + static_cast<size_t>(k) * P + rank];
    }
    if (g < static_cast<size_t>(T)) srcCol[g] = radB[g * N + sourceIdx];
}

// One block per owned triangle; threads stride over lags. Summation order per lag
// matches the sequential algorithm; ties resolve to the smallest lag.
__global__ void crossCorrelationKernel(const val_t* cols, const val_t* srcCol, int T,
                                       val_t* dist, int* gotEnergy) {
    __shared__ val_t sVal[256];
    __shared__ int sLag[256];
    const val_t* pB = cols + static_cast<size_t>(blockIdx.x) * T;

    val_t best = ZERO;
    int bestT = INT_MAX;
    int energy = 0;
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        if (pB[t] > EPSILON) energy = 1;
        val_t sum = ZERO;
        for (int tt = t; tt < T; ++tt) sum += srcCol[tt - t] * pB[tt];
        if (sum > best) {
            best = sum;
            bestT = t;
        }
    }
    int anyEnergy = __syncthreads_or(energy);

    sVal[threadIdx.x] = best;
    sLag[threadIdx.x] = bestT;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            val_t ov = sVal[threadIdx.x + s];
            int ol = sLag[threadIdx.x + s];
            if (ov > sVal[threadIdx.x] || (ov == sVal[threadIdx.x] && ol < sLag[threadIdx.x])) {
                sVal[threadIdx.x] = ov;
                sLag[threadIdx.x] = ol;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        int lag = (sLag[0] == INT_MAX) ? 0 : sLag[0];
        dist[blockIdx.x] = WAVE_SPEED * static_cast<val_t>(lag);
        gotEnergy[blockIdx.x] = anyEnergy;
    }
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    const size_t P = static_cast<size_t>(state.nranks);

    std::vector<val_t> sendBuf(state.maxLocal, ZERO);
    std::vector<int> energyLocal(std::max<size_t>(state.nLocal, 1), 0);

    if (state.nLocal > 0 && T > 0) {
        val_t* d_cols = deviceAlloc<val_t>(state.nLocal * T);
        val_t* d_src = deviceAlloc<val_t>(T);
        val_t* d_dist = deviceAlloc<val_t>(state.nLocal);
        int* d_energy = deviceAlloc<int>(state.nLocal);

        size_t total = std::max(state.nLocal * T, T);
        extractColumnsKernel<<<static_cast<unsigned int>((total + 255) / 256), 256>>>(
            state.d_radB, d_cols, d_src, static_cast<int>(N), static_cast<int>(T),
            static_cast<int>(state.nLocal), state.nranks, state.rank, static_cast<int>(state.sourceIndex));
        crossCorrelationKernel<<<static_cast<unsigned int>(state.nLocal), 256>>>(
            d_cols, d_src, static_cast<int>(T), d_dist, d_energy);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(sendBuf.data(), d_dist, state.nLocal * sizeof(val_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(energyLocal.data(), d_energy, state.nLocal * sizeof(int), cudaMemcpyDeviceToHost));
        cudaFree(d_cols); cudaFree(d_src); cudaFree(d_dist); cudaFree(d_energy);
    }

    std::vector<val_t> recvBuf(state.maxLocal * P, ZERO);
    MPI_Allgather(sendBuf.data(), static_cast<int>(state.maxLocal), MPI_FLOAT,
                  recvBuf.data(), static_cast<int>(state.maxLocal), MPI_FLOAT, MPI_COMM_WORLD);
    for (size_t i = 0; i < N; ++i) {
        state.distances[i] = recvBuf[(i % P) * state.maxLocal + i / P];
    }

    long long localEnergy = 0;
    if (T > 0) {
        for (size_t k = 0; k < state.nLocal; ++k) localEnergy += energyLocal[k];
    }
    MPI_Allreduce(&localEnergy, &state.receivedEnergy, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
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
    printf("  Triangles receiving energy: %lld/%zu\n", state.receivedEnergy, state.numTriangles);

    if (state.receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    size_t total = state.numTriangles * state.numTriangles;
    printf("  Non-zero form factors: %lld/%zu (%.2f%%)\n",
           state.nonZeroKij, total,
           100.0f * state.nonZeroKij / static_cast<val_t>(total));

    if (state.nonZeroKij == 0) {
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
    #pragma omp parallel for reduction(^ : hash) schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        uint32_t bits;
        memcpy(&bits, &state.distances[i], sizeof(bits));
        hash ^= (static_cast<uint64_t>(bits) + i) * 0x9e3779b97f4a7c15ULL;
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

static long elapsedMs(double t0, double t1) {
    return static_cast<long>((t1 - t0) * 1000.0);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    SimulationState state;
    MPI_Comm_rank(MPI_COMM_WORLD, &state.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &state.nranks);

    // Only rank 0 writes to stdout
    if (state.rank != 0) {
        if (!freopen("/dev/null", "w", stdout)) return 1;
    }

    // One GPU per rank (round-robin over the GPUs visible on the node)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, state.rank, MPI_INFO_NULL, &local);
        int localRank = 0;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_free(&local);
        int numDevices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&numDevices));
        if (numDevices == 0) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % numDevices));
        CUDA_CHECK(cudaFree(0));
    }

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
            MPI_Finalize();
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
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
    printf("Parallelization: %d MPI ranks x %d OpenMP threads, CUDA\n", state.nranks, omp_get_max_threads());
    printf("\n");

    // Initialize
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    double startPre = MPI_Wtime();

    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    double endPre = MPI_Wtime();
    long preDuration = elapsedMs(startPre, endPre);

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    double startSim = MPI_Wtime();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    double endSim = MPI_Wtime();
    long simDuration = elapsedMs(startSim, endSim);

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    double startDist = MPI_Wtime();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    double endDist = MPI_Wtime();
    long distDuration = elapsedMs(startDist, endDist);

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
    int status = 0;
    if (validate) {
        if (!validateResults(state)) {
            status = 1;
        }
    }

    releaseDevice(state);
    fflush(stdout);
    MPI_Finalize();
    return status;
}
