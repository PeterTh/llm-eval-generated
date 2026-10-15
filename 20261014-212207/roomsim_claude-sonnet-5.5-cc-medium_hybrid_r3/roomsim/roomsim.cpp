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

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define HD __host__ __device__

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,      \
                    __LINE__, cudaGetErrorString(err_));                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
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

// Flattened octree node for GPU traversal
struct GpuNode {
    float cx, cy, cz;       // box center
    float hx, hy, hz;       // box half extent
    int child[8];           // child node indices (-1 if absent)
    int triStart, triCount; // leaf triangle range (triCount > 0 for leaves)
};

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

        // Overlap masks are computed in parallel for large nodes; the per-child
        // index lists are then filled serially to keep the original ordering.
        std::vector<uint8_t> masks(indices.size());
        #pragma omp parallel for schedule(static) if (indices.size() > 2048)
        for (size_t k = 0; k < indices.size(); ++k) {
            const Triangle& tri = (*allTriangles)[indices[k]];
            uint8_t m = 0;
            for (int i = 0; i < 8; ++i) {
                Vec3 childCenter = center;
                childCenter.x += (i & 1) ? childHalfSize.x : -childHalfSize.x;
                childCenter.y += (i & 2) ? childHalfSize.y : -childHalfSize.y;
                childCenter.z += (i & 4) ? childHalfSize.z : -childHalfSize.z;

                if (triangleBoxOverlap(childCenter, childHalfSize, tri)) m |= (1u << i);
            }
            masks[k] = m;
        }
        for (size_t k = 0; k < indices.size(); ++k) {
            for (int i = 0; i < 8; ++i) {
                if (masks[k] & (1u << i)) childIndices[i].push_back(indices[k]);
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
    // Append this subtree to flat arrays (root ends up at index 0); returns node index
    int flatten(std::vector<GpuNode>& nodes, std::vector<int>& tris, int depth, int& maxDepth) const {
        maxDepth = std::max(maxDepth, depth);
        int me = static_cast<int>(nodes.size());
        nodes.push_back(GpuNode{});
        GpuNode n{};
        n.cx = center.x; n.cy = center.y; n.cz = center.z;
        n.hx = halfExtent.x; n.hy = halfExtent.y; n.hz = halfExtent.z;
        n.triStart = static_cast<int>(tris.size());
        n.triCount = static_cast<int>(triangleIndices.size());
        for (size_t idx : triangleIndices) tris.push_back(static_cast<int>(idx));
        for (int i = 0; i < 8; ++i) {
            n.child[i] = -1;
            if (children[i] && triangleIndices.empty()) {
                n.child[i] = children[i]->flatten(nodes, tris, depth + 1, maxDepth);
            }
        }
        nodes[me] = n;
        return me;
    }

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
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

HD val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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

// Compute the cosine of angle between vector and triangle normal
HD val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// ============================================================================
// Simulation State
// ============================================================================

// Rows of the N x N matrices (receiver triangles i) are distributed over MPI
// ranks in cyclic blocks of ROW_BLOCK rows. Each rank drives one GPU.
constexpr size_t ROW_BLOCK = 32;
constexpr int MAX_STACK = 160;
constexpr uint64_t RNG_SEED = 42;

struct MpiInfo {
    int rank = 0;
    int size = 1;
};

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix, filled on request)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure (host build)
    std::vector<GpuNode> nodes;     // Flattened octree
    std::vector<int> nodeTris;
    int octreeDepth = 0;

    size_t sourceIndex;
    unsigned long long nonZeroKij = 0;  // Global count of Kij > EPSILON (validation only)

    // MPI row distribution
    MpiInfo mpi;
    std::vector<int> myRows;        // global row index of each local row
    std::vector<int> counts, displs;// per-rank row counts / displacements (gather layout)
    std::vector<int> perm;          // gather position -> global row index

    // Device data
    float4* dCent = nullptr;        // triangle centers (w = area)
    Triangle* dTris = nullptr;
    GpuNode* dNodes = nullptr;
    int* dNodeTris = nullptr;
    int* dRows = nullptr;
    val_t* dKT = nullptr;           // Kij transposed: [j][local row]
    val_t* dRadB = nullptr;         // T x N
    val_t* dRadE = nullptr;         // T x N
    val_t* dRho = nullptr;          // N
    val_t* dLocal = nullptr;        // rows (scratch for gathers)

    size_t localRows() const { return myRows.size(); }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// GPU kernels
// ============================================================================

// Counter-based RNG (splitmix64 finalizer): every (i, j, ray) gets its own
// independent random numbers, so the form factors do not depend on how the
// work is distributed.
__device__ __forceinline__ uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Generate a random point inside a triangle using barycentric coordinates
__device__ __forceinline__ Vec3 randomPointInTriangle(const Triangle& t, uint64_t key) {
    uint64_t h = mix64(key);
    val_t u = static_cast<val_t>(h >> 40) * (1.0f / 16777216.0f);
    val_t v = static_cast<val_t>((h >> 16) & 0xFFFFFFu) * (1.0f / 16777216.0f);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

__device__ __forceinline__ bool rayIntersectsBox(const GpuNode& n, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - Vec3(n.cx, n.cy, n.cz);
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > n.hx + ad.x) return false;
    if (fabsf(c.y) > n.hy + ad.y) return false;
    if (fabsf(c.z) > n.hz + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > n.hy * ad.z + n.hz * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > n.hz * ad.x + n.hx * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > n.hx * ad.y + n.hy * ad.x + EPSILON) return false;

    return true;
}

// Check if a ray between two triangles is blocked by any other triangle
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const GpuNode* __restrict__ nodes,
                             const int* __restrict__ nodeTris,
                             const Triangle* __restrict__ tris,
                             int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[MAX_STACK];
    int sp = 0;
    stack[sp++] = 0;  // root is visited without a box test (as in the original)
    while (sp > 0) {
        const GpuNode& n = nodes[stack[--sp]];
        if (n.triCount > 0) {
            for (int k = 0; k < n.triCount; ++k) {
                int idx = nodeTris[n.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const Triangle& tri = tris[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int i = 7; i >= 0; --i) {
                int c = n.child[i];
                if (c >= 0 && rayIntersectsBox(nodes[c], from, to)) stack[sp++] = c;
            }
        }
    }
    return false;
}

// One thread per (receiver i, emitter j) pair; consecutive threads handle
// consecutive receivers (spatially coherent, coalesced stores into KT[j][il]).
__global__ void formFactorKernel(val_t* __restrict__ KT, const int* __restrict__ rows, size_t numRows,
                                 size_t N, size_t j0, size_t count,
                                 const GpuNode* __restrict__ nodes, const int* __restrict__ nodeTris,
                                 const Triangle* __restrict__ tris) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    size_t il = idx % numRows;
    size_t j = j0 + idx / numRows;
    size_t i = static_cast<size_t>(rows[il]);
    val_t* out = KT + j * numRows + il;

    if (i == j) { *out = ZERO; return; }

    const Triangle triI = tris[i];
    const Triangle triJ = tris[j];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) { *out = ZERO; return; }

    val_t kij = ZERO;
    uint64_t base = (static_cast<uint64_t>(i) * N + j) * (2 * NUM_RAYS) + RNG_SEED * 0x9e3779b97f4a7c15ULL;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle(triI, base + 2 * r);
        Vec3 pJ = randomPointInTriangle(triJ, base + 2 * r + 1);

        if (isRayBlocked(pI, pJ, nodes, nodeTris, tris, static_cast<int>(i), static_cast<int>(j))) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.normal());
        val_t cosPhiJ = cosPhi(-v, triJ.normal());

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    *out = kij * INV_NUM_RAYS;
}

// One wave propagation step for the local rows. Each thread accumulates its
// row sequentially over j (same summation order as the sequential code).
__global__ void propagateKernel(size_t t, size_t N, size_t numRows, const int* __restrict__ rows,
                                const val_t* __restrict__ KT, const float4* __restrict__ cent,
                                const val_t* __restrict__ rho, const val_t* __restrict__ radE,
                                val_t* __restrict__ radB, val_t* __restrict__ local) {
    size_t il = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (il >= numRows) return;
    size_t i = static_cast<size_t>(rows[il]);
    float4 ci = cent[i];
    val_t sumB = ZERO;

    #pragma unroll 4
    for (size_t j = 0; j < N; ++j) {
        val_t kij = KT[j * numRows + il];
        if (kij <= ZERO || j == i) continue;

        float4 cj = cent[j];
        Vec3 diff = Vec3(ci.x, ci.y, ci.z) - Vec3(cj.x, cj.y, cj.z);
        int tauij = static_cast<int>(ceilf(diff.norm() * INV_WAVE_SPEED));

        // Skip if wave hasn't yet propagated from j to i
        if (static_cast<int>(t) < tauij) continue;

        size_t srcTime = t - static_cast<size_t>(tauij);
        val_t radJ = radB[srcTime * N + j];
        if (radJ <= ZERO) continue;

        sumB += fminf(kij * cj.w, ONE) * radJ;
    }

    val_t result = rho[i] * sumB + radE[t * N + i];
    radB[t * N + i] = result;
    local[il] = result;
}

// Discrete cross-correlation of each local row with the source signal
__global__ void distanceKernel(size_t N, size_t T, size_t numRows, const int* __restrict__ rows,
                               const val_t* __restrict__ radB, size_t source,
                               val_t* __restrict__ local) {
    size_t il = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (il >= numRows) return;
    size_t i = static_cast<size_t>(rows[il]);

    val_t maxCorr = ZERO;
    int bestT = 0;
    for (size_t t = 0; t < T; ++t) {
        val_t sum = ZERO;
        for (size_t tt = t; tt < T; ++tt) {
            val_t pB = radB[tt * N + i];
            val_t pS = radB[(tt - t) * N + source];
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }
    local[il] = WAVE_SPEED * static_cast<val_t>(bestT);
}

__global__ void countKijKernel(const val_t* __restrict__ KT, size_t total, unsigned long long* out) {
    unsigned long long c = 0;
    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < total;
         k += static_cast<size_t>(gridDim.x) * blockDim.x) {
        if (KT[k] > EPSILON) ++c;
    }
    for (int o = 16; o > 0; o >>= 1) c += __shfl_down_sync(0xffffffffu, c, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, c);
}

// ============================================================================
// Initialization
// ============================================================================

template <typename T>
T* deviceCopy(const std::vector<T>& v) {
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, std::max<size_t>(v.size(), 1) * sizeof(T)));
    if (!v.empty()) CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    const bool root = state.mpi.rank == 0;

    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    const size_t N = state.numTriangles;

    if (root) printf("Generated icosphere mesh with %zu triangles\n", N);

    // Build octree for spatial acceleration (replicated on every rank)
    if (root) printf("Building octree...\n");
    state.octree.build(state.triangles);
    state.octree.flatten(state.nodes, state.nodeTris, 0, state.octreeDepth);
    if (7 * state.octreeDepth + 8 > MAX_STACK) {
        fprintf(stderr, "Octree too deep (%d) for GPU traversal stack\n", state.octreeDepth);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize areas and centers
    state.areas.resize(N);
    std::vector<float4> cent(N);
    #pragma omp parallel for schedule(static) if (N > 4096)
    for (size_t i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
        Vec3 c = state.triangles[i].center();
        cent[i] = make_float4(c.x, c.y, c.z, state.areas[i]);
    }

    // Initialize reflectivity
    state.rho.assign(N, reflectivity);

    // Emission: source active for first half of timesteps
    state.radE.assign(timesteps * N, ZERO);
    size_t timeOff = timesteps / 2;
    for (size_t t = 0; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
    state.distances.assign(N, ZERO);

    // Row distribution: cyclic blocks of ROW_BLOCK rows
    const int P = state.mpi.size;
    const size_t numBlocks = (N + ROW_BLOCK - 1) / ROW_BLOCK;
    state.counts.assign(P, 0);
    state.displs.assign(P, 0);
    state.perm.clear();
    for (int r = 0; r < P; ++r) {
        state.displs[r] = static_cast<int>(state.perm.size());
        for (size_t b = r; b < numBlocks; b += P) {
            for (size_t i = b * ROW_BLOCK; i < std::min(N, (b + 1) * ROW_BLOCK); ++i) {
                state.perm.push_back(static_cast<int>(i));
                if (r == state.mpi.rank) state.myRows.push_back(static_cast<int>(i));
            }
        }
        state.counts[r] = static_cast<int>(state.perm.size()) - state.displs[r];
    }
    const size_t rows = state.localRows();

    // Device setup
    state.dCent = deviceCopy(cent);
    state.dTris = deviceCopy(state.triangles);
    state.dNodes = deviceCopy(state.nodes);
    state.dNodeTris = deviceCopy(state.nodeTris);
    state.dRows = deviceCopy(state.myRows);
    state.dRho = deviceCopy(state.rho);
    state.dRadE = deviceCopy(state.radE);
    CUDA_CHECK(cudaMalloc(&state.dRadB, std::max<size_t>(timesteps * N, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, std::max<size_t>(timesteps * N, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dKT, std::max<size_t>(rows * N, 1) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dLocal, std::max<size_t>(rows, 1) * sizeof(val_t)));
}

void freeDevice(SimulationState& state) {
    cudaFree(state.dCent);
    cudaFree(state.dTris);
    cudaFree(state.dNodes);
    cudaFree(state.dNodeTris);
    cudaFree(state.dRows);
    cudaFree(state.dRho);
    cudaFree(state.dRadE);
    cudaFree(state.dRadB);
    cudaFree(state.dKT);
    cudaFree(state.dLocal);
}

// Gather per-rank local values (state.dLocal) into a global array ordered by triangle
void gatherLocal(SimulationState& state, std::vector<val_t>& global) {
    const size_t rows = state.localRows();
    std::vector<val_t> mine(std::max<size_t>(rows, 1));
    if (rows) CUDA_CHECK(cudaMemcpy(mine.data(), state.dLocal, rows * sizeof(val_t), cudaMemcpyDeviceToHost));
    std::vector<val_t> gathered(state.numTriangles);
    MPI_Allgatherv(mine.data(), static_cast<int>(rows), MPI_FLOAT, gathered.data(),
                   state.counts.data(), state.displs.data(), MPI_FLOAT, MPI_COMM_WORLD);
    #pragma omp parallel for schedule(static) if (state.numTriangles > 65536)
    for (size_t k = 0; k < state.numTriangles; ++k) global[state.perm[k]] = gathered[k];
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    const bool root = state.mpi.rank == 0;
    if (root) printf("Computing form factors (Kij)...\n");

    const size_t N = state.numTriangles;
    const size_t rows = state.localRows();
    if (rows > 0) {
        // Process emitters j in slabs so that progress can be reported
        const size_t slabJ = std::max<size_t>(1, (size_t(1) << 24) / rows);
        const size_t reportEvery = std::max<size_t>(slabJ, N / 10);
        size_t nextReport = reportEvery;
        const int threads = 128;
        for (size_t j0 = 0; j0 < N; j0 += slabJ) {
            size_t nj = std::min(slabJ, N - j0);
            size_t count = nj * rows;
            size_t blocks = (count + threads - 1) / threads;
            formFactorKernel<<<static_cast<unsigned>(blocks), threads>>>(
                state.dKT, state.dRows, rows, N, j0, count,
                state.dNodes, state.dNodeTris, state.dTris);
            CUDA_CHECK(cudaGetLastError());
            if (root && j0 + nj >= nextReport) {
                CUDA_CHECK(cudaDeviceSynchronize());
                printf("  Progress: %zu/%zu emitters (rank 0)\n", j0 + nj, N);
                fflush(stdout);
                nextReport += reportEvery;
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (root) printf("  Progress: %zu/%zu triangles\n", N, N);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    const bool root = state.mpi.rank == 0;
    if (root) printf("Running wave propagation simulation...\n");

    const size_t N = state.numTriangles;
    const size_t rows = state.localRows();
    std::vector<val_t> row(N);
    const int threads = 128;
    const unsigned blocks = static_cast<unsigned>((rows + threads - 1) / threads);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (rows > 0) {
            propagateKernel<<<blocks, threads>>>(t, N, rows, state.dRows, state.dKT, state.dCent,
                                                 state.dRho, state.dRadE, state.dRadB, state.dLocal);
            CUDA_CHECK(cudaGetLastError());
        }
        // Exchange the new radiosity row; later steps depend on it (tau >= 1)
        gatherLocal(state, row);
        CUDA_CHECK(cudaMemcpy(state.dRadB + t * N, row.data(), N * sizeof(val_t), cudaMemcpyHostToDevice));

        if (root && ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.mpi.rank == 0) printf("Computing distances via cross-correlation...\n");

    const size_t rows = state.localRows();
    if (rows > 0) {
        const int threads = 64;
        distanceKernel<<<static_cast<unsigned>((rows + threads - 1) / threads), threads>>>(
            state.numTriangles, state.numTimesteps, rows, state.dRows, state.dRadB,
            state.sourceIndex, state.dLocal);
        CUDA_CHECK(cudaGetLastError());
    }
    gatherLocal(state, state.distances);
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
    unsigned long long nonZeroKij = state.nonZeroKij;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    SimulationState state;
    MPI_Comm_rank(MPI_COMM_WORLD, &state.mpi.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &state.mpi.size);
    const bool root = state.mpi.rank == 0;

    // Bind each rank to a GPU (round-robin over the GPUs of its node)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, state.mpi.rank, MPI_INFO_NULL, &local);
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
        CUDA_CHECK(cudaFree(nullptr));
    }

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;
    int exitCode = -1;

    // Parse command line arguments
    for (int i = 1; i < argc && exitCode < 0; ++i) {
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
            if (root) printUsage(argv[0]);
            exitCode = 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
        }
    }
    if (exitCode >= 0) {
        MPI_Finalize();
        return exitCode;
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (root) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", state.mpi.size, omp_get_max_threads());
        printf("\n");
    }

    // Initialize
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (root) printf("\n");

    // Precomputation (time delays are evaluated on the fly in the simulation kernel)
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (root) {
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("\n");
    }

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (root) {
        printf("Simulation time: %ld ms\n", simDuration);
        printf("\n");
    }

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (root) {
        printf("Distance computation time: %ld ms\n", distDuration);
        printf("\n");
    }

    // Data needed only for validation
    if (validate) {
        const size_t rows = state.localRows();
        unsigned long long local = 0;
        unsigned long long* dCount = nullptr;
        CUDA_CHECK(cudaMalloc(&dCount, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(dCount, 0, sizeof(unsigned long long)));
        if (rows > 0) {
            countKijKernel<<<1024, 256>>>(state.dKT, rows * state.numTriangles, dCount);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaMemcpy(&local, dCount, sizeof(local), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dCount));
        MPI_Allreduce(&local, &state.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

        state.radB.assign(state.numTimesteps * state.numTriangles, ZERO);
        if (!state.radB.empty()) {
            CUDA_CHECK(cudaMemcpy(state.radB.data(), state.dRadB, state.radB.size() * sizeof(val_t),
                                  cudaMemcpyDeviceToHost));
        }
    }
    freeDevice(state);

    if (root) {
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

        // Memory usage (aggregate over all ranks)
        size_t memKij = n * n * sizeof(val_t);
        size_t memRad = 2 * t * n * sizeof(val_t);
        size_t totalMem = memKij + memRad;
        printf("  Memory usage: %.2f MB (Kij distributed over %d ranks)\n",
               totalMem / (1024.0 * 1024.0), state.mpi.size);

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
            if (!validateResults(state)) exitCode = 1;
        }
    }

    if (exitCode < 0) exitCode = 0;
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
