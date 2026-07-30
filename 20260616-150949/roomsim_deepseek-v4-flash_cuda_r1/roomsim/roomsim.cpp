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

#include "../common/results_output.hpp"

// CUDA runtime
#include <cuda_runtime.h>
#include <cfloat>

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
// Forward Declarations (for CUDA kernel and GPU function ordering)
// ============================================================================

struct Triangle;
struct SimulationState;
struct GPUOctreeNode;

void initGPU(class SimulationState& state);
void cleanupGPU(class SimulationState& state);

__global__ void timeDelaysKernel(const Triangle* triangles, int* tau, int numTriangles);
__global__ void formFactorsKernel(const Triangle* triangles, const GPUOctreeNode* nodes,
                                   const int* triIndices, val_t* kij, int numTriangles);
__global__ void simulationKernel(const val_t* kij, const int* tau, const val_t* radE,
                                  val_t* radB, const val_t* areas, const val_t* rho,
                                  int numTriangles, int timestep);
__global__ void distancesKernel(const val_t* radB, val_t* distances, int numTriangles,
                                 int numTimesteps, int sourceIndex);

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

    __host__ __device__ Triangle() = default;
    __host__ __device__ Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
    __host__ __device__ Vec3 normal() const { return _normal; }

    __host__ __device__ val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    __host__ __device__ bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
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
};

// ============================================================================
// GPU-Accelerated Octree Structures
// ============================================================================

// Flat octree node for GPU traversal
struct GPUOctreeNode {
    Vec3 minBound, maxBound;
    int children[8];  // -1 if child does not exist
    int triOffset;    // start index in triangle index array (-1 for internal nodes)
    int triCount;     // number of triangles in this leaf (0 for internal)
};

// Flatten CPU octree into flat GPU-compatible arrays (pre-order traversal)
int flattenOctreeGPU(const Octree& node, std::vector<GPUOctreeNode>& flatNodes,
                     std::vector<int>& flatTriIndices) {
    int myIdx = static_cast<int>(flatNodes.size());
    flatNodes.push_back(GPUOctreeNode());
    flatNodes[myIdx].minBound = node.minBound;
    flatNodes[myIdx].maxBound = node.maxBound;

    for (int i = 0; i < 8; ++i)
        flatNodes[myIdx].children[i] = -1;

    if (!node.triangleIndices.empty()) {
        flatNodes[myIdx].triOffset = static_cast<int>(flatTriIndices.size());
        flatNodes[myIdx].triCount = static_cast<int>(node.triangleIndices.size());
        flatTriIndices.insert(flatTriIndices.end(),
            node.triangleIndices.begin(), node.triangleIndices.end());
    } else {
        flatNodes[myIdx].triOffset = -1;
        flatNodes[myIdx].triCount = 0;
        for (int i = 0; i < 8; ++i) {
            if (node.children[i]) {
                flatNodes[myIdx].children[i] =
                    flattenOctreeGPU(*node.children[i], flatNodes, flatTriIndices);
            }
        }
    }
    return myIdx;
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
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

__host__ __device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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

// Compute the cosine of angle between vector and triangle normal
__host__ __device__ val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// ============================================================================
// GPU Device Helpers (RNG, Ray-Box Intersection, Visibility)
// ============================================================================

// XORShift32 random number generator for GPU
__device__ uint32_t gpuRand(uint32_t& state) {
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

// Uniform [0, 1) random float
__device__ val_t uniform01(uint32_t& state) {
    return (val_t)gpuRand(state) / 4294967296.0f;
}

// Deterministic seed from thread indices
__device__ uint32_t seedFromIJK(int i, int j, int k) {
    uint32_t h = 42u;
    h = (h ^ (uint32_t)i) * 0x9e3779b9u;
    h = (h ^ (uint32_t)j) * 0x9e3779b9u;
    h = (h ^ (uint32_t)k) * 0x9e3779b9u;
    return h ^ (h >> 16);
}

// Generate a random point in a triangle (GPU version)
__device__ Vec3 randomPointInTriangleGPU(const Triangle& t, uint32_t& rngState) {
    val_t u = uniform01(rngState);
    val_t v = uniform01(rngState);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// Ray-AABB intersection test (GPU version using node bounds)
__device__ bool rayIntersectsBoxGPU(const Vec3& p1, const Vec3& p2,
                                     const Vec3& bmin, const Vec3& bmax) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 center = (bmin + bmax) * 0.5f;
    Vec3 halfExtent = (bmax - bmin) * 0.5f;
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

// GPU octree-accelerated ray visibility test
__device__ bool isRayBlockedGPU(const Vec3& from, const Vec3& to,
                                 const GPUOctreeNode* nodes,
                                 const int* triIndices,
                                 const Triangle* tris,
                                 int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = sqrtf(dir.squaredNorm());
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    // Stack-based iterative traversal
    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0; // root node

    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        const GPUOctreeNode& node = nodes[nodeIdx];

        if (!rayIntersectsBoxGPU(from, to, node.minBound, node.maxBound)) continue;

        if (node.triCount > 0) {
            // Leaf: test triangles
            for (int t = 0; t < node.triCount; ++t) {
                int triIdx = triIndices[node.triOffset + t];
                if (triIdx == srcTriIdx || triIdx == dstTriIdx) continue;

                val_t dist = rayTriangleIntersect(from, dirNorm,
                    tris[triIdx].a, tris[triIdx].b, tris[triIdx].c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            // Internal: push children (reverse order so first child is processed first)
            for (int c = 7; c >= 0; --c) {
                if (node.children[c] != -1) {
                    stack[stackPtr++] = node.children[c];
                }
            }
        }
    }
    return false;
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

    // GPU device pointers
    Triangle* d_triangles = nullptr;
    GPUOctreeNode* d_nodes = nullptr;
    int* d_triIndices = nullptr;
    int d_numNodes = 0;
    int d_numTriIndices = 0;
    val_t* d_kij = nullptr;
    int* d_tau = nullptr;
    val_t* d_radE = nullptr;
    val_t* d_radB = nullptr;
    val_t* d_distances = nullptr;
    val_t* d_areas = nullptr;
    val_t* d_rho = nullptr;

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

    // Initialize GPU memory and upload data
    initGPU(state);
}

// ============================================================================
// GPU Memory Management
// ============================================================================

void initGPU(SimulationState& state) {
    // Flatten CPU octree into GPU-compatible arrays
    std::vector<GPUOctreeNode> flatNodes;
    std::vector<int> flatTriIndices;
    flattenOctreeGPU(state.octree, flatNodes, flatTriIndices);
    state.d_numNodes = static_cast<int>(flatNodes.size());
    state.d_numTriIndices = static_cast<int>(flatTriIndices.size());

    printf("GPU: Flattened octree to %d nodes, %d tri indices\n",
           state.d_numNodes, state.d_numTriIndices);

    // Allocate device memory
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;

    cudaMalloc(&state.d_triangles,    n * sizeof(Triangle));
    cudaMalloc(&state.d_nodes,        flatNodes.size() * sizeof(GPUOctreeNode));
    cudaMalloc(&state.d_triIndices,   flatTriIndices.size() * sizeof(int));
    cudaMalloc(&state.d_kij,          n * n * sizeof(val_t));
    cudaMalloc(&state.d_tau,          n * n * sizeof(int));
    cudaMalloc(&state.d_radE,         t * n * sizeof(val_t));
    cudaMalloc(&state.d_radB,         t * n * sizeof(val_t));
    cudaMalloc(&state.d_distances,    n * sizeof(val_t));
    cudaMalloc(&state.d_areas,        n * sizeof(val_t));
    cudaMalloc(&state.d_rho,          n * sizeof(val_t));

    // Upload data to device
    cudaMemcpy(state.d_triangles,  state.triangles.data(),  n * sizeof(Triangle),  cudaMemcpyHostToDevice);
    cudaMemcpy(state.d_nodes,      flatNodes.data(),        flatNodes.size() * sizeof(GPUOctreeNode), cudaMemcpyHostToDevice);
    cudaMemcpy(state.d_triIndices, flatTriIndices.data(),   flatTriIndices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(state.d_areas,      state.areas.data(),      n * sizeof(val_t),     cudaMemcpyHostToDevice);
    cudaMemcpy(state.d_rho,        state.rho.data(),        n * sizeof(val_t),     cudaMemcpyHostToDevice);
    cudaMemcpy(state.d_radE,       state.radE.data(),       t * n * sizeof(val_t), cudaMemcpyHostToDevice);

    // Initialize radB to zero on device
    cudaMemset(state.d_radB, 0, t * n * sizeof(val_t));

    cudaDeviceSynchronize();
}

void cleanupGPU(SimulationState& state) {
    cudaFree(state.d_triangles);
    cudaFree(state.d_nodes);
    cudaFree(state.d_triIndices);
    cudaFree(state.d_kij);
    cudaFree(state.d_tau);
    cudaFree(state.d_radE);
    cudaFree(state.d_radB);
    cudaFree(state.d_distances);
    cudaFree(state.d_areas);
    cudaFree(state.d_rho);

    state.d_triangles = nullptr;
    state.d_nodes = nullptr;
    state.d_triIndices = nullptr;
    state.d_kij = nullptr;
    state.d_tau = nullptr;
    state.d_radE = nullptr;
    state.d_radB = nullptr;
    state.d_distances = nullptr;
    state.d_areas = nullptr;
    state.d_rho = nullptr;
}

// ============================================================================
// GPU-Accelerated Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau) on GPU...\n");

    int n = static_cast<int>(state.numTriangles);
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (n + 15) / 16);

    timeDelaysKernel<<<grid, block>>>(state.d_triangles, state.d_tau, n);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(state.tau.data(), state.d_tau,
               n * n * sizeof(int), cudaMemcpyDeviceToHost);
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij) on GPU...\n");

    int n = static_cast<int>(state.numTriangles);
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (n + 15) / 16);

    formFactorsKernel<<<grid, block>>>(
        state.d_triangles, state.d_nodes, state.d_triIndices,
        state.d_kij, n);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(state.kij.data(), state.d_kij,
               n * n * sizeof(val_t), cudaMemcpyDeviceToHost);
}

// ============================================================================
// GPU-Accelerated Simulation Phase
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation on GPU...\n");

    int n = static_cast<int>(state.numTriangles);
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        simulationKernel<<<gridSize, blockSize>>>(
            state.d_kij, state.d_tau, state.d_radE, state.d_radB,
            state.d_areas, state.d_rho, n, static_cast<int>(t));

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
            cudaDeviceSynchronize();
        }
    }
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(state.radB.data(), state.d_radB,
               state.numTimesteps * n * sizeof(val_t), cudaMemcpyDeviceToHost);
}

// ============================================================================
// GPU-Accelerated Distance Computation
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation on GPU...\n");

    int n = static_cast<int>(state.numTriangles);
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    distancesKernel<<<gridSize, blockSize>>>(
        state.d_radB, state.d_distances, n,
        static_cast<int>(state.numTimesteps),
        static_cast<int>(state.sourceIndex));
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(state.distances.data(), state.d_distances,
               n * sizeof(val_t), cudaMemcpyDeviceToHost);
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
// CUDA Kernels
// ============================================================================

// Kernel 1: Compute time delays (Tau) for all (i,j) pairs
__global__ void timeDelaysKernel(
    const Triangle* triangles,
    int* tau,
    int numTriangles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= numTriangles || j >= numTriangles || i == j) return;

    Vec3 ci = triangles[i].center();
    Vec3 cj = triangles[j].center();
    val_t dist = sqrtf((ci - cj).squaredNorm());
    tau[i * numTriangles + j] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// Kernel 2: Compute form factors (Kij) for all (i,j) pairs
__global__ void formFactorsKernel(
    const Triangle* triangles,
    const GPUOctreeNode* nodes,
    const int* triIndices,
    val_t* kij,
    int numTriangles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= numTriangles || j >= numTriangles) return;
    if (i == j) { kij[i * numTriangles + j] = ZERO; return; }

    // Cull triangles facing the same direction
    if (triangles[i].normal().dot(triangles[j].normal()) > 0.99f) {
        kij[i * numTriangles + j] = ZERO;
        return;
    }

    // Initialize per-thread RNG with deterministic seed
    uint32_t rngState = seedFromIJK(i, j, 0);

    val_t k = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangleGPU(triangles[i], rngState);
        Vec3 pJ = randomPointInTriangleGPU(triangles[j], rngState);

        if (isRayBlockedGPU(pI, pJ, nodes, triIndices, triangles, i, j))
            continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triangles[i].normal());
        val_t cosPhiJ = cosPhi(-v, triangles[j].normal());

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        k += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kij[i * numTriangles + j] = k * INV_NUM_RAYS;
}

// Kernel 3: Wave propagation simulation for one timestep
__global__ void simulationKernel(
    const val_t* kij,
    const int* tau,
    const val_t* radE,
    val_t* radB,
    const val_t* areas,
    const val_t* rho,
    int numTriangles,
    int timestep) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numTriangles) return;

    val_t sumB = ZERO;

    for (int j = 0; j < numTriangles; ++j) {
        if (i == j) continue;

        int tauij = tau[i * numTriangles + j];
        if (timestep < tauij) continue;

        val_t k = kij[i * numTriangles + j];
        if (k <= ZERO) continue;

        int srcTime = timestep - tauij;
        val_t radJ = radB[srcTime * numTriangles + j];
        if (radJ <= ZERO) continue;

        sumB += fminf(k * areas[j], ONE) * radJ;
    }

    radB[timestep * numTriangles + i] = rho[i] * sumB + radE[timestep * numTriangles + i];
}

// Kernel 4: Cross-correlation distance computation
__global__ void distancesKernel(
    const val_t* radB,
    val_t* distances,
    int numTriangles,
    int numTimesteps,
    int sourceIndex) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numTriangles) return;

    val_t maxCorr = ZERO;
    int bestT = 0;

    for (int t = 0; t < numTimesteps; ++t) {
        val_t sum = ZERO;
        for (int tt = t; tt < numTimesteps; ++tt) {
            sum += radB[(tt - t) * numTriangles + sourceIndex] *
                   radB[tt * numTriangles + i];
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }

    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
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
    if (validate) {
        if (!validateResults(state)) {
            cleanupGPU(state);
            return 1;
        }
    }

    cleanupGPU(state);
    return 0;
}
