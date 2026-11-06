/**
 * Room Response Simulation Benchmark
 * 
 * This is a CUDA implementation of room impulse response
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

    __host__ __device__ bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
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


};

// Compact preorder octree.  escape is the first node after the subtree,
// allowing a ray to traverse the tree without per-thread stack storage.
struct GpuOctreeNode {
    Vec3 center;
    Vec3 halfExtent;
    uint32_t first;
    uint32_t count;
    uint32_t escape;
};

void flattenOctree(const Octree& tree, std::vector<GpuOctreeNode>& nodes,
                   std::vector<uint32_t>& indices) {
    const size_t nodeIndex = nodes.size();
    GpuOctreeNode node{};
    node.center = tree.center;
    node.halfExtent = tree.halfExtent;
    node.first = static_cast<uint32_t>(indices.size());
    node.count = static_cast<uint32_t>(tree.triangleIndices.size());
    nodes.push_back(node);
    for (size_t idx : tree.triangleIndices) indices.push_back(static_cast<uint32_t>(idx));
    for (int i = 0; i < 8; ++i) {
        if (tree.children[i]) flattenOctree(*tree.children[i], nodes, indices);
    }
    nodes[nodeIndex].escape = static_cast<uint32_t>(nodes.size());
}

__device__ bool rayIntersectsBoxDevice(const Vec3& p1, const Vec3& p2,
                                       const GpuOctreeNode& node) {
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

__device__ val_t rayTriangleIntersectDevice(const Vec3& orig, const Vec3& dir,
                                             const Triangle& tri) {
    Vec3 e1 = tri.b - tri.a;
    Vec3 e2 = tri.c - tri.a;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);
    if (fabsf(det) < EPSILON) return 3.402823466e+38F;
    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - tri.a;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return 3.402823466e+38F;
    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 3.402823466e+38F;
    return e2.dot(qvec) * invDet;
}

__device__ bool isRayBlockedDevice(const Vec3& from, const Vec3& to,
                                   const GpuOctreeNode* nodes, const uint32_t* indices,
                                   const Triangle* triangles, size_t src, size_t dst) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;
    uint32_t nodeIndex = 0;
    const uint32_t end = nodes[0].escape;
    while (nodeIndex < end) {
        const GpuOctreeNode& node = nodes[nodeIndex];
        // The CPU starts at the root without testing its bounds.
        if (nodeIndex != 0 && !rayIntersectsBoxDevice(from, to, node)) {
            nodeIndex = node.escape;
            continue;
        }
        for (uint32_t k = 0; k < node.count; ++k) {
            uint32_t idx = indices[node.first + k];
            if (idx == src || idx == dst) continue;
            val_t dist = rayTriangleIntersectDevice(from, dirNorm, triangles[idx]);
            if (dist > EPSILON && dist < rayLen - EPSILON) return true;
        }
        ++nodeIndex;
    }
    return false;
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
// Random Number Generation
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
};

// ============================================================================
// CUDA kernels
// ============================================================================

__device__ Vec3 randomPointDevice(const Triangle& tri, const val_t* randoms) {
    val_t u = randoms[0];
    val_t v = randoms[1];
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = tri.b - tri.a;
    Vec3 ac = tri.c - tri.a;
    return tri.a + ab * u + ac * v;
}

__device__ val_t cosPhiDevice(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t value = v.dot(normal) / vNorm;
    return value > ZERO ? value : ZERO;
}

__global__ void formFactorKernel(const Triangle* triangles,
                                 const GpuOctreeNode* nodes,
                                 const uint32_t* indices,
                                 const val_t* randoms,
                                 val_t* kij, int* tau, size_t n,
                                 size_t pairStart, size_t pairCount) {
    size_t localPair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (localPair >= pairCount) return;
    size_t pair = pairStart + localPair;
    size_t i = pair / n;
    size_t j = pair % n;
    size_t output = j * n + i;
    if (i == j) {
        kij[output] = ZERO;
        tau[output] = 0;
        return;
    }

    const Triangle& triI = triangles[i];
    const Triangle& triJ = triangles[j];
    val_t dist = (triI.center() - triJ.center()).norm();
    tau[output] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));

    const val_t* pairRandoms = randoms + localPair * (NUM_RAYS * 4);
    // The host marks pairs culled while advancing the reference RNG stream.
    if (pairRandoms[0] < ZERO) {
        kij[output] = ZERO;
        return;
    }

    val_t result = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointDevice(triI, pairRandoms + 4 * r);
        Vec3 pJ = randomPointDevice(triJ, pairRandoms + 4 * r + 2);
        if (isRayBlockedDevice(pI, pJ, nodes, indices, triangles, i, j)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;
        val_t cosI = cosPhiDevice(v, triI.normal());
        val_t cosJ = cosPhiDevice(-v, triJ.normal());
        if (cosI <= ZERO || cosJ <= ZERO) continue;
        result += (cosI * cosJ) / (PI * distSqr);
    }
    kij[output] = result * INV_NUM_RAYS;
}

__global__ void simulationKernel(const val_t* kij, const int* tau,
                                 const val_t* areas, const val_t* rho,
                                 const val_t* radE, val_t* radB,
                                 size_t n, size_t timestep) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    val_t sumB = ZERO;
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        size_t pair = j * n + i;
        int delay = tau[pair];
        if (static_cast<int>(timestep) < delay) continue;
        val_t factor = kij[pair];
        if (factor <= ZERO) continue;
        val_t radJ = radB[(timestep - static_cast<size_t>(delay)) * n + j];
        if (radJ <= ZERO) continue;
        val_t weighted = factor * areas[j];
        if (weighted > ONE) weighted = ONE;
        sumB += weighted * radJ;
    }
    radB[timestep * n + i] = rho[i] * sumB + radE[timestep * n + i];
}

__global__ void distanceKernel(const val_t* radB, val_t* distances,
                               size_t n, size_t timesteps, size_t sourceIndex) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    val_t maxCorr = ZERO;
    int bestT = 0;
    for (size_t t = 0; t < timesteps; ++t) {
        val_t sum = ZERO;
        for (size_t tt = t; tt < timesteps; ++tt) {
            val_t pB = radB[tt * n + i];
            val_t pS = radB[(tt - t) * n + sourceIndex];
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
    std::vector<val_t> kij;         // Transposed form factors, copied only for validation
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity, bool validate) {
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
    if (validate) state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    if (validate) state.radB.resize(timesteps * state.numTriangles, ZERO);
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

static void cudaCheck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
static void deviceAllocate(T*& ptr, size_t count) {
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&ptr), std::max(count, size_t(1)) * sizeof(T)), "allocation");
}

template <typename T>
static void copyToDevice(T* dst, const std::vector<T>& src) {
    if (!src.empty()) cudaCheck(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), "host-to-device copy");
}

struct GpuSimulation {
    Triangle* triangles = nullptr;
    GpuOctreeNode* nodes = nullptr;
    uint32_t* indices = nullptr;
    val_t *kij = nullptr, *areas = nullptr, *rho = nullptr, *radE = nullptr, *radB = nullptr, *distances = nullptr;
    int* tau = nullptr;

    ~GpuSimulation() {
        cudaFree(triangles); cudaFree(nodes); cudaFree(indices); cudaFree(kij); cudaFree(tau);
        cudaFree(areas); cudaFree(rho); cudaFree(radE); cudaFree(radB); cudaFree(distances);
    }
};

static GpuSimulation gpu;

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const size_t n = state.numTriangles;
    const size_t pairs = n * n;
    std::vector<GpuOctreeNode> nodes;
    std::vector<uint32_t> indices;
    flattenOctree(state.octree, nodes, indices);
    deviceAllocate(gpu.triangles, n);
    deviceAllocate(gpu.nodes, nodes.size());
    deviceAllocate(gpu.indices, indices.size());
    deviceAllocate(gpu.kij, pairs);
    deviceAllocate(gpu.tau, pairs);
    deviceAllocate(gpu.areas, n);
    deviceAllocate(gpu.rho, n);
    deviceAllocate(gpu.radE, state.radE.size());
    const size_t radiosityCount = state.numTimesteps * n;
    deviceAllocate(gpu.radB, radiosityCount);
    deviceAllocate(gpu.distances, n);
    copyToDevice(gpu.triangles, state.triangles);
    copyToDevice(gpu.nodes, nodes);
    copyToDevice(gpu.indices, indices);
    copyToDevice(gpu.areas, state.areas);
    copyToDevice(gpu.rho, state.rho);
    copyToDevice(gpu.radE, state.radE);
    if (radiosityCount != 0)
        cudaCheck(cudaMemset(gpu.radB, 0, radiosityCount * sizeof(val_t)), "zero radiosity");

    // Preserve the serial generator's sequence while overlapping generation,
    // transfer, and ray tracing in bounded chunks.
    constexpr size_t chunkPairs = 8192;
    constexpr size_t samplesPerPair = 4 * NUM_RAYS;
    val_t* hostRandom[2]{};
    val_t* deviceRandom[2]{};
    cudaStream_t streams[2]{};
    for (int b = 0; b < 2; ++b) {
        cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&hostRandom[b]), chunkPairs * samplesPerPair * sizeof(val_t), cudaHostAllocDefault), "pinned allocation");
        deviceAllocate(deviceRandom[b], chunkPairs * samplesPerPair);
        cudaCheck(cudaStreamCreate(&streams[b]), "stream creation");
    }
    RandomGenerator rng(42);
    size_t chunk = 0;
    for (size_t start = 0; start < pairs; start += chunkPairs, ++chunk) {
        int b = static_cast<int>(chunk & 1);
        cudaCheck(cudaStreamSynchronize(streams[b]), "random buffer reuse");
        size_t count = std::min(chunkPairs, pairs - start);
        for (size_t local = 0; local < count; ++local) {
            size_t pair = start + local;
            size_t i = pair / n, j = pair % n;
            if (i == j || state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) {
                hostRandom[b][local * samplesPerPair] = -1.0f;
                continue;
            }
            for (size_t k = 0; k < samplesPerPair; ++k)
                hostRandom[b][local * samplesPerPair + k] = rng.rand();
        }
        cudaCheck(cudaMemcpyAsync(deviceRandom[b], hostRandom[b], count * samplesPerPair * sizeof(val_t), cudaMemcpyHostToDevice, streams[b]), "random transfer");
        formFactorKernel<<<static_cast<unsigned>((count + 127) / 128), 128, 0, streams[b]>>>(
            gpu.triangles, gpu.nodes, gpu.indices, deviceRandom[b], gpu.kij, gpu.tau, n, start, count);
        cudaCheck(cudaGetLastError(), "form factor launch");
    }
    for (int b = 0; b < 2; ++b) {
        cudaCheck(cudaStreamSynchronize(streams[b]), "form factor computation");
        cudaCheck(cudaStreamDestroy(streams[b]), "stream destruction");
        cudaCheck(cudaFree(deviceRandom[b]), "random buffer release");
        cudaCheck(cudaFreeHost(hostRandom[b]), "pinned buffer release");
    }
    // Validation only counts positive entries; the device stores the matrix transposed.
    if (!state.kij.empty())
        cudaCheck(cudaMemcpy(state.kij.data(), gpu.kij, pairs * sizeof(val_t), cudaMemcpyDeviceToHost), "form factor copy");
    printf("  Progress: %zu/%zu triangles\n", n, n);
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    // The pair kernel computes delays together with form factors.
    (void)state;
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const size_t n = state.numTriangles;
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        simulationKernel<<<static_cast<unsigned>((n + 127) / 128), 128>>>(
            gpu.kij, gpu.tau, gpu.areas, gpu.rho, gpu.radE, gpu.radB, n, t);
        cudaCheck(cudaGetLastError(), "simulation launch");
        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    cudaCheck(cudaDeviceSynchronize(), "simulation");
    if (!state.radB.empty())
        cudaCheck(cudaMemcpy(state.radB.data(), gpu.radB, state.radB.size() * sizeof(val_t), cudaMemcpyDeviceToHost), "radiosity copy");
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const size_t n = state.numTriangles;
    distanceKernel<<<static_cast<unsigned>((n + 127) / 128), 128>>>(
        gpu.radB, gpu.distances, n, state.numTimesteps, state.sourceIndex);
    cudaCheck(cudaGetLastError(), "distance launch");
    cudaCheck(cudaMemcpy(state.distances.data(), gpu.distances, n * sizeof(val_t), cudaMemcpyDeviceToHost), "distance copy");
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

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity, validate);

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
            return 1;
        }
    }

    return 0;
}
