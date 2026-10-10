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
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <vector>

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
constexpr val_t NO_INTERSECTION = std::numeric_limits<val_t>::max();

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

    if (std::abs(det) < EPSILON) return NO_INTERSECTION;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return NO_INTERSECTION;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return NO_INTERSECTION;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct CudaSimulation;

struct SimulationState {
    std::unique_ptr<CudaSimulation> gpu;
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Form factors (N x N matrix, row-major)
    std::vector<int> tau;           // Time delays (N x N matrix, row-major)
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
    // Dense matrices and radiosity are allocated on the GPU. Host copies are
    // materialized only when validation needs them.
    state.distances.resize(state.numTriangles, ZERO);

    // The propagation kernel applies the source emission directly.
}

// ============================================================================
// CUDA execution and storage
// ============================================================================

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

size_t checkedProduct(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b) {
        throw std::runtime_error("simulation size overflow");
    }
    return a * b;
}

template<class T>
class DeviceBuffer {
    T* pointer = nullptr;
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (pointer) cudaFree(pointer); }
    void allocate(size_t count) {
        if (pointer) cudaCheck(cudaFree(pointer), "cudaFree");
        pointer = nullptr;
        if (count) {
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&pointer), checkedProduct(count, sizeof(T))),
                      "cudaMalloc");
        }
    }
    T* get() const { return pointer; }
    void upload(const T* source, size_t count) {
        if (count) cudaCheck(cudaMemcpy(pointer, source, count * sizeof(T), cudaMemcpyHostToDevice),
                             "upload simulation data");
    }
    void download(T* destination, size_t count) const {
        if (count) cudaCheck(cudaMemcpy(destination, pointer, count * sizeof(T), cudaMemcpyDeviceToHost),
                             "download simulation data");
    }
};

// Preorder nodes with an escape index allow stackless traversal on the GPU.
// Keeping the original octree's leaves also preserves its visibility semantics.
struct DeviceOctreeNode {
    Vec3 center, halfExtent;
    uint32_t firstTriangle, triangleCount, escape;
};

void flattenOctree(const Octree& tree, std::vector<DeviceOctreeNode>& nodes,
                   std::vector<uint32_t>& indices) {
    const size_t nodeIndex = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, static_cast<uint32_t>(indices.size()),
                     static_cast<uint32_t>(tree.triangleIndices.size()), 0});
    for (size_t index : tree.triangleIndices) indices.push_back(static_cast<uint32_t>(index));
    for (const auto& child : tree.children) {
        if (child) flattenOctree(*child, nodes, indices);
    }
    nodes[nodeIndex].escape = static_cast<uint32_t>(nodes.size());
}

__device__ bool intersectsNode(const Vec3& from, const Vec3& to, const DeviceOctreeNode& node) {
    const Vec3 d = (to - from) * 0.5f;
    const Vec3 c = from + d - node.center;
    const Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    if (fabsf(c.x) > node.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > node.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > node.halfExtent.z + ad.z) return false;
    if (fabsf(d.y * c.z - d.z * c.y) > node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPSILON) return false;
    return true;
}

__device__ bool rayBlocked(const Vec3& from, const Vec3& to,
                          const Triangle* triangles, const DeviceOctreeNode* nodes,
                          const uint32_t* indices, size_t source, size_t destination) {
    const Vec3 direction = to - from;
    const val_t length = direction.norm();
    if (length < EPSILON) return true;
    const Vec3 normalized = direction / length;
    uint32_t nodeIndex = 0;
    const uint32_t end = nodes[0].escape;
    while (nodeIndex < end) {
        const DeviceOctreeNode& node = nodes[nodeIndex];
        // The original traversal does not test the root's bounding box.
        if (nodeIndex && !intersectsNode(from, to, node)) {
            nodeIndex = node.escape;
            continue;
        }
        for (uint32_t k = 0; k < node.triangleCount; ++k) {
            const uint32_t triangleIndex = indices[node.firstTriangle + k];
            if (triangleIndex == source || triangleIndex == destination) continue;
            const Triangle& triangle = triangles[triangleIndex];
            const val_t distance = rayTriangleIntersect(from, normalized, triangle.a, triangle.b, triangle.c);
            if (distance > EPSILON && distance < length - EPSILON) return true;
        }
        ++nodeIndex;
    }
    return false;
}

__device__ Vec3 sampleTriangle(const Triangle& triangle, val_t u, val_t v) {
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return triangle.a + (triangle.b - triangle.a) * u + (triangle.c - triangle.a) * v;
}

// Each checkpoint describes the exact position in the original mt19937 stream.
// Advance untempered state on the host; expand independent chunks in parallel on
// the GPU. This avoids changing the seed, assigning per-thread random streams,
// or transferring all 64 random values required by each active triangle pair.
constexpr unsigned MT_SIZE = 624;
constexpr unsigned RNG_CHUNK = MT_SIZE * 8;
constexpr unsigned PAIR_BATCH = 65536;
constexpr unsigned MAX_CHECKPOINTS = (PAIR_BATCH * NUM_RAYS * 4 + RNG_CHUNK - 1) / RNG_CHUNK;
// CUB accepts an int-sized scan. Batch-aligned shards keep all matrix addressing
// in size_t and allow larger meshes without overflowing scan offsets.
constexpr size_t SCAN_BATCHES = std::numeric_limits<int>::max() / PAIR_BATCH;
constexpr size_t SCAN_PAIRS = SCAN_BATCHES * PAIR_BATCH;

struct RandomCheckpoint {
    uint32_t state[MT_SIZE];
    unsigned position;
};

class OrderedRandomStream {
    RandomCheckpoint current;
    void twist() {
        for (unsigned k = 0; k < MT_SIZE - 397; ++k) {
            const uint32_t bits = (current.state[k] & 0x80000000u) | (current.state[k + 1] & 0x7fffffffu);
            current.state[k] = current.state[k + 397] ^ (bits >> 1) ^ ((bits & 1) ? 0x9908b0dfu : 0u);
        }
        for (unsigned k = MT_SIZE - 397; k < MT_SIZE - 1; ++k) {
            const uint32_t bits = (current.state[k] & 0x80000000u) | (current.state[k + 1] & 0x7fffffffu);
            current.state[k] = current.state[k - (MT_SIZE - 397)] ^ (bits >> 1) ^ ((bits & 1) ? 0x9908b0dfu : 0u);
        }
        const uint32_t bits = (current.state[MT_SIZE - 1] & 0x80000000u) | (current.state[0] & 0x7fffffffu);
        current.state[MT_SIZE - 1] = current.state[396] ^ (bits >> 1) ^ ((bits & 1) ? 0x9908b0dfu : 0u);
        current.position = 0;
    }
    void advance(unsigned count) {
        while (count) {
            if (current.position == MT_SIZE) twist();
            const unsigned take = std::min(count, MT_SIZE - current.position);
            current.position += take;
            count -= take;
        }
    }
public:
    OrderedRandomStream() {
        current.state[0] = 42;
        for (unsigned k = 1; k < MT_SIZE; ++k) {
            const uint32_t previous = current.state[k - 1];
            current.state[k] = 1812433253u * (previous ^ (previous >> 30)) + k;
        }
        current.position = MT_SIZE;
    }
    unsigned checkpoints(RandomCheckpoint* output, unsigned count) {
        unsigned chunks = 0;
        while (count) {
            output[chunks++] = current;
            const unsigned take = std::min(count, RNG_CHUNK);
            advance(take);
            count -= take;
        }
        return chunks;
    }
};

__device__ uint32_t twistWord(uint32_t source, uint32_t first, uint32_t second) {
    const uint32_t bits = (first & 0x80000000u) | (second & 0x7fffffffu);
    return source ^ (bits >> 1) ^ ((bits & 1) ? 0x9908b0dfu : 0u);
}

__global__ void expandRandomKernel(const RandomCheckpoint* checkpoints, val_t* random, unsigned count) {
    __shared__ uint32_t state[MT_SIZE];
    __shared__ uint32_t next[MT_SIZE];
    const unsigned thread = threadIdx.x;
    const RandomCheckpoint& checkpoint = checkpoints[blockIdx.x];
    for (unsigned k = thread; k < MT_SIZE; k += blockDim.x) state[k] = checkpoint.state[k];
    unsigned position = checkpoint.position;
    const unsigned start = blockIdx.x * RNG_CHUNK;
    const unsigned length = min(RNG_CHUNK, count - start);
    unsigned produced = 0;
    __syncthreads();
    while (produced < length) {
        if (position == MT_SIZE) {
            // The second and third ranges depend on already-twisted words.
            // Separate old/new arrays prevent neighbor read/write races.
            if (thread < 227) next[thread] = twistWord(state[thread + 397], state[thread], state[thread + 1]);
            __syncthreads();
            if (thread < 227) {
                const unsigned k = thread + 227;
                const uint32_t value = twistWord(next[thread], state[k], state[k + 1]);
                next[k] = value;
                if (thread < 169) {
                    const unsigned k2 = thread + 454;
                    next[k2] = twistWord(value, state[k2], state[k2 + 1]);
                }
            }
            __syncthreads();
            if (thread == 0) next[623] = twistWord(next[396], state[623], next[0]);
            __syncthreads();
            for (unsigned k = thread; k < MT_SIZE; k += blockDim.x) state[k] = next[k];
            __syncthreads();
            position = 0;
        }
        const unsigned take = min(MT_SIZE - position, length - produced);
        for (unsigned k = thread; k < take; k += blockDim.x) {
            uint32_t value = state[position + k];
            value ^= value >> 11;
            value ^= (value << 7) & 0x9d2c5680u;
            value ^= (value << 15) & 0xefc60000u;
            value ^= value >> 18;
            // Match uniform_real_distribution<float>, including its upper clamp.
            random[start + produced + k] = fminf(__uint2float_rn(value) * 0x1p-32f, 0x1.fffffep-1f);
        }
        __syncthreads();
        position += take;
        produced += take;
    }
}

__global__ void centersKernel(const Triangle* triangles, Vec3* centers, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) centers[i] = triangles[i].center();
}

__global__ void geometryKernel(const Triangle* triangles, const Vec3* centers,
                               int* tau, uint32_t* activePrefix, size_t n) {
    const size_t count = n * n;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t pair = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < count; pair += stride) {
        const size_t i = pair / n;
        const size_t j = pair % n;
        tau[pair] = i == j ? 0 : static_cast<int>(ceilf((centers[i] - centers[j]).norm() * INV_WAVE_SPEED));
        activePrefix[pair] = i != j && !(triangles[i].normal().dot(triangles[j].normal()) > 0.99f);
    }
}

__global__ void batchTotalsKernel(const uint32_t* prefix, uint32_t* totals, size_t count, size_t batches) {
    const size_t batch = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (batch < batches) {
        const size_t end = min(count, (batch + 1) * PAIR_BATCH);
        totals[batch] = prefix[end - 1];
    }
}

__global__ void formFactorsKernel(const Triangle* triangles, const DeviceOctreeNode* nodes,
                                  const uint32_t* indices, const uint32_t* prefix,
                                  const val_t* random, val_t* kij, size_t n,
                                  size_t begin, unsigned pairs, uint32_t randomBase) {
    // Sixteen lanes sample one pair's rays concurrently. The final additions
    // retain the serial ray order, rather than using a floating-point reduction.
    const unsigned localPair = (blockIdx.x * blockDim.x + threadIdx.x) / NUM_RAYS;
    const unsigned ray = threadIdx.x % NUM_RAYS;
    if (localPair >= pairs) return;
    const size_t pair = begin + localPair;
    const uint32_t rank = prefix[pair];
    if (rank == (localPair ? prefix[pair - 1] : randomBase)) return;
    const size_t i = pair / n;
    const size_t j = pair % n;
    const Triangle& receiver = triangles[i];
    const Triangle& emitter = triangles[j];
    const size_t offset = static_cast<size_t>(rank - randomBase - 1) * (NUM_RAYS * 4) + ray * 4;
    const Vec3 from = sampleTriangle(receiver, random[offset], random[offset + 1]);
    const Vec3 to = sampleTriangle(emitter, random[offset + 2], random[offset + 3]);
    val_t contribution = ZERO;
    if (!rayBlocked(from, to, triangles, nodes, indices, i, j)) {
        const Vec3 v = to - from;
        const val_t squaredDistance = v.squaredNorm();
        if (squaredDistance >= EPSILON) {
            const val_t length = v.norm();
            const val_t cosineI = length <= EPSILON ? ZERO : fmaxf(ZERO, v.dot(receiver.normal()) / length);
            const val_t cosineJ = length <= EPSILON ? ZERO : fmaxf(ZERO, (-v).dot(emitter.normal()) / length);
            if (cosineI > ZERO && cosineJ > ZERO) {
                contribution = (cosineI * cosineJ) / (PI * squaredDistance);
            }
        }
    }
    // Each half warp has its own mask: the other pair can have been culled.
    const unsigned mask = 0xffffu << ((threadIdx.x % 32) / NUM_RAYS * NUM_RAYS);
    val_t sum = ZERO;
    #pragma unroll
    for (unsigned r = 0; r < NUM_RAYS; ++r) {
        const val_t value = __shfl_sync(mask, contribution, r, NUM_RAYS);
        if (ray == 0) sum += value;
    }
    if (ray == 0) kij[pair] = sum * INV_NUM_RAYS;
}

__global__ void propagateKernel(const val_t* __restrict__ kij, const int* __restrict__ tau,
                                const val_t* __restrict__ areas, const val_t* __restrict__ rho,
                                val_t* __restrict__ radiosity, size_t n, size_t time,
                                size_t timeOff, size_t source) {
    const size_t i = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const unsigned lane = threadIdx.x % 32;
    if (i >= n) return;
    val_t sum = ZERO;
    for (size_t begin = 0; begin < n; begin += 32) {
        const size_t j = begin + lane;
        val_t contribution = ZERO;
        if (j < n && i != j) {
            const size_t pair = i * n + j;
            const int delay = tau[pair];
            if (static_cast<int>(time) >= delay) {
                const val_t factor = kij[pair];
                if (factor > ZERO) {
                    const val_t incoming = radiosity[(time - static_cast<size_t>(delay)) * n + j];
                    if (!(incoming <= ZERO)) {
                        const val_t weight = factor * areas[j];
                        contribution = (weight > ONE ? ONE : weight) * incoming;
                    }
                }
            }
        }
        if (__ballot_sync(0xffffffffu, contribution != ZERO)) {
            #pragma unroll
            for (unsigned k = 0; k < 32; ++k) {
                const val_t value = __shfl_sync(0xffffffffu, contribution, k);
                if (lane == 0) sum += value;
            }
        }
    }
    const val_t emission = i == source && time < timeOff ? ONE : ZERO;
    if (lane == 0) radiosity[time * n + i] = rho[i] * sum + emission;
}

__global__ void correlationKernel(const val_t* __restrict__ radiosity, val_t* __restrict__ correlations,
                                  size_t n, size_t timesteps, size_t source) {
    const size_t count = n * timesteps;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        const size_t lag = index / n;
        const size_t i = index % n;
        val_t sum = ZERO;
        for (size_t tt = lag; tt < timesteps; ++tt) {
            sum += radiosity[(tt - lag) * n + source] * radiosity[tt * n + i];
        }
        correlations[index] = sum;
    }
}

__global__ void bestDistanceKernel(const val_t* correlations, val_t* distances, size_t n, size_t timesteps) {
    __shared__ val_t warpValues[4];
    __shared__ unsigned warpLags[4];
    const size_t i = blockIdx.x;
    val_t best = ZERO;
    unsigned bestLag = 0;
    for (size_t lag = threadIdx.x; lag < timesteps; lag += blockDim.x) {
        const val_t value = correlations[lag * n + i];
        if (value > best) {
            best = value;
            bestLag = static_cast<unsigned>(lag);
        }
    }
    for (unsigned offset = 16; offset; offset /= 2) {
        const val_t other = __shfl_down_sync(0xffffffffu, best, offset);
        const unsigned otherLag = __shfl_down_sync(0xffffffffu, bestLag, offset);
        if (other > best || (other == best && otherLag < bestLag)) {
            best = other;
            bestLag = otherLag;
        }
    }
    const unsigned lane = threadIdx.x % 32;
    const unsigned warp = threadIdx.x / 32;
    if (lane == 0) {
        warpValues[warp] = best;
        warpLags[warp] = bestLag;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned k = 1; k < 4; ++k) {
            if (warpValues[k] > best || (warpValues[k] == best && warpLags[k] < bestLag)) {
                best = warpValues[k];
                bestLag = warpLags[k];
            }
        }
        distances[i] = WAVE_SPEED * static_cast<val_t>(bestLag);
    }
}

struct RayBatch {
    DeviceBuffer<RandomCheckpoint> checkpoints;
    DeviceBuffer<val_t> random;
    RandomCheckpoint* hostCheckpoints = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t complete = nullptr;
    bool submitted = false;
    ~RayBatch() {
        if (stream) cudaStreamSynchronize(stream);
        if (complete) cudaEventDestroy(complete);
        if (stream) cudaStreamDestroy(stream);
        if (hostCheckpoints) cudaFreeHost(hostCheckpoints);
    }
    void initialize() {
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create ray stream");
        cudaCheck(cudaEventCreateWithFlags(&complete, cudaEventDisableTiming), "create ray event");
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&hostCheckpoints),
                                 MAX_CHECKPOINTS * sizeof(RandomCheckpoint)), "allocate random checkpoints");
        checkpoints.allocate(MAX_CHECKPOINTS);
        random.allocate(PAIR_BATCH * NUM_RAYS * 4);
    }
};

struct PropagationGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~PropagationGraph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};

struct CudaSimulation {
    DeviceBuffer<Triangle> triangles;
    DeviceBuffer<Vec3> centers;
    DeviceBuffer<DeviceOctreeNode> nodes;
    DeviceBuffer<uint32_t> triangleIndices, activePrefix;
    DeviceBuffer<val_t> areas, rho, kij, radiosity, distances;
    DeviceBuffer<int> tau;
    RayBatch batches[2];
    PropagationGraph propagation;
    std::vector<uint32_t> batchTotals;

    explicit CudaSimulation(const SimulationState& state) {
        const size_t n = state.numTriangles;
        const size_t pairs = checkedProduct(n, n);
        int device;
        cudaCheck(cudaGetDevice(&device), "get CUDA device");
        cudaDeviceProp properties;
        cudaCheck(cudaGetDeviceProperties(&properties, device), "get CUDA device properties");
        printf("CUDA device: %s\n", properties.name);
        triangles.allocate(n);
        triangles.upload(state.triangles.data(), n);
        centers.allocate(n);
        areas.allocate(n);
        areas.upload(state.areas.data(), n);
        rho.allocate(n);
        rho.upload(state.rho.data(), n);
        kij.allocate(pairs);
        tau.allocate(pairs);
        activePrefix.allocate(pairs);
        radiosity.allocate(checkedProduct(n, state.numTimesteps));
        distances.allocate(n);
        cudaCheck(cudaMemset(kij.get(), 0, pairs * sizeof(val_t)), "initialize form factors");
        cudaCheck(cudaMemset(distances.get(), 0, n * sizeof(val_t)), "initialize distances");
        if (state.numTimesteps) {
            cudaCheck(cudaMemset(radiosity.get(), 0, n * state.numTimesteps * sizeof(val_t)), "initialize radiosity");
        }
        std::vector<DeviceOctreeNode> hostNodes;
        std::vector<uint32_t> hostIndices;
        flattenOctree(state.octree, hostNodes, hostIndices);
        nodes.allocate(hostNodes.size());
        nodes.upload(hostNodes.data(), hostNodes.size());
        triangleIndices.allocate(hostIndices.size());
        triangleIndices.upload(hostIndices.data(), hostIndices.size());
        batchTotals.resize((pairs + PAIR_BATCH - 1) / PAIR_BATCH);
        for (auto& batch : batches) batch.initialize();
        centersKernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(triangles.get(), centers.get(), n);
        cudaCheck(cudaGetLastError(), "launch triangle centers");
        cudaCheck(cudaDeviceSynchronize(), "initialize CUDA simulation");
        if (state.numTimesteps) {
            // Record the causal kernel sequence once, so propagation needs one
            // host launch regardless of the number of timesteps.
            cudaCheck(cudaStreamBeginCapture(batches[0].stream, cudaStreamCaptureModeThreadLocal),
                      "begin propagation graph");
            for (size_t time = 0; time < state.numTimesteps; ++time) {
                propagateKernel<<<static_cast<unsigned>((n + 3) / 4), 128, 0, batches[0].stream>>>(
                    kij.get(), tau.get(), areas.get(), rho.get(), radiosity.get(),
                    n, time, state.numTimesteps / 2, state.sourceIndex);
            }
            cudaCheck(cudaGetLastError(), "record propagation kernels");
            cudaCheck(cudaStreamEndCapture(batches[0].stream, &propagation.graph), "end propagation graph");
            cudaCheck(cudaGraphInstantiate(&propagation.executable, propagation.graph, nullptr, nullptr, 0),
                      "instantiate propagation graph");
        }
    }
};

unsigned linearBlocks(size_t count, unsigned threads = 256) {
    return static_cast<unsigned>(std::min<size_t>((count + threads - 1) / threads, 65535));
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    auto& gpu = *state.gpu;
    const size_t pairs = state.numTriangles * state.numTriangles;
    geometryKernel<<<linearBlocks(pairs), 256>>>(gpu.triangles.get(), gpu.centers.get(), gpu.tau.get(),
                                                gpu.activePrefix.get(), state.numTriangles);
    cudaCheck(cudaGetLastError(), "launch time delays and pair culling");
    size_t bytes = 0;
    cudaCheck(cub::DeviceScan::InclusiveSum(nullptr, bytes, gpu.activePrefix.get(), gpu.activePrefix.get(),
                                           static_cast<int>(std::min(pairs, SCAN_PAIRS))), "size random stream scan");
    DeviceBuffer<unsigned char> scratch;
    scratch.allocate(bytes);
    for (size_t begin = 0; begin < pairs; begin += SCAN_PAIRS) {
        size_t available = bytes;
        cudaCheck(cub::DeviceScan::InclusiveSum(scratch.get(), available, gpu.activePrefix.get() + begin,
                                               gpu.activePrefix.get() + begin,
                                               static_cast<int>(std::min(pairs - begin, SCAN_PAIRS))),
                  "scan random stream offsets");
    }
    DeviceBuffer<uint32_t> totals;
    totals.allocate(gpu.batchTotals.size());
    batchTotalsKernel<<<linearBlocks(gpu.batchTotals.size()), 256>>>(gpu.activePrefix.get(), totals.get(),
                                                                   pairs, gpu.batchTotals.size());
    cudaCheck(cudaGetLastError(), "launch batch totals");
    totals.download(gpu.batchTotals.data(), gpu.batchTotals.size());
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    auto& gpu = *state.gpu;
    OrderedRandomStream randomStream;
    uint32_t previousTotal = 0;
    const size_t pairs = state.numTriangles * state.numTriangles;
    for (size_t b = 0; b < gpu.batchTotals.size(); ++b) {
        if (b % SCAN_BATCHES == 0) previousTotal = 0;
        const uint32_t total = gpu.batchTotals[b];
        const unsigned count = (total - previousTotal) * NUM_RAYS * 4;
        if (count) {
            RayBatch& batch = gpu.batches[b % 2];
            if (batch.submitted) cudaCheck(cudaEventSynchronize(batch.complete), "wait for ray batch");
            const unsigned chunks = randomStream.checkpoints(batch.hostCheckpoints, count);
            cudaCheck(cudaMemcpyAsync(batch.checkpoints.get(), batch.hostCheckpoints,
                                      chunks * sizeof(RandomCheckpoint), cudaMemcpyHostToDevice, batch.stream),
                      "upload random checkpoints");
            expandRandomKernel<<<chunks, 256, 0, batch.stream>>>(batch.checkpoints.get(), batch.random.get(), count);
            cudaCheck(cudaGetLastError(), "launch ordered random expansion");
            const size_t begin = b * PAIR_BATCH;
            const unsigned batchPairs = static_cast<unsigned>(std::min<size_t>(PAIR_BATCH, pairs - begin));
            formFactorsKernel<<<(batchPairs * NUM_RAYS + 255) / 256, 256, 0, batch.stream>>>(
                gpu.triangles.get(), gpu.nodes.get(), gpu.triangleIndices.get(), gpu.activePrefix.get(),
                batch.random.get(), gpu.kij.get(), state.numTriangles, begin, batchPairs, previousTotal);
            cudaCheck(cudaGetLastError(), "launch form factors");
            cudaCheck(cudaEventRecord(batch.complete, batch.stream), "record ray batch completion");
            batch.submitted = true;
        }
        previousTotal = total;
    }
    for (auto& batch : gpu.batches) cudaCheck(cudaStreamSynchronize(batch.stream), "finish form factors");
    // Prefix offsets are no longer needed once all visibility rays have finished.
    gpu.activePrefix.allocate(0);
    printf("  Progress: %zu/%zu triangles\n", state.numTriangles, state.numTriangles);
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    auto& gpu = *state.gpu;
    // Off-diagonal delays on this mesh are strictly positive. A kernel boundary
    // between timesteps therefore preserves all causal dependencies. A warp
    // loads neighboring emitters together, then adds their contributions in
    // the original order. Empty tiles need no additions.
    if (gpu.propagation.executable) {
        cudaCheck(cudaGraphLaunch(gpu.propagation.executable, nullptr), "launch wave propagation graph");
    }
    for (size_t time = 10; time <= state.numTimesteps; time += 10) {
        printf("  Timestep %zu/%zu\n", time, state.numTimesteps);
    }
    if (state.numTimesteps % 10) {
        printf("  Timestep %zu/%zu\n", state.numTimesteps, state.numTimesteps);
    }
    cudaCheck(cudaDeviceSynchronize(), "finish wave propagation");
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    auto& gpu = *state.gpu;
    const size_t count = state.numTriangles * state.numTimesteps;
    if (count) {
        DeviceBuffer<val_t> correlations;
        correlations.allocate(count);
        correlationKernel<<<linearBlocks(count, 128), 128>>>(gpu.radiosity.get(), correlations.get(),
                                                            state.numTriangles, state.numTimesteps, state.sourceIndex);
        cudaCheck(cudaGetLastError(), "launch cross-correlation");
        bestDistanceKernel<<<static_cast<unsigned>(state.numTriangles), 128>>>(
            correlations.get(), gpu.distances.get(), state.numTriangles, state.numTimesteps);
        cudaCheck(cudaGetLastError(), "launch correlation maxima");
    }
    gpu.distances.download(state.distances.data(), state.numTriangles);
}

void downloadValidationData(SimulationState& state) {
    const size_t n = state.numTriangles;
    const size_t pairs = n * n;
    state.kij.resize(pairs);
    state.tau.resize(pairs);
    state.radB.resize(n * state.numTimesteps);
    state.gpu->kij.download(state.kij.data(), pairs);
    state.gpu->tau.download(state.tau.data(), pairs);
    state.gpu->radiosity.download(state.radB.data(), state.radB.size());
}

// ============================================================================

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
        uint32_t bits;
        std::memcpy(&bits, &state.distances[i], sizeof(bits));
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

int runBenchmark(int argc, char** argv) {
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

    state.gpu = std::make_unique<CudaSimulation>(state);

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
        downloadValidationData(state);
        if (!validateResults(state)) {
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    try {
        return runBenchmark(argc, argv);
    } catch (const std::exception& error) {
        fprintf(stderr, "roomsim: %s\n", error.what());
        return 1;
    }
}
