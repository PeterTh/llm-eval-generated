/**
 * Room Response Simulation Benchmark
 * 
 * This is a CUDA-parallel implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 */

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
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

    if (std::abs(det) < EPSILON) return FLT_MAX;

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
// Room Response Simulation State
// ============================================================================

// CUDA allocations are owned by the simulation; no CPU execution fallback.
void cudaCheck(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(result));
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void upload(const T* src, size_t count) {
        if (count) cudaCheck(cudaMemcpy(data, src, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* dst, size_t count) const {
        if (count) cudaCheck(cudaMemcpy(dst, data, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

// Depth-first layout with escape links permits stack-free device traversal.
struct DeviceNode {
    Vec3 center, half;
    unsigned begin, count, escape;
};

void flattenTree(const Octree& tree, std::vector<DeviceNode>& nodes,
                 std::vector<unsigned>& indices) {
    size_t at = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent,
                     static_cast<unsigned>(indices.size()),
                     static_cast<unsigned>(tree.triangleIndices.size()), 0});
    for (size_t i : tree.triangleIndices) indices.push_back(static_cast<unsigned>(i));
    for (const auto& child : tree.children)
        if (child) flattenTree(*child, nodes, indices);
    nodes[at].escape = static_cast<unsigned>(nodes.size());
}

struct GpuState {
    DeviceBuffer<Triangle> triangles;
    DeviceBuffer<DeviceNode> nodes;
    DeviceBuffer<unsigned> indices;
    DeviceBuffer<float> kij, areas, rho, radB, distances;
    DeviceBuffer<int> tau;
    GpuState(size_t n, size_t t, size_t nodeCount, size_t indexCount)
        : triangles(n), nodes(nodeCount), indices(indexCount), kij(n*n),
          areas(n), rho(n), radB(t*n), distances(n), tau(n*n) {}
};

struct SimulationState {
    std::unique_ptr<GpuState> gpu;
    bool retainValidationData = false;
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Form factors (N x N matrix, row-major)
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

    // Only validation needs host copies of the dense device results.
    if (state.retainValidationData) {
        state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
        state.radB.resize(timesteps * state.numTriangles, ZERO);
    }
    state.distances.resize(state.numTriangles, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// One block advances the exact standard MT19937 stream. The staged twist
// phases reproduce its in-place recurrence, including the final wraparound.
// Full batches contain a multiple of 624 draws, so no random values are lost.
__global__ void generateRandoms(unsigned* state, float* output, size_t count) {
    __shared__ unsigned mt[624];
    unsigned k = threadIdx.x;
    mt[k] = state[k];
    __syncthreads();
    for (size_t base = 0; base < count; base += 624) {
        unsigned y = 0, next = 0;
        if (k < 227) {
            y = (mt[k] & 0x80000000u) | (mt[k+1] & 0x7fffffffu);
            next = mt[k+397] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfu : 0);
        }
        __syncthreads();
        if (k < 227) mt[k] = next;
        __syncthreads();
        // The second phase itself has a dependency at k=454.
        if (k >= 227 && k < 454) {
            y = (mt[k] & 0x80000000u) | (mt[k+1] & 0x7fffffffu);
            next = mt[k-227] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfu : 0);
        }
        __syncthreads();
        if (k >= 227 && k < 454) mt[k] = next;
        __syncthreads();
        if (k >= 454 && k < 623) {
            y = (mt[k] & 0x80000000u) | (mt[k+1] & 0x7fffffffu);
            next = mt[k-227] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfu : 0);
        }
        __syncthreads();
        if (k >= 454 && k < 623) mt[k] = next;
        __syncthreads();
        if (k == 623) {
            y = (mt[623] & 0x80000000u) | (mt[0] & 0x7fffffffu);
            mt[623] = mt[396] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfu : 0);
        }
        __syncthreads();
        y = mt[k];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= y >> 18;
        if (base + k < count)
            output[base+k] = fminf(static_cast<float>(y) * 0x1p-32f, 0x1.fffffep-1f);
        __syncthreads();
    }
    state[k] = mt[k];
}

__device__ bool intersectsBox(const Vec3& p1, const Vec3& p2, const DeviceNode& node) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3& h = node.half;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y || fabsf(c.z) > h.z + ad.z) return false;
    if (fabsf(d.y*c.z-d.z*c.y) > h.y*ad.z+h.z*ad.y+EPSILON) return false;
    if (fabsf(d.z*c.x-d.x*c.z) > h.z*ad.x+h.x*ad.z+EPSILON) return false;
    if (fabsf(d.x*c.y-d.y*c.x) > h.x*ad.y+h.y*ad.x+EPSILON) return false;
    return true;
}

__device__ bool blocked(const Vec3& from, const Vec3& to, unsigned i, unsigned j,
                       const Triangle* triangles, const DeviceNode* nodes,
                       const unsigned* indices) {
    Vec3 dir = to - from;
    float len = dir.norm();
    if (len < EPSILON) return true;
    dir = dir / len;
    unsigned node = 0;
    while (node < nodes[0].escape) {
        const DeviceNode& box = nodes[node];
        if (node != 0 && !intersectsBox(from, to, box)) {
            node = box.escape;
            continue;
        }
        for (unsigned p = box.begin; p < box.begin + box.count; ++p) {
            unsigned idx = indices[p];
            if (idx == i || idx == j) continue;
            const Triangle& tri = triangles[idx];
            float d = rayTriangleIntersect(from, dir, tri.a, tri.b, tri.c);
            if (d > EPSILON && d < len - EPSILON) return true;
        }
        ++node;
    }
    return false;
}

__device__ Vec3 samplePoint(const Triangle& tri, float u, float v) {
    if (u + v > 1.0f) { u = 1.0f-u; v = 1.0f-v; }
    return tri.a + (tri.b-tri.a)*u + (tri.c-tri.a)*v;
}

// A half warp traces the 16 independent samples for each pair. Shuffle reads
// accumulate in the original ray order rather than changing floating-point sums.
__global__ void formFactorsKernel(const Triangle* triangles, const DeviceNode* nodes,
                                 const unsigned* indices, const size_t* pairs,
                                 const float* randoms, float* kij, size_t n, size_t count) {
    size_t pair = (size_t(blockIdx.x)*blockDim.x+threadIdx.x)/16;
    unsigned lane = threadIdx.x % 16;
    if (pair >= count) return;
    size_t ij = pairs[pair];
    unsigned i = ij/n, j = ij%n;
    const Triangle& a = triangles[i];
    const Triangle& b = triangles[j];
    const float* r = randoms + pair*64 + lane*4;
    Vec3 p = samplePoint(a, r[0], r[1]);
    Vec3 q = samplePoint(b, r[2], r[3]);
    float value = 0;
    if (!blocked(p, q, i, j, triangles, nodes, indices)) {
        Vec3 v = q-p;
        float d2 = v.squaredNorm();
        if (d2 >= EPSILON) {
            float ci = cosPhi(v, a.normal()), cj = cosPhi(-v, b.normal());
            if (ci > 0 && cj > 0) value = (ci*cj)/(PI*d2);
        }
    }
    unsigned mask = 0xffffu << (threadIdx.x & 16);
    float sum = 0;
    for (int r = 0; r < NUM_RAYS; ++r)
        sum += __shfl_sync(mask, value, r, 16);
    if (lane == 0) kij[size_t(j)*n+i] = sum*INV_NUM_RAYS;
}

__global__ void timeDelaysKernel(const Triangle* triangles, int* tau, size_t n) {
    size_t p = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (p >= n*n) return;
    size_t i = p%n, j = p/n;
    tau[p] = i == j ? 0 : static_cast<int>(ceilf((triangles[i].center()-triangles[j].center()).norm()*INV_WAVE_SPEED));
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau) on CUDA...\n");
    std::vector<DeviceNode> nodes;
    std::vector<unsigned> indices;
    flattenTree(state.octree, nodes, indices);
    size_t n = state.numTriangles;
    state.gpu = std::make_unique<GpuState>(n, state.numTimesteps, nodes.size(), indices.size());
    auto& gpu = *state.gpu;
    gpu.triangles.upload(state.triangles.data(), n);
    gpu.nodes.upload(nodes.data(), nodes.size());
    gpu.indices.upload(indices.data(), indices.size());
    gpu.areas.upload(state.areas.data(), n);
    gpu.rho.upload(state.rho.data(), n);
    cudaCheck(cudaMemset(gpu.kij.data, 0, n*n*sizeof(float)));
    timeDelaysKernel<<<(n*n+255)/256, 256>>>(gpu.triangles.data, gpu.tau.data, n);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij) on CUDA...\n");
    // 9984*64 is exactly 1024 MT19937 generations. Scratch space stays
    // bounded even for large meshes; excluded pairs consume no random draws.
    constexpr size_t batchSize = 9984;
    DeviceBuffer<unsigned> mt(624);
    std::array<unsigned, 624> seed;
    seed[0] = 42;
    for (unsigned i = 1; i < 624; ++i)
        seed[i] = 1812433253u*(seed[i-1]^(seed[i-1]>>30))+i;
    mt.upload(seed.data(), seed.size());
    DeviceBuffer<float> randoms(batchSize*64);
    DeviceBuffer<size_t> pairs(batchSize);
    std::vector<size_t> pending;
    pending.reserve(batchSize);
    auto& gpu = *state.gpu;
    size_t n = state.numTriangles;
    auto flush = [&]() {
        if (pending.empty()) return;
        pairs.upload(pending.data(), pending.size());
        generateRandoms<<<1,624>>>(mt.data, randoms.data, pending.size()*64);
        cudaCheck(cudaGetLastError());
        formFactorsKernel<<<(pending.size()*16+127)/128,128>>>(
            gpu.triangles.data, gpu.nodes.data, gpu.indices.data, pairs.data,
            randoms.data, gpu.kij.data, n, pending.size());
        cudaCheck(cudaGetLastError());
        pending.clear();
    };
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j || state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) continue;
            pending.push_back(i*n+j);
            if (pending.size() == batchSize) flush();
        }
    }
    flush();
    cudaCheck(cudaDeviceSynchronize());
    // Keep the public host matrix row-major for validation and inspection.
    if (state.retainValidationData) {
        gpu.kij.download(state.kij.data(), n*n);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i+1; j < n; ++j)
                std::swap(state.kij[i*n+j], state.kij[j*n+i]);
    }
}

// Each receiver sums in emitter order, using column-major matrices so a warp
// reads adjacent receivers. Distinct triangles in this mesh have positive
// delays; separate launches in one stream enforce all timestep dependencies.
__global__ void propagationKernel(const float* kij, const int* tau, const float* areas,
                                  const float* rho, float* radB, size_t n, size_t t,
                                  size_t timesteps, size_t source) {
    size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= n) return;
    float sum = 0;
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        int delay = tau[j*n+i];
        if (t < static_cast<size_t>(delay)) continue;
        float k = kij[j*n+i];
        if (k <= 0) continue;
        float rad = radB[(t-delay)*n+j];
        if (rad <= 0) continue;
        sum += fminf(k*areas[j], ONE)*rad;
    }
    radB[t*n+i] = rho[i]*sum + ((i == source && t < timesteps/2) ? ONE : ZERO);
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation on CUDA...\n");
    auto& gpu = *state.gpu;
    size_t n = state.numTriangles;
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        propagationKernel<<<(n+127)/128,128>>>(gpu.kij.data, gpu.tau.data,
            gpu.areas.data, gpu.rho.data, gpu.radB.data, n, t,
            state.numTimesteps, state.sourceIndex);
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaDeviceSynchronize());
    if (state.retainValidationData)
        gpu.radB.download(state.radB.data(), n*state.numTimesteps);
}

// Parallelize both receiver and lag; each correlation retains its time-order
// sum. The second kernel uses a strict comparison to retain the earliest tie.
__global__ void correlationKernel(const float* radB, float* correlations, size_t n,
                                  size_t timesteps, size_t source) {
    size_t p = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (p >= n*timesteps) return;
    size_t i = p%n, lag = p/n;
    float sum = 0;
    for (size_t tt = lag; tt < timesteps; ++tt)
        sum += radB[(tt-lag)*n+source]*radB[tt*n+i];
    correlations[p] = sum;
}

__global__ void distancesKernel(const float* correlations, float* distances,
                                size_t n, size_t timesteps) {
    size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i >= n) return;
    float maxCorr = 0;
    size_t best = 0;
    for (size_t t = 0; t < timesteps; ++t) {
        float sum = correlations[t*n+i];
        if (sum > maxCorr) { maxCorr = sum; best = t; }
    }
    distances[i] = WAVE_SPEED*static_cast<float>(best);
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation on CUDA...\n");
    auto& gpu = *state.gpu;
    size_t n = state.numTriangles, t = state.numTimesteps;
    DeviceBuffer<float> correlations(n*t);
    if (t) {
        correlationKernel<<<(n*t+127)/128,128>>>(gpu.radB.data, correlations.data,
                                               n, t, state.sourceIndex);
        cudaCheck(cudaGetLastError());
    }
    distancesKernel<<<(n+127)/128,128>>>(correlations.data, gpu.distances.data, n, t);
    cudaCheck(cudaGetLastError());
    gpu.distances.download(state.distances.data(), n);
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

int main(int argc, char** argv) try {
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
    state.retainValidationData = validate;
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
            return 1;
        }
    }

    return 0;
} catch (const std::exception& error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
}
