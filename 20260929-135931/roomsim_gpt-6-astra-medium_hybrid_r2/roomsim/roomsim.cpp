/**
 * Room Response Simulation Benchmark
 * 
 * This is a hybrid MPI/OpenMP/CUDA implementation of room impulse response
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
#include <climits>
#include <cfloat>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// MPI calls are confined to the main thread; each process owns one local GPU.
int worldRank = 0, worldSize = 1;
void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", worldRank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)
#define HD __host__ __device__

struct ParallelRuntime {
    ParallelRuntime(int& argc, char**& argv) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
        if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &local);
        int localRank;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_free(&local);
        int devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (!devices) {
            fprintf(stderr, "roomsim requires a CUDA device on every MPI rank.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % devices));
        // Respect explicit OpenMP settings and MPI CPU affinity.
        if (!getenv("OMP_NUM_THREADS")) omp_set_num_threads(std::min(8, omp_get_num_procs()));
    }
    ~ParallelRuntime() { MPI_Finalize(); }
};

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void allocate(size_t n) {
        if (n) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
};

template<class T> struct PinnedBuffer {
    T* data = nullptr;
    explicit PinnedBuffer(size_t n) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data), std::max(size_t(1), n) * sizeof(T)));
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    ~PinnedBuffer() { cudaFreeHost(data); }
};

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

    HD Triangle() = default;
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
// Random Number Generation
// ============================================================================

// MT19937 with the same sequence as std::mt19937, using packed 32-bit words.
// Splitting the twist at its dependence boundaries makes checkpoint advancement
// vectorizable. Fixed-width checkpoints can also be sent with MPI_UINT32_T.
class RandomGenerator {
    struct Engine {
        using result_type = uint32_t;
        std::array<uint32_t, 625> state{}; // last word is the next output index
        static constexpr result_type min() { return 0; }
        static constexpr result_type max() { return UINT32_MAX; }
        explicit Engine(uint32_t seed) {
            state[0] = seed;
            for (uint32_t i = 1; i < 624; ++i)
                state[i] = 1812433253U * (state[i-1] ^ (state[i-1] >> 30)) + i;
            state[624] = 624;
        }
        void twist() {
            #pragma omp simd
            for (int k = 0; k < 227; ++k) {
                uint32_t y = (state[k] & 0x80000000U) | (state[k+1] & 0x7fffffffU);
                state[k] = state[k+397] ^ (y >> 1) ^ ((0U - (y & 1U)) & 0x9908b0dfU);
            }
            #pragma omp simd
            for (int k = 227; k < 454; ++k) {
                uint32_t y = (state[k] & 0x80000000U) | (state[k+1] & 0x7fffffffU);
                state[k] = state[k-227] ^ (y >> 1) ^ ((0U - (y & 1U)) & 0x9908b0dfU);
            }
            #pragma omp simd
            for (int k = 454; k < 623; ++k) {
                uint32_t y = (state[k] & 0x80000000U) | (state[k+1] & 0x7fffffffU);
                state[k] = state[k-227] ^ (y >> 1) ^ ((0U - (y & 1U)) & 0x9908b0dfU);
            }
            uint32_t y = (state[623] & 0x80000000U) | (state[0] & 0x7fffffffU);
            state[623] = state[396] ^ (y >> 1) ^ ((0U - (y & 1U)) & 0x9908b0dfU);
            state[624] = 0;
        }
        result_type operator()() {
            if (state[624] == 624) twist();
            uint32_t y = state[state[624]++];
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c5680U;
            y ^= (y << 15) & 0xefc60000U;
            return y ^ (y >> 18);
        }
        void discard(size_t count) {
            while (count) {
                if (state[624] == 624) twist();
                size_t step = std::min(count, size_t(624 - state[624]));
                state[624] += step;
                count -= step;
            }
        }
    } rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
    void discard(size_t count) { rng.discard(count); }
    uint32_t* checkpoint() { return rng.state.data(); }
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
HD val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t cosine = v.dot(normal) / vNorm;
    return cosine > ZERO ? cosine : ZERO;
}

// A preorder octree with escape links permits stackless traversal on the GPU.
struct FlatNode {
    Vec3 center, halfExtent;
    int escape, first, count;
};
int flattenTree(const Octree& tree, std::vector<FlatNode>& nodes, std::vector<size_t>& indices) {
    int index = static_cast<int>(nodes.size());
    nodes.push_back({tree.center, tree.halfExtent, 0, static_cast<int>(indices.size()),
                     static_cast<int>(tree.triangleIndices.size())});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children) if (child) flattenTree(*child, nodes, indices);
    nodes[index].escape = static_cast<int>(nodes.size());
    return index;
}

__device__ bool intersectsBox(const Vec3& p1, const Vec3& p2, const FlatNode& node) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    const Vec3& h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y || fabsf(c.z) > h.z + ad.z) return false;
    if (fabsf(d.y*c.z-d.z*c.y) > h.y*ad.z+h.z*ad.y+EPSILON) return false;
    if (fabsf(d.z*c.x-d.x*c.z) > h.z*ad.x+h.x*ad.z+EPSILON) return false;
    if (fabsf(d.x*c.y-d.y*c.x) > h.x*ad.y+h.y*ad.x+EPSILON) return false;
    return true;
}

__device__ bool blocked(const Vec3& from, const Vec3& to, size_t i, size_t j,
                        const Triangle* triangles, const FlatNode* nodes, const size_t* indices) {
    Vec3 dir = to - from;
    val_t length = dir.norm();
    if (length < EPSILON) return true;
    dir = dir / length;
    int node = 0;
    while (node < nodes[0].escape) {
        const FlatNode& box = nodes[node];
        // The original traversal always visits the root, including a root leaf.
        if (node != 0 && !intersectsBox(from, to, box)) { node = box.escape; continue; }
        for (int k = 0; k < box.count; ++k) {
            size_t other = indices[box.first + k];
            if (other == i || other == j) continue;
            const Triangle& tri = triangles[other];
            val_t distance = rayTriangleIntersect(from, dir, tri.a, tri.b, tri.c);
            if (distance > EPSILON && distance < length - EPSILON) return true;
        }
        ++node;
    }
    return false;
}

__device__ Vec3 samplePoint(const Triangle& tri, val_t u, val_t v) {
    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
    return tri.a + (tri.b - tri.a) * u + (tri.c - tri.a) * v;
}

// One warp per pair, one lane per visibility sample. Lane zero adds samples in
// the original order, avoiding a floating-point reduction's change of semantics.
__global__ void formFactorsKernel(const Triangle* triangles, const FlatNode* nodes,
    const size_t* indices, const val_t* random, val_t* kij, int* tau, int* minimumDelay,
    size_t n, size_t localRows, size_t firstRow, size_t batchRow, size_t pairs) {
    size_t pair = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    unsigned lane = threadIdx.x % 32;
    if (pair >= pairs) return;
    size_t localI = batchRow + pair / n, j = pair % n, i = firstRow + localI;
    const Triangle& a = triangles[i];
    const Triangle& b = triangles[j];
    val_t value = ZERO;
    bool active = i != j && a.normal().dot(b.normal()) <= 0.99f;
    if (active && lane < NUM_RAYS) {
        const val_t* r = random + pair * (4 * NUM_RAYS) + lane * 4;
        Vec3 p = samplePoint(a, r[0], r[1]);
        Vec3 q = samplePoint(b, r[2], r[3]);
        if (!blocked(p, q, i, j, triangles, nodes, indices)) {
            Vec3 v = q - p;
            val_t d2 = v.squaredNorm();
            if (d2 >= EPSILON) {
                val_t ci = cosPhi(v, a.normal()), cj = cosPhi(-v, b.normal());
                if (ci > ZERO && cj > ZERO) value = (ci * cj) / (PI * d2);
            }
        }
    }
    val_t sum = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) sum += __shfl_sync(0xffffffff, value, r);
    if (lane == 0) {
        size_t at = j * localRows + localI;
        kij[at] = sum * INV_NUM_RAYS;
        int delay = i == j ? 0 : static_cast<int>(ceilf((a.center() - b.center()).norm() * INV_WAVE_SPEED));
        tau[at] = delay;
        if (kij[at] > ZERO) atomicMin(minimumDelay, delay);
    }
}

__global__ void propagateKernel(const val_t* kij, const int* tau, const val_t* areas,
    val_t* history, val_t* outgoing, size_t n, size_t rows, size_t first, size_t start,
    size_t steps, size_t totalSteps, size_t source, val_t rho) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * steps) return;
    size_t i = index % rows, t = start + index / rows;
    val_t sum = ZERO;
    for (size_t j = 0; j < n; ++j) {
        if (j == first + i) continue;
        size_t at = j * rows + i;
        int delay = tau[at];
        if (t < size_t(delay)) continue;
        val_t k = kij[at];
        if (k <= ZERO) continue;
        val_t rad = history[(t - delay) * n + j];
        if (rad <= ZERO) continue;
        sum += fminf(k * areas[j], ONE) * rad;
    }
    val_t emission = first + i == source && t < totalSteps / 2 ? ONE : ZERO;
    val_t value = rho * sum + emission;
    history[t * n + first + i] = value;
    outgoing[index] = value;
}

__global__ void unpackHistory(const val_t* packed, val_t* history, size_t n,
                              size_t start, size_t steps, int ranks) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n * steps) return;
    size_t j = index % n, t = index / n;
    size_t owner = ((j + 1) * ranks - 1) / n;
    size_t first = n * owner / ranks, last = n * (owner + 1) / ranks;
    history[(start + t) * n + j] = packed[steps * first + t * (last - first) + j - first];
}

__global__ void correlationKernel(const val_t* history, val_t* correlations, size_t n,
                                  size_t rows, size_t first, size_t steps, size_t source) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * steps) return;
    size_t i = first + index % rows, lag = index / rows;
    val_t sum = ZERO;
    for (size_t tt = lag; tt < steps; ++tt)
        sum += history[(tt - lag) * n + source] * history[tt * n + i];
    correlations[index] = sum;
}

__global__ void distancesKernel(const val_t* correlations, val_t* distances, size_t rows, size_t steps) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    val_t maximum = ZERO;
    size_t best = 0;
    for (size_t t = 0; t < steps; ++t) {
        val_t value = correlations[t * rows + i];
        if (value > maximum) { maximum = value; best = t; }
    }
    distances[i] = WAVE_SPEED * static_cast<val_t>(best);
}

struct SimulationState {
    size_t numTriangles, numTimesteps, sourceIndex, firstRow, localRows;
    val_t reflectivity;
    int minimumDelay = 1;
    unsigned long long nonZeroKij = 0;
    std::vector<Triangle> triangles;
    std::vector<val_t> areas, radB, distances;
    Octree octree;
    DeviceBuffer<Triangle> dTriangles;
    DeviceBuffer<FlatNode> dNodes;
    DeviceBuffer<size_t> dIndices;
    DeviceBuffer<val_t> dKij, dAreas, dHistory;
    DeviceBuffer<int> dTau;
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    const size_t n = state.triangles.size();
    state.numTriangles = n;
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % n;
    state.reflectivity = reflectivity;
    state.firstRow = n * worldRank / worldSize;
    state.localRows = n * (worldRank + 1) / worldSize - state.firstRow;
    if (worldRank == 0) printf("Generated icosphere mesh with %zu triangles\nBuilding octree...\n", n);
    state.octree.build(state.triangles);
    std::vector<FlatNode> nodes;
    std::vector<size_t> indices;
    flattenTree(state.octree, nodes, indices);
    state.areas.resize(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) state.areas[i] = state.triangles[i].area();
    state.distances.resize(n);
    state.dTriangles.allocate(n);
    state.dNodes.allocate(nodes.size());
    state.dIndices.allocate(indices.size());
    state.dAreas.allocate(n);
    state.dKij.allocate(n * state.localRows);
    state.dTau.allocate(n * state.localRows);
    state.dHistory.allocate(n * timesteps);
    CUDA_CHECK(cudaMemcpy(state.dTriangles.data, state.triangles.data(), n*sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dNodes.data, nodes.data(), nodes.size()*sizeof(FlatNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dIndices.data, indices.data(), indices.size()*sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dAreas.data, state.areas.data(), n*sizeof(val_t), cudaMemcpyHostToDevice));
    if (timesteps) CUDA_CHECK(cudaMemset(state.dHistory.data, 0, n*timesteps*sizeof(val_t)));
}

void computeFormFactors(SimulationState& state) {
    if (worldRank == 0) printf("Computing time delays and form factors on CUDA devices...\n");
    const size_t n = state.numTriangles, rows = state.localRows;
    // Snapshot the exact sequential MT19937 stream at row boundaries. Passing
    // the state between ranks avoids repeating the prefix on every process.
    RandomGenerator rng(42);
    std::vector<RandomGenerator> rowRng;
    rowRng.reserve(rows);
    std::vector<size_t> activeCounts(rows);
    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < rows; ++row) {
        size_t i = state.firstRow + row, count = 0;
        for (size_t j = 0; j < n; ++j)
            if (i != j && state.triangles[i].normal().dot(state.triangles[j].normal()) <= 0.99f) ++count;
        activeCounts[row] = count;
    }
    if (worldRank) MPI_Recv(rng.checkpoint(), 625, MPI_UINT32_T, worldRank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (size_t row = 0; row < rows; ++row) {
        rowRng.push_back(rng);
        rng.discard(activeCounts[row] * (4 * NUM_RAYS));
    }
    if (worldRank+1 < worldSize) MPI_Send(rng.checkpoint(), 625, MPI_UINT32_T, worldRank+1, 0, MPI_COMM_WORLD);

    // Bounded, double-buffered sample tiles: OpenMP prepares the next tile while
    // CUDA traces the previous one. No N*N*NUM_RAYS allocation is needed.
    const size_t tileRows = std::max(size_t(1), std::min(size_t(32), size_t(262144) / n));
    const size_t tilePairs = tileRows * n, randomCount = tilePairs * (4 * NUM_RAYS);
    PinnedBuffer<val_t> host0(randomCount), host1(randomCount);
    val_t* host[2] = {host0.data, host1.data};
    DeviceBuffer<val_t> random[2];
    cudaStream_t streams[2];
    for (int k = 0; k < 2; ++k) {
        random[k].allocate(randomCount);
        CUDA_CHECK(cudaStreamCreate(&streams[k]));
    }
    DeviceBuffer<int> minimum;
    minimum.allocate(1);
    int minDelay = INT_MAX;
    CUDA_CHECK(cudaMemcpy(minimum.data, &minDelay, sizeof(int), cudaMemcpyHostToDevice));
    for (size_t first = 0, batch = 0; first < rows; first += tileRows, ++batch) {
        int slot = batch % 2;
        CUDA_CHECK(cudaStreamSynchronize(streams[slot]));
        size_t count = std::min(tileRows, rows-first);
        #pragma omp parallel for schedule(static)
        for (size_t row = 0; row < count; ++row) {
            RandomGenerator generator = rowRng[first+row];
            size_t i = state.firstRow + first + row;
            for (size_t j = 0; j < n; ++j) {
                if (i == j || state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) continue;
                val_t* dest = host[slot] + (row*n+j) * (4 * NUM_RAYS);
                for (int k = 0; k < 4 * NUM_RAYS; ++k) dest[k] = generator.rand();
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(random[slot].data, host[slot], count*n*(4*NUM_RAYS)*sizeof(val_t),
                                   cudaMemcpyHostToDevice, streams[slot]));
        formFactorsKernel<<<(count*n+3)/4, 128, 0, streams[slot]>>>(
            state.dTriangles.data, state.dNodes.data, state.dIndices.data, random[slot].data,
            state.dKij.data, state.dTau.data, minimum.data, n, rows, state.firstRow, first, count*n);
        CUDA_CHECK(cudaGetLastError());
    }
    for (int k = 0; k < 2; ++k) {
        CUDA_CHECK(cudaStreamSynchronize(streams[k]));
        CUDA_CHECK(cudaStreamDestroy(streams[k]));
    }
    CUDA_CHECK(cudaMemcpy(&minDelay, minimum.data, sizeof(int), cudaMemcpyDeviceToHost));
    MPI_Allreduce(&minDelay, &state.minimumDelay, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (state.minimumDelay == INT_MAX) state.minimumDelay = std::max(size_t(1), state.numTimesteps);
}

void runSimulation(SimulationState& state) {
    if (worldRank == 0) printf("Running distributed CUDA wave propagation...\n");
    size_t n = state.numTriangles, rows = state.localRows, total = state.numTimesteps;
    // Positive geometric delays let several timesteps be evaluated independently
    // and exchanged in one collective. Limit MPI counts to the portable int API.
    size_t batchSize = std::min(total, size_t(std::max(1, state.minimumDelay)));
    batchSize = std::min(batchSize, size_t(INT_MAX) / n);
    if (!total) return;
    DeviceBuffer<val_t> outgoing, incoming;
    outgoing.allocate(rows * batchSize);
    if (worldSize > 1) incoming.allocate(n * batchSize);
    PinnedBuffer<val_t> send(rows * batchSize), receive(worldSize > 1 ? n * batchSize : 1);
    std::vector<int> counts(worldSize), offsets(worldSize);
    for (size_t start = 0; start < total; start += batchSize) {
        size_t steps = std::min(batchSize, total-start);
        if (rows) {
            propagateKernel<<<(rows*steps+127)/128, 128>>>(state.dKij.data, state.dTau.data,
                state.dAreas.data, state.dHistory.data, outgoing.data, n, rows, state.firstRow,
                start, steps, total, state.sourceIndex, state.reflectivity);
            CUDA_CHECK(cudaGetLastError());
        }
        if (worldSize > 1) {
            if (rows) CUDA_CHECK(cudaMemcpy(send.data, outgoing.data, rows*steps*sizeof(val_t), cudaMemcpyDeviceToHost));
            for (int rank = 0; rank < worldSize; ++rank) {
                offsets[rank] = steps * (n * rank / worldSize);
                counts[rank] = steps * (n * (rank+1) / worldSize - n * rank / worldSize);
            }
            MPI_Allgatherv(send.data, rows*steps, MPI_FLOAT, receive.data, counts.data(), offsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(incoming.data, receive.data, n*steps*sizeof(val_t), cudaMemcpyHostToDevice));
            unpackHistory<<<(n*steps+255)/256, 256>>>(incoming.data, state.dHistory.data, n, start, steps, worldSize);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
}

void computeDistances(SimulationState& state) {
    if (worldRank == 0) printf("Computing cross-correlations on CUDA devices...\n");
    size_t rows = state.localRows, steps = state.numTimesteps, n = state.numTriangles;
    DeviceBuffer<val_t> correlations, distances;
    correlations.allocate(rows*steps);
    distances.allocate(rows);
    if (rows) {
        if (steps) {
            correlationKernel<<<(rows*steps+127)/128,128>>>(state.dHistory.data, correlations.data,
                n, rows, state.firstRow, steps, state.sourceIndex);
            CUDA_CHECK(cudaGetLastError());
        }
        distancesKernel<<<(rows+127)/128,128>>>(correlations.data, distances.data, rows, steps);
        CUDA_CHECK(cudaGetLastError());
    }
    std::vector<val_t> local(rows);
    if (rows) CUDA_CHECK(cudaMemcpy(local.data(), distances.data, rows*sizeof(val_t), cudaMemcpyDeviceToHost));
    std::vector<int> counts(worldSize), offsets(worldSize);
    for (int rank = 0; rank < worldSize; ++rank) {
        offsets[rank] = n * rank / worldSize;
        counts[rank] = n * (rank+1) / worldSize - offsets[rank];
    }
    MPI_Gatherv(local.data(), rows, MPI_FLOAT, state.distances.data(), counts.data(), offsets.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
}

__global__ void countFactors(const val_t* kij, size_t count, unsigned long long* result) {
    unsigned long long local = 0;
    for (size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x; i < count; i += size_t(gridDim.x)*blockDim.x)
        if (kij[i] > EPSILON) ++local;
    __shared__ unsigned long long sums[256];
    sums[threadIdx.x] = local;
    __syncthreads();
    for (unsigned stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x+stride];
        __syncthreads();
    }
    if (!threadIdx.x) atomicAdd(result, sums[0]);
}

void collectValidation(SimulationState& state) {
    DeviceBuffer<unsigned long long> count;
    count.allocate(1);
    CUDA_CHECK(cudaMemset(count.data, 0, sizeof(unsigned long long)));
    countFactors<<<256,256>>>(state.dKij.data, state.numTriangles*state.localRows, count.data);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long local;
    CUDA_CHECK(cudaMemcpy(&local, count.data, sizeof(local), cudaMemcpyDeviceToHost));
    MPI_Reduce(&local, &state.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (worldRank == 0) {
        state.radB.resize(state.numTriangles*state.numTimesteps);
        if (!state.radB.empty()) CUDA_CHECK(cudaMemcpy(state.radB.data(), state.dHistory.data,
            state.radB.size()*sizeof(val_t), cudaMemcpyDeviceToHost));
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

int main(int argc, char** argv) {
    ParallelRuntime runtime(argc, argv);
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
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) printf("Unknown option: %s\n", argv[i]);
            if (worldRank == 0) printUsage(argv[0]);
            return 1;
        }
    }

    if (timesteps < 0) {
        if (worldRank == 0) fprintf(stderr, "Timesteps must be non-negative.\n");
        return 1;
    }
    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (worldRank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("MPI ranks: %d, OpenMP threads per rank: %d\n", worldSize, omp_get_max_threads());
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
    }

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    if (worldRank == 0) printf("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    if (worldRank == 0) printf("Precomputation time: %ld ms\n", preDuration);
    if (worldRank == 0) printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    if (worldRank == 0) printf("Simulation time: %ld ms\n", simDuration);
    if (worldRank == 0) printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    if (worldRank == 0) printf("Distance computation time: %ld ms\n", distDuration);
    if (worldRank == 0) printf("\n");

    if (validate) collectValidation(state);
    int exitCode = 0;
    if (worldRank == 0) {
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

        // Persistent matrices on rank zero (sampling and communication use bounded tiles).
        size_t memKij = state.localRows * n * sizeof(val_t);
        size_t memTau = state.localRows * n * sizeof(int);
        size_t memRad = t * n * sizeof(val_t);
        size_t totalMem = memKij + memTau + memRad;
        printf("  Rank 0 matrix storage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

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
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return exitCode;
}
