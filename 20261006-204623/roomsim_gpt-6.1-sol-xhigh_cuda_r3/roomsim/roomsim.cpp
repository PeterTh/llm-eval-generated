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
    __host__ __device__ val_t norm() const { return sqrtf(squaredNorm()); }
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
// CUDA storage and error handling
// ============================================================================

void checkCuda(cudaError_t result, const char* expression, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at line %d (%s): %s\n", line, expression,
                cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(expression) checkCuda((expression), #expression, __LINE__)

template <typename T>
class DeviceArray {
    T* data_ = nullptr;
public:
    DeviceArray() = default;
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    ~DeviceArray() { if (data_) cudaFree(data_); }

    void allocate(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            fprintf(stderr, "Device allocation size overflow\n");
            std::exit(EXIT_FAILURE);
        }
        if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }
    void upload(const std::vector<T>& values) {
        allocate(values.size());
        if (!values.empty()) {
            CUDA_CHECK(cudaMemcpy(data_, values.data(), values.size() * sizeof(T),
                                  cudaMemcpyHostToDevice));
        }
    }
    T* data() const { return data_; }
};

// Depth-first nodes with escape links support stackless traversal on the GPU.
// Triangle duplication, child ordering and overlap tests are unchanged.
struct DeviceOctreeNode {
    Vec3 center, halfExtent;
    uint32_t firstTriangle, triangleCount, escape;
};

void flattenOctree(const Octree& tree, std::vector<DeviceOctreeNode>& nodes,
                   std::vector<idx_t>& indices) {
    size_t index = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, static_cast<uint32_t>(indices.size()),
                     static_cast<uint32_t>(tree.triangleIndices.size()), 0});
    for (size_t triangle : tree.triangleIndices) indices.push_back(static_cast<idx_t>(triangle));
    for (const auto& child : tree.children) {
        if (child) flattenOctree(*child, nodes, indices);
    }
    nodes[index].escape = static_cast<uint32_t>(nodes.size());
}

__device__ bool rayIntersectsBox(const Vec3& from, const Vec3& to,
                                 const DeviceOctreeNode& node) {
    Vec3 d = (to - from) * 0.5f;
    Vec3 c = from + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3& h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y ||
        fabsf(c.z) > h.z + ad.z) return false;
    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;
    return true;
}

__device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                      const Triangle& tri) {
    Vec3 e1 = tri.b - tri.a;
    Vec3 e2 = tri.c - tri.a;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);
    if (fabsf(det) < EPSILON) return NO_INTERSECTION;
    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - tri.a;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return NO_INTERSECTION;
    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return NO_INTERSECTION;
    return e2.dot(qvec) * invDet;
}

__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const Triangle* triangles, const DeviceOctreeNode* nodes,
                             const idx_t* indices, idx_t src, idx_t dst) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;
    uint32_t end = nodes[0].escape;
    for (uint32_t nodeIndex = 0; nodeIndex < end;) {
        const DeviceOctreeNode& node = nodes[nodeIndex];
        // The original traversal visits the root unconditionally.
        if (nodeIndex != 0 && !rayIntersectsBox(from, to, node)) {
            nodeIndex = node.escape;
            continue;
        }
        for (uint32_t k = 0; k < node.triangleCount; ++k) {
            idx_t index = indices[node.firstTriangle + k];
            if (index == src || index == dst) continue;
            val_t distance = rayTriangleIntersect(from, dirNorm, triangles[index]);
            if (distance > EPSILON && distance < rayLen - EPSILON) return true;
        }
        ++nodeIndex;
    }
    return false;
}

// ============================================================================
// Exact original Mersenne Twister stream, expanded in parallel from checkpoints
// ============================================================================

constexpr int MT_WORDS = 624;
constexpr int MT_ROUNDS = 16;
constexpr int RANDOM_BLOCKS = 640;
constexpr size_t DRAWS_PER_PAIR = 4 * NUM_RAYS;
constexpr size_t DRAWS_PER_BLOCK = MT_WORDS * MT_ROUNDS;
static_assert(DRAWS_PER_BLOCK % DRAWS_PER_PAIR == 0);

struct MtState { uint32_t words[MT_WORDS]; };

__host__ __device__ uint32_t mtMix(uint32_t a, uint32_t b) {
    uint32_t y = (a & 0x80000000u) | (b & 0x7fffffffu);
    return (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
}

class RandomCheckpoints {
    MtState state;
public:
    explicit RandomCheckpoints(uint32_t seed) {
        state.words[0] = seed;
        for (uint32_t i = 1; i < MT_WORDS; ++i) {
            uint32_t previous = state.words[i - 1];
            state.words[i] = 1812433253u * (previous ^ (previous >> 30)) + i;
        }
    }
    void advance() {
        // Separate arrays expose the independent portions of the standard
        // in-place MT recurrence to the host compiler's vectorizer.
        MtState next;
        for (int k = 0; k < 227; ++k)
            next.words[k] = state.words[k + 397] ^ mtMix(state.words[k], state.words[k + 1]);
        for (int k = 227; k < 454; ++k)
            next.words[k] = next.words[k - 227] ^ mtMix(state.words[k], state.words[k + 1]);
        for (int k = 454; k < 623; ++k)
            next.words[k] = next.words[k - 227] ^ mtMix(state.words[k], state.words[k + 1]);
        next.words[623] = next.words[396] ^ mtMix(state.words[623], next.words[0]);
        state = next;
    }
    void fill(MtState* checkpoints, size_t count) {
        for (size_t block = 0; block < count; ++block) {
            checkpoints[block] = state;
            for (int round = 0; round < MT_ROUNDS; ++round) advance();
        }
    }
};

__global__ void expandRandomKernel(const MtState* checkpoints, val_t* draws,
                                   size_t drawCount) {
    __shared__ uint32_t storage[2][MT_WORDS];
    uint32_t* current = storage[0];
    uint32_t* next = storage[1];
    int lane = threadIdx.x;
    for (int k = lane; k < MT_WORDS; k += blockDim.x)
        current[k] = checkpoints[blockIdx.x].words[k];
    __syncthreads();

    for (int round = 0; round < MT_ROUNDS; ++round) {
        if (lane < 227)
            next[lane] = current[lane + 397] ^ mtMix(current[lane], current[lane + 1]);
        __syncthreads();
        if (lane < 227) {
            int k = lane + 227;
            next[k] = next[k - 227] ^ mtMix(current[k], current[k + 1]);
        }
        __syncthreads();
        if (lane < 169) {
            int k = lane + 454;
            next[k] = next[k - 227] ^ mtMix(current[k], current[k + 1]);
        }
        if (lane == 169) next[623] = next[396] ^ mtMix(current[623], next[0]);
        __syncthreads();
        uint32_t* swap = current;
        current = next;
        next = swap;

        for (int k = lane; k < MT_WORDS; k += blockDim.x) {
            uint32_t word = current[k];
            word ^= word >> 11;
            word ^= (word << 7) & 0x9d2c5680u;
            word ^= (word << 15) & 0xefc60000u;
            word ^= word >> 18;
            // Match uniform_real_distribution<float>, including its upper-end
            // correction when uint32 -> float rounds up to 2^32.
            val_t value = __uint2float_rn(word) * 0x1p-32f;
            if (value >= ONE) value = 0x1.fffffep-1f;
            size_t offset = static_cast<size_t>(blockIdx.x) * DRAWS_PER_BLOCK +
                            round * MT_WORDS + k;
            if (offset < drawCount) draws[offset] = value;
        }
    }
}

__device__ Vec3 randomPointInTriangle(const Triangle& triangle, val_t u, val_t v) {
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return triangle.a + (triangle.b - triangle.a) * u + (triangle.c - triangle.a) * v;
}

__global__ void formFactorsKernel(const uint64_t* pairs, size_t pairCount,
                                  const val_t* draws, const Triangle* triangles,
                                  const DeviceOctreeNode* nodes, const idx_t* indices,
                                  const val_t* areas, val_t* weights, size_t n,
                                  unsigned long long* nonZeroCount, bool validate) {
    size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t pair = thread / NUM_RAYS;
    int ray = thread % NUM_RAYS;
    val_t contribution = ZERO;
    idx_t i = 0, j = 0;
    bool active = pair < pairCount;
    if (active) {
        uint64_t index = pairs[pair];
        i = static_cast<idx_t>(index / n);
        j = static_cast<idx_t>(index % n);
        const Triangle& triI = triangles[i];
        const Triangle& triJ = triangles[j];
        const val_t* random = draws + pair * DRAWS_PER_PAIR + ray * 4;
        Vec3 pI = randomPointInTriangle(triI, random[0], random[1]);
        Vec3 pJ = randomPointInTriangle(triJ, random[2], random[3]);
        if (!isRayBlocked(pI, pJ, triangles, nodes, indices, i, j)) {
            Vec3 v = pJ - pI;
            val_t distSqr = v.squaredNorm();
            if (distSqr >= EPSILON) {
                val_t length = sqrtf(distSqr);
                val_t cosI = length <= EPSILON ? ZERO : fmaxf(ZERO, v.dot(triI._normal) / length);
                val_t cosJ = length <= EPSILON ? ZERO : fmaxf(ZERO, (-v).dot(triJ._normal) / length);
                if (cosI > ZERO && cosJ > ZERO)
                    contribution = (cosI * cosJ) / (PI * distSqr);
            }
        }
    }
    // Each half warp computes one pair. Sum rays in their original order;
    // a tree reduction would change rounding and subsequent correlation ties.
    val_t kij = ZERO;
    #pragma unroll
    for (int r = 0; r < NUM_RAYS; ++r)
        kij += __shfl_sync(0xffffffffu, contribution, r, NUM_RAYS);
    kij *= INV_NUM_RAYS;
    if (active && ray == 0) weights[static_cast<size_t>(i) * n + j] = fminf(kij * areas[j], ONE);
    if (validate) {
        unsigned mask = __ballot_sync(0xffffffffu, active && ray == 0 && kij > EPSILON);
        if ((threadIdx.x & 31) == 0 && mask) atomicAdd(nonZeroCount, static_cast<unsigned long long>(__popc(mask)));
    }
}

// Double buffering overlaps CPU checkpoint preparation, transfers, and GPU
// expansion/visibility work. Temporary memory is independent of mesh size.
struct FormFactorBatch {
    MtState* hostCheckpoints = nullptr;
    uint64_t* hostPairs = nullptr;
    DeviceArray<MtState> checkpoints;
    DeviceArray<uint64_t> pairs;
    DeviceArray<val_t> draws;
    cudaStream_t stream = nullptr;
    cudaEvent_t done = nullptr;
    bool inFlight = false;

    explicit FormFactorBatch(size_t blocks) {
        size_t capacity = blocks * DRAWS_PER_BLOCK / DRAWS_PER_PAIR;
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostCheckpoints), blocks * sizeof(MtState)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostPairs), capacity * sizeof(uint64_t)));
        checkpoints.allocate(blocks);
        pairs.allocate(capacity);
        draws.allocate(capacity * DRAWS_PER_PAIR);
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    }
    ~FormFactorBatch() {
        if (inFlight) cudaEventSynchronize(done);
        cudaEventDestroy(done);
        cudaStreamDestroy(stream);
        cudaFreeHost(hostPairs);
        cudaFreeHost(hostCheckpoints);
    }
    void wait() {
        if (inFlight) {
            CUDA_CHECK(cudaEventSynchronize(done));
            inFlight = false;
        }
    }
};

__global__ void timeDelaysKernel(const Vec3* centers, int* tau, size_t n) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n * n) return;
    size_t i = index / n, j = index % n;
    tau[index] = i == j ? 0 : static_cast<int>(ceilf((centers[i] - centers[j]).norm() * INV_WAVE_SPEED));
}

__global__ void propagationKernel(const val_t* __restrict__ weights, const int* __restrict__ tau,
                                  const val_t* __restrict__ rho, val_t* __restrict__ radB, size_t n,
                                  size_t time, size_t source, size_t timeOff) {
    size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t i = thread / 32;
    unsigned lane = threadIdx.x & 31;
    if (i >= n) return; // Entire warps agree on this condition.
    val_t sum = ZERO;
    // A warp fetches 32 consecutive emitters together, then accumulates their
    // contributions in increasing j order. This exposes matrix and history
    // loads to parallelism without changing the serial floating-point sum.
    for (size_t first = 0; first < n; first += 32) {
        size_t j = first + lane;
        val_t contribution = ZERO;
        if (j < n && i != j) {
            size_t index = i * n + j;
            int delay = tau[index];
            if (static_cast<int>(time) >= delay) {
                val_t weight = weights[index];
                if (weight > ZERO) {
                    val_t radiosity = radB[(time - static_cast<size_t>(delay)) * n + j];
                    if (!(radiosity <= ZERO)) contribution = weight * radiosity;
                }
            }
        }
        #pragma unroll
        for (int emitter = 0; emitter < 32; ++emitter) {
            val_t value = __shfl_sync(0xffffffffu, contribution, emitter);
            if (lane == 0) sum += value;
        }
    }
    if (lane == 0) {
        val_t emission = i == source && time < timeOff ? ONE : ZERO;
        radB[time * n + i] = rho[i] * sum + emission;
    }
}

__global__ void correlationKernel(const val_t* radB, val_t* correlations,
                                  size_t n, size_t timesteps, size_t source,
                                  size_t firstLag) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t lag = firstLag + static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (i >= n || lag >= timesteps) return;
    val_t sum = ZERO;
    for (size_t tt = lag; tt < timesteps; ++tt)
        sum += radB[(tt - lag) * n + source] * radB[tt * n + i];
    correlations[lag * n + i] = sum;
}

__global__ void distancesKernel(const val_t* correlations, val_t* distances,
                                size_t n, size_t timesteps) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    val_t maxCorr = ZERO;
    int bestT = 0;
    for (size_t t = 0; t < timesteps; ++t) {
        val_t sum = correlations[t * n + i];
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
    size_t numTriangles = 0;
    size_t numTimesteps = 0;
    size_t sourceIndex = 0;
    bool enableValidation = false;
    unsigned long long nonZeroFormFactors = 0;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;
    std::vector<val_t> rho;
    std::vector<val_t> radB;        // Downloaded only for requested validation
    std::vector<val_t> distances;
    Octree octree;

    DeviceArray<Triangle> deviceTriangles;
    DeviceArray<Vec3> centers;
    DeviceArray<DeviceOctreeNode> nodes;
    DeviceArray<idx_t> triangleIndices;
    DeviceArray<val_t> deviceAreas, deviceRho;
    // Receiver-major layout makes a warp's emitter accesses contiguous.
    // Store min(Kij * area[j], 1) once instead of recomputing it each timestep.
    DeviceArray<val_t> weights;
    DeviceArray<int> delays;
    DeviceArray<val_t> radiosity, deviceDistances;
    DeviceArray<unsigned long long> nonZeroCount;

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // CUDA is required; errors are reported instead of falling back to the CPU.
    CUDA_CHECK(cudaFree(nullptr));
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    size_t n = state.numTriangles;
    printf("Generated icosphere mesh with %zu triangles\n", n);
    printf("Building octree...\n");
    state.octree.build(state.triangles);

    std::vector<DeviceOctreeNode> nodes;
    std::vector<idx_t> indices;
    flattenOctree(state.octree, nodes, indices);
    state.nodes.upload(nodes);
    state.triangleIndices.upload(indices);
    state.deviceTriangles.upload(state.triangles);

    std::vector<Vec3> centers(n);
    state.areas.resize(n);
    for (size_t i = 0; i < n; ++i) {
        state.areas[i] = state.triangles[i].area();
        centers[i] = state.triangles[i].center();
    }
    state.centers.upload(centers);
    state.deviceAreas.upload(state.areas);
    state.rho.resize(n, reflectivity);
    state.deviceRho.upload(state.rho);
    state.distances.resize(n, ZERO);

    state.weights.allocate(n * n);
    state.delays.allocate(n * n);
    state.radiosity.allocate(timesteps * n);
    state.deviceDistances.allocate(n);
    state.nonZeroCount.allocate(1);
    CUDA_CHECK(cudaMemset(state.weights.data(), 0, n * n * sizeof(val_t)));
    if (timesteps) CUDA_CHECK(cudaMemset(state.radiosity.data(), 0, timesteps * n * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.nonZeroCount.data(), 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    size_t count = state.numTriangles * state.numTriangles;
    timeDelaysKernel<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
        state.centers.data(), state.delays.data(), state.numTriangles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    RandomCheckpoints rng(42);
    size_t n = state.numTriangles;
    constexpr size_t PAIRS_PER_RANDOM_BLOCK = DRAWS_PER_BLOCK / DRAWS_PER_PAIR;
    size_t randomBlocks = std::min(static_cast<size_t>(RANDOM_BLOCKS),
                                  (n * (n - 1) + PAIRS_PER_RANDOM_BLOCK - 1) / PAIRS_PER_RANDOM_BLOCK);
    size_t capacity = randomBlocks * PAIRS_PER_RANDOM_BLOCK;
    size_t batchCount = n * (n - 1) > capacity ? 2 : 1;
    std::unique_ptr<FormFactorBatch> batches[2];
    for (size_t b = 0; b < batchCount; ++b) batches[b] = std::make_unique<FormFactorBatch>(randomBlocks);
    size_t batchIndex = 0;
    size_t pairCount = 0;

    auto submit = [&]() {
        FormFactorBatch& batch = *batches[batchIndex];
        size_t drawCount = pairCount * DRAWS_PER_PAIR;
        size_t blocks = (drawCount + DRAWS_PER_BLOCK - 1) / DRAWS_PER_BLOCK;
        rng.fill(batch.hostCheckpoints, blocks);
        CUDA_CHECK(cudaMemcpyAsync(batch.checkpoints.data(), batch.hostCheckpoints,
                                  blocks * sizeof(MtState), cudaMemcpyHostToDevice, batch.stream));
        CUDA_CHECK(cudaMemcpyAsync(batch.pairs.data(), batch.hostPairs,
                                  pairCount * sizeof(uint64_t), cudaMemcpyHostToDevice, batch.stream));
        expandRandomKernel<<<static_cast<unsigned>(blocks), 256, 0, batch.stream>>>(
            batch.checkpoints.data(), batch.draws.data(), drawCount);
        CUDA_CHECK(cudaGetLastError());
        formFactorsKernel<<<static_cast<unsigned>((pairCount * NUM_RAYS + 127) / 128),
                            128, 0, batch.stream>>>(
            batch.pairs.data(), pairCount, batch.draws.data(), state.deviceTriangles.data(),
            state.nodes.data(), state.triangleIndices.data(), state.deviceAreas.data(),
            state.weights.data(), n, state.nonZeroCount.data(), state.enableValidation);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(batch.done, batch.stream));
        batch.inFlight = true;
        batchIndex = (batchIndex + 1) % batchCount;
        batches[batchIndex]->wait();
        pairCount = 0;
    };

    // Enumerate in exactly the original row-major order. Culled pairs consume
    // no random values, and full batches end on an MT checkpoint boundary.
    for (size_t i = 0; i < n; ++i) {
        const Vec3& normalI = state.triangles[i]._normal;
        for (size_t j = 0; j < n; ++j) {
            if (i == j || normalI.dot(state.triangles[j]._normal) > 0.99f) continue;
            batches[batchIndex]->hostPairs[pairCount++] = static_cast<uint64_t>(i) * n + j;
            if (pairCount == capacity) submit();
        }
    }
    if (pairCount) submit();
    for (size_t b = 0; b < batchCount; ++b) batches[b]->wait();
    if (state.enableValidation) {
        CUDA_CHECK(cudaMemcpy(&state.nonZeroFormFactors, state.nonZeroCount.data(),
                              sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    }
    printf("  Progress: %zu/%zu triangles\n", n, n);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    // Distinct icosphere centers have strictly positive off-diagonal delays,
    // so every receiver reads only earlier timesteps. Stream ordering provides
    // the global barrier between timesteps without host synchronization.
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        propagationKernel<<<static_cast<unsigned>((state.numTriangles + 3) / 4), 128>>>(
            state.weights.data(), state.delays.data(), state.deviceRho.data(),
            state.radiosity.data(), state.numTriangles, t, state.sourceIndex, state.numTimesteps / 2);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (state.numTimesteps) printf("  Timestep %zu/%zu\n", state.numTimesteps, state.numTimesteps);
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    size_t n = state.numTriangles, timesteps = state.numTimesteps;
    DeviceArray<val_t> correlations;
    correlations.allocate(n * timesteps);
    dim3 threads(32, 8);
    // CUDA's grid.y is limited to 65535. Splitting large lag ranges keeps all
    // command-line timestep counts supported without truncating the grid.
    constexpr size_t LAGS_PER_LAUNCH = 65535 * 8;
    for (size_t first = 0; first < timesteps; first += LAGS_PER_LAUNCH) {
        size_t count = std::min(timesteps - first, LAGS_PER_LAUNCH);
        dim3 blocks(static_cast<unsigned>((n + 31) / 32), static_cast<unsigned>((count + 7) / 8));
        correlationKernel<<<blocks, threads>>>(state.radiosity.data(), correlations.data(),
                                               n, timesteps, state.sourceIndex, first);
        CUDA_CHECK(cudaGetLastError());
    }
    distancesKernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(
        correlations.data(), state.deviceDistances.data(), n, timesteps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.deviceDistances.data(),
                          n * sizeof(val_t), cudaMemcpyDeviceToHost));
    if (state.enableValidation && timesteps) {
        state.radB.resize(n * timesteps);
        CUDA_CHECK(cudaMemcpy(state.radB.data(), state.radiosity.data(),
                              n * timesteps * sizeof(val_t), cudaMemcpyDeviceToHost));
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
    unsigned long long nonZeroKij = state.nonZeroFormFactors;
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

    if (timesteps < 0) {
        fprintf(stderr, "Number of timesteps must be non-negative\n");
        return 1;
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
    state.enableValidation = validate;
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
}
