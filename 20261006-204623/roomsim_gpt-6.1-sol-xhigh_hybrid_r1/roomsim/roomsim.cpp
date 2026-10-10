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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <climits>
#include <cfloat>
#include <cstdarg>
#include <exception>
#include <mpi.h>
#ifdef ROOMSIM_MPI_CUDA_QUERY
#include <mpi-ext.h>
#endif
#include <omp.h>
#include <cuda_runtime.h>
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
                Octree* child = children[i].get();
                #pragma omp task firstprivate(child, i, childMin, childMax) shared(childIndices) if(indices.size() > 128)
                child->buildNode(childIndices[i], childMin, childMax);
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
        triangles.resize(faces.size());
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < faces.size(); ++i) {
            const auto& face = faces[i];
            triangles[i] = Triangle(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Hybrid runtime and device storage
// ============================================================================

int worldRank = 0, worldSize = 1;
bool cudaAwareMPI = false;

void logMessage(const char* format, ...) {
    if (worldRank != 0) return;
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

void mpiCheck(int error, const char* operation) {
    if (error == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(error, message, &length);
    fprintf(stderr, "Rank %d: %s: %.*s\n", worldRank, operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
}
#define MPI_CHECK(call) mpiCheck((call), #call)

void cudaCheck(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    fprintf(stderr, "Rank %d: %s: %s\n", worldRank, operation, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 1);
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

template<typename T> class DeviceBuffer {
    T* data_ = nullptr;
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    void allocate(size_t count) {
        if (count > SIZE_MAX / sizeof(T)) {
            fprintf(stderr, "Device allocation size overflow\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }
    T* get() const { return data_; }
    void upload(const std::vector<T>& values) {
        allocate(values.size());
        if (!values.empty()) CUDA_CHECK(cudaMemcpy(data_, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};

template<typename T> class PinnedBuffer {
    T* data_ = nullptr;
public:
    PinnedBuffer() = default;
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    ~PinnedBuffer() { if (data_) cudaFreeHost(data_); }
    void allocate(size_t count) {
        if (count) CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }
    T* get() const { return data_; }
};

// A preorder octree with subtree escape links gives stackless GPU traversal,
// retaining the original child order, box tests, and triangle tests.
struct FlatNode {
    Vec3 center, halfExtent;
    uint32_t escape, firstTriangle, numTriangles;
};

void flattenOctree(const Octree& tree, std::vector<FlatNode>& nodes, std::vector<idx_t>& indices) {
    size_t position = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0,
                     static_cast<uint32_t>(indices.size()), static_cast<uint32_t>(tree.triangleIndices.size())});
    for (size_t index : tree.triangleIndices) indices.push_back(static_cast<idx_t>(index));
    for (const auto& child : tree.children) {
        if (child) flattenOctree(*child, nodes, indices);
    }
    nodes[position].escape = static_cast<uint32_t>(nodes.size());
}

__device__ bool rayIntersectsBox(const Vec3& p1, const Vec3& p2, const FlatNode& node) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    if (fabsf(c.x) > node.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > node.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > node.halfExtent.z + ad.z) return false;
    if (fabsf(d.y * c.z - d.z * c.y) > node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPSILON) return false;
    return true;
}

__device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir, const Triangle& tri) {
    Vec3 e1 = tri.b - tri.a, e2 = tri.c - tri.a;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);
    if (fabsf(det) < EPSILON) return FLT_MAX;
    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - tri.a;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;
    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;
    return e2.dot(qvec) * invDet;
}

__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const FlatNode* nodes, const idx_t* indices,
                             const Triangle* triangles, size_t src, size_t dst) {
    Vec3 dir = to - from;
    val_t length = dir.norm();
    if (length < EPSILON) return true;
    Vec3 dirNorm = dir / length;
    uint32_t nodeIndex = 0;
    while (nodeIndex < nodes[0].escape) {
        const FlatNode& node = nodes[nodeIndex];
        if (nodeIndex != 0 && !rayIntersectsBox(from, to, node)) {
            nodeIndex = node.escape;
            continue;
        }
        for (uint32_t k = 0; k < node.numTriangles; ++k) {
            idx_t index = indices[node.firstTriangle + k];
            if (index == src || index == dst) continue;
            val_t distance = rayTriangleIntersect(from, dirNorm, triangles[index]);
            if (distance > EPSILON && distance < length - EPSILON) return true;
        }
        ++nodeIndex;
    }
    return false;
}

// MT19937 in its original word recurrence. A checkpoint stores the 624 words
// immediately preceding the next draw, in logical order. Discarding does not
// temper words, and the CUDA blocks generate/temper their own random numbers.
constexpr int MT_WORDS = 624;
constexpr int MT_DEGREE = 19937;
constexpr int POLY_WORDS = (MT_DEGREE + 63) / 64;
using Polynomial = std::array<uint64_t, POLY_WORDS>;

struct MTStream {
    std::array<uint32_t, MT_WORDS> words{};
    int cursor = 0;
    explicit MTStream(uint32_t seed = 42) {
        words[0] = seed;
        for (uint32_t i = 1; i < MT_WORDS; ++i)
            words[i] = 1812433253U * (words[i-1] ^ (words[i-1] >> 30)) + i;
    }
    uint32_t advance() {
        int next = cursor + 1 == MT_WORDS ? 0 : cursor + 1;
        int other = cursor + 397;
        if (other >= MT_WORDS) other -= MT_WORDS;
        uint32_t value = (words[cursor] & 0x80000000U) | (words[next] & 0x7fffffffU);
        uint32_t result = words[other] ^ (value >> 1) ^ ((value & 1U) ? 0x9908b0dfU : 0U);
        words[cursor] = result;
        cursor = next;
        return result;
    }
    void discard(uint64_t count) {
        while (cursor != 0 && count) { advance(); --count; }
        while (count >= MT_WORDS) {
            for (int i = 0; i < MT_WORDS; ++i) {
                uint32_t value = (words[i] & 0x80000000U) | (words[(i+1) % MT_WORDS] & 0x7fffffffU);
                words[i] = words[(i+397) % MT_WORDS] ^ (value >> 1) ^ ((value & 1U) ? 0x9908b0dfU : 0U);
            }
            count -= MT_WORDS;
        }
        while (count--) advance();
    }
    void checkpoint(uint32_t* target) const {
        std::copy(words.begin() + cursor, words.end(), target);
        std::copy(words.begin(), words.begin() + cursor, target + MT_WORDS - cursor);
    }
    void applyJump(const Polynomial& polynomial) {
        std::array<uint32_t, MT_WORDS> result{};
        for (int bit = 0; bit < MT_DEGREE; ++bit) {
            if ((polynomial[bit / 64] >> (bit % 64)) & 1U) {
                int first = MT_WORDS - cursor;
                for (int i = 0; i < first; ++i) result[i] ^= words[cursor + i];
                for (int i = first; i < MT_WORDS; ++i) result[i] ^= words[i - first];
            }
            advance();
        }
        words = result;
        cursor = 0;
    }
};

// Derive the characteristic polynomial from MT19937 itself using packed
// Berlekamp-Massey, then square modulo it. This avoids external RNG libraries
// and provides logarithmic skip-ahead for rank/thread stream splitting.
Polynomial mtCharacteristic() {
    constexpr int packedWords = (2 * MT_DEGREE + 64) / 64;
    std::array<uint64_t, packedWords> c{}, b{}, history{};
    c[0] = b[0] = 1;
    int degree = 0, last = -1;
    MTStream stream;
    for (int n = 0; n < 2 * MT_DEGREE; ++n) {
        uint64_t carry = stream.advance() & 1U;
        for (int w = 0; w <= MT_DEGREE / 64; ++w) {
            uint64_t next = history[w] >> 63;
            history[w] = (history[w] << 1) | carry;
            carry = next;
        }
        unsigned discrepancy = 0;
        for (int w = 0; w <= degree / 64; ++w)
            discrepancy ^= static_cast<unsigned>(__builtin_parityll(c[w] & history[w]));
        if (!discrepancy) continue;
        auto previous = c;
        int shift = n - last, whole = shift / 64, part = shift % 64;
        for (int w = 0; w + whole < packedWords; ++w) {
            c[w + whole] ^= b[w] << part;
            if (part && w + whole + 1 < packedWords) c[w + whole + 1] ^= b[w] >> (64 - part);
        }
        if (2 * degree <= n) {
            degree = n + 1 - degree;
            last = n;
            b = previous;
        }
    }
    if (degree != MT_DEGREE) {
        fprintf(stderr, "MT19937 characteristic degree mismatch: %d\n", degree);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    Polynomial result{};
    // The leading x^19937 coefficient is implicit.
    for (int i = 0; i < MT_DEGREE; ++i)
        if ((c[(MT_DEGREE-i) / 64] >> ((MT_DEGREE-i) % 64)) & 1U)
            result[i / 64] |= uint64_t(1) << (i % 64);
    return result;
}

Polynomial squarePolynomial(const Polynomial& input, const Polynomial& modulus) {
    std::array<uint64_t, 2 * POLY_WORDS> product{};
    int top = 0;
    for (int w = 0; w < POLY_WORDS; ++w) {
        uint64_t bits = input[w];
        while (bits) {
            int bit = 64 * w + __builtin_ctzll(bits);
            product[(2*bit) / 64] ^= uint64_t(1) << ((2*bit) % 64);
            top = 2 * bit;
            bits &= bits - 1;
        }
    }
    for (int bit = top; bit >= MT_DEGREE; --bit) {
        if (!((product[bit / 64] >> (bit % 64)) & 1U)) continue;
        product[bit / 64] ^= uint64_t(1) << (bit % 64);
        int shift = bit - MT_DEGREE, whole = shift / 64, part = shift % 64;
        for (int w = 0; w < POLY_WORDS; ++w) {
            product[w + whole] ^= modulus[w] << part;
            if (part) product[w + whole + 1] ^= modulus[w] >> (64 - part);
        }
    }
    Polynomial result{};
    std::copy_n(product.begin(), POLY_WORDS, result.begin());
    return result;
}

std::vector<Polynomial> makeJumpPowers(uint64_t maxOffset) {
    std::vector<Polynomial> powers;
    // Small streams are faster to discard directly. Large streams must not
    // repeat a linear-length prefix on every rank or OpenMP worker.
    if (maxOffset < (uint64_t(1) << 23)) return powers;
    int bits = 64 - __builtin_clzll(maxOffset);
    powers.resize(bits);
    if (worldRank == 0) {
        Polynomial modulus = mtCharacteristic();
        powers[0][0] = 2;
        for (int i = 1; i < bits; ++i) powers[i] = squarePolynomial(powers[i-1], modulus);
    }
    MPI_CHECK(MPI_Bcast(powers.data(), bits * sizeof(Polynomial), MPI_BYTE, 0, MPI_COMM_WORLD));
    return powers;
}

void skipTo(MTStream& stream, uint64_t offset, const std::vector<Polynomial>& powers) {
    if (powers.empty() || offset < (uint64_t(1) << 23)) {
        stream.discard(offset);
        return;
    }
    // Skip large powers with polynomials, and discard the short remainder.
    for (int bit = 23; bit < static_cast<int>(powers.size()); ++bit)
        if ((offset >> bit) & 1U) stream.applyJump(powers[bit]);
    stream.discard(offset & ((uint64_t(1) << 23) - 1));
}

constexpr int PAIRS_PER_BLOCK = 128;
constexpr int DRAWS_PER_PAIR = 4 * NUM_RAYS;

__device__ uint32_t temperMT(uint32_t value) {
    value ^= value >> 11;
    value ^= (value << 7) & 0x9d2c5680U;
    value ^= (value << 15) & 0xefc60000U;
    return value ^ (value >> 18);
}

__device__ Vec3 randomPointInTriangle(const Triangle& tri, val_t u, val_t v) {
    if (u + v > ONE) { u = ONE - u; v = ONE - v; }
    return tri.a + (tri.b - tri.a) * u + (tri.c - tri.a) * v;
}

__global__ void formFactorKernel(const Triangle* triangles, const val_t* areas,
                                 const FlatNode* nodes, const idx_t* indices,
                                 const uint32_t* checkpoints, const size_t* blockIndices,
                                 size_t pairCount, size_t n, size_t rowBegin, size_t rows,
                                 val_t* weights, uint8_t* delays, unsigned long long* nonZero) {
    __shared__ uint32_t mt[MT_WORDS];
    __shared__ val_t randoms[PAIRS_PER_BLOCK * DRAWS_PER_PAIR];
    __shared__ int prefix[PAIRS_PER_BLOCK];
    __shared__ unsigned positives[PAIRS_PER_BLOCK];
    int lane = threadIdx.x;
    size_t block = blockIndices[blockIdx.x];
    size_t pair = block * PAIRS_PER_BLOCK + lane;
    size_t i = rowBegin + pair / n, j = pair % n;
    bool valid = pair < pairCount;
    Triangle triI, triJ;
    bool eligible = false;
    if (valid) {
        triI = triangles[i]; triJ = triangles[j];
        eligible = i != j && triI.normal().dot(triJ.normal()) <= 0.99f;
    }
    prefix[lane] = eligible ? 1 : 0;
    for (int k = lane; k < MT_WORDS; k += PAIRS_PER_BLOCK)
        mt[k] = checkpoints[static_cast<size_t>(blockIdx.x) * MT_WORDS + k];
    __syncthreads();
    for (int offset = 1; offset < PAIRS_PER_BLOCK; offset *= 2) {
        int add = lane >= offset ? prefix[lane-offset] : 0;
        __syncthreads();
        prefix[lane] += add;
        __syncthreads();
    }
    int draws = prefix[PAIRS_PER_BLOCK-1] * DRAWS_PER_PAIR;
    for (int start = 0; start < draws; start += MT_WORDS) {
        // The in-place MT twist has three dependency bands, each parallel.
        for (int band = 0; band < 3; ++band) {
            int begin = band * 227, end = min(begin + 227, MT_WORDS);
            // Read the whole band before writing; neighboring words are inputs.
            uint32_t outputs[2];
            for (int k = begin + lane, slot = 0; k < end; k += PAIRS_PER_BLOCK, ++slot) {
                uint32_t value = (mt[k] & 0x80000000U) | (mt[(k+1) % MT_WORDS] & 0x7fffffffU);
                outputs[slot] = mt[(k+397) % MT_WORDS] ^ (value >> 1) ^ ((value & 1U) ? 0x9908b0dfU : 0U);
            }
            __syncthreads();
            for (int k = begin + lane, slot = 0; k < end; k += PAIRS_PER_BLOCK, ++slot) mt[k] = outputs[slot];
            __syncthreads();
        }
        for (int k = lane; k < MT_WORDS && start + k < draws; k += PAIRS_PER_BLOCK) {
            val_t value = static_cast<val_t>(temperMT(mt[k])) * 0x1p-32f;
            randoms[start+k] = fminf(value, 0x1.fffffep-1f);
        }
        __syncthreads();
    }
    val_t kij = ZERO;
    if (eligible) {
        const val_t* samples = randoms + (prefix[lane]-1) * DRAWS_PER_PAIR;
        for (int ray = 0; ray < NUM_RAYS; ++ray) {
            Vec3 pI = randomPointInTriangle(triI, samples[4*ray], samples[4*ray+1]);
            Vec3 pJ = randomPointInTriangle(triJ, samples[4*ray+2], samples[4*ray+3]);
            if (isRayBlocked(pI, pJ, nodes, indices, triangles, i, j)) continue;
            Vec3 v = pJ - pI;
            val_t distSqr = v.squaredNorm();
            if (distSqr < EPSILON) continue;
            val_t norm = v.norm();
            val_t cosI = norm <= EPSILON ? ZERO : fmaxf(ZERO, v.dot(triI.normal()) / norm);
            val_t cosJ = norm <= EPSILON ? ZERO : fmaxf(ZERO, (-v).dot(triJ.normal()) / norm);
            if (cosI <= ZERO || cosJ <= ZERO) continue;
            kij += (cosI * cosJ) / (PI * distSqr);
        }
        kij *= INV_NUM_RAYS;
    }
    if (valid) {
        // Transpose the local rows for coalesced propagation loads.
        size_t index = j * rows + (i-rowBegin);
        weights[index] = fminf(kij * areas[j], ONE);
        delays[index] = i == j ? 0 : static_cast<uint8_t>(ceilf((triI.center()-triJ.center()).norm() * INV_WAVE_SPEED));
    }
    positives[lane] = kij > EPSILON ? 1U : 0U;
    __syncthreads();
    for (int stride = PAIRS_PER_BLOCK / 2; stride; stride /= 2) {
        if (lane < stride) positives[lane] += positives[lane+stride];
        __syncthreads();
    }
    if (lane == 0 && positives[0]) atomicAdd(nonZero, static_cast<unsigned long long>(positives[0]));
}

struct SimulationState {
    size_t numTriangles = 0, numTimesteps = 0, sourceIndex = 0;
    size_t rowBegin = 0, localRows = 0;
    int minDelay = 1, historyLength = 1;
    val_t reflectivity = 0.8f;
    uint64_t nonZeroKij = 0;
    std::vector<Triangle> triangles;
    std::vector<val_t> areas, distances;
    std::vector<int> rowCounts, rowOffsets;
    DeviceBuffer<Triangle> deviceTriangles;
    DeviceBuffer<val_t> deviceAreas, weights, history, radiosity, batch, source, deviceDistances;
    DeviceBuffer<uint8_t> delays;
    DeviceBuffer<FlatNode> nodes;
    DeviceBuffer<idx_t> indices;
    DeviceBuffer<unsigned long long> nonZero;
};

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.reflectivity = reflectivity;
    state.rowCounts.resize(worldSize);
    state.rowOffsets.resize(worldSize);
    for (int rank = 0; rank < worldSize; ++rank) {
        size_t begin = state.numTriangles * rank / worldSize;
        size_t end = state.numTriangles * (rank+1) / worldSize;
        state.rowOffsets[rank] = static_cast<int>(begin);
        state.rowCounts[rank] = static_cast<int>(end-begin);
    }
    state.rowBegin = state.rowOffsets[worldRank];
    state.localRows = state.rowCounts[worldRank];
    logMessage("Generated icosphere mesh with %zu triangles\nBuilding octree...\n", state.numTriangles);
    Octree octree;
    octree.build(state.triangles);
    std::vector<FlatNode> nodes;
    std::vector<idx_t> indices;
    flattenOctree(octree, nodes, indices);
    state.nodes.upload(nodes);
    state.indices.upload(indices);
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) state.areas[i] = state.triangles[i].area();
    state.deviceTriangles.upload(state.triangles);
    state.deviceAreas.upload(state.areas);
    size_t pairs = state.localRows * state.numTriangles;
    state.weights.allocate(pairs);
    state.delays.allocate(pairs);
    state.radiosity.allocate(state.localRows * timesteps);
    state.deviceDistances.allocate(state.localRows);
    state.source.allocate(timesteps);
    state.nonZero.allocate(1);
    CUDA_CHECK(cudaMemset(state.nonZero.get(), 0, sizeof(unsigned long long)));
    if (worldRank == 0) state.distances.resize(state.numTriangles);
}

void computeFormFactors(SimulationState& state) {
    logMessage("Computing time delays and form factors (Kij) on CUDA...\n");
    size_t n = state.numTriangles, pairs = state.localRows * n;
    size_t blocks = (pairs + PAIRS_PER_BLOCK - 1) / PAIRS_PER_BLOCK;
    std::vector<uint64_t> offsets(blocks+1, 0);
    int minDelay = INT_MAX, maxDelay = 0;
    #pragma omp parallel for schedule(static) reduction(min:minDelay) reduction(max:maxDelay)
    for (size_t block = 0; block < blocks; ++block) {
        uint64_t count = 0;
        for (size_t pair = block * PAIRS_PER_BLOCK; pair < std::min(pairs, (block+1)*PAIRS_PER_BLOCK); ++pair) {
            size_t i = state.rowBegin + pair / n, j = pair % n;
            if (i == j) continue;
            const Triangle& a = state.triangles[i];
            const Triangle& b = state.triangles[j];
            if (a.normal().dot(b.normal()) <= 0.99f) count += DRAWS_PER_PAIR;
            int delay = static_cast<int>(std::ceil((a.center()-b.center()).norm() * INV_WAVE_SPEED));
            minDelay = std::min(minDelay, delay);
            maxDelay = std::max(maxDelay, delay);
        }
        offsets[block+1] = count;
    }
    for (size_t block = 0; block < blocks; ++block) offsets[block+1] += offsets[block];
    uint64_t rankOffset = 0, localDraws = offsets.back(), totalDraws = 0;
    MPI_CHECK(MPI_Exscan(&localDraws, &rankOffset, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD));
    if (worldRank == 0) rankOffset = 0;
    MPI_CHECK(MPI_Allreduce(&localDraws, &totalDraws, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &minDelay, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &maxDelay, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    // Distinct mesh centers always have strictly positive delays.
    if (minDelay <= 0 || maxDelay > UINT8_MAX) {
        fprintf(stderr, "Unsupported mesh propagation delay\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    state.minDelay = minDelay;
    state.historyLength = maxDelay + minDelay;
    state.history.allocate(n * state.historyLength);
    CUDA_CHECK(cudaMemset(state.history.get(), 0, n * state.historyLength * sizeof(val_t)));
    state.batch.allocate(n * minDelay);
    auto jumpPowers = makeJumpPowers(totalDraws);

    // Bound staging memory independently of N. The states are only 624 words
    // per 128 pairs, rather than transferring 64 random floats per pair.
    constexpr size_t batchBlocks = 8192;
    size_t capacity = std::min(blocks, batchBlocks);
    PinnedBuffer<uint32_t> checkpoints;
    checkpoints.allocate(capacity * MT_WORDS);
    DeviceBuffer<uint32_t> deviceCheckpoints;
    deviceCheckpoints.allocate(capacity * MT_WORDS);
    PinnedBuffer<size_t> blockIndices;
    blockIndices.allocate(capacity);
    DeviceBuffer<size_t> deviceBlockIndices;
    deviceBlockIndices.allocate(capacity);
    int workers = static_cast<int>(std::min<uint64_t>(omp_get_max_threads(), std::max<uint64_t>(1, (localDraws+65535)/65536)));
    std::vector<MTStream> streams(workers);
    std::vector<size_t> positions(workers), ends(workers), chunkOffsets(workers+1);
    #pragma omp parallel for schedule(static) num_threads(workers)
    for (int worker = 0; worker < workers; ++worker) {
        positions[worker] = blocks * worker / workers;
        ends[worker] = blocks * (worker+1) / workers;
        skipTo(streams[worker], rankOffset + offsets[positions[worker]], jumpPowers);
    }
    // Keep every host worker active in each staging batch, even for very large
    // matrices. Each worker's stream advances through its own contiguous range.
    size_t blocksPerWorker = std::max<size_t>(1, capacity / workers);
    for (;;) {
        chunkOffsets[0] = 0;
        for (int worker = 0; worker < workers; ++worker)
            chunkOffsets[worker+1] = chunkOffsets[worker] + std::min(blocksPerWorker, ends[worker]-positions[worker]);
        size_t count = chunkOffsets.back();
        if (!count) break;
        #pragma omp parallel for schedule(static) num_threads(workers)
        for (int worker = 0; worker < workers; ++worker) {
            size_t chunkSize = chunkOffsets[worker+1] - chunkOffsets[worker];
            for (size_t k = 0; k < chunkSize; ++k) {
                size_t block = positions[worker]++, target = chunkOffsets[worker]+k;
                blockIndices.get()[target] = block;
                streams[worker].checkpoint(checkpoints.get() + target * MT_WORDS);
                streams[worker].discard(offsets[block+1]-offsets[block]);
            }
        }
        CUDA_CHECK(cudaMemcpy(deviceCheckpoints.get(), checkpoints.get(), count * MT_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceBlockIndices.get(), blockIndices.get(), count * sizeof(size_t), cudaMemcpyHostToDevice));
        formFactorKernel<<<static_cast<unsigned>(count), PAIRS_PER_BLOCK>>>(
            state.deviceTriangles.get(), state.deviceAreas.get(), state.nodes.get(), state.indices.get(),
            deviceCheckpoints.get(), deviceBlockIndices.get(), pairs, n, state.rowBegin, state.localRows,
            state.weights.get(), state.delays.get(), state.nonZero.get());
        CUDA_CHECK(cudaGetLastError());
        // The next host batch can be prepared while this kernel executes.
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long localNonZero = 0;
    CUDA_CHECK(cudaMemcpy(&localNonZero, state.nonZero.get(), sizeof(localNonZero), cudaMemcpyDeviceToHost));
    uint64_t count = localNonZero;
    MPI_CHECK(MPI_Allreduce(&count, &state.nonZeroKij, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD));
}

// Each thread retains the original ascending-j accumulation order. Transposed
// matrix storage coalesces loads across receiver threads without changing sums.
// A batch is at most the minimum delay, so it has no internal dependencies.
__global__ void propagationKernel(const val_t* weights, const uint8_t* delays,
                                  const val_t* history, val_t* radiosity, val_t* batch,
                                  size_t n, size_t rows, size_t rowBegin, size_t timesteps,
                                  size_t firstTime, int count, int historyLength,
                                  size_t sourceIndex, val_t reflectivity) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int step = blockIdx.y;
    if (i >= rows || step >= count) return;
    size_t t = firstTime + step;
    val_t sum = ZERO;
    for (size_t j = 0; j < n; ++j) {
        int delay = delays[j * rows + i];
        if (j == rowBegin+i || t < static_cast<size_t>(delay)) continue;
        val_t weight = weights[j * rows + i];
        if (weight <= ZERO) continue;
        size_t sourceTime = t - delay;
        val_t rad = history[j * historyLength + sourceTime % historyLength];
        if (rad <= ZERO) continue;
        sum += weight * rad;
    }
    val_t emission = rowBegin+i == sourceIndex && t < timesteps / 2 ? ONE : ZERO;
    val_t result = reflectivity * sum + emission;
    radiosity[i * timesteps + t] = result;
    batch[(rowBegin+i) * count + step] = result;
}

__global__ void installHistoryKernel(const val_t* batch, val_t* history, size_t n,
                                     size_t firstTime, int count, int historyLength) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n * count) return;
    size_t i = index / count, step = index % count;
    history[i * historyLength + (firstTime+step) % historyLength] = batch[index];
}

void runSimulation(SimulationState& state) {
    logMessage("Running wave propagation simulation...\n");
    PinnedBuffer<val_t> incoming;
    if (worldSize > 1 && !cudaAwareMPI) incoming.allocate(state.numTriangles * state.minDelay);
    std::vector<int> counts(worldSize), offsets(worldSize);
    for (size_t t = 0; t < state.numTimesteps; t += state.minDelay) {
        int count = static_cast<int>(std::min<size_t>(state.minDelay, state.numTimesteps-t));
        if (state.localRows) {
            dim3 grid(static_cast<unsigned>((state.localRows+127)/128), count);
            propagationKernel<<<grid, 128>>>(state.weights.get(), state.delays.get(), state.history.get(),
                state.radiosity.get(), state.batch.get(), state.numTriangles, state.localRows, state.rowBegin,
                state.numTimesteps, t, count, state.historyLength, state.sourceIndex, state.reflectivity);
            CUDA_CHECK(cudaGetLastError());
        }
        if (worldSize > 1) {
            for (int rank = 0; rank < worldSize; ++rank) {
                counts[rank] = state.rowCounts[rank] * count;
                offsets[rank] = state.rowOffsets[rank] * count;
            }
            if (cudaAwareMPI) {
                CUDA_CHECK(cudaStreamSynchronize(nullptr));
                MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, state.batch.get(),
                    counts.data(), offsets.data(), MPI_FLOAT, MPI_COMM_WORLD));
            } else {
                size_t bytes = state.localRows * count * sizeof(val_t);
                size_t offset = state.rowBegin * count;
                if (bytes) CUDA_CHECK(cudaMemcpy(incoming.get()+offset, state.batch.get()+offset, bytes, cudaMemcpyDeviceToHost));
                MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, incoming.get(),
                    counts.data(), offsets.data(), MPI_FLOAT, MPI_COMM_WORLD));
                CUDA_CHECK(cudaMemcpy(state.batch.get(), incoming.get(), state.numTriangles * count * sizeof(val_t), cudaMemcpyHostToDevice));
            }
        }
        size_t entries = state.numTriangles * count;
        installHistoryKernel<<<static_cast<unsigned>((entries+255)/256), 256>>>(
            state.batch.get(), state.history.get(), state.numTriangles, t, count, state.historyLength);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void distanceKernel(const val_t* radiosity, const val_t* source, val_t* distances,
                               size_t timesteps) {
    __shared__ val_t maxima[128];
    __shared__ unsigned bestLags[128];
    size_t i = blockIdx.x;
    int lane = threadIdx.x;
    val_t maxCorr = ZERO;
    unsigned best = 0;
    for (size_t lag = lane; lag < timesteps; lag += blockDim.x) {
        val_t sum = ZERO;
        for (size_t tt = lag; tt < timesteps; ++tt)
            sum += source[tt-lag] * radiosity[i * timesteps + tt];
        if (sum > maxCorr) { maxCorr = sum; best = static_cast<unsigned>(lag); }
    }
    maxima[lane] = maxCorr;
    bestLags[lane] = best;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (lane < stride) {
            if (maxima[lane+stride] > maxima[lane] ||
                (maxima[lane+stride] == maxima[lane] && bestLags[lane+stride] < bestLags[lane])) {
                maxima[lane] = maxima[lane+stride];
                bestLags[lane] = bestLags[lane+stride];
            }
        }
        __syncthreads();
    }
    if (lane == 0) distances[i] = WAVE_SPEED * static_cast<val_t>(bestLags[0]);
}

void computeDistances(SimulationState& state) {
    logMessage("Computing distances via cross-correlation...\n");
    int owner = 0;
    while (owner+1 < worldSize && state.sourceIndex >= static_cast<size_t>(state.rowOffsets[owner]+state.rowCounts[owner])) ++owner;
    if (state.numTimesteps == 0) {
        // The original empty correlation returns a zero distance for every row.
    } else if (worldSize == 1) {
        CUDA_CHECK(cudaMemcpy(state.source.get(), state.radiosity.get() + state.sourceIndex * state.numTimesteps,
            state.numTimesteps * sizeof(val_t), cudaMemcpyDeviceToDevice));
    } else if (cudaAwareMPI) {
        if (worldRank == owner) CUDA_CHECK(cudaMemcpy(state.source.get(),
            state.radiosity.get() + (state.sourceIndex-state.rowBegin) * state.numTimesteps,
            state.numTimesteps * sizeof(val_t), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        MPI_CHECK(MPI_Bcast(state.source.get(), static_cast<int>(state.numTimesteps), MPI_FLOAT, owner, MPI_COMM_WORLD));
    } else {
        PinnedBuffer<val_t> source;
        source.allocate(state.numTimesteps);
        if (worldRank == owner) CUDA_CHECK(cudaMemcpy(source.get(),
            state.radiosity.get() + (state.sourceIndex-state.rowBegin) * state.numTimesteps,
            state.numTimesteps * sizeof(val_t), cudaMemcpyDeviceToHost));
        MPI_CHECK(MPI_Bcast(source.get(), static_cast<int>(state.numTimesteps), MPI_FLOAT, owner, MPI_COMM_WORLD));
        CUDA_CHECK(cudaMemcpy(state.source.get(), source.get(), state.numTimesteps * sizeof(val_t), cudaMemcpyHostToDevice));
    }
    if (state.localRows) {
        distanceKernel<<<static_cast<unsigned>(state.localRows), 128>>>(
            state.radiosity.get(), state.source.get(), state.deviceDistances.get(), state.numTimesteps);
        CUDA_CHECK(cudaGetLastError());
    }
    std::vector<val_t> distances(state.localRows);
    if (state.localRows) CUDA_CHECK(cudaMemcpy(distances.data(), state.deviceDistances.get(),
        state.localRows * sizeof(val_t), cudaMemcpyDeviceToHost));
    MPI_CHECK(MPI_Gatherv(distances.data(), static_cast<int>(state.localRows), MPI_FLOAT,
        state.distances.data(), state.rowCounts.data(), state.rowOffsets.data(), MPI_FLOAT, 0, MPI_COMM_WORLD));
}

__global__ void energyCountKernel(const val_t* radiosity, size_t rows, size_t timesteps, unsigned* received) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    for (size_t t = 0; t < timesteps; ++t) {
        if (radiosity[i * timesteps+t] > EPSILON) { atomicAdd(received, 1U); break; }
    }
}

bool validateResults(SimulationState& state) {
    DeviceBuffer<unsigned> received;
    received.allocate(1);
    CUDA_CHECK(cudaMemset(received.get(), 0, sizeof(unsigned)));
    if (state.localRows) {
        energyCountKernel<<<static_cast<unsigned>((state.localRows+127)/128), 128>>>(
            state.radiosity.get(), state.localRows, state.numTimesteps, received.get());
        CUDA_CHECK(cudaGetLastError());
    }
    unsigned local = 0, total = 0;
    CUDA_CHECK(cudaMemcpy(&local, received.get(), sizeof(local), cudaMemcpyDeviceToHost));
    MPI_CHECK(MPI_Allreduce(&local, &total, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD));
    bool success = true;
    if (worldRank == 0) {
        printf("\nValidation:\n");
        val_t minDist = std::numeric_limits<val_t>::max(), maxDist = std::numeric_limits<val_t>::lowest();
        val_t sumDist = ZERO;
        int nonZeroCount = 0;
        for (size_t i = 0; i < state.numTriangles; ++i) {
            val_t d = state.distances[i];
            if (d < ZERO || !std::isfinite(d)) {
                printf("  ERROR: Invalid distance at triangle %zu: %f\n", i, d);
                success = false;
            }
            minDist = std::min(minDist, d); maxDist = std::max(maxDist, d); sumDist += d;
            if (d > EPSILON) ++nonZeroCount;
        }
        printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
        printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
        printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);
        if (state.distances[state.sourceIndex] > WAVE_SPEED * 2)
            printf("  WARNING: Source triangle distance is non-zero: %.4f\n", state.distances[state.sourceIndex]);
        printf("  Triangles receiving energy: %u/%zu\n", total, state.numTriangles);
        uint64_t pairs = state.numTriangles * state.numTriangles;
        printf("  Non-zero form factors: %llu/%llu (%.2f%%)\n",
            static_cast<unsigned long long>(state.nonZeroKij), static_cast<unsigned long long>(pairs),
            100.0 * state.nonZeroKij / pairs);
        if (!total || !state.nonZeroKij) {
            printf("  ERROR: Simulation produced no energy or form factors\n");
            success = false;
        }
        if (success) printf("  Validation: PASSED\n");
    }
    int result = success ? 1 : 0;
    MPI_CHECK(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return result != 0;
}

uint64_t computeHash(const SimulationState& state) {
    uint64_t hash = 0;
    for (size_t i = 0; i < state.distances.size(); ++i) {
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
    if (worldRank != 0) return;
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
            logMessage("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (timesteps < 0 || !std::isfinite(reflectivity)) {
        logMessage("Timesteps must be non-negative and reflectivity must be finite.\n");
        return 1;
    }
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: roomsim requires a CUDA device\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm localComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &localComm));
    int localRank = 0, localSize = 1;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));
    MPI_CHECK(MPI_Comm_size(localComm, &localSize));
    CUDA_CHECK(cudaSetDevice(localRank % devices));
#ifdef ROOMSIM_MPI_CUDA_QUERY
    int supported = MPIX_Query_cuda_support() ? 1 : 0;
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &supported, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    cudaAwareMPI = supported != 0;
#endif
    MPI_CHECK(MPI_Comm_free(&localComm));
    if (!getenv("OMP_NUM_THREADS")) omp_set_num_threads(std::max(1, std::min(32, omp_get_num_procs()/localSize)));
    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    logMessage("Room Response Simulation Benchmark\n");
    logMessage("===================================\n");
    logMessage("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    logMessage("Timesteps: %d\n", timesteps);
    logMessage("Source triangle: %d\n", sourceIdx);
    logMessage("Reflectivity: %.2f\n", reflectivity);
    logMessage("Validation: %s\n", validate ? "enabled" : "disabled");
    logMessage("Hybrid execution: %d MPI ranks, up to %d OpenMP threads/rank, CUDA\n", worldSize, omp_get_max_threads());
    logMessage("MPI GPU transfers: %s\n", cudaAwareMPI ? "direct" : "pinned host staging");
    logMessage("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    logMessage("\n");

    // Precomputation
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double startPre = MPI_Wtime();
    computeFormFactors(state);

    double preSeconds = MPI_Wtime() - startPre;
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &preSeconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    long preDuration = static_cast<long>(preSeconds * 1000);

    logMessage("Precomputation time: %ld ms\n", preDuration);
    logMessage("\n");

    // Simulation
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double startSim = MPI_Wtime();

    runSimulation(state);

    double simSeconds = MPI_Wtime() - startSim;
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &simSeconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    long simDuration = static_cast<long>(simSeconds * 1000);

    logMessage("Simulation time: %ld ms\n", simDuration);
    logMessage("\n");

    // Distance computation
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double startDist = MPI_Wtime();

    computeDistances(state);

    double distSeconds = MPI_Wtime() - startDist;
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &distSeconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    long distDuration = static_cast<long>(distSeconds * 1000);

    logMessage("Distance computation time: %ld ms\n", distDuration);
    logMessage("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    logMessage("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    logMessage("\nPerformance:\n");
    logMessage("  Triangles: %zu\n", n);
    logMessage("  Timesteps: %zu\n", t);
    logMessage("  Form factor computations: %.2e\n", kijOps);
    logMessage("  Simulation operations: %.2e\n", simOps);
    logMessage("  Distance computations: %.2e\n", distOps);
    logMessage("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

    // Memory usage
    size_t memKij = state.localRows * n * sizeof(val_t);
    size_t memTau = state.localRows * n * sizeof(uint8_t);
    size_t memRad = (t * state.localRows + n * (state.historyLength + state.minDelay) + t) * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    logMessage("  Matrix/history memory on rank 0: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    logMessage("  Result hash: %016lX\n", hash);
    logMessage("\n");
    
    // Print results for external validation
    if (printResults && worldRank == 0) {
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

// MPI calls remain on the main thread; OpenMP workers only prepare host data.
int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    int error = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (error != MPI_SUCCESS) return 1;
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int result = 1;
    try {
        result = runBenchmark(argc, argv);
    } catch (const std::exception& exception) {
        fprintf(stderr, "Rank %d: %s\n", worldRank, exception.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_CHECK(MPI_Finalize());
    return result;
}
