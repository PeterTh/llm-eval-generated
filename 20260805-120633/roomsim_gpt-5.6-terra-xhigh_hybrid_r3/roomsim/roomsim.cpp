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
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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

// The benchmark is intentionally a three-level parallel program: MPI owns
// disjoint receiver rows, OpenMP accelerates the irregular visibility work on
// each host, and CUDA executes the regular dense kernels on the local GPU.
// MPI is initialized before any of these helpers are used.
void checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    fprintf(stderr, "CUDA failure during %s: %s\n", operation, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
    std::abort();
}

struct MpiContext {
    int rank = 0;
    int size = 1;
    int localRank = 0;
    int localSize = 1;
};

MpiContext initializeMpi(int& argc, char**& argv) {
    int provided = MPI_THREAD_SINGLE;
    const int initStatus = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (initStatus != MPI_SUCCESS || provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI does not provide the required FUNNELED thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }

    MpiContext context;
    MPI_Comm_rank(MPI_COMM_WORLD, &context.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &context.size);

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, context.rank,
                        MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &context.localRank);
    MPI_Comm_size(localComm, &context.localSize);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "querying CUDA devices");
    if (deviceCount == 0) {
        if (context.rank == 0) fprintf(stderr, "No CUDA device is available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
    // Use local, rather than global, rank so the mapping repeats correctly on
    // every node in a multi-node job.
    checkCuda(cudaSetDevice(context.localRank % deviceCount), "selecting local CUDA device");
    return context;
}

struct RowPartition {
    size_t first = 0;
    size_t count = 0;
    std::vector<int> counts;
    std::vector<int> displacements;
};

RowPartition makeRowPartition(size_t rows, const MpiContext& mpi) {
    if (rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (mpi.rank == 0) fprintf(stderr, "MPI row count exceeds MPI int count range\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }

    RowPartition partition;
    partition.counts.resize(mpi.size);
    partition.displacements.resize(mpi.size);
    const size_t base = rows / static_cast<size_t>(mpi.size);
    const size_t remainder = rows % static_cast<size_t>(mpi.size);
    size_t offset = 0;
    for (int rank = 0; rank < mpi.size; ++rank) {
        const size_t rankCount = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
        partition.counts[rank] = static_cast<int>(rankCount);
        partition.displacements[rank] = static_cast<int>(offset);
        if (rank == mpi.rank) {
            partition.first = offset;
            partition.count = rankCount;
        }
        offset += rankCount;
    }
    return partition;
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    ~DeviceBuffer() { release(); }

    void allocate(size_t count) {
        release();
        count_ = count;
        if (count_ != 0) checkCuda(cudaMalloc(&ptr_, count_ * sizeof(T)), "allocating device buffer");
    }
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return count_; }

private:
    void release() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* ptr_ = nullptr;
    size_t count_ = 0;
};

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    constexpr Vec3() : x(0), y(0), z(0) {}
    constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }

    val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    val_t squaredNorm() const { return x * x + y * y + z * z; }
    val_t norm() const { return std::sqrt(squaredNorm()); }
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
// Random Number Generation
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }

    std::string serialize() const {
        std::ostringstream stream;
        stream << rng;
        return stream.str();
    }

    bool restore(const std::string& state) {
        std::istringstream stream(state);
        stream >> rng;
        return static_cast<bool>(stream);
    }
};

// Generate a random point inside a triangle using barycentric coordinates
Vec3 randomPointInTriangle(const Triangle& t, RandomGenerator& rng) {
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

val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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

// ============================================================================
// Visibility Testing (Octree-accelerated)
// ============================================================================

// Check if a ray between two triangles is blocked by any other triangle
// Uses octree for O(log N) instead of O(N) search
bool isRayBlocked(const Vec3& from, const Vec3& to,
                  const Octree& octree,
                  size_t srcTriIdx, size_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    return octree.applyToTris(from, to, [&](size_t idx, const Triangle& tri) {
        if (idx == srcTriIdx || idx == dstTriIdx) return false;

        val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
        if (dist > EPSILON && dist < rayLen - EPSILON) {
            return true;  // Ray is blocked
        }
        return false;
    });
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return std::max(ZERO, v.dot(normal) / vNorm);
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle(triI, rng);
        Vec3 pJ = randomPointInTriangle(triJ, rng);

        if (isRayBlocked(pI, pJ, octree, idxI, idxJ)) continue;

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

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;
    size_t localFirst;
    size_t localCount;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    // Each rank stores only its receiver rows of the dense matrices.  The
    // previous radiosity timeline is replicated because every receiver row
    // depends on all emitters at delayed timesteps.
    std::vector<val_t> kij;         // Local receiver rows, localCount x N
    std::vector<val_t> radB;        // Replicated T x N reflected radiosity
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    size_t idx2d(size_t localI, size_t j) const { return localI * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity,
                          const RowPartition& partition, const MpiContext& mpi) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    state.localFirst = partition.first;
    state.localCount = partition.count;

    if (mpi.rank == 0) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if (mpi.rank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.kij.resize(state.localCount * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

struct RayPairSamples {
    val_t values[NUM_RAYS][4];  // pI.u, pI.v, pJ.u, pJ.v for each ray
};

Vec3 pointFromBarycentrics(const Triangle& triangle, val_t u, val_t v) {
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return triangle.a + (triangle.b - triangle.a) * u + (triangle.c - triangle.a) * v;
}

val_t computeKijFromSamples(size_t idxI, size_t idxJ,
                            const std::vector<Triangle>& triangles,
                            const Octree& octree,
                            const RayPairSamples& samples) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];
    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        const val_t* values = samples.values[r];
        const Vec3 pI = pointFromBarycentrics(triI, values[0], values[1]);
        const Vec3 pJ = pointFromBarycentrics(triJ, values[2], values[3]);

        if (isRayBlocked(pI, pJ, octree, idxI, idxJ)) continue;

        const Vec3 v = pJ - pI;
        const val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        const val_t cosPhiI = cosPhi(v, triI.normal());
        const val_t cosPhiJ = cosPhi(-v, triJ.normal());
        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }
    return kij * INV_NUM_RAYS;
}

// The original program used one MT19937 stream.  Preserve that stream exactly
// while still exposing every pair calculation to OpenMP by handing each MPI
// rank the serialized generator state at the first row it owns.  Per-row
// samples are produced in source order before the independent ray casts run in
// parallel, so results do not depend on rank count or OpenMP scheduling.
std::string generatorStateForLocalRows(const SimulationState& state,
                                       const RowPartition& partition,
                                       const MpiContext& mpi) {
    std::vector<std::string> rankStates;
    if (mpi.rank == 0) {
        rankStates.resize(mpi.size);
        RandomGenerator generator(42);
        for (size_t i = 0; i <= state.numTriangles; ++i) {
            for (int rank = 0; rank < mpi.size; ++rank) {
                if (static_cast<size_t>(partition.displacements[rank]) == i) {
                    rankStates[rank] = generator.serialize();
                }
            }
            if (i == state.numTriangles) break;

            for (size_t j = 0; j < state.numTriangles; ++j) {
                if (i == j || state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) continue;
                for (int sample = 0; sample < NUM_RAYS * 4; ++sample) generator.rand();
            }
        }
    }

    int maxLength = 0;
    if (mpi.rank == 0) {
        for (const std::string& stateString : rankStates) {
            maxLength = std::max(maxLength, static_cast<int>(stateString.size() + 1));
        }
    }
    MPI_Bcast(&maxLength, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<char> encoded(static_cast<size_t>(maxLength) * static_cast<size_t>(mpi.size), '\0');
    if (mpi.rank == 0) {
        for (int rank = 0; rank < mpi.size; ++rank) {
            std::memcpy(encoded.data() + static_cast<size_t>(rank) * maxLength,
                        rankStates[rank].c_str(), rankStates[rank].size());
        }
    }
    std::vector<char> localEncoded(maxLength, '\0');
    MPI_Scatter(encoded.data(), maxLength, MPI_CHAR, localEncoded.data(), maxLength,
                MPI_CHAR, 0, MPI_COMM_WORLD);
    return std::string(localEncoded.data());
}

void computeFormFactors(SimulationState& state, const RowPartition& partition,
                        const MpiContext& mpi) {
    if (mpi.rank == 0) printf("Computing form factors (Kij) with MPI + OpenMP...\n");
    RandomGenerator generator;
    const std::string localState = generatorStateForLocalRows(state, partition, mpi);
    if (!generator.restore(localState)) {
        if (mpi.rank == 0) fprintf(stderr, "Unable to restore deterministic MT19937 state\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }

    std::vector<RayPairSamples> samples(state.numTriangles);
    for (size_t localI = 0; localI < state.localCount; ++localI) {
        const size_t i = state.localFirst + localI;

        // Generate this row in its original serial order.  This small
        // preparation pass is dwarfed by the octree traversal it enables to
        // run concurrently below.
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j || state.triangles[i].normal().dot(state.triangles[j].normal()) > 0.99f) continue;
            for (int r = 0; r < NUM_RAYS; ++r) {
                for (int value = 0; value < 4; ++value) samples[j].values[r][value] = generator.rand();
            }
        }

#pragma omp parallel for schedule(dynamic, 1)
        for (long long j = 0; j < static_cast<long long>(state.numTriangles); ++j) {
            const size_t column = static_cast<size_t>(j);
            if (i == column || state.triangles[i].normal().dot(state.triangles[column].normal()) > 0.99f) continue;
            state.kij[state.idx2d(localI, column)] = computeKijFromSamples(
                i, column, state.triangles, state.octree, samples[column]);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

constexpr int CUDA_THREADS_PER_BLOCK = 256;

__global__ void computeTauKernel(const float3* centers, int* tau,
                                 size_t localRows, size_t numTriangles,
                                 size_t globalFirst) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= localRows * numTriangles) return;
    const size_t localI = index / numTriangles;
    const size_t j = index - localI * numTriangles;
    const size_t i = globalFirst + localI;
    if (i == j) {
        tau[index] = 0;
        return;
    }
    const float3 from = centers[i];
    const float3 to = centers[j];
    const val_t dx = from.x - to.x;
    const val_t dy = from.y - to.y;
    const val_t dz = from.z - to.z;
    tau[index] = static_cast<int>(ceilf(sqrtf(dx * dx + dy * dy + dz * dz) * INV_WAVE_SPEED));
}

__global__ void propagateKernel(const val_t* kij, const int* tau, const val_t* areas,
                                const val_t* rho, const val_t* radiosity,
                                val_t* output, size_t localRows, size_t numTriangles,
                                size_t globalFirst, size_t timestep, size_t emissionEnd,
                                size_t sourceIndex) {
    const size_t localI = blockIdx.x;
    if (localI >= localRows) return;
    const size_t globalI = globalFirst + localI;
    const size_t offset = localI * numTriangles;
    val_t sumB = ZERO;
    // A block owns a receiver row.  The independent emitter contributions are
    // striped across the block and reduced locally before MPI synchronizes the
    // completed receiver rows between ranks.
    for (size_t j = threadIdx.x; j < numTriangles; j += blockDim.x) {
        if (globalI == j) continue;
        const int delay = tau[offset + j];
        if (static_cast<int>(timestep) < delay) continue;
        const val_t formFactor = kij[offset + j];
        if (formFactor <= ZERO) continue;
        const val_t sourceRadiosity = radiosity[
            (timestep - static_cast<size_t>(delay)) * numTriangles + j];
        if (sourceRadiosity <= ZERO) continue;
        sumB += fminf(formFactor * areas[j], ONE) * sourceRadiosity;
    }
    __shared__ val_t partial[CUDA_THREADS_PER_BLOCK];
    partial[threadIdx.x] = sumB;
    __syncthreads();
    for (int stride = CUDA_THREADS_PER_BLOCK / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < static_cast<unsigned int>(stride)) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const val_t emission = (timestep < emissionEnd && globalI == sourceIndex) ? ONE : ZERO;
        output[localI] = rho[localI] * partial[0] + emission;
    }
}

__global__ void correlationKernel(const val_t* radiosity, val_t* distances,
                                  size_t localRows, size_t numTriangles,
                                  size_t numTimesteps, size_t globalFirst,
                                  size_t sourceIndex) {
    const size_t localI = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (localI >= localRows) return;
    const size_t globalI = globalFirst + localI;
    val_t maxCorrelation = ZERO;
    int bestTimestep = 0;
    for (size_t t = 0; t < numTimesteps; ++t) {
        val_t sum = ZERO;
        for (size_t tt = t; tt < numTimesteps; ++tt) {
            sum += radiosity[(tt - t) * numTriangles + sourceIndex] *
                   radiosity[tt * numTriangles + globalI];
        }
        if (sum > maxCorrelation) {
            maxCorrelation = sum;
            bestTimestep = static_cast<int>(t);
        }
    }
    distances[localI] = WAVE_SPEED * static_cast<val_t>(bestTimestep);
}

template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t count) { allocate(count); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    ~PinnedBuffer() { if (ptr_ != nullptr) cudaFreeHost(ptr_); }

    void allocate(size_t count) {
        if (ptr_ != nullptr) checkCuda(cudaFreeHost(ptr_), "releasing pinned host buffer");
        ptr_ = nullptr;
        if (count != 0) checkCuda(cudaMallocHost(&ptr_, count * sizeof(T)), "allocating pinned host buffer");
    }
    T* get() { return ptr_; }

private:
    T* ptr_ = nullptr;
};

class GpuSimulation {
public:
    explicit GpuSimulation(const SimulationState& state)
        : numTriangles_(state.numTriangles), numTimesteps_(state.numTimesteps),
          localRows_(state.localCount), globalFirst_(state.localFirst),
          centers_(state.numTriangles), areas_(state.numTriangles), rho_(state.localCount),
          kij_(state.localCount * state.numTriangles), tau_(state.localCount * state.numTriangles),
          radiosity_(state.numTimesteps * state.numTriangles), rowDevice_(state.localCount),
          distancesDevice_(state.localCount), localRow_(state.localCount), gatheredRow_(state.numTriangles),
          localDistances_(state.localCount) {
        std::vector<float3> centers(state.numTriangles);
        std::vector<val_t> localRho(state.localCount);
        for (size_t i = 0; i < state.numTriangles; ++i) {
            const Vec3 center = state.triangles[i].center();
            centers[i] = make_float3(center.x, center.y, center.z);
        }
        for (size_t i = 0; i < state.localCount; ++i) localRho[i] = state.rho[state.localFirst + i];
        checkCuda(cudaMemcpy(centers_.get(), centers.data(), centers.size() * sizeof(float3), cudaMemcpyHostToDevice),
                  "uploading triangle centers");
        checkCuda(cudaMemcpy(areas_.get(), state.areas.data(), state.areas.size() * sizeof(val_t), cudaMemcpyHostToDevice),
                  "uploading triangle areas");
        if (localRows_ != 0) {
            checkCuda(cudaMemcpy(rho_.get(), localRho.data(), localRho.size() * sizeof(val_t), cudaMemcpyHostToDevice),
                      "uploading local reflectivity");
        }
        checkCuda(cudaMemset(radiosity_.get(), 0, radiosity_.size() * sizeof(val_t)),
                  "initializing device radiosity");
    }

    void launchTimeDelayComputation() {
        const size_t elements = localRows_ * numTriangles_;
        if (elements == 0) return;
        const int blocks = static_cast<int>((elements + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
        computeTauKernel<<<blocks, CUDA_THREADS_PER_BLOCK>>>(centers_.get(), tau_.get(), localRows_, numTriangles_, globalFirst_);
        checkCuda(cudaGetLastError(), "launching CUDA time-delay kernel");
    }

    void uploadFormFactors(const SimulationState& state) {
        checkCuda(cudaDeviceSynchronize(), "completing CUDA time-delay computation");
        if (localRows_ != 0) {
            checkCuda(cudaMemcpy(kij_.get(), state.kij.data(), state.kij.size() * sizeof(val_t), cudaMemcpyHostToDevice),
                      "uploading local form factors");
        }
    }

    void runWavePropagation(SimulationState& state, const RowPartition& partition, const MpiContext& mpi) {
        if (mpi.rank == 0) printf("Running wave propagation simulation on CUDA GPUs...\n");
        for (size_t t = 0; t < numTimesteps_; ++t) {
            if (localRows_ != 0) {
                const int blocks = static_cast<int>(localRows_);
                propagateKernel<<<blocks, CUDA_THREADS_PER_BLOCK>>>(
                    kij_.get(), tau_.get(), areas_.get(), rho_.get(), radiosity_.get(), rowDevice_.get(),
                    localRows_, numTriangles_, globalFirst_, t, numTimesteps_ / 2, state.sourceIndex);
                checkCuda(cudaGetLastError(), "launching CUDA propagation kernel");
                checkCuda(cudaMemcpy(localRow_.get(), rowDevice_.get(), localRows_ * sizeof(val_t), cudaMemcpyDeviceToHost),
                          "downloading local radiosity row");
            }
            MPI_Allgatherv(localRow_.get(), static_cast<int>(localRows_), MPI_FLOAT,
                           gatheredRow_.get(), partition.counts.data(), partition.displacements.data(),
                           MPI_FLOAT, MPI_COMM_WORLD);
            std::memcpy(state.radB.data() + t * numTriangles_, gatheredRow_.get(), numTriangles_ * sizeof(val_t));
            checkCuda(cudaMemcpy(radiosity_.get() + t * numTriangles_, gatheredRow_.get(),
                                 numTriangles_ * sizeof(val_t), cudaMemcpyHostToDevice),
                      "uploading synchronized radiosity row");
            if (mpi.rank == 0 && ((t + 1) % 10 == 0 || t + 1 == numTimesteps_)) {
                printf("  Timestep %zu/%zu\n", t + 1, numTimesteps_);
            }
        }
    }

    void computeDistances(SimulationState& state, const RowPartition& partition, const MpiContext& mpi) {
        if (mpi.rank == 0) printf("Computing distances via CUDA cross-correlation...\n");
        if (localRows_ != 0) {
            const int blocks = static_cast<int>((localRows_ + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK);
            correlationKernel<<<blocks, CUDA_THREADS_PER_BLOCK>>>(radiosity_.get(), distancesDevice_.get(), localRows_,
                numTriangles_, numTimesteps_, globalFirst_, state.sourceIndex);
            checkCuda(cudaGetLastError(), "launching CUDA correlation kernel");
            checkCuda(cudaMemcpy(localDistances_.get(), distancesDevice_.get(), localRows_ * sizeof(val_t), cudaMemcpyDeviceToHost),
                      "downloading local distances");
        }
        MPI_Allgatherv(localDistances_.get(), static_cast<int>(localRows_), MPI_FLOAT,
                       state.distances.data(), partition.counts.data(), partition.displacements.data(),
                       MPI_FLOAT, MPI_COMM_WORLD);
    }

private:
    size_t numTriangles_;
    size_t numTimesteps_;
    size_t localRows_;
    size_t globalFirst_;
    DeviceBuffer<float3> centers_;
    DeviceBuffer<val_t> areas_;
    DeviceBuffer<val_t> rho_;
    DeviceBuffer<val_t> kij_;
    DeviceBuffer<int> tau_;
    DeviceBuffer<val_t> radiosity_;
    DeviceBuffer<val_t> rowDevice_;
    DeviceBuffer<val_t> distancesDevice_;
    PinnedBuffer<val_t> localRow_;
    PinnedBuffer<val_t> gatheredRow_;
    PinnedBuffer<val_t> localDistances_;
};

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, const MpiContext& mpi) {
    unsigned long long localNonZeroKij = 0;
    for (const val_t value : state.kij) {
        if (value > EPSILON) ++localNonZeroKij;
    }
    unsigned long long nonZeroKij = 0;
    MPI_Allreduce(&localNonZeroKij, &nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    int result = 1;
    if (mpi.rank == 0) {
        printf("\nValidation:\n");
        bool allNonNegative = true;
        val_t minDist = std::numeric_limits<val_t>::max();
        val_t maxDist = std::numeric_limits<val_t>::lowest();
        val_t sumDist = ZERO;
        int nonZeroCount = 0;
        for (size_t i = 0; i < state.numTriangles; ++i) {
            const val_t d = state.distances[i];
            if (d < ZERO || !std::isfinite(d)) {
                allNonNegative = false;
                printf("  ERROR: Invalid distance at triangle %zu: %f\n", i, d);
            }
            minDist = std::min(minDist, d);
            maxDist = std::max(maxDist, d);
            sumDist += d;
            if (d > EPSILON) ++nonZeroCount;
        }
        printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
        printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
        printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);
        if (state.distances[state.sourceIndex] > WAVE_SPEED * 2) {
            printf("  WARNING: Source triangle distance is non-zero: %.4f\n", state.distances[state.sourceIndex]);
        }

        int receivedEnergy = 0;
        for (size_t i = 0; i < state.numTriangles; ++i) {
            for (size_t t = 0; t < state.numTimesteps; ++t) {
                if (state.radB[state.idxTN(t, i)] > EPSILON) {
                    ++receivedEnergy;
                    break;
                }
            }
        }
        const size_t totalPairs = state.numTriangles * state.numTriangles;
        printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);
        printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n", nonZeroKij, totalPairs,
               100.0 * static_cast<double>(nonZeroKij) / static_cast<double>(totalPairs));
        result = (allNonNegative && receivedEnergy != 0 && nonZeroKij != 0) ? 1 : 0;
        printf("  Validation: %s\n", result ? "PASSED" : "FAILED");
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return result != 0;
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
    const MpiContext mpi = initializeMpi(argc, argv);
    omp_set_dynamic(0);

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
            if (mpi.rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi.rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    size_t actualTriangles = 20;
    for (int level = 0; level < subdivisions; ++level) actualTriangles *= 4;
    const RowPartition partition = makeRowPartition(actualTriangles, mpi);

    int exitCode = 0;
    {
        if (mpi.rank == 0) {
            printf("Room Response Simulation Benchmark\n");
            printf("===================================\n");
            printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
            printf("Timesteps: %d\n", timesteps);
            printf("Source triangle: %d\n", sourceIdx);
            printf("Reflectivity: %.2f\n", reflectivity);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Hybrid execution: %d MPI ranks, up to %d OpenMP threads/rank\n\n",
                   mpi.size, omp_get_max_threads());
        }

        SimulationState state;
        initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                             static_cast<size_t>(sourceIdx), reflectivity, partition, mpi);
        GpuSimulation gpu(state);

        MPI_Barrier(MPI_COMM_WORLD);
        const auto startPre = std::chrono::high_resolution_clock::now();
        gpu.launchTimeDelayComputation();
        computeFormFactors(state, partition, mpi);
        gpu.uploadFormFactors(state);
        const auto endPre = std::chrono::high_resolution_clock::now();
        long preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

        MPI_Barrier(MPI_COMM_WORLD);
        const auto startSim = std::chrono::high_resolution_clock::now();
        gpu.runWavePropagation(state, partition, mpi);
        const auto endSim = std::chrono::high_resolution_clock::now();
        long simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

        MPI_Barrier(MPI_COMM_WORLD);
        const auto startDist = std::chrono::high_resolution_clock::now();
        gpu.computeDistances(state, partition, mpi);
        const auto endDist = std::chrono::high_resolution_clock::now();
        long distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

        long globalPreDuration = 0;
        long globalSimDuration = 0;
        long globalDistDuration = 0;
        MPI_Reduce(&preDuration, &globalPreDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&simDuration, &globalSimDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&distDuration, &globalDistDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

        if (mpi.rank == 0) {
            const long totalTime = globalPreDuration + globalSimDuration + globalDistDuration;
            const size_t n = state.numTriangles;
            const size_t t = state.numTimesteps;
            printf("\nPrecomputation time: %ld ms\n", globalPreDuration);
            printf("Simulation time: %ld ms\n", globalSimDuration);
            printf("Distance computation time: %ld ms\n", globalDistDuration);
            printf("Total computation time: %ld ms\n", totalTime);
            printf("\nPerformance:\n");
            printf("  Triangles: %zu\n", n);
            printf("  Timesteps: %zu\n", t);
            printf("  Form factor computations: %.2e\n", static_cast<double>(n) * n);
            printf("  Simulation operations: %.2e\n", static_cast<double>(n) * n * t);
            printf("  Distance computations: %.2e\n", static_cast<double>(n) * t * t);
            printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);
            const size_t globalDenseBytes = n * n * (sizeof(val_t) + sizeof(int));
            const size_t replicatedRadBytes = t * n * sizeof(val_t);
            printf("  Per-rank dense matrix storage: %.2f MB (row-partitioned)\n",
                   globalDenseBytes / (static_cast<double>(mpi.size) * 1024.0 * 1024.0));
            printf("  Per-rank radiosity storage: %.2f MB\n", replicatedRadBytes / (1024.0 * 1024.0));
            printf("  Result hash: %016lX\n\n", computeHash(state));
            if (printResults) {
                std::vector<double> distData(state.distances.begin(), state.distances.end());
                print_results(distData, "Distances");
            }
        }

        if (validate && !validateResults(state, mpi)) exitCode = 1;
    }
    MPI_Finalize();
    return exitCode;
}
