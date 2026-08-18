/**
 * Room Response Simulation Benchmark
 * 
 * This is a hybrid MPI, OpenMP, and CUDA implementation of room impulse response
 * simulation using radiosity-based wave propagation.  MPI partitions receiver
 * rows, OpenMP accelerates the irregular octree visibility work, and CUDA runs
 * the dense delay, wave-propagation, and correlation phases.  It models how
 * sound/light waves propagate between surfaces in a room by computing:
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
// Stateless Sampling
// ============================================================================

// Form-factor samples are indexed by (receiver, emitter, ray, coordinate), so
// the same samples are generated regardless of MPI partitioning or OpenMP
// scheduling.  This preserves the original uniform Monte-Carlo estimator while
// making every matrix entry independently computable.
inline val_t sampleUniform(uint64_t counter) {
    counter += 0x9e3779b97f4a7c15ULL;
    counter = (counter ^ (counter >> 30)) * 0xbf58476d1ce4e5b9ULL;
    counter = (counter ^ (counter >> 27)) * 0x94d049bb133111ebULL;
    counter ^= counter >> 31;
    return static_cast<val_t>(counter >> 40) * (1.0f / 16777216.0f);
}

// Generate a uniformly sampled point inside a triangle from two deterministic
// random values.
Vec3 randomPointInTriangle(const Triangle& t, val_t u, val_t v) {
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
                  const Octree& octree) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        const uint64_t sampleBase =
            ((static_cast<uint64_t>(idxI) * triangles.size() + idxJ) * NUM_RAYS + r) * 4ULL;
        Vec3 pI = randomPointInTriangle(triI, sampleUniform(sampleBase),
                                        sampleUniform(sampleBase + 1));
        Vec3 pJ = randomPointInTriangle(triJ, sampleUniform(sampleBase + 2),
                                        sampleUniform(sampleBase + 3));

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
// CUDA and MPI Runtime Support
// ============================================================================

[[noreturn]] void cudaFailure(cudaError_t status, const char* expression,
                              const char* file, int line) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "MPI rank %d: CUDA error at %s:%d while evaluating %s: %s\n",
            rank, file, line, expression, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
    std::abort();
}

#define CUDA_CHECK(expression) do { \
    const cudaError_t cudaStatus = (expression); \
    if (cudaStatus != cudaSuccess) cudaFailure(cudaStatus, #expression, __FILE__, __LINE__); \
} while (false)

template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t count) { resize(count); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    ~PinnedBuffer() {
        if (data_ != nullptr) cudaFreeHost(data_);
    }

    void resize(size_t count) {
        if (data_ != nullptr) {
            CUDA_CHECK(cudaFreeHost(data_));
            data_ = nullptr;
            size_ = 0;
        }
        if (count != 0) {
            void* raw = nullptr;
            CUDA_CHECK(cudaMallocHost(&raw, count * sizeof(T)));
            data_ = static_cast<T*>(raw);
            size_ = count;
            std::fill(data_, data_ + size_, T{});
        }
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    T& operator[](size_t index) { return data_[index]; }
    const T& operator[](size_t index) const { return data_[index]; }
    size_t size() const { return size_; }

private:
    T* data_ = nullptr;
    size_t size_ = 0;
};

struct DeviceTriangle {
    val_t centerX, centerY, centerZ;
};

constexpr int CUDA_THREADS = 256;

__global__ void computeTauKernel(const DeviceTriangle* triangles, int* tau,
                                 int numTriangles, int firstTriangle, int localTriangles) {
    const size_t entry = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(localTriangles) * numTriangles;
    if (entry >= total) return;

    const size_t localI = entry / numTriangles;
    const int j = static_cast<int>(entry - localI * numTriangles);
    const int i = firstTriangle + static_cast<int>(localI);
    if (i == j) {
        tau[entry] = 0;
        return;
    }

    const DeviceTriangle receiver = triangles[static_cast<size_t>(i)];
    const DeviceTriangle emitter = triangles[static_cast<size_t>(j)];
    const val_t dx = receiver.centerX - emitter.centerX;
    const val_t dy = receiver.centerY - emitter.centerY;
    const val_t dz = receiver.centerZ - emitter.centerZ;
    tau[entry] = static_cast<int>(ceilf(sqrtf(dx * dx + dy * dy + dz * dz) * INV_WAVE_SPEED));
}

__global__ void propagateKernel(const val_t* kij, const int* tau, const val_t* areas,
                                const val_t* radB, val_t* localCurrent,
                                int numTriangles, int firstTriangle, int localTriangles,
                                int timestep, int sourceIndex, int emissionEnd,
                                val_t reflectivity) {
    const int localI = blockIdx.x;
    if (localI >= localTriangles) return;

    const int i = firstTriangle + localI;
    val_t partial = ZERO;
    const val_t* kijRow = kij + static_cast<size_t>(localI) * numTriangles;
    const int* tauRow = tau + static_cast<size_t>(localI) * numTriangles;
    for (int j = threadIdx.x; j < numTriangles; j += blockDim.x) {
        if (i == j) continue;
        const int tauij = tauRow[j];
        if (timestep < tauij) continue;

        const val_t formFactor = kijRow[j];
        if (formFactor <= ZERO) continue;

        const val_t sourceRadiosity =
            radB[static_cast<size_t>(timestep - tauij) * numTriangles + j];
        if (sourceRadiosity <= ZERO) continue;
        partial += fminf(formFactor * areas[j], ONE) * sourceRadiosity;
    }

    extern __shared__ val_t partialSums[];
    partialSums[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partialSums[threadIdx.x] += partialSums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const val_t emission = (i == sourceIndex && timestep < emissionEnd) ? ONE : ZERO;
        localCurrent[localI] = reflectivity * partialSums[0] + emission;
    }
}

__global__ void distanceKernel(const val_t* radB, val_t* distances,
                               int numTriangles, int numTimesteps,
                               int firstTriangle, int localTriangles, int sourceIndex) {
    const int localI = blockIdx.x;
    if (localI >= localTriangles) return;

    val_t bestCorrelation = ZERO;
    int bestTime = 0;
    const int i = firstTriangle + localI;
    for (int lag = threadIdx.x; lag < numTimesteps; lag += blockDim.x) {
        val_t correlation = ZERO;
        for (int time = lag; time < numTimesteps; ++time) {
            correlation += radB[static_cast<size_t>(time) * numTriangles + i] *
                           radB[static_cast<size_t>(time - lag) * numTriangles + sourceIndex];
        }
        if (correlation > bestCorrelation) {
            bestCorrelation = correlation;
            bestTime = lag;
        }
    }

    extern __shared__ unsigned char sharedMemory[];
    val_t* correlations = reinterpret_cast<val_t*>(sharedMemory);
    int* times = reinterpret_cast<int*>(correlations + blockDim.x);
    correlations[threadIdx.x] = bestCorrelation;
    times[threadIdx.x] = bestTime;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const val_t otherCorrelation = correlations[threadIdx.x + stride];
            const int otherTime = times[threadIdx.x + stride];
            if (otherCorrelation > correlations[threadIdx.x] ||
                (otherCorrelation == correlations[threadIdx.x] && otherTime < times[threadIdx.x])) {
                correlations[threadIdx.x] = otherCorrelation;
                times[threadIdx.x] = otherTime;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) distances[localI] = WAVE_SPEED * static_cast<val_t>(times[0]);
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;
    size_t firstTriangle;
    size_t localTriangles;
    int mpiRank;
    int mpiSize;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Locally owned Kij rows (localN x N)
    std::vector<int> tau;           // Locally owned Tau rows (localN x N)
    PinnedBuffer<val_t> radB;       // Globally replicated B history (T x N)
    std::vector<val_t> localDistances;
    std::vector<val_t> distances;   // Root-only globally ordered result
    std::vector<int> rowCounts;
    std::vector<int> rowDisplacements;

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity,
                          int mpiRank, int mpiSize) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.mpiRank = mpiRank;
    state.mpiSize = mpiSize;

    const size_t baseRows = state.numTriangles / static_cast<size_t>(mpiSize);
    const size_t remainder = state.numTriangles % static_cast<size_t>(mpiSize);
    state.localTriangles = baseRows + (static_cast<size_t>(mpiRank) < remainder ? 1 : 0);
    state.firstTriangle = static_cast<size_t>(mpiRank) * baseRows +
                          std::min(static_cast<size_t>(mpiRank), remainder);
    state.rowCounts.resize(mpiSize);
    state.rowDisplacements.resize(mpiSize);
    int displacement = 0;
    for (int rank = 0; rank < mpiSize; ++rank) {
        const size_t rows = baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
        state.rowCounts[rank] = static_cast<int>(rows);
        state.rowDisplacements[rank] = displacement;
        displacement += state.rowCounts[rank];
    }

    if (mpiRank == 0) printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    if (mpiRank == 0) printf("Building octree...\n");
    state.octree.build(state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.numTriangles); ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    const size_t localEntries = state.localTriangles * state.numTriangles;
    state.kij.resize(localEntries, ZERO);
    state.tau.resize(localEntries, 0);
    state.radB.resize(timesteps * state.numTriangles);
    state.localDistances.resize(state.localTriangles, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    // Each rank computes a disjoint, contiguous range of receiver rows.  The
    // stateless sample generator makes this decomposition bitwise repeatable
    // for a fixed rank count and independent of OpenMP scheduling.
#pragma omp parallel for schedule(dynamic, 1)
    for (long long localI = 0; localI < static_cast<long long>(state.localTriangles); ++localI) {
        const size_t i = state.firstTriangle + static_cast<size_t>(localI);
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.kij[static_cast<size_t>(localI) * state.numTriangles + j] =
                computeKij(i, j, state.triangles, state.octree);
        }
    }
}

class CudaExecutor {
public:
    explicit CudaExecutor(const SimulationState& state) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        allocate(triangles_, state.numTriangles);
        allocate(areas_, state.numTriangles);
        allocate(kij_, state.kij.size());
        allocate(tau_, state.tau.size());
        allocate(radB_, state.radB.size());
        allocate(localCurrent_, state.localTriangles);
        allocate(localDistances_, state.localTriangles);

        std::vector<DeviceTriangle> deviceTriangles(state.numTriangles);
        for (size_t i = 0; i < state.numTriangles; ++i) {
            const Vec3 center = state.triangles[i].center();
            deviceTriangles[i] = {center.x, center.y, center.z};
        }
        CUDA_CHECK(cudaMemcpyAsync(triangles_, deviceTriangles.data(),
                                   state.numTriangles * sizeof(DeviceTriangle),
                                   cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(areas_, state.areas.data(),
                                   state.numTriangles * sizeof(val_t),
                                   cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemsetAsync(radB_, 0, state.radB.size() * sizeof(val_t), stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    CudaExecutor(const CudaExecutor&) = delete;
    CudaExecutor& operator=(const CudaExecutor&) = delete;

    ~CudaExecutor() {
        if (stream_ != nullptr) cudaStreamSynchronize(stream_);
        if (triangles_ != nullptr) cudaFree(triangles_);
        if (areas_ != nullptr) cudaFree(areas_);
        if (kij_ != nullptr) cudaFree(kij_);
        if (tau_ != nullptr) cudaFree(tau_);
        if (radB_ != nullptr) cudaFree(radB_);
        if (localCurrent_ != nullptr) cudaFree(localCurrent_);
        if (localDistances_ != nullptr) cudaFree(localDistances_);
        if (stream_ != nullptr) cudaStreamDestroy(stream_);
    }

    void launchTimeDelays(const SimulationState& state) {
        const size_t entries = state.localTriangles * state.numTriangles;
        if (entries == 0) return;
        const unsigned int blocks = static_cast<unsigned int>(
            (entries + CUDA_THREADS - 1) / CUDA_THREADS);
        computeTauKernel<<<blocks, CUDA_THREADS, 0, stream_>>>(
            triangles_, tau_, static_cast<int>(state.numTriangles),
            static_cast<int>(state.firstTriangle), static_cast<int>(state.localTriangles));
        CUDA_CHECK(cudaGetLastError());
    }

    void finishTimeDelays(SimulationState& state) {
        CUDA_CHECK(cudaMemcpyAsync(state.tau.data(), tau_, state.tau.size() * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    void uploadFormFactors(const SimulationState& state) {
        if (!state.kij.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(kij_, state.kij.data(), state.kij.size() * sizeof(val_t),
                                       cudaMemcpyHostToDevice, stream_));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    void runSimulation(SimulationState& state, bool retainHostHistory) {
        PinnedBuffer<val_t> localRow(state.localTriangles);
        PinnedBuffer<val_t> globalRow(state.numTriangles);
        const int emissionEnd = static_cast<int>(state.numTimesteps / 2);
        for (size_t time = 0; time < state.numTimesteps; ++time) {
            if (state.localTriangles != 0) {
                propagateKernel<<<static_cast<int>(state.localTriangles), CUDA_THREADS,
                                  CUDA_THREADS * sizeof(val_t), stream_>>>(
                    kij_, tau_, areas_, radB_, localCurrent_, static_cast<int>(state.numTriangles),
                    static_cast<int>(state.firstTriangle), static_cast<int>(state.localTriangles),
                    static_cast<int>(time), static_cast<int>(state.sourceIndex), emissionEnd,
                    state.rho[0]);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(localRow.data(), localCurrent_,
                                           state.localTriangles * sizeof(val_t),
                                           cudaMemcpyDeviceToHost, stream_));
                CUDA_CHECK(cudaStreamSynchronize(stream_));
            }

            MPI_Allgatherv(localRow.data(), static_cast<int>(state.localTriangles), MPI_FLOAT,
                           globalRow.data(), state.rowCounts.data(), state.rowDisplacements.data(),
                           MPI_FLOAT, MPI_COMM_WORLD);
            if (retainHostHistory) {
                std::memcpy(state.radB.data() + time * state.numTriangles, globalRow.data(),
                            state.numTriangles * sizeof(val_t));
            }
            CUDA_CHECK(cudaMemcpyAsync(radB_ + time * state.numTriangles, globalRow.data(),
                                       state.numTriangles * sizeof(val_t),
                                       cudaMemcpyHostToDevice, stream_));

            if (state.mpiRank == 0 && ((time + 1) % 10 == 0 || time + 1 == state.numTimesteps)) {
                printf("  Timestep %zu/%zu\n", time + 1, state.numTimesteps);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    void computeDistances(SimulationState& state) {
        if (state.localTriangles != 0) {
            constexpr int distanceThreads = 256;
            const size_t sharedBytes = distanceThreads * (sizeof(val_t) + sizeof(int));
            distanceKernel<<<static_cast<int>(state.localTriangles), distanceThreads, sharedBytes, stream_>>>(
                radB_, localDistances_, static_cast<int>(state.numTriangles),
                static_cast<int>(state.numTimesteps), static_cast<int>(state.firstTriangle),
                static_cast<int>(state.localTriangles), static_cast<int>(state.sourceIndex));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(state.localDistances.data(), localDistances_,
                                       state.localTriangles * sizeof(val_t),
                                       cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }

        if (state.mpiRank == 0) state.distances.resize(state.numTriangles);
        MPI_Gatherv(state.localDistances.data(), static_cast<int>(state.localTriangles), MPI_FLOAT,
                    state.mpiRank == 0 ? state.distances.data() : nullptr,
                    state.rowCounts.data(), state.rowDisplacements.data(), MPI_FLOAT,
                    0, MPI_COMM_WORLD);
    }

private:
    template <typename T>
    void allocate(T*& pointer, size_t count) {
        void* raw = nullptr;
        CUDA_CHECK(cudaMalloc(&raw, std::max<size_t>(count, 1) * sizeof(T)));
        pointer = static_cast<T*>(raw);
    }

    cudaStream_t stream_ = nullptr;
    DeviceTriangle* triangles_ = nullptr;
    val_t* areas_ = nullptr;
    val_t* kij_ = nullptr;
    int* tau_ = nullptr;
    val_t* radB_ = nullptr;
    val_t* localCurrent_ = nullptr;
    val_t* localDistances_ = nullptr;
};

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    long long localNonZeroKij = 0;
#pragma omp parallel for reduction(+:localNonZeroKij) schedule(static)
    for (long long i = 0; i < static_cast<long long>(state.kij.size()); ++i) {
        if (state.kij[static_cast<size_t>(i)] > EPSILON) ++localNonZeroKij;
    }
    long long nonZeroKij = 0;
    MPI_Reduce(&localNonZeroKij, &nonZeroKij, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    int validationPassed = 1;
    if (state.mpiRank == 0) {
        printf("\nValidation:\n");
        bool allNonNegative = true;
        val_t minDist = std::numeric_limits<val_t>::max();
        val_t maxDist = std::numeric_limits<val_t>::lowest();
        val_t sumDist = ZERO;
        int nonZeroCount = 0;

        for (size_t i = 0; i < state.numTriangles; ++i) {
            const val_t d = state.distances[i];
            if (d < 0) {
                allNonNegative = false;
                printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
            }
            if (!std::isfinite(d)) {
                allNonNegative = false;
                printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            }
            minDist = std::min(minDist, d);
            maxDist = std::max(maxDist, d);
            sumDist += d;
            if (d > EPSILON) ++nonZeroCount;
        }

        printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
        printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
        printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

        const val_t sourceDistance = state.distances[state.sourceIndex];
        if (sourceDistance > WAVE_SPEED * 2) {
            printf("  WARNING: Source triangle distance is non-zero: %.4f\n", sourceDistance);
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
        printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);
        printf("  Non-zero form factors: %lld/%zu (%.2f%%)\n",
               nonZeroKij, state.numTriangles * state.numTriangles,
               100.0f * static_cast<val_t>(nonZeroKij) /
                   static_cast<val_t>(state.numTriangles * state.numTriangles));

        validationPassed = (allNonNegative && receivedEnergy != 0 && nonZeroKij != 0) ? 1 : 0;
        if (validationPassed) {
            printf("  Validation: PASSED\n");
        } else {
            printf("  Validation: FAILED\n");
        }
    }
    MPI_Bcast(&validationPassed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return validationPassed != 0;
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
    // subdivisions: 0->20, 1->80, 2->320, 3->1280, 4->5120, 5->20480, 6->81920
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
    printf("               20, 80, 320, 1280, 5120, 20480, 81920\n");
    printf("  -t <num>     Number of timesteps (default: 50)\n");
    printf("  -s <num>     Source triangle index (default: 0)\n");
    printf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    printf("  -v           Enable validation\n");
    printf("  -o           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &providedThreadLevel);
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int mpiRank = 0;
    int mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;

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
            showHelp = true;
        } else {
            if (mpiRank == 0) printf("Unknown option: %s\n", argv[i]);
            parseError = true;
        }
    }

    if (showHelp || parseError || timesteps <= 0 || targetTriangles <= 0) {
        if (mpiRank == 0) {
            if (timesteps <= 0 || targetTriangles <= 0) {
                printf("Triangle count and timestep count must both be positive.\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseError || timesteps <= 0 || targetTriangles <= 0 ? 1 : 0;
    }

    // One rank uses one accelerator when possible.  Multiple local ranks round
    // robin over the node's visible GPUs, which also supports scheduler-provided
    // CUDA_VISIBLE_DEVICES masks without any program options.
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (mpiRank == 0) fprintf(stderr, "No CUDA accelerator is visible to this MPI job.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpiRank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    int exitCode = 0;
    {
        if (mpiRank == 0) {
            printf("Room Response Simulation Benchmark\n");
            printf("===================================\n");
            printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
            printf("Timesteps: %d\n", timesteps);
            printf("Source triangle: %d\n", sourceIdx);
            printf("Reflectivity: %.2f\n", reflectivity);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Hybrid execution: %d MPI ranks, %d OpenMP threads/rank, CUDA device %d on rank 0\n\n",
                   mpiSize, omp_get_max_threads(), deviceCount == 0 ? -1 : 0);
        }

        SimulationState state;
        initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                             static_cast<size_t>(sourceIdx), reflectivity, mpiRank, mpiSize);
        CudaExecutor cuda(state);

        MPI_Barrier(MPI_COMM_WORLD);
        const auto startPre = std::chrono::high_resolution_clock::now();
        if (mpiRank == 0) printf("\nComputing time delays (Tau) and form factors (Kij)...\n");
        cuda.launchTimeDelays(state);
        computeFormFactors(state);
        cuda.finishTimeDelays(state);
        cuda.uploadFormFactors(state);
        const auto endPre = std::chrono::high_resolution_clock::now();
        const long long localPreDuration =
            std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();
        long long preDuration = 0;
        MPI_Reduce(&localPreDuration, &preDuration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        if (mpiRank == 0) printf("Precomputation time (slowest rank): %lld ms\n\n", preDuration);

        MPI_Barrier(MPI_COMM_WORLD);
        const auto startSim = std::chrono::high_resolution_clock::now();
        if (mpiRank == 0) printf("Running wave propagation simulation...\n");
        cuda.runSimulation(state, validate);
        const auto endSim = std::chrono::high_resolution_clock::now();
        const long long localSimDuration =
            std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();
        long long simDuration = 0;
        MPI_Reduce(&localSimDuration, &simDuration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        if (mpiRank == 0) printf("Simulation time (slowest rank): %lld ms\n\n", simDuration);

        MPI_Barrier(MPI_COMM_WORLD);
        const auto startDist = std::chrono::high_resolution_clock::now();
        if (mpiRank == 0) printf("Computing distances via cross-correlation...\n");
        cuda.computeDistances(state);
        const auto endDist = std::chrono::high_resolution_clock::now();
        const long long localDistDuration =
            std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();
        long long distDuration = 0;
        MPI_Reduce(&localDistDuration, &distDuration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

        if (mpiRank == 0) {
            printf("Distance computation time (slowest rank): %lld ms\n\n", distDuration);
            const long long totalTime = preDuration + simDuration + distDuration;
            const size_t n = state.numTriangles;
            const size_t t = state.numTimesteps;
            const double kijOps = static_cast<double>(n) * n;
            const double simOps = static_cast<double>(n) * n * t;
            const double distOps = static_cast<double>(n) * t * t;
            const size_t localMatrixBytes = state.kij.size() * sizeof(val_t) +
                                            state.tau.size() * sizeof(int);
            printf("Total computation time: %lld ms\n", totalTime);
            printf("\nPerformance:\n");
            printf("  Triangles: %zu\n", n);
            printf("  Timesteps: %zu\n", t);
            printf("  Form factor computations: %.2e\n", kijOps);
            printf("  Simulation operations: %.2e\n", simOps);
            printf("  Distance computations: %.2e\n", distOps);
            printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);
            printf("  Rank-0 distributed matrix storage: %.2f MB\n",
                   localMatrixBytes / (1024.0 * 1024.0));
            printf("  Result hash: %016lX\n\n", computeHash(state));

            if (printResults) {
                std::vector<double> distData(state.distances.begin(), state.distances.end());
                print_results(distData, "Distances");
            }
        }

        if (validate && !validateResults(state)) exitCode = 1;
    }

    MPI_Finalize();
    return exitCode;
}
