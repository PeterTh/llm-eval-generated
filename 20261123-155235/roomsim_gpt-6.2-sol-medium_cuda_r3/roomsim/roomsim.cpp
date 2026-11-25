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

#include <cuda_runtime.h>

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

    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }

    // Match the CPU benchmark's rounding at each multiplication and
    // addition. Control contraction explicitly in the CUDA kernels.
    __host__ __device__ val_t dot(const Vec3& o) const {
#ifdef __CUDA_ARCH__
        return fmaf(z, o.z, fmaf(x, o.x, y * o.y));
#else
        return x * o.x + y * o.y + z * o.z;
#endif
    }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
#ifdef __CUDA_ARCH__
        return {fmaf(y, o.z, -z * o.y), fmaf(z, o.x, -x * o.z), fmaf(x, o.y, -y * o.x)};
#else
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
#endif
    }

    __host__ __device__ val_t squaredNorm() const {
#ifdef __CUDA_ARCH__
        return dot(*this);
#else
        return x * x + y * y + z * z;
#endif
    }
    __host__ __device__ val_t norm() const { return sqrtf(squaredNorm()); }
    Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    Vec3 center() const { return (a + b + c) / 3.0f; }

    val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }
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
// CUDA storage and geometry
// ============================================================================

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call)

template<class T, bool Pinned = false>
class CudaBuffer {
    T* ptr = nullptr;

public:
    CudaBuffer() = default;
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    ~CudaBuffer() {
        if (ptr) {
            if constexpr (Pinned) cudaFreeHost(ptr);
            else cudaFree(ptr);
        }
    }

    void allocate(size_t count, bool managed = false) {
        if (count == 0) return;
        if constexpr (Pinned) {
            CUDA_CHECK(cudaMallocHost(&ptr, count * sizeof(T)));
        } else if (managed) {
            CUDA_CHECK(cudaMallocManaged(&ptr, count * sizeof(T)));
        } else {
            CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
        }
    }

    void upload(const std::vector<T>& values) {
        allocate(values.size());
        CUDA_CHECK(cudaMemcpy(ptr, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }

    T* get() const { return ptr; }
};

class CudaStream {
    cudaStream_t stream;

public:
    CudaStream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    ~CudaStream() {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    operator cudaStream_t() const { return stream; }
};

constexpr idx_t InternalNode = UINT32_MAX;
constexpr idx_t LastLeafTriangle = 1u << 31;

struct alignas(16) DeviceNode {
    Vec3 center, halfExtent;
    idx_t escape;
    idx_t firstTriangle;
};

void flattenOctree(const Octree& tree, std::vector<DeviceNode>& nodes, std::vector<idx_t>& indices) {
    size_t node = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0, InternalNode});
    if (tree.triangleIndices.empty()) {
        for (const auto& child : tree.children) {
            if (child) flattenOctree(*child, nodes, indices);
        }
    } else {
        nodes[node].firstTriangle = static_cast<idx_t>(indices.size());
        indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
        indices.back() |= LastLeafTriangle;
    }
    nodes[node].escape = static_cast<idx_t>(nodes.size());
}

struct Surface {
    Vec3 a, ab, ac, normal, center;
    val_t area;
};

struct Geometry {
    const Surface* surfaces;
    const DeviceNode* nodes;
    const idx_t* indices;
    idx_t numNodes;
    idx_t firstNode;
};

__device__ bool rayIntersectsBox(const DeviceNode& node, const Vec3& midpoint,
                                 const Vec3& d, const Vec3& ad) {
    Vec3 c = midpoint - node.center;
    Vec3 h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y || fabsf(c.z) > h.z + ad.z)
        return false;

    return fabsf(fmaf(d.y, c.z, -d.z * c.y)) <= fmaf(h.y, ad.z, h.z * ad.y) + EPSILON &&
           fabsf(fmaf(d.z, c.x, -d.x * c.z)) <= fmaf(h.z, ad.x, h.x * ad.z) + EPSILON &&
           fabsf(fmaf(d.x, c.y, -d.y * c.x)) <= fmaf(h.x, ad.y, h.y * ad.x) + EPSILON;
}

// Möller–Trumbore intersection against the open segment, with the original
// tolerance at both endpoints.
__device__ bool rayIntersectsTriangle(const Vec3& from, const Vec3& dir, val_t limit,
                                      const Surface& tri) {
    Vec3 pvec = dir.cross(tri.ac);
    val_t det = tri.ab.dot(pvec);
    if (fabsf(det) < EPSILON) return false;

    val_t invDet = ONE / det;
    Vec3 tvec = from - tri.a;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < ZERO || u > ONE) return false;

    Vec3 qvec = tvec.cross(tri.ab);
    val_t v = dir.dot(qvec) * invDet;
    if (v < ZERO || u + v > ONE) return false;

    val_t dist = tri.ac.dot(qvec) * invDet;
    return dist > EPSILON && dist < limit;
}

__device__ bool isRayBlocked(const Geometry& geom, idx_t i, idx_t j, const Vec3& from,
                             const Vec3& delta, val_t rayLen) {
    Vec3 d = delta * 0.5f;
    Vec3 midpoint = from + d;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    Vec3 dir = delta / rayLen;
    val_t limit = rayLen - EPSILON;

    // Depth-first order and escape indices let a ray skip a missed subtree
    // without a per-thread stack. The original traversal skips the root test.
    for (idx_t n = geom.firstNode; n < geom.numNodes;) {
        const DeviceNode& node = geom.nodes[n];
        if (n != 0 && !rayIntersectsBox(node, midpoint, d, ad)) {
            n = node.escape;
            continue;
        }

        idx_t k = node.firstTriangle;
        if (k != InternalNode) {
            idx_t entry;
            do {
                entry = geom.indices[k++];
                idx_t tri = entry & ~LastLeafTriangle;
                if (tri != i && tri != j &&
                    rayIntersectsTriangle(from, dir, limit, geom.surfaces[tri]))
                    return true;
            } while (!(entry & LastLeafTriangle));
        }
        ++n;
    }
    return false;
}

// ============================================================================
// Deterministic sampling and form factors
// ============================================================================

constexpr int MtSize = 624;
constexpr int MtOffset = 397;
constexpr int MtStep = MtSize - MtOffset;
constexpr int PairsPerBlock = 64;
constexpr int WordsPerPair = NUM_RAYS * 4;
constexpr int TraceThreads = 256;

__host__ __device__ uint32_t mtTwist(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t bits = (a & 0x80000000u) | (b & 0x7fffffffu);
    return c ^ (bits >> 1) ^ ((0u - (bits & 1u)) & 0x9908b0dfu);
}

struct alignas(16) MtCheckpoint {
    uint32_t state[MtSize];
};

class MtStream {
    alignas(64) uint32_t states[2][MtSize];
    uint64_t position = 0;
    int slot = 0;

    // Advance by an MT19937 state. Each loop can be vectorized on the host.
    void advance() {
        const uint32_t* prev = states[slot];
        slot ^= 1;
        uint32_t* next = states[slot];
        for (int i = 0; i < MtStep; ++i)
            next[i] = mtTwist(prev[i], prev[i + 1], prev[i + MtOffset]);
        for (int i = MtStep; i < MtSize - 1; ++i)
            next[i] = mtTwist(prev[i], prev[i + 1], next[i - MtStep]);
        next[MtSize - 1] = mtTwist(prev[MtSize - 1], next[0], next[MtOffset - 1]);
        position += MtSize;
    }

public:
    MtStream() {
        auto& seed = states[0];
        seed[0] = 42;
        for (uint32_t i = 1; i < MtSize; ++i)
            seed[i] = 1812433253u * (seed[i - 1] ^ (seed[i - 1] >> 30)) + i;
    }

    void checkpoint(uint64_t word, MtCheckpoint& result) {
        while (position < word) advance();

        // The 624 words immediately before this block's first sample may
        // straddle two states. Seed each CUDA block at its exact offset.
        size_t tail = position - word;
        size_t head = MtSize - tail;
        if (tail != 0)
            memcpy(result.state, states[slot ^ 1] + head, tail * sizeof(uint32_t));
        memcpy(result.state + tail, states[slot], head * sizeof(uint32_t));
    }
};

__device__ val_t uniformFloat(uint32_t word) {
    word ^= word >> 11;
    word ^= (word << 7) & 0x9d2c5680u;
    word ^= (word << 15) & 0xefc60000u;
    word ^= word >> 18;
    // Match uniform_real_distribution<float> over the 32-bit MT19937 range.
    return fminf(__uint2float_rn(word) * 0x1p-32f, 0x1.fffffep-1f);
}

__device__ Vec3 samplePoint(const Surface& tri, const uint32_t* words) {
    val_t u = uniformFloat(words[0]);
    val_t v = uniformFloat(words[1]);
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return {fmaf(tri.ac.x, v, fmaf(tri.ab.x, u, tri.a.x)),
            fmaf(tri.ac.y, v, fmaf(tri.ab.y, u, tri.a.y)),
            fmaf(tri.ac.z, v, fmaf(tri.ab.z, u, tri.a.z))};
}

__device__ bool needsSamples(const Surface* surfaces, idx_t i, idx_t j) {
    return i != j && surfaces[i].normal.dot(surfaces[j].normal) <= 0.99f;
}

__global__ void countSampledPairs(const Surface* surfaces, idx_t n, size_t first,
                                  size_t pairCount, int* counts) {
    size_t pair = first + static_cast<size_t>(blockIdx.x) * PairsPerBlock + threadIdx.x;
    idx_t i = pair / n;
    idx_t j = pair - static_cast<size_t>(i) * n;
    bool sample = pair < pairCount && needsSamples(surfaces, i, j);
    int count = __popc(__ballot_sync(0xffffffffu, sample));
    __shared__ int secondWarpCount;
    if (threadIdx.x == 32) secondWarpCount = count;
    __syncthreads();
    if (threadIdx.x == 0) counts[blockIdx.x] = count + secondWarpCount;
}

__device__ val_t traceSample(const Geometry& geom, idx_t i, idx_t j, const uint32_t* words) {
    const Surface& a = geom.surfaces[i];
    const Surface& b = geom.surfaces[j];
    Vec3 from = samplePoint(a, words);
    Vec3 to = samplePoint(b, words + 2);
    Vec3 delta = to - from;
    val_t distSqr = delta.squaredNorm();
    if (distSqr < EPSILON) return ZERO;

    val_t rayLen = sqrtf(distSqr);
    val_t cosI = delta.dot(a.normal) / rayLen;
    val_t cosJ = (-delta).dot(b.normal) / rayLen;
    if (cosI <= ZERO || cosJ <= ZERO || isRayBlocked(geom, i, j, from, delta, rayLen))
        return ZERO;

    return (cosI * cosJ) / (PI * distSqr);
}

struct Propagation {
    val_t* weights;
    uint8_t* delays;
    val_t* radiosity;
    idx_t n;
    int timesteps;
    idx_t source;
    val_t reflectivity;
};

using count_t = unsigned long long;

__global__ void computeFormFactors(Geometry geom, Propagation wave, size_t first,
                                    size_t pairCount, const MtCheckpoint* checkpoints,
                                    count_t* nonZeroKij) {
    __shared__ uint32_t words[MtSize + PairsPerBlock * WordsPerPair];
    __shared__ val_t rays[PairsPerBlock * NUM_RAYS];
    __shared__ idx_t receivers[PairsPerBlock], emitters[PairsPerBlock];
    __shared__ int warpCounts[PairsPerBlock / 32];

    const int lane = threadIdx.x;
    const size_t blockStart = first + static_cast<size_t>(blockIdx.x) * PairsPerBlock;
    const size_t pair = blockStart + lane;
    bool active = false;
    idx_t i = 0, j = 0;
    int rank = 0;
    if (lane < PairsPerBlock) {
        i = pair / wave.n;
        j = pair - static_cast<size_t>(i) * wave.n;
        active = pair < pairCount && needsSamples(geom.surfaces, i, j);
        unsigned mask = __ballot_sync(0xffffffffu, active);
        int warpLane = lane % 32;
        if (warpLane == 0) warpCounts[lane / 32] = __popc(mask);
        rank = __popc(mask & ((1u << warpLane) - 1u));
    }
    __syncthreads();

    int sampledPairs = warpCounts[0] + warpCounts[1];
    if (sampledPairs != 0) {
        const MtCheckpoint& checkpoint = checkpoints[blockIdx.x];
        if (active) {
            if (lane >= 32) rank += warpCounts[0];
            receivers[rank] = i;
            emitters[rank] = j;
        }

        for (int k = lane; k < MtSize; k += blockDim.x)
            words[k] = checkpoint.state[k];

        // Every group of MtStep words can be generated in parallel from the
        // preceding words. Leave the raw words in shared memory for the rays.
        const int used = sampledPairs * WordsPerPair;
        for (int base = 0; base < used; base += MtStep) {
            __syncthreads();
            int k = base + lane;
            if (lane < MtStep && k < used)
                words[k + MtSize] = mtTwist(words[k], words[k + 1], words[k + MtOffset]);
        }
        __syncthreads();

        for (int k = lane; k < sampledPairs * NUM_RAYS; k += blockDim.x) {
            int p = k / NUM_RAYS;
            rays[k] = traceSample(geom, receivers[p], emitters[p], words + MtSize + 4 * k);
        }
        __syncthreads();
    }

    if (lane < PairsPerBlock) {
        val_t kij = ZERO;
        if (active) {
            #pragma unroll
            for (int r = 0; r < NUM_RAYS; ++r) kij += rays[rank * NUM_RAYS + r];
            kij *= INV_NUM_RAYS;
        }

        if (wave.weights && pair < pairCount) {
            // A zero delay encodes an inactive pair. Every pair with a
            // nonzero form factor connects distinct icosphere centroids.
            int delay = 0;
            if (kij > ZERO) {
                val_t dist = (geom.surfaces[i].center - geom.surfaces[j].center).norm();
                delay = __float2int_ru(dist * INV_WAVE_SPEED);
            }
            wave.delays[pair] = delay;
            if (delay != 0)
                wave.weights[pair] = fminf(kij * geom.surfaces[j].area, ONE);
        }

        if (nonZeroKij) {
            int count = __popc(__ballot_sync(0xffffffffu, kij > EPSILON));
            if (lane % 32 == 0 && count != 0) atomicAdd(nonZeroKij, count_t(count));
        }
    }
}

// ============================================================================
// Wave propagation and correlation
// ============================================================================

constexpr int PropagationThreads = 128;

__global__ void propagateTimestep(Propagation wave, int time) {
    const idx_t receiver = blockIdx.x;
    const size_t row = static_cast<size_t>(receiver) * wave.n;
    __shared__ val_t contribution[PropagationThreads];
    val_t sum = ZERO;

    if (time != 0) {
        for (idx_t base = 0; base < wave.n; base += blockDim.x) {
            idx_t j = base + threadIdx.x;
            val_t term = ZERO;
            if (j < wave.n) {
                int delay = wave.delays[row + j];
                if (delay != 0 && delay <= time) {
                    val_t rad = wave.radiosity[static_cast<size_t>(time - delay) * wave.n + j];
                    if (!(rad <= ZERO)) term = wave.weights[row + j] * rad;
                }
            }
            contribution[threadIdx.x] = term;
            __syncthreads();
            if (threadIdx.x == 0) {
                // Preserve the original emitter order, including rounding
                // after each addition.
                #pragma unroll
                for (int k = 0; k < PropagationThreads; ++k) sum += contribution[k];
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        val_t emission = receiver == wave.source && time < wave.timesteps / 2 ? ONE : ZERO;
        wave.radiosity[static_cast<size_t>(time) * wave.n + receiver] =
            fmaf(wave.reflectivity, sum, emission);
    }
}

constexpr int CorrelationReceivers = 8;
constexpr int CorrelationLanes = 16;

__global__ void correlate(Propagation wave, val_t* distances, count_t* receivedEnergy) {
    const idx_t i = blockIdx.x * CorrelationReceivers + threadIdx.x;
    const int lane = threadIdx.y;
    __shared__ val_t bestCorr[CorrelationLanes][CorrelationReceivers];
    __shared__ int bestLag[CorrelationLanes][CorrelationReceivers];

    val_t best = ZERO;
    int bestT = 0;
    bool received = false;
    if (i < wave.n) {
        for (size_t lag = lane; lag < static_cast<size_t>(wave.timesteps); lag += CorrelationLanes) {
            val_t sum = ZERO;
            for (size_t t = lag; t < static_cast<size_t>(wave.timesteps); ++t) {
                val_t rad = wave.radiosity[t * wave.n + i];
                val_t src = wave.radiosity[(t - lag) * wave.n + wave.source];
                sum += src * rad;
                if (lag == 0 && rad > EPSILON) received = true;
            }
            if (sum > best) {
                best = sum;
                bestT = static_cast<int>(lag);
            }
        }
    }

    bestCorr[lane][threadIdx.x] = best;
    bestLag[lane][threadIdx.x] = bestT;
    __syncthreads();
    if (lane == 0 && i < wave.n) {
        #pragma unroll
        for (int l = 1; l < CorrelationLanes; ++l) {
            val_t corr = bestCorr[l][threadIdx.x];
            int lag = bestLag[l][threadIdx.x];
            if (corr > best || (corr == best && lag < bestT)) {
                best = corr;
                bestT = lag;
            }
        }
        distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
        if (receivedEnergy && received) atomicAdd(receivedEnergy, count_t(1));
    }
}

// ============================================================================
// Benchmark orchestration
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;
    size_t sourceIndex;
    std::vector<val_t> distances;
    count_t nonZeroKij = 0;
    count_t receivedEnergy = 0;
};

struct PrecomputeBatch {
    CudaBuffer<int> counts;
    CudaBuffer<int, true> hostCounts;
    CudaBuffer<MtCheckpoint> checkpoints;
    CudaBuffer<MtCheckpoint, true> hostCheckpoints;
    CudaStream stream;

    void allocate(int blocks) {
        counts.allocate(blocks);
        hostCounts.allocate(blocks);
        checkpoints.allocate(blocks);
        hostCheckpoints.allocate(blocks);
    }
};

class CudaSimulation {
    CudaBuffer<val_t> weights, radiosity, distances;
    CudaBuffer<uint8_t> delays;
    CudaBuffer<count_t> validationCounts;
    std::vector<Surface> surfaces;
    std::vector<DeviceNode> nodes;
    std::vector<idx_t> indices;
    Propagation wave;
    bool validate;

public:
    CudaSimulation(SimulationState& state, int subdivisions, int timesteps,
                   size_t sourceIdx, val_t reflectivity, bool validate)
        : validate(validate) {
        IcosphereMesh mesh(subdivisions, 10.0f);
        const size_t n = mesh.triangles.size();
        state.numTriangles = n;
        state.numTimesteps = timesteps;
        state.sourceIndex = sourceIdx % n;
        state.distances.resize(n);

        printf("Generated icosphere mesh with %zu triangles\n", n);
        printf("Building octree...\n");
        Octree octree;
        octree.build(mesh.triangles);
        flattenOctree(octree, nodes, indices);

        surfaces.reserve(n);
        for (const Triangle& tri : mesh.triangles)
            surfaces.push_back({tri.a, tri.b - tri.a, tri.c - tri.a,
                                tri._normal, tri.center(), tri.area()});

        radiosity.allocate(n * timesteps);
        distances.allocate(n);
        if (validate) {
            validationCounts.allocate(2);
            CUDA_CHECK(cudaMemset(validationCounts.get(), 0, 2 * sizeof(count_t)));
        }

        if (timesteps > 1) {
            size_t freeBytes, totalBytes;
            CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
            // Managed storage allows the largest meshes to exceed VRAM.
            // Ordinary workloads keep both matrices in device memory.
            bool managed = n * n * (sizeof(val_t) + sizeof(uint8_t)) > freeBytes - freeBytes / 8;
            weights.allocate(n * n, managed);
            delays.allocate(n * n, managed);
        }

        wave = {weights.get(), delays.get(), radiosity.get(), static_cast<idx_t>(n),
                timesteps, static_cast<idx_t>(state.sourceIndex), reflectivity};
    }

    size_t memoryBytes() const {
        size_t n = wave.n;
        return n * wave.timesteps * sizeof(val_t) +
               (weights.get() ? n * n * (sizeof(val_t) + sizeof(uint8_t)) : 0);
    }

    void precompute() {
        printf("Computing time delays (Tau)...\n");
        printf("Computing form factors (Kij)...\n");
        CudaBuffer<Surface> deviceSurfaces;
        CudaBuffer<DeviceNode> deviceNodes;
        CudaBuffer<idx_t> deviceIndices;
        deviceSurfaces.upload(surfaces);
        deviceNodes.upload(nodes);
        deviceIndices.upload(indices);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        Geometry geom{deviceSurfaces.get(), deviceNodes.get(), deviceIndices.get(),
                      static_cast<idx_t>(nodes.size()), nodes[0].firstTriangle == InternalNode ? 1u : 0u};

        constexpr int MaxBatchBlocks = 4096;
        const size_t pairs = static_cast<size_t>(wave.n) * wave.n;
        const size_t blocks = (pairs + PairsPerBlock - 1) / PairsPerBlock;
        const int capacity = static_cast<int>(std::min<size_t>(MaxBatchBlocks, blocks));
        PrecomputeBatch batches[2];
        int numBatches = blocks > static_cast<size_t>(capacity) ? 2 : 1;
        for (int i = 0; i < numBatches; ++i) batches[i].allocate(capacity);

        MtStream random;
        uint64_t word = 0;
        size_t progress = 0;
        // Culled pairs consume no samples. Count each block's remaining
        // pairs before assigning their positions in the original stream.
        for (size_t base = 0; base < blocks; base += capacity) {
            auto& batch = batches[(base / capacity) % numBatches];
            int count = static_cast<int>(std::min<size_t>(capacity, blocks - base));
            size_t first = base * PairsPerBlock;
            countSampledPairs<<<count, PairsPerBlock, 0, batch.stream>>>(
                geom.surfaces, wave.n, first, pairs, batch.counts.get());
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(batch.hostCounts.get(), batch.counts.get(), count * sizeof(int),
                                       cudaMemcpyDeviceToHost, batch.stream));
            CUDA_CHECK(cudaStreamSynchronize(batch.stream));

            // The copy and ray kernel from this buffer's previous use have
            // finished. The other stream can still run while we prepare it.
            if (base >= 2 * static_cast<size_t>(capacity))
                reportProgress((base - capacity) * PairsPerBlock / wave.n, progress);

            for (int b = 0; b < count; ++b) {
                random.checkpoint(word, batch.hostCheckpoints.get()[b]);
                word += batch.hostCounts.get()[b] * WordsPerPair;
            }
            CUDA_CHECK(cudaMemcpyAsync(batch.checkpoints.get(), batch.hostCheckpoints.get(),
                                       count * sizeof(MtCheckpoint), cudaMemcpyHostToDevice, batch.stream));
            computeFormFactors<<<count, TraceThreads, 0, batch.stream>>>(
                geom, wave, first, pairs, batch.checkpoints.get(), validationCounts.get());
            CUDA_CHECK(cudaGetLastError());
        }

        for (int i = 0; i < numBatches; ++i) CUDA_CHECK(cudaStreamSynchronize(batches[i].stream));
        reportProgress(wave.n, progress);
    }

    void propagate() {
        printf("Running wave propagation simulation...\n");
        for (int time = 0; time < wave.timesteps; ++time) {
            propagateTimestep<<<wave.n, PropagationThreads>>>(wave, time);
            CUDA_CHECK(cudaGetLastError());
            if ((time + 1) % 10 == 0 || time + 1 == wave.timesteps) {
                CUDA_CHECK(cudaDeviceSynchronize());
                printf("  Timestep %d/%d\n", time + 1, wave.timesteps);
            }
        }
    }

    void computeDistances(SimulationState& state) {
        printf("Computing distances via cross-correlation...\n");
        dim3 threads(CorrelationReceivers, CorrelationLanes);
        correlate<<<(wave.n + threads.x - 1) / threads.x, threads>>>(
            wave, distances.get(), validate ? validationCounts.get() + 1 : nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(state.distances.data(), distances.get(), wave.n * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
        if (validate) {
            count_t counts[2];
            CUDA_CHECK(cudaMemcpy(counts, validationCounts.get(), sizeof(counts), cudaMemcpyDeviceToHost));
            state.nonZeroKij = counts[0];
            state.receivedEnergy = counts[1];
        }
    }

private:
    void reportProgress(size_t completed, size_t& reported) const {
        for (size_t i = reported + 100; i <= completed; i += 100) {
            printf("  Progress: %zu/%u triangles\n", i, wave.n);
            reported = i;
        }
        if (completed == wave.n && reported != completed)
            printf("  Progress: %zu/%u triangles\n", completed, wave.n);
    }
};

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

    printf("  Triangles receiving energy: %llu/%zu\n", state.receivedEnergy, state.numTriangles);

    if (state.receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    count_t nonZeroKij = state.nonZeroKij;
    size_t pairCount = state.numTriangles * state.numTriangles;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           nonZeroKij, pairCount, 100.0f * nonZeroKij / static_cast<val_t>(pairCount));

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

    if (timesteps < 0) {
        fprintf(stderr, "Timesteps must be non-negative\n");
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
    CudaSimulation simulation(state, subdivisions, timesteps,
                              static_cast<size_t>(sourceIdx), reflectivity, validate);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    simulation.precompute();

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    simulation.propagate();

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    simulation.computeDistances(state);

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

    printf("  Memory usage: %.2f MB\n", simulation.memoryBytes() / (1024.0 * 1024.0));

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
    fprintf(stderr, "roomsim: %s\n", error.what());
    return 1;
}
