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
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
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
    __host__ __device__ val_t norm() const { return std::sqrt(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
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

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
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

__device__ Vec3 randomPointInTriangle(const Triangle& t, float u, float v) {
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

__device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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

// ============================================================================
// Visibility Testing (Octree-accelerated)
// ============================================================================

// Preorder storage lets a ray skip a missed subtree without a per-thread stack.
struct FlatNode {
    Vec3 center, halfExtent;
    unsigned end, first, count;

    __device__ bool intersects(Vec3 d, Vec3 midpoint, Vec3 ad) const {
        Vec3 c = midpoint - center;
        if (std::abs(c.x) > halfExtent.x + ad.x) return false;
        if (std::abs(c.y) > halfExtent.y + ad.y) return false;
        if (std::abs(c.z) > halfExtent.z + ad.z) return false;

        if (std::abs(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
        if (std::abs(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
        if (std::abs(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

        return true;
    }
};

void flatten(const Octree& tree, std::vector<FlatNode>& nodes, std::vector<unsigned>& ids) {
    size_t root = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0, static_cast<unsigned>(ids.size()),
                     static_cast<unsigned>(tree.triangleIndices.size())});
    ids.insert(ids.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children)
        if (child) flatten(*child, nodes, ids);
    nodes[root].end = static_cast<unsigned>(nodes.size());
}

__device__ bool blocked(Vec3 from, Vec3 dir, float length, const Triangle* triangles,
                        const FlatNode* nodes, const unsigned* ids, unsigned i, unsigned j) {
    Vec3 d = dir * 0.5f;
    Vec3 mid = from + d;
    Vec3 ad{fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    Vec3 unit = dir / length;
    for (unsigned k = 0; k < nodes[0].end;) {
        const FlatNode& node = nodes[k];
        if (k && !node.intersects(d, mid, ad)) {
            k = node.end;
            continue;
        }
        for (unsigned m = 0; m < node.count; ++m) {
            unsigned id = ids[node.first + m];
            if (id == i || id == j) continue;
            const Triangle& tri = triangles[id];
            float hit = rayTriangleIntersect(from, unit, tri.a, tri.b, tri.c);
            if (hit > EPSILON && hit < length - EPSILON) return true;
        }
        ++k;
    }
    return false;
}

void check(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

template<class T>
struct DeviceBuffer {
    T* ptr = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    void allocate(size_t count) {
        if (count) check(cudaMalloc(&ptr, count * sizeof(T)));
    }
    void upload(const std::vector<T>& src) {
        allocate(src.size());
        check(cudaMemcpy(ptr, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
};

struct SimulationState {
    unsigned numTriangles, numTimesteps, sourceIndex;
    float reflectivity;
    std::vector<Triangle> triangles;
    std::vector<float> distances;
    DeviceBuffer<Triangle> geometry;
    DeviceBuffer<FlatNode> nodes;
    DeviceBuffer<unsigned> ids;
    DeviceBuffer<float> areas, weights, radB;
    DeviceBuffer<unsigned char> delays;
    DeviceBuffer<unsigned> minDelay;
    DeviceBuffer<unsigned long long> nonZero;
};

void initializeSimulation(SimulationState& s, int subdivisions, size_t times,
                          size_t sourceIdx, float reflectivity) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    s.triangles = std::move(mesh.triangles);
    s.numTriangles = s.triangles.size();
    s.numTimesteps = times;
    s.sourceIndex = sourceIdx % s.numTriangles;
    s.reflectivity = reflectivity;
    printf("Generated icosphere mesh with %u triangles\n", s.numTriangles);
    printf("Building octree...\n");
    Octree tree;
    tree.build(s.triangles);
    std::vector<FlatNode> nodes;
    std::vector<unsigned> ids;
    flatten(tree, nodes, ids);
    s.nodes.upload(nodes);
    s.ids.upload(ids);
    s.geometry.upload(s.triangles);
    std::vector<float> areas;
    for (const auto& tri : s.triangles) areas.push_back(tri.area());
    s.areas.upload(areas);

    size_t pairs = size_t(s.numTriangles) * s.numTriangles;
    s.weights.allocate(pairs);
    check(cudaMemset(s.weights.ptr, 0, pairs * sizeof(float)));
    s.delays.allocate(pairs);
    s.minDelay.allocate(1);
    check(cudaMemset(s.minDelay.ptr, 0xff, sizeof(unsigned)));
    s.nonZero.allocate(1);
    check(cudaMemset(s.nonZero.ptr, 0, sizeof(unsigned long long)));
    s.radB.allocate(times * s.numTriangles);
    s.distances.resize(s.numTriangles);
}

__global__ void timeDelays(const Triangle* triangles, unsigned n, unsigned char* delays,
                           unsigned* minDelay) {
    unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned minimum = UINT_MAX;
    if (i < n && j < n) {
        const Triangle& a = triangles[i];
        const Triangle& b = triangles[j];
        Vec3 d = a.center() - b.center();
        // A diameter of 20 units bounds the delay at 40 timesteps.
        unsigned delay = static_cast<unsigned>(ceilf(d.norm() * INV_WAVE_SPEED));
        delays[size_t(i) * n + j] = delay;
        if (i != j) minimum = delay;
    }
    for (int step = 16; step; step /= 2)
        minimum = min(minimum, __shfl_down_sync(0xffffffff, minimum, step));
    if (threadIdx.x == 0) atomicMin(minDelay, minimum);
}

void computeTimeDelays(SimulationState& s) {
    printf("Computing time delays (Tau)...\n");
    unsigned n = s.numTriangles;
    timeDelays<<<dim3((n + 31) / 32, (n + 7) / 8), dim3(32, 8)>>>(s.geometry.ptr, n, s.delays.ptr, s.minDelay.ptr);
    check(cudaGetLastError());
}

__device__ float uniformSample(unsigned bits) {
    return fminf(float(bits) * 0x1p-32f, 0x1.fffffep-1f);
}

__global__ void formFactors(const uint2* pairs, const uint4* draws, unsigned count,
                            const Triangle* triangles, const FlatNode* nodes, const unsigned* ids,
                            const float* areas, unsigned n, float* weights, unsigned long long* nonZero) {
    unsigned thread = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned pair = thread / NUM_RAYS;
    if (pair >= count) return;
    uint2 ij = pairs[pair];
    uint4 u = draws[thread];
    const Triangle& a = triangles[ij.x];
    const Triangle& b = triangles[ij.y];
    Vec3 from = randomPointInTriangle(a, uniformSample(u.x), uniformSample(u.y));
    Vec3 to = randomPointInTriangle(b, uniformSample(u.z), uniformSample(u.w));
    Vec3 dir = to - from;
    float squared = dir.squaredNorm();
    float len = sqrtf(squared);
    float contribution = 0;
    if (squared >= EPSILON) {
        float ca = dir.dot(a._normal) / len;
        float cb = (-dir).dot(b._normal) / len;
        if (ca > 0 && cb > 0 && !blocked(from, dir, len, triangles, nodes, ids, ij.x, ij.y))
            contribution = (ca * cb) / (PI * squared);
    }

    // Exchange ray results within a half warp, preserving scalar addition order.
    unsigned mask = 0xffffu << (threadIdx.x & 16u);
    float sum = 0;
    #pragma unroll
    for (int ray = 0; ray < NUM_RAYS; ++ray)
        sum += __shfl_sync(mask, contribution, ray, NUM_RAYS);
    if (threadIdx.x % NUM_RAYS == 0) {
        float kij = sum * INV_NUM_RAYS;
        weights[size_t(ij.x) * n + ij.y] = fminf(kij * areas[ij.y], ONE);
        if (kij > EPSILON) atomicAdd(nonZero, 1ULL);
    }
}

// Generate whole blocks of the reference MT19937 stream so the host compiler
// can vectorize the recurrence and tempering across adjacent 32-bit words.
class SampleStream {
    using MT = std::mt19937;
    static constexpr size_t N = MT::state_size, M = MT::shift_size;
    std::array<uint32_t, N> words;
    size_t next = N;

    static uint32_t mix(uint32_t a, uint32_t b) {
        uint32_t joined = (a & 0x80000000u) | (b & 0x7fffffffu);
        return (joined >> 1) ^ ((0u - (joined & 1u)) & uint32_t(MT::xor_mask));
    }

    void twist() {
        for (size_t k = 0; k < N - M; ++k)
            words[k] = words[k + M] ^ mix(words[k], words[k + 1]);
        for (size_t k = N - M; k < N - 1; ++k)
            words[k] = words[k - (N - M)] ^ mix(words[k], words[k + 1]);
        words.back() = words[M - 1] ^ mix(words.back(), words.front());
        next = 0;
    }

public:
    explicit SampleStream(uint32_t seed) {
        words[0] = seed;
        for (size_t k = 1; k < N; ++k) {
            uint32_t prev = words[k - 1];
            words[k] = uint32_t(MT::initialization_multiplier) * (prev ^ (prev >> 30)) + k;
        }
    }

    void fill(uint32_t* output, size_t count) {
        while (count) {
            if (next == N) twist();
            size_t chunk = std::min(count, N - next);
            for (size_t k = 0; k < chunk; ++k) {
                uint32_t y = words[next + k];
                y ^= y >> MT::tempering_u;
                y ^= (y << MT::tempering_s) & MT::tempering_b;
                y ^= (y << MT::tempering_t) & MT::tempering_c;
                output[k] = y ^ (y >> MT::tempering_l);
            }
            next += chunk;
            output += chunk;
            count -= chunk;
        }
    }
};

// Bounded, double-buffered batches overlap the original MT19937 stream with GPU tracing.
constexpr unsigned batchSize = 4096;
struct SampleBatch {
    cudaStream_t stream;
    uint2* pairs;
    uint32_t* draws;
    DeviceBuffer<uint2> devicePairs;
    DeviceBuffer<uint4> deviceDraws;
    SampleBatch() {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaMallocHost(&pairs, batchSize * sizeof(uint2)));
        check(cudaMallocHost(&draws, batchSize * NUM_RAYS * sizeof(uint4)));
        devicePairs.allocate(batchSize);
        deviceDraws.allocate(batchSize * NUM_RAYS);
    }
    void submit(const SimulationState& s, unsigned count, SampleStream& rng) {
        rng.fill(draws, count * NUM_RAYS * 4);
        check(cudaMemcpyAsync(devicePairs.ptr, pairs, count * sizeof(uint2), cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(deviceDraws.ptr, draws, count * NUM_RAYS * sizeof(uint4), cudaMemcpyHostToDevice, stream));
        formFactors<<<(count * NUM_RAYS + 127) / 128, 128, 0, stream>>>(devicePairs.ptr, deviceDraws.ptr,
            count, s.geometry.ptr, s.nodes.ptr, s.ids.ptr, s.areas.ptr, s.numTriangles, s.weights.ptr, s.nonZero.ptr);
        check(cudaGetLastError());
    }
    ~SampleBatch() {
        cudaStreamDestroy(stream);
        cudaFreeHost(draws);
        cudaFreeHost(pairs);
    }
};

void computeFormFactors(SimulationState& s) {
    printf("Computing form factors (Kij)...\n");
    check(cudaDeviceSynchronize());
    SampleBatch batches[2];
    unsigned slot = 0, count = 0;
    SampleStream rng(42);
    for (unsigned i = 0; i < s.numTriangles; ++i) {
        for (unsigned j = 0; j < s.numTriangles; ++j) {
            if (i == j || s.triangles[i]._normal.dot(s.triangles[j]._normal) > 0.99f) continue;
            auto& batch = batches[slot];
            batch.pairs[count] = make_uint2(i, j);
            if (++count == batchSize) {
                batch.submit(s, count, rng);
                count = 0;
                slot ^= 1;
                check(cudaStreamSynchronize(batches[slot].stream));
            }
        }
    }
    if (count) batches[slot].submit(s, count, rng);
    check(cudaDeviceSynchronize());
}

// A warp gathers adjacent emitters for one receiver. Shuffles replay the
// gathered values in emitter order so the multiply-add chain stays deterministic.
__global__ void propagate(unsigned firstTime, unsigned times, unsigned n, unsigned source,
                          float rho, const float* weights, const unsigned char* delays,
                          float* radiosity) {
    const unsigned lane = threadIdx.x;
    unsigned i = blockIdx.x * blockDim.y + threadIdx.y;
    unsigned t = firstTime + blockIdx.y;
    if (i >= n) return;
    size_t row = size_t(i) * n;
    float sum = 0;
    for (unsigned base = 0; base < n; base += 32) {
        unsigned j = base + lane;
        float weight = 0, rad = 0;
        if (j < n) {
            unsigned delay = delays[row + j];
            if (delay && delay <= t) {
                weight = weights[row + j];
                if (weight > 0) {
                    rad = radiosity[size_t(t - delay) * n + j];
                    if (rad <= 0) rad = 0;
                }
            }
        }
        if (__any_sync(0xffffffff, rad != 0)) {
            #pragma unroll
            for (int k = 0; k < 32; ++k) {
                float w = __shfl_sync(0xffffffff, weight, k);
                float r = __shfl_sync(0xffffffff, rad, k);
                sum = fmaf(w, r, sum);
            }
        }
    }
    if (lane == 0) {
        float emission = i == source && t < times / 2 ? ONE : ZERO;
        radiosity[size_t(t) * n + i] = rho * sum + emission;
    }
}

void runSimulation(SimulationState& s) {
    printf("Running wave propagation simulation...\n");
    unsigned minDelay;
    check(cudaMemcpy(&minDelay, s.minDelay.ptr, sizeof(minDelay), cudaMemcpyDeviceToHost));
    // All non-diagonal links reach strictly earlier batches.
    for (unsigned t = 0; t < s.numTimesteps; t += minDelay) {
        unsigned steps = std::min(minDelay, s.numTimesteps - t);
        propagate<<<dim3((s.numTriangles + 3) / 4, steps), dim3(32, 4)>>>(
            t, s.numTimesteps, s.numTriangles, s.sourceIndex, s.reflectivity,
            s.weights.ptr, s.delays.ptr, s.radB.ptr);
    }
    check(cudaDeviceSynchronize());
}

__global__ void correlations(unsigned n, unsigned times, unsigned source, const float* radiosity,
                             float* corr) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lag = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n) return;
    for (; lag < times; lag += gridDim.y * blockDim.y) {
        float sum = 0;
        for (unsigned tt = lag; tt < times; ++tt)
            sum += radiosity[size_t(tt - lag) * n + source] * radiosity[size_t(tt) * n + i];
        corr[size_t(lag) * n + i] = sum;
    }
}

__global__ void bestDistances(unsigned n, unsigned times, const float* corr, float* distances) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned best = 0;
    float peak = 0;
    for (unsigned t = 0; t < times; ++t) {
        float sum = corr[size_t(t) * n + i];
        if (sum > peak) { peak = sum; best = t; }
    }
    distances[i] = WAVE_SPEED * best;
}

void computeDistances(SimulationState& s) {
    printf("Computing distances via cross-correlation...\n");
    DeviceBuffer<float> corr, distances;
    unsigned n = s.numTriangles, times = s.numTimesteps;
    corr.allocate(size_t(n) * times);
    distances.allocate(n);
    if (times)
        correlations<<<dim3((n + 31) / 32, std::min(65535u, (times + 3) / 4)), dim3(32, 4)>>>(
            n, times, s.sourceIndex, s.radB.ptr, corr.ptr);
    bestDistances<<<(n + 127) / 128, 128>>>(n, times, corr.ptr, distances.ptr);
    check(cudaMemcpy(s.distances.data(), distances.ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Validation
// ============================================================================

__global__ void countReceivers(unsigned n, unsigned times, const float* radiosity, unsigned* count) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (unsigned t = 0; t < times; ++t) {
        if (radiosity[size_t(t) * n + i] > EPSILON) {
            atomicAdd(count, 1u);
            break;
        }
    }
}

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
    printf("  Non-zero distances: %d/%u\n", nonZeroCount, state.numTriangles);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    DeviceBuffer<unsigned> count;
    count.allocate(1);
    check(cudaMemset(count.ptr, 0, sizeof(unsigned)));
    countReceivers<<<(state.numTriangles + 127) / 128, 128>>>(
        state.numTriangles, state.numTimesteps, state.radB.ptr, count.ptr);
    unsigned receivedEnergy;
    check(cudaMemcpy(&receivedEnergy, count.ptr, sizeof(receivedEnergy), cudaMemcpyDeviceToHost));
    printf("  Triangles receiving energy: %u/%u\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    unsigned long long nonZeroKij;
    check(cudaMemcpy(&nonZeroKij, state.nonZero.ptr, sizeof(nonZeroKij), cudaMemcpyDeviceToHost));
    size_t pairs = size_t(state.numTriangles) * state.numTriangles;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n", nonZeroKij, pairs,
           100.0f * nonZeroKij / static_cast<float>(pairs));

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
    size_t memTau = n * n * sizeof(unsigned char);
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
