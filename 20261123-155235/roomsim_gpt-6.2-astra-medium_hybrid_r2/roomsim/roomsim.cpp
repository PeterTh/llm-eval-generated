/**
 * Room Response Simulation Benchmark
 * 
 * This is a distributed MPI/OpenMP/CUDA implementation of room impulse response
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
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cfloat>
#include <climits>
#include <cstdint>
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
    __host__ __device__ val_t norm() const { return sqrtf(squaredNorm()); }
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
};

struct FlatNode {
    Vec3 center, halfExtent;
    int end, first, count;
    // Check if a ray intersects this node's bounding box
    __device__ bool rayIntersectsBox(const Vec3& p1, const Vec3& p2) const {
        Vec3 d = (p2 - p1) * 0.5f;
        Vec3 c = p1 + d - center;
        Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

        if (fabsf(c.x) > halfExtent.x + ad.x) return false;
        if (fabsf(c.y) > halfExtent.y + ad.y) return false;
        if (fabsf(c.z) > halfExtent.z + ad.z) return false;

        if (fabsf(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
        if (fabsf(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
        if (fabsf(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

        return true;
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
        int threads = std::min<size_t>(omp_get_max_threads(), (faces.size() + 127) / 128);
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (size_t i = 0; i < faces.size(); ++i) {
            const auto& face = faces[i];
            triangles[i] = Triangle(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Random Number Generation
// ============================================================================

// MT19937 with the serial benchmark's seed and float conversion. Work in
// contiguous blocks so the host compiler can vectorize the twist and temper.
class SampleStream {
    static constexpr int N = 624, M = 397, degree = 19937;
    static constexpr uint32_t twist = 0x9908b0dfU;
    std::array<uint32_t, N> words;
    int cursor = 0;

    static uint32_t mix(uint32_t high, uint32_t low) {
        uint32_t y = (high & 0x80000000U) | (low & 0x7fffffffU);
        return (y >> 1) ^ ((0U - (y & 1U)) & twist);
    }

    void advance() {
        #pragma omp simd
        for (int i = 0; i < N-M; ++i)
            words[i] = words[i+M] ^ mix(words[i], words[i+1]);
        // The wrapped references read the newly generated words.
        for (int i = N-M; i < N-1; ++i)
            words[i] = words[i-(N-M)] ^ mix(words[i], words[i+1]);
        words[N-1] = words[M-1] ^ mix(words[N-1], words[0]);
        cursor = 0;
    }

    // Compute x^offset modulo the MT characteristic polynomial over GF(2).
    // With y = x^624 + x^397, the polynomial is y^32 plus one term
    // x^min(k,31) y^(32-k) per set bit k-1 of the twist mask, all divided
    // by x^31. Only submasks of an exponent survive a binomial expansion.
    void jump(uint64_t offset) {
        std::bitset<degree+1> characteristic;
        for (int k = 0; k <= 32; ++k) {
            if (k && !(twist & (1U << (k-1)))) continue;
            int power = 32-k;
            for (int i = 0; i <= power; ++i) {
                if ((i & power) == i)
                    characteristic.flip(std::min(k,31) + power*M + i*(N-M) - 31);
            }
        }
        std::vector<int> terms;
        for (int i = 0; i <= degree; ++i)
            if (characteristic[i]) terms.push_back(i);

        std::bitset<2*degree> polynomial(1), squared;
        for (int bit = 63; bit >= 0; --bit) {
            squared.reset();
            int shift = (offset >> bit) & 1;
            for (int i = 0; i < degree; ++i)
                if (polynomial[i]) squared[2*i + shift] = true;
            for (int i = 2*degree-1; i >= degree; --i)
                if (squared[i])
                    for (int term : terms) squared.flip(i-degree+term);
            polynomial = squared;
        }

        // Apply the jump polynomial to consecutive untempered MT states.
        std::vector<uint32_t> raw(degree+N);
        for (size_t i = 0; i < raw.size(); ++i) {
            if (cursor == N) advance();
            raw[i] = words[cursor++];
        }
        words.fill(0);
        for (int i = 0; i < degree; ++i) {
            if (!polynomial[i]) continue;
            #pragma omp simd
            for (int j = 0; j < N; ++j) words[j] ^= raw[i+j];
        }
        cursor = 0;
    }

public:
    explicit SampleStream(uint64_t offset) {
        words[0] = 42;
        for (uint32_t i = 1; i < N; ++i)
            words[i] = 1812433253U * (words[i-1] ^ (words[i-1] >> 30)) + i;
        advance();
        if (offset >= (1ULL << 27)) {
            jump(offset);
        } else {
            for (; offset >= N; offset -= N) advance();
            cursor = static_cast<int>(offset);
        }
    }

    void fill(float* out, int count) {
        while (count) {
            if (cursor == N) advance();
            int take = std::min(N-cursor, count);
            #pragma omp simd
            for (int i = 0; i < take; ++i) {
                uint32_t y = words[cursor+i];
                y ^= y >> 11;
                y ^= (y << 7) & 0x9d2c5680U;
                y ^= (y << 15) & 0xefc60000U;
                y ^= y >> 18;
                out[i] = std::min(float(y) * 0x1p-32f, 0x1.fffffep-1f);
            }
            cursor += take;
            out += take;
            count -= take;
        }
    }
};

__device__ Vec3 randomPointInTriangle(const Triangle& t, float u, float v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    return t.a + (t.b - t.a) * u + (t.c - t.a) * v;
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

// Preorder storage lets a ray skip an entire missed subtree in one step.
void flatten(const Octree& tree, std::vector<FlatNode>& nodes, std::vector<int>& indices) {
    const size_t here = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0, static_cast<int>(indices.size()),
                     static_cast<int>(tree.triangleIndices.size())});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children) {
        if (child) flatten(*child, nodes, indices);
    }
    nodes[here].end = static_cast<int>(nodes.size());
}

__device__ float sampleRay(const Triangle* triangles, const FlatNode* nodes, const int* indices,
                           int i, int j, float4 uv) {
    const Triangle& receiver = triangles[i];
    const Triangle& emitter = triangles[j];
    Vec3 from = randomPointInTriangle(receiver, uv.x, uv.y);
    Vec3 to = randomPointInTriangle(emitter, uv.z, uv.w);
    Vec3 v = to - from;
    float distanceSquared = v.squaredNorm();
    if (distanceSquared < EPSILON) return ZERO;
    float length = sqrtf(distanceSquared);
    float ci = v.dot(receiver._normal) / length;
    float cj = (-v).dot(emitter._normal) / length;
    if (ci <= ZERO || cj <= ZERO) return ZERO;

    Vec3 dir = v / length;
    for (int k = 0; k < nodes[0].end;) {
        const FlatNode& node = nodes[k];
        if (k && !node.rayIntersectsBox(from, to)) {
            k = node.end;
            continue;
        }
        for (int t = 0; t < node.count; ++t) {
            int idx = indices[node.first + t];
            if (idx == i || idx == j) continue;
            const Triangle& tri = triangles[idx];
            float d = rayTriangleIntersect(from, dir, tri.a, tri.b, tri.c);
            if (d > EPSILON && d < length - EPSILON) return ZERO;
        }
        ++k;
    }
    return (ci * cj) / (PI * distanceSquared);
}

int rank, ranks;

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: CUDA: %s\n", rank, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

template<class T>
struct DeviceBuffer {
    T* data = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { cudaFree(data); }
    void allocate(size_t n) {
        checkCuda(cudaMalloc(&data, std::max(n, size_t(1)) * sizeof(T)));
    }
    void upload(const std::vector<T>& host) {
        allocate(host.size());
        checkCuda(cudaMemcpy(data, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};

struct SimulationState {
    size_t numTriangles, numTimesteps, sourceIndex;
    int rows, first;
    float reflectivity;
    std::vector<int> counts, offsets;
    std::vector<Triangle> triangles;
    std::vector<float> distances;
    DeviceBuffer<float> weights, history;
    DeviceBuffer<unsigned char> delays;
    unsigned long long nonZeroKij = 0;
    int receivedEnergy = 0;
    int exchangeInterval = 1;
};

void initializeSimulation(SimulationState& s, int subdivisions, int steps, size_t source, float rho) {
    s.triangles = std::move(IcosphereMesh(subdivisions, 10.0f).triangles);
    s.numTriangles = s.triangles.size();
    s.numTimesteps = steps;
    s.sourceIndex = source % s.numTriangles;
    s.reflectivity = rho;
    s.counts.resize(ranks);
    s.offsets.resize(ranks);
    for (int r = 0; r < ranks; ++r) {
        s.offsets[r] = s.numTriangles * r / ranks;
        s.counts[r] = s.numTriangles * (r + 1) / ranks - s.offsets[r];
    }
    s.rows = s.counts[rank];
    s.first = s.offsets[rank];
    s.weights.allocate(s.rows * s.numTriangles);
    s.delays.allocate(s.rows * s.numTriangles);
    s.history.allocate(s.numTimesteps * s.numTriangles);
    if (!rank) s.distances.resize(s.numTriangles);
}

constexpr int THREADS = 128;
constexpr int PAIRS_PER_BATCH = 16384;

__global__ void formFactors(const Triangle* triangles, const FlatNode* nodes, const int* indices,
                            const Vec3* centers, const float* areas, const int* offsets,
                            const float4* samples, size_t start, int pairs, int first, int n,
                            float* weights, unsigned char* delays, unsigned long long* nonzero,
                            int* minimumDelay) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int pair = thread / NUM_RAYS;
    int lane = thread % NUM_RAYS;
    __shared__ float values[THREADS];
    float ray = ZERO;
    size_t pos = start + pair;
    int i = first + pos / n;
    int j = pos % n;
    if (pair < pairs && offsets[pair] >= 0) {
        ray = sampleRay(triangles, nodes, indices, i, j, samples[offsets[pair] + lane]);
    }
    values[threadIdx.x] = ray;
    __syncthreads();
    bool significant = false;
    int delayBound = INT_MAX;
    if (pair < pairs && lane == 0) {
        float sum = ZERO;
        #pragma unroll
        for (int r = 0; r < NUM_RAYS; ++r) sum += values[threadIdx.x + r];
        float kij = sum * INV_NUM_RAYS;
        significant = kij > EPSILON;
        weights[pos] = fminf(kij * areas[j], ONE);
        // The radius is 10 and wave speed is 0.5, so every delay fits in a byte.
        delays[pos] = static_cast<unsigned char>(ceilf((centers[i] - centers[j]).norm() * INV_WAVE_SPEED));
        if (kij > ZERO) delayBound = delays[pos];
    }
    if (minimumDelay) {
        // A warp contains two receiver/emitter pairs, with results in lanes 0 and 16.
        int other = __shfl_xor_sync(0xffffffff, delayBound, NUM_RAYS);
        if (threadIdx.x % 32 == 0 && min(delayBound, other) < INT_MAX)
            atomicMin(minimumDelay, min(delayBound, other));
    }
    if (nonzero) {
        unsigned mask = __ballot_sync(0xffffffff, significant);
        if (threadIdx.x % 32 == 0 && mask) atomicAdd(nonzero, (unsigned long long)__popc(mask));
    }
}

bool samplesPair(const std::vector<Triangle>& triangles, int i, int j) {
    return i != j && triangles[i]._normal.dot(triangles[j]._normal) <= 0.99f;
}

void computeFormFactors(SimulationState& s, bool validate) {
    int n = s.numTriangles;
    std::vector<Vec3> centers(n);
    std::vector<float> areas(n);
    int geometryThreads = std::min(omp_get_max_threads(), (n + 127) / 128);
    #pragma omp parallel for schedule(static) num_threads(geometryThreads)
    for (int i = 0; i < n; ++i) {
        centers[i] = s.triangles[i].center();
        areas[i] = s.triangles[i].area();
    }

    std::vector<FlatNode> nodes;
    std::vector<int> indices;
    {
        Octree tree;
        tree.build(s.triangles);
        flatten(tree, nodes, indices);
    }
    DeviceBuffer<Triangle> dTriangles;
    DeviceBuffer<FlatNode> dNodes;
    DeviceBuffer<int> dIndices;
    DeviceBuffer<Vec3> dCenters;
    DeviceBuffer<float> dAreas;
    dTriangles.upload(s.triangles);
    dNodes.upload(nodes);
    dIndices.upload(indices);
    dCenters.upload(centers);
    dAreas.upload(areas);
    DeviceBuffer<unsigned long long> counter;
    if (validate) {
        counter.allocate(1);
        checkCuda(cudaMemset(counter.data, 0, sizeof(unsigned long long)));
    }

    unsigned long long preceding = 0;
    DeviceBuffer<int> minDelay;
    if (ranks > 1) {
        minDelay.upload(std::vector<int>{INT_MAX});
        unsigned long long active = 0;
        int threads = std::min(omp_get_max_threads(), std::max(1, s.rows));
        #pragma omp parallel for reduction(+:active) schedule(static) num_threads(threads)
        for (int row = 0; row < s.rows; ++row) {
            for (int j = 0; j < n; ++j) active += samplesPair(s.triangles, s.first + row, j);
        }
        MPI_Exscan(&active, &preceding, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    }
    checkCuda(cudaStreamSynchronize(nullptr));
    SampleStream rng(rank && s.rows ? preceding * NUM_RAYS * 4 : 0);

    // Double-buffer the bounded sample stream so the host can fill the next
    // batch while CUDA traces rays from the preceding batch.
    struct Batch {
        cudaStream_t stream;
        float* samples;
        int* offsets;
        DeviceBuffer<float4> dSamples;
        DeviceBuffer<int> dOffsets;
    };
    Batch batches[2];
    size_t total = s.rows * s.numTriangles;
    int capacity = std::min<size_t>(PAIRS_PER_BATCH, std::max(size_t(1), total));
    for (auto& b : batches) {
        checkCuda(cudaStreamCreateWithFlags(&b.stream, cudaStreamNonBlocking));
        checkCuda(cudaMallocHost(&b.samples, capacity * NUM_RAYS * sizeof(float4)));
        checkCuda(cudaMallocHost(&b.offsets, capacity * sizeof(int)));
        b.dSamples.allocate(capacity * NUM_RAYS);
        b.dOffsets.allocate(capacity);
    }
    for (size_t start = 0; start < total; start += capacity) {
        auto& b = batches[start / capacity % 2];
        checkCuda(cudaStreamSynchronize(b.stream));
        int count = std::min<size_t>(capacity, total - start);
        int used = 0;
        for (int k = 0; k < count; ++k) {
            size_t pos = start + k;
            b.offsets[k] = -1;
            if (!samplesPair(s.triangles, s.first + pos / n, pos % n)) continue;
            b.offsets[k] = used;
            used += NUM_RAYS;
        }
        rng.fill(b.samples, used * 4);
        checkCuda(cudaMemcpyAsync(b.dOffsets.data, b.offsets, count * sizeof(int),
                                  cudaMemcpyHostToDevice, b.stream));
        checkCuda(cudaMemcpyAsync(b.dSamples.data, b.samples, used * sizeof(float4),
                                  cudaMemcpyHostToDevice, b.stream));
        formFactors<<<(count * NUM_RAYS + THREADS - 1) / THREADS, THREADS, 0, b.stream>>>(
            dTriangles.data, dNodes.data, dIndices.data, dCenters.data, dAreas.data,
            b.dOffsets.data, b.dSamples.data, start, count, s.first, n, s.weights.data,
            s.delays.data, counter.data, minDelay.data);
        checkCuda(cudaGetLastError());
    }
    for (auto& b : batches) {
        checkCuda(cudaStreamSynchronize(b.stream));
        checkCuda(cudaStreamDestroy(b.stream));
        checkCuda(cudaFreeHost(b.samples));
        checkCuda(cudaFreeHost(b.offsets));
    }
    if (ranks > 1) {
        int bound;
        checkCuda(cudaMemcpy(&bound, minDelay.data, sizeof(bound), cudaMemcpyDeviceToHost));
        MPI_Allreduce(&bound, &s.exchangeInterval, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }
    if (validate) {
        unsigned long long local;
        checkCuda(cudaMemcpy(&local, counter.data, sizeof(local), cudaMemcpyDeviceToHost));
        MPI_Reduce(&local, &s.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
}

__global__ void propagate(const float* weights, const unsigned char* delays, float* history,
                          int n, int first, int source, int time, int timeOff, float rho) {
    int row = blockIdx.x;
    size_t base = size_t(row) * n;
    __shared__ float w[THREADS], b[THREADS];
    float sum = ZERO;
    for (int j = 0; j < n; j += THREADS) {
        int col = j + threadIdx.x;
        float weight = ZERO, energy = ZERO;
        if (col < n) {
            int delay = delays[base + col];
            if (delay && delay <= time) {
                weight = weights[base + col];
                if (weight > ZERO) energy = history[size_t(time - delay) * n + col];
            }
        }
        w[threadIdx.x] = weight;
        b[threadIdx.x] = energy;
        __syncthreads();
        // Keep the serial emitter order, including rounding near correlation ties.
        if (!threadIdx.x) {
            for (int k = 0; k < min(THREADS, n - j); ++k) {
                if (b[k] > ZERO) sum += w[k] * b[k];
            }
        }
        __syncthreads();
    }
    if (!threadIdx.x) {
        history[size_t(time) * n + first + row] =
            rho * sum + float(first + row == source && time < timeOff);
    }
}

// With timesteps stored in rows, a resized column type lets Allgatherv
// exchange several timesteps while retaining the ordinary triangle offsets.
MPI_Datatype historyColumn(int steps, int n) {
    MPI_Datatype column, resized;
    MPI_Type_vector(steps, 1, n, MPI_FLOAT, &column);
    MPI_Type_create_resized(column, 0, sizeof(float), &resized);
    MPI_Type_commit(&resized);
    MPI_Type_free(&column);
    return resized;
}

void runSimulation(SimulationState& s) {
    if (s.numTimesteps == 0) return;
    int steps = s.numTimesteps;
    // Every nonzero coupling reads at least this far back in history.
    int interval = ranks == 1 ? steps : std::min(steps, s.exchangeInterval);
    float* window = nullptr;
    MPI_Datatype column = MPI_FLOAT, tail = MPI_FLOAT;
    if (ranks > 1) {
        checkCuda(cudaMallocHost(&window, interval * s.numTriangles * sizeof(float)));
        if (interval > 1) column = historyColumn(interval, s.numTriangles);
        if (steps % interval > 1) tail = historyColumn(steps % interval, s.numTriangles);
    }
    for (int begin = 0; begin < steps;) {
        int count = std::min(interval, steps - begin);
        if (s.rows) {
            for (int t = begin; t < begin + count; ++t) {
                propagate<<<s.rows, THREADS>>>(s.weights.data, s.delays.data, s.history.data,
                    s.numTriangles, s.first, s.sourceIndex, t, steps / 2, s.reflectivity);
                checkCuda(cudaGetLastError());
            }
        }
        if (ranks > 1) {
            float* current = s.history.data + begin * s.numTriangles;
            size_t pitch = s.numTriangles * sizeof(float);
            if (s.rows) checkCuda(cudaMemcpy2D(window + s.first, pitch, current + s.first, pitch,
                                               s.rows * sizeof(float), count, cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, window, s.counts.data(), s.offsets.data(),
                           count == interval ? column : tail, MPI_COMM_WORLD);
            checkCuda(cudaMemcpy(current, window, count * pitch, cudaMemcpyHostToDevice));
        }
        begin += count;
    }
    checkCuda(cudaDeviceSynchronize());
    if (window) checkCuda(cudaFreeHost(window));
    if (column != MPI_FLOAT) MPI_Type_free(&column);
    if (tail != MPI_FLOAT) MPI_Type_free(&tail);
}

// Vectorize across lags: each lane still accumulates its time samples in
// ascending order, so the original rounding and first-maximum rule are kept.
float correlationDistance(const std::vector<float>& source, const std::vector<float>& signal) {
    constexpr int width = 16;
    size_t steps = signal.size();
    float best = ZERO;
    size_t bestLag = 0;
    for (size_t base = 0; base < steps; base += width) {
        float sums[width] = {};
        int lanes = std::min<size_t>(width, steps - base);
        size_t common = steps - base - lanes + 1;
        for (size_t t = 0; t < common; ++t) {
            #pragma omp simd
            for (int lane = 0; lane < lanes; ++lane)
                sums[lane] += source[t] * signal[t + base + lane];
        }
        for (size_t t = common; t < steps - base; ++t) {
            int remaining = steps - base - t;
            #pragma omp simd
            for (int lane = 0; lane < remaining; ++lane)
                sums[lane] += source[t] * signal[t + base + lane];
        }
        for (int lane = 0; lane < lanes; ++lane) {
            if (sums[lane] > best) {
                best = sums[lane];
                bestLag = base + lane;
            }
        }
    }
    return WAVE_SPEED * bestLag;
}

void computeDistances(SimulationState& s, bool validate) {
    std::vector<float> local(s.numTimesteps * s.rows), source(s.numTimesteps), distances(s.rows);
    if (s.numTimesteps && s.rows) {
        checkCuda(cudaMemcpy2D(local.data(), s.rows * sizeof(float), s.history.data + s.first,
            s.numTriangles * sizeof(float), s.rows * sizeof(float), s.numTimesteps, cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy2D(source.data(), sizeof(float), s.history.data + s.sourceIndex,
            s.numTriangles * sizeof(float), sizeof(float), s.numTimesteps, cudaMemcpyDeviceToHost));
    }
    int received = 0;
    int threads = std::min(omp_get_max_threads(), std::max(1, s.rows));
    #pragma omp parallel reduction(+:received) num_threads(threads)
    {
        std::vector<float> signal(s.numTimesteps);
        #pragma omp for schedule(static)
        for (int row = 0; row < s.rows; ++row) {
            bool hasEnergy = false;
            for (size_t t = 0; t < s.numTimesteps; ++t) {
                signal[t] = local[t * s.rows + row];
                hasEnergy |= signal[t] > EPSILON;
            }
            received += hasEnergy;
            distances[row] = correlationDistance(source, signal);
        }
    }
    MPI_Gatherv(distances.data(), s.rows, MPI_FLOAT, s.distances.data(), s.counts.data(),
                s.offsets.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (validate) MPI_Reduce(&received, &s.receivedEnergy, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    printf("\nValidation:\n");

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = FLT_MAX;
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
    int receivedEnergy = state.receivedEnergy;
    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    auto nonZeroKij = state.nonZeroKij;
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

int main(int argc, char** argv) {
    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);

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
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (!rank) printf("Unknown option: %s\n", argv[i]);
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (timesteps < 0) {
        if (!rank) fprintf(stderr, "Timesteps must be non-negative\n");
        MPI_Finalize();
        return 1;
    }
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    checkCuda(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d requires a CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % devices));
    SimulationState state;
    initializeSimulation(state, subdivisions, timesteps, static_cast<size_t>(sourceIdx), reflectivity);
    if (!rank) {
        printf("Room Response Simulation Benchmark\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\nSource triangle: %d\nReflectivity: %.2f\n", timesteps, sourceIdx, reflectivity);
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    computeFormFactors(state, validate);
    MPI_Barrier(MPI_COMM_WORLD);
    double pre = MPI_Wtime();
    runSimulation(state);
    MPI_Barrier(MPI_COMM_WORLD);
    double sim = MPI_Wtime();
    computeDistances(state, validate);
    double end = MPI_Wtime();
    int status = 0;
    if (!rank) {
        long preDuration = static_cast<long>((pre - start) * 1000);
        long simDuration = static_cast<long>((sim - pre) * 1000);
        long distDuration = static_cast<long>((end - sim) * 1000);
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("Simulation time: %ld ms\n", simDuration);
        printf("Distance computation time: %ld ms\n", distDuration);

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

        size_t deviceMatrices = n * n * (sizeof(float) + sizeof(unsigned char))
                              + ranks * n * t * sizeof(float);
        printf("  Memory usage (distributed simulation matrices): %.2f MB\n",
               deviceMatrices / (1024.0 * 1024.0));

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

        if (validate && !validateResults(state)) status = 1;
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
