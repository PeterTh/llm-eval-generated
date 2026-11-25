/**
 * Room Response Simulation Benchmark
 * 
 * This is a distributed, GPU-accelerated implementation of room impulse response
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
#include <math_constants.h>
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>
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
#include <random>
#include <sstream>
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
    __device__ bool rayIntersectsBox(const Vec3& p1, const Vec3& p2) const {
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
};

// Preorder layout lets a ray skip an entire missed subtree in one step.
void flatten(const Octree& tree, std::vector<FlatNode>& nodes, std::vector<int>& indices) {
    size_t i = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0, static_cast<int>(indices.size()),
                     static_cast<int>(tree.triangleIndices.size())});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children)
        if (child) flatten(*child, nodes, indices);
    nodes[i].end = static_cast<int>(nodes.size());
}

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

    if (std::abs(det) < EPSILON) return CUDART_INF_F;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return CUDART_INF_F;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return CUDART_INF_F;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Visibility Testing (Octree-accelerated)
// ============================================================================


void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

template<typename T>
struct DeviceArray {
    DeviceArray() = default;
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    T* data = nullptr;
    void allocate(size_t count) {
        if (count) checkCuda(cudaMalloc(&data, count * sizeof(T)));
    }
    void upload(const std::vector<T>& v) {
        allocate(v.size());
        if (!v.empty())
            checkCuda(cudaMemcpy(data, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~DeviceArray() { if (data) cudaFree(data); }
};

__device__ bool blocked(const Vec3& from, const Vec3& to, const Triangle* triangles,
                        const FlatNode* nodes, const int* indices, int i, int j) {
    Vec3 delta = to - from;
    float length = delta.norm();
    if (length < EPSILON) return true;
    Vec3 direction = delta / length;
    for (int node = 0; node < nodes[0].end;) {
        const FlatNode& box = nodes[node];
        if (node && !box.rayIntersectsBox(from, to)) {
            node = box.end;
            continue;
        }
        for (int k = 0; k < box.count; ++k) {
            int id = indices[box.first + k];
            if (id == i || id == j) continue;
            const Triangle& tri = triangles[id];
            float hit = rayTriangleIntersect(from, direction, tri.a, tri.b, tri.c);
            if (hit > EPSILON && hit < length - EPSILON) return true;
        }
        ++node;
    }
    return false;
}

// Sixteen adjacent lanes trace one pair. Keep the original ray summation order.
__global__ void formFactors(const Triangle* triangles, const Vec3* centers, const float* areas,
                            const FlatNode* nodes, const int* indices, const float4* samples,
                            size_t start, int pairs, int firstRow, int n,
                            float* weights, unsigned char* delays, unsigned long long* nonZero) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int pair = thread / NUM_RAYS;
    unsigned mask = __ballot_sync(0xffffffffU, pair < pairs);
    if (pair >= pairs) return;
    int lane = thread % NUM_RAYS;
    size_t local = start + pair;
    int row = local / n, i = firstRow + row, j = local % n;
    float4 sample = samples[thread];
    float value = 0;
    if (sample.x >= 0) {
        const Triangle& a = triangles[i];
        const Triangle& b = triangles[j];
        Vec3 from = randomPointInTriangle(a, sample.x, sample.y);
        Vec3 to = randomPointInTriangle(b, sample.z, sample.w);
        Vec3 v = to - from;
        float sq = v.squaredNorm();
        if (sq >= EPSILON) {
            float len = sqrtf(sq);
            float ci = v.dot(a._normal) / len, cj = (-v).dot(b._normal) / len;
            if (ci > 0 && cj > 0 && !blocked(from, to, triangles, nodes, indices, i, j))
                value = (ci * cj) / (PI * sq);
        }
    }
    float sum = 0;
    #pragma unroll
    for (int k = 0; k < NUM_RAYS; ++k)
        sum += __shfl_sync(mask, value, k, NUM_RAYS);
    if (lane == 0) {
        float kij = sum * INV_NUM_RAYS;
        weights[local] = fminf(kij * areas[j], ONE);
        delays[local] = static_cast<unsigned char>(ceilf((centers[i] - centers[j]).norm() * INV_WAVE_SPEED));
        if (nonZero && kij > EPSILON) atomicAdd(nonZero, 1ULL);
    }
}

// At radius ten the longest delay is forty steps. Allow one extra for roundoff.
constexpr int HISTORY = 42;

__global__ void propagate(const float* weights, const unsigned char* delays, float* history,
                          int rows, int first, int n, int t, int source, float emission, float rho) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int i = thread / warpSize, lane = thread % warpSize;
    if (i >= rows) return;
    float sum = 0;
    for (int base = 0; base < n; base += warpSize) {
        int j = base + lane;
        float weight = 0, rad = 0;
        if (j < n) {
            size_t p = static_cast<size_t>(i) * n + j;
            int lag = delays[p];
            if (lag && lag <= t && weights[p] > ZERO) {
                weight = weights[p];
                rad = fmaxf(ZERO, history[static_cast<size_t>((t - lag) % HISTORY) * n + j]);
            }
        }
        // All lanes load contiguous coefficients. Fold their contributions in
        // emitter order, including across warp-sized tiles.
        if (__any_sync(0xffffffffU, rad > ZERO)) {
            #pragma unroll
            for (int k = 0; k < 32; ++k)
                sum += __shfl_sync(0xffffffffU, weight, k) * __shfl_sync(0xffffffffU, rad, k);
        }
    }
    if (lane == 0)
        history[static_cast<size_t>(t % HISTORY) * n + first + i] =
            rho * sum + (first + i == source ? emission : ZERO);
}

struct SimulationState {
    int rank, ranks, first, rows;
    size_t numTriangles, numTimesteps, sourceIndex;
    float reflectivity;
    std::vector<int> counts, offsets;
    std::vector<Triangle> triangles;
    std::vector<float> distances, localHistory, sourceHistory;
    DeviceArray<float> weights;
    DeviceArray<unsigned char> delays;
    unsigned long long nonZeroKij = 0;
    int receivedEnergy = 0;
};

void initializeSimulation(SimulationState& s, int subdivisions, int steps, int source, float rho) {
    s.triangles = std::move(IcosphereMesh(subdivisions, 10.0f).triangles);
    s.numTriangles = s.triangles.size();
    s.numTimesteps = steps;
    s.sourceIndex = static_cast<size_t>(source) % s.numTriangles;
    s.reflectivity = rho;
    int n = s.numTriangles;
    s.counts.resize(s.ranks);
    s.offsets.resize(s.ranks);
    for (int r = 0; r < s.ranks; ++r) {
        s.counts[r] = n / s.ranks + (r < n % s.ranks);
        s.offsets[r] = r * (n / s.ranks) + std::min(r, n % s.ranks);
    }
    s.rows = s.counts[s.rank];
    s.first = s.offsets[s.rank];
    s.weights.allocate(size_t(s.rows) * n);
    s.delays.allocate(size_t(s.rows) * n);
    s.localHistory.resize(size_t(s.rows) * steps);
    s.sourceHistory.resize(steps);
    if (!s.rank) s.distances.resize(n);
}

// Jump to a receiver partition in logarithmic time. Each bit of the tempered
// MT19937 output obeys the same degree-19937 recurrence over GF(2). Reducing
// x^offset modulo its characteristic polynomial expresses the shifted stream
// as an XOR of the first 19937 shifts of the original stream.
std::mt19937 samplingEngine(uint64_t offset) {
    std::mt19937 engine(42);
    if (offset < (1ULL << 24)) {
        engine.discard(offset);
        return engine;
    }
    constexpr int degree = 19937, words = (degree + 63) / 64;
    // Non-leading exponents of the MT19937 characteristic polynomial.
    constexpr int taps[] = {
        0,1189,1416,1585,1643,1870,2493,2773,3000,3227,3454,3681,3908,4135,
        4362,4753,5661,6337,6569,7129,7477,7525,7583,7752,7979,8206,9505,9901,
        9969,10128,10693,10761,10920,11089,11147,11157,11215,11321,11374,11384,
        11485,11611,11712,11717,11838,11881,11944,11997,12277,12335,12393,12504,
        12509,12620,12673,12731,12736,12789,12905,12958,12963,13137,13185,13190,
        13243,13301,13412,13528,13533,13639,13697,13760,13813,13866,14093,14151,
        14209,14320,14325,14436,14547,14552,14605,14721,14774,14779,14953,15001,
        15006,15059,15117,15228,15344,15349,15455,15513,15576,15629,15682,15909,
        15967,16025,16136,16141,16252,16363,16368,16421,16537,16590,16595,16817,
        16822,16875,16933,17044,17160,17271,17329,17445,17498,17725,17783,17841,
        17952,18068,18179,18237,18406,18633,18691,18860,19087,19314
    };
    std::array<uint64_t, words> poly{1};
    for (int bit = 63 - __builtin_clzll(offset); bit >= 0; --bit) {
        std::array<uint64_t, 2 * words + 1> square{};
        for (int w = 0; w < words; ++w) {
            for (uint64_t bits = poly[w]; bits; bits &= bits - 1) {
                int power = 2 * (64 * w + __builtin_ctzll(bits)) + ((offset >> bit) & 1);
                square[power / 64] |= uint64_t(1) << (power % 64);
            }
        }
        // Eliminate 64 high coefficients together. The next highest tap lies
        // over 64 places below the leading term, so the group is independent.
        for (int shift = degree / 64 * 64; shift >= 0; shift -= 64) {
            int p = degree + shift;
            uint64_t high = (square[p / 64] >> (p % 64)) |
                            (square[p / 64 + 1] << (64 - p % 64));
            if (!high) continue;
            for (int tap : taps) {
                int dest = (tap + shift) / 64, k = tap % 64;
                square[dest] ^= high << k;
                if (k) square[dest + 1] ^= high >> (64 - k);
            }
        }
        std::copy_n(square.begin(), words, poly.begin());
        poly.back() &= (uint64_t(1) << (degree % 64)) - 1;
    }

    constexpr size_t stateSize = std::mt19937::state_size;
    std::array<uint32_t, degree + stateSize> sequence;
    for (auto& word : sequence) word = engine();
    std::array<uint32_t, stateSize> state{};
    for (int w = 0; w < words; ++w) {
        for (uint64_t bits = poly[w]; bits; bits &= bits - 1) {
            int shift = 64 * w + __builtin_ctzll(bits);
            for (size_t k = 0; k < stateSize; ++k) state[k] ^= sequence[shift + k];
        }
    }
    // Undo the four tempering operations to load the standard engine state.
    std::stringstream serialized;
    for (uint32_t x : state) {
        x ^= x >> 18;
        x ^= (x << 15) & 0xefc60000U;
        uint32_t y = x;
        for (int k = 0; k < 4; ++k) x = y ^ ((x << 7) & 0x9d2c5680U);
        x ^= (x >> 11) ^ (x >> 22);
        serialized << x << ' ';
    }
    serialized << 0;
    serialized >> engine;
    return engine;
}

void computeFormFactors(SimulationState& s, bool validate) {
    int n = s.numTriangles;
    std::vector<Vec3> centers(n);
    std::vector<float> areas(n);
    int geometryThreads = std::min(omp_get_max_threads(), 1 + n / 256);
    #pragma omp parallel for schedule(static) num_threads(geometryThreads)
    for (int i = 0; i < n; ++i) {
        centers[i] = s.triangles[i].center();
        areas[i] = s.triangles[i].area();
    }
    auto sampled = [&](int i, int j) {
        return i != j && s.triangles[i]._normal.dot(s.triangles[j]._normal) <= 0.99f;
    };
    unsigned long long active = 0, preceding = 0;
    int countThreads = std::min<long long>(omp_get_max_threads(), 1 + int64_t(s.rows) * n / 8192);
    #pragma omp parallel for reduction(+:active) schedule(static) num_threads(countThreads)
    for (int i = s.first; i < s.first + s.rows; ++i)
        for (int j = 0; j < n; ++j)
            active += sampled(i, j);
    MPI_Exscan(&active, &preceding, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (!s.rank) preceding = 0;
    DeviceArray<unsigned long long> count;
    if (validate) count.upload({0});
    if (s.rows) {
        auto rng = samplingEngine(preceding * (4 * NUM_RAYS));
        std::uniform_real_distribution<float> uniform(0, 1);
        std::vector<FlatNode> nodes;
        std::vector<int> indices;
        Octree tree;
        tree.build(s.triangles);
        flatten(tree, nodes, indices);
        DeviceArray<Triangle> geometry;
        DeviceArray<Vec3> center;
        DeviceArray<float> area;
        DeviceArray<FlatNode> boxes;
        DeviceArray<int> leaves;
        geometry.upload(s.triangles);
        center.upload(centers);
        area.upload(areas);
        boxes.upload(nodes);
        leaves.upload(indices);

        constexpr int batch = 4096;
        size_t total = size_t(s.rows) * n;
        size_t capacity = std::min(total, size_t(batch)) * NUM_RAYS;
        float4* host[2];
        DeviceArray<float4> device[2];
        cudaStream_t stream[2];
        for (int k = 0; k < 2; ++k) {
            checkCuda(cudaMallocHost(&host[k], capacity * sizeof(float4)));
            device[k].allocate(capacity);
            checkCuda(cudaStreamCreate(&stream[k]));
        }
        for (size_t start = 0; start < total; start += batch) {
            int k = (start / batch) % 2;
            int pairs = std::min(size_t(batch), total - start);
            checkCuda(cudaStreamSynchronize(stream[k]));
            for (int p = 0; p < pairs; ++p) {
                size_t id = start + p;
                bool draw = sampled(s.first + id / n, id % n);
                for (int ray = 0; ray < NUM_RAYS; ++ray) {
                    float4& v = host[k][p * NUM_RAYS + ray];
                    if (draw) v = {uniform(rng), uniform(rng), uniform(rng), uniform(rng)};
                    else v = {-1, 0, 0, 0};
                }
            }
            int threads = pairs * NUM_RAYS;
            checkCuda(cudaMemcpyAsync(device[k].data, host[k], threads * sizeof(float4),
                                      cudaMemcpyHostToDevice, stream[k]));
            formFactors<<<(threads + 127) / 128, 128, 0, stream[k]>>>(
                geometry.data, center.data, area.data, boxes.data, leaves.data, device[k].data,
                start, pairs, s.first, n, s.weights.data, s.delays.data, count.data);
            checkCuda(cudaGetLastError());
        }
        for (int k = 0; k < 2; ++k) {
            checkCuda(cudaStreamSynchronize(stream[k]));
            checkCuda(cudaStreamDestroy(stream[k]));
            checkCuda(cudaFreeHost(host[k]));
        }
    }
    if (validate) {
        unsigned long long local;
        checkCuda(cudaMemcpy(&local, count.data, sizeof(local), cudaMemcpyDeviceToHost));
        MPI_Reduce(&local, &s.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
}

void runSimulation(SimulationState& s) {
    int n = s.numTriangles;
    DeviceArray<float> history;
    history.allocate(std::min(s.numTimesteps, size_t(HISTORY)) * n);
    float* slice;
    checkCuda(cudaMallocHost(&slice, n * sizeof(float)));
    for (int t = 0; t < int(s.numTimesteps); ++t) {
        float* current = history.data + size_t(t % HISTORY) * n;
        if (s.rows) {
            propagate<<<(s.rows + 3) / 4, 128>>>(s.weights.data, s.delays.data, history.data,
                s.rows, s.first, n, t, s.sourceIndex, t < int(s.numTimesteps / 2), s.reflectivity);
            checkCuda(cudaGetLastError());
            checkCuda(cudaMemcpy(slice + s.first, current + s.first, s.rows * sizeof(float),
                                 cudaMemcpyDeviceToHost));
            std::copy_n(slice + s.first, s.rows, s.localHistory.data() + size_t(t) * s.rows);
        }
        if (s.ranks > 1) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, slice, s.counts.data(),
                           s.offsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
            checkCuda(cudaMemcpy(current, slice, n * sizeof(float), cudaMemcpyHostToDevice));
        }
        s.sourceHistory[t] = slice[s.sourceIndex];
    }
    checkCuda(cudaFreeHost(slice));
}

void computeDistances(SimulationState& s, bool validate) {
    std::vector<float> local(s.rows);
    int received = 0;
    int maxThreads = std::min(omp_get_max_threads(), std::max(1, s.rows));
    double products = double(s.rows) * s.numTimesteps * s.numTimesteps / 2;
    int threads = static_cast<int>(std::min(double(maxThreads), 1 + products / 32768));
    #pragma omp parallel reduction(+:received) num_threads(threads)
    {
        std::vector<float> signal(s.numTimesteps);
        #pragma omp for schedule(static)
        for (int i = 0; i < s.rows; ++i) {
            for (size_t t = 0; t < signal.size(); ++t)
                signal[t] = s.localHistory[t * s.rows + i];
            if (validate)
                received += std::any_of(signal.begin(), signal.end(), [](float x){ return x > EPSILON; });
            float maxCorr = ZERO;
            size_t best = 0;
            for (size_t lag = 0; lag < signal.size(); ++lag) {
                float sum = ZERO;
                for (size_t t = lag; t < signal.size(); ++t)
                    sum += s.sourceHistory[t - lag] * signal[t];
                if (sum > maxCorr) {
                    maxCorr = sum;
                    best = lag;
                }
            }
            local[i] = WAVE_SPEED * static_cast<float>(best);
        }
    }
    MPI_Gatherv(local.data(), s.rows, MPI_FLOAT, s.distances.data(), s.counts.data(),
                s.offsets.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (validate)
        MPI_Reduce(&received, &s.receivedEnergy, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
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
    int support;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &support);
    if (support < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    SimulationState state;
    MPI_Comm_rank(MPI_COMM_WORLD, &state.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &state.ranks);
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
            if (!state.rank) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (!state.rank) printf("Unknown option: %s\n", argv[i]);
            if (!state.rank) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (timesteps < 0) {
        if (!state.rank) fprintf(stderr, "Timesteps must be non-negative\n");
        MPI_Finalize();
        return 1;
    }
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, state.rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    checkCuda(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d has no visible CUDA device\n", state.rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % devices));
    initializeSimulation(state, subdivisions, timesteps, sourceIdx, reflectivity);

    if (!state.rank) {
        printf("Room Response Simulation Benchmark\n");
        printf("MPI ranks: %d\n", state.ranks);
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\nSource triangle: %d\nReflectivity: %.2f\n",
               timesteps, sourceIdx, reflectivity);
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
    double localSeconds = end - start;
    double maxSeconds = 0.0;
    MPI_Reduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int status = 0;
    if (!state.rank) {
        double totalTime = maxSeconds * 1000;
        printf("Precomputation time: %.2f ms\n", (pre - start) * 1000);
        printf("Simulation time: %.2f ms\n", (sim - pre) * 1000);
        printf("Distance computation time: %.2f ms\n", (end - sim) * 1000);
        printf("Total computation time: %.2f ms\n", totalTime);

        // Performance metrics
        size_t n = state.numTriangles;
        size_t t = state.numTimesteps;
        double kijOps = static_cast<double>(n * n);
        double simOps = kijOps * t;
        double distOps = double(n) * t * t;

        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        // Aggregate storage of the coefficient matrices and radiosity histories.
        double matrixBytes = kijOps * (sizeof(float) + sizeof(unsigned char));
        double historyBytes = sizeof(float) * (double(n) * t + double(state.ranks) *
            (t + n * std::min(t, size_t(HISTORY))));
        printf("  Memory usage: %.2f MB\n", (matrixBytes + historyBytes) / (1024 * 1024));

        // Hash
        uint64_t hash = computeHash(state);
        printf("  Result hash: %016" PRIX64 "\n", hash);
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
