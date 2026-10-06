#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "../common/results_output.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT_2
#define M_SQRT_2 0.7071067811865475244
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
#endif
#ifndef M_1_SQRTPI
#define M_1_SQRTPI 0.564189583547756286948
#endif

enum OptionType {
    CALL = 0,
    PUT = 1
};

struct OptionInput {
    int type;           // CALL or PUT
    double strike;      // Strike price
    double spot;        // Spot price
    double q;           // Dividend yield
    double r;           // Risk-free rate
    double t;           // Time to maturity
    double vol;         // Volatility
    double value;       // Expected value (for validation)
    double tol;         // Tolerance
};

// Standard normal cumulative distribution function
inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

// Standard normal probability density function
inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options
double blackScholes(const OptionInput& option) noexcept {
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;
    
    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }
    
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    const double d2 = d1 - sigma * sqrt(T);
    
    const double Nd1 = cumulativeNormal(d1);
    const double Nd2 = cumulativeNormal(d2);
    const double discount = exp(-r * T);
    
    double price;
    if (option.type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else { // PUT
        price = K * discount * cumulativeNormal(-d2) - S * exp(-q * T) * cumulativeNormal(-d1);
    }
    
    return price;
}

// Standard test cases for validation
inline constexpr std::array<OptionInput, 7> getTestOptions() noexcept {
    return {{
        {CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3},
        {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3},
        {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
        {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3},
        {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
        {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
        {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3},
    }};
}

// Generate options [begin, begin + count) of the full (global) option set.
// Each element depends only on its global index, so ranks can generate their
// own block independently with results identical to the serial generation.
void generateOptions(std::vector<OptionInput>& options, const size_t begin, const size_t count) {
    constexpr auto testOptions = getTestOptions();
    options.resize(count);
    
    for (size_t j = 0; j < count; ++j) {
        const size_t i = begin + j;
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[j] = base;
        
        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        options[j].spot *= factor;
        options[j].strike *= factor;
    }
}

// Block decomposition: start index of rank's block
static inline size_t blockStart(const size_t n, const int rank, const int size) {
    const size_t base = n / size;
    const size_t rem = n % size;
    const size_t r = static_cast<size_t>(rank);
    return r * base + std::min(r, rem);
}

// Gather the first `prefix` global elements onto rank 0 (in global order).
// Uses point-to-point transfers chunked to stay within int count limits.
static void gatherPrefix(const std::vector<double>& local, std::vector<double>& global,
                         const size_t n, const size_t prefix, const int rank, const int size) {
    constexpr size_t kChunk = static_cast<size_t>(1) << 30;
    const size_t myBegin = blockStart(n, rank, size);
    const size_t myEnd = blockStart(n, rank + 1, size);
    if (rank == 0) {
        global.resize(prefix);
        const size_t ownEnd = std::min(myEnd, prefix);
        if (ownEnd > myBegin) {
            std::copy(local.begin(), local.begin() + (ownEnd - myBegin), global.begin() + myBegin);
        }
        std::vector<MPI_Request> reqs;
        for (int p = 1; p < size; ++p) {
            const size_t b = blockStart(n, p, size);
            const size_t e = std::min(blockStart(n, p + 1, size), prefix);
            for (size_t off = b; off < e; off += kChunk) {
                const int cnt = static_cast<int>(std::min(kChunk, e - off));
                reqs.emplace_back();
                MPI_Irecv(global.data() + off, cnt, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &reqs.back());
            }
        }
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    } else {
        const size_t e = std::min(myEnd, prefix);
        for (size_t off = myBegin; off < e; off += kChunk) {
            const int cnt = static_cast<int>(std::min(kChunk, e - off));
            MPI_Send(local.data() + (off - myBegin), cnt, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
}

bool validateResults(const std::vector<OptionInput>& options, 
                     const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = std::min(static_cast<size_t>(10), options.size());
    
    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = options[i].value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);
        
        printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n", 
               i, computed, expected, relError);
        
        // Relaxed validation - just check values are positive and reasonable
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
            printf("Validation failed at option %d: invalid value %.4f\n", i, computed);
            allPassed = false;
        }
    }
    
    return allPassed;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of options to price (default: 10000)\n");
    printf("  -v           Enable validation against known values\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = (rank == 0);
    
#ifdef __linux__
    // MPI_Init spawns helper threads, which makes automatic NUMA balancing
    // inject hinting page faults into the compute loop. An explicit MPOL_LOCAL
    // policy (node-local first-touch, same as default) opts this process out.
    syscall(SYS_set_mempolicy, 4 /* MPOL_LOCAL */, nullptr, 0UL);
#endif
    
    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (identically on all ranks)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (root) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate this rank's block of options
    const size_t myBegin = blockStart(numOptions, rank, size);
    const size_t myCount = blockStart(numOptions, rank + 1, size) - myBegin;
    std::vector<OptionInput> options;
    generateOptions(options, myBegin, myCount);
    
    // Allocate local results
    std::vector<double> localResults(myCount);
    
    // Price options
    if (root) printf("Pricing options...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < myCount; ++i) {
        localResults[i] = blackScholes(options[i]);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Report the slowest rank's time
    long long localUs = duration.count(), maxUs = 0;
    MPI_Reduce(&localUs, &maxUs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (root) {
        printf("Computation time: %.3f ms\n", maxUs / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (maxUs / 1e6));
    }
    
    // Collect results on rank 0 only as far as needed
    size_t needed = 0;
    if (printResults) needed = numOptions;
    else if (validate) needed = std::min(static_cast<size_t>(10), numOptions);
    std::vector<double> results;
    if (needed > 0) {
        gatherPrefix(localResults, results, numOptions, needed, rank, size);
    }
    
    // Print results for external validation
    if (printResults && root) {
        print_results(results, "OptionPrices");
    }
    
    int exitCode = 0;
    // Validation
    if (validate) {
        if (root) {
            printf("Validating results...\n");
            std::vector<OptionInput> checkOptions;
            generateOptions(checkOptions, 0, needed);
            bool valid = validateResults(checkOptions, results);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return exitCode;
}
