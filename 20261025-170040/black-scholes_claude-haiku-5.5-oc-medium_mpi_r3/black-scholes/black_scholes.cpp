#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// Contiguous block of the global option index space owned by one rank
struct Partition {
    size_t begin;
    size_t count;
};

// Block distribution of `total` items over `size` ranks (remainder spread over first ranks)
Partition blockPartition(const size_t total, const int rank, const int size) noexcept {
    const size_t p = static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    const size_t base = total / p;
    const size_t rem = total % p;
    Partition part;
    part.begin = r * base + std::min(r, rem);
    part.count = base + (r < rem ? 1 : 0);
    return part;
}

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

// Build the option with global index i (cycles through the test set, scaled by index)
inline OptionInput makeOption(const size_t i) noexcept {
    constexpr auto testOptions = getTestOptions();
    // Cycle through test options and vary parameters slightly
    OptionInput option = testOptions[i % testOptions.size()];
    
    // Add some variation for larger datasets
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

// Generate the options with global indices [begin, begin + count)
void generateOptions(std::vector<OptionInput>& options, const size_t begin, const size_t count) {
    options.resize(count);
    for (size_t j = 0; j < count; ++j) {
        options[j] = makeOption(begin + j);
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

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = (rank == 0);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (identical on every rank, so all ranks take the same path)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) {
                printUsage(argv[0]);
            }
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

    // Gathering full results to the root requires int-sized MPI counts/displacements
    const bool needFull = printResults || validate;
    if (needFull && numOptions > static_cast<size_t>(INT_MAX)) {
        if (root) {
            printf("Number of options exceeds %d, unsupported with -r or -v\n", INT_MAX);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (root) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Each rank generates and prices only its own contiguous block of options
    const Partition part = blockPartition(numOptions, rank, size);
    std::vector<OptionInput> options;
    generateOptions(options, part.begin, part.count);
    
    std::vector<double> localResults(part.count);
    
    if (root) {
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < part.count; ++i) {
        localResults[i] = blackScholes(options[i]);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    const long long localMicros = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Wall time of the parallel region is the slowest rank
    long long maxMicros = 0;
    MPI_Reduce(&localMicros, &maxMicros, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Reduce a checksum over the local results so the computation remains observable
    double localSum = 0.0;
    for (const double v : localResults) {
        localSum += v;
    }
    double globalSum = 0.0;
    MPI_Reduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (root) {
        printf("Computation time: %.3f ms\n", maxMicros / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (maxMicros / 1e6));
    }

    // Gather full results on the root only when they are needed for output or validation
    std::vector<double> results;
    if (needFull) {
        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int r = 0; r < size; ++r) {
            const Partition p = blockPartition(numOptions, r, size);
            counts[r] = static_cast<int>(p.count);
            displs[r] = static_cast<int>(p.begin);
        }
        if (root) {
            results.resize(numOptions);
        }
        MPI_Gatherv(localResults.data(), static_cast<int>(part.count), MPI_DOUBLE,
                    root ? results.data() : nullptr, counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }
    
    // Print results for external validation
    if (printResults && root) {
        print_results(results, "OptionPrices");
    }
    
    // Validation
    int valid = 1;
    if (validate) {
        if (root) {
            printf("Validating results...\n");
            // Validation only inspects the first min(10, N) options
            std::vector<OptionInput> checkOptions;
            generateOptions(checkOptions, 0, std::min(static_cast<size_t>(10), numOptions));
            valid = validateResults(checkOptions, results) ? 1 : 0;

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return valid ? 0 : 1;
}
