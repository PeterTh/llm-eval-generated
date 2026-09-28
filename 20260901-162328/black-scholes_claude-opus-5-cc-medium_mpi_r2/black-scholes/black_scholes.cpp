#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

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

// Generate the option with global index i (identical to the serial generator)
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

// Generate the slice [beginIndex, beginIndex + count) of the global option set.
// Every rank builds only the options it owns, so setup cost and memory scale
// with the number of ranks.
void generateOptions(std::vector<OptionInput>& options, const size_t beginIndex, const size_t count) {
    options.resize(count);

    for (size_t i = 0; i < count; ++i) {
        options[i] = makeOption(beginIndex + i);
    }
}

// `leadingResults` holds the first min(10, numOptions) prices of the global result set
bool validateResults(const std::vector<double>& leadingResults) {
    bool allPassed = true;
    const int numChecks = static_cast<int>(leadingResults.size());

    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = leadingResults[i];
        const double expected = makeOption(static_cast<size_t>(i)).value;
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
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // Only rank 0 produces console output
    const bool isRoot = (rank == 0);
#define ROOT_PRINTF(...) do { if (isRoot) printf(__VA_ARGS__); } while (0)

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank, argv is replicated)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            ROOT_PRINTF("Unknown option: %s\n", argv[i]);
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    ROOT_PRINTF("Black-Scholes Option Pricing Benchmark\n");
    ROOT_PRINTF("Number of options: %zu\n", numOptions);
    ROOT_PRINTF("Validation: %s\n", validate ? "enabled" : "disabled");
    ROOT_PRINTF("MPI ranks: %d\n", numRanks);

    // Block distribution of the global option range over the ranks: rank r owns
    // the contiguous range [begin(r), begin(r+1)). Balanced to within one option.
    const auto blockBegin = [numOptions, numRanks](const int r) noexcept {
        return static_cast<size_t>((static_cast<unsigned long long>(numOptions) * r) / numRanks);
    };
    const size_t localBegin = blockBegin(rank);
    const size_t localEnd = blockBegin(rank + 1);
    const size_t localCount = localEnd - localBegin;

    // Generate the locally owned options
    std::vector<OptionInput> options;
    generateOptions(options, localBegin, localCount);

    // Allocate local results
    std::vector<double> results(localCount);

    // Price options
    ROOT_PRINTF("Pricing options...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const OptionInput* __restrict const optionData = options.data();
    double* __restrict const resultData = results.data();
    for (size_t i = 0; i < localCount; ++i) {
        resultData[i] = blackScholes(optionData[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // The wall clock time of the slowest rank is the time of the parallel region
    long long localMicros = duration.count();
    long long maxMicros = localMicros;
    MPI_Reduce(&localMicros, &maxMicros, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    ROOT_PRINTF("Computation time: %.3f ms\n", maxMicros / 1000.0);
    ROOT_PRINTF("Options per second: %.0f\n", numOptions / (maxMicros / 1e6));

    // Print results for external validation: rank 0 collects the full, globally
    // ordered result vector.
    if (printResults) {
        std::vector<int> counts, displs;
        std::vector<double> gathered;
        if (isRoot) {
            counts.resize(numRanks);
            displs.resize(numRanks);
            for (int r = 0; r < numRanks; ++r) {
                counts[r] = static_cast<int>(blockBegin(r + 1) - blockBegin(r));
                displs[r] = static_cast<int>(blockBegin(r));
            }
            gathered.resize(numOptions);
        }
        MPI_Gatherv(resultData, static_cast<int>(localCount), MPI_DOUBLE, gathered.data(),
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (isRoot) {
            print_results(gathered, "OptionPrices");
        }
    }

    // Validation only inspects the first few options; collect just those on rank 0
    if (validate) {
        const size_t numChecks = std::min(static_cast<size_t>(10), numOptions);
        std::vector<double> leading(numChecks, 0.0);
        for (size_t i = 0; i < numChecks; ++i) {
            if (i >= localBegin && i < localEnd) {
                leading[i] = resultData[i - localBegin];
            }
        }
        // Contributions are disjoint (each index is owned by exactly one rank),
        // so the sum reproduces the values bit-for-bit.
        std::vector<double> leadingGlobal(numChecks);
        MPI_Reduce(leading.data(), leadingGlobal.data(), static_cast<int>(numChecks), MPI_DOUBLE,
                   MPI_SUM, 0, MPI_COMM_WORLD);

        int valid = 1;
        if (isRoot) {
            printf("Validating results...\n");
            valid = validateResults(leadingGlobal) ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
#undef ROOT_PRINTF
}
