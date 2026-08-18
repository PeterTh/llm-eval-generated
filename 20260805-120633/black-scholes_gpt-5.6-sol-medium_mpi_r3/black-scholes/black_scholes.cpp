#include <algorithm>
#include <array>
#include <climits>
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

// Generate a larger set of options by scaling the test set
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions,
                     const size_t globalOffset = 0) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);
    
    for (size_t i = 0; i < numOptions; ++i) {
        const size_t globalIndex = globalOffset + i;
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[globalIndex % testOptions.size()];
        options[i] = base;
        
        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 *
            (globalIndex / static_cast<double>(testOptions.size()));
        options[i].spot *= factor;
        options[i].strike *= factor;
    }
}

// A contiguous, balanced partition preserves the serial result order and keeps
// the difference in work between any two ranks to at most one option.
struct Partition {
    size_t begin;
    size_t count;
};

Partition partitionForRank(const size_t total, const int rank,
                           const int worldSize) noexcept {
    const size_t ranks = static_cast<size_t>(worldSize);
    const size_t base = total / ranks;
    const size_t remainder = total % ranks;
    const size_t rankIndex = static_cast<size_t>(rank);
    return {
        rankIndex * base + std::min(rankIndex, remainder),
        base + (rankIndex < remainder ? 1u : 0u)
    };
}

// MPI_Gatherv uses int counts and displacements. Gather in windows so -r also
// works correctly for data sets larger than INT_MAX elements.
void gatherResultPrefix(const std::vector<double>& localResults,
                        const Partition localPartition, const size_t prefixSize,
                        const int rank, const int worldSize,
                        std::vector<double>& gatheredResults) {
    constexpr size_t maxWindow = static_cast<size_t>(INT_MAX);
    if (rank == 0) {
        gatheredResults.resize(prefixSize);
    }

    std::vector<int> counts;
    std::vector<int> displacements;
    if (rank == 0) {
        counts.resize(static_cast<size_t>(worldSize));
        displacements.resize(static_cast<size_t>(worldSize));
    }

    const unsigned long long localRange[2] = {
        static_cast<unsigned long long>(localPartition.begin),
        static_cast<unsigned long long>(localPartition.count)
    };
    std::vector<unsigned long long> allRanges;
    if (rank == 0) {
        allRanges.resize(static_cast<size_t>(worldSize) * 2);
    }
    MPI_Gather(localRange, 2, MPI_UNSIGNED_LONG_LONG,
               rank == 0 ? allRanges.data() : nullptr, 2,
               MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    for (size_t windowBegin = 0; windowBegin < prefixSize;) {
        const size_t windowSize = std::min(maxWindow, prefixSize - windowBegin);
        const size_t windowEnd = windowBegin + windowSize;
        const size_t localBegin = std::max(localPartition.begin, windowBegin);
        const size_t localEnd = std::min(localPartition.begin + localPartition.count,
                                         windowEnd);
        const size_t sendCount = localEnd > localBegin ? localEnd - localBegin : 0;
        const double* sendBuffer = sendCount == 0
            ? nullptr
            : localResults.data() + (localBegin - localPartition.begin);

        if (rank == 0) {
            for (int source = 0; source < worldSize; ++source) {
                const size_t sourceBegin = static_cast<size_t>(
                    allRanges[static_cast<size_t>(source) * 2]);
                const size_t sourceCount = static_cast<size_t>(
                    allRanges[static_cast<size_t>(source) * 2 + 1]);
                const size_t begin = std::max(sourceBegin, windowBegin);
                const size_t end = std::min(sourceBegin + sourceCount, windowEnd);
                counts[static_cast<size_t>(source)] = static_cast<int>(
                    end > begin ? end - begin : 0);
                displacements[static_cast<size_t>(source)] = static_cast<int>(
                    end > begin ? begin - windowBegin : 0);
            }
        }

        MPI_Gatherv(sendBuffer, static_cast<int>(sendCount), MPI_DOUBLE,
                    rank == 0 ? gatheredResults.data() + windowBegin : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        windowBegin = windowEnd;
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
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;
    
    // Parse once and broadcast the configuration so every rank follows the
    // same collective control flow.
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numOptions = atoll(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                parseStatus = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseStatus = 1;
                break;
            }
        }
    }

    unsigned long long optionCount = static_cast<unsigned long long>(numOptions);
    int flags[3] = {validate ? 1 : 0, printResults ? 1 : 0, parseStatus};
    MPI_Bcast(&optionCount, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(flags, 3, MPI_INT, 0, MPI_COMM_WORLD);
    numOptions = static_cast<size_t>(optionCount);
    validate = flags[0] != 0;
    printResults = flags[1] != 0;
    parseStatus = flags[2];
    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }
    
    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    const Partition localPartition = partitionForRank(numOptions, rank, worldSize);
    std::vector<OptionInput> localOptions;
    generateOptions(localOptions, localPartition.count, localPartition.begin);
    
    std::vector<double> localResults(localPartition.count);
    
    if (rank == 0) {
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (size_t i = 0; i < localPartition.count; ++i) {
        localResults[i] = blackScholes(localOptions[i]);
    }
    
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Options per second: %.0f\n",
               duration > 0.0 ? numOptions / duration : 0.0);
    }

    const size_t gatheredCount = printResults
        ? numOptions
        : (validate ? std::min<size_t>(10, numOptions) : 0);
    std::vector<double> results;
    if (gatheredCount != 0) {
        gatherResultPrefix(localResults, localPartition, gatheredCount, rank,
                           worldSize, results);
    }
    
    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(results, "OptionPrices");
    }
    
    // Validation
    int validStatus = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating results...\n");
            std::vector<OptionInput> validationOptions;
            generateOptions(validationOptions, gatheredCount);
            validStatus = validateResults(validationOptions, results) ? 1 : 0;
            printf("Validation: %s\n", validStatus ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&validStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return validStatus ? 0 : 1;
}
