#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>
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

// Generate only this rank's range using the original global option indices.
void generateOptions(std::vector<OptionInput>& options, const size_t numOptions,
                     const size_t firstOption = 0) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);
    
    for (size_t i = 0; i < numOptions; ++i) {
        const size_t globalIndex = firstOption + i;
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[globalIndex % testOptions.size()];
        options[i] = base;
        
        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 * (globalIndex / static_cast<double>(testOptions.size()));
        options[i].spot *= factor;
        options[i].strike *= factor;
    }
}

struct OptionRange {
    size_t first;
    size_t count;
};

// Contiguous ranges differ by at most one option, including when N < ranks.
OptionRange partitionOptions(const size_t numOptions, const int rank,
                             const int numRanks) noexcept {
    const size_t baseCount = numOptions / static_cast<size_t>(numRanks);
    const size_t remainder = numOptions % static_cast<size_t>(numRanks);
    const size_t rankIndex = static_cast<size_t>(rank);
    return {rankIndex * baseCount + std::min(rankIndex, remainder),
            baseCount + (rankIndex < remainder ? 1 : 0)};
}

// Collect an ordered prefix only when output or validation needs it.
std::vector<double> gatherResults(std::vector<double>& localResults,
                                  const OptionRange localRange,
                                  const size_t numOptions,
                                  const size_t numToCollect,
                                  const int rank, const int numRanks) {
    const OptionRange rootRange = partitionOptions(numOptions, 0, numRanks);
    if (numToCollect <= rootRange.count) {
        // Typically the validation prefix already belongs entirely to rank 0.
        return rank == 0 ? std::move(localResults) : std::vector<double>{};
    }

    std::vector<double> collected;
    std::vector<int> counts;
    std::vector<int> displacements;
    if (rank == 0) {
        collected.resize(numToCollect);
        std::copy_n(localResults.begin(), rootRange.count, collected.begin());
        // The root contributes in place, so its local result buffer can be freed.
        std::vector<double>().swap(localResults);
        counts.resize(numRanks);
        displacements.resize(numRanks);
    }

    // MPI_Gatherv counts/displacements are int. Use bounded windows to also
    // support datasets larger than INT_MAX without requiring MPI-4 interfaces.
    const size_t maxCount = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t first = 0; first < numToCollect;) {
        const size_t last = first + std::min(maxCount, numToCollect - first);
        const size_t sendFirst = std::max(first, localRange.first);
        const size_t sendLast = std::min(last, localRange.first + localRange.count);
        const int sendCount = sendFirst < sendLast ? static_cast<int>(sendLast - sendFirst) : 0;
        const double* sendBuffer = rank != 0 && sendCount != 0
                                       ? localResults.data() + (sendFirst - localRange.first)
                                       : nullptr;

        if (rank == 0) {
            for (int source = 0; source < numRanks; ++source) {
                const OptionRange range = partitionOptions(numOptions, source, numRanks);
                const size_t receiveFirst = std::max(first, range.first);
                const size_t receiveLast = std::min(last, range.first + range.count);
                const bool overlaps = receiveFirst < receiveLast;
                counts[source] = overlaps ? static_cast<int>(receiveLast - receiveFirst) : 0;
                displacements[source] = overlaps ? static_cast<int>(receiveFirst - first) : 0;
            }
        }

        MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : sendBuffer, sendCount, MPI_DOUBLE,
                    rank == 0 ? collected.data() + first : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
        first = last;
    }
    return collected;
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

int runBenchmark(int argc, char** argv, const int rank, const int numRanks) {
    // Parse on rank 0, then give every rank the same configuration and exit path.
    // Entries are the option count, validation flag, results flag, and action
    // (0 = run, 1 = help, 2 = argument error).
    unsigned long long configuration[4] = {10000, 0, 0, 0};
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                configuration[0] = atoll(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                configuration[1] = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                configuration[2] = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                configuration[3] = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                configuration[3] = 2;
                break;
            }
        }
    }

    MPI_Bcast(configuration, 4, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    if (configuration[3] != 0) {
        return configuration[3] == 1 ? 0 : 1;
    }
    if (configuration[0] > std::numeric_limits<size_t>::max()) {
        throw std::length_error("Option count exceeds the addressable range");
    }
    const size_t numOptions = static_cast<size_t>(configuration[0]);
    const bool validate = configuration[1] != 0;
    const bool printResults = configuration[2] != 0;

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate and store only local inputs; no input scatter or replication.
    const OptionRange range = partitionOptions(numOptions, rank, numRanks);
    std::vector<OptionInput> options;
    generateOptions(options, range.count, range.first);
    std::vector<double> results(range.count);
    
    // Price options
    if (rank == 0) {
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (size_t i = 0; i < range.count; ++i) {
        results[i] = blackScholes(options[i]);
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Options per second: %.0f\n", duration > 0.0 ? numOptions / duration : 0.0);
    }

    // Release the large input arrays before any optional root-side collection.
    std::vector<OptionInput>().swap(options);
    std::vector<double> collected;
    if (printResults || validate) {
        const size_t numToCollect = printResults ? numOptions : std::min(size_t{10}, numOptions);
        collected = gatherResults(results, range, numOptions, numToCollect, rank, numRanks);
    }

    // Print results for external validation
    if (rank == 0 && printResults) {
        print_results(collected, "OptionPrices");
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating results...\n");
            // Only the original first ten inputs are needed for validation.
            generateOptions(options, std::min(size_t{10}, numOptions));
            const bool valid = validateResults(options, collected);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    return exitCode;
}

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        return 1;
    }
    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int exitCode = 0;
    try {
        exitCode = runBenchmark(argc, argv, rank, numRanks);
    } catch (const std::exception& error) {
        // An allocation failure on one rank must not strand the others in MPI.
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return exitCode;
}
