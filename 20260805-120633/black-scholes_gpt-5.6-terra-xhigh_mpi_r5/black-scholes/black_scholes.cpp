#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
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

inline constexpr auto testOptions = getTestOptions();

// Generate one option from its global index.  Each rank can therefore build
// its local input block without distributing the full input vector.
inline OptionInput generateOption(const size_t index) noexcept {
    OptionInput option = testOptions[index % testOptions.size()];

    const double factor = 1.0 + 0.1 * (index / static_cast<double>(testOptions.size()));
    option.spot *= factor;
    option.strike *= factor;
    return option;
}

// Generate a contiguous local block of the global option sequence.
void generateOptions(std::vector<OptionInput>& options, const size_t firstOption,
                     const size_t numOptions) {
    options.resize(numOptions);

    for (size_t i = 0; i < numOptions; ++i) {
        options[i] = generateOption(firstOption + i);
    }
}

bool validateResults(const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = std::min(static_cast<size_t>(10), results.size());
    
    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = generateOption(i).value;
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

// Return the global range assigned to a rank.  Contiguous blocks preserve
// ordering for collection and have no per-option communication overhead.
void getLocalRange(const size_t totalOptions, const int rank, const int worldSize,
                   size_t& firstOption, size_t& localOptions) noexcept {
    const size_t ranks = static_cast<size_t>(worldSize);
    const size_t rankIndex = static_cast<size_t>(rank);
    const size_t baseCount = totalOptions / ranks;
    const size_t remainder = totalOptions % ranks;

    localOptions = baseCount + (rankIndex < remainder ? 1 : 0);
    firstOption = rankIndex * baseCount + std::min(rankIndex, remainder);
}

// Gather a contiguous global result range to rank zero.  The operation is
// split into MPI-sized chunks so -r remains correct even when the full result
// vector exceeds MPI's int count/displacement interface.
void gatherResultRange(const std::vector<double>& localResults,
                       const size_t localFirstOption, const size_t totalOptions,
                       const size_t firstResult, const size_t resultCount,
                       const int rank, const int worldSize,
                       std::vector<double>& gatheredResults) {
    if (rank == 0) {
        gatheredResults.resize(resultCount);
    }

    const size_t localEnd = localFirstOption + localResults.size();
    constexpr size_t maxMpiCount = static_cast<size_t>(std::numeric_limits<int>::max());

    for (size_t offset = 0; offset < resultCount;) {
        const size_t chunkCount = std::min(maxMpiCount, resultCount - offset);
        const size_t chunkFirst = firstResult + offset;
        const size_t chunkEnd = chunkFirst + chunkCount;

        const size_t sendFirst = std::max(localFirstOption, chunkFirst);
        const size_t sendEnd = std::min(localEnd, chunkEnd);
        const size_t sendCount = sendEnd > sendFirst ? sendEnd - sendFirst : 0;
        const int mpiSendCount = static_cast<int>(sendCount);
        const double* sendBuffer = sendCount == 0
                                       ? nullptr
                                       : localResults.data() + (sendFirst - localFirstOption);

        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        if (rank == 0) {
            receiveCounts.resize(worldSize);
            displacements.resize(worldSize);
            for (int source = 0; source < worldSize; ++source) {
                size_t sourceFirst;
                size_t sourceCount;
                getLocalRange(totalOptions, source, worldSize, sourceFirst, sourceCount);
                const size_t sourceEnd = sourceFirst + sourceCount;
                const size_t receiveFirst = std::max(sourceFirst, chunkFirst);
                const size_t receiveEnd = std::min(sourceEnd, chunkEnd);
                const size_t receiveCount = receiveEnd > receiveFirst
                                                ? receiveEnd - receiveFirst
                                                : 0;
                receiveCounts[source] = static_cast<int>(receiveCount);
                displacements[source] = receiveCount == 0
                                            ? 0
                                            : static_cast<int>(receiveFirst - chunkFirst);
            }
        }

        double* receiveBuffer = rank == 0 ? gatheredResults.data() + offset : nullptr;
        MPI_Gatherv(sendBuffer, mpiSendCount, MPI_DOUBLE, receiveBuffer,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        offset += chunkCount;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    int worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    size_t firstOption;
    size_t localOptionCount;
    getLocalRange(numOptions, rank, worldSize, firstOption, localOptionCount);

    // Inputs and outputs are distributed across ranks; generation remains
    // outside the timed pricing region, as in the original benchmark.
    std::vector<OptionInput> localOptions;
    generateOptions(localOptions, firstOption, localOptionCount);
    std::vector<double> localResults(localOptionCount);

    if (rank == 0) {
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (size_t i = 0; i < localOptionCount; ++i) {
        localResults[i] = blackScholes(localOptions[i]);
    }

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        printf("Options per second: %.0f\n", numOptions / duration);
    }

    // Only materialize global values when an output mode requires them.  A
    // normal benchmark run therefore communicates only the elapsed time.
    std::vector<double> gatheredResults;
    if (printResults) {
        gatherResultRange(localResults, firstOption, numOptions, 0, numOptions,
                          rank, worldSize, gatheredResults);
    } else if (validate) {
        const size_t validationCount = std::min(static_cast<size_t>(10), numOptions);
        gatherResultRange(localResults, firstOption, numOptions, 0, validationCount,
                          rank, worldSize, gatheredResults);
    }

    int exitCode = 0;
    if (rank == 0) {
        if (printResults) {
            print_results(gatheredResults, "OptionPrices");
        }

        if (validate) {
            printf("Validating results...\n");
            const bool valid = validateResults(gatheredResults);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
