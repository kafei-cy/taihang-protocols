/****************************************************************************
 * @file      bench_alsz_ote.cpp
 * @brief     Computation and communication benchmark for ALSZ OT extension.
 * @details   Projects decimal million-scale costs from 1/2/4-thread calibrations.
 * @author    Yu Chen
 *****************************************************************************/

#include <taihang/common/config.hpp>
#include <taihang/crypto/prg.hpp>
#include <taihang/mpc/ot/alsz_ote.hpp>

#include <openssl/obj_mac.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <future>
#include <iomanip>
#include <iostream>
#include <latch>
#include <string>
#include <vector>

using namespace taihang;
using namespace taihang::mpc::alsz_ote;

namespace {

using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;

constexpr size_t kBaseOtLen = 128;
constexpr size_t kCalibrationOtCount = 1ULL << 20;
constexpr int kCurveNid = NID_X9_62_prime256v1;
constexpr char kAddress[] = "127.0.0.1";
constexpr uint16_t kPort = 12347;
constexpr double kBytesPerMebibyte = 1024.0 * 1024.0;
constexpr double kBytesPerKibibyte = 1024.0;
constexpr std::array<int, 3> kThreadCounts{{1, 2, 4}};

struct ProjectionCase {
    const char* scale;
    size_t ot_count;
};

struct CommunicationCost {
    uint64_t base_ot_bytes;
    uint64_t extension_bytes;
    uint64_t sender_to_receiver_bytes;
    uint64_t receiver_to_sender_bytes;

    uint64_t total_bytes() const {
        return sender_to_receiver_bytes + receiver_to_sender_bytes;
    }
};

struct BenchmarkResult {
    int threads_per_party;
    size_t ot_count;
    double sender_ms;
    double receiver_ms;
    double wall_ms;
    double cpu_ms;
    CommunicationCost communication;
    bool verified;
};

struct ProjectionResult {
    int threads_per_party;
    ProjectionCase projection_case;
    double sender_ms;
    double receiver_ms;
    double wall_ms;
    double cpu_ms;
    CommunicationCost communication;
};

CommunicationCost communication_cost(size_t ot_count,
                                     size_t base_ot_count,
                                     size_t point_bytes) {
    const uint64_t n = ot_count;
    const uint64_t k = base_ot_count;
    const uint64_t block_bytes = sizeof(Block);

    // The ALSZ sender acts as the receiver in the Naor-Pinkas base OTs.
    const uint64_t base_sender_to_receiver = point_bytes * k;
    const uint64_t base_receiver_to_sender =
        point_bytes * (k + 1) + 2 * k * block_bytes;

    const uint64_t ciphertexts = 2 * n * block_bytes;
    const uint64_t adjustment_matrix = n * k / 8;

    return {
        base_sender_to_receiver + base_receiver_to_sender,
        ciphertexts + adjustment_matrix,
        base_sender_to_receiver + ciphertexts,
        base_receiver_to_sender + adjustment_matrix,
    };
}

BenchmarkResult run_calibration(const PublicParameters& pp,
                                int threads_per_party) {
    prg::Seed seed = prg::set_seed(nullptr, 0);
    std::vector<Block> messages_0 =
        prg::gen_random_blocks(seed, kCalibrationOtCount);
    std::vector<Block> messages_1 =
        prg::gen_random_blocks(seed, kCalibrationOtCount);
    std::vector<uint8_t> choices =
        prg::gen_random_bits(seed, kCalibrationOtCount);

    std::vector<Block> received;
    std::latch parties_ready(2);
    std::latch start_protocol(1);

    auto sender_future = std::async(std::launch::async, [&]() {
        net::NetIO io("server", kAddress, kPort);
        parties_ready.count_down();
        start_protocol.wait();

        const auto begin = Clock::now();
        sender<BlockPolicy>(io, pp, messages_0, messages_1,
                            kCalibrationOtCount);
        return Milliseconds(Clock::now() - begin).count();
    });

    auto receiver_future = std::async(std::launch::async, [&]() {
        net::NetIO io("client", kAddress, kPort);
        parties_ready.count_down();
        start_protocol.wait();

        const auto begin = Clock::now();
        received = receiver<BlockPolicy>(io, pp, choices,
                                         kCalibrationOtCount);
        return Milliseconds(Clock::now() - begin).count();
    });

    parties_ready.wait();
    const std::clock_t cpu_begin = std::clock();
    const auto wall_begin = Clock::now();
    start_protocol.count_down();

    const double sender_ms = sender_future.get();
    const double receiver_ms = receiver_future.get();
    const double wall_ms = Milliseconds(Clock::now() - wall_begin).count();
    const double cpu_ms = 1000.0 * static_cast<double>(std::clock() - cpu_begin)
                          / static_cast<double>(CLOCKS_PER_SEC);

    bool verified = received.size() == kCalibrationOtCount;
    for (size_t i = 0; verified && i < kCalibrationOtCount; ++i) {
        const Block& expected = choices[i] == 0 ? messages_0[i] : messages_1[i];
        verified = received[i] == expected;
    }

    const size_t point_bytes = pp.base_ot_pp.group_ctx->get_point_byte_len();
    return {
        threads_per_party,
        kCalibrationOtCount,
        sender_ms,
        receiver_ms,
        wall_ms,
        cpu_ms,
        communication_cost(kCalibrationOtCount, pp.base_len, point_bytes),
        verified,
    };
}

ProjectionResult project(const BenchmarkResult& calibration,
                         const ProjectionCase& projection_case,
                         const PublicParameters& pp) {
    const double ratio = static_cast<double>(projection_case.ot_count)
                         / static_cast<double>(calibration.ot_count);
    const size_t point_bytes = pp.base_ot_pp.group_ctx->get_point_byte_len();

    return {
        calibration.threads_per_party,
        projection_case,
        calibration.sender_ms * ratio,
        calibration.receiver_ms * ratio,
        calibration.wall_ms * ratio,
        calibration.cpu_ms * ratio,
        communication_cost(projection_case.ot_count, pp.base_len, point_bytes),
    };
}

double seconds(double milliseconds) {
    return milliseconds / 1000.0;
}

double mebibytes(uint64_t bytes) {
    return static_cast<double>(bytes) / kBytesPerMebibyte;
}

double million_ots_per_second(size_t ot_count, double wall_ms) {
    return static_cast<double>(ot_count) / wall_ms / 1000.0;
}

double cpu_nanoseconds_per_ot(size_t ot_count, double cpu_ms) {
    return cpu_ms * 1'000'000.0 / static_cast<double>(ot_count);
}

void print_configuration(const PublicParameters& pp) {
    std::cout
        << "\n===============================================================================================\n"
        << "Taihang ALSZ OT Extension Benchmark\n"
        << "===============================================================================================\n"
        << "Mode                  : single-machine loopback\n"
        << "Protocol              : semi-honest ALSZ, 128-bit Block messages\n"
        << "Base OTs              : " << pp.base_len << " Naor-Pinkas OTs on NIST P-256\n"
        << "EC point encoding     : "
        << pp.base_ot_pp.group_ctx->get_point_byte_len() << " bytes (compressed)\n"
        << "OpenMP threads/party  : 1, 2, and 4 (measured independently)\n"
        << "Calibration batch     : " << kCalibrationOtCount << " OTs (2^20)\n"
        << "Timing scope          : protocol only; input generation, connection setup, and verification excluded\n"
        << "Communication scope   : application payload bytes; TCP/IP framing excluded\n"
        << "===============================================================================================\n";
}

void print_calibration(const std::vector<BenchmarkResult>& calibrations) {
    std::cout
        << "\nMeasured calibration by threads per party\n"
        << "------------------------------------------------------------------------------------------------\n"
        << std::left << std::setw(13) << "Threads/party"
        << std::right << std::setw(14) << "OT count"
        << std::setw(15) << "Sender (s)"
        << std::setw(15) << "Receiver (s)"
        << std::setw(13) << "Wall (s)"
        << std::setw(13) << "CPU (s)"
        << std::setw(13) << "M OT/s"
        << std::setw(12) << "CPU ns/OT"
        << std::setw(12) << "Status" << "\n"
        << "------------------------------------------------------------------------------------------------\n";

    for (const auto& calibration : calibrations) {
        std::cout
            << std::left << std::setw(13) << calibration.threads_per_party
            << std::right << std::setw(14) << calibration.ot_count
            << std::fixed << std::setprecision(3)
            << std::setw(15) << seconds(calibration.sender_ms)
            << std::setw(15) << seconds(calibration.receiver_ms)
            << std::setw(13) << seconds(calibration.wall_ms)
            << std::setw(13) << seconds(calibration.cpu_ms)
            << std::setw(13)
            << million_ots_per_second(calibration.ot_count, calibration.wall_ms)
            << std::setw(12) << std::setprecision(1)
            << cpu_nanoseconds_per_ot(calibration.ot_count, calibration.cpu_ms)
            << std::setw(12) << (calibration.verified ? "PASS" : "FAIL")
            << "\n";
    }
    std::cout << "------------------------------------------------------------------------------------------------\n";
}

void print_computation_report(const std::vector<ProjectionResult>& projections) {
    std::cout
        << "\nInferred computation and end-to-end performance\n"
        << "------------------------------------------------------------------------------------------------------------------\n"
        << std::left << std::setw(10) << "Scale"
        << std::setw(13) << "Threads/party"
        << std::right << std::setw(14) << "Target OTs"
        << std::setw(15) << "Sender est.(s)"
        << std::setw(17) << "Receiver est.(s)"
        << std::setw(13) << "Wall est.(s)"
        << std::setw(12) << "CPU est.(s)"
        << std::setw(13) << "M OT/s"
        << std::setw(14) << "CPU ns/OT" << "\n"
        << "------------------------------------------------------------------------------------------------------------------\n";

    for (const auto& projection : projections) {
        std::cout
            << std::left << std::setw(10) << projection.projection_case.scale
            << std::setw(13) << projection.threads_per_party
            << std::right << std::setw(14) << projection.projection_case.ot_count
            << std::fixed << std::setprecision(3)
            << std::setw(15) << seconds(projection.sender_ms)
            << std::setw(17) << seconds(projection.receiver_ms)
            << std::setw(13) << seconds(projection.wall_ms)
            << std::setw(12) << seconds(projection.cpu_ms)
            << std::setw(13)
            << million_ots_per_second(projection.projection_case.ot_count,
                                      projection.wall_ms)
            << std::setw(14) << std::setprecision(1)
            << cpu_nanoseconds_per_ot(projection.projection_case.ot_count,
                                      projection.cpu_ms)
            << "\n";
    }
    std::cout << "------------------------------------------------------------------------------------------------------------------\n"
              << "Estimates scale the measured 2^20 amortized timings linearly with OT count.\n"
              << "CPU time is aggregate across both parties and all OpenMP workers.\n"
              << "The projection does not model cache, memory-bandwidth, or network saturation effects.\n";
}

void print_communication_report(const std::vector<ProjectionResult>& projections) {
    std::cout
        << "\nCommunication payload\n"
        << "---------------------------------------------------------------------------------------------------------------------------------------------\n"
        << std::left << std::setw(10) << "Scale"
        << std::setw(13) << "Threads/party"
        << std::right << std::setw(14) << "Target OTs"
        << std::setw(14) << "Base OT KiB"
        << std::setw(16) << "Extension MiB"
        << std::setw(14) << "S -> R MiB"
        << std::setw(14) << "R -> S MiB"
        << std::setw(16) << "Total bytes"
        << std::setw(13) << "Total MiB"
        << std::setw(12) << "Bytes/OT" << "\n"
        << "---------------------------------------------------------------------------------------------------------------------------------------------\n";

    for (const auto& projection : projections) {
        const auto& cost = projection.communication;
        std::cout
            << std::left << std::setw(10) << projection.projection_case.scale
            << std::setw(13) << projection.threads_per_party
            << std::right << std::setw(14) << projection.projection_case.ot_count
            << std::fixed << std::setprecision(2)
            << std::setw(14) << static_cast<double>(cost.base_ot_bytes) / kBytesPerKibibyte
            << std::setw(16) << mebibytes(cost.extension_bytes)
            << std::setw(14) << mebibytes(cost.sender_to_receiver_bytes)
            << std::setw(14) << mebibytes(cost.receiver_to_sender_bytes)
            << std::setw(16) << cost.total_bytes()
            << std::setw(13) << mebibytes(cost.total_bytes())
            << std::setw(12) << static_cast<double>(cost.total_bytes())
                                  / projection.projection_case.ot_count
            << "\n";
    }
    std::cout << "---------------------------------------------------------------------------------------------------------------------------------------------\n"
              << "S -> R and R -> S refer to the ALSZ sender and receiver roles.\n"
              << "Payload is derived from the protocol and retains one fixed base-OT cost per target batch.\n"
              << "For 128-bit messages, the extension payload approaches 48 bytes per OT.\n";
}

} // namespace

int main() {
    constexpr std::array<ProjectionCase, 2> projection_cases{{
        {"10^6", 1'000'000},
        {"10^7", 10'000'000},
    }};

    config::use_point_compression = true;
    const PublicParameters pp = setup(kCurveNid, kBaseOtLen);
    print_configuration(pp);

    std::vector<BenchmarkResult> calibrations;
    calibrations.reserve(kThreadCounts.size());

    std::vector<ProjectionResult> projections;
    projections.reserve(kThreadCounts.size() * projection_cases.size());
    for (const int threads_per_party : kThreadCounts) {
        config::thread_num = threads_per_party;
        std::cout << "\nRunning calibration with " << threads_per_party
                  << " OpenMP thread(s) per party ...\n";
        calibrations.push_back(run_calibration(pp, threads_per_party));
        const auto& calibration = calibrations.back();
        for (const auto& projection_case : projection_cases) {
            projections.push_back(project(calibration, projection_case, pp));
        }
    }

    print_calibration(calibrations);
    print_computation_report(projections);
    print_communication_report(projections);

    for (const auto& calibration : calibrations) {
        if (!calibration.verified) {
            return 1;
        }
    }
    return 0;
}
