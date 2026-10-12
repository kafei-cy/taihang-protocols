/****************************************************************************
 * @file      bench_mqrpmt_pso.cpp
 * @brief     Benchmark suite for the unified mqRPMT-based PSO framework.
 * @author    This file is part of Taihang.
 *****************************************************************************/

#include <taihang/mpc/pso/mqrpmt_pso.hpp>
#include <taihang/common/bench_setting.hpp>
#include <taihang/common/config.hpp>
#include <taihang/crypto/prg.hpp>
#include <taihang/crypto/x25519_simd.hpp>
#include <openssl/obj_mac.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <future>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace taihang;
using namespace taihang::mpc;
namespace pso = taihang::mpc::mqrpmt_pso;

namespace {

using Clock = std::chrono::high_resolution_clock;
using Milliseconds = std::chrono::duration<double, std::milli>;

constexpr size_t kLogSenderLen = 20;
constexpr size_t kLogReceiverLen = 20;
constexpr size_t kLogSumBound = 48;
constexpr size_t kLogValueBound = 16;
constexpr size_t kStatisticalSecurityParameter = 40;
constexpr uint64_t kValueBound = uint64_t{1} << kLogValueBound;
constexpr int kSecp256r1 = 415;

struct Dataset {
    std::vector<Block> vec_x;
    std::vector<Block> vec_y;
    size_t intersection_size = 0;
    size_t union_size = 0;
    uint64_t intersection_sum = 0;
};

uint64_t value_at(size_t index) {
    return static_cast<uint64_t>(index % (kValueBound - 1)) + 1;
}

Dataset make_dataset(size_t sender_len, size_t receiver_len) {
    Dataset dataset;
    dataset.vec_x.resize(sender_len);
    dataset.vec_y.resize(receiver_len);
    dataset.intersection_size =
        std::min(sender_len / 2, receiver_len);
    dataset.union_size =
        sender_len + receiver_len - dataset.intersection_size;

    Block seed_block =
        make_block(0x123456789ABCDEF0ULL, 0x0FEDCBA987654321ULL);
    prg::Seed seed = prg::set_seed(&seed_block, 0);
    prg::gen_random_blocks(seed, dataset.vec_y.data(), receiver_len);

    for (size_t i = 0; i < sender_len; ++i) {
        if (i < dataset.intersection_size) {
            dataset.vec_x[i] = dataset.vec_y[i];
            dataset.intersection_sum += value_at(i);
        } else {
            // A separate domain keeps sender-only items disjoint from Y.
            dataset.vec_x[i] =
                make_block(0x5858585858585858ULL,
                           static_cast<uint64_t>(i + 1));
        }
    }
    return dataset;
}

struct BenchConfig {
    pso::PsoMode operation;
    int mqrpmt_curve_id;
    cwprf_mqrpmt::MembershipMode membership_mode;
    uint16_t port;
    std::string label;
};

const std::vector<BenchConfig> kConfigs = [] {
    struct Operation {
        pso::PsoMode mode;
        const char* name;
    };
    struct Curve {
        int id;
        const char* name;
    };
    struct Membership {
        cwprf_mqrpmt::MembershipMode mode;
        const char* name;
    };

    constexpr std::array<Operation, 4> operations = {{
        {pso::PsoMode::kIntersection, "PSI"},
        {pso::PsoMode::kUnion, "PSU"},
        {pso::PsoMode::kCard, "PSI-Card"},
        {pso::PsoMode::kCardSum, "PSI-Card-Sum"},
    }};
    constexpr std::array<Curve, 2> curves = {{
        {kSecp256r1, "Secp256r1"},
        {NID_X25519, "X25519"},
    }};
    constexpr std::array<Membership, 2> memberships = {{
        {cwprf_mqrpmt::MembershipMode::BloomFilter, "BloomFilter"},
        {cwprf_mqrpmt::MembershipMode::PlainSet, "PlainSet"},
    }};

    std::vector<BenchConfig> configs;
    uint16_t port = 12500;
    for (const auto& operation : operations) {
        for (const auto& curve : curves) {
            for (const auto& membership : memberships) {
                configs.push_back({
                    operation.mode,
                    curve.id,
                    membership.mode,
                    port++,
                    std::string(operation.name) + " / " + curve.name +
                        " / " + membership.name,
                });
            }
        }
    }
    return configs;
}();

pso::PublicParameters make_public_parameters(const BenchConfig& config) {
    const bool needs_ring = config.operation == pso::PsoMode::kCardSum;
    const std::optional<size_t> ssp =
        config.membership_mode == cwprf_mqrpmt::MembershipMode::BloomFilter
            ? std::optional<size_t>(kStatisticalSecurityParameter)
            : std::nullopt;

    return pso::setup(kSecp256r1,
                      config.mqrpmt_curve_id,
                      kLogSenderLen,
                      kLogReceiverLen,
                      needs_ring ? kLogSumBound : 0,
                      needs_ring ? kLogValueBound : 0,
                      config.membership_mode,
                      ssp);
}

std::vector<ZnElement> make_values(const pso::PublicParameters& pp,
                                   size_t sender_len) {
    std::vector<ZnElement> values;
    values.reserve(sender_len);
    for (size_t i = 0; i < sender_len; ++i) {
        values.emplace_back(pp.ring_ctx, BigInt(value_at(i)));
    }
    return values;
}

struct BenchResult {
    double value_preparation_ms = 0.0;
    double sender_ms = 0.0;
    double receiver_ms = 0.0;
    double wall_ms = 0.0;
    size_t result_size = 0;
    size_t sender_cardinality = 0;
    size_t receiver_cardinality = 0;
    uint64_t recovered_sum = 0;
    bool passed = false;
};

bool validate_result(const BenchConfig& config,
                     const Dataset& dataset,
                     const BenchResult& result) {
    const bool is_plain =
        config.membership_mode == cwprf_mqrpmt::MembershipMode::PlainSet;
    const size_t sender_len = dataset.vec_x.size();

    switch (config.operation) {
        case pso::PsoMode::kIntersection:
            return is_plain
                       ? result.result_size == dataset.intersection_size
                       : result.result_size >= dataset.intersection_size &&
                             result.result_size <= sender_len;

        case pso::PsoMode::kUnion:
            return is_plain
                       ? result.result_size == dataset.union_size
                       : result.result_size >= dataset.vec_y.size() &&
                             result.result_size <= dataset.union_size;

        case pso::PsoMode::kCard:
            return is_plain
                       ? result.receiver_cardinality ==
                             dataset.intersection_size
                       : result.receiver_cardinality >=
                                 dataset.intersection_size &&
                             result.receiver_cardinality <= sender_len;

        case pso::PsoMode::kCardSum: {
            const bool cardinality_matches_sender =
                result.sender_cardinality == result.receiver_cardinality;
            if (is_plain) {
                return cardinality_matches_sender &&
                       result.receiver_cardinality ==
                           dataset.intersection_size &&
                       result.recovered_sum == dataset.intersection_sum;
            }
            return cardinality_matches_sender &&
                   result.receiver_cardinality >= dataset.intersection_size &&
                   result.receiver_cardinality <= sender_len &&
                   result.recovered_sum >= dataset.intersection_sum;
        }
    }
    return false;
}

BenchResult run_once(const BenchConfig& config,
                     const pso::PublicParameters& pp,
                     const Dataset& dataset) {
    const std::string address = "127.0.0.1";
    std::vector<ZnElement> values;

    BenchResult result;
    if (config.operation == pso::PsoMode::kCardSum) {
        const auto preparation_begin = Clock::now();
        values = make_values(pp, dataset.vec_x.size());
        result.value_preparation_ms =
            Milliseconds(Clock::now() - preparation_begin).count();
    }

    const auto wall_begin = Clock::now();

    auto sender_future = std::async(std::launch::async, [&] {
        net::NetIO io("server", address, config.port);
        const auto begin = Clock::now();
        pso::SenderOutput output = pso::pso_sender(
            io, pp, dataset.vec_x, config.operation, values);
        return std::make_pair(
            std::move(output), Milliseconds(Clock::now() - begin).count());
    });

    auto receiver_future = std::async(std::launch::async, [&] {
        net::NetIO io("client", address, config.port);
        const auto begin = Clock::now();
        pso::ReceiverOutput output =
            pso::pso_receiver(io, pp, dataset.vec_y, config.operation);
        return std::make_pair(
            std::move(output), Milliseconds(Clock::now() - begin).count());
    });

    auto sender_result = sender_future.get();
    auto receiver_result = receiver_future.get();
    result.wall_ms = Milliseconds(Clock::now() - wall_begin).count();
    result.sender_ms = sender_result.second;
    result.receiver_ms = receiver_result.second;
    result.result_size = receiver_result.first.set_result.size();
    result.sender_cardinality = sender_result.first.cardinality;
    result.receiver_cardinality = receiver_result.first.cardinality;

    if (config.operation == pso::PsoMode::kCardSum) {
        result.recovered_sum =
            sender_result.first.card_sum.value.to_uint64();
    }
    result.passed = validate_result(config, dataset, result);
    return result;
}

double kelem_per_second(size_t element_count, double milliseconds) {
    return static_cast<double>(element_count) / milliseconds;
}

void print_header(const BenchConfig& config,
                  size_t sender_len,
                  size_t receiver_len) {
    std::cout
        << "\n"
        << "================================================================================\n"
        << config.label << "\n"
        << "--------------------------------------------------------------------------------\n"
        << "Sender set   (X): 2^" << kLogSenderLen << " = " << sender_len
        << "\n"
        << "Receiver set (Y): 2^" << kLogReceiverLen << " = "
        << receiver_len << "\n"
        << "Intersection    : 50% of X\n";
    if (config.mqrpmt_curve_id == NID_X25519) {
        std::cout << "X25519 Backend  : "
                  << crypto::x25519_simd::backend_name(crypto::x25519_simd::Backend::Auto)
                  << "\n";
    }
    std::cout << "================================================================================\n";
}

void print_result(const BenchConfig& config,
                  const Dataset& dataset,
                  const BenchResult& result) {
    std::cout << std::fixed << std::setprecision(2)
              << "Wall time        : " << result.wall_ms / 1000.0 << " s\n"
              << "Sender time      : " << result.sender_ms / 1000.0 << " s ("
              << kelem_per_second(dataset.vec_x.size(), result.sender_ms)
              << " Kelem/s)\n"
              << "Receiver time    : " << result.receiver_ms / 1000.0
              << " s ("
              << kelem_per_second(dataset.vec_x.size(), result.receiver_ms)
              << " Kelem/s)\n";

    if (config.operation == pso::PsoMode::kCardSum) {
        std::cout << "Value preparation: "
                  << result.value_preparation_ms / 1000.0 << " s\n";
    }

    switch (config.operation) {
        case pso::PsoMode::kIntersection:
            std::cout << "Intersection size : " << result.result_size
                      << " (expected " << dataset.intersection_size << ")\n";
            break;
        case pso::PsoMode::kUnion:
            std::cout << "Union size        : " << result.result_size
                      << " (expected " << dataset.union_size << ")\n";
            break;
        case pso::PsoMode::kCard:
            std::cout << "Cardinality       : "
                      << result.receiver_cardinality << " (expected "
                      << dataset.intersection_size << ")\n";
            break;
        case pso::PsoMode::kCardSum:
            std::cout << "Cardinality       : "
                      << result.receiver_cardinality << " (expected "
                      << dataset.intersection_size << ")\n"
                      << "Intersection sum  : " << result.recovered_sum
                      << " (expected " << dataset.intersection_sum << ")\n";
            break;
    }
    std::cout << "Validation        : "
              << (result.passed ? "PASS" : "FAIL") << "\n";
}

void print_summary(
    const std::vector<std::pair<BenchConfig, BenchResult>>& results) {
    constexpr int kLabelWidth = 45;

    std::cout
        << "\n\n"
        << "=========================================================================================================\n"
        << "                         Taihang Unified mqRPMT-PSO Benchmark Summary\n"
        << "=========================================================================================================\n"
        << "Threads: " << config::thread_num << " per party ("
        << config::thread_num * 2
        << " concurrent total on single-machine loopback)\n"
        << "---------------------------------------------------------------------------------------------------------\n"
        << std::left << std::setw(kLabelWidth) << "Configuration"
        << std::right << std::setw(12) << "Sender(s)"
        << std::setw(13) << "Receiver(s)" << std::setw(11) << "Wall(s)"
        << std::setw(14) << "Kelem/s" << std::setw(10) << "Status"
        << "\n"
        << "---------------------------------------------------------------------------------------------------------\n";

    for (const auto& [config_entry, result] : results) {
        std::cout << std::left << std::setw(kLabelWidth)
                  << config_entry.label << std::right << std::fixed
                  << std::setprecision(2) << std::setw(12)
                  << result.sender_ms / 1000.0 << std::setw(13)
                  << result.receiver_ms / 1000.0 << std::setw(11)
                  << result.wall_ms / 1000.0 << std::setw(14)
                  << kelem_per_second(size_t{1} << kLogSenderLen,
                                      result.wall_ms)
                  << std::setw(10) << (result.passed ? "PASS" : "FAIL")
                  << "\n";
    }
    std::cout
        << "=========================================================================================================\n";
}

}  // namespace

int main() {
    const size_t sender_len = size_t{1} << kLogSenderLen;
    const size_t receiver_len = size_t{1} << kLogReceiverLen;

    thread_configuration(BenchmarkMode::SingleMachine);

    std::cout << "Generating benchmark dataset...\n";
    const auto dataset_begin = Clock::now();
    const Dataset dataset = make_dataset(sender_len, receiver_len);
    std::cout << "Dataset ready in " << std::fixed << std::setprecision(2)
              << Milliseconds(Clock::now() - dataset_begin).count()
              << " ms\n";

    std::vector<std::pair<BenchConfig, BenchResult>> summary;
    summary.reserve(kConfigs.size());

    for (const BenchConfig& config : kConfigs) {
        print_header(config, sender_len, receiver_len);
        const pso::PublicParameters pp = make_public_parameters(config);
        const BenchResult result = run_once(config, pp, dataset);
        print_result(config, dataset, result);
        summary.emplace_back(config, result);
    }

    print_summary(summary);
    return 0;
}
