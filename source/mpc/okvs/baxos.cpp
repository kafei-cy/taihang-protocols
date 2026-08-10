/****************************
 * @file      baxos.cpp
 * @brief     Batched Paxos OKVS implementation.
 * @details   Implements a binned Paxos OKVS for "Blazing Fast PSI from
 *            Improved OKVS and Subfield VOLE":
 *            <https://eprint.iacr.org/2022/320>
 *            The implementation is modified from:
 *            <https://github.com/Visa-Research/volepsi.git>:
 *            (1) simplify the design;
 *            (2) support multi-thread programming with OpenMP.
 * @author    Yang Cao
 ****************************/

#include <taihang/mpc/okvs/baxos.hpp>
#include <taihang/common/check.hpp>
#include <taihang/crypto/aes.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace taihang::mpc::okvs {

template <DenseType dense_type, typename value_type>
Baxos<dense_type, value_type>::Baxos(const uint64_t item_num, const uint64_t bin_size,
                         const uint8_t sparse_weight, const uint8_t statistical_security_parameter, const prg::Seed *input_seed)
    : item_num(item_num),
      bin_num(0),
      sparse_weight(sparse_weight),
      statistical_security_parameter(statistical_security_parameter)
{
    if (item_num == 0)
    {
        throw std::invalid_argument("Baxos item_num must be positive.");
    }
    if (bin_size == 0)
    {
        throw std::invalid_argument("Baxos bin_size must be positive.");
    }
    if (sparse_weight < 2)
    {
        throw std::invalid_argument("Baxos requires sparse_weight >= 2.");
    }

    bin_num = (item_num - 1) / bin_size + 1;

    // Calculate the number of elements that can be safely stored in each bin
    item_num_per_bin = hashtable_bin_size(bin_num, item_num, statistical_security_parameter + std::log2(bin_num));
    if (item_num_per_bin == 0)
    {
        throw std::invalid_argument("Baxos parameters exceed the supported hashtable capacity table.");
    }

    seed = input_seed ? *input_seed : prg::set_seed(&kZeroBlock, 0);

    // Calculate sparse_size and dense_size for each bin
    {
        double logN = log2(item_num_per_bin);
        if (sparse_weight == 2)
        {
            double a = 7.529, b = 0.61, c = 2.556;
            double lambdaVsGap = a / (logN - c) + b;

            g_limit = static_cast<uint64_t>(std::ceil(statistical_security_parameter / lambdaVsGap + 1.9));
            sparse_size = 2 * item_num_per_bin;
        }
        else
        {
            double ee = 0;
            if (sparse_weight == 3)
                ee = 1.223;
            else if (sparse_weight == 4)
                ee = 1.293;
            else if (sparse_weight >= 5)
                ee = 0.1485 * sparse_weight + 0.6845;

            double logW = std::log2(sparse_weight);
            double logLambdaVsE = 0.555 * logN + 0.093 * std::pow(logW, 3) - 1.01 * std::pow(logW, 2) + 2.925 * logW - 0.133;
            double lambdaVsE = std::pow(2, logLambdaVsE);

            double b = -9.2 - lambdaVsE * ee;
            double e = (statistical_security_parameter - b) / lambdaVsE;
            g_limit = std::floor(statistical_security_parameter / ((sparse_weight - 2) * std::log2(e * item_num_per_bin)));
            sparse_size = item_num_per_bin * e;
        }
    }

    dense_size = g_limit + (dense_type == DenseType::Binary ? statistical_security_parameter : 0);
    total_size = sparse_size + dense_size;
}
template <DenseType dense_type, typename value_type>
template <typename idx_type>
void Baxos<dense_type, value_type>::impl_solve(const std::vector<Block> &keys, const std::vector<value_type> &values, std::vector<value_type> &output, prg::Seed *prng, uint8_t thread_num)
{
    if (thread_num == 0)
        thread_num = 1;

    if (bin_num == 1)
    {
        // If there is only one bin, then call single-threaded OKVS
        OKVS<idx_type, dense_type, value_type> paxos(item_num_per_bin, sparse_weight, statistical_security_parameter, &seed);
        paxos.set_keys(keys.data());
        output = paxos.encode(values, prng);
        return;
    }
    else
    {
        Divider bin_divider = gen_divider(bin_num);
        const uint64_t keys_size = keys.size();
        const Block *keys_data = keys.data();
        uint64_t bin_size_per_thread = 0;
        uint64_t bin_size_all_thread = 0;
        std::vector<std::vector<idx_type>> bin_size_thread;
        std::vector<uint64_t> total_bin_sizes(bin_num);
        std::unique_ptr<uint64_t[]> item_to_bin_thread;
        std::unique_ptr<Block[]> hash_to_bin_thread;
        std::atomic<bool> capacity_error{false};

#pragma omp parallel num_threads(thread_num)
        {
            const uint64_t thread_id = omp_get_thread_num();
            const uint64_t actual_thread_num = omp_get_num_threads();

#pragma omp single
            {
                auto item_num_per_thread =
                    (keys_size + actual_thread_num - 1) / actual_thread_num;
                bin_size_per_thread = hashtable_bin_size(
                    bin_num, item_num_per_thread, statistical_security_parameter);

                if (bin_size_per_thread == 0 ||
                    actual_thread_num > std::numeric_limits<uint64_t>::max() /
                                            bin_size_per_thread)
                {
                    capacity_error.store(true, std::memory_order_relaxed);
                }
                else
                {
                    bin_size_all_thread = actual_thread_num * bin_size_per_thread;
                    if (bin_num > std::numeric_limits<uint64_t>::max() /
                                      bin_size_all_thread)
                    {
                        capacity_error.store(true, std::memory_order_relaxed);
                    }
                    else
                    {
                        bin_size_thread.resize(actual_thread_num);
                        for (auto &bin_sizes : bin_size_thread)
                            bin_sizes.assign(bin_num, 0);
                        item_to_bin_thread.reset(
                            new uint64_t[bin_num * bin_size_all_thread]);
                        hash_to_bin_thread.reset(
                            new Block[bin_num * bin_size_all_thread]);
                    }
                }
            }

            if (!capacity_error.load(std::memory_order_relaxed))
            {
                const uint64_t begin =
                    (keys_size * thread_id) / actual_thread_num;
                const uint64_t len =
                    keys_size * (thread_id + 1) / actual_thread_num - begin;

                const Block *keys_thread_pointer = keys_data + begin;
                auto &bin_sizes = bin_size_thread[thread_id];

                // Hash all inputs in this thread's range into their bins.
                // Each thread has its own bins which are merged later.
                std::array<Block, 32> hashes;
                std::array<uint64_t, 32> bin_idxes;
                uint64_t i = 0;
                auto idx = begin;
                for (; i + 32 <= len; i += 32, keys_thread_pointer += 32)
                {
                    aes::encrypt_ecb(seed.aes_key, keys_thread_pointer, hashes.data(), 32);
                    for (auto j = 0; j < 32; j++)
                    {
                        hashes[j] ^= keys_thread_pointer[j];
                        const uint64_t *h_pointer64 = (uint64_t *)(hashes.data() + j);
                        const uint32_t *h_pointer32 = (uint32_t *)(h_pointer64);
                        bin_idxes[j] = h_pointer64[0] ^ h_pointer64[1] ^ h_pointer32[3];
                    }
                    reduce_mod32(bin_idxes.data(), &bin_divider, bin_num);
                    auto bin_idx_pointer = bin_idxes.data();
                    for (auto j = 0; j < 32; j++, idx++, bin_idx_pointer++)
                    {
                        auto bin_idx = *bin_idx_pointer;
                        auto &bin_size = bin_sizes[bin_idx];
                        if (bin_size >= bin_size_per_thread)
                        {
                            capacity_error.store(true, std::memory_order_relaxed);
                            continue;
                        }

                        auto bin_position = bin_size++;
                        auto bin_begin = bin_idx * bin_size_all_thread;
                        auto thread_begin = thread_id * bin_size_per_thread;
                        item_to_bin_thread[bin_begin + thread_begin + bin_position] = idx;
                        hash_to_bin_thread[bin_begin + thread_begin + bin_position] = hashes[j];
                    }
                }

                for (; i < len; i++, keys_thread_pointer++, idx++)
                {
                    auto hash_pointer = hashes.data();

                    aes::encrypt_ecb(seed.aes_key, keys_thread_pointer, hash_pointer, 1);
                    *hash_pointer ^= *keys_thread_pointer;

                    const uint64_t *hash_pointer64 = (uint64_t *)hash_pointer;
                    const uint32_t *hash_pointer32 = (uint32_t *)hash_pointer;
                    const uint64_t bin_idx = (hash_pointer64[0] ^ hash_pointer64[1] ^ hash_pointer32[3]) % bin_num;
                    auto &bin_size = bin_sizes[bin_idx];
                    if (bin_size >= bin_size_per_thread)
                    {
                        capacity_error.store(true, std::memory_order_relaxed);
                        continue;
                    }

                    auto bin_position = bin_size++;
                    auto bin_begin = bin_idx * bin_size_all_thread;
                    auto thread_begin = thread_id * bin_size_per_thread;
                    item_to_bin_thread[bin_begin + thread_begin + bin_position] = idx;
                    hash_to_bin_thread[bin_begin + thread_begin + bin_position] = hashes[0];
                }
            }

#pragma omp barrier
            if (!capacity_error.load(std::memory_order_relaxed))
            {
                // This thread iterates over its assigned bins and aggregates all items
                // mapped to the bin by different threads.
                for (uint64_t bin_idx = thread_id;
                     bin_idx < bin_num;
                     bin_idx += actual_thread_num)
                {
                    uint64_t bin_size = 0;
                    for (const auto &bin_size_thread_bin : bin_size_thread)
                        bin_size += bin_size_thread_bin[bin_idx];

                    total_bin_sizes[bin_idx] = bin_size;
                    if (bin_size > item_num_per_bin)
                        capacity_error.store(true, std::memory_order_relaxed);
                }
            }

#pragma omp barrier
            if (!capacity_error.load(std::memory_order_relaxed))
            {
                for (uint64_t bin_idx = thread_id;
                     bin_idx < bin_num;
                     bin_idx += actual_thread_num)
                {
                    const uint64_t bin_size = total_bin_sizes[bin_idx];

                    // Initialize small-sized single-threaded OKVS
                    OKVS<idx_type, dense_type, value_type> paxos;
                    paxos.item_num = bin_size;
                    paxos.sparse_weight = sparse_weight;
                    paxos.sparse_size = sparse_size;
                    paxos.dense_size = dense_size;
                    paxos.total_size = total_size;
                    paxos.seed = seed;
                    paxos.statistical_security_parameter = statistical_security_parameter;
                    paxos.g_limit = g_limit;

                    // Allocate storage space for variables, the process is similar to the OKVS::allocate() function
                    auto allocate_size = sizeof(idx_type) * (item_num_per_bin * sparse_weight * 2 + sparse_size) + sizeof(idx_type *) * sparse_size;
                    std::unique_ptr<uint8_t[]> storage(new uint8_t[allocate_size]);
                    uint8_t *iter = storage.get();

                    paxos.h_sparse.resize(iter, item_num_per_bin, sparse_weight);
                    iter += item_num_per_bin * sparse_weight * sizeof(idx_type);

                    paxos.col_weights = (idx_type *)iter;
                    iter += sparse_size * sizeof(idx_type);

                    iter += sparse_size * sizeof(idx_type *);

                    paxos.h_cols.resize(iter, sparse_size);
                    iter += item_num_per_bin * sparse_weight * sizeof(idx_type);

                    TAIHANG_ASSERT(iter == storage.get() + allocate_size, "Baxos bin storage layout mismatch.");

                    auto bin_begin = bin_idx * bin_size_all_thread;
                    auto item_pointer = item_to_bin_thread.get() + bin_begin;
                    auto hashes_pointer = hash_to_bin_thread.get() + bin_begin;
                    auto output_pointer = output.data() + bin_idx * total_size;

                    // For each thread, copy the hashes and input indexes that it mapped to this bin.

                    auto bin_pos = bin_size_thread[0][bin_idx];
                    for (uint64_t thread_idx = 1; thread_idx < actual_thread_num; thread_idx++)
                    {
                        auto size = bin_size_thread[thread_idx][bin_idx];
                        auto thread_begin = thread_idx * bin_size_per_thread;
                        auto item_thread = item_to_bin_thread.get() + bin_begin + thread_begin;
                        auto hash_thread = hash_to_bin_thread.get() + bin_begin + thread_begin;

                        memmove(item_pointer + bin_pos, item_thread, size * sizeof(uint64_t));
                        memmove(hashes_pointer + bin_pos, hash_thread, size * sizeof(Block));
                        bin_pos += size;
                    }

                    std::vector<value_type> bin_values(bin_size);
                    for (uint64_t j = 0; j < bin_size; ++j)
                        bin_values[j] = values[item_pointer[j]];

                    // Initialization process, similar to OKVS::set_keys
                    memset(paxos.col_weights, 0, sizeof(idx_type) * sparse_size);
                    {
                        paxos.h_dense = hashes_pointer;
                        paxos.sparse_weight = sparse_weight;
                        paxos.weight_nodes.reset(new typename OKVS<idx_type, dense_type, value_type>::weight_node[sparse_size]);

                        paxos.weight_set.resize(200);
                        paxos.mModVals.resize(sparse_weight);
                        paxos.mMods.resize(sparse_weight);
                        for (uint8_t ii = 0; ii < sparse_weight; ++ii)
                        {
                            const idx_type temp = sparse_size - ii;
                            paxos.mModVals[ii] = (temp);
                            paxos.mMods[ii] = (gen_divider(temp));
                        }
                        // Compute the rows and count the column weight.
                        paxos.set_sparse();
                        paxos.weight_statistic();
                        paxos.init_hcols();
                    }
                    paxos.encode(bin_values.data(), output_pointer, prng);
                }
            }
        }
        if (capacity_error.load(std::memory_order_relaxed))
            throw std::runtime_error("Baxos bin capacity exceeded.");
    }
}

uint64_t get_bin_idx(Block *p)
{
    auto p64 = (uint64_t *)p;
    auto p32 = (uint32_t *)p;
    return p64[0] ^ p64[1] ^ p32[3];
}

template <DenseType dense_type, typename value_type>
template <typename idx_type>
void Baxos<dense_type, value_type>::impl_decode_batch(Block *keys, value_type *values, uint64_t batch_len, value_type *output)
{
    // Decode the given inputs based on the Paxos data. The output is written to values.
    // Decode is performed in units of decode_size groups
    auto decode_size = std::min(uint64_t(512), batch_len);
    std::vector<std::vector<Block>> batches(bin_num);
    std::vector<std::vector<uint64_t>> keys_idxes(bin_num);

    // Initialize small-sized single-threaded OKVS
    OKVS<idx_type, dense_type, value_type> paxos;
    {
        paxos.item_num = decode_size;
        paxos.sparse_weight = sparse_weight;
        paxos.sparse_size = sparse_size;
        paxos.dense_size = dense_size;
        paxos.total_size = total_size;
        paxos.seed = seed;
        paxos.statistical_security_parameter = statistical_security_parameter;
        paxos.g_limit = g_limit;
        for (uint8_t i = 0; i < sparse_weight; ++i)
        {
            auto temp = sparse_size - i;
            paxos.mModVals.emplace_back(temp);
            paxos.mMods.emplace_back(gen_divider(temp));
        }
    }
    std::array<Block, 32> buffer;
    std::vector<value_type> value_buffer(decode_size);
    std::array<uint64_t, 32> bin_idxes;
    Divider bin_divider = gen_divider(bin_num);
    uint64_t i = 0;
    // Iterate over the input.
    for (; i + 32 <= batch_len; i += 32, keys += 32)
    {
        paxos.set_dense(keys, 32, buffer.data());

        for (auto j = 0; j < 32; j += 8)
        {
            auto bin_idx_pointer = bin_idxes.data() + j;
            auto buffer_pointer = buffer.data() + j;

            *bin_idx_pointer++ = get_bin_idx(buffer_pointer++);
            *bin_idx_pointer++ = get_bin_idx(buffer_pointer++);
            *bin_idx_pointer++ = get_bin_idx(buffer_pointer++);
            *bin_idx_pointer++ = get_bin_idx(buffer_pointer++);
            *bin_idx_pointer++ = get_bin_idx(buffer_pointer++);
            *bin_idx_pointer++ = get_bin_idx(buffer_pointer++);
            *bin_idx_pointer++ = get_bin_idx(buffer_pointer++);
            *bin_idx_pointer = get_bin_idx(buffer_pointer);
        }
        reduce_mod32(bin_idxes.data(), &bin_divider, bin_num);

        for (auto k = 0; k < 32; k++)
        {

            auto bin_idx = bin_idxes[k];
            auto &batch = batches[bin_idx];
            auto &idxes = keys_idxes[bin_idx];
            batch.push_back(buffer[k]);
            idxes.push_back(i + k);
            // If after processing the current key,
            // the decode_size size group (the unit of decoding) is just filled,
            // then start decoding immediately
            if (batch.size() == decode_size)
            {
                auto output_pointer = output + bin_idx * total_size;
                paxos.h_dense = batch.data();
                paxos.decode(nullptr, decode_size, output_pointer, value_buffer.data(), batch.data());

                for (uint64_t ii = 0; ii < decode_size; ii++)
                {
                    values[idxes[ii]] = value_buffer[ii];
                }
                batch.clear();
                idxes.clear();
            }
        }
    }
    // Perform decoding preprocessing on the remaining groups of less than 32 elements
    for (; i < batch_len; i++, keys++)
    {
        paxos.set_dense(keys, 1, buffer.data());
        auto bin_idx = get_bin_idx(buffer.data()) % bin_num;

        auto &batch_bin = batches[bin_idx];
        auto &idxes = keys_idxes[bin_idx];
        batch_bin.push_back(buffer[0]);
        idxes.push_back(i);

        // Similarly, once the number of processing reaches decode_size, start decoding immediately
        if (batch_bin.size() == decode_size)
        {
            auto output_pointer = output + bin_idx * total_size;
            paxos.h_dense = batch_bin.data();
            paxos.decode(nullptr, decode_size, output_pointer, value_buffer.data(), batch_bin.data());
            for (uint64_t ii = 0; ii < decode_size; ii++)
            {
                values[idxes[ii]] = value_buffer[ii];
            }
            batch_bin.clear();
            idxes.clear();
        }
    }

    // It is no longer required that the unit of decoding must be a group of decode_size size,
    // and handle bins with insufficient size
    for (uint64_t bin_idx = 0; bin_idx < bin_num; bin_idx++)
    {
        const auto batch_size = batches[bin_idx].size();
        if (batch_size)
        {
            auto output_pointer = output + bin_idx * total_size;
            paxos.decode(nullptr, batch_size, output_pointer, value_buffer.data(), batches[bin_idx].data());
            for (uint64_t ii = 0; ii < batch_size; ii++)
            {
                values[keys_idxes[bin_idx][ii]] = value_buffer[ii];
            }
        }
    }
}
template <DenseType dense_type, typename value_type>
template <typename idx_type>
void Baxos<dense_type, value_type>::impl_decode(const std::vector<Block> &keys, std::vector<value_type> &values, const std::vector<value_type> &output, uint8_t thread_num)
{
    if (thread_num == 0)
        thread_num = 1;

    if (bin_num == 1)
    {
        OKVS<idx_type, dense_type, value_type> paxos(item_num_per_bin, sparse_weight, statistical_security_parameter, &seed);
        paxos.decode(keys.data(), keys.size(), output.data(), values.data());
        return;
    }
    auto keys_size = keys.size();
    auto keys_begin = keys.data();
    auto values_begin = values.data();
#pragma omp parallel num_threads(thread_num)
    {
        // Create the desired number of threads and split up the work.
        const uint64_t thread_id = omp_get_thread_num();
        const uint64_t actual_thread_num = omp_get_num_threads();
        uint64_t begin = (keys_size * thread_id) / actual_thread_num;
        uint64_t len = keys_size * (thread_id + 1) / actual_thread_num - begin;

        auto keys_pointer = keys_begin + begin;
        auto values_pointer = values_begin + begin;
        if (len != 0)
            impl_decode_batch<idx_type>((Block *)keys_pointer, values_pointer, len, (value_type *)output.data());
    }
}

template <DenseType dense_type, typename value_type>
void Baxos<dense_type, value_type>::solve(const std::vector<Block> &keys, const std::vector<value_type> &values, std::vector<value_type> &output, prg::Seed *prng, uint8_t thread_num)
{
    if (keys.size() != values.size() || keys.size() != item_num)
    {
        throw std::invalid_argument("Baxos solve input size mismatch.");
    }
    if (total_size != 0 && bin_num > std::numeric_limits<size_t>::max() / total_size)
    {
        throw std::invalid_argument("Baxos output size overflows size_t.");
    }
    if (output.size() < bin_num * total_size)
    {
        throw std::invalid_argument("Baxos solve output size mismatch.");
    }
    if (thread_num == 0)
        thread_num = 1;

    // Select the smallest index type which will work.
    auto bit_len = log2_ceil(sparse_size + 1);

    if (bit_len <= 8)
    {
        impl_solve<uint8_t>(keys, values, output, prng, thread_num);
    }
    else if (bit_len <= 16)
    {
        impl_solve<uint16_t>(keys, values, output, prng, thread_num);
    }
    else if (bit_len <= 32)
    {
        impl_solve<uint32_t>(keys, values, output, prng, thread_num);
    }
    else if (bit_len <= 64)
    {
        impl_solve<uint64_t>(keys, values, output, prng, thread_num);
    }
}

template <DenseType dense_type, typename value_type>
void Baxos<dense_type, value_type>::decode(const std::vector<Block> &keys, std::vector<value_type> &values, const std::vector<value_type> &output, uint8_t thread_num)
{
    if (values.size() < keys.size())
    {
        throw std::invalid_argument("Baxos decode value size mismatch.");
    }
    if (total_size != 0 && bin_num > std::numeric_limits<size_t>::max() / total_size)
    {
        throw std::invalid_argument("Baxos output size overflows size_t.");
    }
    if (output.size() < bin_num * total_size)
    {
        throw std::invalid_argument("Baxos decode output size mismatch.");
    }
    if (thread_num == 0)
        thread_num = 1;

    // Select the smallest index type which will work.
    auto bit_len = log2_ceil(sparse_size + 1);
    if (bit_len <= 8)
    {
        impl_decode<uint8_t>(keys, values, output, thread_num);
    }
    else if (bit_len <= 16)
    {
        impl_decode<uint16_t>(keys, values, output, thread_num);
    }
    else if (bit_len <= 32)
    {
        impl_decode<uint32_t>(keys, values, output, thread_num);
    }
    else if (bit_len <= 64)
    {
        impl_decode<uint64_t>(keys, values, output, thread_num);
    }
}

template class Baxos<DenseType::Binary, Block>;
template class Baxos<DenseType::Gf128, Block>;

} // namespace taihang::mpc::okvs
