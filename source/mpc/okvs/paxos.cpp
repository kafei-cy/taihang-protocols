/****************************
 * @file      paxos.cpp
 * @brief     Paxos OKVS implementation.
 * @details   Implements the core Paxos OKVS algorithm for "Blazing Fast PSI
 *            from Improved OKVS and Subfield VOLE":
 *            <https://eprint.iacr.org/2022/320>
 *            The implementation is modified from:
 *            <https://github.com/Visa-Research/volepsi.git>:
 *            (1) simplify the design;
 *            (2) add serialize/deserialize interfaces for variables such as matrices;
 *            (3) fix two overflow issues when the weight is not 3.
 * @author    Yang Cao
 ****************************/

#include <taihang/mpc/okvs/paxos.hpp>
#include <taihang/common/check.hpp>
#include <taihang/crypto/aes.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace taihang::mpc::okvs {

Block block_and(const Block& lhs, const Block& rhs)
{
#if defined(TAIHANG_ARCH_X64)
   return _mm_and_si128(lhs.mm, rhs.mm);
#else
   return vandq_u8(lhs.mm, rhs.mm);
#endif
}

template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::allocate()
{
   const uint64_t h_sparse_capacity = item_num * sparse_weight * sizeof(idx_type);

   const uint64_t h_dense_capacity = item_num * sizeof(Block);

   const uint64_t col_weights_pointer = h_sparse_capacity + h_dense_capacity;
   const uint64_t col_weights_capacity = sparse_size * sizeof(idx_type);

   const uint64_t h_cols_pointer = col_weights_pointer + col_weights_capacity;
   const uint64_t h_cols_capacity = h_sparse_capacity;

   const uint64_t capacity = h_cols_pointer + h_cols_capacity;
   storage.reset(new uint8_t[capacity]());

   const uint8_t *iter = storage.get();

   h_dense = (Block *)(iter);
   h_sparse.resize(iter + h_dense_capacity, item_num, sparse_weight);

   col_weights = (idx_type *)(iter + col_weights_pointer);

   h_cols.resize(iter + h_cols_pointer, sparse_size);
   weight_nodes.reset(new weight_node[sparse_size]);

   weight_set.resize(200);
}

template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::calculate_sparse_size()
{
   double logN = log2(item_num);
   if (sparse_weight < 2)
   {
      throw std::invalid_argument("OKVS requires sparse_weight >= 2.");
   }
   else if (sparse_weight == 2)
   {
      double a = 7.529, b = 0.61, c = 2.556;
      double lambdaVsGap = a / (logN - c) + b;

      g_limit = static_cast<uint64_t>(std::ceil(statistical_security_parameter / lambdaVsGap + 1.9));
      sparse_size = 2 * item_num;
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
      g_limit = std::floor(statistical_security_parameter / ((sparse_weight - 2) * std::log2(e * item_num)));
      sparse_size = item_num * e;
   }
}

template <typename idx_type, DenseType dense_type, typename value_type>
OKVS<idx_type, dense_type, value_type>::OKVS(const idx_type item_num, const uint8_t sparse_weight, const uint8_t statistical_security_parameter, const prg::Seed *input_seed)
    : item_num(item_num), statistical_security_parameter(statistical_security_parameter), sparse_weight(sparse_weight)
{

   seed = input_seed ? *input_seed : prg::set_seed(&kZeroBlock, 0);

   // Calculate g_limit, sparse_size and dense_size
   calculate_sparse_size();

   // allocate storage for variables
   if (storage == nullptr)
      allocate();

   dense_size = g_limit + (dense_type == DenseType::Binary ? statistical_security_parameter : 0);
   total_size = sparse_size + dense_size;

   // Initialize mModvals and mMods to prepare for set_sparse(set_Mtx)
   mModVals.resize(sparse_weight);
   mMods.resize(sparse_weight);
   for (uint8_t i = 0; i < sparse_weight; ++i)
   {
      auto temp = sparse_size - i;
      mModVals[i] = temp;
      mMods[i] = (gen_divider(temp));
   }
}

template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::set_dense(const Block *keys, idx_type n, Block *dest)
{
   if (n == 0)
      n = item_num;
   if (dest == nullptr)
      dest = h_dense;
   aes::encrypt_ecb(seed.aes_key, (Block *)keys, dest, n);
   auto dense_pointer = dest;
   auto key_pointer = keys;
   uint64_t i = 0;
   for (; i + 8 <= n; i += 8, dense_pointer += 8, key_pointer += 8)
   {
      dense_pointer[0] ^= key_pointer[0];
      dense_pointer[1] ^= key_pointer[1];
      dense_pointer[2] ^= key_pointer[2];
      dense_pointer[3] ^= key_pointer[3];
      dense_pointer[4] ^= key_pointer[4];
      dense_pointer[5] ^= key_pointer[5];
      dense_pointer[6] ^= key_pointer[6];
      dense_pointer[7] ^= key_pointer[7];
   }
   for (; i < n; i++, dense_pointer++, key_pointer++)
   {
      *dense_pointer ^= *key_pointer;
   }
}
template <typename idx_type, DenseType dense_type, typename value_type>
__attribute__((target("avx2")))
void OKVS<idx_type, dense_type, value_type>::set_sparse_32(const Block *dense, idx_type *sparse)
{
   if (sparse_weight == 3)
   {
      Block row128_[3][16];
      for (uint8_t i = 0; i < sparse_weight; i++)
      {
         uint64_t *ll = (uint64_t *)row128_[i];

         for (uint8_t j = 0; j < 32; j++)
         {
            memcpy(&ll[j], (uint32_t *)(dense + j) + i, sizeof(uint64_t));
         }

         mod32(ll, i);
      }

      for (uint8_t i = 0; i < 2; i++)
      {
         std::array<Block, 8> mask, max, min;
         std::array<Block *, 3> row128{
             row128_[0] + i * 8,
             row128_[1] + i * 8,
             row128_[2] + i * 8};
         auto weight_0_idx = row128[0];
         auto weight_1_idx = row128[1];
         auto weight_2_idx = row128[2];

         // mask = (weight_0 > weight_1) ? -1:0;
         mask[0] = _mm_cmpgt_epi64(weight_0_idx[0], weight_1_idx[0]);
         mask[1] = _mm_cmpgt_epi64(weight_0_idx[1], weight_1_idx[1]);
         mask[2] = _mm_cmpgt_epi64(weight_0_idx[2], weight_1_idx[2]);
         mask[3] = _mm_cmpgt_epi64(weight_0_idx[3], weight_1_idx[3]);
         mask[4] = _mm_cmpgt_epi64(weight_0_idx[4], weight_1_idx[4]);
         mask[5] = _mm_cmpgt_epi64(weight_0_idx[5], weight_1_idx[5]);
         mask[6] = _mm_cmpgt_epi64(weight_0_idx[6], weight_1_idx[6]);
         mask[7] = _mm_cmpgt_epi64(weight_0_idx[7], weight_1_idx[7]);

         // min = weight_0 ^ weight_1
         min[0] = weight_0_idx[0] ^ weight_1_idx[0];
         min[1] = weight_0_idx[1] ^ weight_1_idx[1];
         min[2] = weight_0_idx[2] ^ weight_1_idx[2];
         min[3] = weight_0_idx[3] ^ weight_1_idx[3];
         min[4] = weight_0_idx[4] ^ weight_1_idx[4];
         min[5] = weight_0_idx[5] ^ weight_1_idx[5];
         min[6] = weight_0_idx[6] ^ weight_1_idx[6];
         min[7] = weight_0_idx[7] ^ weight_1_idx[7];

         // max = min & mask = mask = 0 <=> weight_1 > weight_0 ? 0 : weight_0 ^ weihgt 1;
         max[0] = block_and(min[0], mask[0]);
         max[1] = block_and(min[1], mask[1]);
         max[2] = block_and(min[2], mask[2]);
         max[3] = block_and(min[3], mask[3]);
         max[4] = block_and(min[4], mask[4]);
         max[5] = block_and(min[5], mask[5]);
         max[6] = block_and(min[6], mask[6]);
         max[7] = block_and(min[7], mask[7]);

         // max = mask =0 <=> weight_1 > weight_0 ? weight_1 : weight_0;
         // Therefore, max = max(weight_1,weight_0);
         max[0] = max[0] ^ weight_1_idx[0];
         max[1] = max[1] ^ weight_1_idx[1];
         max[2] = max[2] ^ weight_1_idx[2];
         max[3] = max[3] ^ weight_1_idx[3];
         max[4] = max[4] ^ weight_1_idx[4];
         max[5] = max[5] ^ weight_1_idx[5];
         max[6] = max[6] ^ weight_1_idx[6];
         max[7] = max[7] ^ weight_1_idx[7];

         // min = weight_0 ^ weight_1 ^ max = min(weight_0,weihgt_1)
         min[0] = min[0] ^ max[0];
         min[1] = min[1] ^ max[1];
         min[2] = min[2] ^ max[2];
         min[3] = min[3] ^ max[3];
         min[4] = min[4] ^ max[4];
         min[5] = min[5] ^ max[5];
         min[6] = min[6] ^ max[6];
         min[7] = min[7] ^ max[7];

         // mask = max==weight_1 ? -1 : 0;
         mask[0] = _mm_cmpeq_epi64(max[0], weight_1_idx[0]);
         mask[1] = _mm_cmpeq_epi64(max[1], weight_1_idx[1]);
         mask[2] = _mm_cmpeq_epi64(max[2], weight_1_idx[2]);
         mask[3] = _mm_cmpeq_epi64(max[3], weight_1_idx[3]);
         mask[4] = _mm_cmpeq_epi64(max[4], weight_1_idx[4]);
         mask[5] = _mm_cmpeq_epi64(max[5], weight_1_idx[5]);
         mask[6] = _mm_cmpeq_epi64(max[6], weight_1_idx[6]);
         mask[7] = _mm_cmpeq_epi64(max[7], weight_1_idx[7]);

         // if max==weight_1 : weight_1++,that is,weight_1 -= mask
         weight_1_idx[0] = _mm_sub_epi64(weight_1_idx[0], mask[0]);
         weight_1_idx[1] = _mm_sub_epi64(weight_1_idx[1], mask[1]);
         weight_1_idx[2] = _mm_sub_epi64(weight_1_idx[2], mask[2]);
         weight_1_idx[3] = _mm_sub_epi64(weight_1_idx[3], mask[3]);
         weight_1_idx[4] = _mm_sub_epi64(weight_1_idx[4], mask[4]);
         weight_1_idx[5] = _mm_sub_epi64(weight_1_idx[5], mask[5]);
         weight_1_idx[6] = _mm_sub_epi64(weight_1_idx[6], mask[6]);
         weight_1_idx[7] = _mm_sub_epi64(weight_1_idx[7], mask[7]);

         // if max=weight_1 : max++,that is, max -= mask
         max[0] = _mm_sub_epi64(max[0], mask[0]);
         max[1] = _mm_sub_epi64(max[1], mask[1]);
         max[2] = _mm_sub_epi64(max[2], mask[2]);
         max[3] = _mm_sub_epi64(max[3], mask[3]);
         max[4] = _mm_sub_epi64(max[4], mask[4]);
         max[5] = _mm_sub_epi64(max[5], mask[5]);
         max[6] = _mm_sub_epi64(max[6], mask[6]);
         max[7] = _mm_sub_epi64(max[7], mask[7]);

         // compare min(weight_0,weight_1) and weight_2
         // if weight_2 > min,mask= -1 else mask 0;
         mask[0] = _mm_cmpgt_epi64(min[0], weight_2_idx[0]);
         mask[1] = _mm_cmpgt_epi64(min[1], weight_2_idx[1]);
         mask[2] = _mm_cmpgt_epi64(min[2], weight_2_idx[2]);
         mask[3] = _mm_cmpgt_epi64(min[3], weight_2_idx[3]);
         mask[4] = _mm_cmpgt_epi64(min[4], weight_2_idx[4]);
         mask[5] = _mm_cmpgt_epi64(min[5], weight_2_idx[5]);
         mask[6] = _mm_cmpgt_epi64(min[6], weight_2_idx[6]);
         mask[7] = _mm_cmpgt_epi64(min[7], weight_2_idx[7]);
         mask[0] = mask[0] ^ kAllOneBlock;
         mask[1] = mask[1] ^ kAllOneBlock;
         mask[2] = mask[2] ^ kAllOneBlock;
         mask[3] = mask[3] ^ kAllOneBlock;
         mask[4] = mask[4] ^ kAllOneBlock;
         mask[5] = mask[5] ^ kAllOneBlock;
         mask[6] = mask[6] ^ kAllOneBlock;
         mask[7] = mask[7] ^ kAllOneBlock;

         // if weight_2>min,weight_2++ <=> weight_2-=mask;
         weight_2_idx[0] = _mm_sub_epi64(weight_2_idx[0], mask[0]);
         weight_2_idx[1] = _mm_sub_epi64(weight_2_idx[1], mask[1]);
         weight_2_idx[2] = _mm_sub_epi64(weight_2_idx[2], mask[2]);
         weight_2_idx[3] = _mm_sub_epi64(weight_2_idx[3], mask[3]);
         weight_2_idx[4] = _mm_sub_epi64(weight_2_idx[4], mask[4]);
         weight_2_idx[5] = _mm_sub_epi64(weight_2_idx[5], mask[5]);
         weight_2_idx[6] = _mm_sub_epi64(weight_2_idx[6], mask[6]);
         weight_2_idx[7] = _mm_sub_epi64(weight_2_idx[7], mask[7]);

         // mask = weight_2>max?-1:0;
         mask[0] = _mm_cmpgt_epi64(max[0], weight_2_idx[0]);
         mask[1] = _mm_cmpgt_epi64(max[1], weight_2_idx[1]);
         mask[2] = _mm_cmpgt_epi64(max[2], weight_2_idx[2]);
         mask[3] = _mm_cmpgt_epi64(max[3], weight_2_idx[3]);
         mask[4] = _mm_cmpgt_epi64(max[4], weight_2_idx[4]);
         mask[5] = _mm_cmpgt_epi64(max[5], weight_2_idx[5]);
         mask[6] = _mm_cmpgt_epi64(max[6], weight_2_idx[6]);
         mask[7] = _mm_cmpgt_epi64(max[7], weight_2_idx[7]);
         mask[0] = mask[0] ^ kAllOneBlock;
         mask[1] = mask[1] ^ kAllOneBlock;
         mask[2] = mask[2] ^ kAllOneBlock;
         mask[3] = mask[3] ^ kAllOneBlock;
         mask[4] = mask[4] ^ kAllOneBlock;
         mask[5] = mask[5] ^ kAllOneBlock;
         mask[6] = mask[6] ^ kAllOneBlock;
         mask[7] = mask[7] ^ kAllOneBlock;

         // if weight_2>max,weight_2++ <=> weight_2-=mask;
         weight_2_idx[0] = _mm_sub_epi64(weight_2_idx[0], mask[0]);
         weight_2_idx[1] = _mm_sub_epi64(weight_2_idx[1], mask[1]);
         weight_2_idx[2] = _mm_sub_epi64(weight_2_idx[2], mask[2]);
         weight_2_idx[3] = _mm_sub_epi64(weight_2_idx[3], mask[3]);
         weight_2_idx[4] = _mm_sub_epi64(weight_2_idx[4], mask[4]);
         weight_2_idx[5] = _mm_sub_epi64(weight_2_idx[5], mask[5]);
         weight_2_idx[6] = _mm_sub_epi64(weight_2_idx[6], mask[6]);
         weight_2_idx[7] = _mm_sub_epi64(weight_2_idx[7], mask[7]);

#ifndef NDEBUG
         for (auto j = 0; j < sparse_weight; j++)
         {
            uint64_t *__restrict row64 = (uint64_t *)(row128[j]);
            for (auto k = 0; k < 16; k++)
            {
               TAIHANG_ASSERT(row64[k] < sparse_size, "OKVS sparse index out of bounds.");
            }
         }
#endif

         for (uint8_t j = 0; j < sparse_weight; ++j)
         {
            idx_type *__restrict rowi = sparse + sparse_weight * (16 * i);
            uint64_t *__restrict row64 = (uint64_t *)(row128[j]);
            auto iter = j;
            rowi[iter] = row64[0];
            iter += sparse_weight;
            rowi[iter] = row64[1];
            iter += sparse_weight;
            rowi[iter] = row64[2];
            iter += sparse_weight;
            rowi[iter] = row64[3];
            iter += sparse_weight;
            rowi[iter] = row64[4];
            iter += sparse_weight;
            rowi[iter] = row64[5];
            iter += sparse_weight;
            rowi[iter] = row64[6];
            iter += sparse_weight;
            rowi[iter] = row64[7];
            iter += sparse_weight;

            if (!is_decoding)
            {
               auto row64_temp = row64;
               col_weights[*row64_temp++]++;
               col_weights[*row64_temp++]++;
               col_weights[*row64_temp++]++;
               col_weights[*row64_temp++]++;
               col_weights[*row64_temp++]++;
               col_weights[*row64_temp++]++;
               col_weights[*row64_temp++]++;
               col_weights[*row64_temp]++;
            }

            rowi += (iter - j);
            row64 += 8;

            iter = j;
            rowi[iter] = row64[0];
            iter += sparse_weight;
            rowi[iter] = row64[1];
            iter += sparse_weight;
            rowi[iter] = row64[2];
            iter += sparse_weight;
            rowi[iter] = row64[3];
            iter += sparse_weight;
            rowi[iter] = row64[4];
            iter += sparse_weight;
            rowi[iter] = row64[5];
            iter += sparse_weight;
            rowi[iter] = row64[6];
            iter += sparse_weight;
            rowi[iter] = row64[7];
            iter += sparse_weight;

            if (!is_decoding)
            {
               auto row64_temp = row64;
               col_weights[*row64_temp++]++;
               col_weights[*row64_temp++]++;
               col_weights[*row64_temp++]++;
               col_weights[*row64_temp++]++;
               col_weights[*row64_temp++]++;
               col_weights[*row64_temp++]++;
               col_weights[*row64_temp++]++;
               col_weights[*row64_temp]++;
            }
         }
      }
   }
   else
   {
      for (uint8_t i = 0; i < 32; i++)
      {
         set_sparse_1(dense + i, sparse + i * sparse_weight);
      }
   }
}
template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::set_sparse_32(const idx_type row)
{
   Block *dense = h_dense + row;
   idx_type *sparse = h_sparse[row];
   set_sparse_32(dense, sparse);
}

template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::set_sparse_1(const Block *dense, idx_type *sparse)
{
   if (sparse_weight == 3)
   {
      uint32_t *rr = (uint32_t *)dense;
      auto rr0 = *(uint64_t *)(&rr[0]);
      auto rr1 = *(uint64_t *)(&rr[1]);
      auto rr2 = *(uint64_t *)(&rr[2]);

      auto Row = sparse;

      Row[0] = divide_u64_do(rr0, &mMods[0]);
      Row[1] = divide_u64_do(rr1, &mMods[1]);
      Row[2] = divide_u64_do(rr2, &mMods[2]);

      Row[0] = rr0 - Row[0] * mModVals[0];
      Row[1] = rr1 - Row[1] * mModVals[1];
      Row[2] = rr2 - Row[2] * mModVals[2];
      auto min = std::min<idx_type>(Row[0], Row[1]);
      auto max = Row[0] + Row[1] - min;

      if (max == Row[1])
      {
         ++Row[1];
         ++max;
      }

      if (Row[2] >= min)
         ++Row[2];

      if (Row[2] >= max)
         ++Row[2];
      if (!is_decoding)
      {
         for (uint8_t i = 0; i < 3; i++)
         {
            col_weights[Row[i]]++;
         }
      }
   }
   else
   {
      uint8_t j = 0;
      Block dense_temp = *dense;
      for (; j + 3 < sparse_weight; j += 3)
      {
         for (auto j2 = 0; j2 < 3; j2++)
         {
            auto iter = j + j2;
            uint64_t col = *(uint64_t *)(&((uint32_t *)(&dense_temp))[j2]);
            col = col % (sparse_size - iter);
            uint8_t k = 0, end = 0;
            for (k = 0; k < iter; k++)
            {
               if (col >= sparse[k])
                  col++;
               else
                  break;
            }
            for (end = iter; end > k; end--)
            {
               sparse[end] = sparse[end - 1];
            }
            sparse[end] = static_cast<idx_type>(col);
            if (!is_decoding)
               col_weights[col]++;
         }
         dense_temp = gf128_mul(dense_temp, dense_temp);
      }
      for (; j < sparse_weight; j++)
      {
         uint64_t col = *(uint64_t *)(&((uint32_t *)(&dense_temp))[j % 3]);
         col = col % (sparse_size - j);
         uint8_t k = 0, end = 0;
         for (k = 0; k < j; k++)
         {
            if (col >= sparse[k])
               col++;
            else
               break;
         }
         for (end = j; end > k; end--)
         {
            sparse[end] = sparse[end - 1];
         }
         sparse[end] = static_cast<idx_type>(col);
         if (!is_decoding)
            col_weights[col]++;
      }
   }
}

template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::set_sparse_1(const idx_type row)
{
   auto dense = h_dense + row;
   auto sparse = h_sparse[row];
   set_sparse_1(dense, sparse);
}

template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::set_sparse()
{
   idx_type i = 0;
   for (i = 0; i + 32 <= item_num; i += 32)
   {
      set_sparse_32(i);
   }

   for (; i < item_num; i++)
   {
      set_sparse_1(i);
   }
}

template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::weight_statistic()
{
   // Initialize the row_begin of h_cols to determine the starting position of each column in the allocated storage space
   // And assign values to weight_nodes and weight_set
   uint64_t begin = 0;
   uint8_t max_weight = 0;

   for (idx_type col = 0; col < sparse_size; col++)
   {
      uint64_t weight = col_weights[col];
      h_cols.set_row_begin(col, begin);
      begin += weight;

      OKVS::weight_node node(weight, empty_node, empty_node, col);
      weight_nodes[col] = node;

      // auto &node = weight_nodes[col];
      // node.weight = weight;
      // node.next = empty_node;
      // node.prev = empty_node;
      // node.col_idx = col;

      // Use the head insertion method of the linked list to assign weight_set
      auto first_node = weight_set[weight];
      if (first_node != nullptr)
      {
         first_node->prev = node.col_idx;
         weight_nodes[col].next = first_node->col_idx;
      }
      weight_set[weight] = &weight_nodes[col];
      if (weight > max_weight)
         max_weight = weight;
   }
   // Count the maximum weight in the process and use it to reduce the size of weight_set
   // to avoid subsequent unnecessary traversal

   weight_set.resize(max_weight + 1);
   weight_set.shrink_to_fit();
   TAIHANG_ASSERT(begin <= h_sparse.item_num, "OKVS sparse row traversal exceeded allocated rows.");
}

template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::init_hcols()
{
   // Traversing the h_sparse of the row-major order
   // to assign values to the h_cols of the column-major order

   std::unique_ptr<uint8_t[]> col_weights_now(new uint8_t[sparse_size]());
   auto begin_pointer = h_sparse[0];
   for (idx_type row = 0; row < item_num; row++)
   {
      for (auto k = 0; k < sparse_weight; k++, begin_pointer++)
      {
         idx_type col = *begin_pointer;
         TAIHANG_ASSERT(col_weights_now[col] < col_weights[col], "OKVS column weight exceeded expected count.");
         h_cols[col][col_weights_now[col]++] = row;
      }
   }
}

template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::set_Mtx(const Block *keys)
{
   // This function receives a pointer to a key array and uses it to construct h_sparse and h_dense.
   // The functions completed are equivalent to the two functions set_dense and set_sparse

   idx_type i = 0;
   Block *keys_pointer = (Block *)keys;
   Block *dense_pointer = h_dense;
   auto sparse_pointer = h_sparse[0];

   // Traverse the key array, construct h_dense through aes_hash, and then use h_dense to construct h_sparse

   // accelerate performance in groups of 32
   for (; i + 32 <= item_num; i += 32, keys_pointer += 32, dense_pointer += 32, sparse_pointer += sparse_weight << 5)
   {
      set_dense(keys_pointer, 32, dense_pointer);
      set_sparse_32(dense_pointer, sparse_pointer);
   }

   // process the last group (less than 32) that may exist
   if (item_num > i)
      set_dense(keys_pointer, item_num - i, dense_pointer);
   for (; i < item_num; i++, dense_pointer++, sparse_pointer += sparse_weight)
   {
      set_sparse_1(dense_pointer, sparse_pointer);
   }
}

template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::set_keys(const Block *keys)
{
   // This function receives a pointer to a key array, and uses it to construct h_sparse, h_dense and other related variables

   set_Mtx(keys);
   // set_dense(keys);

   // set_sparse();

#ifndef NDEBUG
   // check the correctness of col_weights
   uint64_t sum = 0;
   for (idx_type i = 0; i < sparse_size; i++)
   {
      sum += col_weights[i];
   }
   TAIHANG_ASSERT(sum == item_num * sparse_weight, "OKVS column weight checksum mismatch.");
#endif

   weight_statistic();

   init_hcols();
}

template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::get_FC_1()
{
   std::vector<idx_type> colPermutation(total_size, -1); // map
   uint8_t g = gap_rows.size();
   FC_1.resize(g);
   idx_type delta = triangular_c_rows.size();

   for (auto i = 0; i < g; ++i)
   {
      auto gap_rows_i_0 = gap_rows[i];
      auto gap_rows_i_1 = gap_rows_first_row[i];

      auto h_sparse_gap_0 = h_sparse[gap_rows_i_0];
      auto h_sparse_gap_1 = h_sparse[gap_rows_i_1];
      if (std::memcmp(
              h_sparse_gap_0,
              h_sparse_gap_1,
              sparse_weight * sizeof(idx_type)) == 0)
         FC_1[i].push_back(gap_rows_i_1);
      else
      {
         for (idx_type j = 0; j < delta; j++)
            colPermutation[triangular_c_cols[delta - j - 1]] = j;
         std::set<idx_type, std::greater<idx_type>> F_i;
         auto begin_pointer = h_sparse_gap_0;
         const idx_type *end_pointer = begin_pointer + sparse_weight;
         for (; begin_pointer < end_pointer; begin_pointer++)
         {
            auto col = *begin_pointer;
            if (colPermutation[col] != empty_node)
               F_i.insert(colPermutation[col]);
         }
         while (F_i.size())
         {
            auto col_permuted = *F_i.begin();
            auto row_permuted = col_permuted;
            [[maybe_unused]] auto temp = col_permuted;
            auto row_in_c = triangular_c_rows[delta - row_permuted - 1];
            FC_1[i].push_back(row_in_c);

            begin_pointer = h_sparse[row_in_c];
            end_pointer = begin_pointer + sparse_weight;
            for (; begin_pointer < end_pointer; begin_pointer++)
            {
               auto col = *begin_pointer;

               col_permuted = colPermutation[col];

               if (col_permuted != empty_node)
               {
                  TAIHANG_ASSERT(col_permuted <= temp, "OKVS permutation ordering invariant violated.");
                  auto col_permuted_iter = F_i.find(col_permuted);
                  if (col_permuted_iter != F_i.end())
                     F_i.erase(col_permuted_iter);
                  else
                     F_i.insert(col_permuted);
               }
            }
         }
      }
   }
}

template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::backfill_BlockArrayValue128(value_type *values, value_type *output, prg::Seed *prng)
{
   uint32_t valuetype_len = sizeof(value_type)/sizeof(Block);
   uint8_t g = gap_rows.size();
   if (g && (gap_cols.size() == 0))
   {
      if (g > g_limit)
         throw std::runtime_error("OKVS gap size exceeds g_limit.");
      get_FC_1();

      auto len = prng ? dense_size : g;
      std::vector<value_type> values_(len);

      E_gf128.resize(len, std::vector<Block>(len));

      for (auto i = 0; i < g; i++)
      {
         auto common_ratio_e = h_dense[gap_rows[i]];
         auto E_i_k = common_ratio_e;
         E_gf128[i][0] = E_i_k;
         for (auto k = 1; k < len; k++)
         {
            E_i_k = gf128_mul(E_i_k, common_ratio_e);
            E_gf128[i][k] = E_i_k;
         }

         for (auto col : FC_1[i])
         {
            auto common_ratio_b = h_dense[col];
            auto B_col_k = common_ratio_b;
            E_gf128[i][0] = E_gf128[i][0] ^ B_col_k;
            for (auto k = 1; k < len; k++)
            {
               B_col_k = gf128_mul(B_col_k, common_ratio_b);
               E_gf128[i][k] = E_gf128[i][k] ^ B_col_k;
            }
         }
      }

      if (prng)
      {
         std::vector<Block> random_blocks = prg::gen_random_blocks(*prng, (dense_size - g)*valuetype_len);
         auto iter = 0;
         for (auto i = g; i < dense_size; i++)
         {
            E_gf128[i] = prg::gen_random_blocks(*prng, dense_size);
            values_[i] = ((value_type*)(&random_blocks[0]))[iter++];
         }
         TAIHANG_ASSERT(iter == (dense_size - g), "OKVS GF(2^128) dense tail size mismatch.");
      }

      if (!check_invert_gf128(E_gf128))
      {
         throw std::runtime_error("OKVS GF(2^128) dense matrix is not invertible.");
      }

      for (auto i = 0; i < g; i++)
      {
         values_[i] = values[gap_rows[i]];
         for (auto j : FC_1[i])
         {
            values_[i] ^= values[j];
         }
      }

      for (auto i = 0; i < len; i++)
      {
         auto &output_ = output[sparse_size + i];
         for (auto j = 0; j < len; j++)
         {
            output_ ^= gf128_mul(values_[j], E_gf128[i][j]);
         }
      }
   }
   else if (prng)
   {
      auto random_blocks = prg::gen_random_blocks(*prng, dense_size*valuetype_len);
      for (idx_type i = sparse_size; i < total_size; ++i)
         output[i] = ((value_type*)(&random_blocks[0]))[i - sparse_size];
   }

   value_type temp_value;
   auto delta = triangular_c_rows.size();
   auto iter = delta;

   for (idx_type k = 0; k < delta; k++)
   {
      iter--;
      auto row = triangular_c_rows[iter];
      auto col = triangular_c_cols[iter];
      temp_value = values[row];

      auto row_data = h_sparse[row];
      for (auto j = 0; j < sparse_weight; j++)
      {
         auto cc = row_data[j];    // 6925//26953//31661//45321
         temp_value ^= output[cc]; // 00//00//00output[row_data[1]]
      }

      if (g || prng)
      {
         auto common_ratio_d = h_dense[row];
         auto d_iter_j = common_ratio_d;
         temp_value ^= gf128_mul(output[sparse_size], d_iter_j);
         for (auto j = 1; j < dense_size; j++)
         {
            d_iter_j = gf128_mul(d_iter_j, common_ratio_d);
            temp_value ^= gf128_mul(output[sparse_size + j], d_iter_j);
         }
      }

      output[col] = temp_value;
   }
}
 


template <typename idx_type, DenseType dense_type, typename value_type>
template<typename T>
void OKVS<idx_type, dense_type, value_type>::backfill_gf128(T *values, T *output, prg::Seed *prng)
{
   throw std::runtime_error("Invalid value_type for backfill_gf128");
}

template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::backfill_gf128(Block *values, Block *output, prg::Seed *prng)
{
   uint8_t g = gap_rows.size();
   if (g && (gap_cols.size() == 0))
   {
      if (g > g_limit)
         throw std::runtime_error("OKVS gap size exceeds g_limit.");
      get_FC_1();

      auto len = prng ? dense_size : g;
      std::vector<Block> values_(len);

      E_gf128.resize(len, std::vector<Block>(len));

      //      |dense[r0]^1, dense[r0]^2, ... |
      // E =  |dense[r1]^1, dense[r1]^2, ... |
      //      |dense[r2]^1, dense[r2]^2, ... |
      // EE = E - FC^{-1} * B.
      for (auto i = 0; i < g; i++)
      {
         auto common_ratio_e = h_dense[gap_rows[i]];
         auto E_i_k = common_ratio_e;
         E_gf128[i][0] = E_i_k;
         for (auto k = 1; k < len; k++)
         {
            E_i_k = gf128_mul(E_i_k, common_ratio_e);
            E_gf128[i][k] = E_i_k;
         }

         for (auto col : FC_1[i])
         {
            auto common_ratio_b = h_dense[col];
            auto B_col_k = common_ratio_b;
            E_gf128[i][0] = E_gf128[i][0] ^ B_col_k;
            for (auto k = 1; k < len; k++)
            {
               B_col_k = gf128_mul(B_col_k, common_ratio_b);
               E_gf128[i][k] = E_gf128[i][k] ^ B_col_k;
            }
         }
      }

      if (prng)
      {
         std::vector<Block> random_blocks = prg::gen_random_blocks(*prng, dense_size - g);
         auto iter = 0;
         for (auto i = g; i < dense_size; i++)
         {
            E_gf128[i] = prg::gen_random_blocks(*prng, dense_size);
            values_[i] = random_blocks[iter++];
         }
         TAIHANG_ASSERT(iter == (dense_size - g), "OKVS binary dense tail size mismatch.");
      }

      if (!check_invert_gf128(E_gf128))
      {
         throw std::runtime_error("OKVS GF(2^128) dense matrix is not invertible.");
      }

      // xx = x' - FC^{-1} * x.
      for (auto i = 0; i < g; i++)
      {
         values_[i] = values[gap_rows[i]];
         for (auto j : FC_1[i])
         {
            values_[i] ^= values[j];
         }
      }

      // p' = (E - FC^{-1} * B)^{-1} * (x' - FC^{-1} * x).
      for (auto i = 0; i < len; i++)
      {
         auto &output_ = output[sparse_size + i];
         for (auto j = 0; j < len; j++)
         {
            output_ ^= gf128_mul(values_[j], E_gf128[i][j]);
         }
      }
   }
   else if (prng)
   {
      auto random_blocks = prg::gen_random_blocks(*prng, dense_size);
      for (idx_type i = sparse_size; i < total_size; ++i)
         output[i] = random_blocks[i - sparse_size];
   }

   Block temp_block;
   auto delta = triangular_c_rows.size();
   auto iter = delta;

   for (idx_type k = 0; k < delta; k++)
   {
      iter--;
      auto row = triangular_c_rows[iter];
      auto col = triangular_c_cols[iter];
      temp_block = values[row];

      auto row_data = h_sparse[row];
      for (auto j = 0; j < sparse_weight; j++)
      {
         auto cc = row_data[j];    // 6925//26953//31661//45321
         temp_block ^= output[cc]; // 00//00//00output[row_data[1]]
      }

      if (g || prng)
      {
         auto common_ratio_d = h_dense[row];
         auto d_iter_j = common_ratio_d;
         temp_block ^= gf128_mul(output[sparse_size], d_iter_j);
         for (auto j = 1; j < dense_size; j++)
         {
            d_iter_j = gf128_mul(d_iter_j, common_ratio_d);
            temp_block ^= gf128_mul(output[sparse_size + j], d_iter_j);
         }
      }

      output[col] = temp_block;
   }
}
template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::backfill_binary(value_type *values, value_type *output, prg::Seed *prng)
{
   //==================   backfill  ==================
   // We are solving the system H * P = X. Divide the matrix into
   // H = | A B C; D E F |, where C is a square lower-triangular matrix
   // and E is a dense g*g matrix with the gap rows.

   uint8_t g = gap_rows.size();

   if (g && (gap_cols.size() == 0))
   {
      if (g > g_limit)
         throw std::runtime_error("OKVS gap size exceeds g_limit.");
      // Allocate space for the value corresponding to gap cols
      std::vector<value_type> values_(g, ((value_type *)(&kZeroBlock))[0]);

      // Calculate the auxiliary Mtx F^C^{-1},
      // C represents the triangularized Mtx, F is the gap rows corresponding to the same column
      get_FC_1();

      // The part about dense cols in gap rows
      std::vector<Block> E_Row(g, kZeroBlock);
      // To consider the impact on the rest of the Mtx of the process of emptying F with C, Update E and D
      for (auto i = 0; i < g; i++)
      {
         Block FC_B_i = kZeroBlock;
         for (auto col : FC_1[i])
            FC_B_i = FC_B_i ^ h_dense[col];
         E_Row[i] = h_dense[gap_rows[i]] ^ FC_B_i;
      }
      E_.resize(g, std::vector<uint8_t>(g));

      // get_gap_cols
      // In dense cols, g columns are selected, rows corresponding to the gap are found,
      // and the Mtx of g * g (E') is formed, which is required to be invertible

      // std::vector<std::vector<uint8_t>> col_combinations;
      // auto combination_num = choose_n_k(dense_size, g);
      gap_cols.resize(g);
      for (int i = 0; i < g; ++i)
      {
         gap_cols[i] = i;
      }

      // for (uint64_t i = 0; i < combination_num; i++)
      while (1)
      {
         if (prev_combination(gap_cols, dense_size) == false)
            throw std::runtime_error("OKVS failed to find an invertible binary dense submatrix.");
         // gap_cols = ith_combination(i, dense_size, g);
         for (auto row = 0; row < g; row++)
         {
            if (dense_size > 64)
            {
               auto pointer = (uint8_t *)(&E_Row[row]);
               for (auto j = 0; j < g; j++)
               {
                  auto col = gap_cols[j];
                  auto loc = col >> 3;
                  auto shift = col & 0b111;

                  // E_[row][j] = bitset<50>(block_to_int64(E_Row[row]))[col];
                  E_[row][j] = ((pointer[loc]) >> shift) & 1;
               }
            }
            else
            {
               auto E_row_low64 = block_to_int64(E_Row[row]);
               for (auto j = 0; j < g; j++)
               {
                  auto col = gap_cols[j];
                  E_[row][j] = (E_row_low64 >> col) & 1;
               }
            }
         }
         // Construct E' from the selected column to check its reversibility
         if (check_invert(E_))
            break;
      }

      // Use collection data structures to speed up the find process
      std::unordered_set<uint8_t> gap_cols_set(gap_cols.begin(), gap_cols.end());

      if (prng)
      {
         // Randomize all unknowns for unselected columns
         std::vector<Block> random_blocks(prg::gen_random_blocks(*prng, dense_size - g));
         auto iter = 0;
         for (auto i = sparse_size; i < total_size; i++)
         {
            if (gap_cols_set.find(i - sparse_size) == gap_cols_set.end())
            {
               TAIHANG_ASSERT(i < total_size, "OKVS output index out of bounds.");
               TAIHANG_ASSERT(iter < dense_size - g, "OKVS random dense tail index out of bounds.");
               output[i] = ((value_type *)(&random_blocks[iter++]))[0];
            }
         }
         TAIHANG_ASSERT(iter == (dense_size - g), "OKVS binary dense tail size mismatch.");
      }

      // getX2Prime:Consider the effect of emptying the F process on values
      for (auto i = 0; i < g; i++)
      {
         values_[i] = values[gap_rows[i]];
         auto &v2_i = values_[i];
         for (auto j : FC_1[i])
         {
            v2_i ^= values[j];
         }
      }
      if (prng)
      {
         // So the gap rows is left with E' unknown, so we start solving the equations,
         // and we move the D term, and we update the values
         if (dense_size >= 64)
         {
            auto loc = 0;
            auto i = 0;
            // unroll the loop
            for (; i + 8 < dense_size; i += 8, loc++)
            {
               for (auto shift = 0; shift < 8; shift++)
               {
                  if (gap_cols_set.find(i + shift) == gap_cols_set.end())
                  {
                     auto output_dense_i = output[sparse_size + i + shift];
                     for (auto j = 0; j < g; j++)
                     {
                        auto dense = h_dense[gap_rows[j]];

                        for (auto k : FC_1[j])
                           dense ^= h_dense[k];
                        auto pointer = (uint8_t *)&dense;

                        if ((pointer[loc] >> shift) & 1)
                           values_[j] ^= output_dense_i;
                     }
                  }
               }
            }
            for (auto shift = 0; shift < dense_size - i; shift++)
            {
               if (gap_cols_set.find(i + shift) == gap_cols_set.end())
               {
                  auto output_dense_i = output[sparse_size + i + shift];
                  for (auto j = 0; j < g; j++)
                  {
                     auto dense = h_dense[gap_rows[j]];

                     for (auto k : FC_1[j])
                        dense ^= h_dense[k];
                     auto pointer = (uint8_t *)&dense;
                     if ((pointer[loc] >> shift) & 1)
                        values_[j] ^= output_dense_i;
                  }
               }
            }
         }
         else
         {
            for (auto i = 0; i < dense_size; i++)
            {
               if (gap_cols_set.find(i) == gap_cols_set.end())
               {
                  auto output_dense_i = output[sparse_size + i];
                  for (auto j = 0; j < g; j++)
                  {
                     auto dense = h_dense[gap_rows[j]];

                     for (auto k : FC_1[j])
                        dense ^= h_dense[k];

                     if ((block_to_int64(dense) >> i) & 1)
                        values_[j] ^= output_dense_i;
                  }
               }
            }
         }
      }
      // getEPrime:E_,done

      // Solve for the unknown quantity corresponding to E', by E'^{-1} * V'
      for (auto i = 0; i < g; i++)
      {
         auto &output_ = output[sparse_size + gap_cols[i]];
         for (auto j = 0; j < g; j++)
         {
            if (E_[i][j])
               output_ ^= values_[j];
         }
      }
   }
   else if (prng)
   {
      // g=0 -> the Redundant columns are set to random amounts
      auto random_blocks = prg::gen_random_blocks(*prng, dense_size);
      auto output_pointer = output + sparse_size;
      auto random_pointer = (value_type *)random_blocks.data();
      for (idx_type i = 0; i < dense_size; ++i)
      {
         *output_pointer = *random_pointer;
         output_pointer++;
         random_pointer++;
      }
   }

   // Start to solve the unknown corresponding to the triangular Mtx
   value_type temp_out;
   auto delta = item_num - g;

   for (idx_type iter = delta - 1; iter != idx_type(-1); iter--)
   {
      auto row = triangular_c_rows[iter];
      auto col = triangular_c_cols[iter];
      temp_out = values[row];

      // Begin Solving System of Equations by Gaussian Elimination
      auto row_data = h_sparse[row];
      if (sparse_weight == 3)
      {
         auto col_1 = *row_data++;
         auto col_2 = *row_data++;
         auto col_3 = *row_data;
         auto temp_block_ = output[col_1] ^ output[col_2];
         temp_out ^= output[col_3];
         temp_out ^= temp_block_;
      }
      else
      {
         for (auto j = 0; j < sparse_weight; j++)
         {
            auto cc = row_data[j];
            temp_out ^= output[cc];
         }
         // if (row == 0)
      }
      if (prng)
      {
         if (dense_size > 64)
         {
            auto pointer = (uint8_t *)(h_dense + row);
            auto loc = 0;
            auto j = 0;
            for (; j + 8 < dense_size; j += 8, loc++)
            {
               for (auto shift = 0; shift < 8; shift++)
               {
                  if ((pointer[loc] >> shift) & 1)
                  {
                     temp_out ^= output[sparse_size + j + shift];
                  }
               }
            }
            for (auto shift = 0; shift < dense_size - j; shift++)
            {
               if ((pointer[loc] >> shift) & 1)
               {
                  temp_out ^= output[sparse_size + j + shift];
               }
            }
         }
         else
         {
            auto h_dense_row_low64 = block_to_int64(h_dense[row]);
            for (auto j = 0; j < dense_size; j++)
            {
               if ((h_dense_row_low64 >> j) & 1)
               {
                  temp_out ^= output[sparse_size + j];
               }
            }
         }
      }
      else
      {
         if (dense_size > 64)
         {
            auto pointer = (uint8_t *)(h_dense + row);
            auto output_dense_begin = output + sparse_size;
            for (auto gap_cols_j : gap_cols)
            {
               auto loc = gap_cols_j >> 3;
               auto shift = gap_cols_j & 0b111;
               if ((pointer[loc] >> shift) & 1)
                  temp_out ^= output_dense_begin[gap_cols_j];
            }
         }
         else
         {
            auto h_dense_row_low64 = block_to_int64(h_dense[row]);
            auto output_dense_begin = output + sparse_size;
            for (auto gap_cols_j : gap_cols)
               if ((h_dense_row_low64 >> gap_cols_j) & 1)
                  temp_out ^= output_dense_begin[gap_cols_j];
         }
      }

      output[col] = temp_out;
   }
}

template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::triangulate()
{
   // Complete the second step of encode in the paper: Triangulate

   triangular_c_rows.reserve(item_num);
   triangular_c_cols.reserve(item_num);

   //==================   triangulate  ==================

   // Used to determine which rows have been considered (eliminated)
   std::vector<uint8_t> row_sign(item_num);

   // idx_type ii = 0;
   while (weight_set.size() > 1)
   {
      const idx_type min_idx = find_pop_min_node();

      // "first" determines whether a row is the first row of the currently considered column (column min_idx)
      bool first = true;

      // Traverse the min_idx column to determine all rows

      // for (auto row_idx = 0; row_idx < col_weights[min_idx]; row_idx++)
      auto begin_pointer = h_cols[min_idx];
      const idx_type *end_pointer = begin_pointer + col_weights[min_idx];
      // {
      //    auto row = h_cols[min_idx][row_idx];
      for (; begin_pointer < end_pointer; begin_pointer++)
      {
         const idx_type row = *begin_pointer;
         if (row_sign[row] == 0)
         {
            // This row has not been cleared, traverse all the columns in this row,
            // reduce the weight of the corresponding column node by one,
            // and update the node to the correct position in weight_set
            row_sign[row] = 1;
            auto begin_pointer_ = h_sparse[row];
            const idx_type *end_pointer_ = begin_pointer_ + sparse_weight;
            for (; begin_pointer_ < end_pointer_; begin_pointer_++)
            {
               auto col = *begin_pointer_;
               auto &node = weight_nodes[col];
               if (node.weight)
               {
                  pop(node);
                  node.weight--;
                  push(node);

                  if (node.weight == 1)
                     _mm_prefetch((const char *)h_cols[col], _MM_HINT_T0);
               }
            }

            // If the column is the first row, put the row into the triangular Mtx
            if (first)
            {
               triangular_c_rows.emplace_back(row);
               triangular_c_cols.emplace_back(min_idx);
               first = false;
            }
            else
            {
               const idx_type first_row = triangular_c_rows.back();
               if (is_equal(h_dense[row], h_dense[first_row]))
               {
                  throw std::invalid_argument("OKVS received duplicate keys.");
               }
               else
               {
                  // If it is not the first row, put it into gap rows
                  gap_rows.emplace_back(row);
                  gap_rows_first_row.emplace_back(first_row);
               }
            }
         }
      }
   }
   //    show_time(start, "triangulate over");
}

template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::encode(value_type *values,value_type *output, prg::Seed *prng)
{
   if (triangular_c_rows.size() == 0)
   {
      triangulate();
   }

   if (prng)
   {
      // If the Paxos data should be randomized, then provide a PRNG.
      uint32_t valuetype_len = sizeof(value_type)/sizeof(Block);
      auto single_value_len = valuetype_len ? valuetype_len : 1; 
      // Assign random numbers to the unknowns corresponding to those columns whose weight is 0
      if (weight_set[0] != 0)
      {
         OKVS::weight_node *node_pointer = weight_set[0];
         uint8_t remain_len;
         while (1)
         {
            remain_len = 8;
            std::vector<Block> output_(prg::gen_random_blocks(*prng, 8*single_value_len));
            while (node_pointer != nullptr && remain_len > 0)
            {
               auto col_idx = node_pointer->col_idx;
               output[col_idx] = ((value_type *)(&output_[(remain_len - 1)*single_value_len]))[0];
               if (node_pointer->next == empty_node)
                  node_pointer = nullptr;
               else
               {
                  node_pointer = &weight_nodes[node_pointer->next];
                  remain_len--;
               }
            }
            if (node_pointer == nullptr)
               break;
         }
      }
      else if (weight_0_list.size())
      {
         auto weight_0_size = weight_0_list.size();
         std::vector<Block> output_(prg::gen_random_blocks(*prng, weight_0_size*single_value_len));
         for (idx_type i = 0; i < weight_0_size; i++)
         {
            output[weight_0_list[i]] = ((value_type *)(&output_[i*single_value_len]))[0];
         }
      }
      else
      {
         throw std::runtime_error("OKVS encode requires weight-zero columns for randomized encoding.");
      }
   }
   // Continue to perform the next steps of encode according to the type of dense column
   if (dense_type == DenseType::Binary)
      backfill_binary(values, output, prng);
   else if (dense_type == DenseType::Gf128 && std::is_same<value_type, Block>::value)
      backfill_gf128(values, output, prng);
   else if (dense_type == DenseType::Gf128 && std::is_same<value_type, BlockArrayValue>::value)
      backfill_BlockArrayValue128(values, output, prng);
   

   // return output;
}

template <typename idx_type, DenseType dense_type, typename value_type>
value_type OKVS<idx_type, dense_type, value_type>::decode_1(const Block *key, const std::vector<value_type> &output)
{
   return decode_1(key, output.data());
}

template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::decode_1(const Block *key, const value_type *output, value_type *value, Block *with_dense)
{
   // Decode one input based on the Paxos data structure.
   Block dense;
   Block *dense_pointer;
   // set_dense
   if (!with_dense)
   {
      set_dense(key, 1, &dense);
      dense_pointer = &dense;
   }
   else
   {
      dense_pointer = with_dense;
   }

   std::vector<idx_type> sparse;
   sparse.resize(sparse_weight);
   // set_sparse
   set_sparse_1(dense_pointer, sparse.data());

   *value = output[sparse[0]];
   for (auto i = 1; i < sparse_weight; i++)
   {
      *value ^= output[sparse[i]]; // sparse[0]
   }

   if (dense_type == DenseType::Binary)
   {
      if (dense_size > 64)
      {
         // auto dense_low64 = block_to_int64(dense);
         auto pointer = (uint8_t *)(dense_pointer);
         for (auto i = 0; i < dense_size; i++)
         {
            auto loc = i >> 3;
            auto shift = i & 0b111;
            if ((pointer[loc] >> shift) & 1)
            {
               *value ^= output[sparse_size + i];
            }
         }
      }
      else
      {
         auto dense_low64 = block_to_int64(*dense_pointer);
         for (auto i = 0; i < dense_size; i++)
         {
            if ((dense_low64 >> i) & 1)
               *value ^= output[sparse_size + i];
         }
      }
   }
   else
   {
      const Block common_ratio_d = *dense_pointer;
      Block d_i = *dense_pointer;
      if (dense_size > 0)
         *value ^= gf128_mul(output[sparse_size], d_i);
      for (auto i = 1; i < dense_size; i++)
      {
         d_i = gf128_mul(d_i, common_ratio_d);
         *value ^= gf128_mul(output[sparse_size + i], d_i);
      }
   }
}
template <typename idx_type, DenseType dense_type, typename value_type>
std::vector<value_type> OKVS<idx_type, dense_type, value_type>::encode(const std::vector<value_type> &values, prg::Seed *prng)
{
   std::vector<value_type> output(total_size);
   TAIHANG_ASSERT(values.size() == item_num, "OKVS encode input size mismatch.");
   encode((value_type *)values.data(), (value_type *)output.data(), prng);
   return output;
}
template <typename idx_type, DenseType dense_type, typename value_type>
value_type OKVS<idx_type, dense_type, value_type>::decode_1(const Block *key, const value_type *output)
{
   value_type ans;
   decode_1(key, output, &ans);
   return ans;
}

template <typename idx_type, DenseType dense_type, typename value_type>
std::vector<value_type> OKVS<idx_type, dense_type, value_type>::decode_32(const Block *keys, const value_type *output)
{
   std::vector<value_type> values(32);
   decode_32(keys, output, values.data());
   return values;
}

template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::decode_32(const Block *keys, const value_type *output, value_type *values, Block *with_dense)
{
   // Decode 32 inputs based on the Paxos data structure.
   std::array<Block, 32> dense;
   Block *dense_pointer;
   // set_dense
   if (!with_dense)
   {
      set_dense(keys, 32, dense.data());
      dense_pointer = dense.data();
   }
   else
   {
      dense_pointer = with_dense;
   }
   // std::vector<idx_type> sparse(output.size(),0);
   std::unique_ptr<idx_type[]> sparse(new idx_type[sparse_weight * 32]);
   // set_sparse
   set_sparse_32(dense_pointer, sparse.get());

   for (auto j = 0; j < 4; j++)
   {
      const idx_type *sparse_8 = sparse.get() + j * 8 * sparse_weight;

      auto sparse_0_0 = sparse_8[0 * sparse_weight + 0];
      auto sparse_1_0 = sparse_8[1 * sparse_weight + 0];
      auto sparse_2_0 = sparse_8[2 * sparse_weight + 0];
      auto sparse_3_0 = sparse_8[3 * sparse_weight + 0];
      auto sparse_4_0 = sparse_8[4 * sparse_weight + 0];
      auto sparse_5_0 = sparse_8[5 * sparse_weight + 0];
      auto sparse_6_0 = sparse_8[6 * sparse_weight + 0];
      auto sparse_7_0 = sparse_8[7 * sparse_weight + 0];

      auto values_pointer = values + j * 8;

      auto values_p0 = values_pointer + 0;
      auto values_p1 = values_pointer + 1;
      auto values_p2 = values_pointer + 2;
      auto values_p3 = values_pointer + 3;
      auto values_p4 = values_pointer + 4;
      auto values_p5 = values_pointer + 5;
      auto values_p6 = values_pointer + 6;
      auto values_p7 = values_pointer + 7;

      *values_p0 = output[sparse_0_0];
      *values_p1 = output[sparse_1_0];
      *values_p2 = output[sparse_2_0];
      *values_p3 = output[sparse_3_0];
      *values_p4 = output[sparse_4_0];
      *values_p5 = output[sparse_5_0];
      *values_p6 = output[sparse_6_0];
      *values_p7 = output[sparse_7_0];
   }

   for (auto j = 1; j < sparse_weight; j++)
   {
      for (auto k = 0; k < 4; k++)
      {
         const idx_type *sparse_8 = sparse.get() + k * 8 * sparse_weight;
         auto values_8 = values + k * 8;

         auto sparse_0_j = sparse_8[sparse_weight * 0 + j];
         auto sparse_1_j = sparse_8[sparse_weight * 1 + j];
         auto sparse_2_j = sparse_8[sparse_weight * 2 + j];
         auto sparse_3_j = sparse_8[sparse_weight * 3 + j];
         auto sparse_4_j = sparse_8[sparse_weight * 4 + j];
         auto sparse_5_j = sparse_8[sparse_weight * 5 + j];
         auto sparse_6_j = sparse_8[sparse_weight * 6 + j];
         auto sparse_7_j = sparse_8[sparse_weight * 7 + j];

         auto values_p0 = values_8 + 0;
         auto values_p1 = values_8 + 1;
         auto values_p2 = values_8 + 2;
         auto values_p3 = values_8 + 3;
         auto values_p4 = values_8 + 4;
         auto values_p5 = values_8 + 5;
         auto values_p6 = values_8 + 6;
         auto values_p7 = values_8 + 7;

         *values_p0 ^= output[sparse_0_j];
         *values_p1 ^= output[sparse_1_j];
         *values_p2 ^= output[sparse_2_j];
         *values_p3 ^= output[sparse_3_j];
         *values_p4 ^= output[sparse_4_j];
         *values_p5 ^= output[sparse_5_j];
         *values_p6 ^= output[sparse_6_j];
         *values_p7 ^= output[sparse_7_j];
      }
   }
   if (dense_type == DenseType::Binary)
   {
      if (dense_size > 64)
      {
         for (auto k = 0; k < 4; k++)
         {
            auto values_8 = values + k * 8;
            const Block *dense_8 = dense_pointer + k * 8;

            auto dense_0 = (uint8_t *)dense_8;
            auto dense_1 = (uint8_t *)(dense_8 + 1);
            auto dense_2 = (uint8_t *)(dense_8 + 2);
            auto dense_3 = (uint8_t *)(dense_8 + 3);
            auto dense_4 = (uint8_t *)(dense_8 + 4);
            auto dense_5 = (uint8_t *)(dense_8 + 5);
            auto dense_6 = (uint8_t *)(dense_8 + 6);
            auto dense_7 = (uint8_t *)(dense_8 + 7);

            for (auto i = 0; i < dense_size; i++)
            {
               auto loc = i >> 3;
               auto shift = i & 0b111;
               auto flag_0 = (dense_0[loc] >> shift) & 1;
               auto flag_1 = (dense_1[loc] >> shift) & 1;
               auto flag_2 = (dense_2[loc] >> shift) & 1;
               auto flag_3 = (dense_3[loc] >> shift) & 1;
               auto flag_4 = (dense_4[loc] >> shift) & 1;
               auto flag_5 = (dense_5[loc] >> shift) & 1;
               auto flag_6 = (dense_6[loc] >> shift) & 1;
               auto flag_7 = (dense_7[loc] >> shift) & 1;

               auto values_p0 = values_8 + 0;
               auto values_p1 = values_8 + 1;
               auto values_p2 = values_8 + 2;
               auto values_p3 = values_8 + 3;
               auto values_p4 = values_8 + 4;
               auto values_p5 = values_8 + 5;
               auto values_p6 = values_8 + 6;
               auto values_p7 = values_8 + 7;

               auto output_gap_i = output[sparse_size + i];

               *values_p0 ^= flag_0?output_gap_i:value_type();
               *values_p1 ^= flag_1?output_gap_i:value_type();
               *values_p2 ^= flag_2?output_gap_i:value_type();
               *values_p3 ^= flag_3?output_gap_i:value_type();
               *values_p4 ^= flag_4?output_gap_i:value_type();
               *values_p5 ^= flag_5?output_gap_i:value_type();
               *values_p6 ^= flag_6?output_gap_i:value_type();
               *values_p7 ^= flag_7?output_gap_i:value_type();
            }
         }
      }
      else
      {

         for (auto k = 0; k < 4; k++)
         {
            auto values_8 = values + k * 8;
            const Block *dense_8 = dense_pointer + k * 8;

            auto dense_0 = block_to_int64(dense_8[0]);
            auto dense_1 = block_to_int64(dense_8[1]);
            auto dense_2 = block_to_int64(dense_8[2]);
            auto dense_3 = block_to_int64(dense_8[3]);
            auto dense_4 = block_to_int64(dense_8[4]);
            auto dense_5 = block_to_int64(dense_8[5]);
            auto dense_6 = block_to_int64(dense_8[6]);
            auto dense_7 = block_to_int64(dense_8[7]);

            for (auto i = 0; i < dense_size; i++)
            {

               auto flag_0 = dense_0 & 1;
               auto flag_1 = dense_1 & 1;
               auto flag_2 = dense_2 & 1;
               auto flag_3 = dense_3 & 1;
               auto flag_4 = dense_4 & 1;
               auto flag_5 = dense_5 & 1;
               auto flag_6 = dense_6 & 1;
               auto flag_7 = dense_7 & 1;

               auto values_p0 = values_8 + 0;
               auto values_p1 = values_8 + 1;
               auto values_p2 = values_8 + 2;
               auto values_p3 = values_8 + 3;
               auto values_p4 = values_8 + 4;
               auto values_p5 = values_8 + 5;
               auto values_p6 = values_8 + 6;
               auto values_p7 = values_8 + 7;

               auto output_gap_i = output[sparse_size + i];

               *values_p0 ^= flag_0?output_gap_i:value_type();
               *values_p1 ^= flag_1?output_gap_i:value_type();
               *values_p2 ^= flag_2?output_gap_i:value_type();
               *values_p3 ^= flag_3?output_gap_i:value_type();
               *values_p4 ^= flag_4?output_gap_i:value_type();
               *values_p5 ^= flag_5?output_gap_i:value_type();
               *values_p6 ^= flag_6?output_gap_i:value_type();
               *values_p7 ^= flag_7?output_gap_i:value_type();

               dense_0 >>= 1;
               dense_1 >>= 1;
               dense_2 >>= 1;
               dense_3 >>= 1;
               dense_4 >>= 1;
               dense_5 >>= 1;
               dense_6 >>= 1;
               dense_7 >>= 1;
            }
         }
      }
   }
   else if (dense_type == DenseType::Gf128)
   {

      if (dense_size > 0)
      {
         for (auto k = 0; k < 4; k++)
         {
            auto values_8 = values + k * 8;
            auto common_ratio_d_8 = dense_pointer + k * 8;

            auto values_p0 = values_8 + 0;
            auto values_p1 = values_8 + 1;
            auto values_p2 = values_8 + 2;
            auto values_p3 = values_8 + 3;
            auto values_p4 = values_8 + 4;
            auto values_p5 = values_8 + 5;
            auto values_p6 = values_8 + 6;
            auto values_p7 = values_8 + 7;
            auto output_dense_0 = output[sparse_size];

            *values_p0 ^= gf128_mul(output_dense_0, common_ratio_d_8[0]);
            *values_p1 ^= gf128_mul(output_dense_0, common_ratio_d_8[1]);
            *values_p2 ^= gf128_mul(output_dense_0, common_ratio_d_8[2]);
            *values_p3 ^= gf128_mul(output_dense_0, common_ratio_d_8[3]);
            *values_p4 ^= gf128_mul(output_dense_0, common_ratio_d_8[4]);
            *values_p5 ^= gf128_mul(output_dense_0, common_ratio_d_8[5]);
            *values_p6 ^= gf128_mul(output_dense_0, common_ratio_d_8[6]);
            *values_p7 ^= gf128_mul(output_dense_0, common_ratio_d_8[7]);
         }
      }

      std::array<Block, 32> d_i_32;
      memcpy(d_i_32.data(), dense_pointer, sizeof(Block) * 32);

      auto iter = sparse_size + 1;
      for (auto i = 1; i < dense_size; i++, iter++)
      {
         auto output_dense_i = output[iter];
         for (auto k = 0; k < 4; k++)
         {
            auto values_8 = values + k * 8;
            auto common_ratio_d_8 = dense_pointer + k * 8;
            auto d_i = d_i_32.data() + k * 8;

            d_i[0] = gf128_mul(d_i[0], common_ratio_d_8[0]);
            d_i[1] = gf128_mul(d_i[1], common_ratio_d_8[1]);
            d_i[2] = gf128_mul(d_i[2], common_ratio_d_8[2]);
            d_i[3] = gf128_mul(d_i[3], common_ratio_d_8[3]);
            d_i[4] = gf128_mul(d_i[4], common_ratio_d_8[4]);
            d_i[5] = gf128_mul(d_i[5], common_ratio_d_8[5]);
            d_i[6] = gf128_mul(d_i[6], common_ratio_d_8[6]);
            d_i[7] = gf128_mul(d_i[7], common_ratio_d_8[7]);

            auto values_p0 = values_8 + 0;
            auto values_p1 = values_8 + 1;
            auto values_p2 = values_8 + 2;
            auto values_p3 = values_8 + 3;
            auto values_p4 = values_8 + 4;
            auto values_p5 = values_8 + 5;
            auto values_p6 = values_8 + 6;
            auto values_p7 = values_8 + 7;

            *values_p0 ^= gf128_mul(output_dense_i, d_i[0]);
            *values_p1 ^= gf128_mul(output_dense_i, d_i[1]);
            *values_p2 ^= gf128_mul(output_dense_i, d_i[2]);
            *values_p3 ^= gf128_mul(output_dense_i, d_i[3]);
            *values_p4 ^= gf128_mul(output_dense_i, d_i[4]);
            *values_p5 ^= gf128_mul(output_dense_i, d_i[5]);
            *values_p6 ^= gf128_mul(output_dense_i, d_i[6]);
            *values_p7 ^= gf128_mul(output_dense_i, d_i[7]);
         }
      }
   }
}

template <typename idx_type, DenseType dense_type, typename value_type>
void OKVS<idx_type, dense_type, value_type>::decode(const Block *keys, const idx_type key_num, const value_type *output, value_type *values, Block *with_dense)
{
   // Decode the given input vector and write the result to values.
   if (sparse_size == 0)
   {
      throw std::runtime_error("OKVS must be initialized before decode.");
   }
   is_decoding = true;
   idx_type i = 0;
   Block *key_iter = (Block *)keys;
   value_type *value_iter = values;
   for (; i + 32 <= key_num; i += 32)
   {
      decode_32(key_iter, output, value_iter, with_dense);
      if (with_dense)
      {
         with_dense += 32;
      }
      else
      {
         key_iter += 32;
      }
      value_iter += 32;
   }
   for (; i < key_num; i++)
   {
      // if(i==28414)
      decode_1(key_iter, output, value_iter, with_dense);
      if (with_dense)
      {
         with_dense++;
      }
      else
      {
         key_iter++;
      }
      value_iter++;
   }
   is_decoding = false;
}

template <typename idx_type, DenseType dense_type, typename value_type>
std::vector<value_type> OKVS<idx_type, dense_type, value_type>::decode(const std::vector<Block> &keys, const std::vector<value_type> &output, Block *with_dense)
{
   auto key_num = keys.size();
   std::vector<value_type> values(key_num);
   decode(keys.data(), key_num, output.data(), values.data(), with_dense);
   return values;
}

template class OKVS<uint8_t, DenseType::Binary, Block>;
template class OKVS<uint16_t, DenseType::Binary, Block>;
template class OKVS<uint32_t, DenseType::Binary, Block>;
template class OKVS<uint64_t, DenseType::Binary, Block>;
template class OKVS<uint8_t, DenseType::Gf128, Block>;
template class OKVS<uint16_t, DenseType::Gf128, Block>;
template class OKVS<uint32_t, DenseType::Gf128, Block>;
template class OKVS<uint64_t, DenseType::Gf128, Block>;

} // namespace taihang::mpc::okvs
