/****************************
 * @file      paxos.hpp
 * @brief     Low-level Paxos OKVS engine.
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

#ifndef TAIHANG_MPC_OKVS_PAXOS_HPP
#define TAIHANG_MPC_OKVS_PAXOS_HPP

#include <taihang/common/check.hpp>
#include <taihang/mpc/okvs/okvs.hpp>
#include <taihang/crypto/prg.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <list>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace taihang::mpc::okvs {

// A generic matrix class template that stores a matrix of values of type T.
// It provides methods for resizing the matrix, setting the values of its rows,
// and accessing elements of the matrix using the [] operator.

template <typename T>
class Mtx
{
   uint64_t mRow = 0;                               // The number of rows in the matrix
   uint64_t mCol = 0;                               // The number of columns in the matrix
   std::unique_ptr<T[]> allocate;                   // A unique pointer to the matrix data
   std::unique_ptr<uint64_t[]> row_begin = nullptr; // A unique pointer to the beginning of each row

public:
   uint64_t item_num;  // The total number of elements in the matrix
   T *mView = nullptr; // A pointer to the matrix data

   Mtx() = default; // Default constructor

   // Resizes the matrix to have the specified number of rows and columns, and allocates
   // memory to store the matrix data. If a storage pointer is provided, the matrix data
   // is set to point to the storage.
   inline void resize(const uint8_t *storage = nullptr, uint64_t rows = 0, uint64_t cols = 0)
   {
      mRow = rows;
      mCol = cols;
      item_num = rows * cols;
      if (storage == nullptr)
      {
         allocate.reset(new T[item_num]()); // Allocate memory for the matrix data
         mView = allocate.get();            // Set the pointer to point to the matrix data
      }
      else
         mView = (T *)storage;
      if (rows && !cols)
         row_begin.reset(new uint64_t[rows]); // Allocate memory for the beginning of each row
   }

   // Resizes the matrix to have the specified row weights, where each element of
   // row_weights represents the number of elements in the corresponding row.
   inline void resize_row(uint64_t *row_weights)
   {
      uint64_t begin = 0;
      for (auto i = 0; i < mRow; i++)
      {
         row_begin[i] = begin;    // Set the beginning of the current row
         begin += row_weights[i]; // Calculate the beginning of the next row
      }
   }

   // Sets the beginning of the specified row to the specified value.
   inline void set_row_begin(uint64_t row, uint64_t begin)
   {
      TAIHANG_ASSERT(row_begin != nullptr, "OKVS matrix row storage is not allocated.");
      TAIHANG_ASSERT(row < mRow, "OKVS matrix row index out of bounds.");
      if (mCol)
         TAIHANG_ASSERT(begin < mRow * mCol, "OKVS matrix row offset out of bounds.");
      row_begin[row] = begin;
   }

   // Returns a pointer to the specified row of the matrix.
   T *operator[](uint64_t row_num)
   {
      if (mCol != 0)
      {
         return mView + row_num * mCol;
      }
      else
      {
         return mView + row_begin[row_num];
      }
   }
};

// The core Paxos algorithm. idx_type should be large enough to fit the Paxos size value.
template <typename idx_type = uint64_t, DenseType dense_type = DenseType::Binary, typename value_type = Block>
class OKVS
{
public:
   // The number of key-value pairs in the store.
   idx_type item_num;

   // statistical security parameters
   uint8_t statistical_security_parameter;

   // The upper bound of g
   uint8_t g_limit;

   // The number of columns in the sparse part of the store.
   idx_type sparse_size = 0;

   // The number of columns in the dense part of the store.
   uint8_t dense_size;

   // The number of columns of the store.
   uint64_t total_size;

   // provide randomness to set_dense(set_Mtx)
   prg::Seed seed;

   // A matrix that stores the rows corresponding to each valid bit of each column in the sparse part of the store.
   Mtx<idx_type> h_cols;

   // The value used to represent an empty weight_node.
   const idx_type empty_node = idx_type(-1);

   // This struct defines a custom linked list node used to construct weight_nodes and weight_set.
   // Each node corresponds to a particular column in the matrix and contains information about its weight and its neighboring columns.
   struct weight_node
   {
      // The weight of the column corresponding to this node.
      uint8_t weight;
      // The index of the next node in the linked list.
      idx_type next;
      // The index of the previous node in the linked list.
      idx_type prev;
      // The index of the column in the matrix that this node corresponds to.
      idx_type col_idx;
      // Constructor for creating a new node with the specified weight, next and previous node indices, and column index.
      weight_node(const uint8_t w, idx_type n, idx_type p, idx_type c)
      {
         weight = w;
         next = n;
         prev = p;
         col_idx = c;
      }
      weight_node() {}
   };

   // Pointer to an array of weight_node objects for each column
   std::unique_ptr<weight_node[]> weight_nodes;
   // an array of linked lists composed of different weight columns
   std::vector<weight_node *> weight_set;
   // an array of nodes with weight 0.
   std::vector<idx_type> weight_0_list;

   // Pop operation for linked list(weight_set[node.weight]);
   inline void pop(weight_node &node)
   {
      if (node.prev == empty_node)
      {
         auto &weight_set_at_weight = weight_set[node.weight];

         TAIHANG_ASSERT(weight_set_at_weight == &node, "OKVS weight-set head mismatch.");
         if (node.next == empty_node)
         {
            weight_set_at_weight = nullptr;
            while (weight_set.back() == nullptr)
               weight_set.pop_back();
         }
         else
         {
            weight_set_at_weight = &weight_nodes[node.next];
            weight_set_at_weight->prev = empty_node;
         }
      }
      else
      {
         auto &prev = weight_nodes[node.prev];

         if (node.next == empty_node)
         {
            prev.next = empty_node;
         }
         else
         {
            auto &next = weight_nodes[node.next];
            prev.next = next.col_idx;
            next.prev = prev.col_idx;
         }
      }

      node.prev = empty_node;
      node.next = empty_node;
   }

   // Push operation for linked list(weight_set[node.weight]);
   inline void push(weight_node &node)
   {
      TAIHANG_ASSERT(node.next == empty_node, "OKVS weight node next pointer is not empty.");
      TAIHANG_ASSERT(node.prev == empty_node, "OKVS weight node previous pointer is not empty.");

      if (weight_set.size() <= node.weight)
      {
         weight_set.resize(node.weight + 1, nullptr);
      }

      if (weight_set[node.weight] == nullptr)
      {
         weight_set[node.weight] = &node;
      }
      else
      {
         TAIHANG_ASSERT(weight_set[node.weight]->prev == empty_node,
                        "OKVS weight-set head previous pointer is not empty.");
         weight_set[node.weight]->prev = node.col_idx;
         node.next = weight_set[node.weight]->col_idx;
         weight_set[node.weight] = &node;
      }
   }

   // Find the head node of the linked list with the minimum non-zero weight in the weight_set, which contains multiple linked lists, and pop it out.
   inline idx_type find_pop_min_node()
   {
      for (uint8_t w = 1; w < weight_set.size(); w++)
      {
         OKVS::weight_node *first_node_pointer = weight_set[w];
         if (first_node_pointer)
         {
            auto &min_node = *first_node_pointer;
            pop(min_node);
            min_node.weight = 0;
            return min_node.col_idx;
         }
      }
      return 0;
   }

   // two auxiliary variables used for quick modulo calculation using round-up division.
   std::vector<Divider> mMods;     // variable for round-up division
   std::vector<idx_type> mModVals; // the modulus for modulo calculation

   std::unique_ptr<uint8_t[]> storage; // Memory Pool

   // The correspondence between the rows and columns of a triangular matrix and the original matrix.
   // 'triangular[i]=idx' means that the i-th row (or column) of the triangular matrix corresponds to the idx-th row (or column) of the original matrix.
   std::vector<idx_type> triangular_c_rows, triangular_c_cols;

   // The rows left over after approximating a matrix into a triangular form, also known as gap rows.
   std::vector<idx_type> gap_rows;
   // The first row of the corresponding column when a gap row is selected during the process of approximating a matrix into a triangular form.
   std::vector<idx_type> gap_rows_first_row;
   // The column that corresponds to the selected gap row.
   std::vector<uint8_t> gap_cols;

   // The weight of each column
   // std::vector<idx_type>col_weights;
   idx_type *col_weights;

   // The Hamming weight of each row in the sparse part.
   uint8_t sparse_weight;

   // The sparse matrix corresponding to the set of keys.
   // Each row contains 'sparse_weight' indices representing the source positions of the Hamming weight.
   Mtx<idx_type> h_sparse;
   //  std::vector<Block> h_dense;

   // The dense matrix corresponding to the set of keys.
   Block *h_dense;

   /*
      H=[A B C
         D E F]
      C is a triangular matrix.
   */
   std::vector<std::list<idx_type>> FC_1;   // F*C^{-1}
   std::vector<std::vector<uint8_t>> E_;    // E^{-1} for binary
   std::vector<std::vector<Block>> E_gf128; // E^{-1} for gf_128

   bool is_decoding = false;

   OKVS() = default;
   OKVS(const idx_type item_num, const uint8_t sparse_weight = 3, const uint8_t statistical_security_parameter = 40, const prg::Seed *seed = nullptr);

   // Calculate the number of columns in a sparse matrix based on the size of the set of keys,
   // the weight of the sparse vector, and statistical security parameters.
   void calculate_sparse_size();

   // Allocate space for the memory pool.
   void allocate();

   // Calculate the corresponding matrix based on the input set of keys and compute its weight.
   void set_keys(const Block *keys);

   // Calculate the corresponding matrix
   void set_Mtx(const Block *keys);

   // Calculate the corresponding dense matrix
   void set_dense(const Block *keys, idx_type n = 0, Block *dest = nullptr);

   // Calculate the corresponding sparse vector for a key
   void set_sparse_1(const Block *dense, idx_type *sparse);
   void set_sparse_1(const idx_type row);

   // Calculate corresponding sparse vectors for 32 keys
   void set_sparse_32(const Block *dense, idx_type *sparse);
   void set_sparse_32(const idx_type row);

   // Calculate the corresponding sparse matrix
   void set_sparse();

   // Compute the weight of each column
   void weight_statistic();

   // Initialize a column-major matrix
   void init_hcols();

   // Approximate the matrix into a triangular form
   void triangulate();

   // Encode the given value set based on the already set input.
   // The Paxos data structure is written to output.
   std::vector<value_type> encode(const std::vector<value_type> &values, prg::Seed *prng = nullptr);
   void encode(value_type *values, value_type *output, prg::Seed *prng = nullptr);

   // Decode for a key.
   value_type decode_1(const Block *key, const std::vector<value_type> &output);
   value_type decode_1(const Block *key, const value_type *output);
   void decode_1(const Block *key, const value_type *output, value_type *value, Block *with_dense = nullptr);

   // Decode for 32 keys.
   std::vector<value_type> decode_32(const Block *keys, const value_type *output);
   void decode_32(const Block *keys, const value_type *output, value_type *values, Block *with_dense = nullptr);
   std::vector<value_type> decode(const std::vector<Block> &keys, const std::vector<value_type> &output, Block *with_dense = nullptr);

   // Decode the given inputs based on the Paxos data structure.
   // The output is written to values.
   void decode(const Block *keys, const idx_type key_num, const value_type *output, value_type *values, Block *with_dense = nullptr);

   // A fast method for performing modulo 32.
   void mod32(uint64_t *vals, const uint64_t modIdx)
   {
      auto div = &mMods[modIdx];
      auto modVal = mModVals[modIdx];
      reduce_mod32(vals, div, modVal);
   }

   // Calculate F*C^{-1}
   void get_FC_1();

   // The backfill algorithm for a binary OKVS.
   void backfill_binary(value_type *values, value_type *output, prg::Seed *prng);
   // The backfill algorithm for a OKVS whose dense_type is gf_128.
   template<typename T>
   void backfill_gf128(T *values, T *output, prg::Seed *prng);

   void backfill_gf128(Block *values, Block *output, prg::Seed *prng);
   
   // The backfill algorithm for a OKVS whose dense_type is gf_128 and value_type is BlockArrayValue(Block[]).
   void backfill_BlockArrayValue128(value_type *values, value_type *output, prg::Seed *prng);
};

extern template class OKVS<uint8_t, DenseType::Binary, Block>;
extern template class OKVS<uint16_t, DenseType::Binary, Block>;
extern template class OKVS<uint32_t, DenseType::Binary, Block>;
extern template class OKVS<uint64_t, DenseType::Binary, Block>;
extern template class OKVS<uint8_t, DenseType::Gf128, Block>;
extern template class OKVS<uint16_t, DenseType::Gf128, Block>;
extern template class OKVS<uint32_t, DenseType::Gf128, Block>;
extern template class OKVS<uint64_t, DenseType::Gf128, Block>;

} // namespace taihang::mpc::okvs

#endif // TAIHANG_MPC_OKVS_PAXOS_HPP
