/*
 * Code for Fast k-Nearest Neighbour Search via Prioritized DCI
 * 
 * This code implements the method described in the Prioritized DCI paper, 
 * which can be found at https://arxiv.org/abs/1703.00440
 * 
 * This file is a part of the Dynamic Continuous Indexing reference 
 * implementation.
 * 
 * 
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. 
 * 
 * Copyright (C) 2017    Ke Li
 */

#ifndef DCI_H
#define DCI_H

// If this is a C++ compiler, use C linkage
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <Python.h>
#include "btree_p.h"
#include "hashtable_p.h"
#include "stack.h"
#include "bf16_util.h"

typedef struct dci{
    int dim;                        // (Ambient) dimensionality of data
    int num_comp_indices;           // Number of composite cell_indices
    int num_simp_indices;           // Number of simple cell_indices in each composite index
    int num_points;
    int num_levels;
    long long next_point_id;
    additional_info* root; // this contains the root of our tree, i.e., the additional_info containing the cell_indices of the highest level
    bf16_t* proj_vec;              // BF16: Assuming column-major layout, matrix of size dim x (num_comp_indices*num_simp_indices)
    bf16_t* add_proj_vec;          // BF16: projection for transformation
    int* num_points_on_level;  // Number of points of each level
    additional_info*** points_on_level;  // point address of each level
    int* max_num_on_level;  // Maximum number of points of each level
    float promotion_prob;   // Probability of promoting a point to the next level
    float promotion_prob_subseq;  // Probability of promoting a point to the subsequent level
    hashtable_p* inserted_points;
    int* token2nodeIndex;  // token id to leaf node index
    int* token2nodeOffset;  // token id to leaf node offset
    // btree_p_leaf_node** nodeIndex2Address;  // leaf node index to node address
    int num_leaf_nodes;
    int next_target_level;  // next target level for insertion
    int max_volume;
    float max_sq_norm;
    float* sq_norm_list;
    bool transform;
    int parallel_level;
    int inner_threads;  // Number of threads for inner parallel loops (level 2)
    int inner_inner_threads;  // Number of threads for innermost parallel loops (level 3)
    int numa_threshold;  // Threshold for switching to NUMA-optimized threading
    additional_info** sub_root_list;
    int max_leaves;
    Stack* stack;  // stack for memory management
    bool *page_status;  // page status -- 1: update, 0: not update
    bool update_addr;  // update address for the tokens in leaves or not
    btree_p_leaf_node **leaf_list;  // leaf list (leave id -> leaf address)
    bool debug;
} dci;

// Setting num_to_retrieve and prop_to_retrieve has no effect when blind is true
// Setting field_of_view has no effect when there is only one level
// min_num_finest_level_points is for internal use only; setting it has no effect (since it will be overwritten)
typedef struct dci_query_config {
    bool blind;
    // Querying algorithm terminates whenever we have visited max(num_visited, prop_visited*num_points) points or retrieved max(num_retrieved, prop_retrieved*num_points) points, whichever happens first
    int num_to_visit;
    int num_to_retrieve;
    float prop_to_visit;
    float prop_to_retrieve;
    int field_of_view;
    int min_num_finest_level_points;
    int target_level;
} dci_query_config;

void data_projection(int num_indices, dci* const dci_inst, const int dim, const int num_points, const bf16_t* const data, float** p_data_proj, const bool* mask, bool pre_computed);

void dci_init(dci* const dci_inst, const int dim, const int num_comp_indices, const int num_simp_indices, float promotion_prob, float promotion_prob_subseq, int max_volume, bool transform, int parallel_level, bool debug, bf16_t* proj_vec);

// Note: the data itself is not kept in the index and must be kept in-place
long long dci_add(dci* const dci_inst, const int dim, const int num_points, const bf16_t* const data, const bf16_t* const value, const int num_levels, dci_query_config construction_query_config, const long long* const data_ids, int target_level, float* new_data_proj, bool* mask, bool random, int interval, int X, float anchor_threshold);

int dci_delete(dci*const dci_inst, const int num_points, const long long *const data_ids, dci_query_config deletion_config, long long* duplicate_delete_ids);

// CAUTION: This function allocates memory for each nearest_neighbours[j], nearest_neighbour_dists[j], so we need to deallocate them outside of this function!
void dci_query(dci* const dci_inst, const int dim, const int num_queries, const bf16_t* const query, int num_neighbours, dci_query_config query_config, bool* mask, int** const nearest_neighbours, float** const nearest_neighbour_dists, int* const num_returned);

void dci_clear(dci* const dci_inst);

void free_instance(additional_info *root, int num_levels, int num_indices, dci* dci_inst);

// Clear cell_indices and reset the projection directions
void dci_reset(dci* const dci_inst);

void dci_free(dci* const dci_inst);

long long dci_add_query(dci* const dci_inst, const int dim, const int num_points, const bf16_t* const data, const bf16_t* const value, const int num_levels,
    dci_query_config construction_query_config, const long long* const data_ids, int target_level, float* new_data_proj, bool* add_mask,
    const int num_queries, const bf16_t* const query, int num_neighbours, dci_query_config query_config, bool* delete_mask,
    int** const nearest_neighbours, float** const nearest_neighbour_dists, int* const num_returned, bool random, int interval, int X, float anchor_threshold);

void dci_address_update(dci* const dci_inst, int* indices, PyObject* new_address, int update_num, int offset);

// Helper functions
void pack_points_to_matrix(additional_info** upper_cells, int num_points, int dim, bf16_t* points_matrix);
void compute_points_sq_norms(const bf16_t* points_matrix, int num_points, int dim, float* sq_norms);

#ifdef __cplusplus
}
#endif

#endif // DCI_H
