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

#ifndef BTREE_P_H
#define BTREE_P_H
#include <omp.h> 

// If this is a C++ compiler, use C linkage
#ifdef __cplusplus
extern "C" {
#endif

#include "btree_common.h"
#include "bf16_util.h"

typedef struct data_pt data_pt;
typedef struct bulk_data_pt bulk_data_pt;
typedef struct additional_info additional_info;
typedef struct idx_arr idx_arr;
typedef struct Stack Stack;

typedef struct btree_p_inner_node {
    // Level in the B+ tree, if level == 0 -> leaf node
    unsigned short level;

    // Number of keys used, so number of valid children or data
    // pointers
    unsigned short num_slots_used;

    // Keys of children or data pointers
    float* slot_keys;

    // Pointers to children
    void** slot_ptrs;

} btree_p_inner_node;

typedef struct btree_p_leaf_node {

    // Level in the B+ tree, if level == 0 -> leaf node
    unsigned short level;

    // Number of keys used, so number of valid children or data
    // pointers
    unsigned short num_slots_used;

    // float linked list pointers to traverse the leaves
    struct btree_p_leaf_node *prev_leaf;

    // float linked list pointers to traverse the leaves
    struct btree_p_leaf_node *next_leaf;

    // Keys of children or data pointers
    float* slot_keys;

    // Array of data
    data_pt* slot_data;

    // ID of the node
    int id;

    // Address of the space for the saved data (BF16 format)
    bf16_t* data_loc;

    // Address of the incidential data (BF16 format)
    bf16_t* inc_data_loc;

    // Lower Bound 1
    float lower_bound1;

    // Lower Bound 2
    float lower_bound2;

} btree_p_leaf_node;

typedef struct btree_p {
    // number of data currently in the tree
    unsigned int num_data;

    // Base B+ tree parameter: The number of key/data slots in each leaf
    unsigned short leaf_max_num_slots;

    // Base B+ tree parameter: The number of key slots in each inner node,
    // this can differ from slots in each leaf.
    unsigned short inner_max_num_slots;

    // Computed B+ tree parameter: The minimum number of key/data slots used
    // in a leaf. If fewer slots are used, the leaf will be merged or slots
    // shifted from it's siblings.
    unsigned short leaf_min_num_slots;

    // Computed B+ tree parameter: The minimum number of key slots used
    // in an inner node. If fewer slots are used, the inner node will be
    // merged or slots shifted from it's siblings.
    unsigned short inner_min_num_slots;
    // Pointer to the B+ tree's root node, either leaf or inner node
    void* root;

    // Pointer to first leaf in the float linked leaf chain
    btree_p_leaf_node *first_leaf;

    // Pointer to last leaf in the float linked leaf chain
    btree_p_leaf_node *last_leaf;

    // Number of leaf nodes in the B+ tree
    unsigned int num_leaf_nodes;
} btree_p;

struct additional_info {
    long long id;
    int local_id;
    float* local_dist;
    int* num_finest_level_points;
    int* num_finest_level_nodes;
    bf16_t* data_loc;        // BF16 format: actual data vectors
    bf16_t* inc_data_loc;    // BF16 format: incidental data
    float max_child_dist;
    float parent_dist;  // distance to the parent
    additional_info* parent_info;  // parent's additional_info, useful for deletion
    btree_p* cell_indices;  // indices of the cell (points in the lower level which this point is the parent of)
    idx_arr* arr_indices;
    bool flag;  // if arr_indices needs to update or not
    float max_sq_norm; // maximum square norm applied to the children in this cell
    omp_lock_t lock;  // lock for multi-thread
};


struct data_pt {
    additional_info* info;
};

struct idx_arr {
    float key;
    int local_id;
    additional_info* info;
};

struct bulk_data_pt {
    data_pt data_pt;
    float local_dist;  // local_distance to the parent  // Add these two line will slow down the query time
    long long parent_id;                                                           //  from 0.037 to 0.041 (slow down construction more: 0.55 to 0.61)
};

typedef struct btree_p_search_res {
	btree_p_leaf_node* n;
	int slot;
} btree_p_search_res;


void btree_p_init(btree_p* tree);
void btree_p_clear(btree_p* tree, int* num_leaf_nodes, Stack *stack);
bool btree_p_insert(btree_p* const tree, const float key, const data_pt value, int *token2nodeIndex, int *token2nodeOffset, int* num_leaf_nodes, Stack *stack, bool **page_status, int dim, bool update_addr, btree_p_leaf_node ***leaf_list, int* max_leaves);
void btree_p_bulk_load(btree_p* const tree, float* keybegin, float* keyend, data_pt* databegin, data_pt* dataend, int *token2nodeIndex, int *token2nodeOffset, int* num_leaf_nodes, bool update_node, Stack *stack, bool **page_status, int dim, bool update_addr, btree_p_leaf_node ***leaf_list, int* max_leaves);
// Node is considered a match if its key is within the range between key-(1e-8) and 
// key+(1e-8) inclusive and its ID is exactly equal to value. 
bool btree_p_delete(btree_p* const tree, const float key, const long long value, int *token2nodeIndex, int *token2nodeOffset, int* num_leaf_nodes, Stack *stack, bool *page_status, int dim, bool update_addr);
btree_p_search_res btree_p_search(btree_p* const tree, const float key);
bool btree_p_is_end(const btree_p* const tree, const btree_p_search_res src);
btree_p_search_res btree_p_find_prev(btree_p_search_res src);
btree_p_search_res btree_p_find_next(btree_p_search_res src);
btree_p_search_res btree_p_first(const btree_p* const tree);
btree_p_search_res btree_p_last(const btree_p* const tree);
void btree_p_dump(btree_p* const tree);
float btree_p_keyof(const btree_p_search_res src);
data_pt btree_p_valueof(const btree_p_search_res src);

#ifdef __cplusplus
}
#endif

#endif
