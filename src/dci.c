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

 //#include <malloc.h>
#include "dci.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <string.h>
#include <Python.h>
#include "btree_i.h"
#include "btree_common.h"
#include "hashtable_d.h"
#include "hashtable_i.h"
#include "hashtable_p.h"
#include<immintrin.h>
#include <x86intrin.h>
#include <limits.h>
#include "bf16_util.h"
#include "util.h"
#include "debug.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif  // USE_OPENMP

#define INT_SIZE     (8 * sizeof(unsigned int))
static const int SLOT_NUM = 256/INT_SIZE;    // # of int in one SIMD register

#define BITSLOT(b) ((b) / INT_SIZE)
#define BITMASK(b) (1 << ((b) % INT_SIZE))
#define BITSET(a, b) ((a)[BITSLOT(b)] |= BITMASK(b))
#define BITNSLOTS(nb) ((nb + INT_SIZE - 1) / INT_SIZE)   // # of int for nb bits
#define BITTEST(a, b) ((a)[BITSLOT(b)] & BITMASK(b))

#define CLOSEST 128
#define cblas_enabled 1

static int seed = 3;  // For Debugging

void pack_points_to_matrix(
    additional_info** upper_cells,
    int num_points,
    int dim,
    bf16_t* points_matrix)
{
    for (int i = 0; i < num_points; i++) {
        memcpy(points_matrix + i * dim,
               upper_cells[i]->data_loc,
               dim * sizeof(bf16_t));
    }
}

void compute_points_sq_norms(
    const bf16_t* points_matrix,
    int num_points,
    int dim,
    float* sq_norms)
{
#if defined(__AVX512BF16__)
    // AVX512_BF16 path
    for (int i = 0; i < num_points; i++) {
        const bf16_t* point = points_matrix + (size_t)i * (size_t)dim;

        // Two accumulators to hide dpbf16 latency
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();

        int j = 0;

        // Process 64 BF16 values per iter (2x 32-wide dpbf16)
        for (; j + 63 < dim; j += 64) {
            __m512bh p0 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(point + j));
            __m512bh p1 = (__m512bh)_mm512_loadu_epi16((const __m512i*)(point + j + 32));
            acc0 = _mm512_dpbf16_ps(acc0, p0, p0);
            acc1 = _mm512_dpbf16_ps(acc1, p1, p1);
        }

        // One more full 32-wide chunk if present
        __m512 acc = _mm512_add_ps(acc0, acc1);
        for (; j + 31 < dim; j += 32) {
            __m512bh p = (__m512bh)_mm512_loadu_epi16((const __m512i*)(point + j));
            acc = _mm512_dpbf16_ps(acc, p, p);
        }

        // Tail: masked load of remaining BF16s (0..31), zero-fill inactive lanes
        int rem = dim - j;
        if (rem > 0) {
            // Use 64-bit shift to avoid UB if rem==32
            __mmask32 km = (rem >= 32) ? (__mmask32)0xFFFFFFFFu
                                       : (__mmask32)(((uint64_t)1u << rem) - 1u);

            __m512i v = _mm512_maskz_loadu_epi16(km, (const void*)(point + j));
            __m512bh p = (__m512bh)v;
            acc = _mm512_dpbf16_ps(acc, p, p);
        }

        float norm = _mm512_reduce_add_ps(acc);
        sq_norms[i] = norm;
    }
#else
    // Software fallback - convert to FP32 and compute
    for (int i = 0; i < num_points; i++) {
        const bf16_t* point = points_matrix + i * dim;
        float norm = 0.0f;

        for (int j = 0; j < dim; j++) {
            float val = bf16_to_f32(point[j]);
            norm += val * val;
        }

        sq_norms[i] = norm;
    }
#endif
}

static inline void BitAnd(const unsigned int* const x, unsigned int* const y, const int k) {
    __m256i X, Y; // 256-bit values
	long i=0;
	for (i = 0; i < k; i += SLOT_NUM) {
		X = _mm256_load_si256(x + i); // load chunk of ints
		Y = _mm256_load_si256(y + i);
		_mm256_store_si256(y + i, _mm256_and_si256(X, Y));
	}
}

static inline void BitNot_And(const unsigned int* const x, unsigned int* const y, const int k) {
    __m256i X, Y; // 256-bit values
    __m256i mask = _mm256_set1_epi32(-1);
	long i=0;
	for (i = 0; i < k; i += SLOT_NUM) {
		X = _mm256_load_si256(x + i); // load chunk of ints
		Y = _mm256_load_si256(y + i);
		_mm256_store_si256(y + i, _mm256_and_si256(_mm256_xor_si256(X, mask), Y));
	}
}

static inline float abs_d(float x) { return x > 0 ? x : -x; }

static inline int min_i(int a, int b) { return a < b ? a : b; }

static inline int max_i(int a, int b) { return a > b ? a : b; }

typedef struct tree_node {
    additional_info* parent;
    long long child;
    float dist;
} tree_node;

void free_cell(struct additional_info* cell, int num_indices, dci* dci_inst) {
    if(cell == NULL) return;
    // free indices only if needed
    int i;
    if (cell->cell_indices) {
        for (i = 0; i < num_indices; i++) {
            btree_p_clear(&(cell->cell_indices[i]), &(dci_inst->num_leaf_nodes), dci_inst->stack);
        }
        free(cell->cell_indices);
    }
    if (cell->arr_indices)
        free(cell->arr_indices);
    if (cell->num_finest_level_points)
        free(cell->num_finest_level_points);
    if (cell->num_finest_level_nodes)
        free(cell->num_finest_level_nodes);
    if (cell->local_dist)
        free(cell->local_dist);
    #ifdef USE_OPENMP
    omp_destroy_lock(&(cell->lock));
    #endif  // USE_OPENMP
    free(cell);
}

void free_instance(additional_info* root, int num_levels, int num_indices, dci* dci_inst) {
    if (root == NULL)
        return;
    int i, j;

    free_cell(root, num_indices, dci_inst);
    for (i = 0; i < num_levels; i++) {
        for (j = 0; j < dci_inst->num_points_on_level[i]; j++) {
            free_cell(dci_inst->points_on_level[i][j], num_indices, dci_inst);
        }
    }
}

static inline void realloc_(dci* const dci_inst) {
    hashtable_p_extend(dci_inst->inserted_points, dci_inst->max_volume);
    dci_inst->token2nodeIndex = (int*)realloc(dci_inst->token2nodeIndex, sizeof(int) * dci_inst->max_volume);
    dci_inst->token2nodeOffset = (int*)realloc(dci_inst->token2nodeOffset, sizeof(int) * dci_inst->max_volume);
}

static inline int add_to_list(int num_candidates, int num_neighbours, idx_arr* top_candidates, int* num_returned, 
                        float cur_dist, additional_info* cur_points, float* last_top_candidate_dist, dci_query_config query_config, 
                        int* num_returned_finest_level_points, int* last_top_candidate, int i, float init, int num_finest) {
    if (num_candidates < num_neighbours) {
        top_candidates[*num_returned].key = cur_dist;
        top_candidates[*num_returned].info = cur_points;
        if (cur_dist > (*last_top_candidate_dist)) {
            (*last_top_candidate_dist) = cur_dist;
            (*last_top_candidate) = *num_returned;
        }
        (*num_returned)++;
        if (query_config.min_num_finest_level_points) {
            (*num_returned_finest_level_points) += num_finest;
        }
    }
    else if (cur_dist < (*last_top_candidate_dist)) {
        int tmp;
        if (query_config.min_num_finest_level_points) {
            tmp = top_candidates[(*last_top_candidate)].info->num_finest_level_nodes[query_config.target_level];
            if (query_config.target_level == 0)
                tmp -= 1;
        }
        if (query_config.min_num_finest_level_points &&
            (*num_returned_finest_level_points) + num_finest - tmp <
            query_config.min_num_finest_level_points) { // Add
            top_candidates[*num_returned].key = cur_dist;
            top_candidates[*num_returned].info = cur_points;
            (*num_returned)++;
            (*num_returned_finest_level_points) += num_finest;
        }
        else {
            // Replace
            if (query_config.min_num_finest_level_points) {
                (*num_returned_finest_level_points) += num_finest - tmp;
            }
            top_candidates[(*last_top_candidate)].key = cur_dist;
            top_candidates[(*last_top_candidate)].info = cur_points;
            (*last_top_candidate_dist) = init;
            for (int j = 0; j < *num_returned; j++) {
                if (top_candidates[j].key > (*last_top_candidate_dist)) {
                    (*last_top_candidate_dist) = top_candidates[j].key;
                    (*last_top_candidate) = j;
                }
            }
        }
    }
    else if (query_config.min_num_finest_level_points && 
        (*num_returned_finest_level_points) <  query_config.min_num_finest_level_points) { // Also Add
        top_candidates[*num_returned].key = cur_dist;
        top_candidates[*num_returned].info = cur_points;
        (*num_returned)++;
        (*num_returned_finest_level_points) += num_finest;
    }
}

static inline int add_to_list_(int num_candidates, int num_neighbours, idx_arr* top_candidates, int* num_returned, 
                        float cur_dist, additional_info* cur_points, float* last_top_candidate_dist, dci_query_config query_config, 
                        int* num_returned_finest_level_points, int* last_top_candidate, int i, float init, int num_finest) {
    if (num_candidates < num_neighbours) {
        top_candidates[*num_returned].key = cur_dist;
        top_candidates[*num_returned].info = cur_points;
        if (cur_dist > (*last_top_candidate_dist)) {
            (*last_top_candidate_dist) = cur_dist;
            (*last_top_candidate) = *num_returned;
        }
        (*num_returned)++;
        if (query_config.min_num_finest_level_points) {
            (*num_returned_finest_level_points) += num_finest;
        }
    }
    else if (cur_dist < (*last_top_candidate_dist)) {
        if (query_config.min_num_finest_level_points &&
            (*num_returned_finest_level_points) + num_finest -
            top_candidates[(*last_top_candidate)].info->num_finest_level_points[query_config.target_level] <
            query_config.min_num_finest_level_points) { // Add
            top_candidates[*num_returned].key = cur_dist;
            top_candidates[*num_returned].info = cur_points;
            (*num_returned)++;
            (*num_returned_finest_level_points) += num_finest;
        }
        else {
            // Replace
            if (query_config.min_num_finest_level_points) {
                (*num_returned_finest_level_points) += num_finest -
                    top_candidates[(*last_top_candidate)].info->num_finest_level_points[query_config.target_level];
            }
            top_candidates[(*last_top_candidate)].key = cur_dist;
            top_candidates[(*last_top_candidate)].info = cur_points;
            (*last_top_candidate_dist) = init;
            for (int j = 0; j < *num_returned; j++) {
                if (top_candidates[j].key > (*last_top_candidate_dist)) {
                    (*last_top_candidate_dist) = top_candidates[j].key;
                    (*last_top_candidate) = j;
                }
            }
        }
    }
    else if (query_config.min_num_finest_level_points && 
        (*num_returned_finest_level_points) <  query_config.min_num_finest_level_points) { // Also Add
        top_candidates[*num_returned].key = cur_dist;
        top_candidates[*num_returned].info = cur_points;
        (*num_returned)++;
        (*num_returned_finest_level_points) += num_finest;
    }
}

static void dci_gen_proj_vec(bf16_t* const proj_vec, const int dim, const int num_indices) {
    // FP32 staging buffer (dim * num_indices)
    float* tmp = (float*)aligned_alloc(64, (size_t)dim * (size_t)num_indices * sizeof(float));
    if (!tmp) return; // handle OOM as you prefer

    // 1) generate N(0,1)
    for (int j = 0; j < num_indices; j++) {
        float sq = 0.0f;
        float* v = tmp + (size_t)j * (size_t)dim;

        for (int i = 0; i < dim; i++) {
            float x = rand_normal();
            v[i] = x;
            sq += x * x;
        }

        // 2) normalize in FP32
        float inv_norm = 1.0f / sqrtf(sq);
        for (int i = 0; i < dim; i++) {
            v[i] *= inv_norm;
        }

        // 3) convert once to BF16
        for (int i = 0; i < dim; i++) {
            proj_vec[(size_t)j * (size_t)dim + i] = f32_to_bf16(v[i]);
        }
    }

    free(tmp);
}

void data_projection(int num_indices, dci* const dci_inst, const int dim,
    const int num_points, const bf16_t* const data, float** p_data_proj, const bool* mask, bool pre_computed) {

    int i, j;
    // True if data_proj is (# of points) x (# of cell_indices)
    // column-major; used only for error-checking

    float* data_proj;

    if (pre_computed) {
        data_proj = *p_data_proj;
    }
    else {
        if (posix_memalign((void**)&data_proj, 64, sizeof(float) * num_indices * num_points) != 0) {
            perror("Memory allocation failed!\n");
            return;
        }
        *p_data_proj = data_proj;

        // data_proj is (# of cell_indices) x (# of points) column-major
        matmul(num_indices, num_points, dim, dci_inst->proj_vec, data, data_proj);
    }
    int cur_next_point_id = dci_inst->next_point_id;
    bool expand_flag = 0;
    while (cur_next_point_id + num_points >= dci_inst->max_volume) {
        dci_inst->max_volume *= 2;
        expand_flag = 1;
    }
    if (expand_flag) {
        realloc_(dci_inst);
    }
    if (dci_inst->transform) {
        if (cur_next_point_id == 0) {
            dci_inst->sq_norm_list = (float*)malloc(sizeof(float) * (dci_inst->max_volume));
        }
        else if (expand_flag) {
            dci_inst->sq_norm_list = (float*)realloc(dci_inst->sq_norm_list, sizeof(float) * (dci_inst->max_volume));
        }
        // Calculate the norm of all points
        float temp_norm;
        float* norms_out = dci_inst->sq_norm_list + (size_t)cur_next_point_id;
        compute_points_sq_norms(data, num_points, dim, norms_out);

        for (int i = 0; i < num_points; i++) {
            if (mask == NULL || mask[i]) {
                float temp_norm = norms_out[i];
                if (dci_inst->max_sq_norm < temp_norm) {
                    dci_inst->max_sq_norm = temp_norm;
                }
            }
        }

        float add_proj_vec_f32[num_indices];
        for (int j = 0; j < num_indices; j++) {
            add_proj_vec_f32[j] = bf16_to_f32(dci_inst->add_proj_vec[j]);
        }

        // key_transform
        // float sqt = sqrt(dci_inst->max_sq_norm);
        for (i = 0; i < num_points; i++) {
            if (mask == NULL || mask[i]) {
                float diff = dci_inst->max_sq_norm - norms_out[i];
                if (diff < 0.0f) diff = 0.0f;          // clamp tiny negatives due to rounding
                float scale = sqrtf(diff);

                float* proj_row = data_proj + (size_t)i * (size_t)num_indices;
                for (int j = 0; j < num_indices; j++) {
                    proj_row[j] += scale * add_proj_vec_f32[j];
                }
            }
        }
    }
}

static inline int dci_next_closest_proj(const idx_arr* const index, int* const left_pos, int* const right_pos, const float query_proj, const int num_elems, int* returned_ids, float* index_priority) {
    
    int returned_num = 0;
    int temp_right = *right_pos;
    int temp_left = *left_pos;
    int i = 0;
    int num = 0;
    if (temp_left == -1 && temp_right == num_elems) {
        return 0;
    } else if (temp_left == -1) {
        if (temp_right <= num_elems - CLOSEST) {
            returned_num = CLOSEST;
            for (i = 0; i < CLOSEST; i++) {
                returned_ids[num++] = index[temp_right].local_id;
                ++temp_right;
            }
        }
        else {
            returned_num = num_elems - temp_right;
            for (i = 0; temp_right < num_elems; i++) {
                returned_ids[num++] = index[temp_right].local_id;
                ++temp_right;
            }
        }
        *index_priority = abs_d(index[temp_right-1].key - query_proj);
    } else if (temp_right == num_elems) {
        if (temp_left >= CLOSEST-1) {
            returned_num = CLOSEST;
            for (i = 0; i < CLOSEST; i++) {
                returned_ids[num++] = index[temp_left].local_id;
                --temp_left;
            }
        }
        else {
            returned_num = temp_left + 1;
            for (i = 0; temp_left > -1; i++) {
                returned_ids[num++] = index[temp_left].local_id;
                --temp_left;
            }
        }
        *index_priority = abs_d(index[temp_left+1].key - query_proj);
    } else if (index[temp_right].key - query_proj < query_proj - index[temp_left].key) {
        if (temp_right <= num_elems - CLOSEST) {
            returned_num = CLOSEST;
            for (i = 0; i < CLOSEST; i++) {
                returned_ids[num++] = index[temp_right].local_id;
                ++temp_right;
            }
            *index_priority = abs_d(index[temp_right-1].key - query_proj);
        }
        else {
            returned_num = num_elems - temp_right;
            for (i = 0; temp_right < num_elems; i++) {
                returned_ids[num++] = index[temp_right].local_id;
                ++temp_right;
            }
            if (temp_left >= CLOSEST-1-i) {
                returned_num = CLOSEST;
                for (; i < CLOSEST; i++) {
                    returned_ids[num++] = index[temp_left].local_id;
                    --temp_left;
                }
            }
            else {
                returned_num += temp_left + 1;
                for (; temp_left > -1; i++) {
                    returned_ids[num++] = index[temp_left].local_id;
                    --temp_left;
                }
            }
            *index_priority = abs_d(index[temp_left+1].key - query_proj);
        }
    } else {
        if (temp_left >= CLOSEST-1) {
            returned_num = CLOSEST;
            for (i = 0; i < CLOSEST; i++) {
                returned_ids[num++] = index[temp_left].local_id;
                --temp_left;
            }
            *index_priority = abs_d(index[(temp_left)+1].key - query_proj);
        }
        else {
            returned_num = (temp_left) + 1;
            for (i = 0; temp_left > -1; i++) {
                returned_ids[num++] = index[temp_left].local_id;
                --temp_left;
            }
            if (temp_right <= num_elems - CLOSEST + i) {
                returned_num = CLOSEST;
                for (; i < CLOSEST; i++) {
                    returned_ids[num++] = index[temp_right].local_id;
                    ++temp_right;
                }
            }
            else {
                returned_num += num_elems - (temp_right);
                for (; temp_right < num_elems; i++) {
                    returned_ids[num++] = index[temp_right].local_id;
                    ++temp_right;
                }
            }
            *index_priority = abs_d(index[temp_right-1].key - query_proj);
        }
    }
    (*left_pos) = temp_left;
    (*right_pos) = temp_right;
    return returned_num;
}

static inline int dci_next_closest_proj_(const idx_arr* const index, int* const left_pos, int* const right_pos, const float query_proj, const int num_elems) {

    int cur_pos;
    if (*left_pos == -1 && *right_pos == num_elems) {
        cur_pos = -1;
    } else if (*left_pos == -1) {
        cur_pos = *right_pos;
        ++(*right_pos);
    } else if (*right_pos == num_elems) {
        cur_pos = *left_pos;
        --(*left_pos);
    } else if (index[*right_pos].key - query_proj < query_proj - index[*left_pos].key) {
        cur_pos = *right_pos;
        ++(*right_pos);
    } else {
        cur_pos = *left_pos;
        --(*left_pos);
    }
    return cur_pos;
}

// Returns the index of the element whose key is the largest that is less than the key
// Returns an integer from -1 to num_elems - 1 inclusive
// Could return -1 if all elements are greater or equal to key
static inline int dci_search_index(const idx_arr* const index, const float key, const int num_elems) {
    int start_pos, end_pos, cur_pos;
    
    start_pos = -1;
    end_pos = num_elems - 1;
    cur_pos = (start_pos + end_pos + 2) / 2;
    
    while (start_pos < end_pos) {
        if (index[cur_pos].key < key) {
            start_pos = cur_pos;
        } else {
            end_pos = cur_pos - 1;
        }
        cur_pos = (start_pos + end_pos + 2) / 2;
    }
    
    return start_pos;
}

static inline int dci_compare_data_pt_parent(const void *a, const void *b) {
    float key_diff = ((bulk_data_pt *)a)->parent_id - ((bulk_data_pt *)b)->parent_id;
    return (key_diff > 0) - (key_diff < 0);
}

static inline int dci_compare_data_pt_dist(const void *a, const void *b) {
    float key_diff = ((bulk_data_pt *)a)->local_dist - ((bulk_data_pt *)b)->local_dist;
    return (key_diff > 0) - (key_diff < 0);
}

static inline int dci_compare_data_idx_arr_dist(const void *a, const void *b) {
    float key_diff = ((idx_arr *)a)->key - ((idx_arr *)b)->key;
    return (key_diff > 0) - (key_diff < 0);
}

void update_arr_indices(int num_indices, additional_info* point) {
    if (point->flag == 0)
        return;
        
    int num_points = point->cell_indices[0].num_data;
    if (num_points == 0) {
        point->arr_indices = NULL;
    }
    else {
        data_pt cur_point;
        idx_arr* arr_indices = (idx_arr*)malloc(sizeof(idx_arr)*(num_indices*num_points));
        for (int i = 0; i < num_indices; i++) {
            btree_p_search_res cur = btree_p_first(&(point->cell_indices[i]));
            for (int j = 0; j < num_points; j++) {
                cur_point = btree_p_valueof(cur);
                if (i==0) {
                    cur_point.info->local_id = j;
                    arr_indices[i*num_points+j].local_id = j;
                }
                else {
                    arr_indices[i*num_points+j].local_id = cur_point.info->local_id;
                }
                arr_indices[i*num_points+j].info = cur_point.info;
                arr_indices[i*num_points+j].key = cur_point.info->local_dist[i];
                cur = btree_p_find_next(cur);
            }
        }
        free(point->arr_indices);
        point->arr_indices = arr_indices;
    }
    point->flag = 0;
}

static void update_local_dist(int num_indices, additional_info* point, float max_sq_norm, bf16_t* add_proj_vec, float* sq_norm_list) {
    float o_term = sqrt((point->max_sq_norm) - (sq_norm_list[point->id]));
    float n_term = sqrt((max_sq_norm) - (sq_norm_list[point->id]));
    for (int k = 0; k < num_indices; k++) {
        point->local_dist[k] += (n_term - o_term) * bf16_to_f32(add_proj_vec[k]);
    }
    point->max_sq_norm = max_sq_norm;
}

static void update_max_sq_norm(int num_indices, additional_info* point, float max_sq_norm, bf16_t* add_proj_vec, float* sq_norm_list, int* token2nodeIndex, int* token2nodeOffset, int* num_leaf_nodes, Stack* stack, bool **page_status, int dim, bool update_addr, btree_p_leaf_node ***leaf_list, int* max_leaves) {
    // Step 1. Get all the children of the point and update by the new max_sq_norm
    // Step 2. From each cur_index of the point, get the arr_indices, then bulk_load
    int num_points = point->cell_indices[0].num_data;
    assert(num_points > 0);

    data_pt cur_point;
    idx_arr* arr_indices = (idx_arr*)malloc(sizeof(idx_arr)*(num_indices*num_points));
    bulk_data_pt* bulk = (bulk_data_pt*)malloc(sizeof(bulk_data_pt) * num_points);
    float* bulk_data_proj = (float*)malloc(sizeof(float) * num_points);
    data_pt* bulk_data = (data_pt*)malloc(sizeof(data_pt) * num_points);
    
    if (!arr_indices || !bulk || !bulk_data_proj || !bulk_data) {
        perror("Memory allocation failed in update_max_sq_norm");
        return;
    }

    btree_p_search_res cur = btree_p_first(&(point->cell_indices[0]));
    for (int j = 0; j < num_points; j++) {
        cur_point = btree_p_valueof(cur);
        update_local_dist(num_indices, cur_point.info, max_sq_norm, add_proj_vec, sq_norm_list);
        cur = btree_p_find_next(cur);
    }

    for (int i = 0; i < num_indices; i++) {
        btree_p_search_res cur = btree_p_first(&(point->cell_indices[i]));
        for (int j = 0; j < num_points; j++) {
            cur_point = btree_p_valueof(cur);
            bulk[j].data_pt = cur_point;
            bulk[j].local_dist = cur_point.info->local_dist[i];
            cur = btree_p_find_next(cur);
        }
        qsort(bulk, num_points, sizeof(bulk_data_pt), dci_compare_data_pt_dist);
        for (int j = 0; j < num_points; j++) {
            bulk_data_proj[j] = bulk[j].local_dist;
            bulk_data[j] = bulk[j].data_pt;
            cur_point = bulk[j].data_pt;
            if (i==0) {
                cur_point.info->local_id = j;
                arr_indices[i*num_points+j].local_id = j;
            }
            else {
                arr_indices[i*num_points+j].local_id = cur_point.info->local_id;
            }
            arr_indices[i*num_points+j].info = cur_point.info;
            arr_indices[i*num_points+j].key = bulk[j].local_dist;
        }
        btree_p_clear(&(point->cell_indices[i]), num_leaf_nodes, stack);
        btree_p_init(&(point->cell_indices[i]));
        btree_p_bulk_load(&(point->cell_indices[i]), &(bulk_data_proj[0]), &(bulk_data_proj[num_points]), &(bulk_data[0]), &(bulk_data[num_points]), 
        token2nodeIndex, token2nodeOffset, num_leaf_nodes, 1, stack, page_status, dim, update_addr, leaf_list, max_leaves);
    }
    free(point->arr_indices);
    point->arr_indices = arr_indices;
    // point->max_sq_norm = max_sq_norm;
    point->flag = 0;  // already updated the arr_indices
    
    free(bulk);
    free(bulk_data_proj);
    free(bulk_data);
}

static void add_and_update_max_sq_norm(int num_indices, additional_info* point, float max_sq_norm, bf16_t* add_proj_vec, float* sq_norm_list, data_pt new_point, float* new_data_proj, int* token2nodeIndex, int* token2nodeOffset, int* num_leaf_nodes, Stack* stack, bool **page_status, int dim, bool update_addr, btree_p_leaf_node ***leaf_list, int* max_leaves) {
    // Step 1. Get all the children of the point and update by the new max_sq_norm
    // Step 2. Add the new child to the children list before sorting
    // Step 3. From each cur_index of the point, get the arr_indices, then bulk_load
    int num_points = point->cell_indices[0].num_data;
    assert(num_points > 0);

    num_points += 1;
    data_pt cur_point;
    idx_arr* arr_indices = (idx_arr*)malloc(sizeof(idx_arr)*(num_indices*num_points));
    bulk_data_pt* bulk = (bulk_data_pt*)malloc(sizeof(bulk_data_pt) * num_points);
    float* bulk_data_proj = (float*)malloc(sizeof(float) * num_points);
    data_pt* bulk_data = (data_pt*)malloc(sizeof(data_pt) * num_points);
    
    if (!arr_indices || !bulk || !bulk_data_proj || !bulk_data) {
        perror("Memory allocation failed in add_and_update_max_sq_norm");
        return;
    }

    btree_p_search_res cur = btree_p_first(&(point->cell_indices[0]));
    for (int j = 0; j < num_points - 1; j++) {
        cur_point = btree_p_valueof(cur);
        update_local_dist(num_indices, cur_point.info, max_sq_norm, add_proj_vec, sq_norm_list);
        cur = btree_p_find_next(cur);
    }

    for (int i = 0; i < num_indices; i++) {
        btree_p_search_res cur = btree_p_first(&(point->cell_indices[i]));
        for (int j = 0; j < num_points - 1; j++) {
            cur_point = btree_p_valueof(cur);
            bulk[j].data_pt = cur_point;
            bulk[j].local_dist = cur_point.info->local_dist[i];
            cur = btree_p_find_next(cur);
        }
        bulk[num_points-1].data_pt = new_point;  // Add the new point
        bulk[num_points-1].local_dist = new_data_proj[i];  // Add the new point
        qsort(bulk, num_points, sizeof(bulk_data_pt), dci_compare_data_pt_dist);
        for (int j = 0; j < num_points; j++) {
            bulk_data_proj[j] = bulk[j].local_dist;
            bulk_data[j] = bulk[j].data_pt;
            cur_point = bulk[j].data_pt;
            if (i==0) {
                cur_point.info->local_id = j;
                arr_indices[i*num_points+j].local_id = j;
            }
            else {
                arr_indices[i*num_points+j].local_id = cur_point.info->local_id;
            }
            arr_indices[i*num_points+j].info = cur_point.info;
            arr_indices[i*num_points+j].key = bulk[j].local_dist;
        }
        btree_p_clear(&(point->cell_indices[i]), num_leaf_nodes, stack);
        btree_p_init(&(point->cell_indices[i]));
        btree_p_bulk_load(&(point->cell_indices[i]), &(bulk_data_proj[0]), &(bulk_data_proj[num_points]), &(bulk_data[0]), &(bulk_data[num_points]), 
        token2nodeIndex, token2nodeOffset, num_leaf_nodes, 1, stack, page_status, dim, update_addr, leaf_list, max_leaves);
    }
    free(point->arr_indices);
    point->arr_indices = arr_indices;
    // point->max_sq_norm = max_sq_norm;
    point->flag = 0;  // already updated the arr_indices
    
    free(bulk);
    free(bulk_data_proj);
    free(bulk_data);
}

static void dci_insert_to_indices(dci* dci_inst, bool transform, int num_indices, additional_info* parent, additional_info* point, float max_sq_norm, bf16_t* add_proj_vec, float* sq_norm_list) {
    // Insert the new point to the indices
    // 1. no child or no transformation -> directly insert
    // 2. child exists and transformation -> compare the max_sq_norm between the child and the new point
    //           -> == max_sq_norms -> directly insert
    //           -> != max_sq_norms -> check which part needs to update the local_dist
    //                             -> new point's max_sq_norm < child's max_sq_norm -> update the new point's local_dist, and insert
    //                             -> child's max_sq_norm < new point's max_sq_norm -> call <add_and_update_max_sq_norm>
    int k;
    data_pt data_point;
    data_point.info = point;

    if ((!transform) || (parent->cell_indices[0].num_data == 0)) {
        for (k = 0; k < num_indices; k++) {
            btree_p_insert(&(parent->cell_indices[k]),
                point->local_dist[k], data_point, dci_inst->token2nodeIndex, dci_inst->token2nodeOffset, &(dci_inst->num_leaf_nodes), dci_inst->stack, &(dci_inst->page_status), dci_inst->dim, dci_inst->update_addr, &(dci_inst->leaf_list), &(dci_inst->max_leaves));
        }
        parent->flag = 1;
    }
    else {
        float child_max_sq_norm = btree_p_valueof(btree_p_first(&(parent->cell_indices[0]))).info->max_sq_norm;
        if (point->max_sq_norm == child_max_sq_norm) {
            for (k = 0; k < num_indices; k++) {
                btree_p_insert(&(parent->cell_indices[k]),
                    point->local_dist[k], data_point, dci_inst->token2nodeIndex, dci_inst->token2nodeOffset, &(dci_inst->num_leaf_nodes), dci_inst->stack, &(dci_inst->page_status), dci_inst->dim, dci_inst->update_addr, &(dci_inst->leaf_list), &(dci_inst->max_leaves));
            }
            parent->flag = 1;
        }
        else {
            if (point->max_sq_norm < child_max_sq_norm) {
                update_local_dist(num_indices, point,
                                child_max_sq_norm, add_proj_vec, sq_norm_list);
                for (k = 0; k < num_indices; k++) {
                    btree_p_insert(&(parent->cell_indices[k]),
                        point->local_dist[k], data_point, dci_inst->token2nodeIndex, dci_inst->token2nodeOffset, &(dci_inst->num_leaf_nodes), dci_inst->stack, &(dci_inst->page_status), dci_inst->dim, dci_inst->update_addr, &(dci_inst->leaf_list), &(dci_inst->max_leaves));
                }
                parent->flag = 1;
            }
            else {
                add_and_update_max_sq_norm(num_indices, parent,
                                point->max_sq_norm, add_proj_vec, sq_norm_list,
                                data_point, point->local_dist, dci_inst->token2nodeIndex, dci_inst->token2nodeOffset, &(dci_inst->num_leaf_nodes), dci_inst->stack, &(dci_inst->page_status), dci_inst->dim, dci_inst->update_addr, &(dci_inst->leaf_list), &(dci_inst->max_leaves));
            }
        }
    }

    // Update max_child_dist: calculate distance from parent to the newly added child
    if (parent->id >= 0 && parent->data_loc != NULL && point->data_loc != NULL) {
        float dist;
        if (transform) {
            dist = transform_compute_dist(parent->data_loc, point->data_loc, dci_inst->dim,
                                        dci_inst->max_sq_norm,
                                        dci_inst->sq_norm_list[parent->id],
                                        dci_inst->sq_norm_list[point->id]);
        } else {
            dist = compute_dist(parent->data_loc, point->data_loc, dci_inst->dim);
        }

        // Update max_child_dist if this distance is larger
        if (dist > parent->max_child_dist) {
            parent->max_child_dist = dist;
        }
    }
}

static int dci_query_single_point(
    int num_comp_indices, int num_simp_indices, int dim, int num_levels,
    additional_info* root, int num_populated_levels, int num_neighbours,
    idx_arr* points_to_expand,  idx_arr** points_to_expand_next, int* num_top_candidates,
    const bf16_t* const query, const float* const query_proj,
    dci_query_config query_config, idx_arr* const top_candidates,
    bool cumu, dci* const dci_inst);

static int dci_query_single_point_(
    int num_comp_indices, int num_simp_indices, int dim, int num_levels,
    additional_info* root, int num_populated_levels, int num_neighbours,
    const bf16_t* const query, const float query_proj,
    dci_query_config query_config, idx_arr* const top_candidates,
    bool cumu, dci* const dci_inst, long long query_id, int num_upper_points, additional_info** upper_level_cells_ret,
    bf16_t* points_matrix, float* all_distances, float* points_sq_norms);

static inline void initialize_indices(btree_p* tree, int num_indices) {
    for (int i = 0; i < num_indices; i++) {
        btree_p_init(&(tree[i]));
    }
}

additional_info* create_sub_root(bulk_data_pt* bulk, int num_points_on_cur_levels, int num_indices, float max_sq_norm, int* token2nodeIndex, int* token2nodeOffset, int* num_leaf_nodes, Stack* stack, bool **page_status, int dim, bool update_addr, btree_p_leaf_node ***leaf_list, int* max_leaves) {
    btree_p* cur_index;
    float* bulk_data_proj = (float*)malloc(sizeof(float) * num_points_on_cur_levels);
    if (bulk_data_proj == NULL) {
        perror("Memory allocation failed for bulk_data_proj!\n");
        return NULL;
    }
    data_pt* bulk_data = (data_pt*)malloc(sizeof(data_pt) * num_points_on_cur_levels);
    if (bulk_data == NULL) {
        perror("Memory allocation failed for bulk_data!\n");
        return NULL;
    }

    additional_info* cur_empty_root = (additional_info*)malloc(sizeof(additional_info));
    cur_empty_root->cell_indices = (btree_p*)malloc(sizeof(btree_p) * num_indices);
    cur_empty_root->arr_indices = (idx_arr*)malloc(sizeof(idx_arr)*(num_indices*num_points_on_cur_levels));
    cur_empty_root->num_finest_level_points = NULL;
    cur_empty_root->local_dist = NULL;
    cur_empty_root->max_sq_norm = max_sq_norm;
    cur_empty_root->max_child_dist = 0.;
    #ifdef USE_OPENMP
    omp_init_lock(&(cur_empty_root->lock));
    #endif
    cur_empty_root->id = -1;
    initialize_indices(cur_empty_root->cell_indices, num_indices);

    for (int k = 0; k < num_indices; k++) {
        for (int j = 0; j < num_points_on_cur_levels; j++) {
            bulk[j].local_dist = bulk[j].data_pt.info->local_dist[k];
        }
        qsort(bulk, num_points_on_cur_levels, sizeof(bulk_data_pt), dci_compare_data_pt_dist);
        for (int j = 0; j < num_points_on_cur_levels; j++) {
            bulk_data_proj[j] = bulk[j].local_dist;
            bulk_data[j] = bulk[j].data_pt;
            cur_empty_root->arr_indices[k*num_points_on_cur_levels+j].info = bulk[j].data_pt.info;
            cur_empty_root->arr_indices[k*num_points_on_cur_levels+j].key = bulk[j].local_dist;
            if (k == 0) {
                bulk[j].data_pt.info->local_id = j;
                cur_empty_root->arr_indices[k*num_points_on_cur_levels+j].local_id = j;
            }
            else {
                cur_empty_root->arr_indices[k*num_points_on_cur_levels+j].local_id = bulk[j].data_pt.info->local_id;
            }
        }
        cur_index = &(cur_empty_root->cell_indices[k]);
        btree_p_bulk_load(cur_index, &(bulk_data_proj[0]), &(bulk_data_proj[num_points_on_cur_levels]), &(bulk_data[0]), &(bulk_data[num_points_on_cur_levels]), token2nodeIndex, token2nodeOffset, num_leaf_nodes, 0, stack, page_status, dim, update_addr, leaf_list, max_leaves);
    }
    cur_empty_root->flag = 0;
    
    free(bulk_data_proj);
    free(bulk_data);

    return cur_empty_root;
}

void dci_init(dci* const dci_inst, const int dim,
    const int num_comp_indices, const int num_simp_indices, float promotion_prob,
    float promotion_prob_subseq, int max_volume, bool transform, int parallel_level,
    bool debug, bf16_t* proj_vec) {
    if (debug)
        srand48(seed);
    else
        srand48(time(NULL));
    
    int num_indices = num_comp_indices * num_simp_indices;
    dci_inst->dim = dim;
    dci_inst->num_comp_indices = num_comp_indices;
    dci_inst->num_simp_indices = num_simp_indices;
    dci_inst->num_points = 0;
    dci_inst->num_levels = 0;
    dci_inst->next_point_id = 0;
    dci_inst->next_target_level = 0;
    dci_inst->num_points_on_level = NULL;
    dci_inst->points_on_level = NULL;
    dci_inst->token2nodeIndex = NULL;
    dci_inst->token2nodeOffset = NULL;
    // dci_inst->nodeIndex2Address = NULL;
    dci_inst->num_leaf_nodes = 0;
    dci_inst->max_num_on_level = NULL;
    dci_inst->root = NULL;
    dci_inst->promotion_prob = promotion_prob;
    dci_inst->promotion_prob_subseq = promotion_prob_subseq;
    assert(max_volume > 0);
    int shifted = 1;
    while (shifted < max_volume) { // Keep shifting left until 2^n > max_volume
        shifted <<= 1;
    }
    dci_inst->max_volume = shifted;
    hashtable_p* inserted_points = (hashtable_p*)malloc(sizeof(hashtable_p));
    inserted_points->size = 0;
    dci_inst->inserted_points = inserted_points;
    hashtable_p_init(dci_inst->inserted_points, dci_inst->max_volume, 1);
    dci_inst->token2nodeIndex = (int*)malloc(sizeof(int) * (dci_inst->max_volume));
    dci_inst->token2nodeOffset = (int*)malloc(sizeof(int) * (dci_inst->max_volume));
    dci_inst->max_sq_norm = 0;
    dci_inst->sq_norm_list = NULL;
    dci_inst->transform = transform;
    dci_inst->parallel_level = parallel_level;
    dci_inst->inner_threads = 1;  // Default to 1, will be set dynamically later
    dci_inst->inner_inner_threads = 1;  // Default to 1, will be set dynamically later
    dci_inst->numa_threshold = 3000;  // Default threshold for NUMA optimization
    dci_inst->debug = debug;
    int expected_num_pages = (int)(dci_inst->max_volume / 8);
    dci_inst->max_leaves = expected_num_pages;
    dci_inst->stack = (Stack*)malloc(sizeof(Stack));
    initStack(dci_inst->stack, expected_num_pages);
    dci_inst->page_status = (bool*)malloc(sizeof(bool) * expected_num_pages);
    dci_inst->update_addr = 0;
    dci_inst->leaf_list = (btree_p_leaf_node**)malloc(sizeof(btree_p_leaf_node*) * expected_num_pages);
    dci_inst->sub_root_list = NULL;

    if (transform) {
        if (posix_memalign((void**)&(dci_inst->proj_vec), 64, sizeof(bf16_t) * dim * num_indices) != 0) {
            perror("Memory allocation failed!\n");
            return;
        }
        if (posix_memalign((void**)&(dci_inst->add_proj_vec), 64, sizeof(bf16_t) * 1 * num_indices) != 0) {
            perror("Memory allocation failed!\n");
            return;
        }

        if (proj_vec) {
            for (int j = 0; j < num_indices; j++) {
                for (int i = 0; i < dim; i++) {
                    dci_inst->proj_vec[i + j * (dim)] = proj_vec[i + j * (dim+1)];
                }
                dci_inst->add_proj_vec[j] = proj_vec[dim + j * (dim+1)];
            }
        }
        else {
            bf16_t* temp_proj_vec;
            if (posix_memalign((void**)&(temp_proj_vec), 64, sizeof(bf16_t) * (dim+1) * num_indices) != 0) {
                perror("Memory allocation failed!\n");
                return;
            }
            dci_gen_proj_vec(temp_proj_vec, dim+1, num_indices);

            for (int j = 0; j < num_indices; j++) {
                for (int i = 0; i < dim; i++) {
                    dci_inst->proj_vec[i + j * (dim)] = temp_proj_vec[i + j * (dim+1)];
                }
                dci_inst->add_proj_vec[j] = temp_proj_vec[dim + j * (dim+1)];
            }
            free(temp_proj_vec);
        }
    }
    else {
        if (posix_memalign((void**)&(dci_inst->proj_vec), 64, sizeof(bf16_t) * dim * num_indices) != 0) {
            perror("Memory allocation failed!\n");
            return;
        }
        if (proj_vec) {
            for (int j = 0; j < num_indices; j++) {
                for (int i = 0; i < dim; i++) {
                    dci_inst->proj_vec[i + j * dim] = proj_vec[i + j * dim];
                }
            }
        }
        else
            dci_gen_proj_vec(dci_inst->proj_vec, dim, num_indices);

        dci_inst->add_proj_vec = NULL;
    }
}

void dci_clear(dci* const dci_inst) {
    free_instance(dci_inst->root, dci_inst->num_levels, dci_inst->num_simp_indices * dci_inst->num_comp_indices, dci_inst);
    dci_inst->num_points = 0;
    dci_inst->num_levels = 0;
    dci_inst->next_point_id = 0;
    dci_inst->root = NULL;
    dci_inst->sub_root_list = NULL;
    if(dci_inst->num_points_on_level != NULL) {
        free(dci_inst->num_points_on_level);
        dci_inst->num_points_on_level = NULL;
    }
    if(dci_inst->inserted_points != NULL) {
        hashtable_p_clear(dci_inst->inserted_points);
        free(dci_inst->inserted_points);
        dci_inst->inserted_points = NULL;
    }
    if (dci_inst->sq_norm_list != NULL) {
        free(dci_inst->sq_norm_list);
        dci_inst->sq_norm_list = NULL;
    }
    if (dci_inst->points_on_level != NULL) {
        for (int i = 0; i < dci_inst->num_levels; i++) {
            free(dci_inst->points_on_level[i]);
        }
        free(dci_inst->points_on_level);
        dci_inst->points_on_level = NULL;
    }
    if (dci_inst->max_num_on_level != NULL) {
        for (int i = 0; i < dci_inst->num_levels; i++) {
            free(dci_inst->max_num_on_level[i]);
        }
        free(dci_inst->max_num_on_level);
        dci_inst->max_num_on_level = NULL;
    }
    if (dci_inst->token2nodeIndex != NULL) {
        free(dci_inst->token2nodeIndex);
        dci_inst->token2nodeIndex = NULL;
    }
    if (dci_inst->token2nodeOffset != NULL) {
        free(dci_inst->token2nodeOffset);
        dci_inst->token2nodeOffset = NULL;
    }
    if (dci_inst->page_status != NULL) {
        free(dci_inst->page_status);
        dci_inst->page_status = NULL;
    }
    if (dci_inst->leaf_list != NULL) {
        free(dci_inst->leaf_list);
        dci_inst->leaf_list = NULL;
    }
    if (dci_inst->stack != NULL) {
        freeStack(dci_inst->stack);
        free(dci_inst->stack);
        dci_inst->stack = NULL;
    }
}

void dci_free(dci* const dci_inst) {
    dci_clear(dci_inst);
    free(dci_inst->proj_vec);
    dci_inst->proj_vec = NULL;
    if (dci_inst->transform) {
        free(dci_inst->add_proj_vec);
        dci_inst->add_proj_vec = NULL;
    }
}

void dci_reset(dci* const dci_inst) {
    if (dci_inst->debug)
        srand48(seed);
    else
        srand48(time(NULL));
    if (dci_inst->transform) {
        int num_indices = dci_inst->num_comp_indices*dci_inst->num_simp_indices;
        int dim = dci_inst->dim;
        bf16_t* temp_proj_vec;
        if (posix_memalign((void**)&(temp_proj_vec), 64, sizeof(bf16_t) * (dim+1) * num_indices) != 0) {
            perror("Memory allocation failed!\n");
            return;
        }
        dci_gen_proj_vec(temp_proj_vec, dim+1, num_indices);
        for (int j = 0; j < num_indices; j++) {
            for (int i = 0; i < dim; i++) {
                dci_inst->proj_vec[i + j * (dim)] = temp_proj_vec[i + j * (dim+1)];
            }
            dci_inst->add_proj_vec[j] = temp_proj_vec[dim + j * (dim+1)];
        }
        free(temp_proj_vec);
    }
    else
        dci_gen_proj_vec(dci_inst->proj_vec, dci_inst->dim, dci_inst->num_comp_indices*dci_inst->num_simp_indices);
}

static void dci_assign_parent(
    int num_comp_indices, int num_simp_indices, int dim, int num_levels,
    additional_info* root, const int num_populated_levels, const int num_queries,
    const long long* selected_query_pos, const bf16_t* const query,
    const float* const query_proj, const dci_query_config query_config,
    tree_node* const assigned_parent, dci* const dci_inst, int num_upper_points, additional_info** upper_level_cells_ret,
    bf16_t* points_matrix, float* points_sq_norms);

void construct_new_tree(dci* const dci_inst, additional_info** root, const int dim,
    const int num_points, int* actual_num_levels, int* next_target_level,
    const bf16_t* const data, const bf16_t* const value, int** p_num_points_on_level,
    float** p_data_proj, additional_info**** p_level_cells_ret,
    const dci_query_config construction_query_config, bool* mask, int empty_until,
    bool random, int interval, int X, float anchor_threshold) {

    int h, i, j;
    int num_points_on_cur_levels;
    int max_num_points_on_cur_levels;
    // Only populated when actual_num_levels >= 2
    long long** level_members;
    int num_comp_indices = dci_inst->num_comp_indices;
    int num_simp_indices = dci_inst->num_simp_indices;
    int num_indices = num_comp_indices * num_simp_indices;
    // True if data_proj is (# of points) x (# of cell_indices)
    bool data_proj_transposed = false;
    // column-major; used only for error-checking
    float promotion_prob = dci_inst->promotion_prob;

    if (*p_data_proj == NULL)
        data_projection(num_indices, dci_inst, dim, num_points, data, p_data_proj, mask, 0);
    float* data_proj = *p_data_proj;
    float max_sq_norm = dci_inst->max_sq_norm;

    int* data_levels = (int*)malloc(sizeof(int) * num_points);  // Since num_points is relatively large (<= 1M)

    (*p_num_points_on_level)[0] = 0;

    int* num_points_on_level;

    if (!random) {
        int target_level, next_t_level = 0, cur_num_levels = 0;
        for (i = 0; i < num_points; i++) {
            if (mask == NULL || mask[i]) {
                float promotion_prob = dci_inst->promotion_prob;
                target_level = next_t_level;
                if (target_level > 0)
                    promotion_prob = dci_inst->promotion_prob_subseq;
                
                if (target_level == cur_num_levels) {
                    cur_num_levels++;
                    if (cur_num_levels > 1)
                        (*p_num_points_on_level) = (int*)realloc(*p_num_points_on_level, sizeof(int) * cur_num_levels);
                    (*p_num_points_on_level)[target_level] = 0;
                    next_t_level = 0;
                }
                else if (target_level == cur_num_levels - 1) {
                    int promo = (int)ceil(1/promotion_prob);
                    if ((*p_num_points_on_level)[target_level] + 1 == promo) {
                        next_t_level = target_level + 1;
                    }
                    else {
                        next_t_level = 0;
                    }
                }
                else {
                    if (((*p_num_points_on_level)[target_level] + 1 >= (*p_num_points_on_level)[target_level + 1] / promotion_prob))
                        next_t_level = target_level + 1;
                    else
                        next_t_level = 0;
                }
                data_levels[i] = target_level;
                (*p_num_points_on_level)[target_level] += 1;
            }
        }
        *next_target_level = next_t_level;
        *actual_num_levels = cur_num_levels;
        num_points_on_level = *p_num_points_on_level;

        if (cur_num_levels >= 2) {
            *p_level_cells_ret = (additional_info***)realloc(*p_level_cells_ret, sizeof(additional_info**) * cur_num_levels);
            level_members = (long long**)malloc(sizeof(long long*) * cur_num_levels);
            int cumu_levels[cur_num_levels];
            for (i = 0; i < cur_num_levels; i++) {
                level_members[i] = (long long*)malloc(sizeof(long long) * num_points_on_level[i]);
                cumu_levels[i] = 0;
            }
            for (i = 0; i < num_points; i++) {
                if (mask == NULL || mask[i]) {
                    h = data_levels[i];
                    level_members[h][cumu_levels[h]++] = i;
                }
            }
        }
        else {
            level_members = (long long**)malloc(sizeof(long long*) * (1));
            level_members[0] = (long long*)malloc(sizeof(long long) * num_points);
            int j = 0;
            for (i=0; i < num_points; i++){
                if (mask == NULL || mask[i])
                    level_members[0][j++] = i;
            }
        }
    }
    else {
        int cur_num_levels = 1;
        for (j = 0; j < num_points; j++) {
            if (mask == NULL || mask[j]) {
                i = 0;
                while (1) {
                    if (drand48() > promotion_prob)
                        break;
                    i++;
                }
                if (i >= cur_num_levels) {
                    *p_num_points_on_level = (int*)realloc(*p_num_points_on_level, sizeof(int) * (i + 1));
                    for (int ii = cur_num_levels; ii <= i; ii++) {
                        (*p_num_points_on_level)[ii] = 0;
                    }
                    cur_num_levels = i + 1; 
                }
                (*p_num_points_on_level)[i]++;
                data_levels[j] = i;
            }
        }
        num_points_on_level = *p_num_points_on_level;
        int level_relabelling[cur_num_levels];

        h = 0;
        for (i = 0; i < cur_num_levels; i++) {
            if (num_points_on_level[i] > 0 || i < empty_until) {
                level_relabelling[i] = h;
                h++;
            }
            else {
                level_relabelling[i] = -1;
            }
        }
        for (i = 0; i < cur_num_levels; i++) {
            if (level_relabelling[i] >= 0) {
                num_points_on_level[level_relabelling[i]] = num_points_on_level[i];
            }
        }
        *actual_num_levels = h;

        if (*actual_num_levels >= 2) {
            *p_level_cells_ret = (additional_info***)realloc(*p_level_cells_ret, sizeof(additional_info**) * (*actual_num_levels));
            level_members = (long long**)malloc(sizeof(long long*) * (*actual_num_levels));
            for (i = 0; i < *actual_num_levels; i++) {
                level_members[i] = (long long*)malloc(sizeof(long long) * num_points_on_level[i]);
                h = 0;
                for (j = 0; j < num_points; j++) {
                    if (mask == NULL || mask[j]) {
                        if (level_relabelling[data_levels[j]] == i) {
                            level_members[i][h] = j;
                            h++;
                        }
                    }
                }
                assert(h == num_points_on_level[i]);
            }
        }
        else {
            level_members = (long long**)malloc(sizeof(long long*) * (1));
            level_members[0] = (long long*)malloc(sizeof(long long) * num_points);
            int j = 0;
            for (i=0; i < num_points; i++){
                if (mask == NULL || mask[i])
                    level_members[0][j++] = i;
            }
        }
    }
    free(data_levels);

    additional_info*** level_cells_ret = *p_level_cells_ret;
    int max_num_points_on_level = 0;
    for (i = (*actual_num_levels) - 1; i >= 0; i--) {
        num_points_on_cur_levels = num_points_on_level[i];
        max_num_points_on_cur_levels = 2;
        while (num_points_on_cur_levels > max_num_points_on_cur_levels) {
            max_num_points_on_cur_levels *= 2;
        }
        level_cells_ret[i] = (additional_info**)malloc(
            sizeof(additional_info*) * max_num_points_on_cur_levels);
        // we do this individually to be able to free them independently (for deletion)
        for (j = 0; j < num_points_on_cur_levels; j++) {
            level_cells_ret[i][j] = (additional_info*)malloc(sizeof(additional_info));
        }
        if (num_points_on_cur_levels > max_num_points_on_level) {
            max_num_points_on_level = num_points_on_cur_levels;
        }
    }

    if (empty_until > 0) {
        dci_inst->sub_root_list = (additional_info**)malloc(sizeof(additional_info*) * (*actual_num_levels));
        for (i = 0; i < (*actual_num_levels); i++) {
            dci_inst->sub_root_list[i] = NULL;
        }
    }

    i = (*actual_num_levels) - 1;
    num_points_on_cur_levels = num_points_on_level[i];

    assert(num_points_on_cur_levels > 0);

    *root = (additional_info*)malloc(sizeof(additional_info));
    (*root)->cell_indices = (btree_p*)malloc(sizeof(btree_p) * num_indices);
    (*root)->arr_indices = (idx_arr*)malloc(sizeof(idx_arr)*(num_indices * num_points_on_cur_levels));
    (*root)->num_finest_level_points = NULL;
    (*root)->num_finest_level_nodes = NULL;
    (*root)->local_dist = NULL;
    (*root)->max_sq_norm = max_sq_norm;
    #ifdef USE_OPENMP
    omp_init_lock(&((*root)->lock));
    #endif
    (*root)->id = -1;
    (*root)->flag = 1;
    (*root)->max_child_dist = 0.;
    initialize_indices((*root)->cell_indices, num_indices);

    bulk_data_pt* bulk = (bulk_data_pt*)malloc(sizeof(bulk_data_pt) * max_num_points_on_level);
    tree_node* assigned_parent = (tree_node*)malloc(sizeof(tree_node) * max_num_points_on_level);
    float* bulk_data_proj = (float*)malloc(sizeof(float) * max_num_points_on_level);
    data_pt* bulk_data = (data_pt*)malloc(sizeof(data_pt) * max_num_points_on_level);
    int* parent_idx = (int*)malloc(sizeof(int) * (max_num_points_on_level + 1));

    btree_p* cur_index;
    for (j = 0; j < num_points_on_cur_levels; j++) {
        int k;
        additional_info* cur_cell = level_cells_ret[i][j];
        cur_cell->id = level_members[i][j];
        cur_cell->arr_indices = NULL;
        #ifdef USE_OPENMP
        omp_init_lock(&(cur_cell->lock));
        #endif
        cur_cell->max_sq_norm = max_sq_norm;
        cur_cell->cell_indices = NULL;
        cur_cell->num_finest_level_points = NULL;
        cur_cell->num_finest_level_nodes = NULL;
        if (i) {  // we don't need to allocate for the finest level
            cur_cell->cell_indices = (btree_p*)malloc(sizeof(btree_p) * num_indices);
            initialize_indices(cur_cell->cell_indices, num_indices);
            cur_cell->num_finest_level_points = (int*)malloc(sizeof(int) * (i + 1));
            cur_cell->num_finest_level_nodes = (int*)malloc(sizeof(int)* (i + 1));
            for (int l = i; l >= 0; l--) {
                cur_cell->num_finest_level_points[l] = 0;
                cur_cell->num_finest_level_nodes[l] = 0;
            }
            cur_cell->num_finest_level_points[0] = 1;
            cur_cell->num_finest_level_nodes[0] = 1;
        }
        cur_cell->flag = 0;
        cur_cell->max_child_dist = 0.;
        cur_cell->parent_dist = 0.;
        cur_cell->data_loc = &(data[(cur_cell->id) * dim]);
        cur_cell->inc_data_loc = &(value[(cur_cell->id) * dim]);
        cur_cell->parent_info = *root;
        cur_cell->local_dist = (float*)malloc(sizeof(float) * num_indices);
        for (k = 0; k < num_indices; k++) {
            cur_cell->local_dist[k] = data_proj[k + (cur_cell->id) * num_indices];
        }
        data_pt cur_point;
        cur_point.info = cur_cell;
        bulk[j].data_pt = cur_point;
        bulk[j].parent_id = -1;
    }

    for (int k = 0; k < num_indices; k++) {
        for (j = 0; j < num_points_on_cur_levels; j++) {
            bulk[j].local_dist = bulk[j].data_pt.info->local_dist[k];
        }
        qsort(bulk, num_points_on_cur_levels, sizeof(bulk_data_pt), dci_compare_data_pt_dist);
        for (j = 0; j < num_points_on_cur_levels; j++) {
            bulk_data_proj[j] = bulk[j].local_dist;
            bulk_data[j] = bulk[j].data_pt;
            (*root)->arr_indices[k*num_points_on_cur_levels+j].info = bulk_data[j].info;
            (*root)->arr_indices[k*num_points_on_cur_levels+j].key = bulk[j].local_dist;
            if (k == 0) {
                bulk_data[j].info->local_id = j;
                (*root)->arr_indices[k*num_points_on_cur_levels+j].local_id = j;
            }
            else {
                (*root)->arr_indices[k*num_points_on_cur_levels+j].local_id = bulk_data[j].info->local_id;
            }
        }
        cur_index = &((*root)->cell_indices[k]);
        bool update_node = 0;
        if (empty_until < 0) {
            update_node = 1;
        }
        btree_p_bulk_load(cur_index, &(bulk_data_proj[0]), &(bulk_data_proj[num_points_on_cur_levels]), &(bulk_data[0]), &(bulk_data[num_points_on_cur_levels]), dci_inst->token2nodeIndex, dci_inst->token2nodeOffset, &(dci_inst->num_leaf_nodes), update_node, dci_inst->stack, &(dci_inst->page_status), dim, dci_inst->update_addr, &(dci_inst->leaf_list), &(dci_inst->max_leaves));
    }
    (*root)->flag = 0;

    additional_info* cur_empty_root = NULL;
    int cur_sub_level = (*actual_num_levels) - 1;  // store the index of the currently lowest empty level

    for (i = (*actual_num_levels) - 2; i >= 0; i--) {
        if (num_points_on_level[i] == 0) {
            continue;
        }
        assert(!data_proj_transposed);

        num_points_on_cur_levels = num_points_on_level[i];

        if (num_points_on_level[i + 1] > 0) {
            int target_level = i + 1;
            int num_upper_points = num_points_on_level[target_level];
            bf16_t* points_matrix = NULL;
            float* points_sq_norms = NULL;

            if (cblas_enabled && num_upper_points > 50) {
                points_matrix = (bf16_t*)malloc(num_upper_points * dim * sizeof(bf16_t));
                // Pack points into contiguous matrix
                pack_points_to_matrix(level_cells_ret[target_level], num_upper_points, dim, points_matrix);
                points_sq_norms = (float*)malloc(num_upper_points * sizeof(float));
                compute_points_sq_norms(points_matrix, num_upper_points, dim, points_sq_norms);
            }

            if ((i == 0) && (interval > 0) && (!random) && (num_points_on_cur_levels > dci_inst->numa_threshold)) {
                // printf("interval=%d, num_points_on_cur_levels=%d\n", interval, num_points_on_cur_levels);
                // Anchor point optimization: first assign parents to anchor points
                // Anchors are: multiples of interval (0, interval, 2*interval, ...) + last point if needed
                int num_regular_anchors = (num_points_on_cur_levels + interval - 1) / interval;
                int last_pos = num_points_on_cur_levels - 1;
                bool need_last_anchor = (last_pos % interval != 0) && (last_pos > 0);
                int num_anchors = num_regular_anchors + (need_last_anchor ? 1 : 0);

                long long anchor_indices[num_anchors];
                tree_node anchor_parents[num_anchors];

                // Step 1: Collect anchor point indices
                // Regular anchors at multiples of interval
                for (int anchor_idx = 0; anchor_idx < num_regular_anchors; anchor_idx++) {
                    int pos = anchor_idx * interval;
                    anchor_indices[anchor_idx] = level_members[i][pos];
                }
                // Add last point only if it's not already covered
                if (need_last_anchor) {
                    anchor_indices[num_regular_anchors] = level_members[i][last_pos];
                }

                // Step 2: Assign parents to anchor points using original logic
                if (cur_empty_root != NULL) {
                    dci_assign_parent(num_comp_indices, num_simp_indices, dim,
                        cur_sub_level + 1, cur_empty_root, cur_sub_level - i,
                        num_anchors, anchor_indices, data,
                        data_proj, construction_query_config,
                        anchor_parents, dci_inst, 
                        num_points_on_level[target_level], level_cells_ret[target_level], points_matrix, points_sq_norms);
                } else {
                    dci_assign_parent(num_comp_indices, num_simp_indices, dim,
                        (*actual_num_levels), *root, (*actual_num_levels) - i - 1,
                        num_anchors, anchor_indices, data,
                        data_proj, construction_query_config,
                        anchor_parents, dci_inst, 
                        num_points_on_level[target_level], level_cells_ret[target_level], points_matrix, points_sq_norms);
                }

                // Update the max_child_dist for each anchor parent
                for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
                    additional_info* parent_info = anchor_parents[anchor_idx].parent;
                    float dist_to_child = anchor_parents[anchor_idx].dist;
                    if (dist_to_child > parent_info->max_child_dist) {
                        parent_info->max_child_dist = dist_to_child;
                    }
                }

                // Step 3: For all points, determine parents using anchor optimization
                int num_on_hold = 0;

                long long* points_needing_query = NULL;
                long long* points_needing_query_ids = NULL;
                int num_points_needing_query = 0;
                if (cblas_enabled && num_upper_points > 1000) {
                    // Allocate array to collect points that need query-based assignment
                    points_needing_query = (long long*)malloc(num_points_on_cur_levels * sizeof(long long));
                    points_needing_query_ids = (long long*)malloc(num_points_on_cur_levels * sizeof(long long));
                }

                int adaptive_threads = dci_inst->inner_threads;
                int chunk_size = (num_points_on_cur_levels / adaptive_threads) / 4;
                chunk_size = (chunk_size < 1) ? 1 : chunk_size;
#pragma omp parallel for if(dci_inst->parallel_level >= 2) num_threads(adaptive_threads) schedule(dynamic, chunk_size)
                for (int point_idx = 0; point_idx < num_points_on_cur_levels; point_idx++) {
                    assigned_parent[point_idx].child = level_members[i][point_idx];

                    // If this is an anchor point (multiple of interval or last point), use pre-computed parent
                    if (point_idx % interval == 0) {
                        // Regular anchor point
                        int anchor_idx = point_idx / interval;
                        assigned_parent[point_idx].parent = anchor_parents[anchor_idx].parent;
                        assigned_parent[point_idx].dist = anchor_parents[anchor_idx].dist;
                    }
                    else if (need_last_anchor && point_idx == last_pos) {
                        // Last anchor point
                        assigned_parent[point_idx].parent = anchor_parents[num_regular_anchors].parent;
                        assigned_parent[point_idx].dist = anchor_parents[num_regular_anchors].dist;
                    }
                    else {
                        // Non-anchor point: check X anchors on left and right
                        int left_anchor_base = point_idx / interval;
                        int right_anchor_base = left_anchor_base + 1;

                        long long cur_point_id = level_members[i][point_idx];
                        const bf16_t* cur_point_data = &data[cur_point_id * dim];
                        const float cur_point_proj = data_proj[cur_point_id];

                        // Find the closest anchor parent among X left and X right anchors
                        // First collect unique anchor parents to avoid duplicate distance computations
                        additional_info* unique_parents[2 * X];
                        int unique_parent_count = 0;

                        // Collect X left anchors
                        for (int j = 0; j < X && (left_anchor_base - j) >= 0; j++) {
                            int anchor_idx = left_anchor_base - j;
                            if (anchor_idx < num_anchors && anchor_parents[anchor_idx].parent != NULL) {
                                additional_info* parent = anchor_parents[anchor_idx].parent;
                                // Check if this parent is already in the list
                                bool already_added = false;
                                for (int k = 0; k < unique_parent_count; k++) {
                                    if (unique_parents[k] == parent) {
                                        already_added = true;
                                        break;
                                    }
                                }
                                if (!already_added) {
                                    unique_parents[unique_parent_count++] = parent;
                                }
                            }
                        }

                        // Collect X right anchors
                        for (int j = 0; j < X && (right_anchor_base + j) < num_anchors; j++) {
                            int anchor_idx = right_anchor_base + j;
                            if (anchor_parents[anchor_idx].parent != NULL) {
                                additional_info* parent = anchor_parents[anchor_idx].parent;
                                // Check if this parent is already in the list
                                bool already_added = false;
                                for (int k = 0; k < unique_parent_count; k++) {
                                    if (unique_parents[k] == parent) {
                                        already_added = true;
                                        break;
                                    }
                                }
                                if (!already_added) {
                                    unique_parents[unique_parent_count++] = parent;
                                }
                            }
                        }

                        // Now compute distances to unique parents only
                        additional_info* closest_parent = NULL;
                        float min_dist_to_parent = FLT_MAX;

                        for (int k = 0; k < unique_parent_count; k++) {
                            const bf16_t* parent_data = unique_parents[k]->data_loc;
                            long long parent_id = unique_parents[k]->id;

                            float dist;
                            if (dci_inst->transform) {
                                dist = transform_compute_dist(cur_point_data, parent_data, dim,
                                                             dci_inst->max_sq_norm,
                                                             dci_inst->sq_norm_list[cur_point_id],
                                                             dci_inst->sq_norm_list[parent_id]);
                            } else {
                                dist = compute_dist(cur_point_data, parent_data, dim);
                            }

                            if (dist < min_dist_to_parent) {
                                min_dist_to_parent = dist;
                                closest_parent = unique_parents[k];
                            }
                        }

                        // Now check if max children distance of closest parent > min_dist_to_parent
                        bool use_anchor_parent = false;
                        if (closest_parent != NULL) {
                            // Use pre-computed max_child_dist from additional_info
                            float max_children_dist = closest_parent->max_child_dist;

                            // If max children distance >= current point's distance to parent, use anchor parent
                            if (max_children_dist * anchor_threshold >= min_dist_to_parent) {
                                use_anchor_parent = true;
                            }
                        }

                        if (use_anchor_parent && closest_parent != NULL) {
                            // Use the anchor parent directly
                            assigned_parent[point_idx].parent = closest_parent;
                            assigned_parent[point_idx].dist = min_dist_to_parent;

                            // Update parent's max_child_dist
                            if (min_dist_to_parent > closest_parent->max_child_dist) {
                                closest_parent->max_child_dist = min_dist_to_parent;
                            }
                        } else {
                            num_on_hold++;
                            if (cblas_enabled && num_upper_points > 1000) {
                                // Collect this point for batch query-based assignment
                                int local_idx;
                                #pragma omp atomic capture
                                local_idx = num_points_needing_query++;
                                points_needing_query[local_idx] = point_idx;
                                points_needing_query_ids[local_idx] = cur_point_id;
                            }
                            else {
                                // Assign parent on the fly using dci_assign_parent
                                idx_arr top_candidate;
                                int target_level = i + 1;
                                if (cur_empty_root != NULL) {
                                    dci_query_single_point_(
                                        num_comp_indices, num_simp_indices, dim,
                                        cur_sub_level + 1, cur_empty_root, cur_sub_level - i,
                                        1, cur_point_data, cur_point_proj,
                                        construction_query_config,
                                        &top_candidate, false,
                                        dci_inst, level_members[i][point_idx],
                                        num_points_on_level[target_level], level_cells_ret[target_level], NULL, NULL, NULL);
                                }
                                else {
                                    dci_query_single_point_(
                                        num_comp_indices, num_simp_indices, dim,
                                        (*actual_num_levels), *root, (*actual_num_levels) - i - 1,
                                        1, cur_point_data, cur_point_proj,
                                        construction_query_config,
                                        &top_candidate, false,
                                        dci_inst, level_members[i][point_idx],
                                        num_points_on_level[target_level], level_cells_ret[target_level], NULL, NULL, NULL);
                                }
                                assigned_parent[point_idx].parent = top_candidate.info;
                                assigned_parent[point_idx].dist = top_candidate.key;
                            }
                        }
                    }
                }

                // printf("num_on_hold: %d\n", num_on_hold);

                // Batch assign parents for points that couldn't use anchor optimization
                if (num_points_needing_query > 0) {
                    // Create temporary array for batch assignment results
                    tree_node* batch_assigned_parents = (tree_node*)malloc(num_points_needing_query * sizeof(tree_node));

                    int target_level = i + 1;
                    if (cur_empty_root != NULL) {
                        dci_assign_parent(num_comp_indices, num_simp_indices, dim,
                            cur_sub_level + 1, cur_empty_root, cur_sub_level - i,
                            num_points_needing_query, points_needing_query_ids, data,
                            data_proj, construction_query_config,
                            batch_assigned_parents, dci_inst,
                            num_points_on_level[target_level], level_cells_ret[target_level], points_matrix, points_sq_norms);
                    } else {
                        dci_assign_parent(num_comp_indices, num_simp_indices, dim,
                            (*actual_num_levels), *root, (*actual_num_levels) - i - 1,
                            num_points_needing_query, points_needing_query_ids, data,
                            data_proj, construction_query_config,
                            batch_assigned_parents, dci_inst,
                            num_points_on_level[target_level], level_cells_ret[target_level], points_matrix, points_sq_norms);
                    }

                    // Copy batch results back to assigned_parent and update max_child_dist
                    for (int j = 0; j < num_points_needing_query; j++) {
                        int point_idx = points_needing_query[j];
                        assigned_parent[point_idx].parent = batch_assigned_parents[j].parent;
                        assigned_parent[point_idx].dist = batch_assigned_parents[j].dist;

                        // Update parent's max_child_dist
                        if (batch_assigned_parents[j].parent != NULL &&
                            batch_assigned_parents[j].dist > batch_assigned_parents[j].parent->max_child_dist) {
                            batch_assigned_parents[j].parent->max_child_dist = batch_assigned_parents[j].dist;
                        }
                    }
                    
                    free(batch_assigned_parents);
                }

                free(points_needing_query);
                free(points_needing_query_ids);
                // printf("num_on_hold: %d\n", num_points_needing_query);
            }
            else {
                // Use original assign_parent logic when anchor optimization is not applied
                int target_level = i + 1;
                if (cur_empty_root != NULL) {
                    dci_assign_parent(num_comp_indices, num_simp_indices, dim,
                        cur_sub_level + 1, cur_empty_root, cur_sub_level - i,
                        num_points_on_level[i], level_members[i], data,
                        data_proj, construction_query_config,
                        assigned_parent, dci_inst,
                        num_points_on_level[target_level], level_cells_ret[target_level], points_matrix, points_sq_norms);
                }
                else {
                    dci_assign_parent(num_comp_indices, num_simp_indices, dim,
                        (*actual_num_levels), *root, (*actual_num_levels) - i - 1,
                        num_points_on_level[i], level_members[i], data, 
                        data_proj, construction_query_config,
                        assigned_parent, dci_inst,
                        num_points_on_level[target_level], level_cells_ret[target_level], points_matrix, points_sq_norms);
                }
            }

            if (cblas_enabled && num_upper_points > 1000) {
                free(points_matrix);
                free(points_sq_norms);
            }

            btree_p* cur_index;
            for (j = 0; j < num_points_on_cur_levels; j++) {
                int k;
                long long cur_id = assigned_parent[j].child;

                additional_info* cur_cell = level_cells_ret[i][j];
                cur_cell->id = cur_id;
                cur_cell->arr_indices = NULL;
                #ifdef USE_OPENMP
                omp_init_lock(&(cur_cell->lock));
                #endif
                cur_cell->max_sq_norm = max_sq_norm;
                cur_cell->cell_indices = NULL;
                cur_cell->num_finest_level_points = NULL;
                cur_cell->num_finest_level_nodes = NULL;
                if (i) {  // we don't need to allocate for the finest level
                    cur_cell->cell_indices = (btree_p*)malloc(sizeof(btree_p) * num_indices);
                    initialize_indices(cur_cell->cell_indices, num_indices);
                    cur_cell->num_finest_level_points = (int*)malloc(sizeof(int) * (i + 1));
                    cur_cell->num_finest_level_nodes = (int*)malloc(sizeof(int) * (i + 1));
                    for (int l = i; l >= 0; l--) {
                        cur_cell->num_finest_level_points[l] = 0;
                        cur_cell->num_finest_level_nodes[l] = 0;
                    }
                    cur_cell->num_finest_level_points[0] = 1;
                    cur_cell->num_finest_level_nodes[0] = 1;
                }
                cur_cell->flag = 0;
                cur_cell->data_loc = &(data[cur_id * dim]);
                cur_cell->inc_data_loc = &(value[cur_id * dim]);
                cur_cell->max_child_dist = 0.;
                cur_cell->parent_dist = assigned_parent[j].dist;  // distance to the id
                cur_cell->parent_info = assigned_parent[j].parent;
                if (cur_cell->parent_dist > cur_cell->parent_info->max_child_dist) {
                    cur_cell->parent_info->max_child_dist = cur_cell->parent_dist;
                }
                cur_cell->local_dist = (float*)malloc(sizeof(float) * num_indices);
                for (k = 0; k < num_indices; k++) {
                    cur_cell->local_dist[k] = data_proj[k + cur_id * num_indices];
                }
                data_pt cur_point;
                cur_point.info = cur_cell;
                bulk[j].data_pt = cur_point;
                bulk[j].parent_id = assigned_parent[j].parent->id;
                for (additional_info* temp_cell = cur_cell->parent_info;
                    temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
                    temp_cell->num_finest_level_points[i + 1] += 1;
                    temp_cell->num_finest_level_points[0] += 1;
                }
            }

            // btree_p_bulk_load()
            // first iter: sort by parent id (go over to see the change of p_id: [prev_i: last_i])
            // second iter: for each index sort between [prev_i: last_i]
            // just one array for 14 indices one by one
            qsort(bulk, num_points_on_cur_levels, sizeof(bulk_data_pt), dci_compare_data_pt_parent);
            int p_idx = 0;
            parent_idx[p_idx++] = 0;
            for (j=1; j < num_points_on_cur_levels; j++) {
                if (bulk[j].parent_id != bulk[j-1].parent_id) {
                    bulk[j-1].data_pt.info->parent_info->arr_indices = (idx_arr*)malloc(sizeof(idx_arr)*(num_indices*(j - parent_idx[p_idx-1])));
                    parent_idx[p_idx++] = j;
                }
            }
            parent_idx[p_idx] = num_points_on_cur_levels;
            bulk[j-1].data_pt.info->parent_info->arr_indices = (idx_arr*)malloc(sizeof(idx_arr)*(num_indices*(j - parent_idx[p_idx-1])));
            int p, begin, end;
            for (int k = 0; k < num_indices; k++) {
                for (j=0; j < num_points_on_cur_levels; j++) {
                    bulk[j].local_dist = bulk[j].data_pt.info->local_dist[k];
                }
                for (p=0; p<p_idx; p++) {
                    begin = parent_idx[p];
                    end = parent_idx[p + 1];
                    qsort(&(bulk[begin]), end - begin, sizeof(bulk_data_pt), dci_compare_data_pt_dist);
                    additional_info* parent = bulk[begin].data_pt.info->parent_info;
                    for (j = begin; j < end; j++) {
                        bulk_data_proj[j] = bulk[j].local_dist;
                        bulk_data[j] = bulk[j].data_pt;
                        parent->arr_indices[k*(end-begin)+(j-begin)].info = bulk_data[j].info;
                        parent->arr_indices[k*(end-begin)+(j-begin)].key = bulk_data_proj[j];
                        if (k == 0) {
                            bulk_data[j].info->local_id = j-begin;
                            parent->arr_indices[k*(end-begin)+(j-begin)].local_id = j-begin;
                        }
                        else {
                            parent->arr_indices[k*(end-begin)+(j-begin)].local_id = bulk_data[j].info->local_id;
                        }
                    }
                    cur_index = &(parent->cell_indices[k]);
                    bool update_node = 0;
                    if (empty_until < 0)
                        update_node = 1;
                    btree_p_bulk_load(cur_index, &(bulk_data_proj[begin]), &(bulk_data_proj[end]), &(bulk_data[begin]), &(bulk_data[end]), dci_inst->token2nodeIndex, dci_inst->token2nodeOffset, &(dci_inst->num_leaf_nodes), update_node, dci_inst->stack, &(dci_inst->page_status), dim, dci_inst->update_addr, &(dci_inst->leaf_list), &(dci_inst->max_leaves));
                }
            }
        }
        else {
            btree_p* cur_index;
            for (j = 0; j < num_points_on_cur_levels; j++) {
                int k;
                long long cur_id = level_members[i][j];

                additional_info* cur_cell = level_cells_ret[i][j];
                cur_cell->id = cur_id;
                cur_cell->max_sq_norm = max_sq_norm;
                cur_cell->local_dist = (float*)malloc(sizeof(float) * num_indices);
                cur_cell->cell_indices = NULL;
                #ifdef USE_OPENMP
                omp_init_lock(&(cur_cell->lock));
                #endif
                cur_cell->arr_indices = NULL;
                cur_cell->num_finest_level_points = NULL;
                cur_cell->num_finest_level_nodes = NULL;
                if (i) {  // we don't need to allocate for the finest level
                    cur_cell->cell_indices = (btree_p*)malloc(sizeof(btree_p) * num_indices);
                    initialize_indices(cur_cell->cell_indices, num_indices);
                    cur_cell->num_finest_level_points = (int*)malloc(sizeof(int) * (i + 1));
                    cur_cell->num_finest_level_nodes = (int*)malloc(sizeof(int) * (i + 1));
                    for (int l = i; l >= 0; l--) {
                        cur_cell->num_finest_level_points[l] = 0;
                        cur_cell->num_finest_level_nodes[l] = 0;
                    }
                    cur_cell->num_finest_level_points[0] = 1;
                    cur_cell->num_finest_level_nodes[0] = 1;
                }
                cur_cell->flag = 0;
                cur_cell->data_loc = &(data[cur_id * dim]);
                cur_cell->inc_data_loc = &(value[cur_id * dim]);
                cur_cell->parent_dist = 0.;
                cur_cell->parent_info = NULL;
                data_pt cur_point;
                cur_point.info = cur_cell;
                bulk[j].data_pt = cur_point;
                bulk[j].parent_id = -1;
                
                for (k = 0; k < num_indices; k++) {
                    cur_cell->local_dist[k] = data_proj[k + cur_id * num_indices];
                }
            }

            cur_sub_level = i;
            cur_empty_root = create_sub_root(bulk, num_points_on_cur_levels, num_indices, max_sq_norm, dci_inst->token2nodeIndex, dci_inst->token2nodeOffset, &(dci_inst->num_leaf_nodes), dci_inst->stack, &(dci_inst->page_status), dim, dci_inst->update_addr, &(dci_inst->leaf_list), &(dci_inst->max_leaves));
            dci_inst->sub_root_list[i + 1] = cur_empty_root;
            for (j = 0; j < num_points_on_cur_levels; j++) {
                level_cells_ret[i][j]->parent_info = cur_empty_root;
            }
        }
    }

    free(bulk);
    free(assigned_parent);
    free(bulk_data_proj);
    free(bulk_data);
    free(parent_idx);

    assert(num_points_on_cur_levels == num_points_on_level[0]);

    if (*actual_num_levels >= 2) {
        for (i = 0; i < *actual_num_levels; i++) {
            free(level_members[i]);
        }
    }
    else
        free(level_members[0]); 

    free(level_members);
}

int dci_delete(dci* const dci_inst, const int num_points, const long long* const data_ids, 
    dci_query_config deletion_config, long long* duplicate_delete_ids) {
    additional_info* cur_point;  // contains cells of each level, not the finest level though
    int* num_points_on_level;
    int i, j, k, h;
    int num_points_on_upper_levels = 1, num_points_on_cur_levels = 0;
    int num_indices = dci_inst->num_simp_indices * dci_inst->num_comp_indices;
    btree_p_search_res s;
    btree_p* cur_tree;
    int num_levels = dci_inst->num_levels;
    float lambda = 0.5; // the ratio of the expected number of points
    bf16_t* temp_data_loc = NULL;
    bf16_t* temp_inc_data_loc = NULL;
    int dim = dci_inst->dim;
    bool update_addr = dci_inst->update_addr;

    hashtable_i to_delete; // whether a point should be deleted or not
    hashtable_i_init(&to_delete, num_points, 1);
    for (i = 0; i < num_points; i++) {
        hashtable_i_set(&to_delete, data_ids[i], i);
    }

    // we calculated the points before to avoid missing some points because of deletion in its parent
    float promotion_prob = pow((float)(dci_inst->num_points), -1.0 / dci_inst->num_levels);
    additional_info*** deleted_points = (additional_info***)malloc(sizeof(additional_info**) * (dci_inst->num_levels));
    int* delete_level_num = (int*)malloc(sizeof(int) * dci_inst->num_levels);
    int* expected_level_num = (int*)malloc(sizeof(int) * dci_inst->num_levels);
    int* rest_level_num = (int*)malloc(sizeof(int) * dci_inst->num_levels);

    num_points_on_level = dci_inst->num_points_on_level;

    // Delete the whole tree
    if (dci_inst->num_points == num_points) {
        dci_reset(dci_inst);
        return num_points;
    }

    // Initialize delete arrays
    for (i = dci_inst->num_levels - 1; i >= 0; i--) {
        deleted_points[i] = (additional_info**)malloc(sizeof(additional_info*) * (num_points));
        delete_level_num[i] = 0;
    }

    int duplicate_num = 0;
    // Calculate the delete number for each level
    for (h = 0; h < num_points; h++) {
        addinfo_level* addinfo_l = hashtable_p_get(dci_inst->inserted_points, data_ids[h], NULL);
        if (addinfo_l == NULL) {
            duplicate_delete_ids[duplicate_num++] = data_ids[h];
            continue;
        }
        i = addinfo_l->level;
        deleted_points[i][delete_level_num[i]] = addinfo_l->addinfo;
        delete_level_num[i] += 1;
    }

    // Assure at the beginning, all the level are larger than the expected number
    int ii = 0;
    int iii = 0;
    i = dci_inst->num_levels - 1;
    expected_level_num[i] = (int)ceil((dci_inst->num_points - num_points + duplicate_num) *
        pow(promotion_prob, i) * lambda);
    while (true) {
        for (i = dci_inst->num_levels - 2; i >= 1; i--) {
            expected_level_num[i] = (int)ceil(expected_level_num[i + 1] / promotion_prob) - 1;
            if (expected_level_num[i] <= num_points_on_level[i] && 
                    expected_level_num[i] <= dci_inst->num_points - num_points + duplicate_num) {
                continue;
            }
            else {
                break;
            }
        }
        if (i <= 0) break;
        else expected_level_num[dci_inst->num_levels - 1] -= 1;
        if (expected_level_num[dci_inst->num_levels - 1] == 0) {
            promotion_prob = (5 * promotion_prob + 1) / 6;
            assert(promotion_prob < 1);
            i = dci_inst->num_levels - 1;
            expected_level_num[i] = (int)ceil((dci_inst->num_points - num_points + duplicate_num) *
                pow(promotion_prob, i) * lambda) + 1;
        }
        assert(expected_level_num[dci_inst->num_levels - 1] > 0);
    }
    if (expected_level_num[dci_inst->num_levels - 1] > num_points_on_level[dci_inst->num_levels - 1])
        expected_level_num[dci_inst->num_levels - 1] = num_points_on_level[dci_inst->num_levels - 1];
    expected_level_num[0] = 0;
    
    // Calculate the rest number of each level
    for (i = dci_inst->num_levels - 1; i >= 0; i--) {
        assert(expected_level_num[i] <= num_points_on_level[i]);
        rest_level_num[i] = num_points_on_level[i] - delete_level_num[i] - expected_level_num[i];
    }

    // Do the rehearsal of deletion
    int last_level = 0;
    int down_to = dci_inst->num_levels - 1;

    for (i = dci_inst->num_levels - 1; i > last_level; i--) {
        if (rest_level_num[i] < 0) {
            assert(i - 1 >= last_level);
            for (ii = i - 1; ii >= last_level && ii < i; ii--) {
                if (rest_level_num[ii] + rest_level_num[i] >= 0) {
                    if (ii < down_to) {
                        down_to = ii;
                    }
                    rest_level_num[ii] += rest_level_num[i];
                    rest_level_num[i] = 0;
                    break;
                }
                else if (rest_level_num[ii] > 0) {
                    rest_level_num[i] += rest_level_num[ii];
                    rest_level_num[ii] = 0;
                }
                // run out of all the available points
                if (ii == last_level) {
                    down_to = 0;
                    assert(rest_level_num[ii] == 0);
                    // change the last level by moving one level up
                    last_level += 1;
                    rest_level_num[last_level] += expected_level_num[last_level];
                    expected_level_num[last_level] = 0;
                    assert(rest_level_num[last_level] >= 0);
                    ii += 2;
                }
            }
        }
        assert(rest_level_num[i] >= 0);
    }

    // Finetune the expected_level_num and make expected num be the final num
    if (rest_level_num[last_level] == 0) {
        last_level += 1;
    }

    int total_expected = 0;
    for (i = dci_inst->num_levels - 1; i >= last_level; i--) {
        assert(rest_level_num[i] >= 0);
        expected_level_num[i] += rest_level_num[i];
        total_expected += expected_level_num[i];
    }
    assert(dci_inst->num_points - num_points + duplicate_num == total_expected);

    int num_deleted = 0;
    assert(num_points_on_level[dci_inst->num_levels - 1] == dci_inst->root->cell_indices[0].num_data);

    // START to delete (from top-most level to down_to)
    for (i = dci_inst->num_levels - 1; i >= 0; i--) {
        for (ii = 0; ii < delete_level_num[i]; ii++) {
            num_deleted += 1;
            cur_point = deleted_points[i][ii];

            addinfo_level* addinfo_l = hashtable_p_get(dci_inst->inserted_points, cur_point->id, NULL);
            if (addinfo_l->offset < num_points_on_level[i] - 1) {
                additional_info* shifted_data = dci_inst->points_on_level[i][num_points_on_level[i] - 1];
                dci_inst->points_on_level[i][addinfo_l->offset] = shifted_data;
                hashtable_p_set(dci_inst->inserted_points, shifted_data->id, shifted_data, i, addinfo_l->offset);
            }
            hashtable_p_delete(dci_inst->inserted_points, cur_point->id);
            num_points_on_level[i] -= 1;

            if (cur_point->parent_info != NULL) {  // delete that point from its parent, update num_finest_level_points
                for (k = 0; k < num_indices; k++) {
                    assert(i < dci_inst->num_levels - 1 || cur_point->parent_info == dci_inst->root);
                    bool ret = btree_p_delete(&(cur_point->parent_info->cell_indices[k]),
                        cur_point->local_dist[k], cur_point->id, dci_inst->token2nodeIndex, dci_inst->token2nodeOffset, 
                    &(dci_inst->num_leaf_nodes), dci_inst->stack, dci_inst->page_status, dim, dci_inst->update_addr);
                    assert(ret);
                }
                cur_point->parent_info->flag = 1;
                for (additional_info* temp_cell = cur_point->parent_info;
                    temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
                    temp_cell->num_finest_level_points[i + 1] -= 1;
                    assert(temp_cell->num_finest_level_points[i + 1] >= 0);
                    if (i > 0) {
                        for (k = 0; k <= i; k++) {
                            temp_cell->num_finest_level_points[k] -= cur_point->num_finest_level_points[k];
                            assert(temp_cell->num_finest_level_points[k] >= 0);
                        }
                    }
                    else {
                        temp_cell->num_finest_level_points[0] -= 1;
                        assert(temp_cell->num_finest_level_points[0] >= 0);
                    }
                    if (temp_cell->parent_info == NULL) break;
                }
                cur_point->parent_info = NULL;
            }
            assert(i >= last_level);
            if (num_points_on_level[i] < expected_level_num[i]) {  // if imbalanced, PROMOTE one point from the next lower level by setting a threshold
                int promote = 0;
                additional_info* promote_point = NULL;
                for (iii = i - 1; iii >= 0; iii--) {
                    if (num_points_on_level[iii] - delete_level_num[iii] > expected_level_num[iii]) {
                        for (int jj = num_points_on_level[iii] - 1; jj >= 0; jj--) {
                            if (!hashtable_i_exists(&to_delete, dci_inst->points_on_level[iii][jj]->id)) {  // if the candidate will not be deleted, promote it 
                                promote_point = dci_inst->points_on_level[iii][jj];
                                promote = 1;
                                addinfo_level* addinfo_l = hashtable_p_get(dci_inst->inserted_points, promote_point->id, NULL);
                                assert(addinfo_l->level == iii);
                                assert(jj == addinfo_l->offset);
                                if (addinfo_l->offset < num_points_on_level[iii] - 1) {
                                    additional_info* shifted_data = dci_inst->points_on_level[iii][num_points_on_level[iii] - 1];
                                    dci_inst->points_on_level[iii][addinfo_l->offset] = shifted_data;
                                    hashtable_p_set(dci_inst->inserted_points, shifted_data->id, shifted_data, iii, addinfo_l->offset);   
                                }
                                hashtable_p_set(dci_inst->inserted_points, promote_point->id, promote_point, i, num_points_on_level[i]);
                                if (num_points_on_level[i] >= dci_inst->max_num_on_level[i]) {
                                    dci_inst->max_num_on_level[i] *= 2;
                                    dci_inst->points_on_level[i] = (additional_info**)realloc(dci_inst->points_on_level[i], sizeof(additional_info*) * dci_inst->max_num_on_level[i]);
                                }
                                dci_inst->points_on_level[i][num_points_on_level[i]] = promote_point;
                                num_points_on_level[iii] -= 1;
                                num_points_on_level[i] += 1;
                                break;
                            }
                            assert(jj != 0);
                        }
                        break;
                    }
                }
                assert(promote == 1);

                // if one specific lower level doesn't have enough data, no promotion and go down
                // otherwise promote
                // if promote, do modification FROM TOP TO BOTTOM!
                if (promote) {
                    btree_p* temp_index = promote_point->cell_indices;
                    if (iii >= last_level) {  // delete the promote data from its old parent
                        assert(promote_point->parent_info != NULL);
                        if (promote_point->parent_info != NULL) {
                            if (update_addr) {
                                temp_data_loc = (bf16_t*)malloc(sizeof(bf16_t) * dim);
                                temp_inc_data_loc = (bf16_t*)malloc(sizeof(bf16_t) * dim);
                                for (k = 0; k < dim; k++) {  // copy the data_loc since the old data_loc will be overwritten in btree_p_delete
                                    temp_data_loc[k] = promote_point->data_loc[k];
                                    temp_inc_data_loc[k] = promote_point->inc_data_loc[k];
                                }
                                promote_point->data_loc = temp_data_loc;
                                promote_point->inc_data_loc = temp_inc_data_loc;
                            }
                            for (k = 0; k < num_indices; k++) {
                                bool ret = btree_p_delete(&(promote_point->parent_info->cell_indices[k]),
                                    promote_point->local_dist[k], promote_point->id, dci_inst->token2nodeIndex, dci_inst->token2nodeOffset, &(dci_inst->num_leaf_nodes), 
                                    dci_inst->stack, dci_inst->page_status, dim, dci_inst->update_addr);
                                assert(ret);
                            }
                            promote_point->parent_info->flag = 1;
                            for (additional_info* temp_cell = promote_point->parent_info;
                                temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
                                temp_cell->num_finest_level_points[iii + 1] -= 1;
                                assert(temp_cell->num_finest_level_points[iii + 1] >= 0);
                                if (iii > 0) {
                                    for (k = 0; k <= iii; k++) {
                                        temp_cell->num_finest_level_points[k] -= promote_point->num_finest_level_points[k];
                                        assert(temp_cell->num_finest_level_points[k] >= 0);
                                    }
                                }
                                else {
                                    temp_cell->num_finest_level_points[0] -= 1;
                                    assert(temp_cell->num_finest_level_points[0] >= 0);
                                }
                                if (temp_cell->parent_info == NULL) break;
                            }
                            promote_point->parent_info = NULL;
                        }
                    }

                    // initialize promote_point
                    promote_point->cell_indices = (btree_p*)malloc(sizeof(btree_p) * num_indices);
                    initialize_indices(promote_point->cell_indices, num_indices);
                    #ifdef USE_OPENMP
                    omp_init_lock(&(promote_point->lock));
                    #endif
                    promote_point->arr_indices = NULL;
                    // promote_point->local_dist = (float*)malloc(sizeof(float) * num_indices);
                    promote_point->flag = 1;
                    if (promote_point->num_finest_level_points) {
                        free(promote_point->num_finest_level_points);
                    }
                    promote_point->num_finest_level_points = (int*)malloc(sizeof(int) * (i + 1));
                    for (int l = i; l >= 0; l--) {
                        promote_point->num_finest_level_points[l] = 0;
                    }
                    promote_point->num_finest_level_points[0] += 1;

                    // find new parent for the promoted point
                    assert(!hashtable_i_exists(&to_delete, promote_point->id));
                    if (i == dci_inst->num_levels - 1) { // highest level
                        promote_point->parent_info = dci_inst->root;
                        promote_point->parent_dist = 0.;
                        assert(dci_inst->root->cell_indices != NULL);

                        dci_insert_to_indices(dci_inst, dci_inst->transform, num_indices, 
                            dci_inst->root, promote_point,
                            dci_inst->max_sq_norm, dci_inst->add_proj_vec, dci_inst->sq_norm_list);
                    }
                    else {
                        idx_arr top_candidate;
                        dci_query_single_point_(
                            dci_inst->num_comp_indices, dci_inst->num_simp_indices, dci_inst->dim,
                            dci_inst->num_levels, dci_inst->root,
                            dci_inst->num_levels - i - 1, 1,
                            promote_point->data_loc,
                            promote_point->local_dist[0],
                            deletion_config,
                            &top_candidate,
                            false,
                            dci_inst, promote_point->id, 
                            dci_inst->num_points_on_level[i + 1], dci_inst->points_on_level[i + 1], NULL, NULL, NULL);
                        promote_point->parent_info = top_candidate.info;
                        promote_point->parent_dist = top_candidate.key;

                        assert(promote_point->parent_info->cell_indices != NULL);

                        dci_insert_to_indices(dci_inst, dci_inst->transform, num_indices, 
                            promote_point->parent_info, promote_point,
                            dci_inst->max_sq_norm, dci_inst->add_proj_vec, dci_inst->sq_norm_list);
                        
                        for (additional_info* temp_cell = promote_point->parent_info;
                            temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
                            temp_cell->num_finest_level_points[i + 1] += 1;
                            temp_cell->num_finest_level_points[0] += 1;
                        }
                    }
                    if (update_addr && temp_data_loc != NULL) {
                        free(temp_data_loc);
                        free(temp_inc_data_loc);
                    }

                    if (i > last_level) {  // Find new parents for points in the cell of this deleted point (if needed)
                        for (s = btree_p_first(cur_point->cell_indices); !btree_p_is_end(cur_point->cell_indices, s);
                            s = btree_p_find_next(s)) {
                            additional_info* cur_p = btree_p_valueof(s).info;
                            idx_arr top_candidate;
                            
                            dci_query_single_point_(
                                dci_inst->num_comp_indices, dci_inst->num_simp_indices, dci_inst->dim,
                                dci_inst->num_levels, dci_inst->root,
                                dci_inst->num_levels - i, 1,
                                cur_p->data_loc,
                                cur_p->local_dist[0],
                                deletion_config,
                                &top_candidate,
                                false,
                                dci_inst, cur_p->id, 
                                dci_inst->num_points_on_level[i], dci_inst->points_on_level[i], NULL, NULL, NULL);
                            assert(cur_p->parent_info->cell_indices != NULL);
                            cur_p->parent_info = top_candidate.info;
                            cur_p->parent_dist = top_candidate.key;

                            dci_insert_to_indices(dci_inst, dci_inst->transform, num_indices, 
                                cur_p->parent_info, cur_p,
                                dci_inst->max_sq_norm, dci_inst->add_proj_vec, dci_inst->sq_norm_list);

                            for (additional_info* temp_cell = cur_p->parent_info;
                                temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
                                temp_cell->num_finest_level_points[i] += 1;
                                if (i > 1) {
                                    for (k = 0; k <= i - 1; k++) {
                                        temp_cell->num_finest_level_points[k] += cur_p->num_finest_level_points[k];
                                    }
                                }
                                else {
                                    temp_cell->num_finest_level_points[0] += 1;
                                }
                            }
                        }

                        for (int jj = num_points_on_level[i - 1] - 1; jj >= 0; jj--) {  // for the i-1 level, decide if they need to change parent to promote_point
                            additional_info* lower_cell = dci_inst->points_on_level[i - 1][jj];
                            if (lower_cell->parent_info == promote_point) continue;

                            // caculate the distance
                            float cur_dist;
                            if (dci_inst->transform) {
                                cur_dist = transform_compute_dist(promote_point->data_loc, lower_cell->data_loc, dci_inst->dim, dci_inst->max_sq_norm, dci_inst->sq_norm_list[promote_point->id], dci_inst->sq_norm_list[lower_cell->id]);
                            }
                            else {
                                cur_dist = compute_dist(promote_point->data_loc, lower_cell->data_loc, dci_inst->dim);
                            }
                            
                            // need to change the parent if promote point is better
                            if (lower_cell->parent_dist - cur_dist > 1e-8) {
                                additional_info* prev_parent = lower_cell->parent_info;
                                lower_cell->parent_info = promote_point;
                                lower_cell->parent_dist = cur_dist;

                                assert(prev_parent->id != lower_cell->parent_info->id);
                                assert(lower_cell->parent_info->cell_indices != NULL);

                                if (update_addr) {
                                    temp_data_loc = (bf16_t*)malloc(sizeof(bf16_t) * dim);
                                    temp_inc_data_loc = (bf16_t*)malloc(sizeof(bf16_t) * dim);
                                    for (k = 0; k < dim; k++) {  // copy the data_loc since the old data_loc will be overwritten in btree_p_delete
                                        temp_data_loc[k] = lower_cell->data_loc[k];
                                        temp_inc_data_loc[k] = lower_cell->inc_data_loc[k];
                                    }
                                    lower_cell->data_loc = temp_data_loc;
                                    lower_cell->inc_data_loc = temp_inc_data_loc;
                                }
                                for (k = 0; k < num_indices; k++) {
                                    bool ret = btree_p_delete(&(prev_parent->cell_indices[k]), lower_cell->local_dist[k], lower_cell->id, dci_inst->token2nodeIndex, dci_inst->token2nodeOffset, 
                                                                &(dci_inst->num_leaf_nodes), dci_inst->stack, dci_inst->page_status, dim, dci_inst->update_addr);
                                    assert(ret);
                                }
                                prev_parent->flag = 1;
                                dci_insert_to_indices(dci_inst, dci_inst->transform, num_indices, 
                                    lower_cell->parent_info, lower_cell,
                                    dci_inst->max_sq_norm, dci_inst->add_proj_vec, dci_inst->sq_norm_list);

                                if (update_addr && temp_data_loc != NULL) {
                                    free(temp_data_loc);
                                    free(temp_inc_data_loc);
                                }

                                for (additional_info* temp_cell = prev_parent;
                                    temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
                                    temp_cell->num_finest_level_points[i] -= 1;
                                    assert(temp_cell->num_finest_level_points[i] >= 0);
                                    if (i > 1) {
                                        for (int l = 0; l <= i - 1; l++) {
                                            temp_cell->num_finest_level_points[l] -= lower_cell->num_finest_level_points[l];
                                            assert(temp_cell->num_finest_level_points[l] >= 0);
                                        }
                                    }
                                    else {
                                        temp_cell->num_finest_level_points[0] -= 1;
                                        assert(temp_cell->num_finest_level_points[0] >= 0);
                                    }
                                    if (temp_cell->parent_info == NULL) break;
                                }
                                for (additional_info* temp_cell = lower_cell->parent_info;
                                    temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
                                    temp_cell->num_finest_level_points[i] += 1;
                                    if (i > 1) {
                                        for (int l = 0; l <= i - 1; l++) {
                                            temp_cell->num_finest_level_points[l] += lower_cell->num_finest_level_points[l];
                                        }
                                    }
                                    else {
                                        temp_cell->num_finest_level_points[0] += 1;
                                    }
                                    if (temp_cell->parent_info == NULL) break;
                                }
                            }
                        }
                    }

                    if (iii > last_level) {  // find parents for the old children of the promoted point
                        assert(temp_index != NULL);
                        for (s = btree_p_first(temp_index); !btree_p_is_end(temp_index, s);
                            s = btree_p_find_next(s)) {
                            additional_info* cur_p = btree_p_valueof(s).info;
                            idx_arr top_candidate;
                            dci_query_single_point_(
                                dci_inst->num_comp_indices, dci_inst->num_simp_indices, dci_inst->dim,
                                dci_inst->num_levels, dci_inst->root,
                                dci_inst->num_levels - iii, 1,
                                cur_p->data_loc,
                                cur_p->local_dist[0],
                                deletion_config,
                                &top_candidate,
                                false,
                                dci_inst, cur_p->id,
                                dci_inst->num_points_on_level[iii], dci_inst->points_on_level[iii], NULL, NULL, NULL);
                            assert(top_candidate.info->cell_indices != NULL);
                            cur_p->parent_info = top_candidate.info;
                            cur_p->parent_dist = top_candidate.key;

                            dci_insert_to_indices(dci_inst, dci_inst->transform, num_indices, 
                                    cur_p->parent_info, cur_p,
                                    dci_inst->max_sq_norm, dci_inst->add_proj_vec, dci_inst->sq_norm_list);

                            for (additional_info* temp_cell = cur_p->parent_info;
                                temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
                                temp_cell->num_finest_level_points[iii] += 1;
                                if (iii > 1) {
                                    for (k = 0; k <= iii - 1; k++) {
                                        temp_cell->num_finest_level_points[k] += cur_p->num_finest_level_points[k];
                                    }
                                }
                                else {
                                    temp_cell->num_finest_level_points[0] += 1;
                                }
                                if (temp_cell->parent_info == NULL) break;
                            }
                        }
                    }
                    if (temp_index != NULL) {
                        for (k = 0; k < num_indices; k++) {
                            btree_p_clear(&(temp_index[k]), &(dci_inst->num_leaf_nodes), dci_inst->stack);
                        }
                        free(temp_index);
                    }
                }
            }
            else if (i > last_level) {  // still balanced, no need to promote, find new parents for children of the deleted point (if needed)
                assert(dci_inst->root->cell_indices->num_data != 0);
                for (s = btree_p_first(cur_point->cell_indices); !btree_p_is_end(cur_point->cell_indices, s);
                    s = btree_p_find_next(s)) {
                    additional_info* cur_p = btree_p_valueof(s).info;
                    idx_arr top_candidate;
                    dci_query_single_point_(
                        dci_inst->num_comp_indices, dci_inst->num_simp_indices, dci_inst->dim,
                        dci_inst->num_levels, dci_inst->root,
                        dci_inst->num_levels - i, 1,
                        cur_p->data_loc,
                        cur_p->local_dist[0],
                        deletion_config,
                        &top_candidate,
                        false,
                        dci_inst, cur_p->id, 
                        dci_inst->num_points_on_level[i], dci_inst->points_on_level[i], NULL, NULL, NULL);
                    assert(cur_p->parent_info->cell_indices != NULL);
                    cur_p->parent_info = top_candidate.info;
                    cur_p->parent_dist = top_candidate.key;

                    dci_insert_to_indices(dci_inst, dci_inst->transform, num_indices, 
                                    cur_p->parent_info, cur_p,
                                    dci_inst->max_sq_norm, dci_inst->add_proj_vec, dci_inst->sq_norm_list);

                    for (additional_info* temp_cell = cur_p->parent_info;
                        temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
                        temp_cell->num_finest_level_points[i] += 1;
                        if (i > 1) {
                            for (k = 0; k <= i - 1; k++) {
                                temp_cell->num_finest_level_points[k] += cur_p->num_finest_level_points[k];
                            }
                        }
                        else {
                            temp_cell->num_finest_level_points[0] += 1;
                        }
                        if (temp_cell->parent_info == NULL) break;
                    }
                }
            }
        }
        // if there is a level have no point, then we can infer that its expected num is 0, 
        // which means that there will be no promotion anymore, so there is no points that should be kept below
        // so we could delete them directly 
        if (i == last_level) {  // update statistics and stop the for loop
            assert(num_points_on_level[i] == expected_level_num[i]);
            if (i > 0) {
                for (ii = 0; ii < num_points_on_level[last_level]; ii++) {
                    dci_inst->points_on_level[last_level][ii]->num_finest_level_points = NULL;
                    dci_inst->points_on_level[last_level][ii]->cell_indices = NULL;
                    dci_inst->points_on_level[last_level][ii]->arr_indices = NULL;
                    #ifdef USE_OPENMP
                    omp_init_lock(&(dci_inst->points_on_level[last_level][ii]->lock));
                    #endif
                    dci_inst->points_on_level[last_level][ii]->flag = 1;
                }
                int* new_data_num_points_on_level = (int*)malloc(sizeof(int) * (dci_inst->num_levels - last_level));
                int* new_max_num_on_level = (int*)malloc(sizeof(int) * (dci_inst->num_levels - last_level));
                additional_info*** new_points_on_level = (additional_info***)malloc(sizeof(additional_info**) * (dci_inst->num_levels - last_level));
                for (ii = 0; ii < (dci_inst->num_levels - last_level); ii++) {
                    new_data_num_points_on_level[ii] = dci_inst->num_points_on_level[ii + last_level];
                    new_max_num_on_level[ii] = dci_inst->max_num_on_level[ii + last_level];
                    new_points_on_level[ii] = dci_inst->points_on_level[ii + last_level];
                }
                free(dci_inst->num_points_on_level);
                free(dci_inst->max_num_on_level);
                free(dci_inst->points_on_level);
                dci_inst->num_points_on_level = new_data_num_points_on_level;
                dci_inst->max_num_on_level = new_max_num_on_level;
                dci_inst->points_on_level = new_points_on_level;
                dci_inst->num_levels -= last_level;
            }
            break;
        }
    }

    for (i = last_level - 1; i >= 0; i--) {  // For the rest levels, delete the points directly
        num_deleted += delete_level_num[i];
    }
    assert(num_points - duplicate_num == num_deleted);
    dci_inst->num_points -= num_deleted;
    
    num_points_on_level = dci_inst->num_points_on_level;
    additional_info*** level_points = dci_inst->points_on_level;
    
    if (dci_inst->num_levels >= 2) {
        //  Update num_finest_level_points
        num_points_on_cur_levels = num_points_on_level[1];
        for (j = 0; j < num_points_on_cur_levels; j++) {
            if (level_points[1][j]->num_finest_level_points != NULL) {
                free(level_points[1][j]->num_finest_level_points);
            }
            if (level_points[1][j]->num_finest_level_nodes != NULL) {
                free(level_points[1][j]->num_finest_level_nodes);
            }
            level_points[1][j]->num_finest_level_points = (int*)malloc(sizeof(int) * 2);
            level_points[1][j]->num_finest_level_nodes = (int*)malloc(sizeof(int) * 2);
            int temp_num_data = level_points[1][j]->cell_indices[0].num_data;
            level_points[1][j]->num_finest_level_points[0] = temp_num_data + 1;
            level_points[1][j]->num_finest_level_points[1] = temp_num_data;
            int temp_num_node = level_points[1][j]->cell_indices[0].num_leaf_nodes;
            level_points[1][j]->num_finest_level_nodes[0] = temp_num_node + 1;
            level_points[1][j]->num_finest_level_nodes[1] = temp_num_node;
        }
        for (i = 2; i < dci_inst->num_levels; i++) {
            num_points_on_cur_levels = num_points_on_level[i];
            for (j = 0; j < num_points_on_cur_levels; j++) {
                if (level_points[i][j]->num_finest_level_points != NULL) {
                    free(level_points[i][j]->num_finest_level_points);
                }
                if (level_points[i][j]->num_finest_level_nodes != NULL) {
                    free(level_points[i][j]->num_finest_level_nodes);
                }
                level_points[i][j]->num_finest_level_points = (int*)malloc(sizeof(int) * (i + 1));
                level_points[i][j]->num_finest_level_nodes = (int*)malloc(sizeof(int) * (i + 1));
                for (int l = i; l >= 0; l--) {
                    level_points[i][j]->num_finest_level_points[l] = 0;
                    level_points[i][j]->num_finest_level_nodes[l] = 0;
                }
                level_points[i][j]->num_finest_level_points[i] = level_points[i][j]->cell_indices[0].num_data;
                level_points[i][j]->num_finest_level_points[0] = 1;
                level_points[i][j]->num_finest_level_nodes[i] = level_points[i][j]->cell_indices[0].num_leaf_nodes;
                level_points[i][j]->num_finest_level_nodes[0] = 1;
                cur_tree = level_points[i][j]->cell_indices;
                assert(cur_tree != NULL);
                btree_p_search_res s;

                for (s = btree_p_first(cur_tree); !btree_p_is_end(cur_tree, s);
                    s = btree_p_find_next(s)) {
                    for (int l = i - 1; l >= 0; l--) {
                        level_points[i][j]->num_finest_level_points[l] += s.n->slot_data[s.slot].info->num_finest_level_points[l];
                        level_points[i][j]->num_finest_level_nodes[l] += s.n->slot_data[s.slot].info->num_finest_level_nodes[l];
                    }
                }
                level_points[i][j]->num_finest_level_nodes[0] -= level_points[i][j]->num_finest_level_points[i];
                level_points[i][j]->num_finest_level_nodes[0] += level_points[i][j]->num_finest_level_nodes[i];
            }
        }
    }


    // TODO: UPDATE THE NODE INFORAMATION FOR DELETION FUNCTIONS


    hashtable_i_free(&to_delete);
    free(delete_level_num);
    for (i = 0; i < dci_inst->num_levels; i++) {
        free(deleted_points[i]);
    }
    free(deleted_points);
    free(expected_level_num);
    free(rest_level_num);

    return num_deleted;
}

long long dci_add_one_point(dci* const dci_inst, const int dim, const int num_points,
    const bf16_t* const data, const bf16_t* const value, dci_query_config construction_query_config,
    const long long* const data_ids, int target_level, float* new_data_proj, bool* mask, int interval, int X, float anchor_threshold) {
    int i, j, h, k;
    int new_data_level, num_points_on_cur_levels, num_points_on_upper_levels, new_data_num_points_on_cur_levels;
    additional_info* new_data_root, * cur_cell, * prev_parent;
    int num_comp_indices = dci_inst->num_comp_indices;
    int num_simp_indices = dci_inst->num_simp_indices;
    int num_indices = num_comp_indices * num_simp_indices;
    bool free_data_proj = 0;
    int num_levels = dci_inst->num_levels;
    bf16_t* temp_data_loc = NULL;
    bf16_t* temp_inc_data_loc = NULL;

    assert(num_points == 1);

    if (!dci_inst->root) {
        new_data_root = (additional_info*)malloc(sizeof(additional_info));
        new_data_root->cell_indices = (btree_p*)malloc(sizeof(btree_p) * num_indices);
        new_data_root->arr_indices = NULL;
        new_data_root->num_finest_level_points = NULL;
        new_data_root->num_finest_level_nodes = NULL;
        new_data_root->local_dist = NULL;
        new_data_root->max_sq_norm = dci_inst->max_sq_norm;
        #ifdef USE_OPENMP
        omp_init_lock(&(new_data_root->lock));
        #endif
        new_data_root->id = -1;
        new_data_root->flag = 1;
        initialize_indices(new_data_root->cell_indices, num_indices);

        dci_inst->root = new_data_root;
    }

    long long data_id;
    if (data_ids == NULL)
        data_id = dci_inst->next_point_id;
    else
        data_id = data_ids[0];

    if (data_id >= dci_inst->max_volume) {
        while (data_id >= dci_inst->max_volume) {
            dci_inst->max_volume *= 2;
        }
        realloc_(dci_inst);
        assert(dci_inst->max_volume > dci_inst->num_points);
    }

    // Step 1. Data projection, insert to the tree and update the max_norm
    if (new_data_proj == NULL) {
        free_data_proj = 1;
        data_projection(num_indices, dci_inst, dim, 1, data, &new_data_proj, mask, 0);
    }
    
    // Step 2. Insert the point to the tree
    if (dci_inst->num_points == 0) {
        // Directly add to the root
        additional_info* cur_cell = (additional_info*)malloc(sizeof(additional_info));
        cur_cell->id = data_id;
        cur_cell->arr_indices = NULL;
        #ifdef USE_OPENMP
        omp_init_lock(&(cur_cell->lock));
        #endif
        cur_cell->max_sq_norm = dci_inst->max_sq_norm;

        cur_cell->cell_indices = NULL;
        cur_cell->num_finest_level_points = NULL;
        cur_cell->num_finest_level_nodes = NULL;
        cur_cell->flag = 0;
        cur_cell->max_child_dist = 0.;
        cur_cell->parent_dist = 0.;
        cur_cell->data_loc = data;
        cur_cell->inc_data_loc = value;
        cur_cell->parent_info = dci_inst->root;
        cur_cell->local_dist = (float*)malloc(sizeof(float) * num_indices);
        for (k = 0; k < num_indices; k++) {
            cur_cell->local_dist[k] = new_data_proj[k];
        }
        dci_insert_to_indices(dci_inst, dci_inst->transform, num_indices, 
                            dci_inst->root, cur_cell,
                            dci_inst->max_sq_norm, dci_inst->add_proj_vec, dci_inst->sq_norm_list);
        update_arr_indices(num_indices, dci_inst->root);

        dci_inst->num_points += 1;
        
        dci_inst->num_levels = 1;
        dci_inst->num_points_on_level = (int*)malloc(sizeof(int) * 1);
        dci_inst->points_on_level = (additional_info***)malloc(sizeof(additional_info**)*1);
        dci_inst->max_num_on_level = (int*)malloc(sizeof(int) * 1);

        dci_inst->num_points_on_level[0] = 1;
        dci_inst->points_on_level[0] = (additional_info**)malloc(sizeof(additional_info*) * 2);
        dci_inst->max_num_on_level[0] = 2;
        dci_inst->points_on_level[0][0] = cur_cell;

        hashtable_p_set(dci_inst->inserted_points, cur_cell->id, cur_cell, 0, 0);
    }
    else if (target_level >= num_levels) {
        // Need to add one additional level to the top of the tree
        new_data_root = (additional_info*)malloc(sizeof(additional_info));
        new_data_root->cell_indices = (btree_p*)malloc(sizeof(btree_p) * num_indices);
        new_data_root->arr_indices = NULL;
        new_data_root->num_finest_level_points = NULL;
        new_data_root->num_finest_level_nodes = NULL;
        new_data_root->local_dist = NULL;
        new_data_root->max_sq_norm = dci_inst->max_sq_norm;
        #ifdef USE_OPENMP
        omp_init_lock(&(new_data_root->lock));
        #endif
        new_data_root->id = -1;
        new_data_root->flag = 1;
        initialize_indices(new_data_root->cell_indices, num_indices);

        // Directly add to the new root
        additional_info* cur_cell = (additional_info*)malloc(sizeof(additional_info));
        cur_cell->id = data_id;
        cur_cell->arr_indices = NULL;
        #ifdef USE_OPENMP
        omp_init_lock(&(cur_cell->lock));
        #endif
        cur_cell->max_sq_norm = dci_inst->max_sq_norm;

        cur_cell->cell_indices = (btree_p*)malloc(sizeof(btree_p) * num_indices);
        initialize_indices(cur_cell->cell_indices, num_indices);
        cur_cell->num_finest_level_points = (int*)malloc(sizeof(int)* (num_levels + 1));
        cur_cell->num_finest_level_nodes = (int*)malloc(sizeof(int)* (num_levels + 1));
        for (int l = num_levels; l >= 0; l--) {
            cur_cell->num_finest_level_points[l] = 0;
            cur_cell->num_finest_level_nodes[l] = 0;
        }
        cur_cell->num_finest_level_points[0] = 1;
        cur_cell->num_finest_level_nodes[0] = 1;
        cur_cell->flag = 0;
        cur_cell->max_child_dist = 0.;
        cur_cell->parent_dist = 0.;
        cur_cell->data_loc = data;
        cur_cell->inc_data_loc = value;
        cur_cell->parent_info = new_data_root;
        cur_cell->local_dist = (float*)malloc(sizeof(float) * num_indices);
        for (k = 0; k < num_indices; k++) {
            cur_cell->local_dist[k] = new_data_proj[k];
        }
        dci_insert_to_indices(dci_inst, dci_inst->transform, num_indices, 
                            new_data_root, cur_cell,
                            dci_inst->max_sq_norm, dci_inst->add_proj_vec, dci_inst->sq_norm_list);
        update_arr_indices(num_indices, new_data_root);

        additional_info** points_on_lower_level = dci_inst->points_on_level[num_levels - 1];
        int num_points_on_lower_level = dci_inst->num_points_on_level[num_levels - 1];
        for (int jj = 0; jj < num_points_on_lower_level; jj++) {
            additional_info* lower_cell = points_on_lower_level[jj];
            // caculate the distance
            float cur_dist;
            if (dci_inst->transform) {
                cur_dist = transform_compute_dist(lower_cell->data_loc, data, dim, dci_inst->max_sq_norm, dci_inst->sq_norm_list[lower_cell->id], dci_inst->sq_norm_list[data_id]);
            }
            else {
                cur_dist = compute_dist(lower_cell->data_loc, data, dim);
            }
            lower_cell->parent_info = cur_cell;
            lower_cell->parent_dist = cur_dist;

            dci_insert_to_indices(dci_inst, dci_inst->transform, num_indices,
                        cur_cell, lower_cell,
                        dci_inst->max_sq_norm, dci_inst->add_proj_vec, dci_inst->sq_norm_list);

            cur_cell->num_finest_level_points[num_levels] += 1;
            if (num_levels > 1) {
                for (int l = 0; l <= num_levels - 1; l++) {
                    cur_cell->num_finest_level_points[l] += lower_cell->num_finest_level_points[l];
                    cur_cell->num_finest_level_nodes[l] += lower_cell->num_finest_level_nodes[l];
                }
            }
            else {
                cur_cell->num_finest_level_points[0] += 1;
            }
        }
        cur_cell->num_finest_level_nodes[num_levels] = cur_cell->cell_indices[0].num_leaf_nodes;
        if (num_levels > 1)
            cur_cell->num_finest_level_nodes[0] -= num_points_on_lower_level;
        cur_cell->num_finest_level_nodes[0] += cur_cell->num_finest_level_nodes[num_levels];

        free_cell(dci_inst->root, num_indices, dci_inst);
        dci_inst->root = new_data_root;

        dci_inst->num_levels += 1;
        dci_inst->num_points += 1;

        num_levels = dci_inst->num_levels;

        dci_inst->num_points_on_level = (int*)realloc(dci_inst->num_points_on_level, sizeof(int) * num_levels);
        dci_inst->points_on_level = (additional_info***)realloc(dci_inst->points_on_level, sizeof(additional_info**) * num_levels);
        dci_inst->max_num_on_level = (int*)realloc(dci_inst->max_num_on_level, sizeof(int) * num_levels);

        dci_inst->num_points_on_level[num_levels - 1] = 1;
        // Initialize the array by size of 2 (will be expanded if necessary)
        dci_inst->points_on_level[num_levels - 1] = (additional_info**)malloc(sizeof(additional_info*) * 2);
        dci_inst->max_num_on_level[num_levels - 1] = 2;
        dci_inst->points_on_level[num_levels - 1][0] = cur_cell;

        hashtable_p_set(dci_inst->inserted_points, cur_cell->id, cur_cell, num_levels - 1, 0);
    }
    else if (target_level == num_levels - 1) {
        // Directly add to the root
        additional_info* cur_cell = (additional_info*)malloc(sizeof(additional_info));
        cur_cell->id = data_id;
        cur_cell->arr_indices = NULL;
        #ifdef USE_OPENMP
        omp_init_lock(&(cur_cell->lock));
        #endif
        cur_cell->max_sq_norm = dci_inst->max_sq_norm;

        cur_cell->cell_indices = NULL;
        cur_cell->num_finest_level_points = NULL;
        cur_cell->num_finest_level_nodes = NULL;
        if (target_level) {
            cur_cell->cell_indices = (btree_p*)malloc(sizeof(btree_p) * num_indices);
            initialize_indices(cur_cell->cell_indices, num_indices);
            cur_cell->num_finest_level_points = (int*)malloc(sizeof(int)* num_levels);
            cur_cell->num_finest_level_nodes = (int*)malloc(sizeof(int)* num_levels);
            for (int l = target_level; l >= 0; l--) {
                cur_cell->num_finest_level_points[l] = 0;
                cur_cell->num_finest_level_nodes[l] = 0;
            }
            cur_cell->num_finest_level_points[0] = 1;
            cur_cell->num_finest_level_nodes[0] = 1;
        }
        cur_cell->flag = 0;
        cur_cell->max_child_dist = 0.;
        cur_cell->parent_dist = 0.;
        cur_cell->data_loc = data;
        cur_cell->inc_data_loc = value;
        cur_cell->parent_info = dci_inst->root;
        cur_cell->local_dist = (float*)malloc(sizeof(float) * num_indices);
        for (k = 0; k < num_indices; k++) {
            cur_cell->local_dist[k] = new_data_proj[k];
        }
        dci_insert_to_indices(dci_inst, dci_inst->transform, num_indices, 
                            dci_inst->root, cur_cell,
                            dci_inst->max_sq_norm, dci_inst->add_proj_vec, dci_inst->sq_norm_list);
        update_arr_indices(num_indices, dci_inst->root);

        additional_info** points_on_lower_level = dci_inst->points_on_level[target_level - 1];
        int num_points_on_lower_level = dci_inst->num_points_on_level[target_level - 1];
        // for the target_level-1 level, decide if they need to change parent to new inserted point
        for (int jj = 0; jj < num_points_on_lower_level; jj++) {
            additional_info* lower_cell = points_on_lower_level[jj];
            // caculate the distance
            float cur_dist;
            if (dci_inst->transform) {
                cur_dist = transform_compute_dist(lower_cell->data_loc, data, dim, dci_inst->max_sq_norm, dci_inst->sq_norm_list[lower_cell->id], dci_inst->sq_norm_list[data_id]);
            }
            else {
                cur_dist = compute_dist(lower_cell->data_loc, data, dim);
            }
            // need to change the parent if the new inserted point is better
            if (lower_cell->parent_dist - cur_dist > 1e-8) {
                additional_info* prev_parent = lower_cell->parent_info;
                lower_cell->parent_info = cur_cell;
                lower_cell->parent_dist = cur_dist;

                if (dci_inst->update_addr) {
                    temp_data_loc = (bf16_t*)malloc(sizeof(bf16_t) * dim);
                    temp_inc_data_loc = (bf16_t*)malloc(sizeof(bf16_t) * dim);
                    for (k = 0; k < dim; k++) {  // copy the data_loc since the old data_loc will be overwritten in btree_p_delete
                        temp_data_loc[k] = lower_cell->data_loc[k];
                        temp_inc_data_loc[k] = lower_cell->inc_data_loc[k];
                    }
                    lower_cell->data_loc = temp_data_loc;
                    lower_cell->inc_data_loc = temp_inc_data_loc;
                }
                // Keep track of the previous number of nodes of both old and cur parents
                int prev_old_num_nodes = prev_parent->cell_indices[0].num_leaf_nodes;
                int cur_old_num_nodes = cur_cell->cell_indices[0].num_leaf_nodes;

                for (k = 0; k < num_indices; k++) {
                    bool ret = btree_p_delete(&(prev_parent->cell_indices[k]), lower_cell->local_dist[k], lower_cell->id, dci_inst->token2nodeIndex, dci_inst->token2nodeOffset, 
                                                &(dci_inst->num_leaf_nodes), dci_inst->stack, dci_inst->page_status, dim, dci_inst->update_addr);
                    assert(ret);
                }
                prev_parent->flag = 1;
                dci_insert_to_indices(dci_inst, dci_inst->transform, num_indices,
                        cur_cell, lower_cell,
                        dci_inst->max_sq_norm, dci_inst->add_proj_vec, dci_inst->sq_norm_list);

                if (dci_inst->update_addr && temp_data_loc != NULL) {
                    free(temp_data_loc);
                    free(temp_inc_data_loc);
                }
                // Keep track of the new number of nodes of both old and cur parents
                int prev_new_num_nodes = prev_parent->cell_indices[0].num_leaf_nodes;
                int cur_new_num_nodes = cur_cell->cell_indices[0].num_leaf_nodes;

                for (additional_info* temp_cell = prev_parent;
                    temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
                    temp_cell->num_finest_level_points[target_level] -= 1;
                    if (prev_new_num_nodes != prev_old_num_nodes) {
                        assert(prev_old_num_nodes - prev_new_num_nodes == 1);
                        temp_cell->num_finest_level_nodes[target_level] -= 1;
                        assert(temp_cell->num_finest_level_nodes[target_level] >= 0);
                    }
                    assert(temp_cell->num_finest_level_points[target_level] >= 0);
                    if (target_level > 1) {
                        for (int l = 0; l <= target_level - 1; l++) {
                            temp_cell->num_finest_level_points[l] -= lower_cell->num_finest_level_points[l];
                            assert(temp_cell->num_finest_level_points[l] >= 0);
                            temp_cell->num_finest_level_nodes[l] -= lower_cell->num_finest_level_nodes[l];
                        }
                        if (prev_new_num_nodes == prev_old_num_nodes)
                            temp_cell->num_finest_level_nodes[0] += 1;
                    }
                    else {
                        temp_cell->num_finest_level_points[0] -= 1;
                        assert(temp_cell->num_finest_level_points[0] >= 1);
                        if (prev_new_num_nodes != prev_old_num_nodes)
                            temp_cell->num_finest_level_nodes[0] -= 1;
                    }
                    if (temp_cell->parent_info == NULL) break;
                }
                int diff = cur_new_num_nodes - cur_old_num_nodes;
                for (additional_info* temp_cell = cur_cell;
                    temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
                    temp_cell->num_finest_level_points[target_level] += 1;
                    if (cur_new_num_nodes != cur_old_num_nodes) {
                        temp_cell->num_finest_level_nodes[target_level] += diff;
                    }
                    if (target_level > 1) {
                        for (int l = 0; l <= target_level - 1; l++) {
                            temp_cell->num_finest_level_points[l] += lower_cell->num_finest_level_points[l];
                            temp_cell->num_finest_level_nodes[l] += lower_cell->num_finest_level_nodes[l];
                        }
                        temp_cell->num_finest_level_nodes[0] -= 1;
                        if (cur_new_num_nodes != cur_old_num_nodes)
                            temp_cell->num_finest_level_nodes[0] += diff;
                    }
                    else {
                        temp_cell->num_finest_level_points[0] += 1;
                        if (cur_new_num_nodes != cur_old_num_nodes)
                            temp_cell->num_finest_level_nodes[0] += diff;
                    }
                    if (temp_cell->parent_info == NULL) break;
                }
            }
        }

        dci_inst->num_points += 1;
        hashtable_p_set(dci_inst->inserted_points, cur_cell->id, cur_cell, target_level, dci_inst->num_points_on_level[target_level]);
        if (dci_inst->num_points_on_level[target_level] >= dci_inst->max_num_on_level[target_level]) {
            dci_inst->max_num_on_level[target_level] *= 2;
            dci_inst->points_on_level[target_level] = (additional_info**)realloc(dci_inst->points_on_level[target_level], sizeof(additional_info*) * dci_inst->max_num_on_level[target_level]);
        }
        dci_inst->points_on_level[target_level][dci_inst->num_points_on_level[target_level]] = cur_cell;
        dci_inst->num_points_on_level[target_level] += 1;
    }
    else {
        additional_info* cur_cell = (additional_info*)malloc(sizeof(additional_info));
        cur_cell->id = data_id;

        int cur_num_returned;
        idx_arr top_candidate;

        num_points_on_cur_levels = dci_inst->num_points_on_level[target_level];

        if ((i == 0) && (interval > 0) && (num_points_on_cur_levels > dci_inst->numa_threshold)) {
            // Non-anchor point: check X anchors on the left (smaller indices)
            // Since points are inserted in order, there are no right anchors yet
            long long cur_point_id = data_id;
            const bf16_t* cur_point_data = data;

            // Collect unique anchor parents from X anchors on the left
            // Anchors are at positions: current - interval, current - 2*interval, ..., current - X*interval
            // First collect unique anchor parents to avoid duplicate distance computations
            additional_info* unique_parents[X];
            int unique_parent_count = 0;

            // Collect X left anchors at -interval, -2*interval, -3*interval, ...
            for (int j = 1; j <= X; j++) {
                int anchor_pos = num_points_on_cur_levels - j * interval;
                if (anchor_pos >= 0 && anchor_pos < num_points_on_cur_levels) {
                    // Get the actual anchor point from points_on_level
                    additional_info* anchor_point = dci_inst->points_on_level[target_level][anchor_pos];
                    if (anchor_point != NULL && anchor_point->parent_info != NULL) {
                        additional_info* parent = anchor_point->parent_info;
                        // Check if this parent is already in the list
                        bool already_added = false;
                        for (int k = 0; k < unique_parent_count; k++) {
                            if (unique_parents[k] == parent) {
                                already_added = true;
                                break;
                            }
                        }
                        if (!already_added) {
                            unique_parents[unique_parent_count++] = parent;
                        }
                    }
                }
            }

            // Now compute distances to unique parents only
            additional_info* closest_parent = NULL;
            float min_dist_to_parent = FLT_MAX;

            for (int k = 0; k < unique_parent_count; k++) {
                const bf16_t* parent_data = unique_parents[k]->data_loc;
                long long parent_id = unique_parents[k]->id;

                float dist;
                if (dci_inst->transform) {
                    dist = transform_compute_dist(cur_point_data, parent_data, dim,
                                                    dci_inst->max_sq_norm,
                                                    dci_inst->sq_norm_list[cur_point_id],
                                                    dci_inst->sq_norm_list[parent_id]);
                } else {
                    dist = compute_dist(cur_point_data, parent_data, dim);
                }

                if (dist < min_dist_to_parent) {
                    min_dist_to_parent = dist;
                    closest_parent = unique_parents[k];
                }
            }

            // Now check if max children distance of closest parent > min_dist_to_parent
            bool use_anchor_parent = false;
            if (closest_parent != NULL) {
                // Use pre-computed max_child_dist from additional_info
                float max_children_dist = closest_parent->max_child_dist;

                // If max children distance >= current point's distance to parent, use anchor parent
                if (max_children_dist * anchor_threshold >= min_dist_to_parent) {
                    use_anchor_parent = true;
                }
            }

            if (use_anchor_parent && closest_parent != NULL) {
                top_candidate.info = closest_parent;
                top_candidate.key = min_dist_to_parent;
            } else {
                cur_num_returned = dci_query_single_point_(
                    num_comp_indices, num_simp_indices, dim,
                    num_levels, dci_inst->root, num_levels - target_level - 1, 1,
                    data, new_data_proj[0], construction_query_config,
                    &top_candidate, false, dci_inst, data_id, 
                    dci_inst->num_points_on_level[target_level + 1], dci_inst->points_on_level[target_level + 1], NULL, NULL, NULL);
            }
        }
        else {
            cur_num_returned = dci_query_single_point_(
                num_comp_indices, num_simp_indices, dim,
                num_levels, dci_inst->root, num_levels - target_level - 1, 1,
                data, new_data_proj[0], construction_query_config,
                &top_candidate, false, dci_inst, data_id, 
                dci_inst->num_points_on_level[target_level + 1], dci_inst->points_on_level[target_level + 1], NULL, NULL, NULL);
        }

        // Update parent's max_child_dist
        if (top_candidate.info != NULL && top_candidate.key > top_candidate.info->max_child_dist) {
            top_candidate.info->max_child_dist = top_candidate.key;
        }
        
        cur_cell->arr_indices = NULL;
        #ifdef USE_OPENMP
        omp_init_lock(&(cur_cell->lock));
        #endif
        cur_cell->cell_indices = NULL;
        cur_cell->num_finest_level_points = NULL;
        cur_cell->num_finest_level_nodes = NULL;
        cur_cell->flag = 0;
        cur_cell->max_child_dist = 0.;
        cur_cell->parent_dist = top_candidate.key;  // distance to the id
        cur_cell->data_loc = data;
        cur_cell->inc_data_loc = value;
        cur_cell->parent_info = top_candidate.info;
        cur_cell->local_dist = (float*)malloc(sizeof(float) * num_indices);
        for (k = 0; k < num_indices; k++) {
            cur_cell->local_dist[k] = new_data_proj[k];
        }
        cur_cell->max_sq_norm = dci_inst->max_sq_norm;

        for (additional_info* temp_cell = cur_cell->parent_info;
            temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
            temp_cell->num_finest_level_points[target_level + 1] += 1;
            temp_cell->num_finest_level_points[0] += 1;
        }

        // Keep track of the previous number of nodes
        int old_num_nodes = cur_cell->parent_info->cell_indices[0].num_leaf_nodes;

        dci_insert_to_indices(dci_inst, dci_inst->transform, num_indices, 
                            cur_cell->parent_info, cur_cell,
                            dci_inst->max_sq_norm, dci_inst->add_proj_vec, dci_inst->sq_norm_list);

        int new_num_nodes = cur_cell->parent_info->cell_indices[0].num_leaf_nodes;

        if (new_num_nodes != old_num_nodes) {
            int diff = new_num_nodes - old_num_nodes;
            for (additional_info* temp_cell = cur_cell->parent_info;
                temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
                temp_cell->num_finest_level_nodes[target_level + 1] += diff;
                temp_cell->num_finest_level_nodes[0] += diff;
                assert(temp_cell->num_finest_level_nodes[target_level + 1] >= 0);
                assert(temp_cell->num_finest_level_nodes[0] >= 0);
            }
        }

        if (target_level) {
            // we don't need to allocate for the finest level
            cur_cell->cell_indices = (btree_p*)malloc(sizeof(btree_p) * num_indices);
            initialize_indices(cur_cell->cell_indices, num_indices);
            cur_cell->num_finest_level_points = (int*)malloc(sizeof(int) * (target_level + 1));
            cur_cell->num_finest_level_nodes = (int*)malloc(sizeof(int) * (target_level + 1));
            for (int l = target_level; l >= 0; l--) {
                cur_cell->num_finest_level_points[l] = 0;
                cur_cell->num_finest_level_nodes[l] = 0;
            }
            cur_cell->num_finest_level_points[0] = 1;
            cur_cell->num_finest_level_nodes[0] = 1;

            additional_info** points_on_lower_level = dci_inst->points_on_level[target_level - 1];
            int num_points_on_lower_level = dci_inst->num_points_on_level[target_level - 1];
            // for the target_level-1 level, decide if they need to change parent to new instered point
            for (int jj = 0; jj < num_points_on_lower_level; jj++) {
                additional_info* lower_cell = points_on_lower_level[jj];
                // caculate the distance
                float cur_dist;
                if (dci_inst->transform) {
                    cur_dist = transform_compute_dist(lower_cell->data_loc, data, dim, dci_inst->max_sq_norm, dci_inst->sq_norm_list[lower_cell->id], dci_inst->sq_norm_list[data_id]);
                }
                else {
                    cur_dist = compute_dist(lower_cell->data_loc, data, dim);
                }
                // need to change the parent if promote point is better
                if (lower_cell->parent_dist - cur_dist > 1e-8) {
                    additional_info* prev_parent = lower_cell->parent_info;
                    lower_cell->parent_info = cur_cell;
                    lower_cell->parent_dist = cur_dist;

                    if (dci_inst->update_addr) {
                        temp_data_loc = (bf16_t*)malloc(sizeof(bf16_t) * dim);
                        temp_inc_data_loc = (bf16_t*)malloc(sizeof(bf16_t) * dim);
                        for (k = 0; k < dim; k++) {
                            temp_data_loc[k] = lower_cell->data_loc[k];
                            temp_inc_data_loc[k] = lower_cell->inc_data_loc[k];
                        }
                        lower_cell->data_loc = temp_data_loc;
                        lower_cell->inc_data_loc = temp_inc_data_loc;
                    }

                    // Keep track of the previous number of nodes of both old and cur parents
                    int prev_old_num_nodes = prev_parent->cell_indices[0].num_leaf_nodes;
                    int cur_old_num_nodes = cur_cell->cell_indices[0].num_leaf_nodes;

                    for (k = 0; k < num_indices; k++) {
                        bool ret = btree_p_delete(&(prev_parent->cell_indices[k]), lower_cell->local_dist[k], lower_cell->id, dci_inst->token2nodeIndex, dci_inst->token2nodeOffset, 
                                                    &(dci_inst->num_leaf_nodes), dci_inst->stack, dci_inst->page_status, dim, dci_inst->update_addr);
                        assert(ret);
                    }
                    prev_parent->flag = 1;
                    dci_insert_to_indices(dci_inst, dci_inst->transform, num_indices,
                            cur_cell, lower_cell,
                            dci_inst->max_sq_norm, dci_inst->add_proj_vec, dci_inst->sq_norm_list);

                    if (dci_inst->update_addr && temp_data_loc != NULL) {
                        free(temp_data_loc);
                        free(temp_inc_data_loc);
                    }

                    // Keep track of the new number of nodes of both old and cur parents
                    int prev_new_num_nodes = prev_parent->cell_indices[0].num_leaf_nodes;
                    int cur_new_num_nodes = cur_cell->cell_indices[0].num_leaf_nodes;

                    for (additional_info* temp_cell = prev_parent;
                        temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
                        temp_cell->num_finest_level_points[target_level] -= 1;
                        assert(temp_cell->num_finest_level_points[target_level] >= 0);
                        if (prev_new_num_nodes != prev_old_num_nodes) {
                            assert(prev_old_num_nodes - prev_new_num_nodes == 1);
                            temp_cell->num_finest_level_nodes[target_level] -= 1;
                            assert(temp_cell->num_finest_level_nodes[target_level] >= 0);
                        }
                        if (target_level > 1) {
                            for (int l = 0; l <= target_level - 1; l++) {
                                temp_cell->num_finest_level_points[l] -= lower_cell->num_finest_level_points[l];
                                assert(temp_cell->num_finest_level_points[l] >= 0);
                                temp_cell->num_finest_level_nodes[l] -= lower_cell->num_finest_level_nodes[l];
                            }
                            if (prev_new_num_nodes == prev_old_num_nodes)
                                temp_cell->num_finest_level_nodes[0] += 1;
                        }
                        else {
                            temp_cell->num_finest_level_points[0] -= 1;
                            assert(temp_cell->num_finest_level_points[0] >= 1);
                            if (prev_new_num_nodes != prev_old_num_nodes)
                                temp_cell->num_finest_level_nodes[0] -= 1;
                        }
                        if (temp_cell->parent_info == NULL) break;
                    }
                    int diff = cur_new_num_nodes - cur_old_num_nodes;
                    for (additional_info* temp_cell = cur_cell;
                        temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
                        temp_cell->num_finest_level_points[target_level] += 1;
                        if (cur_new_num_nodes != cur_old_num_nodes) {
                            temp_cell->num_finest_level_nodes[target_level] += diff;
                        }
                        if (target_level > 1) {
                            for (int l = 0; l <= target_level - 1; l++) {
                                temp_cell->num_finest_level_points[l] += lower_cell->num_finest_level_points[l];
                                temp_cell->num_finest_level_nodes[l] += lower_cell->num_finest_level_nodes[l];
                            }
                            temp_cell->num_finest_level_nodes[0] -= 1;
                            if (cur_new_num_nodes != cur_old_num_nodes)
                                temp_cell->num_finest_level_nodes[0] += diff;
                        }
                        else {
                            temp_cell->num_finest_level_points[0] += 1;
                            if (cur_new_num_nodes != cur_old_num_nodes)
                                temp_cell->num_finest_level_nodes[0] += diff;
                        }
                        if (temp_cell->parent_info == NULL) break;
                    }
                }
            }
        }

        dci_inst->num_points += 1;
        hashtable_p_set(dci_inst->inserted_points, cur_cell->id, cur_cell, target_level, dci_inst->num_points_on_level[target_level]);
        if (dci_inst->num_points_on_level[target_level] >= dci_inst->max_num_on_level[target_level]) {
            dci_inst->max_num_on_level[target_level] *= 2;
            dci_inst->points_on_level[target_level] = (additional_info**)realloc(dci_inst->points_on_level[target_level], sizeof(additional_info*) * dci_inst->max_num_on_level[target_level]);
        }
        dci_inst->points_on_level[target_level][dci_inst->num_points_on_level[target_level]] = cur_cell;
        dci_inst->num_points_on_level[target_level] += 1;
    }

    if (free_data_proj)
        free(new_data_proj);

    if (data_ids == NULL)
        dci_inst->next_point_id += 1;
    else {
        if (data_ids[0] >= dci_inst->next_point_id)
            dci_inst->next_point_id = data_ids[0] + 1;
    }

    return data_id;
}

long long dci_add(dci* const dci_inst, const int dim, const int num_points,
    const bf16_t* const data, const bf16_t* const value, const int num_levels, dci_query_config construction_query_config,
    const long long* const data_ids, int target_level, float* new_data_proj, bool* mask, bool random, int interval, int X, float anchor_threshold) {
    int i, j, h, k;
    int new_data_level, next_target_level;
    int num_points_on_cur_levels, num_points_on_upper_levels, new_data_num_points_on_cur_levels, max_new_data_num_points_on_cur_levels;
    additional_info* new_data_root, * cur_cell, * prev_parent;
    int num_comp_indices = dci_inst->num_comp_indices;
    int num_simp_indices = dci_inst->num_simp_indices;
    int num_indices = num_comp_indices * num_simp_indices;
    bool free_data_proj = 0;
    bf16_t* temp_data_loc = NULL;
    bf16_t* temp_inc_data_loc = NULL;

    // num_levels is not used in this version

    if (new_data_proj == NULL)
        free_data_proj = 1;

    if (num_points == 0)
        return 0;
    else if (num_points == 1) {
        long long first_id = dci_add_one_point(dci_inst, dim, num_points, data, value, construction_query_config, data_ids, target_level, new_data_proj, mask, interval, X, anchor_threshold);
        return first_id;
    }

    construction_query_config.num_to_visit = min_i(construction_query_config.num_to_visit, num_points);
    construction_query_config.num_to_retrieve = min_i(construction_query_config.num_to_retrieve, num_points);

    additional_info*** new_data_level_cells = (additional_info***)malloc(sizeof(additional_info**) * 1);  // initialize with one level
    int* new_data_num_points_on_level = (int*)malloc(sizeof(int) * 1);  // initialize with one level

    int empty_until;
    if (!dci_inst->root)
        empty_until = -1;
    else
        empty_until = dci_inst->num_levels;

    bool extend_flag = 0;
    while (dci_inst->num_points + num_points > dci_inst->max_volume) {
        dci_inst->max_volume *= 2;
        extend_flag = 1;
    }
    if (extend_flag)
        realloc_(dci_inst);
    
    construct_new_tree(dci_inst, &new_data_root, dim, num_points,
        &new_data_level, &next_target_level, data, value, &new_data_num_points_on_level,
        &new_data_proj, &new_data_level_cells, construction_query_config, mask,
        empty_until, random, interval, X, anchor_threshold);

    // // For Debugging
    // print_tree(new_data_level, new_data_num_points_on_level, new_data_root);

    if (!dci_inst->root) {  // first time adding data
        dci_inst->root = new_data_root;
        dci_inst->num_levels = new_data_level;
        dci_inst->next_point_id += num_points;  // data_ids is not used for the first time insertion
        dci_inst->num_points += num_points;
        dci_inst->num_points_on_level = (int*)malloc(sizeof(int) * new_data_level);
        dci_inst->max_num_on_level = (int*)malloc(sizeof(int) * new_data_level);
        dci_inst->points_on_level = new_data_level_cells;
        dci_inst->next_target_level = next_target_level;

        // Add new data to inserted_points(hashtable), key is id, value is level_cells[i][j]+level
        for (i = dci_inst->num_levels - 1; i >= 0; i--) {
            new_data_num_points_on_cur_levels = new_data_num_points_on_level[i];
            max_new_data_num_points_on_cur_levels = 2;
            while (new_data_num_points_on_cur_levels > max_new_data_num_points_on_cur_levels) {
                max_new_data_num_points_on_cur_levels *= 2;
            }
            dci_inst->num_points_on_level[i] = new_data_num_points_on_cur_levels;
            dci_inst->max_num_on_level[i] = max_new_data_num_points_on_cur_levels;
            for (j = 0; j < new_data_num_points_on_cur_levels; j++) {
                hashtable_p_set(dci_inst->inserted_points, new_data_level_cells[i][j]->id, new_data_level_cells[i][j], i, j);
            }
        }

        btree_p* cur_tree;
        btree_p_search_res s;
        if (dci_inst->num_levels >= 2) {
            num_points_on_cur_levels = new_data_num_points_on_level[1];
            for (j = 0; j < num_points_on_cur_levels; j++) {
                int temp_num_node = new_data_level_cells[1][j]->cell_indices[0].num_leaf_nodes;
                new_data_level_cells[1][j]->num_finest_level_nodes[0] = temp_num_node + 1;
                new_data_level_cells[1][j]->num_finest_level_nodes[1] = temp_num_node;
            }
            for (i = 2; i < dci_inst->num_levels; i++) {
                num_points_on_cur_levels = new_data_num_points_on_level[i];
                for (j = 0; j < num_points_on_cur_levels; j++) {
                    int* num_finest_level_nodes = new_data_level_cells[i][j]->num_finest_level_nodes;
                    int* num_finest_level_points = new_data_level_cells[i][j]->num_finest_level_points;
                    for (int l = i; l >= 0; l--) {
                        num_finest_level_nodes[l] = 0;
                    }
                    num_finest_level_nodes[i] = new_data_level_cells[i][j]->cell_indices[0].num_leaf_nodes;
                    num_finest_level_nodes[0] = 1;
                    cur_tree = new_data_level_cells[i][j]->cell_indices;
                    assert(cur_tree != NULL);
                    for (s = btree_p_first(cur_tree); !btree_p_is_end(cur_tree, s);
                        s = btree_p_find_next(s)) {
                        for (int l = i - 1; l >= 0; l--) {
                            num_finest_level_nodes[l] += s.n->slot_data[s.slot].info->num_finest_level_nodes[l];
                        }
                    }
                    num_finest_level_nodes[0] -= num_finest_level_points[i];
                    num_finest_level_nodes[0] += num_finest_level_nodes[i];
                }
            }
        }
    }
    else {  // here we merge the new tree with the previous one
        int extend_flag = 0;
        while (dci_inst->num_points + num_points > dci_inst->max_volume) {
            dci_inst->max_volume *= 2;
            extend_flag = 1;
        }
        if (extend_flag)
            realloc_(dci_inst);

        int taller_level, shorter_level;
        int *shorter_num_points_on_levels, *taller_num_points_on_levels;
        additional_info ***shorter_points_on_level, ***taller_points_on_level;
        additional_info *shorter_root, *taller_root;
        int empty_flag;
        int cur_sub_level = new_data_level - 1;  // store the index of the currently lowest empty level
        additional_info* cur_empty_root = NULL;

        // First check which one is taller
        if (dci_inst->num_levels >= new_data_level) {
            shorter_level = new_data_level;
            taller_level = dci_inst->num_levels;

            shorter_num_points_on_levels = new_data_num_points_on_level;
            taller_num_points_on_levels = dci_inst->num_points_on_level;

            shorter_points_on_level = new_data_level_cells;
            taller_points_on_level = dci_inst->points_on_level;

            shorter_root = new_data_root;
            taller_root = dci_inst->root;

            empty_flag = 0;  // no empty level in taller tree
        }
        else {
            shorter_level = dci_inst->num_levels;
            taller_level = new_data_level;

            dci_inst->num_points_on_level = (int*)realloc(dci_inst->num_points_on_level, sizeof(int) * new_data_level);
            dci_inst->points_on_level = (additional_info***)realloc(dci_inst->points_on_level, sizeof(additional_info**) * new_data_level);
            dci_inst->max_num_on_level = (int*)realloc(dci_inst->max_num_on_level, sizeof(int) * new_data_level);

            for (i = dci_inst->num_levels; i < new_data_level; i++) {
                dci_inst->num_points_on_level[i] = 0;
                dci_inst->points_on_level[i] = (additional_info**)malloc(sizeof(additional_info*) * 2);
                dci_inst->max_num_on_level[i] = 2;
            }

            shorter_num_points_on_levels = dci_inst->num_points_on_level;
            taller_num_points_on_levels = new_data_num_points_on_level;

            shorter_points_on_level = dci_inst->points_on_level;
            taller_points_on_level = new_data_level_cells;

            shorter_root = dci_inst->root;
            taller_root = new_data_root;

            empty_flag = 1;  // no empty level in shorter tree
        }

        additional_info*** newT_parents = (additional_info***)malloc(sizeof(additional_info**) * taller_level);  // temporarily store new parent for old-tree
        float** newT_parents_dist = (float**)malloc(sizeof(float*) * taller_level);  // temporarily store new parent dist for old-tree

        // find new parents for both old-tree and new-tree first (and don't insert!)
        for (i = taller_level - 1; i >= 0; i--) {
            int shorter_num_points_on_cur_levels = 0;
            if (i < shorter_level)
                shorter_num_points_on_cur_levels = shorter_num_points_on_levels[i];

            // calculate the number of points in this level in the old-tree
            int taller_num_points_on_cur_levels = taller_num_points_on_levels[i];

            newT_parents[i] = (additional_info**)malloc(sizeof(additional_info*) * (shorter_num_points_on_cur_levels + taller_num_points_on_cur_levels));
            newT_parents_dist[i] = (float*)malloc(sizeof(float) * (shorter_num_points_on_cur_levels + taller_num_points_on_cur_levels));
            
            // Change the new-tree nodes id and add the new-tree nodes to the points_on_level list
            if (i < new_data_level) {
                for (j = 0; j < new_data_num_points_on_level[i]; j++) {
                    if (data_ids == NULL) {
                        new_data_level_cells[i][j]->id += dci_inst->next_point_id;
                    }
                    else {
                        new_data_level_cells[i][j]->id = data_ids[new_data_level_cells[i][j]->id];
                    }
                }

                if (i < new_data_level - 2 && dci_inst->sub_root_list != NULL && dci_inst->sub_root_list[i+1] != NULL) {
                    cur_empty_root = dci_inst->sub_root_list[i+1];
                    cur_sub_level = i;
                }
            }
            
            // Find potential new parent for shorter tree from the taller tree
            if (i < shorter_level) {
                for (j = 0; j < shorter_num_points_on_cur_levels; j++) {
                    cur_cell = shorter_points_on_level[i][j];
                    if (i == taller_level - 1) {  // highest level
                        newT_parents[i][j] = taller_root;
                        newT_parents_dist[i][j] = 0.;
                    }
                    else if (taller_num_points_on_levels[i + 1] == 0) {
                        newT_parents[i][j] = cur_cell->parent_info;
                        newT_parents_dist[i][j] = cur_cell->parent_dist;
                    }
                    else {
                        idx_arr top_candidate;
                        if (empty_flag && cur_empty_root != NULL) {
                            dci_query_single_point_(
                                num_comp_indices, num_simp_indices, dim,
                                cur_sub_level + 1, cur_empty_root,
                                cur_sub_level - i, 1,
                                cur_cell->data_loc,
                                cur_cell->local_dist[0],
                                construction_query_config,
                                &top_candidate, false,
                                dci_inst, cur_cell->id, 
                                dci_inst->num_points_on_level[i+1], dci_inst->points_on_level[i+1], NULL, NULL, NULL);
                        }
                        else {
                            dci_query_single_point_(
                                num_comp_indices, num_simp_indices, dim,
                                taller_level, taller_root,
                                taller_level - i - 1, 1,
                                cur_cell->data_loc,
                                cur_cell->local_dist[0],
                                construction_query_config,
                                &top_candidate, false,
                                dci_inst, cur_cell->id,
                                dci_inst->num_points_on_level[i+1], dci_inst->points_on_level[i+1], NULL, NULL, NULL);
                        } // TODO: need to modified num_points_on_level

                        if (cur_cell->parent_info->id == -1 || i == shorter_level - 1) {
                            newT_parents[i][j] = top_candidate.info;
                            newT_parents_dist[i][j] = top_candidate.key;
                        }
                        else {  // if new parent from old-tree is closer, update; otherwise use the copy of old parent
                            if (cur_cell->parent_dist - top_candidate.key > 1e-8) {
                                newT_parents[i][j] = top_candidate.info;
                                newT_parents_dist[i][j] = top_candidate.key;
                            }
                            else {
                                newT_parents[i][j] = cur_cell->parent_info;
                                newT_parents_dist[i][j] = cur_cell->parent_dist;
                            }
                        }
                    }
                }
            }

            // Find potential new parent for taller tree from the shorter tree
            if (i < shorter_level - 1) {  // no need to change anything for the highest level
                for (j = shorter_num_points_on_cur_levels;
                    j < shorter_num_points_on_cur_levels + taller_num_points_on_cur_levels; j++) {
                    cur_cell = taller_points_on_level[i][j - shorter_num_points_on_cur_levels];
                    if (shorter_num_points_on_levels[i + 1] == 0) {
                        newT_parents[i][j] = NULL;
                        newT_parents_dist[i][j] = -1;
                    }
                    else {
                        idx_arr top_candidate;
                        if (!empty_flag && cur_empty_root != NULL) {
                            dci_query_single_point_(
                                num_comp_indices, num_simp_indices, dim,
                                cur_sub_level + 1, cur_empty_root,
                                cur_sub_level - i, 1,
                                cur_cell->data_loc,
                                cur_cell->local_dist[0],
                                construction_query_config,
                                &top_candidate, false,
                                dci_inst, cur_cell->id, 0, NULL, NULL, NULL, NULL);
                        } // TODO: need to modified num_points_on_level
                        else{
                            dci_query_single_point_(
                                num_comp_indices, num_simp_indices, dim,
                                shorter_level, shorter_root,
                                shorter_level - i - 1, 1,
                                cur_cell->data_loc,
                                cur_cell->local_dist[0],
                                construction_query_config,
                                &top_candidate, false,
                                dci_inst, cur_cell->id, 0, NULL, NULL, NULL, NULL);
                        } // TODO: need to modified num_points_on_level

                        // need to change the parent if the new one is better (don't change immediately, store it first)
                        if (cur_cell->parent_info->id == -1) {
                            newT_parents[i][j] = top_candidate.info;
                            newT_parents_dist[i][j] = top_candidate.key;
                        }
                        else {
                            if (cur_cell->parent_dist - top_candidate.key > 1e-8) {
                                newT_parents[i][j] = top_candidate.info;
                                newT_parents_dist[i][j] = top_candidate.key;
                            }
                            else {
                                newT_parents[i][j] = NULL;
                                newT_parents_dist[i][j] = -1;
                            }
                        }
                    }
                }
            }
        }

        // add new-tree nodes to old-tree finally
        for (i = taller_level - 1; i >= 0; i--) {
            int shorter_num_points_on_cur_levels = shorter_num_points_on_levels[i];
            int taller_num_points_on_cur_levels = taller_num_points_on_levels[i];

            if (i < shorter_level) {  // insert shorter tree nodes to the taller tree
                for (j = 0; j < shorter_num_points_on_cur_levels; j++) {
                    cur_cell = shorter_points_on_level[i][j];

                    if (cur_cell->cell_indices) {
                        for (k = 0; k < num_indices; k++) {
                            btree_p_clear(&(cur_cell->cell_indices[k]), &(dci_inst->num_leaf_nodes), dci_inst->stack);
                        }
                        free(cur_cell->cell_indices);
                    }
                    if (cur_cell->arr_indices)
                        free(cur_cell->arr_indices);
                    if (cur_cell->num_finest_level_points)
                        free(cur_cell->num_finest_level_points);

                    cur_cell->cell_indices = NULL;
                    if (i) {  // we don't need to allocate for the finest level
                        cur_cell->cell_indices = (btree_p*)malloc(sizeof(btree_p) * num_indices);
                        initialize_indices(cur_cell->cell_indices, num_indices);
                    }
                    cur_cell->arr_indices = NULL;
                    cur_cell->num_finest_level_points = NULL;
                    cur_cell->flag = 1;

                    additional_info* new_parent = newT_parents[i][j];
                    if (new_parent != NULL) {
                        dci_insert_to_indices(dci_inst, dci_inst->transform, num_indices, 
                                new_parent, cur_cell,
                                dci_inst->max_sq_norm, dci_inst->add_proj_vec, dci_inst->sq_norm_list);

                        cur_cell->parent_info = new_parent;
                        cur_cell->parent_dist = newT_parents_dist[i][j];
                    }
                }
            }

            if (i < shorter_level - 1) {  // change parent of points in taller tree (no need to change for the highest level)
                for (j = 0; j < taller_num_points_on_cur_levels; j++) {
                    if (newT_parents[i][shorter_num_points_on_cur_levels + j] != NULL) {
                        cur_cell = taller_points_on_level[i][j];
                        data_pt cur_point;
                        cur_point.info = cur_cell;

                        if (dci_inst->update_addr) {
                            temp_data_loc = (bf16_t*)malloc(sizeof(bf16_t) * dim);
                            temp_inc_data_loc = (bf16_t*)malloc(sizeof(bf16_t) * dim);
                            for (k = 0; k < dim; k++) {
                                temp_data_loc[k] = cur_cell->data_loc[k];
                                temp_inc_data_loc[k] = cur_cell->inc_data_loc[k];
                            }
                            cur_cell->data_loc = temp_data_loc;
                            cur_cell->inc_data_loc = temp_inc_data_loc;
                        }
                        prev_parent = cur_cell->parent_info;
                        if (prev_parent->id != -1) {
                            for (k = 0; k < num_indices; k++) {
                                bool ret = btree_p_delete(&(prev_parent->cell_indices[k]), cur_cell->local_dist[k], cur_cell->id, dci_inst->token2nodeIndex, dci_inst->token2nodeOffset, 
                                                            &(dci_inst->num_leaf_nodes), dci_inst->stack, dci_inst->page_status, dim, dci_inst->update_addr);
                                assert(ret);
                            }
                            prev_parent->flag = 1;
                        }

                        additional_info* new_parent = newT_parents[i][shorter_num_points_on_cur_levels + j];
                        if (new_parent != NULL) {
                            dci_insert_to_indices(dci_inst, dci_inst->transform, num_indices, 
                                    new_parent, cur_cell,
                                    dci_inst->max_sq_norm, dci_inst->add_proj_vec, dci_inst->sq_norm_list);

                            cur_cell->parent_info = new_parent;
                            cur_cell->parent_dist = newT_parents_dist[i][shorter_num_points_on_cur_levels + j];
                        }

                        if (dci_inst->update_addr && temp_data_loc != NULL) {
                            free(temp_data_loc);
                            free(temp_inc_data_loc);
                        }
                    }
                }
            }

            if (i < new_data_level) {
                for (j = 0; j < new_data_num_points_on_level[i]; j++) {
                    hashtable_p_set(dci_inst->inserted_points, new_data_level_cells[i][j]->id, new_data_level_cells[i][j], i, dci_inst->num_points_on_level[i]);
                    if (dci_inst->num_points_on_level[i] >= dci_inst->max_num_on_level[i]) {
                        dci_inst->max_num_on_level[i] *= 2;
                        dci_inst->points_on_level[i] = (additional_info**)realloc(dci_inst->points_on_level[i], sizeof(additional_info*) * dci_inst->max_num_on_level[i]);
                    }
                    dci_inst->points_on_level[i][dci_inst->num_points_on_level[i]++] = new_data_level_cells[i][j];
                }
            }
        }

        if (dci_inst->sub_root_list != NULL) {
            for (i = 0; i < new_data_level; i++) {
                if (dci_inst->sub_root_list[i] != NULL) {
                    free_cell(dci_inst->sub_root_list[i], num_indices, dci_inst);
                }
            }
            free(dci_inst->sub_root_list);
            dci_inst->sub_root_list = NULL;
        }

        free_cell(shorter_root, num_indices, dci_inst);
        dci_inst->root = taller_root;
        dci_inst->num_levels = taller_level;

        dci_inst->num_points += num_points;
        if (data_ids == NULL)
            dci_inst->next_point_id += num_points;
        else {
            if (data_ids[num_points - 1] >= dci_inst->next_point_id)
                dci_inst->next_point_id = data_ids[num_points - 1] + 1;
        }

        // Update num_finest_level_points (this part may not be necessary)
        additional_info*** level_cells = dci_inst->points_on_level;
        btree_p* cur_tree;
        if (dci_inst->num_levels >= 2) {
            num_points_on_cur_levels = dci_inst->num_points_on_level[1];
            for (j = 0; j < num_points_on_cur_levels; j++) {
                if (level_cells[1][j]->num_finest_level_points != NULL) {
                    free(level_cells[1][j]->num_finest_level_points);
                }
                level_cells[1][j]->num_finest_level_points = (int*)malloc(sizeof(int) * 2);
                int temp_num_data = level_cells[1][j]->cell_indices[0].num_data;
                level_cells[1][j]->num_finest_level_points[0] = temp_num_data + 1;
                level_cells[1][j]->num_finest_level_points[1] = temp_num_data;

                if (level_cells[1][j]->num_finest_level_nodes != NULL) {
                    free(level_cells[1][j]->num_finest_level_nodes);
                }
                level_cells[1][j]->num_finest_level_nodes = (int*)malloc(sizeof(int) * 2);
                int temp_num_node = level_cells[1][j]->cell_indices[0].num_leaf_nodes;
                level_cells[1][j]->num_finest_level_nodes[0] = temp_num_node + 1;
                level_cells[1][j]->num_finest_level_nodes[1] = temp_num_node;
            }
            for (i = 2; i < dci_inst->num_levels; i++) {
                num_points_on_cur_levels = dci_inst->num_points_on_level[i];
                for (j = 0; j < num_points_on_cur_levels; j++) {
                    if (level_cells[i][j]->num_finest_level_points != NULL) {
                        free(level_cells[i][j]->num_finest_level_points);
                    }
                    level_cells[i][j]->num_finest_level_points = (int*)malloc(sizeof(int) * (i + 1));
                    for (int l = i; l >= 0; l--) {
                        level_cells[i][j]->num_finest_level_points[l] = 0;
                    }
                    level_cells[i][j]->num_finest_level_points[i] = level_cells[i][j]->cell_indices[0].num_data;
                    level_cells[i][j]->num_finest_level_points[0] = 1;

                    if (level_cells[i][j]->num_finest_level_nodes != NULL) {
                        free(level_cells[i][j]->num_finest_level_nodes);
                    }
                    level_cells[i][j]->num_finest_level_nodes = (int*)malloc(sizeof(int) * (i + 1));
                    for (int l = i; l >= 0; l--) {
                        level_cells[i][j]->num_finest_level_nodes[l] = 0;
                    }
                    level_cells[i][j]->num_finest_level_nodes[i] = level_cells[i][j]->cell_indices[0].num_leaf_nodes;
                    level_cells[i][j]->num_finest_level_nodes[0] = 1;

                    cur_tree = level_cells[i][j]->cell_indices;
                    btree_p_search_res s;
                    for (s = btree_p_first(cur_tree); !btree_p_is_end(cur_tree, s);
                        s = btree_p_find_next(s)) {
                        for (int l = i - 1; l >= 0; l--) {
                            level_cells[i][j]->num_finest_level_points[l] += s.n->slot_data[s.slot].info->num_finest_level_points[l];
                            level_cells[i][j]->num_finest_level_nodes[l] += s.n->slot_data[s.slot].info->num_finest_level_nodes[l];
                        }
                    }
                    level_cells[i][j]->num_finest_level_nodes[0] -= level_cells[i][j]->cell_indices[0].num_data;
                    level_cells[i][j]->num_finest_level_nodes[0] += level_cells[i][j]->num_finest_level_nodes[i];
                }
            }
        }
        for (i = 0; i < taller_level; i++) {
            free(newT_parents[i]);
            free(newT_parents_dist[i]);
        }
        free(newT_parents);
        free(newT_parents_dist);

        for (i = 0; i < min_i(dci_inst->num_levels, new_data_level); i++) {
            free(new_data_level_cells[i]);
        }
        free(new_data_level_cells);
    }
    free(new_data_num_points_on_level);
    if (free_data_proj)
        free(new_data_proj);

    if (data_ids == NULL)
        return dci_inst->next_point_id - num_points;
    else
        return data_ids[0];
}

static int dci_query_single_point_single_level(
    int num_comp_indices, int num_simp_indices, int dim,
    additional_info* point, int num_neighbours,
    const bf16_t* const query, const float* const query_proj,
    const dci_query_config query_config, idx_arr* const top_candidates,
    bool cumu, dci* const dci_inst) {
    int i, j, k, m, h, top_h;
    int num_indices = num_comp_indices * num_simp_indices;
    float top_index_priority, cur_dist;
    int num_returned_finest_level_points;
    if (query_config.target_level == 0)
        num_returned_finest_level_points = 1;  // last level only counts the current node once, which is already counted by setting the initial value of num_returned_finest_level_points to 1
    else
        num_returned_finest_level_points = 0;
    int num_candidates = 0;
    int num_points = point->cell_indices[0].num_data;
    if (num_points == 0) return 0;
    int num_nodes = point->cell_indices[0].num_leaf_nodes;
    float init = (-FLT_MAX);

    float last_top_candidate_dist = init;  // The distance of the k^th closest candidate found so far
    int last_top_candidate = -1;
    int num_returned = 0;
    idx_arr* arr_indices;
    int num_finest = 0;

    int left_pos[num_indices];
    int right_pos[num_indices];
    additional_info* cur_points;
    float index_priority[num_indices];
    float candidate_dists[num_points];
    bool checked[num_points];
    int returned_num[num_indices];
    int returned_ids[num_indices*CLOSEST];
    for (i = 0; i < num_indices; i++) {
        returned_num[i] = 0;
    }
    for (i = 0; i < num_indices*CLOSEST; i++) {
        returned_ids[i] = -1;
    }

    int bitnslots = BITNSLOTS(num_points);  // # of int we need to represent the # of points we have
    int min_slot = (bitnslots + SLOT_NUM - 1) / SLOT_NUM * SLOT_NUM;    // (# of int stored in the smallest # of whole SIMD registers that can represent # of points we have)

    assert(num_neighbours > 0);

    int num_points_to_retrieve =
        max_i(query_config.num_to_retrieve,
            (int)ceil(query_config.prop_to_retrieve * num_points));
    int num_projs_to_visit = max_i(
        query_config.num_to_visit * num_simp_indices,
        (int)ceil(query_config.prop_to_visit * num_points * num_simp_indices));

    int count_num = num_indices*min_slot;
    _Alignas(32) unsigned int mask[min_slot];
    _Alignas(32) unsigned int merged_bitarray[min_slot];
    _Alignas(32) unsigned int count_bitarray[count_num];

    for (i = 0; i < count_num; i++) {
        count_bitarray[i] = 0;
    }
    for (i = 0; i < min_slot; i++) {
        merged_bitarray[i] = 0;
    }
    for (i = 0; i < min_slot; i++) {
        mask[i] = -1;
    }

    // need to lock otherwise there could be conflicts
    #ifdef USE_OPENMP
    omp_set_lock(&(point->lock));
    #endif
    if (point->flag == 1) {  // need to update arr_indices
       update_arr_indices(num_indices, point);
    }
    #ifdef USE_OPENMP
    omp_unset_lock(&(point->lock));
    #endif
    arr_indices = point->arr_indices;

    for (i = 0; i < num_points; i++) {
        candidate_dists[i] = init;
    }

    for (i = 0; i < num_points; i++) {
        checked[i] = 1;
    }

    if (num_neighbours >= num_points) {
        for (i = 0; i < num_points; i++) {
            cur_points = arr_indices[i].info;
            // Compute distance
            if (dci_inst->transform) {
                cur_dist = transform_compute_dist_query(cur_points->data_loc, query, dim);
            }
            else {
                cur_dist = compute_dist(cur_points->data_loc, query, dim);
            }
            top_candidates[i].key = cur_dist;
            top_candidates[i].info = cur_points;
        }

        qsort(top_candidates, num_points, sizeof(idx_arr), dci_compare_data_idx_arr_dist);

        return num_points;
    }

    for (i = 0; i < num_indices; i++) {
        left_pos[i] = dci_search_index(&(arr_indices[i*num_points]), query_proj[i], num_points);
        right_pos[i] = left_pos[i] + 1;
    }

    for (i = 0; i < num_indices; i++) {
        returned_num[i] = dci_next_closest_proj(&(arr_indices[i*num_points]), &(left_pos[i]), &(right_pos[i]), query_proj[i], num_points, &(returned_ids[i*CLOSEST]), &(index_priority[i]));
        if (returned_num[i] == 0) return 0;
    }

    int visit_proj = 0;

    k = 0;
    while (k < num_points * num_simp_indices * num_comp_indices) {
        visit_proj = 0;
        for (m = 0; m < num_comp_indices; m++) {
            top_index_priority = DBL_MAX;
            top_h = -1;
            for (h = 0; h < num_simp_indices; h++) {
                if (index_priority[h + m * num_simp_indices] < top_index_priority) {
                    top_index_priority = index_priority[h + m * num_simp_indices];
                    top_h = h;
                }
            }
            if (top_h >= 0) {
                i = top_h + m * num_simp_indices;
                visit_proj += returned_num[i];
                for (j = 0; j < returned_num[i]; j++) {
                    BITSET(&(count_bitarray[i*min_slot]), returned_ids[i*CLOSEST+j]);
                }
                for (j = 0; j < min_slot; j++) {
                    merged_bitarray[j] = count_bitarray[m*num_simp_indices*min_slot+j];
                }
                for (j = m * num_simp_indices+1; j < (m+1) * num_simp_indices; j++) {
                    BitAnd(&(count_bitarray[j*min_slot]), merged_bitarray, min_slot);
                }
                BitAnd(mask, merged_bitarray, min_slot);
                BitNot_And(merged_bitarray, mask, min_slot);
                // --------------------------------------------
                for (j = 0; j < num_points; j++) {
                    if(BITTEST(merged_bitarray, j)) {
                        int local_idx = j;
                        cur_points = arr_indices[local_idx].info;
                        if (query_config.min_num_finest_level_points) {
                            // num_finest = cur_points->num_finest_level_points[query_config.target_level];
                            num_finest = cur_points->num_finest_level_nodes[query_config.target_level];
                            if (query_config.target_level == 0) num_finest -= 1;  // last level only counts the current node once, which is already counted by setting the initial value to 1
                        }
                        if (!(query_config.blind)) {
                            if (checked[local_idx]) {
                                // Compute distance
                                if (dci_inst->transform) {
                                    cur_dist = transform_compute_dist_query(cur_points->data_loc, query, dim);
                                }
                                else {
                                    cur_dist = compute_dist(cur_points->data_loc, query, dim);
                                }
                                candidate_dists[local_idx] = cur_dist;
                                checked[local_idx] = 0;

                                add_to_list(num_candidates, num_neighbours, top_candidates, &num_returned,
                                                    cur_dist, cur_points, &last_top_candidate_dist, query_config,
                                                    &num_returned_finest_level_points, &last_top_candidate, i, init, num_finest);
                                num_candidates++;
                            }
                            else {
                                cur_dist = candidate_dists[local_idx];
                            }
                        }
                        else {
                            if (checked[local_idx]) {
                                if (num_finest > 0) {
                                    candidate_dists[local_idx] = top_index_priority;
                                    checked[local_idx] = 0;
                                    top_candidates[num_candidates].info = cur_points;
                                    num_candidates++;
                                    if (query_config.min_num_finest_level_points) {
                                        num_returned_finest_level_points += num_finest;
                                    }
                                }
                            }
                            else if (top_index_priority > candidate_dists[local_idx]) {
                                candidate_dists[local_idx] = top_index_priority;
                            }
                        }
                    }
                }
                returned_num[i] = dci_next_closest_proj(&(arr_indices[i*num_points]), &(left_pos[i]), &(right_pos[i]), query_proj[i], num_points, &(returned_ids[i*CLOSEST]), &(index_priority[i]));
                if (returned_num[i] == 0) {
                    index_priority[i] = DBL_MAX;
                }
            }
        }
        if (num_candidates >= num_neighbours &&
            num_returned_finest_level_points >= query_config.min_num_finest_level_points) {
            if (k + visit_proj >= num_projs_to_visit || num_candidates >= num_points_to_retrieve) {
                break;
            }
        }
        k += visit_proj;
    }

    if (query_config.blind) {
        for (int j = 0; j < num_candidates; j++) {
            top_candidates[j].key = candidate_dists[top_candidates[j].info->local_id];
        }
        num_returned = min_i(num_candidates, num_points_to_retrieve);
    }
    else {
        if (num_returned > num_neighbours) {
            qsort(top_candidates, num_returned, sizeof(idx_arr), dci_compare_data_idx_arr_dist);
            if (query_config.min_num_finest_level_points) {
                num_returned_finest_level_points = 0;
                if (query_config.target_level == 0)
                    num_returned_finest_level_points = 1;
                int j = 0;
                for (j = 0; j < num_returned-1; j++) {
                    // num_returned_finest_level_points += top_candidates[j].info->num_finest_level_points[query_config.target_level];
                    if (query_config.target_level)
                        num_returned_finest_level_points += top_candidates[j].info->num_finest_level_nodes[query_config.target_level];
                    else
                        num_returned_finest_level_points += (top_candidates[j].info->num_finest_level_nodes[query_config.target_level] - 1);
                    if (num_returned_finest_level_points >= query_config.min_num_finest_level_points) {
                        break;
                    }
                }
                num_returned = max_i(num_neighbours, j + 1);
            }
            else {
                num_returned = num_neighbours;
            }
        }
    }

    return num_returned;
}

static int dci_query_single_point_single_level_(
    int num_comp_indices, int num_simp_indices, int dim,
    additional_info* point, int num_neighbours,
    const bf16_t* const query, const float query_proj,
    const dci_query_config query_config, idx_arr* const top_candidates,
    bool cumu, dci* dci_inst, long long query_id) {
    int i, j, k, m, h, top_h;
    int num_indices = num_comp_indices * num_simp_indices;
    assert(num_indices == 1);
    assert(num_simp_indices == 1);
    assert(num_comp_indices == 1);
    float top_index_priority, cur_dist;
    int cur_pos;
    int num_returned_finest_level_points = 0;
    int num_candidates = 0;
    int num_points = point->cell_indices[0].num_data;
    if (num_points == 0) return 0;
    float init = -1.0;
    int num_finest = 0;

    float last_top_candidate_dist = init;  // The distance of the k^th closest candidate found so far
    int last_top_candidate = -1;
    int num_returned = 0;
    idx_arr* arr_indices;

    int left_pos;
    int right_pos;
    additional_info* cur_points;

    assert(num_neighbours > 0);

    int num_points_to_retrieve =
        max_i(query_config.num_to_retrieve,
            (int)ceil(query_config.prop_to_retrieve * num_points));
    int num_projs_to_visit = max_i(
        query_config.num_to_visit * num_simp_indices,
        (int)ceil(query_config.prop_to_visit * num_points * num_simp_indices));

    // need to lock otherwise there could be conflicts
    #ifdef USE_OPENMP
    omp_set_lock(&(point->lock));
    #endif
    if (dci_inst->transform && point->cell_indices[0].num_data > 0 && btree_p_valueof(btree_p_first(&(point->cell_indices[0]))).info->max_sq_norm < dci_inst->max_sq_norm) {  // need to update the tree based on the new max_sq_norm
        int old_num_nodes = point->cell_indices[0].num_leaf_nodes;
        update_max_sq_norm(num_indices, point, dci_inst->max_sq_norm, dci_inst->add_proj_vec, dci_inst->sq_norm_list,  dci_inst->token2nodeIndex, dci_inst->token2nodeOffset, &(dci_inst->num_leaf_nodes), dci_inst->stack, &(dci_inst->page_status), dci_inst->dim, dci_inst->update_addr, &(dci_inst->leaf_list), &(dci_inst->max_leaves));
        int new_num_nodes = point->cell_indices[0].num_leaf_nodes;
        if (new_num_nodes != old_num_nodes) {
            // Get the level of the point
            int level = 0;
            int total = point->num_finest_level_nodes[0] - 1;
            while (total > 0) {
                total -= point->num_finest_level_nodes[level + 1];
                level += 1;
            }
            int diff = new_num_nodes - old_num_nodes;
            point->num_finest_level_nodes[level] += diff;
            point->num_finest_level_nodes[0] += diff;
            for (additional_info* temp_cell = point->parent_info;
                temp_cell->id != -1; temp_cell = temp_cell->parent_info) {
                temp_cell->num_finest_level_nodes[level] += diff;
                temp_cell->num_finest_level_nodes[0] += diff;
            }
        }
    }
    else if (point->flag == 1) {  // need to update arr_indices
        update_arr_indices(num_indices, point);
    }
    #ifdef USE_OPENMP
    omp_unset_lock(&(point->lock));
    #endif
    arr_indices = point->arr_indices;

    if (num_neighbours >= num_points) {
        int cnt = 0;
        for (i = 0; i < num_points; i++) {
            if (query_config.min_num_finest_level_points > 0 && arr_indices[i].info->num_finest_level_points[query_config.target_level] == 0)
                continue;  // skip
            cur_points = arr_indices[i].info;
            // Compute distance
            if (dci_inst->transform) {
                cur_dist = transform_compute_dist(cur_points->data_loc, query, dim, dci_inst->max_sq_norm, dci_inst->sq_norm_list[cur_points->id], dci_inst->sq_norm_list[query_id]);
            }
            else {
                cur_dist = compute_dist(cur_points->data_loc, query, dim);
            }
            top_candidates[cnt].key = cur_dist;
            top_candidates[cnt++].info = cur_points;
        }

        qsort(top_candidates, cnt, sizeof(idx_arr), dci_compare_data_idx_arr_dist);

        return cnt;
    }

    left_pos = dci_search_index(arr_indices, query_proj, num_points);
    right_pos = left_pos + 1;

    cur_pos = dci_next_closest_proj_(arr_indices, &(left_pos), &(right_pos), query_proj, num_points);
    // assert(cur_pos >= 0);    // There should be at least one point in the index
    if (cur_pos == -1) return 0;
    cur_points = arr_indices[cur_pos].info;

    k = 0;
    while (k < num_points) {
        if (query_config.min_num_finest_level_points) {
            num_finest = cur_points->num_finest_level_points[query_config.target_level];
        }
        if (query_config.min_num_finest_level_points == 0 || num_finest > 0) {
            // Compute distance
            if (dci_inst->transform) {
                cur_dist = transform_compute_dist(cur_points->data_loc, query, dim, dci_inst->max_sq_norm, dci_inst->sq_norm_list[cur_points->id], dci_inst->sq_norm_list[query_id]);
            }
            else {
                cur_dist = compute_dist(cur_points->data_loc, query, dim);
            }
            add_to_list_(num_candidates, num_neighbours, top_candidates, &num_returned,
                                cur_dist, cur_points, &last_top_candidate_dist, query_config,
                                &num_returned_finest_level_points, &last_top_candidate, i, init, num_finest);
            num_candidates++;
        }
        if (num_candidates >= num_neighbours &&
            num_returned_finest_level_points >= query_config.min_num_finest_level_points) {
            if (k + 1 >= num_projs_to_visit || num_candidates >= num_points_to_retrieve) {
                break;
            }
        }
        cur_pos = dci_next_closest_proj_(arr_indices, &(left_pos), &(right_pos), query_proj, num_points);
        if (cur_pos >= 0) {
            cur_points = arr_indices[cur_pos].info;
        }
        k++;
    }

    if (num_returned > num_neighbours) {
        qsort(top_candidates, num_returned, sizeof(idx_arr), dci_compare_data_idx_arr_dist);
        if (query_config.min_num_finest_level_points) {
            num_returned_finest_level_points = 0;
            int j = 0;
            for (j = 0; j < num_returned - 1; j++) {
                num_returned_finest_level_points += top_candidates[j].info->num_finest_level_points[query_config.target_level];
                if (num_returned_finest_level_points >= query_config.min_num_finest_level_points) {
                    break;
                }
            }
            num_returned = max_i(num_neighbours, j + 1);
        }
        else {
            num_returned = num_neighbours;
        }
    }

    return num_returned;
}

static int dci_query_single_point(
    int num_comp_indices, int num_simp_indices, int dim, int num_levels,
    additional_info* root, int num_populated_levels, int num_neighbours,
    idx_arr* points_to_expand, idx_arr** points_to_expand_next, int* num_top_candidates,
    const bf16_t* const query, const float* const query_proj,
    dci_query_config query_config, idx_arr* const top_candidates,
    bool cumu, dci* const dci_inst) {
    int i, j, k;
    float init = (-FLT_MAX);
    int candidates_num = 0;
    int candidates_num_ = num_neighbours;
    float last_top_candidate_dist = init;  // The distance of the k^th closest candidate found so far
    float last_top_candidate_dist_ = init;  // The distance of the k^th closest candidate found so far
    int last_top_candidate = -1;
    int last_top_candidate_ = -1;
    int num_indices = num_comp_indices * num_simp_indices;
    int num_points_to_expand;
    int max_num_points_to_expand = max_i(query_config.field_of_view, num_neighbours);
    if (query_config.blind) {
        max_num_points_to_expand += num_comp_indices - 1;
    }
    int num_finest_level_points_to_expand;
    int* token2nodeIndex = dci_inst->token2nodeIndex;

    assert(num_neighbours > 0);
    assert(cumu);  // Current version only supports cumulative case
    assert(num_populated_levels <= num_levels);

    int temp_idx = 0;
    if (num_populated_levels <= 1) {
        if (query_config.blind) {
            query_config.num_to_retrieve = num_neighbours;
            query_config.prop_to_retrieve = -1.0;
        }
        query_config.min_num_finest_level_points = 0;

        num_points_to_expand = dci_query_single_point_single_level(
            num_comp_indices, num_simp_indices, dim, root,
            num_neighbours, query, query_proj,
            query_config, points_to_expand, cumu, dci_inst);

        temp_idx = num_points_to_expand;
    }
    else {
        assert(query_config.field_of_view > 0);

        if (query_config.blind) {
            query_config.num_to_retrieve = query_config.field_of_view;
            query_config.prop_to_retrieve = -1.0;
        }
        if (cumu) {
            query_config.target_level = 0;
        }
        else {
            query_config.target_level = num_levels - num_populated_levels + 1;
        }
        query_config.min_num_finest_level_points = num_neighbours;
        num_points_to_expand = dci_query_single_point_single_level(
            num_comp_indices, num_simp_indices, dim, root,
            query_config.field_of_view, query,
            query_proj, query_config, points_to_expand, cumu, dci_inst);
        assert(num_points_to_expand > 0);

        for (i = num_levels - 2; i >= num_levels - num_populated_levels + 1; i--) {
            if (cumu && num_populated_levels > 1) {
                int node_ids[num_neighbours];
                int cnt = 0;
                idx_arr cur;
                for (int j = 0; j < num_points_to_expand; j++) {
                    cur = points_to_expand[j];
                    // The first half is for nodes
                    int idx = token2nodeIndex[points_to_expand[j].info->id];
                    int jj = 0;
                    for (; jj < cnt; jj++) {  // If the node is already in the list
                        if (idx == node_ids[jj])
                            break;
                    }
                    if (jj == cnt) {
                        if (candidates_num >= num_neighbours) {
                            if (top_candidates[last_top_candidate].key > cur.key) {
                                top_candidates[last_top_candidate] = cur;
                                last_top_candidate_dist = init;
                                for (int jjj = 0; jjj < candidates_num; jjj++) {
                                    if (top_candidates[jjj].key > last_top_candidate_dist) {
                                        last_top_candidate_dist = top_candidates[jjj].key;
                                        last_top_candidate = jjj;
                                    }
                                }
                            }
                            else
                                break;
                        }
                        else {
                            top_candidates[candidates_num++] = cur;
                            if (cur.key > last_top_candidate_dist) {
                                last_top_candidate_dist = cur.key;
                                last_top_candidate = candidates_num - 1;
                            }
                        }
                        node_ids[cnt] = idx;
                        cnt += 1;
                        if (cnt == num_neighbours)
                            break;
                    }

                    // The second half is for tokens
                    if (candidates_num_ >= num_neighbours * 2) {
                        if (top_candidates[last_top_candidate_].key > cur.key) {
                            top_candidates[last_top_candidate_] = cur;
                            last_top_candidate_dist_ = init;
                            for (int jj = num_neighbours; jj < num_neighbours * 2; jj++) {
                                if (top_candidates[jj].key > last_top_candidate_dist_) {
                                    last_top_candidate_dist_ = top_candidates[jj].key;
                                    last_top_candidate_ = jj;
                                }
                            }
                        }
                    }
                    else {
                        top_candidates[candidates_num_++] = cur;
                        if (cur.key > last_top_candidate_dist_) {
                            last_top_candidate_dist_ = cur.key;
                            last_top_candidate_ = candidates_num_ - 1;
                        }
                    }
                }
            }
// #pragma omp parallel for if(dci_inst->parallel_level >= 3) num_threads(2) schedule(static)
            for (int j = 0; j < num_points_to_expand; j++) {
                additional_info* point = points_to_expand[j].info;
                num_top_candidates[j] = dci_query_single_point_single_level(
                    num_comp_indices, num_simp_indices, dim, point,
                    query_config.field_of_view, query,
                    query_proj, query_config, points_to_expand_next[j], cumu, dci_inst);

                assert(num_top_candidates[j] <= max_num_points_to_expand * BTREE_LEAF_MAX_NUM_SLOTS);
            }

            temp_idx = 0;
            for (int j = 0; j < num_points_to_expand; j++) {
                for (k = 0; k < num_top_candidates[j]; k++) {
                    points_to_expand[temp_idx++] = points_to_expand_next[j][k];
                }
            }
            qsort(points_to_expand, temp_idx, sizeof(idx_arr), dci_compare_data_idx_arr_dist);

            if (num_neighbours > 1) {
                num_finest_level_points_to_expand = 0;
                k = 0;
                int node_ids[temp_idx];
                int cnt = 0;
                for (int j = 0; j < temp_idx; j++) {
                    additional_info* tmp_info = points_to_expand[j].info;
                    int idx = token2nodeIndex[tmp_info->id];
                    assert(tmp_info->num_finest_level_nodes[0] > 0);
                    int jj = 0;
                    for (; jj < cnt; jj++) {  // If the node is already in the list
                        if (idx == node_ids[jj])
                            break;
                    }
                    if (jj == cnt) {
                        num_finest_level_points_to_expand += tmp_info->num_finest_level_nodes[0];
                        node_ids[cnt] = idx;
                        cnt += 1;
                    }
                    else {
                        num_finest_level_points_to_expand += (tmp_info->num_finest_level_nodes[0] - 1);
                    }
                    k++;
                    if (num_finest_level_points_to_expand >= num_neighbours) {
                        break;
                    }
                }
                num_points_to_expand = max_i(min_i(query_config.field_of_view, temp_idx), k);
                if (num_points_to_expand > query_config.field_of_view) {
                    perror("Try to increase field_of_view in the query config\n"); 
                    exit(EXIT_FAILURE);
                }
            }
            else {
                num_points_to_expand = min_i(query_config.field_of_view, temp_idx);
            }
        }
        if (query_config.blind) {
            query_config.num_to_retrieve = num_neighbours;
            query_config.prop_to_retrieve = -1.0;
        }
        query_config.min_num_finest_level_points = 0;

        if (cumu && num_populated_levels > 1) {
            int node_ids[num_neighbours];
            int cnt = 0;
            idx_arr cur;
            for (j = 0; j < num_points_to_expand; j++) {
                cur = points_to_expand[j];
                // The first half is for nodes
                int idx = token2nodeIndex[points_to_expand[j].info->id];
                int jj = 0;
                for (; jj < cnt; jj++) {  // If the node is already in the list
                    if (idx == node_ids[jj])
                        break;
                }
                if (jj == cnt) {
                    if (candidates_num >= num_neighbours) {
                        if (top_candidates[last_top_candidate].key > cur.key) {
                            top_candidates[last_top_candidate] = cur;
                            last_top_candidate_dist = init;
                            for (int jjj = 0; jjj < candidates_num; jjj++) {
                                if (top_candidates[jjj].key > last_top_candidate_dist) {
                                    last_top_candidate_dist = top_candidates[jjj].key;
                                    last_top_candidate = jjj;
                                }
                            }
                        }
                        else
                            break;
                    }
                    else {
                        top_candidates[candidates_num++] = cur;
                        if (cur.key > last_top_candidate_dist) {
                            last_top_candidate_dist = cur.key;
                            last_top_candidate = candidates_num - 1;
                        }
                    }
                    node_ids[cnt] = idx;
                    cnt += 1;
                    if (cnt == num_neighbours)
                        break;
                }

                // The second half is for tokens
                if (candidates_num_ >= num_neighbours * 2) {
                    if (top_candidates[last_top_candidate_].key > cur.key) {
                        top_candidates[last_top_candidate_] = cur;
                        last_top_candidate_dist_ = init;
                        for (int jj = num_neighbours; jj < num_neighbours * 2; jj++) {
                            if (top_candidates[jj].key > last_top_candidate_dist_) {
                                last_top_candidate_dist_ = top_candidates[jj].key;
                                last_top_candidate_ = jj;
                            }
                        }
                    }
                }
                else {
                    top_candidates[candidates_num_++] = cur;
                    if (cur.key > last_top_candidate_dist_) {
                        last_top_candidate_dist_ = cur.key;
                        last_top_candidate_ = candidates_num_ - 1;
                    }
                }
            }
        }
// #pragma omp parallel for if(dci_inst->parallel_level >= 3) num_threads(2) schedule(static)
        for (int j = 0; j < num_points_to_expand; j++) {
            additional_info* point = points_to_expand[j].info;
            num_top_candidates[j] = dci_query_single_point_single_level(
                num_comp_indices, num_simp_indices, dim, point,
                num_neighbours, query, query_proj,
                query_config, points_to_expand_next[j], cumu, dci_inst);

            assert(num_top_candidates[j] <= max_num_points_to_expand * BTREE_LEAF_MAX_NUM_SLOTS);
        }

        temp_idx = 0;
        for (int j = 0; j < num_points_to_expand; j++) {
            for (k = 0; k < num_top_candidates[j]; k++) {
                points_to_expand[temp_idx++] = points_to_expand_next[j][k];
            }
        }
        qsort(points_to_expand, temp_idx, sizeof(idx_arr), dci_compare_data_idx_arr_dist);

        num_points_to_expand = min_i(num_neighbours, temp_idx);
    }

    assert(num_points_to_expand <= temp_idx);

    if (cumu && num_populated_levels > 1) {
        int node_ids[num_neighbours];
        int cnt = 0;
        idx_arr cur;
        for (int j = 0; j < temp_idx; j++) {
            cur = points_to_expand[j];
            // The first half is for nodes
            int idx = token2nodeIndex[points_to_expand[j].info->id];
            int jj = 0;
            for (; jj < cnt; jj++) {  // If the node is already in the list
                if (idx == node_ids[jj])
                    break;
            }
            if (jj == cnt) {
                if (candidates_num >= num_neighbours) {
                    if (top_candidates[last_top_candidate].key > cur.key) {
                        top_candidates[last_top_candidate] = cur;
                        last_top_candidate_dist = init;
                        for (int jjj = 0; jjj < candidates_num; jjj++) {
                            if (top_candidates[jjj].key > last_top_candidate_dist) {
                                last_top_candidate_dist = top_candidates[jjj].key;
                                last_top_candidate = jjj;
                            }
                        }
                    }
                    else
                        break;
                }
                else {
                    top_candidates[candidates_num++] = cur;
                    if (cur.key > last_top_candidate_dist) {
                        last_top_candidate_dist = cur.key;
                        last_top_candidate = candidates_num-1;
                    }
                }
                node_ids[cnt] = idx;
                cnt += 1;
                if (cnt == num_neighbours)
                    break;
            }

            // The second half is for tokens
            if (candidates_num_ >= num_neighbours * 2) {
                if (top_candidates[last_top_candidate_].key > cur.key) {
                    top_candidates[last_top_candidate_] = cur;
                    last_top_candidate_dist_ = init;
                    for (int jj = num_neighbours; jj < num_neighbours * 2; jj++) {
                        if (top_candidates[jj].key > last_top_candidate_dist_) {
                            last_top_candidate_dist_ = top_candidates[jj].key;
                            last_top_candidate_ = jj;
                        }
                    }
                }
            }
            else {
                top_candidates[candidates_num_++] = cur;
                if (cur.key > last_top_candidate_dist_) {
                    last_top_candidate_dist_ = cur.key;
                    last_top_candidate_ = candidates_num_ - 1;
                }
            }
        }
        num_points_to_expand = min_i(num_neighbours, candidates_num);
    }
    else {
        assert(num_points_to_expand > 0);
        idx_arr cur;
        for (j = 0; j < num_points_to_expand; j++) {
            cur = points_to_expand[j];
            top_candidates[candidates_num++] = cur;
            top_candidates[candidates_num_++] = cur;
        }
    }

    qsort(top_candidates, candidates_num, sizeof(idx_arr), dci_compare_data_idx_arr_dist);
    qsort(top_candidates + num_neighbours, candidates_num_ - num_neighbours, sizeof(idx_arr), dci_compare_data_idx_arr_dist);

    return num_points_to_expand;
}

static inline int dci_query_single_point_(
    int num_comp_indices, int num_simp_indices, int dim, int num_levels,
    additional_info* root, int num_populated_levels, int num_neighbours,
    const bf16_t* const query, const float query_proj,
    dci_query_config query_config, idx_arr* const top_candidates,
    bool cumu, dci* const dci_inst, long long query_id, int num_upper_points, additional_info** upper_level_cells_ret,
    bf16_t* points_matrix, float* all_distances, float* points_sq_norms) {
    int i, j, k;
    float init = (-FLT_MAX);
    int candidates_num = 0;
    float last_top_candidate_dist = init;  // The distance of the k^th closest candidate found so far
    int last_top_candidate = -1;
    int num_indices = num_comp_indices * num_simp_indices;
    int num_points_to_expand;
    assert(num_neighbours == 1);
    assert(num_upper_points > 0);
    assert(query_config.blind == false);
    int max_num_points_to_expand = max_i(query_config.field_of_view, num_neighbours);
    if (query_config.blind) {
        max_num_points_to_expand += num_comp_indices - 1;
    }

    if (num_upper_points <= dci_inst->numa_threshold) {
        // Check all points on the upper level
        float dist;
        float min_dist = FLT_MAX;
        additional_info* assigned_parent = NULL;

        if (cblas_enabled && points_matrix != NULL && all_distances != NULL && points_sq_norms != NULL) {
            if (dci_inst->transform) {
                compute_distances_transform_blas(
                    query, points_matrix, num_upper_points, dim,
                    dci_inst->max_sq_norm,
                    dci_inst->sq_norm_list[query_id],
                    points_sq_norms,
                    all_distances);
            } else {
                compute_distances_matrix_blas(
                    query, points_matrix, num_upper_points, dim,
                    points_sq_norms,
                    all_distances);
            }

            int min_idx = find_min_distance_simd(all_distances, num_upper_points, &min_dist);
            assigned_parent = upper_level_cells_ret[min_idx];
        }
        else {
            for (int upper_idx = 0; upper_idx < num_upper_points; upper_idx++) {
                additional_info* upper_cell = upper_level_cells_ret[upper_idx];
                const bf16_t* upper_point_data = upper_cell->data_loc;
                long long parent_id = upper_cell->id;

                if (dci_inst->transform) {
                    dist = transform_compute_dist(query, upper_point_data, dim,
                        dci_inst->max_sq_norm,
                        dci_inst->sq_norm_list[query_id],
                        dci_inst->sq_norm_list[parent_id]);
                } else {
                    dist = compute_dist(query, upper_point_data, dim);
                }

                if (dist < min_dist) {
                    min_dist = dist;
                    assigned_parent = upper_cell;
                }
            }
        }

        // Update parent's max_child_dist
        if (assigned_parent != NULL && min_dist > assigned_parent->max_child_dist) {
            assigned_parent->max_child_dist = min_dist;
        }

        top_candidates[0].info = assigned_parent;
        top_candidates[0].key = min_dist;

        return 1;
    }

    idx_arr* points_to_expand = malloc(sizeof(idx_arr) * max_num_points_to_expand * max_num_points_to_expand);
    idx_arr* points_to_expand_next = malloc(sizeof(idx_arr) * max_num_points_to_expand * max_num_points_to_expand);
    // idx_arr points_to_expand[max_num_points_to_expand * max_num_points_to_expand];
    // idx_arr points_to_expand_next[max_num_points_to_expand * max_num_points_to_expand];

    int num_top_candidates[max_num_points_to_expand];
    int num_finest_level_points_to_expand;

    assert(num_populated_levels <= num_levels);

    int temp_idx = 0;
    if (num_populated_levels <= 1) {
        if (query_config.blind) {
            query_config.num_to_retrieve = num_neighbours;
            query_config.prop_to_retrieve = -1.0;
        }
        query_config.min_num_finest_level_points = 0;

        num_points_to_expand = dci_query_single_point_single_level_(
            num_comp_indices, num_simp_indices, dim, root,
            num_neighbours, query, query_proj,
            query_config, points_to_expand, cumu, dci_inst, query_id);

        temp_idx = num_points_to_expand;
    }
    else {
        assert(query_config.field_of_view > 0);

        if (query_config.blind) {
            query_config.num_to_retrieve = query_config.field_of_view;
            query_config.prop_to_retrieve = -1.0;
        }
        query_config.target_level = num_levels - num_populated_levels + 1;
        query_config.min_num_finest_level_points = num_neighbours;
        num_points_to_expand = dci_query_single_point_single_level_(
            num_comp_indices, num_simp_indices, dim, root,
            query_config.field_of_view, query,
            query_proj, query_config, points_to_expand, cumu, dci_inst, query_id);
        assert(num_points_to_expand > 0);

        for (i = num_levels - 2; i >= num_levels - num_populated_levels + 1; i--) {
// #pragma omp parallel for if(dci_inst->parallel_level >= 3) num_threads(dci_inst->inner_inner_threads)
            for (int j = 0; j < num_points_to_expand; j++) {
                additional_info* point = points_to_expand[j].info;
                num_top_candidates[j] = dci_query_single_point_single_level_(
                    num_comp_indices, num_simp_indices, dim, point,
                    query_config.field_of_view, query,
                    query_proj, query_config, &points_to_expand_next[j * max_num_points_to_expand], cumu, dci_inst, query_id);

                assert(num_top_candidates[j] <= max_num_points_to_expand);
            }

            temp_idx = 0;
            for (int j = 0; j < num_points_to_expand; j++) {
                for (k = 0; k < num_top_candidates[j]; k++) {
                    points_to_expand[temp_idx++] = points_to_expand_next[j * max_num_points_to_expand + k];
                }
            }
            qsort(points_to_expand, temp_idx, sizeof(idx_arr), dci_compare_data_idx_arr_dist);

            num_points_to_expand = min_i(query_config.field_of_view, temp_idx);
        }
        if (query_config.blind) {
            query_config.num_to_retrieve = num_neighbours;
            query_config.prop_to_retrieve = -1.0;
        }
        query_config.min_num_finest_level_points = 0;
// #pragma omp parallel for if(dci_inst->parallel_level >= 3) num_threads(dci_inst->inner_inner_threads)
        for (int j = 0; j < num_points_to_expand; j++) {
            additional_info* point = points_to_expand[j].info;
            num_top_candidates[j] = dci_query_single_point_single_level_(
                num_comp_indices, num_simp_indices, dim, point,
                num_neighbours, query, query_proj,
                query_config, &points_to_expand_next[j * max_num_points_to_expand], cumu, dci_inst, query_id);

            assert(num_top_candidates[j] <= max_num_points_to_expand);
        }

        temp_idx = 0;
        for (int j = 0; j < num_points_to_expand; j++) {
            for (k = 0; k < num_top_candidates[j]; k++) {
                points_to_expand[temp_idx++] = points_to_expand_next[j * max_num_points_to_expand + k];
            }
        }
        qsort(points_to_expand, temp_idx, sizeof(idx_arr), dci_compare_data_idx_arr_dist);

        num_points_to_expand = min_i(num_neighbours, temp_idx);
    }

    assert(num_points_to_expand <= temp_idx);

    assert(num_points_to_expand == 1);
    // idx_arr cur;
    // for (j = 0; j < num_points_to_expand; j++) {
    //     cur = points_to_expand[j];
    //     top_candidates[candidates_num++] = cur;
    // }
    // qsort(top_candidates, candidates_num, sizeof(idx_arr), dci_compare_data_idx_arr_dist);
    top_candidates[0] = points_to_expand[0];

    free(points_to_expand);
    free(points_to_expand_next);
    
    return num_points_to_expand;
}

static void dci_assign_parent(
    int num_comp_indices, int num_simp_indices, int dim, int num_levels,
    additional_info* root, const int num_populated_levels, const int num_queries,
    const long long* selected_query_pos, const bf16_t* const query,
    const float* const query_proj, const dci_query_config query_config,
    tree_node* const assigned_parent, dci* const dci_inst, int num_upper_points, additional_info** upper_level_cells_ret,
    bf16_t* points_matrix, float* points_sq_norms) {
    int num_indices = num_comp_indices * num_simp_indices;
    float* all_distances = NULL;

    // Adaptive thread count
    int adaptive_threads = dci_inst->inner_threads * 4;

    int max_thread = 1;
    #ifdef USE_OPENMP
    if (dci_inst->parallel_level >= 2) {
        max_thread = (num_queries < dci_inst->numa_threshold) ? 2 : adaptive_threads;
    }
    #endif

    if (cblas_enabled && num_upper_points > 50) {
        all_distances = (float*)malloc(num_upper_points * max_thread * sizeof(float));
    }

    // Choose scheduling strategy based on workload size
    if (num_queries < dci_inst->numa_threshold) {
        // Small workload: use static scheduling to minimize overhead
#pragma omp parallel for if(dci_inst->parallel_level >= 2) num_threads(2) schedule(static)
        for (int j = 0; j < num_queries; j++) {
            int t_index = 0;
            #ifdef USE_OPENMP
            if (dci_inst->parallel_level >= 2) {
                t_index = omp_get_thread_num();
            }
            #endif
            int cur_num_returned;
            idx_arr top_candidate;

            cur_num_returned = dci_query_single_point_(
                num_comp_indices, num_simp_indices, dim, num_levels, root,
                num_populated_levels, 1,
                &(query[((long long int) selected_query_pos[j]) * dim]),
                query_proj[selected_query_pos[j]], query_config,
                &top_candidate, false, dci_inst, selected_query_pos[j] + dci_inst->next_point_id,
                num_upper_points, upper_level_cells_ret,
                points_matrix, all_distances + t_index * num_upper_points, points_sq_norms);

            assigned_parent[j].parent = top_candidate.info;
            assigned_parent[j].dist = top_candidate.key;
            assigned_parent[j].child = selected_query_pos[j];
        }
    } else {
        // Large workload: use dynamic scheduling with larger chunks for better NUMA locality
        int chunk_size = num_queries / adaptive_threads / 12;
        chunk_size = (chunk_size < 1) ? 1 : chunk_size;
#pragma omp parallel for if(dci_inst->parallel_level >= 2) num_threads(adaptive_threads) schedule(dynamic, chunk_size)
        for (int j = 0; j < num_queries; j++) {
            int t_index = 0;
            #ifdef USE_OPENMP
            if (dci_inst->parallel_level >= 2) {
                t_index = omp_get_thread_num();
            }
            #endif
            int cur_num_returned;
            idx_arr top_candidate;

            cur_num_returned = dci_query_single_point_(
                num_comp_indices, num_simp_indices, dim, num_levels, root,
                num_populated_levels, 1,
                &(query[((long long int) selected_query_pos[j]) * dim]),
                query_proj[selected_query_pos[j]], query_config,
                &top_candidate, false, dci_inst, selected_query_pos[j] + dci_inst->next_point_id,
                num_upper_points, upper_level_cells_ret,
                points_matrix, all_distances + t_index * num_upper_points, points_sq_norms);

            assigned_parent[j].parent = top_candidate.info;
            assigned_parent[j].dist = top_candidate.key;
            assigned_parent[j].child = selected_query_pos[j];
        }
    }

    if (cblas_enabled && num_upper_points > 50) {
        free(all_distances);
    }
}

void dci_query(dci* const dci_inst, const int dim,
    const int num_queries, const bf16_t* const query,
    int num_neighbours,
    dci_query_config query_config, bool* mask,
    int** const nearest_neighbours,
    float** const nearest_neighbour_dists,
    int* const num_returned) {
    int num_indices = dci_inst->num_comp_indices * dci_inst->num_simp_indices;
    int i, j;
    float* query_proj;
    assert(dci_inst->root != NULL);
    assert(dim == dci_inst->dim);
    assert(num_neighbours > 0);
    if (posix_memalign((void**)&query_proj, 64, sizeof(float) * num_indices * num_queries) != 0) {
        perror("Memory allocation failed!\n");
        return;
    }
    // Convert query to BF16 for matmul
    matmul(num_indices, num_queries, dim, dci_inst->proj_vec, query, query_proj);

    if (dci_inst->transform)
        query_transform(query, num_queries, dim, query_proj, num_indices);
    
    if (num_neighbours > dci_inst->num_points) {
        num_neighbours = dci_inst->num_points;
    }
    if (query_config.field_of_view > dci_inst->num_points) {
        query_config.field_of_view = dci_inst->num_points;
    }
    if (query_config.num_to_visit > dci_inst->num_points) {
        query_config.num_to_visit = dci_inst->num_points;
    }
    if (query_config.num_to_retrieve > dci_inst->num_points) {
        query_config.num_to_retrieve = dci_inst->num_points;
    }
    int max_num_points_to_expand = max_i(query_config.field_of_view, num_neighbours);
    
    int* token2nodeIndex = dci_inst->token2nodeIndex;

    if (num_queries == 1) {
        // Query once
        idx_arr* points_to_expand = (idx_arr *)malloc(sizeof(idx_arr) * max_num_points_to_expand * max_num_points_to_expand * BTREE_LEAF_MAX_NUM_SLOTS);
        idx_arr** points_to_expand_next = (idx_arr**)malloc(sizeof(idx_arr*) * max_num_points_to_expand);
        for (j = 0; j < max_num_points_to_expand; j++) {
            points_to_expand_next[j] = (idx_arr *)malloc(sizeof(idx_arr) * max_num_points_to_expand * BTREE_LEAF_MAX_NUM_SLOTS);
        }
        int* num_top_candidates = (int*)malloc(sizeof(int)*max_num_points_to_expand);

        int cur_num_returned;
        idx_arr top_candidate[num_neighbours * 2];

        cur_num_returned = dci_query_single_point(
            dci_inst->num_comp_indices, dci_inst->num_simp_indices,
            dci_inst->dim, dci_inst->num_levels, dci_inst->root,
            dci_inst->num_levels, num_neighbours, points_to_expand,
            points_to_expand_next, num_top_candidates,
            query, query_proj, query_config, top_candidate, true, dci_inst);

        nearest_neighbours[0] = (int*)malloc(sizeof(int) * cur_num_returned * 2);
        if (num_returned) {
            num_returned[0] = cur_num_returned;
        }
        if (nearest_neighbour_dists) {
            nearest_neighbour_dists[0] = (float*)malloc(sizeof(float) * cur_num_returned);
        }

        for (i = 0; i < cur_num_returned; i++) {
            nearest_neighbours[0][i] = token2nodeIndex[top_candidate[i].info->id];
            if (nearest_neighbour_dists) {
                if (dci_inst->transform)
                    nearest_neighbour_dists[0][i] = -1 * top_candidate[i].key;
                else
                    nearest_neighbour_dists[0][i] = top_candidate[i].key;
            }
        }
        for (i = num_neighbours; i < num_neighbours + cur_num_returned; i++) {
            nearest_neighbours[0][i] = top_candidate[i].info->id;
            if (nearest_neighbour_dists) {
                if (dci_inst->transform)
                    nearest_neighbour_dists[0][i] = -1 * top_candidate[i].key;
                else
                    nearest_neighbour_dists[0][i] = top_candidate[i].key;
            }
        }

        for (int i = 0; i < max_num_points_to_expand; i++) {
            free(points_to_expand_next[i]);
        }
        free(points_to_expand);
        free(points_to_expand_next);
        free(num_top_candidates);
        free(query_proj);
    }
    else{
        int max_thread = 1;
        #ifdef USE_OPENMP
        max_thread = omp_get_max_threads();
        #endif
        idx_arr* points_to_expand = (idx_arr *)malloc(sizeof(idx_arr) * max_num_points_to_expand * max_num_points_to_expand * max_thread);
        idx_arr** points_to_expand_next = (idx_arr**)malloc(sizeof(idx_arr*) * max_num_points_to_expand * max_thread);
        for (j = 0; j < max_num_points_to_expand * max_thread; j++) {
            points_to_expand_next[j] = (idx_arr *)malloc(sizeof(idx_arr) * max_num_points_to_expand);
        }
        int* num_top_candidates = (int*)malloc(sizeof(int)*max_num_points_to_expand * max_thread);

#pragma omp parallel for if(dci_inst->parallel_level >= 2) num_threads(dci_inst->inner_threads)
        for (j = 0; j < num_queries; j++) {
            if (mask != NULL && !(mask[j])) continue;
            int t_index = 0;
            #ifdef USE_OPENMP
            if (dci_inst->parallel_level >= 2) {
                t_index = omp_get_thread_num();
            }
            #endif
            int i;
            int cur_num_returned;
            idx_arr top_candidate[num_neighbours * 2];

            cur_num_returned = dci_query_single_point(
                dci_inst->num_comp_indices, dci_inst->num_simp_indices,
                dci_inst->dim, dci_inst->num_levels, dci_inst->root,
                dci_inst->num_levels, num_neighbours, &(points_to_expand[max_num_points_to_expand * max_num_points_to_expand * t_index]),
                &(points_to_expand_next[max_num_points_to_expand * t_index]), &(num_top_candidates[max_num_points_to_expand * t_index]),
                &(query[j * dim]), &(query_proj[j * num_indices]), query_config, top_candidate, true, dci_inst);

            nearest_neighbours[j] = (int*)malloc(sizeof(int) * cur_num_returned * 2);
            if (num_returned) {
                num_returned[j] = cur_num_returned;
            }
            if (nearest_neighbour_dists) {
                nearest_neighbour_dists[j] = (float*)malloc(sizeof(float) * cur_num_returned);
            }

            for (i = 0; i < cur_num_returned; i++) {
                nearest_neighbours[j][i] = token2nodeIndex[top_candidate[i].info->id];
                if (nearest_neighbour_dists) {
                    if (dci_inst->transform)
                        nearest_neighbour_dists[j][i] = -1 * top_candidate[i].key;
                    else
                        nearest_neighbour_dists[j][i] = top_candidate[i].key;
                }
            }
            for (i = num_neighbours; i < num_neighbours + cur_num_returned; i++) {
                nearest_neighbours[j][i] = top_candidate[i].info->id;
                if (nearest_neighbour_dists) {
                    if (dci_inst->transform)
                        nearest_neighbour_dists[j][i] = -1 * top_candidate[i].key;
                    else
                        nearest_neighbour_dists[j][i] = top_candidate[i].key;
                }
            }
        }

        for (int i = 0; i < max_num_points_to_expand * max_thread; i++) {
            free(points_to_expand_next[i]);
        }
        free(points_to_expand);
        free(points_to_expand_next);
        free(num_top_candidates);
        free(query_proj);
    }
}

long long dci_add_query(dci* const dci_inst, const int dim,
    const int num_points, const bf16_t* const data, const bf16_t* const value, const int num_levels,
    dci_query_config construction_query_config, const long long* const data_ids,
    int target_level, float* new_data_proj, bool* add_mask,
    const int num_queries, const bf16_t* const query,
    int num_neighbours, dci_query_config query_config, bool* delete_mask,
    int** const nearest_neighbours, float** const nearest_neighbour_dists,
    int* const num_returned, bool random, int interval, int X, float anchor_threshold) {

    long long first_id = dci_add(dci_inst, dim, num_points, data, value,
                                num_levels, construction_query_config,
                                data_ids, target_level, new_data_proj,
                                add_mask, random, interval, X, anchor_threshold);

    dci_query(dci_inst, dim, num_queries, query, num_neighbours, query_config, 
            delete_mask, nearest_neighbours, nearest_neighbour_dists, 
            num_returned);

    return first_id;
}

// This part of code is to update the address of the points in DCI
void dci_address_update(dci* const dci_inst, int* indices, PyObject* new_address, int update_num, int offset) {
    // Indices: the indices of the leaf nodes to be updated
    // new_address: the new address of each leaf node to be updated
    // update_num: the number of leaf nodes to be updated
    // offset: the offset of the value embedding regarding the key embedding

    btree_p_leaf_node** leaf_list = dci_inst->leaf_list;

    // Adaptive thread count based on workload size and NUMA considerations
    int adaptive_threads = dci_inst->inner_threads;
    
    if (update_num < dci_inst->numa_threshold) {
        adaptive_threads = 2;
    } else {
        // For large workloads, use full thread count but consider chunk scheduling
        adaptive_threads = dci_inst->inner_threads;
    }

#pragma omp parallel for if(dci_inst->parallel_level >= 2) num_threads(adaptive_threads)
    for (int i = 0; i < update_num; i++) {
        btree_p_leaf_node* leaf = leaf_list[indices[i]];
        PyObject* ptr_obj = PyList_GetItem(new_address, i);
        uintptr_t addr = PyLong_AsUnsignedLongLong(ptr_obj);
        bf16_t* fptr = (bf16_t*) addr;
        // Copy the data to the new address
        if (leaf->data_loc == NULL) {
            for (int j = 0; j < leaf->num_slots_used; j++) {
                memcpy(fptr + (j * dci_inst->dim), leaf->slot_data[j].info->data_loc, (dci_inst->dim) * sizeof(bf16_t));
                memcpy(fptr + offset + (j * dci_inst->dim), leaf->slot_data[j].info->inc_data_loc, (dci_inst->dim) * sizeof(bf16_t));
            }
        }
        else {
            // // For Testing
            // for (int j = 0; j < leaf->num_slots_used; j++) {
            //     if (leaf->slot_data[j].info->data_loc[0] != leaf->data_loc[j * dci_inst->dim])
            //         printf("%d, %d, %f, %f\n", leaf->id, leaf->slot_data[j].info->id, leaf->slot_data[j].info->data_loc[0], leaf->data_loc[j * dci_inst->dim]);
            //     // assert(leaf->slot_data[j].info->data_loc == leaf->data_loc + (j * dci_inst->dim));
            //     // assert(leaf->slot_data[j].info->inc_data_loc == leaf->inc_data_loc + (j * dci_inst->dim));
            // }
            memcpy(fptr, leaf->data_loc, (leaf->num_slots_used) * (dci_inst->dim) * sizeof(bf16_t));
            memcpy(fptr + offset, leaf->inc_data_loc, (leaf->num_slots_used) * (dci_inst->dim) * sizeof(bf16_t));
            free(leaf->data_loc);
            free(leaf->inc_data_loc);
        }
        // Update the address
        leaf->data_loc = fptr;
        leaf->inc_data_loc = fptr + offset;
        for (int j = 0; j < leaf->num_slots_used; j++) {
            leaf->slot_data[j].info->data_loc = fptr + (j * dci_inst->dim);
            leaf->slot_data[j].info->inc_data_loc = fptr + offset + (j * dci_inst->dim);
        }
        dci_inst->page_status[indices[i]] = 0;
    }
}