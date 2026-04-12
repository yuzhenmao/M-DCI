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

#include "Python.h"
#include "numpy/arrayobject.h"
#include "dci.h"
#include "util.h"
#include "debug.h"
#include "hashtable_pp.h"
#include "btree_i.h"
#include<immintrin.h>
#include <x86intrin.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>
_Static_assert(sizeof(bool) == 1, "C bool must be 1 byte to alias NPY_BOOL");

#if PY_MAJOR_VERSION >= 3
#define PY3K
#endif

#ifdef PY3K

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

#endif

// DCI struct with some additional structures for Python-specific bookkeeping
typedef struct py_dci {
    dci dci_inst;
    hashtable_pp hashtable;
    btree_i* cached_tree;
    // PyArrayObject *py_array;
} py_dci;

typedef struct py_dci_list {
    py_dci* dci_inst_list;
    int num_inst;
} py_dci_list;

static inline void capsule_cleanup(PyObject *capsule) {
    void *memory = PyCapsule_GetPointer(capsule, NULL);
    // Use your specific gc implementation in place of free if you have to
    free(memory);
}

// Called automatically by the garbage collector
static void py_dci_free(PyObject *py_dci_inst_wrapper) {
    
    py_dci_list *py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");

    for (int n = 0; n < py_dci_inst_list->num_inst; n++) {
        py_dci *py_dci_inst = &(py_dci_inst_list->dci_inst_list[n]);
        hashtable_pp ht = py_dci_inst->hashtable;
        htentry_pp *x;
        for (int i = 0; i < ht.size; i++) {
            if (ht.entries[i]) {
                x = ht.entries[i];
                while (x) {
                    if (x->value)
                        Py_DECREF(x->value);
                    x = x->next;
                }
            }
        }

        hashtable_pp_free(&(py_dci_inst->hashtable));
        
        if (py_dci_inst->dci_inst.num_points > 0) {
            dci_free(&(py_dci_inst->dci_inst));
        }

        if (py_dci_inst->cached_tree) {
            btree_i_clear(py_dci_inst->cached_tree);
            free(py_dci_inst->cached_tree);
        }
    }

    free(py_dci_inst_list->dci_inst_list);
    free(py_dci_inst_list);
}

static PyObject *py_dci_new(PyObject *self, PyObject *args) {
    
    int dim, num_comp_indices, num_simp_indices, max_volume, num_inst, parallel_level;
    float promotion_prob, promotion_prob_subseq;
    bool debug, transform;
    float* proj_vec;
    PyArrayObject *py_proj_vec;
    
    if (!PyArg_ParseTuple(args, "iiiffibiibO!", &dim, &num_comp_indices, &num_simp_indices, &promotion_prob, &promotion_prob_subseq, &max_volume, &transform, &num_inst, &parallel_level, &debug, &PyArray_Type, &py_proj_vec)) return NULL;

    py_dci_list *py_dci_inst_list = (py_dci_list *)malloc(sizeof(py_dci_list));
    py_dci_inst_list->num_inst = num_inst;

    if (PyArray_DIM(py_proj_vec, 0) == 0)
        proj_vec = NULL;
    else
        proj_vec = (float *)PyArray_DATA(py_proj_vec);
    
    py_dci_inst_list->dci_inst_list = (py_dci *)malloc(sizeof(py_dci) * num_inst);
    py_dci *pdci_inst_list = py_dci_inst_list->dci_inst_list;

    if (promotion_prob_subseq == 0) {
        promotion_prob_subseq = promotion_prob;
    }

    for (int i = 0; i < num_inst; i++) {
        dci_init(&(pdci_inst_list[i].dci_inst), dim, num_comp_indices, num_simp_indices, promotion_prob, promotion_prob_subseq, max_volume, transform, parallel_level, debug, proj_vec);
        hashtable_pp_init(&(pdci_inst_list[i].hashtable), 1, max_volume);
        pdci_inst_list[i].cached_tree = NULL;
    }
    
    // Returns new reference
    PyObject *py_dci_inst_wrapper = PyCapsule_New(py_dci_inst_list, "py_dci_inst_list", py_dci_free);
    
    return py_dci_inst_wrapper;
}

static PyObject *py_dci_address_update(PyObject *self, PyObject *args) {
    int* update_num; 
    int* indices;
    float** new_address;
    PyObject *py_dci_inst_wrapper;
    PyArrayObject *py_indices, *py_update_num;
    PyObject *py_pointer_list;
    py_dci_list *py_dci_inst_list;
    int offset;

    // Two options:
    // 1. Copy data in python
    // 2. Copy data in C
    
    if (!PyArg_ParseTuple(args, "OO!OO!i", &py_dci_inst_wrapper, &PyArray_Type, &py_indices, &py_pointer_list, &PyArray_Type, &py_update_num, &offset)) return NULL;

    if (!py_dci_inst_wrapper) return NULL;

    if (!PyList_Check(py_pointer_list)) return NULL;
    Py_ssize_t list_size = PyList_Size(py_pointer_list);

    indices = (int *)PyArray_DATA(py_indices);
    update_num = (int *)PyArray_DATA(py_update_num);

    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");
    int num_inst = py_dci_inst_list->num_inst;

    assert((int)list_size == num_inst);
    assert(num_inst == PyArray_DIM(py_indices, 0));

    int max_num_pages = PyArray_DIM(py_indices, 1);

#pragma omp parallel for
    for (int idx = 0; idx < num_inst; idx++) {      
        if (update_num[idx] == 0) continue;
        dci *py_dci_inst = &(py_dci_inst_list->dci_inst_list[idx].dci_inst);
        PyObject* sublist = PyList_GetItem(py_pointer_list, idx);
        Py_ssize_t num_ptrs = PyList_Size(sublist);
        assert(num_ptrs == update_num[idx]);
        dci_address_update(py_dci_inst, &(indices[max_num_pages * idx]), sublist, update_num[idx], offset);

        // // We do not handle the reference count here since the original data is stored in an cache object
        // // It can be tricky to use the address of one of the item in a numpy array to get the PyArrayObject* of that numpy array

        // for (int i = 0; i < update_num[idx]; i++) {
        //     long long data_id = indices[max_num_pages * idx + i];
        //     PyArrayObject *py_array = hashtable_pp_get(&(py_dci_inst->hashtable), data_id, NULL);
        //     if (py_array) {
        //         bool ret = hashtable_pp_delete(&(py_dci_inst->hashtable), data_id);
        //         assert(ret);
        //         Py_DECREF(py_array);
        //     }
        //     hashtable_pp_set(&(py_dci_inst->hashtable), data_id, py_data);
        //     Py_INCREF(py_data);
        // }
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *py_dci_cached_tree(PyObject *self, PyObject *args) {
    PyObject *py_dci_inst_wrapper;
    PyArrayObject *py_page_insert_indices, *py_query, *py_returned_num, *py_kept_indices;
    py_dci_list *py_dci_inst_list;
    int *page_insert_indices, *returned_num, *kept_indices;
    float* query;
    int dim, max_returned_num = 0;
    int max_kept_num = 0;
    bool replace;

    if (!PyArg_ParseTuple(args, "OO!O!O!O!b", &py_dci_inst_wrapper, &PyArray_Type, &py_query, &PyArray_Type, &py_page_insert_indices, &PyArray_Type, &py_returned_num, &PyArray_Type, &py_kept_indices, &replace)) return NULL;

    if (!py_dci_inst_wrapper) return NULL;

    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");
    int num_inst = py_dci_inst_list->num_inst;

    if (PyArray_DIM(py_query, 0) == 0) {
        query = NULL;
    }
    else {
        query = (float *)PyArray_DATA(py_query);
        dim = PyArray_DIM(py_query, 1);
    }

    assert(PyArray_DIM(py_page_insert_indices, 0) > 0);
    page_insert_indices = (int *)PyArray_DATA(py_page_insert_indices);
    int num_insert_pages = PyArray_DIM(py_page_insert_indices, 1);

    if (PyArray_DIM(py_returned_num, 0) == 0) {
        returned_num = NULL;
    }
    else {
        returned_num = (int *)PyArray_DATA(py_returned_num);
        max_returned_num = 0;
        for (int i = 0; i < num_inst; i++) {
            if (returned_num[i] > max_returned_num) {
                max_returned_num = returned_num[i];
            }
        }
    }

    if (PyArray_DIM(py_kept_indices, 0) == 0) {
        kept_indices = NULL;
    }
    else {
        kept_indices = (int *)PyArray_DATA(py_kept_indices);
        max_kept_num = PyArray_DIM(py_kept_indices, 1);
    }
    
    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");
    assert(num_inst == PyArray_DIM(py_page_insert_indices, 0));

    npy_intp py_sorted_idx_shape[2];
    py_sorted_idx_shape[0] = num_inst;
    py_sorted_idx_shape[1] = max_returned_num;
    PyArrayObject *py_sorted_page_idx = (PyArrayObject *)PyArray_SimpleNew(2, py_sorted_idx_shape, NPY_INT);
    int *sorted_page_idx_flattened = (int *)PyArray_DATA(py_sorted_page_idx);

#pragma omp parallel for
    for (int idx = 0; idx < num_inst; idx++) {
        bool first_time = false;
        btree_i *cached_tree = py_dci_inst_list->dci_inst_list[idx].cached_tree;
        if (cached_tree != NULL && returned_num != NULL && cached_tree->num_data == returned_num[idx]) {
            // for (int i = 0; i < returned_num[idx]; i++) {
            //     sorted_page_idx_flattened[idx * max_returned_num + i] = -1;
            // }
            btree_i_search_res left_res = btree_i_first(cached_tree);
            int count = 0;
            int prev_num_data = cached_tree->num_data;
            int total_prev_num_data = 2 * prev_num_data;
            for (int i = 0; i < total_prev_num_data; i++) {
                int tmp_value = btree_i_valueof(left_res);
                if (tmp_value < 0) {
                    left_res = btree_i_find_next(left_res);
                    continue;
                }
                assert(tmp_value > 0);
                sorted_page_idx_flattened[idx * max_returned_num + count] = tmp_value - 1;
                left_res = btree_i_find_next(left_res);
                count += 1;
            }
            assert(count == returned_num[idx]);
            btree_i_clear(cached_tree);
            free(cached_tree);
            cached_tree = NULL;
        }
        if (cached_tree == NULL) {
            cached_tree = (btree_i *)malloc(sizeof(btree_i));
            btree_i_init(cached_tree);
            py_dci_inst_list->dci_inst_list[idx].cached_tree = cached_tree;
            first_time = true;
        }
        dci *dci_inst = &(py_dci_inst_list->dci_inst_list[idx].dci_inst);
        btree_p_leaf_node **leaf_list = dci_inst->leaf_list;
        int num_indices = dci_inst->num_comp_indices * dci_inst->num_simp_indices;
        
        float lower_bounds1[num_insert_pages];
        float lower_bounds2[num_insert_pages];

        // ** We only care about 1 comp_indices and 1 simp_indices setting
// #pragma omp parallel for
        int insert_count = 0;
        float max_sq_norm = dci_inst->max_sq_norm;
        float tmp_max_sq_norm, lower_bound1, lower_bound2, proj;
        btree_p_leaf_node *leaf_node;
        bool update;

        for (int i = 0; i < num_insert_pages; i++) {
            if (page_insert_indices[idx * num_insert_pages + i] == -1) break;
            assert(page_insert_indices[idx * num_insert_pages + i] < dci_inst->num_leaf_nodes);
            leaf_node = leaf_list[page_insert_indices[idx * num_insert_pages + i]];
            lower_bound1 = -FLT_MAX;
            lower_bound2 = FLT_MAX;
            tmp_max_sq_norm = leaf_node->slot_data->info->max_sq_norm;
            update = false;
            if (tmp_max_sq_norm < max_sq_norm)
                update = true;
            for (data_pt* d = leaf_node->slot_data; d < leaf_node->slot_data + leaf_node->num_slots_used; d++) {
                if (update) {
                    float p_norm = dci_inst->sq_norm_list[d->info->id];
                    float o_term = sqrt(tmp_max_sq_norm - p_norm);
                    float n_term = sqrt(max_sq_norm - p_norm);
                    proj = d->info->local_dist[0] + (n_term - o_term) * dci_inst->add_proj_vec[0];
                }
                else {
                    proj = d->info->local_dist[0];
                }
                if (proj > lower_bound1) lower_bound1 = proj;
                if (proj < lower_bound2) lower_bound2 = proj;
            }
            lower_bounds1[i] = lower_bound1;
            lower_bounds2[i] = lower_bound2;
            if (first_time) {
                leaf_node->lower_bound1 = lower_bound1;
                leaf_node->lower_bound2 = lower_bound2;
            }
            insert_count += 1;
        }

        if (replace) {
            int temp_p;
            for (int i = 0; i < num_insert_pages; i++) {
                temp_p = page_insert_indices[idx * num_insert_pages + i];
                if (temp_p == -1) break;
                assert(temp_p < dci_inst->num_leaf_nodes);
                leaf_node = leaf_list[temp_p];
                assert(temp_p == leaf_node->id);

                if (leaf_node->lower_bound1 != lower_bounds1[i]) {
                    bool ret;
                    ret = btree_i_delete(cached_tree, leaf_node->lower_bound1, temp_p + 1);
                    assert(ret);
                    leaf_node->lower_bound1 = lower_bounds1[i];
                    btree_i_insert(cached_tree, lower_bounds1[i], temp_p + 1);
                }
                if (leaf_node->lower_bound2 != lower_bounds2[i]) {
                    bool ret;
                    ret = btree_i_delete(cached_tree, leaf_node->lower_bound2, -1 * (temp_p + 1));
                    assert(ret);
                    leaf_node->lower_bound2 = lower_bounds2[i];
                    btree_i_insert(cached_tree, lower_bounds2[i], -1 * (temp_p + 1));
                }
            }
        }
        else if (first_time) {
            float lower_bounds[2 * insert_count];
            int idx_flattened[2 * insert_count];
            for (int i = 0; i < insert_count; i++) {
                lower_bounds[i] = lower_bounds1[i];
                lower_bounds[insert_count + i] = lower_bounds2[i];
                idx_flattened[i] = i;
                idx_flattened[insert_count + i] = i + insert_count;
            }
            qsort_r(idx_flattened, 2 * insert_count, sizeof(int), compare_float, (void *)lower_bounds);
            float lower_bounds_new[2 * insert_count];
            int idx_flattened_new[2 * insert_count];
            for (int i = 0; i < 2 * insert_count; i++) {
                lower_bounds_new[i] = lower_bounds[idx_flattened[i]];
                if (idx_flattened[i] >= insert_count) {
                    idx_flattened_new[i] = -1 * (page_insert_indices[idx * num_insert_pages + idx_flattened[i] - insert_count] + 1);
                }
                else {
                    idx_flattened_new[i] = page_insert_indices[idx * num_insert_pages + idx_flattened[i]] + 1;
                }
            }
            btree_i_bulk_load(cached_tree, &(lower_bounds_new[0]), &(lower_bounds_new[2*insert_count]), &(idx_flattened_new[0]), &(idx_flattened_new[2*insert_count]));

            cached_tree->max_sq_norm = max_sq_norm;
            cached_tree->num_data = insert_count;
        }
        else {
            if (cached_tree->max_sq_norm < max_sq_norm) {
                // rebuild the tree and bulk load
                int prev_num_data = cached_tree->num_data;
                int total_prev_num_data = 2 * prev_num_data;
                float lower_bounds[total_prev_num_data];
                int idx_flattened[total_prev_num_data];

                btree_i_search_res left_res = btree_i_first(cached_tree);
                int tmp_index[prev_num_data];
                int temp_idx = 0;
                for (int i = 0; i < total_prev_num_data; i++) {
                    int tmp_value = btree_i_valueof(left_res);
                    if (tmp_value < 0) {
                        left_res = btree_i_find_next(left_res);
                        continue;
                    }
                    assert(tmp_value > 0);
                    tmp_index[temp_idx] = tmp_value;
                    leaf_node = leaf_list[tmp_value - 1];
                    lower_bound1 = -FLT_MAX;
                    lower_bound2 = FLT_MAX;
                    tmp_max_sq_norm = leaf_node->slot_data->info->max_sq_norm;
                    for (data_pt* d = leaf_node->slot_data; d < leaf_node->slot_data + leaf_node->num_slots_used; d++) {
                        float p_norm = dci_inst->sq_norm_list[d->info->id];
                        float o_term = sqrt(tmp_max_sq_norm - p_norm);
                        float n_term = sqrt(max_sq_norm - p_norm);
                        proj = d->info->local_dist[0] + (n_term - o_term) * dci_inst->add_proj_vec[0];
                        if (proj > lower_bound1) lower_bound1 = proj;
                        if (proj < lower_bound2) lower_bound2 = proj;
                    }
                    leaf_node->lower_bound1 = lower_bound1;
                    leaf_node->lower_bound2 = lower_bound2;

                    lower_bounds[temp_idx] = lower_bound1;
                    lower_bounds[prev_num_data + temp_idx] = lower_bound2;
                    idx_flattened[temp_idx] = temp_idx;
                    idx_flattened[prev_num_data + temp_idx] = prev_num_data + temp_idx;

                    left_res = btree_i_find_next(left_res);
                    temp_idx += 1;
                }
                assert(temp_idx == prev_num_data);

                qsort_r(idx_flattened, total_prev_num_data, sizeof(int), compare_float, (void *)lower_bounds);
                float lower_bounds_new[total_prev_num_data];
                int idx_flattened_new[total_prev_num_data];
                for (int i = 0; i < total_prev_num_data; i++) {
                    lower_bounds_new[i] = lower_bounds[idx_flattened[i]];
                    if (idx_flattened[i] >= prev_num_data) {
                        idx_flattened_new[i] = -1 * (tmp_index[idx_flattened[i] - prev_num_data]);
                    }
                    else {
                        idx_flattened_new[i] = tmp_index[idx_flattened[i]];
                    }
                }
                btree_i_clear(cached_tree);
                free(cached_tree);
                cached_tree = (btree_i *)malloc(sizeof(btree_i));
                btree_i_init(cached_tree);
                py_dci_inst_list->dci_inst_list[idx].cached_tree = cached_tree;

                btree_i_bulk_load(cached_tree, &(lower_bounds_new[0]), &(lower_bounds_new[total_prev_num_data]), &(idx_flattened_new[0]), &(idx_flattened_new[total_prev_num_data]));
                
                cached_tree->max_sq_norm = max_sq_norm;
                cached_tree->num_data = prev_num_data;
            }

            if (returned_num != NULL && returned_num[idx] > 0 && cached_tree->num_data > 0) {
                assert(cached_tree->num_data > returned_num[idx]);
                assert(query != NULL);
                    
                float* query_proj = (float *)malloc(sizeof(float) * num_indices);
                matmul(num_indices, 1, dim, dci_inst->proj_vec, query, query_proj);
                if (dci_inst->transform)
                    query_transform(query, 1, dim, query_proj, num_indices);

                int count = 0;
                float tmp_dists[returned_num[idx]];  // For debugging
                btree_i_search_res left_res = btree_i_first(cached_tree);
                btree_i_search_res right_res = btree_i_last(cached_tree);
                float left_key = btree_i_keyof(left_res) - query_proj[0];
                float right_key = btree_i_keyof(right_res) - query_proj[0];
                left_key = left_key < 0 ? -1 * left_key : left_key;
                right_key = right_key < 0 ? -1 * right_key : right_key;

                while (count < returned_num[idx]) {
                    if (left_res.n == NULL || right_res.n == NULL) break;
                    if (left_key > right_key) {
                        int tmp_value = btree_i_valueof(left_res);
                        tmp_value = tmp_value < 0 ? -1 * tmp_value - 1 : tmp_value - 1;
                        for (int i = 0; i < count; i++) {
                            if (tmp_value == sorted_page_idx_flattened[idx * max_returned_num + i]) {
                                tmp_value = -1;
                                break;
                            }
                        }
                        if (tmp_value >= 0 && kept_indices != NULL) {
                            for (int i = 0; i < max_kept_num; i++) {
                                if (kept_indices[idx * max_kept_num + i] == -1)
                                    break;
                                if (tmp_value == kept_indices[idx * max_kept_num + i]) {
                                    tmp_value = -1;
                                    break;
                                }
                            }
                        }
                        if (tmp_value >= 0) {
                            sorted_page_idx_flattened[idx * max_returned_num + count] = tmp_value;
                            tmp_dists[count] = left_key;
                            count += 1;
                        }
                        left_res = btree_i_find_next(left_res);
                        assert(left_res.n != NULL);
                        left_key = btree_i_keyof(left_res) - query_proj[0];
                        left_key = left_key < 0 ? -1 * left_key : left_key;
                    }
                    else {
                        int tmp_value = btree_i_valueof(right_res);
                        tmp_value = tmp_value < 0 ? -1 * tmp_value - 1 : tmp_value - 1;
                        for (int i = 0; i < count; i++) {
                            if (tmp_value == sorted_page_idx_flattened[idx * max_returned_num + i]) {
                                tmp_value = -1;
                                break;
                            }
                        }
                        if (tmp_value >= 0 && kept_indices != NULL) {
                            for (int i = 0; i < max_kept_num; i++) {
                                if (kept_indices[idx * max_kept_num + i] == -1)
                                    break;
                                if (tmp_value == kept_indices[idx * max_kept_num + i]) {
                                    tmp_value = -1;
                                    break;
                                }
                            }
                        }
                        if (tmp_value >= 0) {
                            sorted_page_idx_flattened[idx * max_returned_num + count] = tmp_value;
                            tmp_dists[count] = right_key;
                            count += 1;
                        }
                        right_res = btree_i_find_prev(right_res);
                        assert(right_res.n != NULL);
                        right_key = btree_i_keyof(right_res) - query_proj[0];
                        right_key = right_key < 0 ? -1 * right_key : right_key;
                    }
                }

                // for (int i = 0; i < returned_num[idx]; i++) {
                //     printf("%d - %f\n", sorted_page_idx_flattened[idx * max_returned_num + i], tmp_dists[i]);
                // }

                free(query_proj);

                // !!! Currently we do not handle the case a page is double inserted and have different lower bounds. This actually should not happen, but we need to handle it in the future.
                for (int i = 0; i < returned_num[idx]; i++) {
                    leaf_node = leaf_list[sorted_page_idx_flattened[idx * max_returned_num + i]];
                    bool ret;
                    ret = btree_i_delete(cached_tree, leaf_node->lower_bound1, leaf_node->id + 1);
                    assert(ret);
                    ret = btree_i_delete(cached_tree, leaf_node->lower_bound2, -1 * (leaf_node->id + 1));
                    assert(ret);
                }
                cached_tree->num_data -= returned_num[idx];
            }

            int temp_p;
            for (int i = 0; i < num_insert_pages; i++) {
                temp_p = page_insert_indices[idx * num_insert_pages + i];
                if (temp_p == -1) break;
                assert(temp_p < dci_inst->num_leaf_nodes);
                leaf_node = leaf_list[temp_p];
                leaf_node->lower_bound1 = lower_bounds1[i];
                leaf_node->lower_bound2 = lower_bounds2[i];
                btree_i_insert(cached_tree, lower_bounds1[i], temp_p + 1);
                btree_i_insert(cached_tree, lower_bounds2[i], -1 * (temp_p + 1));
            }
            cached_tree->num_data += insert_count;
        }
    }

    Py_INCREF(py_sorted_page_idx);
    return py_sorted_page_idx;
}

// Borrows *py_dci_inst_wrapper, py_dci_inst owns at most one copy of *py_data
static PyObject *py_dci_add(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    PyArrayObject *py_data;
    PyArrayObject *py_data_id, *py_token_mask;
    int dim, num_levels, num_to_visit, num_to_retrieve, field_of_view, num_new_points;
    float prop_to_visit, prop_to_retrieve;
    bool blind;
    dci_query_config construction_query_config;
    py_dci *py_dci_inst;
    py_dci_list *py_dci_inst_list;
    float *data;
    long long *d_id;
    
    // start_idx is inclusive, end_idx is exclusive
    if (!PyArg_ParseTuple(args, "OO!O!O!ibiiffi", &py_dci_inst_wrapper, &PyArray_Type, &py_data, &PyArray_Type, &py_data_id, &PyArray_Type, &py_token_mask, 
                                                    &num_levels, &blind, &num_to_visit, &num_to_retrieve, &prop_to_visit, &prop_to_retrieve, &field_of_view)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    if (!py_data) return NULL;
    
    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");
    py_dci_inst = &(py_dci_inst_list->dci_inst_list[0]);
    
    // Assuming row-major layout, py_data->data is N x D, where N is the number of data points and D is the dimensionality
    data = (float *)PyArray_DATA(py_data);
	num_new_points = PyArray_DIM(py_data, 0);
	dim = PyArray_DIM(py_data, 1);
    d_id = (long long *)PyArray_DATA(py_data_id);
    if (PyArray_DIM(py_data_id, 0) == 0) {
        d_id = NULL;
    }

    bool *token_mask = (bool *)PyArray_DATA(py_token_mask);
    // unsigned char *_token_mask = (unsigned char *)PyArray_DATA(py_token_mask);
    // int mask_size = PyArray_DIM(py_token_mask, 0);
    // bool token_mask[mask_size];
    // for (int i = 0; i < mask_size; i++) {
    //     token_mask[i] = (_token_mask[i] != 0);
    // }
	
    if (num_new_points > 0) {
        
        construction_query_config.blind = blind;
        construction_query_config.num_to_visit = num_to_visit;
        construction_query_config.num_to_retrieve = num_to_retrieve;
        construction_query_config.prop_to_visit = prop_to_visit;
        construction_query_config.prop_to_retrieve = prop_to_retrieve;
        construction_query_config.field_of_view = field_of_view;
        construction_query_config.target_level = 0;
        
        long long first_id = dci_add(&(py_dci_inst->dci_inst), dim, num_new_points, data, data, num_levels, construction_query_config, d_id, 0, NULL, token_mask, 0, 0, 1, 1);
        // py_dci_inst->data_idx_offset = start_idx;

        if (d_id == NULL) {
            for (int i = 0; i < num_new_points; i++) {
                hashtable_pp_set(&(py_dci_inst->hashtable), first_id + i, py_data);
                Py_INCREF(py_data);
            }
        }
        else {
            for (int i = 0; i < num_new_points; i++) {
                hashtable_pp_set(&(py_dci_inst->hashtable), d_id[i], py_data);
                Py_INCREF(py_data);
            }
        }
        // py_dci_inst->py_array = py_data;
        
        // py_dci_inst owns a reference to py_data and relinquishes it when database is cleared
        // Py_INCREF(py_data);
    }
    
    Py_INCREF(Py_None);
    return Py_None;
}

// Borrows *py_dci_inst_wrapper, py_dci_inst owns at most one copy of *py_data
static PyObject *py_dci_delete(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    PyArrayObject *py_data, *py_duplicate_delete_ids;
    int dim, num_to_visit, num_to_retrieve, field_of_view, num_delete_points;
    float prop_to_visit, prop_to_retrieve;
    bool blind;
    dci_query_config construction_query_config;
    py_dci *py_dci_inst;
    py_dci_list *py_dci_inst_list;
    long long *data_ids;
    long long *duplicate_delete_ids;
    npy_intp py_duplicate_delete_ids_shape[1];
    
    // start_idx is inclusive, end_idx is exclusive
    if (!PyArg_ParseTuple(args, "OO!biiffi", &py_dci_inst_wrapper, &PyArray_Type, &py_data, &blind, &num_to_visit, &num_to_retrieve, &prop_to_visit, &prop_to_retrieve, &field_of_view)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    if (!py_data) return NULL;
    
    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");
    py_dci_inst = &(py_dci_inst_list->dci_inst_list[0]);
    
    // Assuming row-major layout, py_data->data is N x D, where N is the number of data points and D is the dimensionality
    data_ids = (long long *)PyArray_DATA(py_data);
	num_delete_points = PyArray_DIM(py_data, 0);
	dim = PyArray_DIM(py_data, 1);
	
    if (num_delete_points > 0) {
        
        construction_query_config.blind = blind;
        construction_query_config.num_to_visit = num_to_visit;
        construction_query_config.num_to_retrieve = num_to_retrieve;
        construction_query_config.prop_to_visit = prop_to_visit;
        construction_query_config.prop_to_retrieve = prop_to_retrieve;
        construction_query_config.field_of_view = field_of_view;
        construction_query_config.target_level = 0;
        
        long long tmp_duplicate_delete_ids[num_delete_points];
        int num_deleted = dci_delete(&(py_dci_inst->dci_inst), num_delete_points, data_ids, construction_query_config, tmp_duplicate_delete_ids);
        // py_dci_inst->data_idx_offset = start_idx;
        // py_dci_inst->py_array = py_data;

        py_duplicate_delete_ids_shape[0] = num_delete_points - num_deleted;
        py_duplicate_delete_ids = (PyArrayObject *)PyArray_SimpleNew(1, py_duplicate_delete_ids_shape, NPY_LONGLONG);
        duplicate_delete_ids = (long long *)PyArray_DATA(py_duplicate_delete_ids);
        
        for (int i = 0; i < num_delete_points - num_deleted; i++) {
            duplicate_delete_ids[i] = tmp_duplicate_delete_ids[i];
        }
        
        for (int i = 0; i < num_delete_points; i++) {
            PyArrayObject *py_array = hashtable_pp_get(&(py_dci_inst->hashtable), data_ids[i], NULL);
            if (py_array) {
                bool ret = hashtable_pp_delete(&(py_dci_inst->hashtable), data_ids[i]);
                assert(ret);
                Py_DECREF(py_array);
                num_deleted -= 1;
            }
        }
        assert(num_deleted == 0);

        // py_dci_inst owns a reference to py_data and relinquishes it when database is cleared
        // Py_INCREF(py_data);  // keep track if using py_data, every time use + 1, delete - 1
    }
    
    return Py_BuildValue("N", py_duplicate_delete_ids);
}

static PyObject *py_dci_query(PyObject *self, PyObject *args) {

    PyObject *py_dci_inst_wrapper;
    PyArrayObject *py_query, *py_nearest_neighbour_idx, *py_num_returned, *py_mask;
    int i, j, idx, dim, num_levels, num_inst;
    int num_neighbours, q_num_to_visit, q_num_to_retrieve, q_field_of_view;
    float q_prop_to_visit, q_prop_to_retrieve;
    bool blind;
    dci_query_config query_config;
    bool *mask;
    long long *d_id;
    float *query;
    int *nearest_neighbour_idx;
    int num_comp_indices, num_simp_indices;
    int parallel_level;
    int ratio;
    bool debug, transform;
    py_dci_list *py_dci_inst_list;
    
    // start_idx is inclusive, end_idx is exclusive
    if (!PyArg_ParseTuple(args, "OO!O!ibiiffiii", &py_dci_inst_wrapper, &PyArray_Type, &py_query, &PyArray_Type, &py_mask, 
                                                                                &num_neighbours, &blind, &q_num_to_visit, 
                                                                                &q_num_to_retrieve, &q_prop_to_visit, 
                                                                                &q_prop_to_retrieve, &q_field_of_view,
                                                                                &parallel_level, &ratio))  return NULL;
    
    // Assuming row-major layout, py_query->data is N x D, where N is the number of queries and D is the dimensionality
    query = (float *)PyArray_DATA(py_query);
    mask = (bool *)PyArray_DATA(py_mask);

    query_config.blind = blind;
    query_config.num_to_visit = q_num_to_visit;
    query_config.num_to_retrieve = q_num_to_retrieve;
    query_config.prop_to_visit = q_prop_to_visit;
    query_config.prop_to_retrieve = q_prop_to_retrieve;
    query_config.field_of_view = q_field_of_view;
    query_config.target_level = 0;
    
    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");
    num_inst = py_dci_inst_list->num_inst;

    int num_query_head = num_inst * ratio;
    int num_query = (int)(PyArray_DIM(py_query, 0) / num_query_head);

    npy_intp py_num_returned_shape[1];
    py_num_returned_shape[0] = num_query * num_query_head;
    py_num_returned = (PyArrayObject *)PyArray_SimpleNew(1, py_num_returned_shape, NPY_INT);
    int *num_returned = (int *)PyArray_DATA(py_num_returned);

    npy_intp py_nearest_neighbours_shape[1];
    py_nearest_neighbours_shape[0] = num_query * num_neighbours * num_query_head * 2;
    py_nearest_neighbour_idx = (PyArrayObject *)PyArray_SimpleNew(1, py_nearest_neighbours_shape, NPY_INT);
    nearest_neighbour_idx = (int *)PyArray_DATA(py_nearest_neighbour_idx);

    int **nearest_neighbours = (int **)malloc(sizeof(int *)*num_query*num_query_head);

#pragma omp parallel for if(parallel_level >= 1)
    for (idx = 0; idx < num_query_head; idx++) {
        int dci_idx = idx / ratio;
        dci *py_dci_inst_temp = &(py_dci_inst_list->dci_inst_list[dci_idx].dci_inst);
        py_dci_inst_temp->parallel_level = parallel_level;
        dim = py_dci_inst_temp->dim;

        int **nearest_neighbour_temp = &(nearest_neighbours[num_query * idx]);

        dci_query(py_dci_inst_temp, dim, num_query, &(query[num_query * idx * dim]), num_neighbours, query_config,
            &(mask[num_query * idx]), nearest_neighbour_temp, NULL, 
            &(num_returned[num_query * idx]));

        for (int i = 0; i < num_query; i++) {
            if (!(mask[num_query * idx + i])) continue;
            for (int j = 0; j < num_returned[num_query * idx + i] * 2; j++) {
                nearest_neighbour_idx[num_query * num_neighbours * idx * 2 + i * num_neighbours * 2 + j] = nearest_neighbour_temp[i][j];
            }
        }
        for (int i = 0; i < num_query; i++) {
            if ((mask[num_query * idx + i])) {
                free(nearest_neighbour_temp[i]);
            }
        }
    }
    
    free(nearest_neighbours);

    return Py_BuildValue("NN", py_nearest_neighbour_idx, py_num_returned);
}

static PyObject *py_dci_quick_sort(PyObject *self, PyObject *args) {
    PyObject *py_dci_inst_wrapper;
    PyArrayObject *py_page_indices, *py_input_num, *py_query;
    py_dci_list *py_dci_inst_list;
    int* input_num;
    int* page_indices;
    float* query;

    if (!PyArg_ParseTuple(args, "OO!O!O!", &py_dci_inst_wrapper, &PyArray_Type, &py_query, &PyArray_Type, &py_page_indices, &PyArray_Type, &py_input_num)) return NULL;

    if (!py_dci_inst_wrapper) return NULL;

    page_indices = (int *)PyArray_DATA(py_page_indices);
    input_num = (int *)PyArray_DATA(py_input_num);

    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");
    int num_inst = py_dci_inst_list->num_inst;
    assert(num_inst == PyArray_DIM(py_page_indices, 0));
    int num_pages = PyArray_DIM(py_page_indices, 1);

    query = (float *)PyArray_DATA(py_query);
    int dim = PyArray_DIM(py_query, 1);

    npy_intp py_sorted_idx_shape[2];
    py_sorted_idx_shape[0] = num_inst;
    py_sorted_idx_shape[1] = num_pages;
    PyArrayObject *py_sorted_page_idx = (PyArrayObject *)PyArray_SimpleNew(2, py_sorted_idx_shape, NPY_INT);
    int *sorted_page_idx_flattened = (int *)PyArray_DATA(py_sorted_page_idx);

#pragma omp parallel for
    for (int idx = 0; idx < num_inst; idx++) {
        dci *dci_inst = &(py_dci_inst_list->dci_inst_list[idx].dci_inst);
        btree_p_leaf_node **leaf_list = dci_inst->leaf_list;
        int num_indices = dci_inst->num_comp_indices * dci_inst->num_simp_indices;
        float* query_proj = (float *)malloc(sizeof(float) * num_indices);
        matmul(num_indices, 1, dim, dci_inst->proj_vec, query, query_proj);
        if (dci_inst->transform)
            query_transform(query, 1, dim, query_proj, num_indices);
        
        float lower_bounds[input_num[idx]];
        int idx_flattened[input_num[idx]];

        // ** We only care about 1 comp_indices and 1 simp_indices setting
// #pragma omp parallel for
        for (int i = 0; i < input_num[idx]; i++) {
            btree_p_leaf_node * leaf_node = leaf_list[page_indices[idx * num_pages + i]];
            float lower_bound = -1;
            for (data_pt* d = leaf_node->slot_data; d < leaf_node->slot_data + leaf_node->num_slots_used; d++) {
                float diff = query_proj[0] - d->info->local_dist[0];
                float tmp_dist = diff > 0 ? diff : -diff;
                if (tmp_dist > lower_bound) lower_bound = tmp_dist;
            }
            lower_bounds[i] = lower_bound;
            idx_flattened[i] = i;
        }
        qsort_r(idx_flattened, input_num[idx], sizeof(int), compare_float_r, (void *)lower_bounds);
        for (int i = 0; i < input_num[idx]; i++) {
            sorted_page_idx_flattened[idx * num_pages + i] = page_indices[idx * num_pages + idx_flattened[i]];
        }
        // for (int i = 0; i < input_num[idx]; i++) {
        //     printf("%d - %f\n", sorted_page_idx_flattened[idx * num_pages + i], lower_bounds[idx_flattened[i]]);
        // }
    }

    Py_INCREF(py_sorted_page_idx);
    return py_sorted_page_idx;
}

static PyObject *py_dci_add_query_at_end(PyObject *self, PyObject *args) {

    PyObject *py_dci_inst_wrapper;
    PyArrayObject *py_data, *py_data_id, *py_attention_mask, *py_value, *py_changed_page_list, *py_data_proj;
    PyArrayObject *py_query, *py_nearest_neighbour_idx, *py_num_returned, *py_mask;
    int i, j, idx, dim, num_levels, num_inst, c_num_to_visit, c_num_to_retrieve, c_field_of_view;
    int num_neighbours, q_num_to_visit, q_num_to_retrieve, q_field_of_view, num_query = 0;
    long long max_num_points;
    float c_prop_to_visit, c_prop_to_retrieve, q_prop_to_visit, q_prop_to_retrieve;
    bool blind;
    dci_query_config construction_query_config, query_config;
    float *data, *value;
    bool *mask;
    int* attention_mask;
    long long *d_id;
    float *query;
    int *nearest_neighbour_idx;
    int num_comp_indices, num_simp_indices;
    int parallel_level, interval;
    bool transform, random, do_query, track, update_addr;
    int X;
    float anchor_threshold;
    bool* changed_page_list;
    int old_page_num = 0;
    int ratio;
    float* data_proj_all;
    py_dci_list *py_dci_inst_list;

    // start_idx is inclusive, end_idx is exclusive
    if (!PyArg_ParseTuple(args, "OO!O!O!O!O!iibiiiiffffiiiiLbiO!bbbbO!O!iiif", &py_dci_inst_wrapper, &PyArray_Type, &py_data,
                                                                                &PyArray_Type, &py_data_id, &PyArray_Type, &py_query, &PyArray_Type, &py_value, &PyArray_Type, &py_mask,
                                                                                &num_levels, &num_neighbours, &blind, &c_num_to_visit, &q_num_to_visit,
                                                                                &c_num_to_retrieve, &q_num_to_retrieve, &c_prop_to_visit, &q_prop_to_visit,
                                                                                &c_prop_to_retrieve, &q_prop_to_retrieve, &c_field_of_view, &q_field_of_view,
                                                                                &num_comp_indices, &num_simp_indices, &max_num_points, &transform,
                                                                                &parallel_level, &PyArray_Type, &py_attention_mask, &random, &do_query, &track, &update_addr,
                                                                                &PyArray_Type, &py_changed_page_list, &PyArray_Type, &py_data_proj, &ratio, &interval, &X, &anchor_threshold))  return NULL;
    if (!py_data) return NULL;
    
    // Assuming row-major layout, py_data->data is N x D, where N is the number of data points and D is the dimensionality
    data = (float *)PyArray_DATA(py_data);
	dim = PyArray_DIM(py_data, 1);
    value = (float *)PyArray_DATA(py_value);
    assert(PyArray_DIM(py_data, 0) == PyArray_DIM(py_value, 0));

    d_id = (long long *)PyArray_DATA(py_data_id);  // Never used
    if (PyArray_DIM(py_data_id, 0) == 0) {
        d_id = NULL;
    }

    changed_page_list = (bool *)PyArray_DATA(py_changed_page_list);
    if (!track) {
        changed_page_list = NULL;
    }
    else {
        assert(PyArray_DIM(py_changed_page_list, 0) == PyArray_DIM(py_data, 0));
        old_page_num = PyArray_DIM(py_changed_page_list, 1);
    }

    if (PyArray_DIM(py_data_proj, 0) == 0) {
        data_proj_all = NULL;
    }
    else {
        data_proj_all = (float *)PyArray_DATA(py_data_proj);
        assert(PyArray_DIM(py_data_proj, 0) == PyArray_DIM(py_data, 0));
        assert(PyArray_DIM(py_data_proj, 1) == num_comp_indices * num_simp_indices);
    }
        
    mask = (bool *)PyArray_DATA(py_mask);
    attention_mask = (int *)PyArray_DATA(py_attention_mask);
    
    construction_query_config.blind = blind;
    construction_query_config.num_to_visit = c_num_to_visit;
    construction_query_config.num_to_retrieve = c_num_to_retrieve;
    construction_query_config.prop_to_visit = c_prop_to_visit;
    construction_query_config.prop_to_retrieve = c_prop_to_retrieve;
    construction_query_config.field_of_view = c_field_of_view;
    construction_query_config.target_level = 0;

    if (do_query) {
        query_config.blind = blind;
        query_config.num_to_visit = q_num_to_visit;
        query_config.num_to_retrieve = q_num_to_retrieve;
        query_config.prop_to_visit = q_prop_to_visit;
        query_config.prop_to_retrieve = q_prop_to_retrieve;
        query_config.field_of_view = q_field_of_view;
        query_config.target_level = 0;
    }
    
    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");
    num_inst = py_dci_inst_list->num_inst;

    // Assuming row-major layout, py_query->data is N x D, where N is the number of queries and D is the dimensionality
    int num_query_head = num_inst * ratio;
    if (do_query) {
        query = (float *)PyArray_DATA(py_query);
        num_query = (int)(PyArray_DIM(py_query, 0) / num_query_head);
        assert(num_query == 1);
    }

    int *num_returned = (int *)malloc(sizeof(int)*num_query*num_query_head);

    npy_intp py_nearest_neighbours_shape[1];
    py_nearest_neighbours_shape[0] = num_query * num_neighbours * num_query_head * 2;
    py_nearest_neighbour_idx = (PyArrayObject *)PyArray_SimpleNew(1, py_nearest_neighbours_shape, NPY_INT);
    nearest_neighbour_idx = (int *)PyArray_DATA(py_nearest_neighbour_idx);

    int **nearest_neighbours = (int **)malloc(sizeof(int *)*num_query*num_query_head);
    int num_indices = num_comp_indices * num_simp_indices;

    long long num_points = 0;
    for (i = 0; i < max_num_points; i++) {
        if (mask[i])
            num_points++;
    }

    for (i = 0; i < max_num_points; i++) {
        if (attention_mask[i] > num_points)
            attention_mask[i] = num_points;
    }

    int changed_num_pages[num_inst];
    int **changed_page_idx = (int **)malloc(sizeof(int *) * num_inst);

    // Store original thread count before any modifications
    static int original_max_threads = 0;
    if (original_max_threads == 0) {
        original_max_threads = omp_get_max_threads();
    }
    
    // Enable nested OpenMP parallelism for multi-level threading
    int inner_threads = 1;
    int inner_inner_threads = 1;
    if (parallel_level >= 2) {
        omp_set_nested(1);
        int max_levels = (parallel_level >= 3) ? 3 : 2;
        omp_set_max_active_levels(max_levels);
        // Dynamic thread distribution based on original total available threads
        int total_threads = original_max_threads;
        int outer_threads = (num_inst < total_threads) ? num_inst : total_threads / 2;
        inner_threads = total_threads / outer_threads;
        
        // For 3-level parallelism, further subdivide inner threads
        if (parallel_level >= 3 && inner_threads > 1) {
            inner_inner_threads = (inner_threads > 2) ? 2 : 1;
            inner_threads = (inner_threads > 2) ? inner_threads / 2 : inner_threads;
        }
        
        omp_set_num_threads(outer_threads);
    }

#pragma omp parallel for if(parallel_level >= 1)
    for (idx = 0; idx < num_inst; idx++) {
        dci *py_dci_inst_temp = &(py_dci_inst_list->dci_inst_list[idx].dci_inst);
        py_dci_inst_temp->parallel_level = parallel_level;
        py_dci_inst_temp->inner_threads = inner_threads;
        py_dci_inst_temp->inner_inner_threads = inner_inner_threads;
        py_dci_inst_temp->update_addr = update_addr;
        int prev_num_leaf_nodes = py_dci_inst_temp->num_leaf_nodes;            

        float* data_proj;
        long long first_id;
        int num_indices = num_comp_indices * num_simp_indices;
        bool pre_computed;

        // // Test precomputed data projection
        // if (data_proj_all != NULL) {
        //     py_dci_inst_temp->transform = 0;
        //     float* data_proj_ = &(data_proj_all[max_num_points * idx * num_indices]);
        //     float* data_proj__;
        //     data_projection(num_indices, py_dci_inst_temp, dim, max_num_points,
        //         &(data[max_num_points * idx * dim]), &data_proj__, &(mask[max_num_points * idx]), 0);
        //     for (int i = 0; i < max_num_points * num_indices; i++) {
        //         if (fabsf(data_proj__[i] - data_proj_[i]) > 1e-3) {
        //             printf("%d %d %f %f\n", idx, i, data_proj__[i], data_proj_[i]);
        //         }
        //     }
        //     free(data_proj__);
        // }
        // py_dci_inst_temp->transform = transform;

        if (data_proj_all == NULL) {
            pre_computed = 0;
        }
        else {
            pre_computed = 1;
            data_proj = &(data_proj_all[max_num_points * idx * num_indices]);
        }
        data_projection(num_indices, py_dci_inst_temp, dim, max_num_points,
            &(data[max_num_points * idx * dim]), &data_proj, &(mask[max_num_points * idx]), pre_computed);

        if (max_num_points > 1 && attention_mask[0] > 1) {

            first_id = dci_add(py_dci_inst_temp, dim, attention_mask[0],
                                            &(data[max_num_points * idx * dim]), &(value[max_num_points * idx * dim]), 0,
                                            construction_query_config, NULL, 0,
                                            data_proj, &(mask[max_num_points * idx]), random, interval, X, anchor_threshold);
        }
        else {
            int target_level;
            for (long long i = 0; i < max_num_points; i++) {
                if (!(mask[max_num_points * idx + i])) continue;

                float promotion_prob = py_dci_inst_temp->promotion_prob;

                if (random) {
                    target_level = 0;
                    // Decide which level to add in
                    while (1) {
                        if (target_level > 0)
                            promotion_prob = py_dci_inst_temp->promotion_prob_subseq;
                        if (drand48() > promotion_prob)
                            break;
                        target_level++;
                    }
                }
                else {
                    if (py_dci_inst_temp->num_levels == 0) {
                        target_level = 0;
                        py_dci_inst_temp->next_target_level = 0;
                    }
                    else {
                        target_level = py_dci_inst_temp->next_target_level;
                        if (target_level > 0)
                            promotion_prob = py_dci_inst_temp->promotion_prob_subseq;
                        
                        if (target_level == py_dci_inst_temp->num_levels) {
                            py_dci_inst_temp->next_target_level = 0;
                        }
                        else if (target_level == py_dci_inst_temp->num_levels - 1) {
                            int promo = (int)ceil(1/promotion_prob);
                            if (py_dci_inst_temp->num_points_on_level[target_level] + 1 == promo) {
                                py_dci_inst_temp->next_target_level = target_level + 1;
                            }
                            else {
                                py_dci_inst_temp->next_target_level = 0;
                            }
                        }
                        else {
                            if ((py_dci_inst_temp->num_points_on_level[target_level] + 1 >= py_dci_inst_temp->num_points_on_level[target_level + 1] / promotion_prob))
                                py_dci_inst_temp->next_target_level = target_level + 1;
                            else
                                py_dci_inst_temp->next_target_level = 0;
                        }
                    }
                }

                long long _first_id = dci_add(py_dci_inst_temp, dim, 1, 
                                            &(data[max_num_points * idx * dim + i * dim]), &(value[max_num_points * idx * dim + i * dim]),
                                            py_dci_inst_temp->num_levels, construction_query_config, &(py_dci_inst_temp->next_point_id), target_level, 
                                            &(data_proj[i * num_indices]), &(mask[max_num_points * idx + i]), random, 0, 1, anchor_threshold);
                if (i == 0)
                    first_id = _first_id; 
            }
        }

        if (track) {
            bool* page_status = py_dci_inst_temp->page_status;
            for (int i = 0; i < prev_num_leaf_nodes; i++) {
                if (page_status[i]) {
                    changed_page_list[idx * old_page_num + i] = 1;
                    page_status[i] = 0;
                }
                else {
                    changed_page_list[idx * old_page_num + i] = 0;
                }
            }
        }
        // // We do not handle the reference count here since the original data is stored in an cache object
        // if (d_id == NULL) {
        //     for (int i = 0; i < max_num_points; i++) {
        //         if (!(mask[max_num_points * idx + i])) continue;
        //         hashtable_pp_set(&(py_dci_inst_list->dci_inst_list[idx].hashtable), first_id++, py_data);
        //         Py_INCREF(py_data);
        //     }
        // }
        // else {
        //     for (int i = 0; i < max_num_points; i++) {
        //         if (!(mask[max_num_points * idx + i])) continue;
        //         hashtable_pp_set(&(py_dci_inst_list->dci_inst_list[idx].hashtable), d_id[i], py_data);
        //         Py_INCREF(py_data);
        //     }
        // }
        if (do_query) {
            #pragma omp parallel for if(parallel_level >= 1)
            for (int q_idx = 0; q_idx < ratio; q_idx++) {
                int q_head_idx = ratio * idx + q_idx;
                int **nearest_neighbour_temp = &(nearest_neighbours[num_query * q_head_idx]);
                dci_query(py_dci_inst_temp, dim, num_query, 
                            &(query[num_query * q_head_idx * dim]), num_neighbours, query_config, 
                            &(mask[max_num_points * idx]), nearest_neighbour_temp, NULL, 
                            &(num_returned[num_query * q_head_idx]));
        
                for (int i = 0; i < num_query; i++) {
                    for (int j = 0; j < num_returned[num_query * q_head_idx + i] * 2; j++) {
                        nearest_neighbour_idx[num_query * num_neighbours * q_head_idx * 2 + i * num_neighbours * 2 + j] = nearest_neighbour_temp[i][j];
                    }
                }
                for (int i = 0; i < num_query; i++) {
                    if ((mask[max_num_points * idx + i])) {
                        free(nearest_neighbour_temp[i]);
                    }
                }
            }
        }

        if (!pre_computed)
            free(data_proj);
    }
    
    free(nearest_neighbours);
    free(num_returned);

    // Reset thread count to original value after parallel work
    if (parallel_level >= 2) {
        omp_set_num_threads(original_max_threads);
    }

    // if (track)
    //     return Py_BuildValue("NN", py_nearest_neighbour_idx, py_changed_page_list);

    return Py_BuildValue("NN", py_nearest_neighbour_idx, Py_None);
}

static PyObject *py_dci_add_query(PyObject *self, PyObject *args) {

    PyObject *py_dci_inst_wrapper;
    PyArrayObject *py_data, *py_data_id, *py_attention_mask, *py_data_proj;
    PyArrayObject *py_query, *py_nearest_neighbour_idx, *py_num_returned, *py_mask, *py_value, *py_changed_page_list;
    int i, j, idx, dim, num_levels, num_inst, c_num_to_visit, c_num_to_retrieve, c_field_of_view;
    int num_neighbours, q_num_to_visit, q_num_to_retrieve, q_field_of_view;
    long long max_num_points;
    float c_prop_to_visit, c_prop_to_retrieve, q_prop_to_visit, q_prop_to_retrieve;
    bool blind;
    dci_query_config construction_query_config, query_config;
    float *data, *value;
    bool *mask;
    int* attention_mask;
    long long *d_id;
    float *query;
    int *nearest_neighbour_idx;
    int num_comp_indices, num_simp_indices;
    int parallel_level;
    bool transform, random, do_query, track, update_addr;
    int X;
    float anchor_threshold;
    bool* changed_page_list;
    int old_page_num = 0;
    int ratio, interval;
    float* data_proj_all;
    py_dci_list *py_dci_inst_list;

    // start_idx is inclusive, end_idx is exclusive
    if (!PyArg_ParseTuple(args, "OO!O!O!O!O!iibiiiiffffiiiiLbiO!bbbbO!O!iiif", &py_dci_inst_wrapper, &PyArray_Type, &py_data,
                                                                                &PyArray_Type, &py_data_id, &PyArray_Type, &py_query, &PyArray_Type, &py_value, &PyArray_Type, &py_mask,
                                                                                &num_levels, &num_neighbours, &blind, &c_num_to_visit, &q_num_to_visit,
                                                                                &c_num_to_retrieve, &q_num_to_retrieve, &c_prop_to_visit, &q_prop_to_visit,
                                                                                &c_prop_to_retrieve, &q_prop_to_retrieve, &c_field_of_view, &q_field_of_view,
                                                                                &num_comp_indices, &num_simp_indices, &max_num_points, &transform,
                                                                                &parallel_level, &PyArray_Type, &py_attention_mask, &random, &do_query, &track, &update_addr,
                                                                                &PyArray_Type, &py_changed_page_list, &PyArray_Type, &py_data_proj, &ratio, &interval, &X, &anchor_threshold))  return NULL;
    if (!py_data) return NULL;

    assert(ratio == 1);  // Currently add_query does not support group query
    
    // Assuming row-major layout, py_data->data is N x D, where N is the number of data points and D is the dimensionality
    data = (float *)PyArray_DATA(py_data);
    value = (float *)PyArray_DATA(py_value);
	dim = PyArray_DIM(py_data, 1);
    d_id = (long long *)PyArray_DATA(py_data_id);  // Never used
    if (PyArray_DIM(py_data_id, 0) == 0) {
        d_id = NULL;
    }

    changed_page_list = (bool *)PyArray_DATA(py_changed_page_list);
    if (!track) {
        changed_page_list = NULL;
    }
    else {
        old_page_num = PyArray_DIM(py_changed_page_list, 1);
    }

    if (PyArray_DIM(py_data_proj, 0) == 0) {
        data_proj_all = NULL;
    }
    else {
        data_proj_all = (float *)PyArray_DATA(py_data_proj);
        assert(PyArray_DIM(py_data_proj, 0) == PyArray_DIM(py_data, 0));
        assert(PyArray_DIM(py_data_proj, 1) == num_comp_indices * num_simp_indices);
    }

    // Assuming row-major layout, py_query->data is N x D, where N is the number of queries and D is the dimensionality
    if (do_query)
        query = (float *)PyArray_DATA(py_query);
    mask = (bool *)PyArray_DATA(py_mask);
    attention_mask = (int *)PyArray_DATA(py_attention_mask);
    
    construction_query_config.blind = blind;
    construction_query_config.num_to_visit = c_num_to_visit;
    construction_query_config.num_to_retrieve = c_num_to_retrieve;
    construction_query_config.prop_to_visit = c_prop_to_visit;
    construction_query_config.prop_to_retrieve = c_prop_to_retrieve;
    construction_query_config.field_of_view = c_field_of_view;
    construction_query_config.target_level = 0;

    if (do_query) {
        query_config.blind = blind;
        query_config.num_to_visit = q_num_to_visit;
        query_config.num_to_retrieve = q_num_to_retrieve;
        query_config.prop_to_visit = q_prop_to_visit;
        query_config.prop_to_retrieve = q_prop_to_retrieve;
        query_config.field_of_view = q_field_of_view;
        query_config.target_level = 0;
    }
    
    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");
    num_inst = py_dci_inst_list->num_inst;

    int *num_returned = (int *)malloc(sizeof(int)*max_num_points*num_inst);

    npy_intp py_nearest_neighbours_shape[1];
    py_nearest_neighbours_shape[0] = max_num_points * num_neighbours * num_inst * 2;
    py_nearest_neighbour_idx = (PyArrayObject *)PyArray_SimpleNew(1, py_nearest_neighbours_shape, NPY_INT);
    nearest_neighbour_idx = (int *)PyArray_DATA(py_nearest_neighbour_idx);

    int **nearest_neighbours = (int **)malloc(sizeof(int *)*max_num_points*num_inst);
    int num_indices = num_comp_indices * num_simp_indices;

    long long num_points = 0;
    for (i = 0; i < max_num_points; i++) {
        if (mask[i])
            num_points++;
    }

    for (i = 0; i < max_num_points; i++) {
        if (attention_mask[i] > num_points)
            attention_mask[i] = num_points;
    }

    int changed_num_pages[num_inst];
    int **changed_page_idx = (int **)malloc(sizeof(int *) * num_inst);

    // Store original thread count before any modifications
    static int original_max_threads = 0;
    if (original_max_threads == 0) {
        original_max_threads = omp_get_max_threads();
    }
    
    // Enable nested OpenMP parallelism for multi-level threading
    int inner_threads = 1;
    int inner_inner_threads = 1;
    if (parallel_level >= 2) {
        omp_set_nested(1);
        int max_levels = (parallel_level >= 3) ? 3 : 2;
        omp_set_max_active_levels(max_levels);
        // Dynamic thread distribution based on original total available threads
        int total_threads = original_max_threads;
        int outer_threads = (num_inst < total_threads) ? num_inst : total_threads / 2;
        inner_threads = total_threads / outer_threads;
        
        // For 3-level parallelism, further subdivide inner threads
        if (parallel_level >= 3 && inner_threads > 1) {
            inner_inner_threads = (inner_threads > 2) ? 2 : 1;
            inner_threads = (inner_threads > 2) ? inner_threads / 2 : inner_threads;
        }
        
        omp_set_num_threads(outer_threads);
    }

#pragma omp parallel for if(parallel_level >= 1)
    for (idx = 0; idx < num_inst; idx++) {
        dci *py_dci_inst_temp = &(py_dci_inst_list->dci_inst_list[idx].dci_inst);
        py_dci_inst_temp->parallel_level = parallel_level;
        py_dci_inst_temp->inner_threads = inner_threads;
        py_dci_inst_temp->inner_inner_threads = inner_inner_threads;
        py_dci_inst_temp->update_addr = update_addr;
        int prev_num_leaf_nodes = py_dci_inst_temp->num_leaf_nodes;

        float* data_proj;
        long long first_id;
        int num_indices = num_comp_indices * num_simp_indices;
        bool pre_computed;
        if (data_proj_all == NULL) {
            pre_computed = 0;
        }
        else {
            pre_computed = 1;
            data_proj = &(data_proj_all[max_num_points * idx * num_indices]);
        }
        data_projection(num_indices, py_dci_inst_temp, dim, max_num_points,
            &(data[max_num_points * idx * dim]), &data_proj, &(mask[max_num_points * idx]), pre_computed);

        int **nearest_neighbour_temp = &(nearest_neighbours[max_num_points * idx]);

        if (max_num_points > 1 && attention_mask[0] > 1) {

            if (do_query) {
                dci_add_query(py_dci_inst_temp, dim, attention_mask[0],
                    &(data[max_num_points * idx * dim]), &(value[max_num_points * idx * dim]), num_levels,
                    construction_query_config, NULL, 0, data_proj, &(mask[max_num_points * idx]),
                    attention_mask[0], &(query[max_num_points * idx * dim]), num_neighbours, query_config,
                    &(mask[max_num_points * idx]), nearest_neighbour_temp, NULL,
                    &(num_returned[max_num_points * idx]), random, interval, X, anchor_threshold);

                for (int i = 0; i < attention_mask[0]; i++) {
                    if (!(mask[max_num_points * idx + i])) continue;
                    for (int j = 0; j < num_returned[max_num_points * idx + i] * 2; j++) {
                        nearest_neighbour_idx[max_num_points * num_neighbours * idx * 2 + i * num_neighbours * 2 + j] = nearest_neighbour_temp[i][j];
                    }
                }
                for (int i = 0; i < attention_mask[0]; i++) {
                    if ((mask[max_num_points * idx + i])) {
                        free(nearest_neighbour_temp[i]);
                    }
                }
            }
            else {
                first_id = dci_add(py_dci_inst_temp, dim, attention_mask[0], 
                                                &(data[max_num_points * idx * dim]), &(value[max_num_points * idx * dim]), num_levels,
                                                construction_query_config, NULL, 0, 
                                                data_proj, &(mask[max_num_points * idx]), random, interval, X, anchor_threshold);
            }
        }
        else {
            int target_level;
            for (long long i = 0; i < max_num_points; i++) {
                if (!(mask[max_num_points * idx + i])) continue;

                float promotion_prob = py_dci_inst_temp->promotion_prob;

                if (random) {
                    target_level = 0;
                    // Decide which level to add in
                    while (1) {
                        if (target_level > 0)
                            promotion_prob = py_dci_inst_temp->promotion_prob_subseq;
                        if (drand48() > promotion_prob)
                            break;
                        target_level++;
                    }
                }
                else {
                    if (py_dci_inst_temp->num_levels == 0) {
                        target_level = 0;
                        py_dci_inst_temp->next_target_level = 0;
                    }
                    else {
                        target_level = py_dci_inst_temp->next_target_level;
                        if (target_level > 0)
                            promotion_prob = py_dci_inst_temp->promotion_prob_subseq;
                        
                        if (target_level == py_dci_inst_temp->num_levels) {
                            py_dci_inst_temp->next_target_level = 0;
                        }
                        else if (target_level == py_dci_inst_temp->num_levels - 1) {
                            int promo = (int)ceil(1/promotion_prob);
                            if (py_dci_inst_temp->num_points_on_level[target_level] + 1 == promo) {
                                py_dci_inst_temp->next_target_level = target_level + 1;
                            }
                            else {
                                py_dci_inst_temp->next_target_level = 0;
                            }
                        }
                        else {
                            if ((py_dci_inst_temp->num_points_on_level[target_level] + 1 >= py_dci_inst_temp->num_points_on_level[target_level + 1] / promotion_prob))
                                py_dci_inst_temp->next_target_level = target_level + 1;
                            else
                                py_dci_inst_temp->next_target_level = 0;
                        }
                    }
                }

                // Use the vanilla attention when the number of points is less than the number of neighbours
                // TODO: need to decide the threshold
                if ((py_dci_inst_temp->num_points) < num_neighbours || !do_query) {
                    long long _first_id = dci_add(py_dci_inst_temp, dim, 1, 
                                                &(data[max_num_points * idx * dim + i * dim]), &(value[max_num_points * idx * dim + i * dim]),
                                                py_dci_inst_temp->num_levels, construction_query_config, &(py_dci_inst_temp->next_point_id), target_level, 
                                                &(data_proj[i * num_indices]), &(mask[max_num_points * idx + i]), random, 0, 1, anchor_threshold);
                    if (i == 0)
                        first_id = _first_id;

                    if (do_query) {
                        for (int j = 0; j <= i; j++) {
                            nearest_neighbour_idx[max_num_points * num_neighbours * idx * 2 + i * num_neighbours * 2 + j] = j;
                        }
                    }
                }
                else {
                    long long _first_id = dci_add_query(py_dci_inst_temp, dim, 1, 
                                &(data[max_num_points * idx * dim + i * dim]), &(value[max_num_points * idx * dim + i * dim]),
                                py_dci_inst_temp->num_levels,
                                construction_query_config, &(py_dci_inst_temp->next_point_id), target_level, 
                                &(data_proj[i * num_indices]), &(mask[max_num_points * idx + i]),
                                1, &(query[max_num_points * idx * dim + i * dim]), num_neighbours, query_config, 
                                &(mask[max_num_points * idx + i]), nearest_neighbour_temp, NULL, 
                                &(num_returned[max_num_points * idx]), random, 0, 1, anchor_threshold);
                    
                    if (i == 0)
                        first_id = _first_id;

                    for (int j = 0; j < num_returned[max_num_points * idx] * 2; j++) {
                        nearest_neighbour_idx[max_num_points * num_neighbours * idx * 2 + i * num_neighbours * 2 + j] = nearest_neighbour_temp[0][j];
                    }
                    free(nearest_neighbour_temp[0]);
                }
            }
        }

        if (track) {
            bool* page_status = py_dci_inst_temp->page_status;
            for (int i = 0; i < prev_num_leaf_nodes; i++) {
                if (page_status[i]) {
                    changed_page_list[idx * old_page_num + i] = 1;
                    page_status[i] = 0;
                }
                else {
                    changed_page_list[idx * old_page_num + i] = 0;
                }
            }
        }
        // // We do not handle the reference count here since the original data is stored in an cache object
        // if (d_id == NULL) {
        //     for (int i = 0; i < max_num_points; i++) {
        //         if (!(mask[max_num_points * idx + i])) continue;
        //         hashtable_pp_set(&(py_dci_inst_list->dci_inst_list[idx].hashtable), first_id++, py_data);
        //         Py_INCREF(py_data);
        //     }
        // }
        // else {
        //     for (int i = 0; i < max_num_points; i++) {
        //         if (!(mask[max_num_points * idx + i])) continue;
        //         hashtable_pp_set(&(py_dci_inst_list->dci_inst_list[idx].hashtable), d_id[i], py_data);
        //         Py_INCREF(py_data);
        //     }
        // }
        if (!pre_computed)
            free(data_proj);
    }
    
    free(nearest_neighbours);
    free(num_returned);

    // Reset thread count to original value after parallel work
    if (parallel_level >= 2) {
        omp_set_num_threads(original_max_threads);
    }

    // if (track)
    //     return Py_BuildValue("NN", py_nearest_neighbour_idx, py_changed_page_list);

    return Py_BuildValue("NN", py_nearest_neighbour_idx, Py_None);
}

static PyObject *py_dci_add_query_attention(PyObject *self, PyObject *args) {

    PyArrayObject *py_data, *py_data_id, *py_new_value, *py_attention_mask;
    PyArrayObject *py_query, *py_value, *py_nearest_neighbour_idx, *py_nearest_neighbour_dists, *py_num_returned, *py_mask;
    int i, j, idx, dim, dim_v, num_levels, num_inst, c_num_to_visit, c_num_to_retrieve, c_field_of_view;
    int num_neighbours, q_num_to_visit, q_num_to_retrieve, q_field_of_view;
    long long max_num_points;
    float c_prop_to_visit, c_prop_to_retrieve, q_prop_to_visit, q_prop_to_retrieve, promotion_prob, promotion_prob_subseq;
    bool blind;
    dci_query_config construction_query_config, query_config;
    float *data;
    bool *mask;
    int* attention_mask;
    long long *d_id;
    float *query, *value;
    int *nearest_neighbour_idx;
    int num_comp_indices, num_simp_indices;
    int parallel_level;
    bool debug, transform, random;
    
    // start_idx is inclusive, end_idx is exclusive
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiibiiiiffffiiiiffLbibO!b", &PyArray_Type, &py_data, &PyArray_Type, &py_data_id, 
                                                                                &PyArray_Type, &py_query,  &PyArray_Type, &py_value, &PyArray_Type, &py_mask, 
                                                                                &num_inst, &num_levels, &num_neighbours, &blind, &c_num_to_visit, &q_num_to_visit, 
                                                                                &c_num_to_retrieve, &q_num_to_retrieve, &c_prop_to_visit, &q_prop_to_visit, 
                                                                                &c_prop_to_retrieve, &q_prop_to_retrieve, &c_field_of_view, &q_field_of_view, 
                                                                                &num_comp_indices, &num_simp_indices, &promotion_prob, &promotion_prob_subseq, &max_num_points, &transform, 
                                                                                &parallel_level, &debug, &PyArray_Type, &py_attention_mask, &random))  return NULL;
    if (!py_data) return NULL;
    
    // Assuming row-major layout, py_data->data is N x D, where N is the number of data points and D is the dimensionality
    data = (float *)PyArray_DATA(py_data);
	dim = PyArray_DIM(py_data, 1);
    dim_v = PyArray_DIM(py_value, 1);
    d_id = (long long *)PyArray_DATA(py_data_id);  // Never used
    if (PyArray_DIM(py_data_id, 0) == 0) {
        d_id = NULL;
    }
    // Assuming row-major layout, py_query->data is N x D, where N is the number of queries and D is the dimensionality
    query = (float *)PyArray_DATA(py_query);
    value = (float *)PyArray_DATA(py_value);
    mask = (bool *)PyArray_DATA(py_mask);
    attention_mask = (int *)PyArray_DATA(py_attention_mask);

    if (promotion_prob_subseq == 0) {
        promotion_prob_subseq = promotion_prob;
    }
    
    construction_query_config.blind = blind;
    construction_query_config.num_to_visit = c_num_to_visit;
    construction_query_config.num_to_retrieve = c_num_to_retrieve;
    construction_query_config.prop_to_visit = c_prop_to_visit;
    construction_query_config.prop_to_retrieve = c_prop_to_retrieve;
    construction_query_config.field_of_view = c_field_of_view;
    construction_query_config.target_level = 0;

    query_config.blind = blind;
    query_config.num_to_visit = q_num_to_visit;
    query_config.num_to_retrieve = q_num_to_retrieve;
    query_config.prop_to_visit = q_prop_to_visit;
    query_config.prop_to_retrieve = q_prop_to_retrieve;
    query_config.field_of_view = q_field_of_view;
    query_config.target_level = 0;

    int *num_returned = (int *)malloc(sizeof(int)*max_num_points*num_inst);

    float* new_value_flattened = (float*)calloc(max_num_points * dim_v * num_inst, sizeof(float));
    npy_intp py_new_value_shape[1];
    py_new_value_shape[0] = max_num_points * dim_v * num_inst;
    py_new_value = (PyArrayObject *)PyArray_SimpleNewFromData(1, py_new_value_shape, NPY_FLOAT, new_value_flattened);
    PyObject *capsule = PyCapsule_New(new_value_flattened, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject *) py_new_value, capsule);

    npy_intp py_nearest_neighbours_shape[1];
    py_nearest_neighbours_shape[0] = max_num_points * num_neighbours * num_inst;
    py_nearest_neighbour_idx = (PyArrayObject *)PyArray_SimpleNew(1, py_nearest_neighbours_shape, NPY_INT);
    nearest_neighbour_idx = (int *)PyArray_DATA(py_nearest_neighbour_idx);

    int **nearest_neighbours = (int **)malloc(sizeof(int *)*max_num_points*num_inst);
    float **nearest_neighbour_dists = (float **)malloc(sizeof(float *)*max_num_points*num_inst);

    py_dci *py_dci_inst = (py_dci *)malloc(num_inst * sizeof(py_dci));
    int num_indices = num_comp_indices * num_simp_indices;
    
    for (idx = 0; idx < num_inst; idx++) {
        dci_init(&(py_dci_inst[idx].dci_inst), dim, num_comp_indices, num_simp_indices, promotion_prob, promotion_prob_subseq, max_num_points, transform, parallel_level, debug, NULL);
        hashtable_pp_init(&(py_dci_inst[idx].hashtable), 1, max_num_points);
    }

    long long num_points = 0;
    for (i = 0; i < max_num_points; i++) {
        if (mask[i])
            num_points++;
    }

    for (i = 0; i < max_num_points; i++) {
        if (attention_mask[i] > num_points)
            attention_mask[i] = num_points;
    }

#pragma omp parallel for if(parallel_level >= 1)
    for (idx = 0; idx < num_inst; idx++) {
        dci *py_dci_inst_temp = &(py_dci_inst[idx].dci_inst);
        py_dci_inst_temp->parallel_level = parallel_level;

        float* data_proj;

        int num_indices = num_comp_indices * num_simp_indices;

        data_projection(num_indices, py_dci_inst_temp, dim, max_num_points,
            &(data[max_num_points * idx * dim]), &data_proj, &(mask[max_num_points * idx]), 0);

        float **nearest_neighbour_dists_temp = &(nearest_neighbour_dists[max_num_points * idx]);
        int **nearest_neighbour_temp = &(nearest_neighbours[max_num_points * idx]);
        float *value_temp = &(value[max_num_points * idx * dim_v]);

        assert(py_dci_inst_temp->num_points == 0);

        if (attention_mask[0] > 1) {

            dci_add_query(py_dci_inst_temp, dim, attention_mask[0], 
                &(data[max_num_points * idx * dim]), &(value[max_num_points * idx * dim]), num_levels,
                construction_query_config, NULL, 0, data_proj, &(mask[max_num_points * idx]),
                attention_mask[0], &(query[max_num_points * idx * dim]), num_neighbours, query_config,
                &(mask[max_num_points * idx]), nearest_neighbour_temp, nearest_neighbour_dists_temp, 
                &(num_returned[max_num_points * idx]), random, 0, 1, 1);

            float *new_value_temp;
            __m256 X, Y; // 256-bit values
            __m256 dot = _mm256_setzero_ps(); // set to (0, 0, 0, 0, 0, 0, 0, 0)
            float temp[8];
            for (int i = 0; i < attention_mask[0]; i++) {
                if (!(mask[max_num_points * idx + i])) continue;
                new_value_temp = &(new_value_flattened[(max_num_points * idx + i) * dim_v]);
                for (int j = 0; j < num_returned[max_num_points * idx + i]; j++) {
                    float* x = value_temp + dim_v * nearest_neighbour_temp[i][j];
                    float weight = nearest_neighbour_dists_temp[i][j];
                    float y[8] = {weight, weight, weight, weight, weight, weight, weight, weight};
                    Y = _mm256_loadu_ps(y);
                    int ii;
                    for (ii = 0; ii < dim_v - 8; ii += 8)
                    {
                        X = _mm256_loadu_ps(x + ii); // load chunk of 8 floats
                        dot =  _mm256_mul_ps(X, Y);
                        _mm256_storeu_ps(&temp[0], dot);
                        new_value_temp[ii] += temp[0];
                        new_value_temp[ii+1] += temp[1];
                        new_value_temp[ii+2] += temp[2];
                        new_value_temp[ii+3] += temp[3];
                        new_value_temp[ii+4] += temp[4];
                        new_value_temp[ii+5] += temp[5];
                        new_value_temp[ii+6] += temp[6];
                        new_value_temp[ii+7] += temp[7];
                    }
                    for (; ii < dim_v; ii++)
                        new_value_temp[ii] += x[ii] * weight;
                }
                for (int j = 0; j < num_returned[max_num_points * idx + i] * 2; j++) {
                    nearest_neighbour_idx[max_num_points * num_neighbours * idx * 2 + i * num_neighbours * 2 + j] = nearest_neighbour_temp[i][j];
                }
            }
            for (int i = 0; i < attention_mask[0]; i++) {
                if ((mask[max_num_points * idx + i])) {
                    free(nearest_neighbour_temp[i]);
                    free(nearest_neighbour_dists_temp[i]);
                }
            }
        }
        else {
            int target_level;

            assert(max_num_points > 4);

            for (long long i = 0; i < 4; i++) {
                // Directly conduct dot-product to calcuate the value
                float* q = &(query[max_num_points * idx * dim + i * dim]);
                float max_weight = 0.;
                float weight_list[i + 1];
                // Find the maximum value in the input array to prevent overflow in exp
                for (int j = 0; j <= i; j++) {
                    if (!(mask[max_num_points * idx + j])) continue;
                    float* x = value_temp + dim_v * j;
                    float* k = &(data[max_num_points * idx * dim + j * dim]);
                    float weight = vecmul(k, q, dim);
                    if (weight > max_weight) {
                        max_weight = weight;
                    }
                    weight_list[j] = weight;
                }
                float sum_exp = 0.;
                float sqrt_dim = sqrt(dim);
                // Compute the exponential values and their sum
                for (int j = 0; j <= i; j++) {
                    if (!(mask[max_num_points * idx + j])) continue;
                    weight_list[j] = exp((weight_list[j] - max_weight) / sqrt_dim); // subtract max_input for numerical stability
                    sum_exp += weight_list[j];
                }
                // Normalize the exponential values
                float *new_value_temp;
                __m256 X, Y; // 256-bit values
                __m256 dot = _mm256_setzero_ps(); // set to (0, 0, 0, 0, 0, 0, 0, 0)
                float temp[8];
                new_value_temp = &(new_value_flattened[(max_num_points * idx + i) * dim_v]);
                for (int j = 0; j <= i; j++) {
                    if (!(mask[max_num_points * idx + j])) continue;
                    float* x = value_temp + dim_v * j;
                    float weight = weight_list[j] / sum_exp;
                    float y[8] = {weight, weight, weight, weight, weight, weight, weight, weight};
                    Y = _mm256_loadu_ps(y);
                    int ii;
                    for (ii = 0; ii < dim_v - 8; ii += 8)
                    {
                        X = _mm256_loadu_ps(x + ii); // load chunk of 8 floats
                        dot =  _mm256_mul_ps(X, Y);
                        _mm256_storeu_ps(&temp[0], dot);
                        new_value_temp[ii] += temp[0];
                        new_value_temp[ii+1] += temp[1];
                        new_value_temp[ii+2] += temp[2];
                        new_value_temp[ii+3] += temp[3];
                        new_value_temp[ii+4] += temp[4];
                        new_value_temp[ii+5] += temp[5];
                        new_value_temp[ii+6] += temp[6];
                        new_value_temp[ii+7] += temp[7];
                    }
                    for (; ii < dim_v; ii++)
                        new_value_temp[ii] += x[ii] * weight;
                }
                for (int j = 0; j <= i; j++) {
                    nearest_neighbour_idx[max_num_points * num_neighbours * idx * 2 + i * num_neighbours * 2 + j] = j;
                }
            }

            int* num_points_on_level = (int *)malloc(sizeof(int) * 2);
            num_points_on_level[0] = 0;
            num_points_on_level[1] = 0;
            int next_target_level = 0;

            for (long long i = 4; i < max_num_points; i++) {
                if (!(mask[max_num_points * idx + i])) continue;

                float promo = promotion_prob;

                if (random) {
                    target_level = 0;
                    // Decide which level to add in
                    while (1) {
                        if (target_level > 0)
                            promo = promotion_prob_subseq;
                        if (drand48() > promo)
                            break;
                        target_level++;
                    }
                }
                else {
                    // Decide which level to add in
                    target_level = next_target_level;
                    if (target_level > 0)
                        promo = promotion_prob_subseq;
                    if ((num_points_on_level[target_level] + 1 >= (num_points_on_level[target_level + 1] + 1) / promo))
                        next_target_level = target_level + 1;
                    else
                        next_target_level = 0;

                    if (next_target_level == py_dci_inst_temp->num_levels) {
                        num_points_on_level = (int *)realloc(num_points_on_level, sizeof(int) * (next_target_level + 2));
                        num_points_on_level[next_target_level] = 0;
                        num_points_on_level[next_target_level + 1] = 0;
                    }
                    
                    num_points_on_level[target_level]++;
                }

                // Use the vanilla attention when the number of points is less than the number of neighbours
                // TODO: need to decide the threshold
                if ((py_dci_inst_temp->num_points) < num_neighbours) {
                    long long first_id = dci_add(py_dci_inst_temp, dim, 1, 
                                                &(data[max_num_points * idx * dim + i * dim]), &(value[max_num_points * idx * dim + i * dim]),
                                                py_dci_inst_temp->num_levels, construction_query_config, &i, target_level, 
                                                &(data_proj[i * num_indices]), &(mask[max_num_points * idx + i]), 1, 0, 1, 1);
                    // Directly conduct dot-product to calcuate the value
                    float* q = &(query[max_num_points * idx * dim + i * dim]);
                    float max_weight = 0.;
                    float weight_list[i + 1];
                    // Find the maximum value in the input array to prevent overflow in exp
                    for (int j = 0; j <= i; j++) {
                        if (!(mask[max_num_points * idx + j])) continue;
                        float* x = value_temp + dim_v * j;
                        float* k = &(data[max_num_points * idx * dim + j * dim]);
                        float weight = vecmul(k, q, dim);
                        if (weight > max_weight) {
                            max_weight = weight;
                        }
                        weight_list[j] = weight;
                    }
                    float sum_exp = 0.;
                    float sqrt_dim = sqrt(dim);
                    // Compute the exponential values and their sum
                    for (int j = 0; j <= i; j++) {
                        if (!(mask[max_num_points * idx + j])) continue;
                        weight_list[j] = exp((weight_list[j] - max_weight) / sqrt_dim); // subtract max_input for numerical stability
                        sum_exp += weight_list[j];
                    }
                    // Normalize the exponential values
                    float *new_value_temp;
                    __m256 X, Y; // 256-bit values
                    __m256 dot = _mm256_setzero_ps(); // set to (0, 0, 0, 0, 0, 0, 0, 0)
                    float temp[8];
                    new_value_temp = &(new_value_flattened[(max_num_points * idx + i) * dim_v]);
                    for (int j = 0; j <= i; j++) {
                        if (!(mask[max_num_points * idx + j])) continue;
                        float* x = value_temp + dim_v * j;
                        float weight = weight_list[j] / sum_exp;
                        float y[8] = {weight, weight, weight, weight, weight, weight, weight, weight};
                        Y = _mm256_loadu_ps(y);
                        int ii;
                        for (ii = 0; ii < dim_v - 8; ii += 8)
                        {
                            X = _mm256_loadu_ps(x + ii); // load chunk of 8 floats
                            dot =  _mm256_mul_ps(X, Y);
                            _mm256_storeu_ps(&temp[0], dot);
                            new_value_temp[ii] += temp[0];
                            new_value_temp[ii+1] += temp[1];
                            new_value_temp[ii+2] += temp[2];
                            new_value_temp[ii+3] += temp[3];
                            new_value_temp[ii+4] += temp[4];
                            new_value_temp[ii+5] += temp[5];
                            new_value_temp[ii+6] += temp[6];
                            new_value_temp[ii+7] += temp[7];
                        }
                        for (; ii < dim_v; ii++)
                            new_value_temp[ii] += x[ii] * weight;
                    }
                    for (int j = 0; j <= i; j++) {
                        nearest_neighbour_idx[max_num_points * num_neighbours * idx * 2 + i * num_neighbours * 2 + j] = j;
                    }
                }
                else {
                    dci_add_query(py_dci_inst_temp, dim, 1, 
                                &(data[max_num_points * idx * dim + i * dim]), &(value[max_num_points * idx * dim + i * dim]),
                                py_dci_inst_temp->num_levels,
                                construction_query_config, &i, target_level, 
                                &(data_proj[i * num_indices]), &(mask[max_num_points * idx + i]),
                                1, &(query[max_num_points * idx * dim + i * dim]), num_neighbours, query_config, 
                                &(mask[max_num_points * idx + i]), nearest_neighbour_temp, nearest_neighbour_dists_temp, 
                                &(num_returned[max_num_points * idx]), random, 0, 1, 1);

                    // ===================================================================
                    // Combine the first few tokens with the results returned by the query
                    // ===================================================================
                    float* q = &(query[max_num_points * idx * dim + i * dim]);
                    float max_weight = 0.;
                    float weight_list[4 + num_returned[max_num_points * idx]];
                    // Find the maximum value in the input array to prevent overflow in exp
                    for (int j = 0; j < 4; j++) {
                        if (!(mask[max_num_points * idx + j])) continue;
                        float* k = &(data[max_num_points * idx * dim + j * dim]);
                        float weight = vecmul(k, q, dim);
                        if (weight > max_weight) {
                            max_weight = weight;
                        }
                        weight_list[j] = weight;
                    }
                    for (int j = 0; j < num_returned[max_num_points * idx]; j++) {
                        float weight = nearest_neighbour_dists_temp[0][j];
                        if (weight > max_weight) {
                            max_weight = weight;
                        }
                        weight_list[4+j] = weight;
                    }
                    float sum_exp = 0.;
                    float sqrt_dim = sqrt(dim);
                    // Compute the exponential values and their sum
                    for (int j = 0; j < num_returned[max_num_points * idx] + 4; j++) {
                        if (!(mask[max_num_points * idx + j])) continue;
                        weight_list[j] = exp((weight_list[j] - max_weight) / sqrt_dim); // subtract max_input for numerical stability
                        sum_exp += weight_list[j];
                    }
                    // Normalize the exponential values
                    float *new_value_temp;
                    __m256 X, Y; // 256-bit values
                    __m256 dot = _mm256_setzero_ps(); // set to (0, 0, 0, 0, 0, 0, 0, 0)
                    float temp[8];
                    new_value_temp = &(new_value_flattened[(max_num_points * idx + i) * dim_v]);
                    for (int j = 0; j < num_returned[max_num_points * idx] + 4; j++) {
                        if (!(mask[max_num_points * idx + j])) continue;
                        float* x;
                        if (j < 4)
                            x = value_temp + dim_v * j;
                        else
                            x = value_temp + dim_v * nearest_neighbour_temp[0][j-4];
                        float weight = weight_list[j] / sum_exp;
                        float y[8] = {weight, weight, weight, weight, weight, weight, weight, weight};
                        Y = _mm256_loadu_ps(y);
                        int ii;
                        for (ii = 0; ii < dim_v - 8; ii += 8)
                        {
                            X = _mm256_loadu_ps(x + ii); // load chunk of 8 floats
                            dot = _mm256_mul_ps(X, Y);
                            _mm256_storeu_ps(&temp[0], dot);
                            new_value_temp[ii] += temp[0];
                            new_value_temp[ii+1] += temp[1];
                            new_value_temp[ii+2] += temp[2];
                            new_value_temp[ii+3] += temp[3];
                            new_value_temp[ii+4] += temp[4];
                            new_value_temp[ii+5] += temp[5];
                            new_value_temp[ii+6] += temp[6];
                            new_value_temp[ii+7] += temp[7];
                        }
                        for (; ii < dim_v; ii++)
                            new_value_temp[ii] += x[ii] * weight;
                    }
                    for (int j = 0; j < num_returned[max_num_points * idx]; j++) {
                        nearest_neighbour_idx[max_num_points * num_neighbours * idx * 2 + i * num_neighbours * 2 + j] = nearest_neighbour_temp[0][j];
                    }
                    free(nearest_neighbour_temp[0]);
                    free(nearest_neighbour_dists_temp[0]);
                }
            }

            // for (int i = 0; i < py_dci_inst_temp->num_levels; i++) {
            //     printf("%d   %d\n ", i, py_dci_inst_temp->num_points_on_level[i]);
            // }

            free(num_points_on_level);
        }

        hashtable_pp ht = py_dci_inst[idx].hashtable;
        htentry_pp *x;
        for (int i = 0; i < ht.size; i++) {
            if (ht.entries[i]) {
                x = ht.entries[i];
                while (x) {
                    if (x->value)
                        Py_DECREF(x->value);
                    x = x->next;
                }
            }
        }

        hashtable_pp_free(&(py_dci_inst[idx].hashtable));
        
        dci_free(py_dci_inst_temp);
        free(data_proj);
    }
    
    free(nearest_neighbours);
    free(nearest_neighbour_dists);
    free(num_returned);
    free(py_dci_inst);
    
    return Py_BuildValue("NN", py_new_value, py_nearest_neighbour_idx);
}

// Borrows *py_dci_inst_wrapper, relinquishes *py_data
static PyObject *py_dci_clear(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    py_dci *py_dci_inst;
    py_dci_list *py_dci_inst_list;
    
    if (!PyArg_ParseTuple(args, "O", &py_dci_inst_wrapper)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");
    py_dci_inst = &(py_dci_inst_list->dci_inst_list[0]);
	
    // if (py_dci_inst->py_array) {
    //     Py_DECREF(py_dci_inst->py_array);
    // }
    
    dci_clear(&(py_dci_inst->dci_inst));
    hashtable_pp_clear(&(py_dci_inst->hashtable));
    btree_i_clear(py_dci_inst->cached_tree);
    // py_dci_inst->py_array = NULL;
    // py_dci_inst->data_idx_offset = 0;
    
    Py_INCREF(Py_None);
    return Py_None;
    
}

// Borrows *py_dci_inst_wrapper, relinquishes *py_data
static PyObject *py_dci_reset(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    py_dci *py_dci_inst;
    py_dci_list *py_dci_inst_list;
    
    if (!PyArg_ParseTuple(args, "O", &py_dci_inst_wrapper)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");

    for (int i = 0; i < py_dci_inst_list->num_inst; i++) {
        py_dci_inst = &(py_dci_inst_list->dci_inst_list[i]);
        dci_reset(&(py_dci_inst->dci_inst));
    }
    
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *py_dci_reset_proj(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    PyArrayObject *py_proj_vec;
    bool transform;
    py_dci_list *py_dci_inst_list;
    
    if (!PyArg_ParseTuple(args, "OO!b", &py_dci_inst_wrapper, &PyArray_Type, &py_proj_vec, &transform)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");

    float* temp_proj_vec = (float *)PyArray_DATA(py_proj_vec);

    for (int idx = 0; idx < py_dci_inst_list->num_inst; idx++) {
        dci *dci_inst = &(py_dci_inst_list->dci_inst_list[idx].dci_inst);
        int num_indices = dci_inst->num_comp_indices * dci_inst->num_simp_indices;
        int dim = dci_inst->dim;

        if (transform) {
            for (int j = 0; j < num_indices; j++) {
                for (int i = 0; i < dim; i++) {
                    dci_inst->proj_vec[i + j * (dim)] = temp_proj_vec[i + j * (dim+1)];
                }
                dci_inst->add_proj_vec[j] = temp_proj_vec[dim + j * (dim+1)];
            }
        }
        else {
            for (int j = 0; j < num_indices; j++) {
                for (int i = 0; i < dim; i++) {
                    dci_inst->proj_vec[i + j * dim] = temp_proj_vec[i + j * dim];
                }
            }
        }
    }
    
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *py_dci_get_num_points(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    PyArrayObject *py_num_points;
    py_dci_list *py_dci_inst_list;
    
    if (!PyArg_ParseTuple(args, "O", &py_dci_inst_wrapper)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");

    npy_intp py_num_points_shape[1];
    py_num_points_shape[0] = py_dci_inst_list->num_inst;
    py_num_points = (PyArrayObject *)PyArray_SimpleNew(1, py_num_points_shape, NPY_INT);
    int* num_points = (int *)PyArray_DATA(py_num_points);

    for (int i = 0; i < py_dci_inst_list->num_inst; i++) {
        num_points[i] = (py_dci_inst_list->dci_inst_list[i].dci_inst).num_points;
    }
    
	return Py_BuildValue("N", py_num_points);
}

static PyObject *py_dci_get_num_levels(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    PyArrayObject *py_num_levels;
    py_dci_list *py_dci_inst_list;
    
    if (!PyArg_ParseTuple(args, "O", &py_dci_inst_wrapper)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");

    npy_intp py_num_levels_shape[1];
    py_num_levels_shape[0] = py_dci_inst_list->num_inst;
    py_num_levels = (PyArrayObject *)PyArray_SimpleNew(1, py_num_levels_shape, NPY_INT);
    int* num_levels = (int *)PyArray_DATA(py_num_levels);

    for (int i = 0; i < py_dci_inst_list->num_inst; i++) {
        num_levels[i] = (py_dci_inst_list->dci_inst_list[i].dci_inst).num_levels;
    }
    
	return Py_BuildValue("N", py_num_levels);
}

static PyObject *py_dci_get_num_leaves(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    PyArrayObject *py_num_leaves;
    py_dci_list *py_dci_inst_list;
    
    if (!PyArg_ParseTuple(args, "O", &py_dci_inst_wrapper)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");

    npy_intp py_num_leaves_shape[1];
    py_num_leaves_shape[0] = py_dci_inst_list->num_inst;
    py_num_leaves = (PyArrayObject *)PyArray_SimpleNew(1, py_num_leaves_shape, NPY_INT);
    int* num_leaves = (int *)PyArray_DATA(py_num_leaves);

    for (int i = 0; i < py_dci_inst_list->num_inst; i++) {
        num_leaves[i] = (py_dci_inst_list->dci_inst_list[i].dci_inst).num_leaf_nodes;
    }
    
	return Py_BuildValue("N", py_num_leaves);
}

static PyObject *py_dci_get_proj_vec(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    py_dci *py_dci_inst;
    py_dci_list *py_dci_inst_list;
    PyArrayObject *py_proj_vec;
    npy_intp py_proj_vec_shape[2];
    
    if (!PyArg_ParseTuple(args, "O", &py_dci_inst_wrapper)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");
    py_dci_inst = &(py_dci_inst_list->dci_inst_list[0]);
    
    py_proj_vec_shape[0] = (py_dci_inst->dci_inst).num_comp_indices*(py_dci_inst->dci_inst).num_simp_indices;
    py_proj_vec_shape[1] = (py_dci_inst->dci_inst).dim;
    // Assuming row-major layout, matrix is of size (num_comp_indices*num_simp_indices) x dim
    py_proj_vec = (PyArrayObject *)PyArray_SimpleNewFromData(2, py_proj_vec_shape, NPY_FLOAT, (py_dci_inst->dci_inst).proj_vec);
    // py_proj_vec owns a reference to py_dci_inst_wrapper
    py_proj_vec->base = py_dci_inst_wrapper;
    Py_INCREF(py_dci_inst_wrapper);
    
    return (PyObject *)py_proj_vec;
}

static PyObject *py_dci_get_token2node(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    PyArrayObject *py_token2node_index, *py_token2node_offset;
    py_dci *py_dci_inst;
    py_dci_list *py_dci_inst_list;
    
    if (!PyArg_ParseTuple(args, "O", &py_dci_inst_wrapper)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");
    py_dci_inst = &(py_dci_inst_list->dci_inst_list[0]);

    npy_intp py_token2node_shape[2];
    py_token2node_shape[0] = py_dci_inst_list->num_inst;
    py_token2node_shape[1] = (py_dci_inst_list->dci_inst_list[0].dci_inst).num_points;
    py_token2node_index = (PyArrayObject *)PyArray_SimpleNew(2, py_token2node_shape, NPY_INT);
    py_token2node_offset = (PyArrayObject *)PyArray_SimpleNew(2, py_token2node_shape, NPY_INT);
    int* token2node_index = (int *)PyArray_DATA(py_token2node_index);
    int* token2node_offset = (int *)PyArray_DATA(py_token2node_offset);
    int max_num_points = (py_dci_inst_list->dci_inst_list[0].dci_inst).num_points;
    
    for (int i = 0; i < py_dci_inst_list->num_inst; i++) {
        for (int j = 0; j < max_num_points; j++) {
            token2node_index[i * max_num_points + j] = (py_dci_inst_list->dci_inst_list[i].dci_inst).token2nodeIndex[j];
            token2node_offset[i * max_num_points + j] = (py_dci_inst_list->dci_inst_list[i].dci_inst).token2nodeOffset[j];
        }
    }
    
	return Py_BuildValue("NN", py_token2node_index, py_token2node_offset);
}

static PyObject *py_dci_get_valid_entries(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    PyArrayObject *py_valid_entries, *py_ids;
    int *leaf_ids;
    int max_leaves;
    
    if (!PyArg_ParseTuple(args, "OO!", &py_dci_inst_wrapper, &PyArray_Type, &py_ids)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;

    if (PyArray_DIM(py_ids, 0) == 0) {
        leaf_ids = NULL;
        return NULL;
    }
    else {
        leaf_ids = (int *)PyArray_DATA(py_ids);
        max_leaves = PyArray_DIM(py_ids, 1);
    }
    
    py_dci_list *py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");

    npy_intp py_valid_entries_shape[2];
    py_valid_entries_shape[0] = py_dci_inst_list->num_inst;
    py_valid_entries_shape[1] = max_leaves;
    py_valid_entries = (PyArrayObject *)PyArray_SimpleNew(2, py_valid_entries_shape, NPY_INT);
    int* valid_entries = (int *)PyArray_DATA(py_valid_entries);
    
#pragma omp parallel for
    for (int i = 0; i < py_dci_inst_list->num_inst; i++) {
        btree_p_leaf_node **llist = (py_dci_inst_list->dci_inst_list[i].dci_inst).leaf_list;
        for (int j = 0; j < max_leaves; j++) {
            int lid = leaf_ids[i * max_leaves + j];
            if (lid >= 0)
                valid_entries[i * max_leaves + j] = llist[lid]->num_slots_used;
            else
                valid_entries[i * max_leaves + j] = -1;
        }
    }
    
	return Py_BuildValue("N", py_valid_entries);
}

static PyObject *py_dci_print_dci(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    py_dci *py_dci_inst;
    py_dci_list *py_dci_inst_list;
    int idx;
    
    if (!PyArg_ParseTuple(args, "Oi", &py_dci_inst_wrapper, &idx)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");
    py_dci_inst = &(py_dci_inst_list->dci_inst_list[idx]);
    print_dci(&(py_dci_inst->dci_inst));
    
	Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *py_dci_print_cell_num(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    py_dci *py_dci_inst;
    py_dci_list *py_dci_inst_list;
    int idx;
    
    if (!PyArg_ParseTuple(args, "Oi", &py_dci_inst_wrapper, &idx)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");
    py_dci_inst = &(py_dci_inst_list->dci_inst_list[idx]);
    print_cell_num(&(py_dci_inst->dci_inst));
    
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *py_dci_get_parent_stat(PyObject *self, PyObject *args) {

    PyObject *py_dci_inst_wrapper;
    py_dci_list *py_dci_inst_list;
    int X, interval;

    if (!PyArg_ParseTuple(args, "Oii", &py_dci_inst_wrapper, &X, &interval)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;

    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");
    int num_inst = py_dci_inst_list->num_inst;

    // Get the number of points on the finest level (assuming all instances have the same number)
    int num_points = py_dci_inst_list->dci_inst_list[0].dci_inst.num_points_on_level[0];

    // Create Python arrays to return
    // - parent_in_anchor_set: whether point's parent is in the 2X anchor set
    // - distance_ratios: ratio of (point to anchor's parent) / (anchor to its parent)
    // - parent_consistency: whether point's parent matches closest anchor's parent
    // - min_dist_to_anchor_parents: minimum distance to any parent in the anchor set
    // - max_dist_closest_parent_to_children: max distance from closest anchor parent to its children
    npy_intp shape[2] = {num_inst, num_points};
    PyArrayObject *py_parent_in_anchor_set = (PyArrayObject *)PyArray_SimpleNew(2, shape, NPY_BOOL);
    PyArrayObject *py_distance_ratios = (PyArrayObject *)PyArray_SimpleNew(2, shape, NPY_FLOAT);
    PyArrayObject *py_parent_consistency = (PyArrayObject *)PyArray_SimpleNew(2, shape, NPY_BOOL);
    PyArrayObject *py_min_dist_to_anchor_parents = (PyArrayObject *)PyArray_SimpleNew(2, shape, NPY_FLOAT);
    PyArrayObject *py_max_dist_closest_parent_to_children = (PyArrayObject *)PyArray_SimpleNew(2, shape, NPY_FLOAT);

    bool* py_parent_data = (bool*)PyArray_DATA(py_parent_in_anchor_set);
    float* py_ratio_data = (float*)PyArray_DATA(py_distance_ratios);
    bool* py_consistency_data = (bool*)PyArray_DATA(py_parent_consistency);
    float* py_min_dist_data = (float*)PyArray_DATA(py_min_dist_to_anchor_parents);
    float* py_max_dist_data = (float*)PyArray_DATA(py_max_dist_closest_parent_to_children);

    // Process all instances in parallel
#pragma omp parallel for
    for (int idx = 0; idx < num_inst; idx++) {
        py_dci *py_dci_inst = &(py_dci_inst_list->dci_inst_list[idx]);

        // Allocate arrays for results for this instance
        bool* parent_in_anchor_set = (bool*)malloc(sizeof(bool) * num_points);
        float* distance_ratios = (float*)malloc(sizeof(float) * num_points);
        bool* parent_consistency = (bool*)malloc(sizeof(bool) * num_points);
        float* min_dist_to_anchor_parents = (float*)malloc(sizeof(float) * num_points);
        float* max_dist_closest_parent_to_children = (float*)malloc(sizeof(float) * num_points);

        // Call the C function
        get_parent_stat(&(py_dci_inst->dci_inst), X, interval, parent_in_anchor_set, distance_ratios, parent_consistency, min_dist_to_anchor_parents, max_dist_closest_parent_to_children);

        // Copy data to Python arrays
        for (int i = 0; i < num_points; i++) {
            py_parent_data[idx * num_points + i] = parent_in_anchor_set[i];
            py_ratio_data[idx * num_points + i] = distance_ratios[i];
            py_consistency_data[idx * num_points + i] = parent_consistency[i];
            py_min_dist_data[idx * num_points + i] = min_dist_to_anchor_parents[i];
            py_max_dist_data[idx * num_points + i] = max_dist_closest_parent_to_children[i];
        }

        // Free the C arrays
        free(parent_in_anchor_set);
        free(distance_ratios);
        free(parent_consistency);
        free(min_dist_to_anchor_parents);
        free(max_dist_closest_parent_to_children);
    }

    return Py_BuildValue("NNNNN", py_parent_in_anchor_set, py_distance_ratios, py_parent_consistency, py_min_dist_to_anchor_parents, py_max_dist_closest_parent_to_children);
}

static PyObject *py_dci_print_num_points_on_level(PyObject *self, PyObject *args) {

    PyObject *py_dci_inst_wrapper;
    py_dci *py_dci_inst;
    py_dci_list *py_dci_inst_list;
    int idx;

    if (!PyArg_ParseTuple(args, "Oi", &py_dci_inst_wrapper, &idx)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;

    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");
    py_dci_inst = &(py_dci_inst_list->dci_inst_list[idx]);
    print_num_points_on_level(&(py_dci_inst->dci_inst));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *py_dci_check_dci(PyObject *self, PyObject *args) {
    
    PyObject *py_dci_inst_wrapper;
    py_dci *py_dci_inst;
    py_dci_list *py_dci_inst_list;
    
    if (!PyArg_ParseTuple(args, "O", &py_dci_inst_wrapper)) return NULL;
    if (!py_dci_inst_wrapper) return NULL;
    
    py_dci_inst_list = (py_dci_list *)PyCapsule_GetPointer(py_dci_inst_wrapper, "py_dci_inst_list");

    for (int i = 0; i < py_dci_inst_list->num_inst; i++) {
        py_dci_inst = &(py_dci_inst_list->dci_inst_list[i]);
        check_dci(&(py_dci_inst->dci_inst));
    }
    
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *copy_to_buffer(PyObject *self, PyObject *args) {
    PyArrayObject *py_src_ptr_array;
    int list_size, update_num;
    unsigned long long ptr_dest;
    int offset_s, offset_t, dim, page_size, dtype;  // dtype: 0 = float32, 1 = float16

    if (!PyArg_ParseTuple(args, "O!Kiiiiiii", &PyArray_Type, &py_src_ptr_array, &ptr_dest, &list_size, &update_num,
                          &offset_s, &offset_t, &dim, &page_size, &dtype)) return NULL;
    
    // Get direct pointers to the NumPy array data
    unsigned long long* src_ptr_data = (unsigned long long*)PyArray_DATA(py_src_ptr_array);

    #pragma omp parallel for
    for (int i = 0; i < list_size; ++i) {
        uintptr_t src_addr = (uintptr_t)src_ptr_data[i];

        if (dtype == 0) {  // float32
            float* p_dest = (float*)(uintptr_t)ptr_dest;
            float* current_src_page = (float*)src_addr;
            float* current_dest_page = p_dest + i * page_size;

            size_t copy_len = update_num * dim;
            memcpy(current_dest_page, current_src_page, copy_len * sizeof(float));
            memcpy(current_dest_page + offset_t, current_src_page + offset_s, copy_len * sizeof(float));

        } else if (dtype == 1) {  // float16
            uint16_t* p_dest = (uint16_t*)(uintptr_t)ptr_dest;
            uint16_t* current_src_page = (uint16_t*)src_addr;
            uint16_t* current_dest_page = p_dest + i * page_size;

            size_t copy_len = update_num * dim;
            memcpy(current_dest_page, current_src_page, copy_len * sizeof(uint16_t));
            memcpy(current_dest_page + offset_t, current_src_page + offset_s, copy_len * sizeof(uint16_t));
        }
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *reuse_copy_node(PyObject *self, PyObject *args) {
    PyArrayObject *py_p_index;
    PyArrayObject *py_p_offset;
    PyArrayObject *py_keys;
    PyArrayObject *py_values;
    PyObject *py_new_address;
    int kv_offset;

    if (!PyArg_ParseTuple(args, "O!O!O!O!Oi", &PyArray_Type, &py_p_index, &PyArray_Type, &py_p_offset,
                          &PyArray_Type, &py_keys, &PyArray_Type, &py_values, &py_new_address, &kv_offset)) {
        return NULL;
    }

    // Get direct pointers to the NumPy array data
    int32_t* p_index_data = (int32_t*)PyArray_DATA(py_p_index);
    int32_t* p_offset_data = (int32_t*)PyArray_DATA(py_p_offset);

    // Get array shapes
    npy_intp *keys_shape = PyArray_SHAPE(py_keys);
    npy_intp *values_shape = PyArray_SHAPE(py_values);

    int num_inst = PyList_Size(py_new_address);
    int key_dim = keys_shape[1];
    int value_dim = values_shape[1];
    int num_points = keys_shape[0] / num_inst;

    int small_page_size = kv_offset / num_inst;

    // // Get data type of keys and values arrays
    // int keys_dtype = PyArray_TYPE(py_keys);
    // int values_dtype = PyArray_TYPE(py_values);

    float* keys_data = (float*)PyArray_DATA(py_keys);
    float* values_data = (float*)PyArray_DATA(py_values);

    #pragma omp parallel for
    for (int inst = 0; inst < num_inst; ++inst) {
        PyObject *addr_obj = PyList_GetItem(py_new_address, inst);
        unsigned long long current_address = PyLong_AsUnsignedLongLong(addr_obj);
        float* dest_keys = (float*)(uintptr_t)current_address;

        int32_t *tmp_p_index = p_index_data + inst * num_points;
        int32_t *tmp_p_offset = p_offset_data + inst * num_points;
        float *tmp_keys = keys_data + inst * num_points * key_dim;
        float *tmp_values = values_data + inst * num_points * key_dim;

        int total_threads = omp_get_max_threads();
        int inner_threads = total_threads / num_inst;
        if (inner_threads < 1) inner_threads = 1;
        #pragma omp parallel for num_threads(inner_threads)
        for (int i = 0; i < num_points; ++i) {
            int page_idx = tmp_p_index[i];
            int offset = tmp_p_offset[i];
            
            float* current_key = tmp_keys + i * key_dim;
            float* current_value = tmp_values + i * value_dim;
            float* dest_key_page = dest_keys + page_idx * 2 * kv_offset + offset * key_dim;
            float* dest_value_page = dest_key_page + kv_offset;

            memcpy(dest_key_page, current_key, key_dim * sizeof(float));
            memcpy(dest_value_page, current_value, value_dim * sizeof(float));
        }
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *reuse_update_node(PyObject *self, PyObject *args) {
    PyArrayObject *py_old_p_offset;
    PyArrayObject *py_old_p_index;
    PyArrayObject *py_new_p_index;
    PyArrayObject *py_new_p_offset;
    PyArrayObject *py_keys;
    PyArrayObject *py_values;
    PyArrayObject *py_new_address;
    PyArrayObject *py_changed_flags;
    PyArrayObject *py_num_leaves;
    int kv_offset;

    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!iO!O!",
                          &PyArray_Type, &py_old_p_index,
                          &PyArray_Type, &py_old_p_offset,
                          &PyArray_Type, &py_new_p_index,
                          &PyArray_Type, &py_new_p_offset,
                          &PyArray_Type, &py_keys,
                          &PyArray_Type, &py_values,
                          &PyArray_Type, &py_new_address,
                          &kv_offset,
                          &PyArray_Type, &py_changed_flags,
                          &PyArray_Type, &py_num_leaves
                          )) {
        return NULL;
    }

    // Get direct pointers to the NumPy array data
    int32_t* old_p_offset_data = (int32_t*)PyArray_DATA(py_old_p_offset);
    int32_t* old_p_index_data = (int32_t*)PyArray_DATA(py_old_p_index);
    int32_t* new_p_index_data = (int32_t*)PyArray_DATA(py_new_p_index);
    int32_t* new_p_offset_data = (int32_t*)PyArray_DATA(py_new_p_offset);
    int8_t* changed_flags_data = (int8_t*)PyArray_DATA(py_changed_flags);
    unsigned long long* new_address_data = (unsigned long long*)PyArray_DATA(py_new_address);
    int* num_leaves_data = (int*)PyArray_DATA(py_num_leaves);

    // Get array shapes
    npy_intp *keys_shape = PyArray_SHAPE(py_keys);
    npy_intp *values_shape = PyArray_SHAPE(py_values);
    npy_intp *old_p_offset_shape = PyArray_SHAPE(py_old_p_offset);
    npy_intp *new_p_offset_shape = PyArray_SHAPE(py_new_p_offset);
    npy_intp *changed_flags_shape = PyArray_SHAPE(py_changed_flags);
    npy_intp *new_address_shape = PyArray_SHAPE(py_new_address);

    int num_inst = new_address_shape[0];
    int num_pages_l = new_address_shape[1];
    int num_pages_s = changed_flags_shape[1];
    int key_dim = keys_shape[1];
    int value_dim = values_shape[1];
    int num_new_keys = keys_shape[0] / num_inst;
    int num_old_points = old_p_offset_shape[1];
    int num_new_points = new_p_offset_shape[1];

    int small_page_size = kv_offset / num_inst;
    int page_num_tokens = small_page_size / key_dim;

    float* keys_data = (float*)PyArray_DATA(py_keys);
    float* values_data = (float*)PyArray_DATA(py_values);

    assert(key_dim == value_dim);

    #pragma omp parallel for
    for (int inst = 0; inst < num_inst; ++inst) {
        // Skip if this instance hasn't changed
        unsigned long long* inst_addresses = new_address_data + inst * num_pages_l;
        int8_t* tmp_changed_flags = changed_flags_data + inst * num_pages_s;
        int32_t *tmp_old_p_offset = old_p_offset_data + inst * num_old_points;
        int32_t *tmp_old_p_index = old_p_index_data + inst * num_old_points;
        int32_t *tmp_new_p_index = new_p_index_data + inst * num_new_points;
        int32_t *tmp_new_p_offset = new_p_offset_data + inst * num_new_points;
        int tmp_num_leaves = num_leaves_data[inst];

        float *tmp_keys = keys_data + inst * num_new_keys * key_dim;
        float *tmp_values = values_data + inst * num_new_keys * value_dim;

        int total_threads = omp_get_max_threads();
        int inner_threads = total_threads / num_inst;
        if (inner_threads < 1) inner_threads = 1;

        int count = 0;
        int map_list[tmp_num_leaves];
        int reverse_map_list[tmp_num_leaves];
        for (int i = 0; i < tmp_num_leaves; ++i) {
            if (tmp_changed_flags[i]) {
                reverse_map_list[i] = count;
                map_list[count++] = i;
            }
        }

        bool *track_mask = (bool*)calloc(count * page_num_tokens, sizeof(bool));
        float* tmp_buffer_keys = (float*)malloc(count * small_page_size * sizeof(float));
        float* tmp_buffer_values = (float*)malloc(count * small_page_size * sizeof(float));

        #pragma omp parallel for num_threads(inner_threads)
        for (int i = 0; i < num_new_points; ++i) {
            int new_page_idx = tmp_new_p_index[i];
            int new_offset = tmp_new_p_offset[i];

            if (i >= num_old_points) {
                float* current_key = tmp_keys + (i - num_old_points) * key_dim;
                float* current_value = tmp_values + (i - num_old_points) * value_dim;

                int tmp_offset = reverse_map_list[new_page_idx] * small_page_size + new_offset * key_dim;

                float* dest_key_page = tmp_buffer_keys + tmp_offset;
                float* dest_value_page = tmp_buffer_values + tmp_offset;

                memcpy(dest_key_page, current_key, key_dim * sizeof(float));
                memcpy(dest_value_page, current_value, value_dim * sizeof(float));
                track_mask[reverse_map_list[new_page_idx] * page_num_tokens + new_offset] = true;
                continue;
            }
            
            int old_page_idx = tmp_old_p_index[i];
            int old_offset = tmp_old_p_offset[i];
            
            if (old_page_idx != new_page_idx || old_offset != new_offset) {

                float* orginal_address = (float*)(uintptr_t)inst_addresses[old_page_idx];

                float* current_key = orginal_address + old_offset * key_dim;
                float* current_value = current_key + kv_offset;

                int tmp_offset = reverse_map_list[new_page_idx] * small_page_size + new_offset * key_dim;

                float* dest_key_page = tmp_buffer_keys + tmp_offset;
                float* dest_value_page = tmp_buffer_values + tmp_offset;

                memcpy(dest_key_page, current_key, key_dim * sizeof(float));
                memcpy(dest_value_page, current_value, value_dim * sizeof(float));
                track_mask[reverse_map_list[new_page_idx] * page_num_tokens + new_offset] = true;
            }
        }

        for (int i = 0; i < count; ++i) {
            float* dest_address = (float*)(uintptr_t)inst_addresses[map_list[i]];
            for (int j = 0; j < page_num_tokens; ++j) {
                if (track_mask[i * page_num_tokens + j]) {
                    memcpy(dest_address + j * key_dim, tmp_buffer_keys + i * small_page_size + j * key_dim, key_dim * sizeof(float));
                    memcpy(dest_address + j * value_dim + kv_offset, tmp_buffer_values + i * small_page_size + j * value_dim, value_dim * sizeof(float)); 
                }
            }
        }

        free(tmp_buffer_keys);
        free(tmp_buffer_values);
        free(track_mask);
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *diff_pages_by_head(PyObject *self, PyObject *args) {
    PyArrayObject *py_array_A;
    PyArrayObject *py_array_B;
    PyArrayObject *py_array_mask;
    PyArrayObject *py_array_pid;

    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &py_array_A, &PyArray_Type, &py_array_B, &PyArray_Type, &py_array_mask, &PyArray_Type, &py_array_pid))
        return NULL;

    npy_intp *shape_A = PyArray_SHAPE(py_array_A);
    int num_head = shape_A[0];
    int page_num = shape_A[1];

    npy_intp *shape_pid = PyArray_SHAPE(py_array_pid);
    assert (num_head == shape_pid[0]);
    int total_page_num = shape_pid[1];

    // Allocate output array
    npy_intp dims[2] = {num_head, page_num};
    int32_t *A_data = (int32_t *)PyArray_DATA(py_array_A);
    int32_t *B_data = (int32_t *)PyArray_DATA(py_array_B);
    int8_t *ccc_mask = (int8_t *)PyArray_DATA(py_array_mask);
    int32_t *pid_data = (int32_t *)PyArray_DATA(py_array_pid);

    PyArrayObject *py_result_A = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_INT32, 0);
    PyArrayObject *py_result_B = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_INT32, 0);
    PyArrayObject *py_result_O = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_INT32, 0);
    int32_t *out_data_A = (int32_t *)PyArray_DATA(py_result_A);
    int32_t *out_data_B = (int32_t *)PyArray_DATA(py_result_B);
    int32_t *out_data_O = (int32_t *)PyArray_DATA(py_result_O);

    #pragma omp parallel for
    for (int i = 0; i < num_head; ++i) {
        int32_t *tmp_A = out_data_A + i * page_num;
        int32_t *tmp_B = out_data_B + i * page_num;
        int32_t *tmp_O = out_data_O + i * page_num;
        int32_t *tmp_A_data = A_data + i * page_num;
        int32_t *tmp_B_data = B_data + i * page_num;
        int8_t *tmp_ccc_mask = ccc_mask + i * page_num;
        int32_t *tmp_pid_data = pid_data + i * total_page_num;

        int out_idx = 0;
        for (int j = 0; j < page_num; ++j) {
            int32_t val_a = tmp_A_data[j];
            int found = 0;
            for (int k = 0; k < page_num; ++k) {
                if (val_a == tmp_B_data[k]) {
                    if (tmp_ccc_mask[j])
                        break;
                    found = 1;
                    break;
                }
            }
            if (!found) {
                tmp_A[out_idx++] = val_a;
            }
        }

        int out_idx_ = 0;
        for (int j = 0; j < page_num; ++j) {
            int32_t val_b = tmp_B_data[j];
            int found = 0;
            for (int k = 0; k < page_num; ++k) {
                if (val_b == tmp_A_data[k]) {
                    if (tmp_ccc_mask[k])
                        break;
                    tmp_O[j] = val_b;
                    found = 1;
                    break;
                }
            }
            if (!found) {
                tmp_O[j] = tmp_A[out_idx_];
                tmp_B[out_idx_++] = tmp_pid_data[val_b];
                tmp_pid_data[val_b] = -1;
            }
        }
        for (int j = 0; j < out_idx_; ++j) {
            tmp_pid_data[tmp_A[j]] = tmp_B[j];
        }
        assert(out_idx_ == out_idx);
        // Pad the rest of the row with -1
        for (; out_idx < page_num; ++out_idx) {
            tmp_A[out_idx] = -1;
            tmp_B[out_idx] = -1;
        }
    }

    return Py_BuildValue("NNN", py_result_A, py_result_B, py_result_O);
}


// Methods table - maps names in Python to C functions  
static PyMethodDef py_dci_module_methods[] = {
    {"_dci_new", py_dci_new, METH_VARARGS, "Create new DCI instance."},
    {"_dci_address_update", py_dci_address_update, METH_VARARGS, "Update the address of the points in DCI instance."},
    {"_dci_quick_sort", py_dci_quick_sort, METH_VARARGS, "Quick sort the provided pages using projection."},
    {"_dci_cached_tree", py_dci_cached_tree, METH_VARARGS, "Cached tree construction."},
    {"_dci_add", py_dci_add, METH_VARARGS, "Add data."},
    {"_dci_query", py_dci_query, METH_VARARGS, "Search for nearest neighbours."},
    {"_dci_add_query_at_end", py_dci_add_query_at_end, METH_VARARGS, "Add and Search for nearest neighbours at the end."},
    {"_dci_add_query_attention", py_dci_add_query_attention, METH_VARARGS, "Add and Search for nearest neighbours and then compute attention score."},
    {"_dci_add_query", py_dci_add_query, METH_VARARGS, "Add and Search for nearest neighbours."},
    {"_dci_clear", py_dci_clear, METH_VARARGS, "Delete all data."},
    {"_dci_reset", py_dci_reset, METH_VARARGS, "Regenerate projection directions."},
    {"_dci_reset_proj", py_dci_reset_proj, METH_VARARGS, "Reset projection directions."},
    {"_dci_get_num_points", py_dci_get_num_points, METH_VARARGS, "Get the number of points indexed by DCI instance. "},
    {"_dci_get_num_levels", py_dci_get_num_levels, METH_VARARGS, "Get the number of levels in DCI instance. "},
    {"_dci_get_num_leaves", py_dci_get_num_leaves, METH_VARARGS, "Get the number of nodes in DCI instance. "},
    {"_dci_get_proj_vec", py_dci_get_proj_vec, METH_VARARGS, "Get the projection vectors used by DCI instance. "},
    {"_dci_get_token2node", py_dci_get_token2node, METH_VARARGS, "Get the token2node mapping used by DCI instance. "},
    {"_dci_get_valid_entries", py_dci_get_valid_entries, METH_VARARGS, "Get the number of valid entries in each leaf node. "},
    {"_dci_delete", py_dci_delete, METH_VARARGS, "Delete data."},
    {"_dci_print_dci", py_dci_print_dci, METH_VARARGS, "Print the structure of the whole dci."},
    {"_dci_print_cell_num", py_dci_print_cell_num, METH_VARARGS, "Print the number of children for each cell."},
    {"_dci_print_num_points_on_level", py_dci_print_num_points_on_level, METH_VARARGS, "Print the number of points on each level."},
    {"_dci_get_parent_stat", py_dci_get_parent_stat, METH_VARARGS, "Get parent statistics for the finest level. Returns (parent_in_anchor_set, distance_ratios, parent_consistency, min_dist_to_anchor_parents, max_dist_closest_parent_to_children)."},
    {"_dci_check_dci", py_dci_check_dci, METH_VARARGS, "Check the correctness of the DCI instance."},
    {"_dci_copy_to_buffer", copy_to_buffer, METH_VARARGS, "Copy data to buffer."},
    {"_dci_reuse_copy_node", reuse_copy_node, METH_VARARGS, "Reuse copy to buffer."},
    {"_dci_reuse_update_node", reuse_update_node, METH_VARARGS, "Reuse update to buffer."},
    {"_dci_diff_pages_by_head", diff_pages_by_head, METH_VARARGS, "Diff pages by head."},
    {NULL, NULL, 0, NULL}
};

#ifdef PY3K

static int py_dci_module_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int py_dci_module_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef py_dci_module_def = {
        PyModuleDef_HEAD_INIT,
        "_dci",
        NULL,
        sizeof(struct module_state),
        py_dci_module_methods,
        NULL,
        py_dci_module_traverse,
        py_dci_module_clear,
        NULL
};

// Module name is "_dci"
PyMODINIT_FUNC PyInit__dci(void) {
    PyObject *module = PyModule_Create(&py_dci_module_def);
    import_array();     // Import Numpy
    return module;
}

#else

// Module name is "_dci"
PyMODINIT_FUNC init_dci(void) {
    (void) Py_InitModule("__dci", py_dci_module_methods);
    import_array();     // Import Numpy
}

#endif
