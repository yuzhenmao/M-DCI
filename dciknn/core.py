'''
Code for Fast k-Nearest Neighbour Search via Prioritized DCI

This code implements the method described in the Prioritized DCI paper, 
which can be found at https://arxiv.org/abs/1703.00440

This file is a part of the Dynamic Continuous Indexing reference 
implementation.


This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

Copyright (C) 2017    Ke Li
'''

import numpy as np
import pdb
from ._dci import _dci_new, _dci_get_proj_vec, _dci_get_num_points, _dci_get_num_levels, _dci_add,  _dci_delete,_dci_query, _dci_clear, _dci_reset, _dci_print_dci, _dci_print_cell_num, _dci_add_query, _dci_add_query_attention, _dci_check_dci, _dci_get_token2node, _dci_get_num_leaves, _dci_print_num_points_on_level, _dci_address_update, _dci_add_query_at_end, _dci_quick_sort, _dci_cached_tree, _dci_get_valid_entries, _dci_reset_proj, _dci_copy_to_buffer, _dci_diff_pages_by_head, _dci_reuse_copy_node, _dci_reuse_update_node, _dci_get_parent_stat

class ProtectedArray(object):
    # when_readable is a function that returns True when reading is allowed
    def __init__(self, base_array, when_readable = None, read_error = None, when_writable = None, write_error = None):
        self._base = base_array
        self._when_readable = when_readable
        self._read_error = read_error
        self._when_writable = when_writable
        self._write_error = write_error
        
    def __getitem__(self, indices):
        if self._when_readable is not None and not self._when_readable(indices):
            if self._read_error is None:
               raise RuntimeError("array is not currently readable")
            else:
                raise self._read_error(indices)
        return self._base.__getitem__(indices)
        
    def __setitem__(self, indices, value):
        if self._when_writable is not None and not self._when_writable(indices):
            if self._write_error is None:
               raise RuntimeError("array is not currently writable")
            else:
                raise self._write_error(indices)
        self._base.__setitem__(indices, value)
        
    def __getattr__(self, attr):
        return getattr(self._base, attr)
    
    def __repr__(self):
        return repr(self._base)

class DCI(object):
    
    def __init__(self, dim, num_comp_indices = 2, num_simp_indices = 7, promotion_prob = 0.1, promotion_prob_subseq = 0, num_points = 0, transform = False, num_inst = 1, parallel_level = 0, debug = False, init = True, proj_vec = None):
        
        self._dim = dim
        self._num_comp_indices = num_comp_indices
        self._num_simp_indices = num_simp_indices
        self.promotion_prob = promotion_prob
        self.promotion_prob_subseq = promotion_prob_subseq
        self.num_inst = num_inst
        assert num_points > 0
        assert num_inst > 0
        if proj_vec is None:
            proj_vec = np.array([]).astype(np.float32)
        if init:
            self._dci_inst = _dci_new(dim, num_comp_indices, num_simp_indices, promotion_prob, promotion_prob_subseq, num_points, transform, num_inst, parallel_level, debug, proj_vec)
            self._proj_vec = _dci_get_proj_vec(self._dci_inst)
        self._array = None
        self._orig_indices = None    # Used only when the data is originally discontiguous - translates the indices from the contiguous subset of data to the original indices
        
    @property
    def dim(self):
        return self._dim
        
    @property
    def num_comp_indices(self):
        return self._num_comp_indices
        
    @property
    def num_simp_indices(self):
        return self._num_simp_indices
    
    @property
    def num_points(self):
        return _dci_get_num_points(self._dci_inst)
    
    @property
    def num_levels(self):
        return _dci_get_num_levels(self._dci_inst)
    
    @property
    def num_leaves(self):
        return _dci_get_num_leaves(self._dci_inst)
    
    @property
    def token2node(self):
        return _dci_get_token2node(self._dci_inst)
    
    def print_dci(self, idx = 0):
        _dci_print_dci(self._dci_inst, idx)
    
    def print_cell_num(self, idx = 0):
        _dci_print_cell_num(self._dci_inst, idx)
    
    def print_num_points_on_level(self, idx = 0):
        _dci_print_num_points_on_level(self._dci_inst, idx)

    def get_parent_stat(self, X, interval):
        """
        Get parent statistics for the finest level across all DCI instances.

        For each non-anchor point, this function analyzes the relationship between the point's
        parent and the parents of nearby anchor points (X points on the left, X on the right).

        Args:
            X: Number of anchor points to check on each side
            interval: Interval between anchor points

        Returns:
            Tuple of five numpy arrays (num_inst x num_points):
            - parent_in_anchor_set: Boolean array indicating if the point's parent matches
              ANY of the 2X surrounding anchor points' parents
            - distance_ratios: Float array with ratio = dist(point → closest_anchor's_parent)
              / dist(closest_anchor → its_parent). This measures how far the point is from
              the anchor's parent relative to the anchor's own distance to its parent.
              For anchor points, this is 0.0.
            - parent_consistency: Boolean array indicating if the point's parent matches
              the parent of its closest anchor point (among the 2X anchors)
            - min_dist_to_anchor_parents: Float array with the minimum distance from the point
              to ANY parent in the 2X anchor set. For anchor points, this is 0.0.
            - max_dist_closest_parent_to_children: Float array with the maximum distance from
              the closest anchor parent to any of its children. For anchor points, this is 0.0.
        """
        return _dci_get_parent_stat(self._dci_inst, X, interval)

    @property
    def check_dci(self):
        _dci_check_dci(self._dci_inst)

    @staticmethod
    def copy_to_buffer(src_array, ptr_dest, list_size, update_num, offset_s, offset_t, dim, page_size, dtype=1):
        assert isinstance(src_array, np.ndarray)        
        _dci_copy_to_buffer(src_array, ptr_dest, list_size, update_num, offset_s, offset_t, dim, page_size, dtype)

    @staticmethod
    def reuse_copy_node(p_index, p_offset, keys, values, new_address, kv_offset):  
        _dci_reuse_copy_node(p_index, p_offset, keys, values, new_address, kv_offset)

    @staticmethod
    def reuse_update_node(old_index, old_offset, new_index, new_offset, keys, values, new_address, kv_offset, ccc, num_leaves):
        _dci_reuse_update_node(old_index, old_offset, new_index, new_offset, keys, values, new_address, kv_offset, ccc, num_leaves)
    
    @staticmethod
    def diff_pages_by_head(array_A, array_B, array_mask, array_pid):
        assert array_mask.dtype == bool
        assert array_A.shape == array_B.shape == array_mask.shape
        return _dci_diff_pages_by_head(array_A, array_B, array_mask, array_pid)
    
    @property
    def proj_vec(self):
        return ProtectedArray(self._proj_vec, when_writable = lambda _: self.num_points == 0, write_error = lambda _: AttributeError("can only set projection vectors when the database is empty"))
    
    @proj_vec.setter
    def proj_vec(self, new_proj_vec):
        if self.num_points != 0:
            raise AttributeError("can only set projection vectors when the database is empty")
        # Disallow broadcasting when assigning to proj_vec directly
        new_proj_vec = np.asarray(new_proj_vec)
        if new_proj_vec.shape != self._proj_vec.shape:
            raise ValueError("mismatch between the expected shape of projection vectors (%s) and the supplied shape (%s)" % (repr(self._proj_vec.shape),repr(new_proj_vec.shape)))
        self._proj_vec[...] = new_proj_vec
            
    def _ensure_positive_integer(self, x):
        if not isinstance(x, int):
            raise TypeError("number must be an integer")
        elif x <= 0:
             raise ValueError("number must be positive")
    
    def _check_array(self, arr):
        if arr.shape[1] != self.dim:
            raise ValueError("mismatch between array dimension (%d) and the declared dimension of this DCI instance (%d)" % (arr.shape[1],self.dim))
        if arr.dtype != np.float32:
            raise TypeError("array must consist of float-precision floats")
        if not arr.flags.c_contiguous:
            raise ValueError("the memory layout of array must be in row-major (C-order)")
        
    def _check_attention_mask(self, arr):
        if arr[0] <= 0 or (arr[0] > 1 and arr[0] < 10):
            raise ValueError("the first element of the attention_mask must be greater than 10 or equal to 1")
        if arr.dtype != np.intc:
            raise TypeError("attention mask must consist of ints")
        if not arr.flags.c_contiguous:
            raise ValueError("the memory layout of attention mask must be in row-major (C-order)")
        if not np.all(arr[:-1] <= arr[1:]):
            raise ValueError("attention mask must be sorted in non-descending order")
    
    def _check_and_fix_array(self, arr):
        if arr.shape[1] != self.dim:
            raise ValueError("mismatch between array dimension (%d) and the declared dimension of this DCI instance (%d)" % (arr.shape[1],self.dim))
        if arr.dtype == np.float32 and arr.flags.c_contiguous:
            return arr
        else:
            return np.array(arr, dtype=np.float32, copy=False, order='C')
    
    def _check_is_base_array(self, arr):
        # arr cannot be derived from some other array (except if it's just transposed, in which case the data pointer stays the same)
        if arr.base is not None:
            try:
                arr_addr = arr.data
                base = arr
                while base.base is not None:
                    base = base.base
            except AttributeError:
                arr_addr = None
            # if arr_addr is None or arr_addr != base.data:
            #     raise ValueError("array must not be derived from another array, except via the transpose operator. Pass in the original array and specify the indices or make a copy of the derived array.")
    
    def _check_data(self, data):
        self._check_array(data)
        self._check_is_base_array(data)
    
    def _check_and_fix_indices(self, data, indices):
        check_indices_within_bounds = False
        if indices is None:
            is_contiguous = True
            selected_idx = (0,data.shape[0])
        elif isinstance(indices, slice):
            step = indices.step
            start = indices.start
            stop = indices.stop
            if start is None:
                start = 0
            if step is None:
                step = 1
            if start < 0:
                start = data.shape[0] + start
            if stop < 0:
                stop = data.shape[0] + stop
            start = max(start, 0)
            stop = min(stop, data.shape[0])
            if step == 1:
                is_contiguous = True
                selected_idx = (start,stop)
            else:
                is_contiguous = False
                selected_idx = np.arange(start,stop,step,dtype=np.intc)
        elif isinstance(indices, int):
            if indices < 0:
                cur_idx = data.shape[0] + indices
            else:
                cur_idx = indices
            if cur_idx < 0 or cur_idx >= data.shape[0]:
                raise IndexError("index out of bounds")
            is_contiguous = True
            selected_idx = (cur_idx,cur_idx+1)
        elif isinstance(indices, np.ndarray):
            is_contiguous = False
            if indices.ndim == 1:
                if indices.dtype == np.intc:
                    selected_idx = indices
                    if np.any(selected_idx < 0):
                        selected_idx = np.copy(selected_idx)
                        selected_idx[selected_idx < 0] += data.shape[0]
                    check_indices_within_bounds = True
                elif indices.dtype == bool:
                    if indices.shape[0] == data.shape[0]:
                        selected_idx = np.nonzero(indices)[0].astype(np.intc)
                    else:
                        raise IndexError("mismatch between the number of boolean indices (%d) and array dimension (%d)" % (indices.shape[0],data.shape[0]))
                elif indices.dtype.kind in np.typecodes['AllInteger']:  # Check if dtype is an integer type; also returns true if dtype is bool                
                    selected_idx = indices.astype(np.intc)
                    selected_idx[selected_idx < 0] += data.shape[0]
                    check_indices_within_bounds = True
                else:
                    raise TypeError("indices must be integers or booleans")
            else:
                raise IndexError("indices must be in an one-dimensional array")
        elif isinstance(indices, list):
            is_contiguous = False
            if isinstance(indices[0], bool):
                selected_idx = np.nonzero(indices)[0].astype(np.intc)
            elif isinstance(indices[0], int):   # Also returns true if indices[0] is bool
                selected_idx = np.array(indices,dtype=np.intc)
                selected_idx[selected_idx < 0] += data.shape[0]
                check_indices_within_bounds = True
            elif isinstance(indices[0], list):
                raise IndexError("indices must be in an one-dimensional array")
            else:
                raise TypeError("indices must be integers or booleans")
        else:
            raise TypeError("indices must be None, a slice object, an integer, an array or list of integers")
            
        if check_indices_within_bounds:
            if np.any(selected_idx < 0) or np.any(selected_idx >= data.shape[0]):
                raise IndexError("some indices (e.g. %d) out of bounds" % (indices[(selected_idx < 0) | (selected_idx >= data.shape[0])][0]))
        
        return is_contiguous,selected_idx
    
    def address_update(self, indices, new_address, num_pages, offset):
        _dci_address_update(self._dci_inst, indices, new_address, num_pages, offset)

    # Indices can be None, a slice object, an integer, an array or list of integers - best to use np.intc type
    def add(self, data, data_ids, token_mask = None, indices = None, num_levels = 2, blind = False, num_to_visit = -1, num_to_retrieve = -1, prop_to_visit = -1.0, prop_to_retrieve = -1.0, field_of_view = 10):
        
        num_points = self.num_points
        
        #if num_points > 0:
        #   raise RuntimeError("DCI class does not support insertion of more than one array. Must combine all arrays into one array before inserting")

        if token_mask is None:
            token_mask = np.ones(data.shape[0], dtype=bool)
        
        # if num_levels >= 3:
        #     self._ensure_positive_integer(field_of_view)
        # else:
        #     field_of_view = -1
        self._ensure_positive_integer(field_of_view)
        
        # if num_to_visit > num_points:
        #     num_to_visit = num_points
        
        if prop_to_visit < 0.0:
            if num_to_visit < 0:
                prop_to_visit = 1.0
            else:
                prop_to_visit = -1.0 
        else:
            if prop_to_visit > 1.0:
                prop_to_visit = 1.0
        
        # if num_to_retrieve > num_points:
        #     num_to_retrieve = num_points
        
        if prop_to_retrieve < 0.0:
            if num_to_retrieve < 0:
                prop_to_retrieve = 0.002
            else:
                prop_to_retrieve = -1.0
        else:
            if prop_to_retrieve > 1.0:
                prop_to_retrieve = 1.0
        
        self._check_data(data)
        is_contiguous, _indices = self._check_and_fix_indices(data, indices)
        
        if is_contiguous:
            #pdb.set_trace()
            _dci_add(self._dci_inst, data, data_ids, token_mask, num_levels, blind, num_to_visit, num_to_retrieve, prop_to_visit, prop_to_retrieve, field_of_view)
        else:
            selected_data = data[_indices]
            _dci_add(self._dci_inst, selected_data, data_ids, token_mask, num_levels, blind, num_to_visit, num_to_retrieve, prop_to_visit, prop_to_retrieve, field_of_view)
            self._orig_indices = _indices
        
        self._array = data


        # Indices can be None, a slice object, an integer, an array or list of integers - best to use np.intc type
    def delete(self, data_ids, field_of_view = 10, blind = False, num_to_visit = -1, num_to_retrieve = -1, prop_to_visit = -1.0, prop_to_retrieve = -1.0, num_dci = 1):
        
        num_points = self.num_points
        
        # if num_levels >= 3:
        #     self._ensure_positive_integer(field_of_view)
        # else:
        #     field_of_view = -1
        
        # if num_to_visit > num_points:
        #     num_to_visit = num_points
        
        if prop_to_visit < 0.0:
            if num_to_visit < 0:
                prop_to_visit = 1.0
            else:
                prop_to_visit = -1.0 
        else:
            if prop_to_visit > 1.0:
                prop_to_visit = 1.0
        
        # if num_to_retrieve > num_points:
        #     num_to_retrieve = num_points
        
        if prop_to_retrieve < 0.0:
            if num_to_retrieve < 0:
                prop_to_retrieve = 0.002
            else:
                prop_to_retrieve = -1.0
        else:
            if prop_to_retrieve > 1.0:
                prop_to_retrieve = 1.0
        
        duplicate_delete_ids = _dci_delete(self._dci_inst, data_ids, blind, num_to_visit, num_to_retrieve, prop_to_visit, prop_to_retrieve, field_of_view)

        return duplicate_delete_ids

    
    # query is num_queries x dim
    def query(self, query, mask, num_neighbours = -1, field_of_view = 100, blind = False, num_to_visit = -1, num_to_retrieve = -1, prop_to_visit = -1.0, prop_to_retrieve = -1.0, parallel_level=0, ratio=1):
        _query = self._check_and_fix_array(query)
        
        num_points = self.num_points[0]
        
        if num_neighbours < 0:
            num_neighbours = num_points
        
        self._ensure_positive_integer(num_neighbours)
        self._ensure_positive_integer(ratio)
        
        if self.num_levels[0] >= 2:
            self._ensure_positive_integer(field_of_view)
        else:
            field_of_view = -1
        
        if num_to_visit > num_points:
            num_to_visit = num_points
        
        if prop_to_visit < 0.0:
            if num_to_visit < 0:
                prop_to_visit = 1.0
            else:
                prop_to_visit = -1.0 
        else:
            if prop_to_visit > 1.0:
                prop_to_visit = 1.0
        
        if num_to_retrieve > num_points:
            num_to_retrieve = num_points
        
        if prop_to_retrieve < 0.0:
            if num_to_retrieve < 0:
                prop_to_retrieve = 0.05
            else:
                prop_to_retrieve = -1.0
        else:
            if prop_to_retrieve > 1.0:
                prop_to_retrieve = 1.0
        
        # num_queries x num_neighbours
        _nearest_neighbour_idx, _num_returned = _dci_query(self._dci_inst, _query, mask, num_neighbours, blind, num_to_visit, num_to_retrieve, prop_to_visit, prop_to_retrieve, field_of_view, parallel_level, ratio)
        
        if self._orig_indices is not None:
            _nearest_neighbour_idx = self._orig_indices[_nearest_neighbour_idx]
        
        return _nearest_neighbour_idx, _num_returned
    
    def quick_page_sort(self, query, page_indices, input_num):
        sorted_page_idx = _dci_quick_sort(self._dci_inst, query, page_indices, input_num)

        return sorted_page_idx

    def cached_tree(self, query, page_insert_indices, returned_num, kept_indices, replace=False):
        assert(kept_indices.dtype == np.intc or kept_indices.dtype == np.int32 or kept_indices.shape[0] == 0)
        assert(returned_num.dtype == np.intc or returned_num.dtype == np.int32 or returned_num.shape[0] == 0)
        assert(page_insert_indices.dtype == np.intc or page_insert_indices.dtype == np.int32)
        assert page_insert_indices.shape[0] > 0
        sorted_page_idx = _dci_cached_tree(self._dci_inst, query, page_insert_indices, returned_num, kept_indices, replace)

        return sorted_page_idx
    
    def get_valid_entries(self, leaf_index):
        assert(leaf_index.dtype == np.intc or leaf_index.dtype == np.int32 or leaf_index.shape[0] == 0)
        valid_entries = _dci_get_valid_entries(self._dci_inst, leaf_index)

        return valid_entries
    
    def add_query(self, key, query, value, mask, num_levels = -100, num_points = 0, blind = False, c_num_to_visit = -1, c_num_to_retrieve = -1,
                    c_prop_to_visit = -1.0, c_prop_to_retrieve = -1.0, c_field_of_view = 10, num_neighbours = -1, q_field_of_view = 100,
                    q_num_to_visit = -1, q_num_to_retrieve = -1, q_prop_to_visit = -1.0, q_prop_to_retrieve = -1.0, transform = False,
                    parallel_level = 0, causal = False, attention_mask = None, random = True, paged = False, budget = 0,
                    do_query=True, track=False, update_addr=False, changed_page_list=None, data_proj=None, ratio=1, interval=0, X=0, anchor_threshold=0.9):

        data_ids = np.array([]).astype(np.float32)  # not used in IceFormer

        if data_proj is None:
            data_proj = np.array([]).astype(np.float32)

        if changed_page_list is None:
            changed_page_list = np.array([]).astype(np.bool_)

        assert num_points > 0

        assert ratio == 1  # Currently add_query does not support group query
        
        if c_prop_to_visit < 0.0:
            if c_num_to_visit < 0:
                c_prop_to_visit = 1.0
            else:
                c_prop_to_visit = -1.0 
        else:
            if c_prop_to_visit > 1.0:
                c_prop_to_visit = 1.0
        
        if c_prop_to_retrieve < 0.0:
            if c_num_to_retrieve < 0:
                c_prop_to_retrieve = 0.002
            else:
                c_prop_to_retrieve = -1.0
        else:
            if c_prop_to_retrieve > 1.0:
                c_prop_to_retrieve = 1.0
        
        self._check_data(key)
        is_contiguous, _indices = self._check_and_fix_indices(key, None)
        
        if do_query:
            _query = self._check_and_fix_array(query)
        else:
            _query = np.array([])
        _mask = mask
        
        
        if num_neighbours < 0:
            num_neighbours = num_points

        self._ensure_positive_integer(num_neighbours)

        if budget <= num_neighbours:
            budget = num_neighbours
        
        # num_levels is not used in this version
        if 0 < num_levels <= 2:
            c_field_of_view = -1
            if num_levels == 1:
                q_field_of_view = -1
        
        if q_prop_to_visit < 0.0:
            if q_num_to_visit < 0:
                q_prop_to_visit = 1.0
            else:
                q_prop_to_visit = -1.0 
        else:
            if q_prop_to_visit > 1.0:
                q_prop_to_visit = 1.0
        
        if q_prop_to_retrieve < 0.0:
            if q_num_to_retrieve < 0:
                q_prop_to_retrieve = 0.05
            else:
                q_prop_to_retrieve = -1.0
        else:
            if q_prop_to_retrieve > 1.0:
                q_prop_to_retrieve = 1.0

        if attention_mask is not None:
            attention_mask = attention_mask.astype(np.intc)
        else:
            if causal:
                parallel_level = 1
                attention_mask = np.arange(num_points, dtype=np.intc) + 1
            else:
                attention_mask = (np.ones(num_points, dtype=np.intc) * num_points).astype(np.intc)

        self._check_attention_mask(attention_mask)
        self._ensure_positive_integer(ratio)
        # num_queries x num_neighbours
        if is_contiguous:
            nearest_neighbour_idx, changed_page_idx = _dci_add_query(self._dci_inst, key, data_ids, _query, value, _mask, num_levels, num_neighbours, blind, c_num_to_visit, q_num_to_visit, c_num_to_retrieve,
                                                q_num_to_retrieve, c_prop_to_visit, q_prop_to_visit, c_prop_to_retrieve, q_prop_to_retrieve,
                                                c_field_of_view, q_field_of_view, self.num_comp_indices, self.num_simp_indices,
                                                num_points, transform, parallel_level, attention_mask, random, do_query, track, update_addr, changed_page_list, data_proj, ratio, interval, X, anchor_threshold)
        else:
            selected_data = key[_indices]
            nearest_neighbour_idx, changed_page_idx = _dci_add_query(self._dci_inst, selected_data, data_ids, _query, value, _mask, num_levels, num_neighbours, blind, c_num_to_visit, q_num_to_visit, c_num_to_retrieve,
                                                q_num_to_retrieve, c_prop_to_visit, q_prop_to_visit, c_prop_to_retrieve, q_prop_to_retrieve,
                                                c_field_of_view, q_field_of_view, self.num_comp_indices, self.num_simp_indices,
                                                num_points, transform, parallel_level, attention_mask, random, do_query, track, update_addr, changed_page_list, data_proj, ratio, interval, X, anchor_threshold)
            self._orig_indices = _indices
        
        return nearest_neighbour_idx, changed_page_idx
    

    def add_query_at_end(self, key, query, value, mask, num_levels = -100, num_points = 0, blind = False, c_num_to_visit = -1, c_num_to_retrieve = -1,
                    c_prop_to_visit = -1.0, c_prop_to_retrieve = -1.0, c_field_of_view = 10, num_neighbours = -1, q_field_of_view = 100,
                    q_num_to_visit = -1, q_num_to_retrieve = -1, q_prop_to_visit = -1.0, q_prop_to_retrieve = -1.0, transform = False,
                    parallel_level = 0, causal = False, attention_mask = None, random = True, paged = False, budget = 0,
                    do_query=True, track=False, update_addr=False, changed_page_list=None, data_proj=None, ratio=1, interval=0, X=0, anchor_threshold=0.9):

        data_ids = np.array([]).astype(np.float32)  # not used in IceFormer    

        if data_proj is None:
            data_proj = np.array([]).astype(np.float32)

        if changed_page_list is None:
            changed_page_list = np.array([]).astype(np.bool_)

        assert num_points > 0
        
        if c_prop_to_visit < 0.0:
            if c_num_to_visit < 0:
                c_prop_to_visit = 1.0
            else:
                c_prop_to_visit = -1.0 
        else:
            if c_prop_to_visit > 1.0:
                c_prop_to_visit = 1.0
        
        if c_prop_to_retrieve < 0.0:
            if c_num_to_retrieve < 0:
                c_prop_to_retrieve = 0.002
            else:
                c_prop_to_retrieve = -1.0
        else:
            if c_prop_to_retrieve > 1.0:
                c_prop_to_retrieve = 1.0
        
        self._check_data(key)
        is_contiguous, _indices = self._check_and_fix_indices(key, None)
        
        if do_query:
            _query = self._check_and_fix_array(query)
        else:
            _query = np.array([])
        _mask = mask
        
        
        if num_neighbours < 0:
            num_neighbours = num_points

        self._ensure_positive_integer(num_neighbours)
        self._ensure_positive_integer(ratio)

        if budget <= num_neighbours:
            budget = num_neighbours
        
        # num_levels is not used in this version
        if 0 < num_levels <= 2:
            c_field_of_view = -1
            if num_levels == 1:
                q_field_of_view = -1
        
        if q_prop_to_visit < 0.0:
            if q_num_to_visit < 0:
                q_prop_to_visit = 1.0
            else:
                q_prop_to_visit = -1.0 
        else:
            if q_prop_to_visit > 1.0:
                q_prop_to_visit = 1.0
        
        if q_prop_to_retrieve < 0.0:
            if q_num_to_retrieve < 0:
                q_prop_to_retrieve = 0.05
            else:
                q_prop_to_retrieve = -1.0
        else:
            if q_prop_to_retrieve > 1.0:
                q_prop_to_retrieve = 1.0

        if attention_mask is not None:
            attention_mask = attention_mask.astype(np.intc)
        else:
            if causal:
                parallel_level = 1
                attention_mask = np.arange(num_points, dtype=np.intc) + 1
            else:
                attention_mask = (np.ones(num_points, dtype=np.intc) * num_points).astype(np.intc)

        self._check_attention_mask(attention_mask)
        
        # num_queries x num_neighbours
        if is_contiguous:
            nearest_neighbour_idx, changed_page_idx = _dci_add_query_at_end(self._dci_inst, key, data_ids, _query, value, _mask, num_levels, num_neighbours, blind, c_num_to_visit, q_num_to_visit, c_num_to_retrieve,
                                                q_num_to_retrieve, c_prop_to_visit, q_prop_to_visit, c_prop_to_retrieve, q_prop_to_retrieve,
                                                c_field_of_view, q_field_of_view, self.num_comp_indices, self.num_simp_indices,
                                                num_points, transform, parallel_level, attention_mask, random, do_query, track, update_addr, changed_page_list, data_proj, ratio, interval, X, anchor_threshold)
        else:
            selected_data = key[_indices]
            nearest_neighbour_idx, changed_page_idx = _dci_add_query_at_end(self._dci_inst, selected_data, data_ids, _query, value, _mask, num_levels, num_neighbours, blind, c_num_to_visit, q_num_to_visit, c_num_to_retrieve,
                                                q_num_to_retrieve, c_prop_to_visit, q_prop_to_visit, c_prop_to_retrieve, q_prop_to_retrieve,
                                                c_field_of_view, q_field_of_view, self.num_comp_indices, self.num_simp_indices,
                                                num_points, transform, parallel_level, attention_mask, random, do_query, track, update_addr, changed_page_list, data_proj, ratio, interval, X, anchor_threshold)
            self._orig_indices = _indices
        
        return nearest_neighbour_idx, changed_page_idx

    def add_query_attention(self, key, query, value, mask, num_inst = 1, indices = None, num_levels = 2, num_points = 0, blind = False, c_num_to_visit = -1, c_num_to_retrieve = -1, 
                            c_prop_to_visit = -1.0, c_prop_to_retrieve = -1.0, c_field_of_view = 10, num_neighbours = -1,  q_field_of_view = 100, 
                            q_num_to_visit = -1, q_num_to_retrieve = -1, q_prop_to_visit = -1.0, q_prop_to_retrieve = -1.0, transform = False, parallel_level = 0, 
                            debug = False, causal = False, attention_mask = None, random = True, paged = False, budget = 0):
        
        data_ids = np.array([]).astype(np.float32)  # not used in IceFormer

        assert num_points > 0
        
        if c_prop_to_visit < 0.0:
            if c_num_to_visit < 0:
                c_prop_to_visit = 1.0
            else:
                c_prop_to_visit = -1.0 
        else:
            if c_prop_to_visit > 1.0:
                c_prop_to_visit = 1.0
        
        if c_prop_to_retrieve < 0.0:
            if c_num_to_retrieve < 0:
                c_prop_to_retrieve = 0.002
            else:
                c_prop_to_retrieve = -1.0
        else:
            if c_prop_to_retrieve > 1.0:
                c_prop_to_retrieve = 1.0
        
        self._check_data(key)
        is_contiguous, _indices = self._check_and_fix_indices(key, indices)
        
        _query = self._check_and_fix_array(query)
        _value = self._check_and_fix_array(value)
        _mask = mask
        
        
        if num_neighbours < 0:
            num_neighbours = num_points

        self._ensure_positive_integer(num_neighbours)

        if budget <= num_neighbours:
            budget = num_neighbours
        
        # num_levels is not used in this version
        if 0 < num_levels <= 2:
            c_field_of_view = -1
            if num_levels == 1:
                q_field_of_view = -1
        
        if q_prop_to_visit < 0.0:
            if q_num_to_visit < 0:
                q_prop_to_visit = 1.0
            else:
                q_prop_to_visit = -1.0 
        else:
            if q_prop_to_visit > 1.0:
                q_prop_to_visit = 1.0
        
        if q_prop_to_retrieve < 0.0:
            if q_num_to_retrieve < 0:
                q_prop_to_retrieve = 0.05
            else:
                q_prop_to_retrieve = -1.0
        else:
            if q_prop_to_retrieve > 1.0:
                q_prop_to_retrieve = 1.0

        if causal:
            parallel_level = 1
            attention_mask = np.arange(num_points, dtype=np.intc) + 1
        elif attention_mask is not None:
            attention_mask = attention_mask.astype(np.intc)
        else:
            attention_mask = (np.ones(num_points, dtype=np.intc) * num_points).astype(np.intc)

        self._check_attention_mask(attention_mask)

        num_returned = np.array([])
        
        # num_queries x num_neighbours
        if is_contiguous:
            new_value, nearest_neighbour_idx = _dci_add_query_attention(key, data_ids, _query, _value, _mask, num_inst, num_levels, num_neighbours, blind, c_num_to_visit, q_num_to_visit, c_num_to_retrieve, 
                                        q_num_to_retrieve, c_prop_to_visit, q_prop_to_visit, c_prop_to_retrieve, q_prop_to_retrieve, 
                                        c_field_of_view, q_field_of_view, self.num_comp_indices, self.num_simp_indices, self.promotion_prob, self.promotion_prob_subseq,
                                        num_points, transform, parallel_level, debug, attention_mask, random)
        else:
            selected_data = key[_indices]
            new_value, nearest_neighbour_idx = _dci_add_query_attention(selected_data, data_ids, _query, _value, _mask, num_inst, num_levels, num_neighbours, blind, c_num_to_visit, q_num_to_visit, c_num_to_retrieve, 
                                        q_num_to_retrieve, c_prop_to_visit, q_prop_to_visit, c_prop_to_retrieve, q_prop_to_retrieve, 
                                        c_field_of_view, q_field_of_view, self.num_comp_indices, self.num_simp_indices, self.promotion_prob, self.promotion_prob_subseq,
                                        num_points, transform, parallel_level, debug, attention_mask, random)
            self._orig_indices = _indices
        
        return num_returned, new_value, nearest_neighbour_idx
    
    def clear(self):
        _dci_clear(self._dci_inst)
        self._array = None
        self._orig_indices = None
    
    def reset(self):
        _dci_reset(self._dci_inst)

    def reset_proj(self, proj_vec, transform=False):
        assert(proj_vec.dtype == np.float32)
        _dci_reset_proj(self._dci_inst, proj_vec, transform)
