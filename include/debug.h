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

#ifndef DEBUG_H
#define DEBUG_H

// If this is a C++ compiler, use C linkage
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "btree_p.h"
#include "hashtable_p.h"

void print_dci(dci* dci_inst);

void get_parent_stat(dci* dci_inst, int X, int interval, bool* parent_in_anchor_set, float* distance_ratios, bool* parent_consistency, float* min_dist_to_anchor_parents, float* max_dist_closest_parent_to_children);

void print_cell_num(dci* dci_inst);

void print_tree(int num_levels, int* num_points_on_level, additional_info* root);

void print_num_points_on_level(dci* dci_inst);

void check_dci(dci* dci_inst);

#ifdef __cplusplus
}
#endif

#endif // UTIL_H
