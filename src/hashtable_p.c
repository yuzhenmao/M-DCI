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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include "hashtable_p.h"

#define LOAD_FACTOR 1.0

void hashtable_p_init(hashtable_p* const ht, const int expected_num_entries, const int key_interval) {
    int num_buckets = (int)ceil(expected_num_entries * LOAD_FACTOR);

    ht->size = num_buckets;
    ht->key_interval = key_interval;
    ht->entries = (htentry_p**)calloc(num_buckets, sizeof(htentry_p*));
}

static inline int hashtable_p_hash(const hashtable_p* const ht, const long long key) {
    return ((key / ht->key_interval) + 84751LL) % ht->size;
}

bool hashtable_p_exists(const hashtable_p* const ht, const long long key) {
    int hashval = hashtable_p_hash(ht, key);
    htentry_p *x = ht->entries[hashval];
    
    while (x) {
        if (x->key == key)
            return true;
        x = x->next;
    }
    return false;
}

void hashtable_p_extend(hashtable_p* ht, const int new_num_entries) {  // Extend the hashtable using "dynamic hashing"
    int old_size = ht->size;
    int new_size = (int)ceil(new_num_entries * LOAD_FACTOR);
    ht->size = new_size;
    htentry_p** new_entries = (htentry_p**)realloc(ht->entries, new_size * sizeof(htentry_p*));
    if (new_entries == NULL) {
        perror("Memory allocation failed!\n");
        return;
    }
    ht->entries = new_entries;
    for (int i = old_size; i < new_size; i++)
        ht->entries[i] = NULL;

    // Rehash all entries
    for (int i = 0; i < old_size; i++) {
        htentry_p* entry = ht->entries[i];
        htentry_p* prev_entry = NULL;
        htentry_p* next_entry;
        while (entry) {
            next_entry = entry->next;
            int new_hashval = hashtable_p_hash(ht, entry->key);
            int bit_flag = new_hashval & ((old_size - 1) ^ (new_size - 1));
            if (bit_flag == 0) {  // Keep unchanged
                prev_entry = entry;
            }
            else {  // Move to the new bucket
                if (prev_entry != NULL)
                    prev_entry->next = next_entry;
                else
                    ht->entries[i] = next_entry;
                entry->next = ht->entries[new_hashval];
                ht->entries[new_hashval] = entry;
            }
            entry = next_entry;
        }
    }
}

addinfo_level* hashtable_p_get(const hashtable_p* const ht, const long long key, const addinfo_level* default_value) {
    int hashval = hashtable_p_hash(ht, key);
    htentry_p *x = ht->entries[hashval];
    
    while (x) {
        if (x->key == key)
            return x->value;
        x = x->next;
    }
    return default_value;
}

void hashtable_p_set(const hashtable_p* const ht, const long long key, const additional_info* addinfo, const int level, const int offset) {
    int hashval = hashtable_p_hash(ht, key);
    htentry_p *x = ht->entries[hashval];
    addinfo_level* value = (addinfo_level*)malloc(sizeof(addinfo_level));
    value->addinfo = addinfo;
    value->level = level;
    value->offset = offset;
    
    while (x) {
        if (x->key == key)
            break;
        x = x->next;
    }
    if (x) {
        free(x->value);
        x->value = value;
    } else {
        x = (htentry_p*)malloc(sizeof(htentry_p));
        x->key = key;
        x->value = value;
        x->next = ht->entries[hashval];
        
        ht->entries[hashval] = x;
    }
}

bool hashtable_p_delete(const hashtable_p* const ht, const long long key) {
    int hashval = hashtable_p_hash(ht, key);
    htentry_p **x_loc = &(ht->entries[hashval]);
    htentry_p *to_delete;
    while (*x_loc) {
        if ((*x_loc)->key == key)
            break;
        x_loc = &((*x_loc)->next);
    }
    if (*x_loc) {
        if((*x_loc)->value != NULL) {
            free((*x_loc)->value);
            (*x_loc)->value = NULL;
        }
        to_delete = *x_loc;
        *x_loc = (*x_loc)->next;
        free(to_delete);
        return true;
    } else {
        return false;
    }
}

void hashtable_p_clear(const hashtable_p* const ht) {
    int i;
    htentry_p *x, *next;
    if(ht == NULL) return;
    for (i = 0; i < ht->size; i++) {
        if(ht->entries == NULL) break;
        x = ht->entries[i];
        while (x) {
            if(x->value != NULL) {
                free(x->value);
                x->value = NULL;
            }
            next = x->next;
            free(x);
            x = next;
        }
        ht->entries[i] = NULL;
    }
}

void hashtable_p_free(const hashtable_p* const ht) {
    hashtable_p_clear(ht);
    free(ht->entries);
}

void hashtable_p_dump(const hashtable_p* const ht) {
    int i;
    htentry_p *x;
    for (i = 0; i < ht->size; i++) {
        if (ht->entries[i]) {
            printf("%d: ", i);
            x = ht->entries[i];
            while (x) {
                printf("%lld[%d]->", x->key, x->value->level);
                x = x->next;
            }
            printf("NIL\n");
        }
    }
}
