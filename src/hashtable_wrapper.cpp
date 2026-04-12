#include "hashtable_wrapper.h"
#include <stdexcept>
#include <numpy/ndarrayobject.h>

HashTable::HashTable(int key_interval, int expected_num_entries) 
    : key_interval(key_interval), size(0) {
    table.reserve(expected_num_entries);
}

bool HashTable::exists(long long key) const {
    return table.find(key) != table.end();
}

std::variant<py::array_t<float>, torch::Tensor> HashTable::get(long long key, const std::variant<py::array_t<float>, torch::Tensor>& default_value) const {
    auto it = table.find(key);
    if (it != table.end()) {
        return it->second->value;
    }
    return default_value;
}

void HashTable::set(long long key, const std::variant<py::array_t<float>, torch::Tensor>& value, DataType type) {
    auto entry = std::make_shared<HTEntry>();
    entry->key = key;
    entry->value = value;
    entry->type = type;
    table[key] = entry;
    size++;
}

bool HashTable::remove(long long key) {
    auto it = table.find(key);
    if (it != table.end()) {
        table.erase(it);
        size--;
        return true;
    }
    return false;
}

void HashTable::clear() {
    table.clear();
    size = 0;
}