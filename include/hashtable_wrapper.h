#ifndef HASHTABLE_H
#define HASHTABLE_H

#include <torch/torch.h>
#include <unordered_map>
#include <memory>
#include <variant>

// Include the pybind11 header
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // For py::array_t

namespace py = pybind11;

enum DataType { TORCH, NUMPY };

struct HTEntry {
    long long key;
    std::variant<py::array_t<float>, torch::Tensor> value;
    DataType type;
};

class HashTable {
public:
    HashTable(int key_interval, int expected_num_entries);

    bool exists(long long key) const;
    std::variant<py::array_t<float>, torch::Tensor> get(long long key, const std::variant<py::array_t<float>, torch::Tensor>& default_value) const;
    void set(long long key, const std::variant<py::array_t<float>, torch::Tensor>& value, DataType type);
    bool remove(long long key);
    void clear();
    int getSize() const { return size; }

private:
    int key_interval;
    int size;
    std::unordered_map<long long, std::shared_ptr<HTEntry>> table;
};

#endif // HASHTABLE_H