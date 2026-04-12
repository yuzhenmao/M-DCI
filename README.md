# M-DCI: Multi-level Dynamic Continuous Indexing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/yuzhenmao/M-DCI/blob/main/LICENSE)

> **You are on the `bf16` branch.** This branch builds the DCI index over **bf16** vectors and pairs with the [`bf16` branch of IceCache](https://github.com/yuzhenmao/IceCache/tree/bf16). Your CPU must support bf16. For a float32 build that runs on CPUs without bf16 support, use the [`main` branch](https://github.com/yuzhenmao/M-DCI/tree/main).

**M-DCI** is a reference implementation of **Multi-level Dynamic Continuous Indexing**, a family of randomized algorithms for k-nearest neighbour search that overcomes the curse of dimensionality. Its query time is linear in ambient dimensionality and sublinear in intrinsic dimensionality, making it well-suited for fast CPU-based ANN search in high-dimensional spaces.

This package is the indexing backend used by [**IceCache**](https://github.com/yuzhenmao/IceCache) for retrieving relevant KV pages during long-sequence LLM inference, but it can also be used as a standalone ANN library.

## Features

- Multi-level (composite + simple) DCI indexing
- OpenMP parallelism for both index construction and query
- NumPy API for general-purpose use
- Optimized C core with AVX2 / Apple Accelerate / OpenBLAS backends
- Zero-copy interop with PyTorch tensors

## Requirements

- A CPU with **bf16** support (e.g. Intel Sapphire Rapids / AMD Zen 4 or newer)
- A C compiler (GCC on Linux, Clang on macOS)
- OpenBLAS or MKL

> **CPU threads:** M-DCI is heavily parallelized. For best performance run on a machine with many cores and set `OMP_NUM_THREADS` accordingly:
> ```bash
> export OMP_NUM_THREADS=64
> ```

## Installation

```bash
git clone -b bf16 https://github.com/yuzhenmao/M-DCI.git
cd M-DCI
python3 setup.py install
```

## Repository Structure

```
M-DCI/
├── include/          # C headers (dci.h, btree, hashtable, util)
├── src/              # C / C++ sources
│   ├── dci.c         # Core DCI algorithm
│   ├── py_dci.c      # NumPy/CPython bindings
│   ├── btree_*.c     # B-tree backing structures
│   └── hashtable_*.c # Hashtable backing structures
├── dciknn/           # Python package
│   ├── __init__.py
│   └── core.py       # DCI Python class
└── setup.py
```

## Use with IceCache

[IceCache](https://github.com/yuzhenmao/IceCache) uses M-DCI to index CPU-resident KV pages and retrieve the most relevant ones at each decoding step. After installing IceCache, install M-DCI into the same Python environment:

```bash
git clone -b bf16 https://github.com/yuzhenmao/M-DCI.git
cd M-DCI
python3 setup.py install
```

## Citation

If you use M-DCI in your research, please cite:

```bibtex
@inproceedings{
mao2024iceformer,
title={IceFormer: Accelerated Inference with Long-Sequence Transformers on {CPU}s},
author={Yuzhen Mao and Martin Ester and Ke Li},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=6RR3wU4mSZ}
}
```

```bibtex
@inproceedings{
mao2026icecache,
title={IceCache: Memory-Efficient {KV}-cache Management for Long-Sequence {LLM}s},
author={Yuzhen Mao and Qitong Wang and Martin Ester and Ke Li},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=yHxSKM9kdr}
}
```

The original Prioritized DCI algorithm is described in [Li & Malik, 2017](https://arxiv.org/abs/1703.00440).
