import sys
import os
import platform
from setuptools import setup, find_packages, Extension

try:
    import numpy
except ImportError:
    raise ImportError("NumPy is required. Please install it first: pip install numpy")

def get_dci_numpy_extension(multithreading):
    """
    Build the dciknn._dci C extension using the original NumPy-based API (py_dci.c).
    This is what dciknn.core.DCI uses.
    """
    import platform

    dci_core_sources = [
        'src/dci.c',
        'src/util.c',
        'src/debug.c',
        'src/hashtable_i.c',
        'src/hashtable_d.c',
        'src/btree_i.c',
        'src/btree_p.c',
        'src/hashtable_p.c',
        'src/hashtable_pp.c',
        'src/stack.c',
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    include_dir = os.path.join(script_dir, 'include')
    include_dirs = [include_dir, numpy.get_include()]

    # extra_compile_args = ['-Wall', '-std=gnu99', '-m64', '-g', '-O0']  # for debugging
    extra_compile_args = ['-Wall', '-std=gnu99', '-m64', '-O3']
    extra_link_args = ['-lm']

    if platform.system() == 'Darwin':
        extra_compile_args.append('-march=native')
        extra_link_args.extend(['-framework', 'Accelerate'])
    else:
        extra_compile_args.append('-march=core-avx2')
        extra_link_args.append('-lopenblas')

    if multithreading:
        extra_compile_args.extend(['-fopenmp', '-DUSE_OPENMP'])
        if platform.system() == 'Darwin':
            extra_link_args.append('-lomp')
        else:
            extra_link_args.append('-lgomp')

    return Extension(
        name='dciknn._dci',
        sources=['src/py_dci.c'] + dci_core_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )


def main():
    answer = "y"
    multithreading = answer.lower().startswith("y")

    ext_modules = [get_dci_numpy_extension(multithreading)]

    setup(
        name='dci',
        version='0.2.0',
        packages=find_packages(),
        description="""
            Dynamic Continuous Indexing (DCI) is a family of randomized algorithms for
            exact k-nearest neighbour search that overcomes the curse of dimensionality.
            Its query time complexity is linear in ambient dimensionality and sublinear
            in intrinsic dimensionality. This package contains the reference implementation
            of DCI with both NumPy and PyTorch native APIs.
            """,
        ext_modules=ext_modules,
        author='Yuzhen Mao, Ke Li',
        author_email='yuzhenm@sfu.ca',
        url='https://yuzhenmao.github.io/IceFormer/',
        license='Mozilla Public License 2.0',
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
            'Programming Language :: C',
            'Programming Language :: C++',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
        long_description="""
        Multi-level Dynamic Continuous Indexing (DCI)

        This package provides efficient k-nearest neighbor search with two API options:

        1. **NumPy API** (legacy): Compatible with existing code using NumPy arrays
        2. **PyTorch API** (new): Native PyTorch tensor support with:
           - `add_query_torch()`: Combined add and query operations
           - `query_torch()`: Standalone query operations
           - Efficient GIL release for multi-threaded performance
           - Zero-copy tensor operations

        Both APIs support multi-level indexing, parallel processing, and are optimized
        for CPU-based nearest neighbor search in high-dimensional spaces.
        """,
        install_requires=['torch>=1.8.0', 'numpy>=1.19.0', 'scikit-learn'],
        python_requires='>=3.8',
    )

if __name__ == '__main__':
    main()