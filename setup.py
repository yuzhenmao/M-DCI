import sys
import os
from setuptools import setup, find_packages

try:
    from torch.utils.cpp_extension import CppExtension, BuildExtension
    import torch
except ImportError:
    raise ImportError("PyTorch is required. Please install it first: pip install torch")

try:
    import numpy
except ImportError:
    raise ImportError("NumPy is required. Please install it first: pip install numpy")

try:
    from numpy.distutils.system_info import get_info
except ImportError:
    # numpy.distutils is deprecated, use a fallback
    def get_info(name, notfound_action=0):
        return {}


def get_dci_extension(multithreading, debug):
    """
    Build the DCI C++ extension with PyTorch integration.
    This extension includes both the original NumPy-based API (py_dci.c)
    and the new PyTorch-native API (dci_wrapper.cpp with add_query_torch and query_torch).
    """
    # DCI sources and headers
    dci_sources = [
        'src/dci.c',
        'src/util.c',
        'src/debug.c',
        'src/hashtable_i.c',
        'src/hashtable_d.c',
        'src/btree_i.c',
        'src/btree_p.c',
        'src/hashtable_p.c',
        'src/hashtable_pp.c',
        'src/hashtable_wrapper.cpp',
        'src/stack.c',
        'src/bf16_util.c',
    ]

    dci_headers = [
        'include/dci.h',
        'include/util.h',
        'include/debug.h',
        'include/hashtable_i.h',
        'include/hashtable_d.h',
        'include/btree_i.h',
        'include/btree_p.h',
        'include/hashtable_p.h',
        'include/hashtable_pp.h',
        'include/hashtable_wrapper.h',
        'include/stack.h',
        'include/bf16_util.h',
    ]

    # Try to get BLAS/LAPACK info
    lapack_info = get_info('lapack_opt', 1)
    blas_libraries = []
    blas_library_dirs = []

    if lapack_info:
        blas_libraries = lapack_info.get('libraries', [])
        blas_library_dirs = lapack_info.get('library_dirs', [])

    # If no BLAS found via numpy, try common locations
    if not blas_libraries:
        print("Warning: No BLAS library found via numpy. Trying common BLAS libraries...")
        # Try common BLAS libraries in order of preference
        for blas_name in ['openblas', 'blas', 'cblas', 'Accelerate']:
            blas_libraries = [blas_name]
            print(f"  Attempting to use: {blas_name}")
            break

    # Compilation arguments
    extra_compile_args = ['-std=c++17', '-O3']
    if debug:
        extra_compile_args = ['-std=c++17', '-O0', '-g', '-UNDEBUG']
    extra_link_args = []

    # Platform-specific settings
    import platform
    if platform.system() == 'Darwin':  # macOS
        # Use Accelerate framework on macOS
        extra_link_args.extend(['-framework', 'Accelerate'])
        # Avoid march=core-avx2 on macOS as it may not be compatible with all Macs
        extra_compile_args.append('-march=native')
    else:  # Linux and others
        extra_compile_args.extend(['-mavx512f', '-mavx512bw', '-mavx512dq', '-mavx512vl', '-mavx512bf16', '-mfma'])
        # Link BLAS libraries
        for lib in blas_libraries:
            extra_link_args.append(f'-l{lib}')

    if multithreading:
        extra_compile_args.extend(['-fopenmp', '-DUSE_OPENMP'])
        if platform.system() == 'Darwin':
            # On macOS, may need libomp from Homebrew
            extra_link_args.append('-lomp')
        else:
            extra_link_args.append('-lgomp')

    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative paths
    library_dir = os.path.join(script_dir, 'include')

    # Include directories
    include_dirs = [library_dir, numpy.get_include()]

    # Library directories
    library_dirs = [library_dir] + blas_library_dirs

    return CppExtension(
        name='dci',
        sources=['src/dci_wrapper.cpp'] + dci_sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=blas_libraries if platform.system() != 'Darwin' else [],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        depends=dci_headers
    )

def main():
    answer = "y"
    multithreading = answer.lower().startswith("y")
    debug = answer.lower().startswith("f")

    ext_modules = [get_dci_extension(multithreading, debug)]

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
        cmdclass={
            'build_ext': BuildExtension.with_options(use_ninja=False)
        },
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