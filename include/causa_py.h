#ifndef CAUSA_PY_H
#define CAUSA_PY_H

// Pure CPython C API bindings for the tensor library.
// No external dependencies - requires only Python.h and libpython.
//
// Build:
//   python setup.py build_ext --inplace
//
// Usage:
//   import tensor
//   t = tensor.zeros([2, 3])
//   t[0, 0] = 1.5
//   print(t)

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#endif