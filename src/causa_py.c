#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "causa.h"
#include <string.h>
#include <math.h>

// ============================================================
// Module state
// ============================================================

typedef struct {
    PyObject* arena;         // causa_arena_t* wrapped
    PyObject* backend;       // causa_backend_t* wrapped
} causa_module_t;

static causa_module_t g_module;

// Capsule helpers for passing raw pointers through Python
static const char* ARENA_CAPSULE = "causa.arena";
static const char* BACKEND_CAPSULE = "causa.backend";

static void arena_capsule_destructor(PyObject* cap) {
    causa_arena_t* arena = (causa_arena_t*)PyCapsule_GetPointer(cap, ARENA_CAPSULE);
    // causa_arena_free releases the blocks and the arena itself, so the
    // pointer must not be freed again here.
    if (arena) causa_arena_free(arena);
}

static void backend_capsule_destructor(PyObject* cap) {
    causa_backend_t* backend = (causa_backend_t*)PyCapsule_GetPointer(cap, BACKEND_CAPSULE);
    if (backend) causa_backend_free(backend);
}

// ============================================================
// Tensor object
// ============================================================

typedef struct {
    PyObject_HEAD
    causa_tensor_t* tensor;
    causa_arena_t* arena;
    causa_backend_t* backend;
    PyObject* arena_cap;     // keeps arena alive
    PyObject* backend_cap;   // keeps backend alive
    int tape_idx;            // index into the autograd tape, -1 if leaf
    // Where the tape of this tensor's graph begins. A backward pass walks
    // no further back than this, so that one graph does not disturb
    // another.
    int tape_start;
} PyTensorObject;

static PyTypeObject PyTensor_Type;

// ============================================================
// Autograd tape
//
// Every differentiable operation appends an entry to a global tape.
// backward() walks the tape in reverse, accumulating gradients.
// ============================================================

typedef enum {
    TAPE_ADD, TAPE_SUB, TAPE_MUL, TAPE_DIV, TAPE_MATMUL,
    TAPE_COPY, TAPE_RELU, TAPE_GELU, TAPE_SILU, CAUSA_ACT_TANH_TAPE,
    TAPE_SIGMOID, TAPE_SOFTMAX, TAPE_LOG_SOFTMAX,
    TAPE_SUM, TAPE_MEAN, TAPE_MAX, TAPE_MIN,
    TAPE_TRANSPOSE, TAPE_RESHAPE,
    TAPE_NEG, TAPE_ABS, TAPE_SQRT, TAPE_EXP, TAPE_LOG,
    TAPE_ADD_SCALAR, TAPE_SUB_SCALAR, TAPE_MUL_SCALAR, TAPE_DIV_SCALAR
} tape_op_t;

// The operation with a plain number beside it.
//
// The number is not a tensor, so it is kept in the entry rather than as
// one of its inputs.
static int tape_append_scalar(
    tape_op_t op,
    PyObject** inputs, int num_inputs,
    PyObject* output,
    double value
);

static int tape_append(
    tape_op_t op,
    PyObject** inputs, int num_inputs,
    PyObject* output,
    int32_t dim0, int32_t dim1,
    const int64_t* out_shape, int32_t out_ndim
);

typedef struct {
    tape_op_t op;
    PyObject* inputs[4];      // strong references
    int num_inputs;
    PyObject* output;         // strong reference
    // Whether this entry has been reached by the current backward pass.
    // The tape is shared by every graph, so the walk has to know which
    // entries belong to the graph it is working on.
    bool reached;
    // The number beside the operation, when there was one
    double scalar;
    // extra parameters
    int32_t dim0, dim1;
    int64_t out_shape[CAUSA_MAX_DIMS];
    int32_t out_ndim;
} tape_entry_t;

static tape_entry_t* g_tape = NULL;
static int g_tape_count = 0;
static int g_tape_capacity = 0;

static int tape_reserve(void) {
    if (g_tape_count < g_tape_capacity) return g_tape_count;
    int new_cap = g_tape_capacity ? g_tape_capacity * 2 : 64;
    tape_entry_t* nt = (tape_entry_t*)realloc(g_tape, (size_t)new_cap * sizeof(tape_entry_t));
    if (!nt) return -1;
    g_tape = nt;
    g_tape_capacity = new_cap;
    return g_tape_count;
}

static int tape_append(
    tape_op_t op,
    PyObject** inputs, int num_inputs,
    PyObject* output,
    int32_t dim0, int32_t dim1,
    const int64_t* out_shape, int32_t out_ndim
) {
    int idx = tape_reserve();
    if (idx < 0) return -1;

    tape_entry_t* e = &g_tape[idx];
    e->op = op;
    e->num_inputs = num_inputs;
    // The tape comes from realloc, so the mark has to be cleared. Left
    // over from an earlier entry, it would make the backward pass skip
    // the entry and lose its gradient.
    e->reached = false;

    // The graph of the output begins where the graphs of its inputs begin.
    // The earliest of them is the one that counts, because the graph has to
    // reach all of them.
    int start = -1;
    for (int i = 0; i < num_inputs; i++) {
        PyTensorObject* in = (PyTensorObject*)inputs[i];
        if (in->tape_start < 0) continue;
        if (start < 0 || in->tape_start < start) start = in->tape_start;
    }
    if (start < 0) start = idx;
    if (output) ((PyTensorObject*)output)->tape_start = start;

    for (int i = 0; i < num_inputs; i++) {
        e->inputs[i] = inputs[i];
        Py_INCREF(inputs[i]);
    }
    e->output = output;
    Py_INCREF(output);
    e->dim0 = dim0;
    e->dim1 = dim1;
    e->out_ndim = out_ndim;
    if (out_shape && out_ndim > 0) {
        memcpy(e->out_shape, out_shape, (size_t)out_ndim * sizeof(int64_t));
    }
    g_tape_count = idx + 1;
    return idx;
}

// Record an operation with a plain number beside it.
//
// The number is not a tensor, so it is kept in the entry rather than as
// one of its inputs.
static int tape_append_scalar(
    tape_op_t op,
    PyObject** inputs, int num_inputs,
    PyObject* output,
    double value
) {
    int idx = tape_append(op, inputs, num_inputs, output, 0, 0, NULL, 0);
    if (idx >= 0) g_tape[idx].scalar = value;
    return idx;
}

// Release all tape entries (called at interpreter shutdown or on demand)
static void tape_clear(void) {
    for (int i = 0; i < g_tape_count; i++) {
        for (int j = 0; j < g_tape[i].num_inputs; j++) {
            Py_XDECREF(g_tape[i].inputs[j]);
        }
        Py_XDECREF(g_tape[i].output);
    }
    g_tape_count = 0;
    if (g_tape) {
        free(g_tape);
        g_tape = NULL;
        g_tape_capacity = 0;
    }
}

static void causa_dealloc(PyTensorObject* self) {
    Py_XDECREF(self->arena_cap);
    Py_XDECREF(self->backend_cap);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* causa_new_internal(
    causa_tensor_t* t,
    causa_arena_t* arena,
    causa_backend_t* backend,
    PyObject* arena_cap,
    PyObject* backend_cap
) {
    PyTensorObject* self = PyObject_New(PyTensorObject, &PyTensor_Type);
    if (!self) return NULL;
    self->tensor = t;
    self->arena = arena;
    self->backend = backend;
    // The capsules are borrowed here. The caller keeps its own reference
    // for as long as it needs the arena, and this object holds one of its
    // own so that the arena outlives the tensor.
    Py_XINCREF(arena_cap);
    Py_XINCREF(backend_cap);
    self->arena_cap = arena_cap;
    self->backend_cap = backend_cap;
    self->tape_idx = -1;
    self->tape_start = -1;
    return (PyObject*)self;
}

static PyObject* causa_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs
) {
    (void)type; (void)args; (void)kwargs;
    PyErr_SetString(PyExc_TypeError, "Tensor cannot be constructed directly; use causa.zeros/randn/full/empty");
    return NULL;
}

// The arena and the backend that a tensor made here is given.
//
// The arena holds the tensor and its data. A small block is enough for most
// tensors, and the arena grows if it has to, so a big block would only
// reserve memory that is never used.
//
// The caller frees both if the tensor cannot be made.
static void causa_new_arena(
    causa_arena_t** out_arena,
    causa_backend_t** out_backend
) {
    *out_arena = causa_arena_new(64 * 1024);
    *out_backend = causa_backend_cpu_new();
}

// Make a tensor object that owns its arena.
//
// The capsules are created here and handed to the tensor, which keeps them
// alive. The reference this function starts with is given up again, so
// that the arena is freed when the tensor is. Without that, every tensor
// would keep a megabyte of arena for ever.
static PyObject* causa_new_owned(
    causa_tensor_t* t,
    causa_arena_t* arena,
    causa_backend_t* backend
) {
    PyObject* arena_cap = PyCapsule_New((void*)arena, ARENA_CAPSULE, arena_capsule_destructor);
    PyObject* backend_cap = PyCapsule_New((void*)backend, BACKEND_CAPSULE, backend_capsule_destructor);
    if (!arena_cap || !backend_cap) {
        Py_XDECREF(arena_cap);
        Py_XDECREF(backend_cap);
        return NULL;
    }

    PyObject* result = causa_new_internal(t, arena, backend, arena_cap, backend_cap);

    // The tensor holds a reference of its own now.
    Py_DECREF(arena_cap);
    Py_DECREF(backend_cap);

    return result;
}

// --- shape helpers ---

static PyObject* causa_shape(PyTensorObject* self, void* closure) {
    (void)closure;
    if (!self->tensor) Py_RETURN_NONE;
    PyObject* shape = PyList_New(self->tensor->ndim);
    if (!shape) return NULL;
    for (int32_t i = 0; i < self->tensor->ndim; i++) {
        PyObject* v = PyLong_FromLongLong((long long)self->tensor->shape[i]);
        if (!v) { Py_DECREF(shape); return NULL; }
        PyList_SET_ITEM(shape, i, v);
    }
    return shape;
}

static PyObject* causa_strides(PyTensorObject* self, void* closure) {
    (void)closure;
    if (!self->tensor) Py_RETURN_NONE;
    PyObject* strides = PyList_New(self->tensor->ndim);
    if (!strides) return NULL;
    for (int32_t i = 0; i < self->tensor->ndim; i++) {
        PyObject* v = PyLong_FromLongLong((long long)self->tensor->strides[i]);
        if (!v) { Py_DECREF(strides); return NULL; }
        PyList_SET_ITEM(strides, i, v);
    }
    return strides;
}

static PyObject* causa_numel(PyTensorObject* self, void* closure) {
    (void)closure;
    if (!self->tensor) Py_RETURN_NONE;
    return PyLong_FromLongLong((long long)causa_tensor_numel(self->tensor));
}

static PyObject* causa_nbytes(PyTensorObject* self, void* closure) {
    (void)closure;
    if (!self->tensor) Py_RETURN_NONE;
    return PyLong_FromLongLong((long long)causa_tensor_nbytes(self->tensor));
}

static PyObject* causa_dtype(PyTensorObject* self, void* closure) {
    (void)closure;
    if (!self->tensor) Py_RETURN_NONE;
    return PyUnicode_FromString(causa_dtype_name(self->tensor->dtype));
}

static PyObject* causa_device(PyTensorObject* self, void* closure) {
    (void)closure;
    if (!self->tensor) Py_RETURN_NONE;
    return PyUnicode_FromString(causa_device_name(self->tensor->device));
}

static PyObject* causa_requires_grad(PyTensorObject* self, void* closure) {
    (void)closure;
    if (!self->tensor) Py_RETURN_NONE;
    return PyBool_FromLong(self->tensor->requires_grad);
}

static int causa_set_requires_grad(PyTensorObject* self, PyObject* value, void* closure) {
    (void)closure;
    if (!self->tensor) { PyErr_SetString(PyExc_ValueError, "invalid tensor"); return -1; }
    int truth = PyObject_IsTrue(value);
    if (truth < 0) return -1;
    self->tensor->requires_grad = (truth != 0);
    return 0;
}

static PyObject* causa_grad(PyTensorObject* self, void* closure) {
    (void)closure;
    if (!self->tensor || !self->tensor->grad) Py_RETURN_NONE;
    return causa_new_internal(self->tensor->grad, self->arena, self->backend,
                               self->arena_cap, self->backend_cap);
}

static PyObject* causa_name(PyTensorObject* self, void* closure) {
    (void)closure;
    if (!self->tensor) Py_RETURN_NONE;
    return PyUnicode_FromString(self->tensor->name);
}

static int causa_set_name(PyTensorObject* self, PyObject* value, void* closure) {
    (void)closure;
    if (!self->tensor) { PyErr_SetString(PyExc_ValueError, "invalid tensor"); return -1; }
    const char* s = PyUnicode_AsUTF8(value);
    if (!s) return -1;
    strncpy(self->tensor->name, s, CAUSA_MAX_NAME - 1);
    self->tensor->name[CAUSA_MAX_NAME - 1] = '\0';
    return 0;
}

static PyGetSetDef causa_getset[] = {
    {"shape", (getter)causa_shape, NULL, "Tensor shape as a list", NULL},
    {"strides", (getter)causa_strides, NULL, "Tensor strides as a list", NULL},
    {"numel", (getter)causa_numel, NULL, "Number of elements", NULL},
    {"nbytes", (getter)causa_nbytes, NULL, "Size in bytes", NULL},
    {"dtype", (getter)causa_dtype, NULL, "Data type name", NULL},
    {"device", (getter)causa_device, NULL, "Device name", NULL},
    {"requires_grad", (getter)causa_requires_grad, (setter)causa_set_requires_grad, "Whether gradients are tracked", NULL},
    {"grad", (getter)causa_grad, NULL, "Gradient tensor or None", NULL},
    {"name", (getter)causa_name, (setter)causa_set_name, "Tensor name", NULL},
    {NULL}
};

// --- data access ---

// Build a nested Python list matching the tensor shape
static PyObject* build_nested_list(const float* data, const int64_t* shape, int32_t ndim, int32_t dim) {
    if (dim == ndim - 1) {
        PyObject* list = PyList_New(shape[dim]);
        if (!list) return NULL;
        for (int64_t i = 0; i < shape[dim]; i++) {
            PyObject* v = PyFloat_FromDouble((double)data[i]);
            if (!v) { Py_DECREF(list); return NULL; }
            PyList_SET_ITEM(list, i, v);
        }
        return list;
    }

    PyObject* list = PyList_New(shape[dim]);
    if (!list) return NULL;

    int64_t stride = 1;
    for (int32_t i = dim + 1; i < ndim; i++) stride *= shape[i];

    for (int64_t i = 0; i < shape[dim]; i++) {
        PyObject* sub = build_nested_list(data + i * stride, shape, ndim, dim + 1);
        if (!sub) { Py_DECREF(list); return NULL; }
        PyList_SET_ITEM(list, i, sub);
    }
    return list;
}

static PyObject* causa_to_list(PyTensorObject* self, void* closure) {
    (void)closure;
    if (!self->tensor) Py_RETURN_NONE;

    const float* data = (const float*)self->tensor->data;
    return build_nested_list(data, self->tensor->shape, self->tensor->ndim, 0);
}

// --- operators ---

static PyObject* causa_getitem(PyTensorObject* self, PyObject* key) {
    if (!self->tensor) {
        PyErr_SetString(PyExc_ValueError, "invalid tensor");
        return NULL;
    }

    int64_t n = causa_tensor_numel(self->tensor);
    const float* data = (const float*)self->tensor->data;

    // Integer index
    if (PyIndex_Check(key)) {
        Py_ssize_t idx = PyNumber_AsSsize_t(key, PyExc_IndexError);
        if (idx == -1 && PyErr_Occurred()) return NULL;
        if (idx < 0) idx += n;
        if (idx < 0 || idx >= n) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return NULL;
        }
        return PyFloat_FromDouble((double)data[idx]);
    }

    // Slice
    if (PySlice_Check(key)) {
        Py_ssize_t start, stop, step, slicelen;
        if (PySlice_GetIndicesEx(key, n, &start, &stop, &step, &slicelen) < 0) return NULL;
        PyObject* list = PyList_New(slicelen);
        if (!list) return NULL;
        for (Py_ssize_t i = 0; i < slicelen; i++) {
            PyObject* v = PyFloat_FromDouble((double)data[start + i * step]);
            if (!v) { Py_DECREF(list); return NULL; }
            PyList_SET_ITEM(list, i, v);
        }
        return list;
    }

    PyErr_SetString(PyExc_TypeError, "tensor indices must be integers or slices");
    return NULL;
}

static int causa_setitem(PyTensorObject* self, PyObject* key, PyObject* value) {
    if (!self->tensor) {
        PyErr_SetString(PyExc_ValueError, "invalid tensor");
        return -1;
    }

    int64_t n = causa_tensor_numel(self->tensor);
    float* data = (float*)self->tensor->data;

    if (PyIndex_Check(key)) {
        Py_ssize_t idx = PyNumber_AsSsize_t(key, PyExc_IndexError);
        if (idx == -1 && PyErr_Occurred()) return -1;
        if (idx < 0) idx += n;
        if (idx < 0 || idx >= n) {
            PyErr_SetString(PyExc_IndexError, "index out of range");
            return -1;
        }
        double v = PyFloat_AsDouble(value);
        if (v == -1.0 && PyErr_Occurred()) return -1;
        data[idx] = (float)v;
        return 0;
    }

    if (PySlice_Check(key)) {
        Py_ssize_t start, stop, step, slicelen;
        if (PySlice_GetIndicesEx(key, n, &start, &stop, &step, &slicelen) < 0) return -1;

        // Accept a sequence of values
        PyObject* seq = PySequence_Fast(value, "can only assign an iterable");
        if (!seq) return -1;
        if (PySequence_Fast_GET_SIZE(seq) != slicelen) {
            PyErr_SetString(PyExc_ValueError, "slice length mismatch");
            Py_DECREF(seq);
            return -1;
        }
        for (Py_ssize_t i = 0; i < slicelen; i++) {
            double v = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, i));
            if (v == -1.0 && PyErr_Occurred()) { Py_DECREF(seq); return -1; }
            data[start + i * step] = (float)v;
        }
        Py_DECREF(seq);
        return 0;
    }

    PyErr_SetString(PyExc_TypeError, "tensor indices must be integers or slices");
    return -1;
}

// Binary op dispatch
typedef enum {
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MATMUL
} binop_t;

static PyObject* causa_binop(PyTensorObject* self, PyObject* other, binop_t op, const char* opname) {
    if (!self->tensor) {
        PyErr_SetString(PyExc_ValueError, "invalid tensor");
        return NULL;
    }

    causa_tensor_t* result = NULL;
    PyObject* out = NULL;
    int needs_grad = self->tensor->requires_grad;
    tape_op_t top = TAPE_ADD;
    switch (op) {
        case OP_ADD: top = TAPE_ADD; break;
        case OP_SUB: top = TAPE_SUB; break;
        case OP_MUL: top = TAPE_MUL; break;
        case OP_DIV: top = TAPE_DIV; break;
        case OP_MATMUL: top = TAPE_MATMUL; break;
    }

    if (PyObject_TypeCheck(other, &PyTensor_Type)) {
        PyTensorObject* o = (PyTensorObject*)other;
        needs_grad = needs_grad || o->tensor->requires_grad;

        switch (op) {
            case OP_ADD: result = causa_tensor_add(self->tensor, o->tensor); break;
            case OP_SUB: result = causa_tensor_sub(self->tensor, o->tensor); break;
            case OP_MUL: result = causa_tensor_mul(self->tensor, o->tensor); break;
            case OP_DIV: result = causa_tensor_div(self->tensor, o->tensor); break;
            case OP_MATMUL: result = causa_tensor_matmul(self->tensor, o->tensor); break;
        }

        if (!result) {
            PyErr_Format(PyExc_ValueError, "%s failed (shape or dtype mismatch)", opname);
            return NULL;
        }

        out = causa_new_internal(result, self->arena, self->backend,
                                  self->arena_cap, self->backend_cap);
        if (!out) return NULL;

        // Record on the tape only if an input requires gradients
        if (needs_grad) {
            PyObject* ins[2] = {(PyObject*)self, other};
            int idx = tape_append(top, ins, 2, out, 0, 0, NULL, 0);
            if (idx >= 0) ((PyTensorObject*)out)->tape_idx = idx;
        }
        return out;
    }

    // A plain number on the right of the operator. The library has the
    // operations for these, so there is no reason to refuse them.
    if (PyFloat_Check(other) || PyLong_Check(other) || PyIndex_Check(other)) {
        double value = PyFloat_AsDouble(other);
        if (value == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            PyErr_Format(PyExc_TypeError, "unsupported operand type(s) for %s: 'Tensor' and '%s'",
                         opname, Py_TYPE(other)->tp_name);
            return NULL;
        }

        switch (op) {
            case OP_ADD: result = causa_tensor_add_scalar(self->tensor, value); break;
            case OP_SUB: result = causa_tensor_sub_scalar(self->tensor, value); break;
            case OP_MUL: result = causa_tensor_mul_scalar(self->tensor, value); break;
            case OP_DIV: result = causa_tensor_div_scalar(self->tensor, value); break;
            default:
                PyErr_Format(PyExc_TypeError, "unsupported operand type(s) for %s: 'Tensor' and '%s'",
                             opname, Py_TYPE(other)->tp_name);
                return NULL;
        }

        if (!result) {
            PyErr_Format(PyExc_ValueError, "%s failed", opname);
            return NULL;
        }

        out = causa_new_internal(result, self->arena, self->backend,
                                  self->arena_cap, self->backend_cap);
        if (!out) return NULL;

        if (needs_grad) {
            // The number is not a tensor, so the operation is recorded with
            // the tensor alone and the value beside it.
            PyObject* ins[1] = {(PyObject*)self};
            tape_op_t stop = top;
            switch (op) {
                case OP_ADD: stop = TAPE_ADD_SCALAR; break;
                case OP_SUB: stop = TAPE_SUB_SCALAR; break;
                case OP_MUL: stop = TAPE_MUL_SCALAR; break;
                case OP_DIV: stop = TAPE_DIV_SCALAR; break;
                default: break;
            }
            int idx = tape_append_scalar(stop, ins, 1, out, value);
            if (idx >= 0) ((PyTensorObject*)out)->tape_idx = idx;
        }
        return out;
    }

    PyErr_Format(PyExc_TypeError, "unsupported operand type(s) for %s: 'Tensor' and '%s'",
                 opname, Py_TYPE(other)->tp_name);
    return NULL;
}

static PyObject* causa_add(PyObject* self, PyObject* other) {
    return causa_binop((PyTensorObject*)self, other, OP_ADD, "add");
}
static PyObject* causa_sub(PyObject* self, PyObject* other) {
    return causa_binop((PyTensorObject*)self, other, OP_SUB, "subtract");
}
static PyObject* causa_mul(PyObject* self, PyObject* other) {
    return causa_binop((PyTensorObject*)self, other, OP_MUL, "multiply");
}
static PyObject* causa_div(PyObject* self, PyObject* other) {
    return causa_binop((PyTensorObject*)self, other, OP_DIV, "divide");
}
static PyObject* causa_matmul(PyObject* self, PyObject* other) {
    return causa_binop((PyTensorObject*)self, other, OP_MATMUL, "matmul");
}

// Helper: wrap a unary activation result and record it on the tape
static PyObject* unary_result(PyTensorObject* self, causa_tensor_t* r, tape_op_t top) {
    // A failure has to raise something. Returning nothing without saying
    // why leaves Python to raise an error about the function itself.
    if (!r) {
        PyErr_SetString(PyExc_RuntimeError, "the operation could not be computed");
        return NULL;
    }
    PyObject* out = causa_new_internal(r, self->arena, self->backend,
                                        self->arena_cap, self->backend_cap);
    if (!out) return NULL;
    if (self->tensor->requires_grad) {
        PyObject* ins[1] = {(PyObject*)self};
        int idx = tape_append(top, ins, 1, out, 0, 0, NULL, 0);
        if (idx >= 0) ((PyTensorObject*)out)->tape_idx = idx;
    }
    return out;
}

static PyObject* causa_neg(PyTensorObject* self) {
    if (!self->tensor) Py_RETURN_NONE;
    return unary_result(self, causa_tensor_neg(self->tensor), TAPE_NEG);
}

static PyObject* causa_abs(PyTensorObject* self) {
    if (!self->tensor) Py_RETURN_NONE;
    return unary_result(self, causa_tensor_abs(self->tensor), TAPE_ABS);
}

static PyObject* causa_sqrt(PyTensorObject* self) {
    if (!self->tensor) Py_RETURN_NONE;
    return unary_result(self, causa_tensor_sqrt(self->tensor), TAPE_SQRT);
}

static PyObject* causa_exp(PyTensorObject* self) {
    if (!self->tensor) Py_RETURN_NONE;
    return unary_result(self, causa_tensor_exp(self->tensor), TAPE_EXP);
}

static PyObject* causa_log(PyTensorObject* self) {
    if (!self->tensor) Py_RETURN_NONE;
    return unary_result(self, causa_tensor_log(self->tensor), TAPE_LOG);
}

static PyObject* causa_relu(PyTensorObject* self) {
    if (!self->tensor) Py_RETURN_NONE;
    return unary_result(self, causa_tensor_activation(self->tensor, CAUSA_ACT_RELU, 0.0), TAPE_RELU);
}

static PyObject* causa_gelu(PyTensorObject* self) {
    if (!self->tensor) Py_RETURN_NONE;
    return unary_result(self, causa_tensor_activation(self->tensor, CAUSA_ACT_GELU, 0.0), TAPE_GELU);
}

static PyObject* causa_sigmoid(PyTensorObject* self) {
    if (!self->tensor) Py_RETURN_NONE;
    return unary_result(self, causa_tensor_activation(self->tensor, CAUSA_ACT_SIGMOID, 0.0), TAPE_SIGMOID);
}

static PyObject* causa_tanh(PyTensorObject* self) {
    if (!self->tensor) Py_RETURN_NONE;
    return unary_result(self, causa_tensor_activation(self->tensor, CAUSA_ACT_TANH, 0.0), CAUSA_ACT_TANH_TAPE);
}

static PyObject* causa_softmax(PyTensorObject* self) {
    if (!self->tensor) Py_RETURN_NONE;
    return unary_result(self, causa_tensor_activation(self->tensor, CAUSA_ACT_SOFTMAX, 0.0), TAPE_SOFTMAX);
}

// reduction helper
static PyObject* causa_reduce(PyTensorObject* self, causa_reduce_op_t op, const char* name, tape_op_t top) {
    if (!self->tensor) Py_RETURN_NONE;
    int32_t dims[CAUSA_MAX_DIMS];
    int32_t ndim = self->tensor->ndim;
    for (int32_t i = 0; i < ndim; i++) dims[i] = i;
    // keepdim=true so the result always has at least one dimension
    causa_tensor_t* r = causa_tensor_reduce(self->tensor, op, dims, ndim, true);
    if (!r) {
        PyErr_Format(PyExc_ValueError, "%s failed", name);
        return NULL;
    }
    PyObject* out = causa_new_internal(r, self->arena, self->backend,
                                        self->arena_cap, self->backend_cap);
    if (!out) return NULL;
    if (self->tensor->requires_grad) {
        PyObject* ins[1] = {(PyObject*)self};
        int idx = tape_append(top, ins, 1, out, 0, 0, self->tensor->shape, self->tensor->ndim);
        if (idx >= 0) ((PyTensorObject*)out)->tape_idx = idx;
    }
    return out;
}

static PyObject* causa_sum(PyTensorObject* self) {
    return causa_reduce(self, CAUSA_REDUCE_SUM, "sum", TAPE_SUM);
}
static PyObject* causa_mean(PyTensorObject* self) {
    return causa_reduce(self, CAUSA_REDUCE_MEAN, "mean", TAPE_MEAN);
}
static PyObject* causa_max(PyTensorObject* self) {
    return causa_reduce(self, CAUSA_REDUCE_MAX, "max", TAPE_MAX);
}
static PyObject* causa_min(PyTensorObject* self) {
    return causa_reduce(self, CAUSA_REDUCE_MIN, "min", TAPE_MIN);
}

static PyObject* causa_transpose(PyTensorObject* self, PyObject* args) {
    if (!self->tensor) Py_RETURN_NONE;
    int d0 = 0, d1 = 1;
    if (!PyArg_ParseTuple(args, "ii", &d0, &d1)) return NULL;
    causa_tensor_t* r = causa_tensor_transpose(self->tensor, d0, d1);
    if (!r) { PyErr_SetString(PyExc_ValueError, "transpose failed"); return NULL; }
    PyObject* out = causa_new_internal(r, self->arena, self->backend,
                                        self->arena_cap, self->backend_cap);
    if (!out) return NULL;
    if (self->tensor->requires_grad) {
        PyObject* ins[1] = {(PyObject*)self};
        int idx = tape_append(TAPE_TRANSPOSE, ins, 1, out, d0, d1, NULL, 0);
        if (idx >= 0) ((PyTensorObject*)out)->tape_idx = idx;
    }
    return out;
}

static PyObject* causa_reshape(PyTensorObject* self, PyObject* args) {
    if (!self->tensor) Py_RETURN_NONE;
    PyObject* shape_obj;
    if (!PyArg_ParseTuple(args, "O", &shape_obj)) return NULL;
    PyObject* seq = PySequence_Fast(shape_obj, "shape must be a sequence");
    if (!seq) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n <= 0 || n > CAUSA_MAX_DIMS) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "invalid shape");
        return NULL;
    }
    int64_t shape[CAUSA_MAX_DIMS];
    for (Py_ssize_t i = 0; i < n; i++) {
        shape[i] = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(seq, i));
        if (PyErr_Occurred()) { Py_DECREF(seq); return NULL; }
    }
    Py_DECREF(seq);
    causa_tensor_t* r = causa_tensor_reshape(self->tensor, shape, (int32_t)n);
    if (!r) { PyErr_SetString(PyExc_ValueError, "reshape failed"); return NULL; }
    PyObject* out = causa_new_internal(r, self->arena, self->backend,
                                        self->arena_cap, self->backend_cap);
    if (!out) return NULL;
    if (self->tensor->requires_grad) {
        PyObject* ins[1] = {(PyObject*)self};
        int idx = tape_append(TAPE_RESHAPE, ins, 1, out, 0, 0, shape, (int32_t)n);
        if (idx >= 0) ((PyTensorObject*)out)->tape_idx = idx;
    }
    return out;
}

// backward pass: walk the tape in reverse, accumulating gradients
static PyObject* causa_backward(PyTensorObject* self, PyObject* args, PyObject* kwargs) {
    if (!self->tensor) Py_RETURN_NONE;

    static char* kwlist[] = {"grad", NULL};
    PyObject* grad_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &grad_obj)) return NULL;

    // Where the walk starts and where it stops. It stops at the beginning
    // of this tensor's graph: going further back would reach into other
    // graphs, whose gradients would then be added to this one's.
    int start = self->tape_idx;
    int first = self->tape_start;
    if (first < 0 || first > start) first = 0;

    // Seed gradient
    causa_tensor_options_t gopts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    gopts.dtype = self->tensor->dtype;

    causa_tensor_t* seed = NULL;
    if (grad_obj && PyObject_TypeCheck(grad_obj, &PyTensor_Type)) {
        seed = ((PyTensorObject*)grad_obj)->tensor;
    } else if (grad_obj) {
        PyErr_SetString(PyExc_TypeError, "the gradient must be a tensor");
        return NULL;
    } else {
        seed = causa_tensor_new(self->arena, self->backend,
                                  self->tensor->shape, self->tensor->ndim, &gopts);
        if (!seed) { PyErr_SetString(PyExc_RuntimeError, "failed to allocate seed gradient"); return NULL; }
        causa_tensor_fill(seed, 1.0);
    }

    // The seed has to be the same shape and type as the thing it is the
    // gradient of. A seed that does not fit would be copied over
    // nothing, and the gradient would quietly stay at zero.
    if (causa_tensor_numel(seed) != causa_tensor_numel(self->tensor) ||
        seed->dtype != self->tensor->dtype) {
        PyErr_SetString(PyExc_ValueError, "the gradient must have the same shape and type as the tensor");
        return NULL;
    }

    if (start < 0) {
        // Leaf: the gradient is already on self->tensor->grad
        PyObject* result = causa_new_internal(self->tensor->grad, self->arena, self->backend,
                                               self->arena_cap, self->backend_cap);
        return result;
    }

    // The entries this tensor reaches are found by following the inputs.
    //
    // The tape is shared by every graph, so the entries between the start
    // of this graph and this tensor may belong to others. Counting
    // backwards would apply their operations as well, so the entries that
    // are not reached are skipped.
    {
        unsigned char* seen = (unsigned char*)calloc((size_t)(g_tape_count + 1), 1);
        int* stack = (int*)malloc(sizeof(int) * (size_t)(g_tape_count + 1));
        if (!seen || !stack) {
            free(seen);
            free(stack);
            PyErr_SetString(PyExc_RuntimeError, "could not allocate");
            return NULL;
        }
        int top = 0;
        stack[top++] = start;

        while (top > 0) {
            int index = stack[--top];
            if (index < 0 || index >= g_tape_count) continue;
            if (seen[index]) continue;
            seen[index] = 1;
            g_tape[index].reached = true;

            for (int j = 0; j < g_tape[index].num_inputs; j++) {
                PyObject* in = g_tape[index].inputs[j];
                if (!in || !PyObject_TypeCheck(in, &PyTensor_Type)) continue;
                int in_index = ((PyTensorObject*)in)->tape_idx;
                if (in_index >= 0 && !seen[in_index]) {
                    stack[top++] = in_index;
                }
            }
        }
        free(seen);
        free(stack);
    }

    // The gradients of everything this graph touches are cleared before
    // the walk, so that a second call gives the same answer as the first
    // instead of adding to it.
    //
    // The clearing happens before the seed is put in place, or it would
    // clear the seed as well.
    for (int i = 0; i <= start && i < g_tape_count; i++) {
        if (!g_tape[i].reached) continue;
        tape_entry_t* e = &g_tape[i];
        if (e->output) {
            PyTensorObject* out_obj = (PyTensorObject*)e->output;
            if (out_obj->tensor != self->tensor && out_obj->tensor->grad) {
                causa_tensor_zero_grad(out_obj->tensor);
            }
        }
        for (int j = 0; j < e->num_inputs; j++) {
            PyTensorObject* in_obj = (PyTensorObject*)e->inputs[j];
            if (in_obj->tensor != self->tensor && in_obj->tensor->grad) {
                causa_tensor_zero_grad(in_obj->tensor);
            }
        }
    }

    // The output's own gradient
    if (self->tensor->grad) {
        causa_tensor_zero_grad(self->tensor);
        causa_tensor_copy(self->tensor->grad, seed);
    } else {
        causa_tensor_t* g = causa_tensor_new(self->arena, self->backend,
                                               self->tensor->shape, self->tensor->ndim, &gopts);
        if (!g) { PyErr_SetString(PyExc_RuntimeError, "failed to allocate gradient"); return NULL; }
        causa_tensor_copy(g, seed);
        self->tensor->grad = g;
    }

    for (int i = start; i >= 0; i--) {
        if (i >= g_tape_count) continue;
        // Only the entries this graph reaches are applied.
        if (!g_tape[i].reached) continue;
        tape_entry_t* e = &g_tape[i];
        if (!e->output) continue;

        PyTensorObject* out_obj = (PyTensorObject*)e->output;
        causa_tensor_t* grad = out_obj->tensor->grad;
        if (!grad) continue;

        switch (e->op) {
            case TAPE_ADD: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_t* b = ((PyTensorObject*)e->inputs[1])->tensor;
                causa_tensor_add_backward(a, b, grad);
                break;
            }
            case TAPE_SUB: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_t* b = ((PyTensorObject*)e->inputs[1])->tensor;
                causa_tensor_sub_backward(a, b, grad);
                break;
            }
            case TAPE_MUL: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_t* b = ((PyTensorObject*)e->inputs[1])->tensor;
                causa_tensor_mul_backward(a, b, grad);
                break;
            }
            case TAPE_DIV: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_t* b = ((PyTensorObject*)e->inputs[1])->tensor;
                causa_tensor_div_backward(a, b, grad);
                break;
            }
            case TAPE_MATMUL: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_t* b = ((PyTensorObject*)e->inputs[1])->tensor;
                causa_tensor_matmul_backward(a, b, grad);
                break;
            }
            case TAPE_RELU: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_activation_backward(a, CAUSA_ACT_RELU, 0.0, grad);
                break;
            }
            case TAPE_GELU: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_activation_backward(a, CAUSA_ACT_GELU, 0.0, grad);
                break;
            }
            case TAPE_SIGMOID: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_activation_backward(a, CAUSA_ACT_SIGMOID, 0.0, grad);
                break;
            }
            case CAUSA_ACT_TANH_TAPE: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_activation_backward(a, CAUSA_ACT_TANH, 0.0, grad);
                break;
            }
            case TAPE_SOFTMAX: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_activation_backward(a, CAUSA_ACT_SOFTMAX, 0.0, grad);
                break;
            }
            case TAPE_SUM:
            case TAPE_MEAN: {
                // The gradient of a sum is the seed spread over every
                // element. It is added to what is already there, because
                // the input may also be reached by another path.
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                if (!a->grad) {
                    causa_tensor_t* g = causa_tensor_new(a->arena, a->backend,
                                                           a->shape, a->ndim, &gopts);
                    if (!g) break;
                    a->grad = g;
                    causa_tensor_fill(a->grad, 0.0);
                }
                int64_t n = causa_tensor_numel(a);
                float scale = 1.0f;
                if (e->op == TAPE_MEAN) scale = 1.0f / (float)(n ? n : 1);
                int64_t gn = causa_tensor_numel(grad);
                for (int64_t k = 0; k < n; k++) {
                    float g = causa_tensor_load_f32(grad, k < gn ? k : gn - 1) * scale;
                    causa_tensor_store_f32(a->grad, k, causa_tensor_load_f32(a->grad, k) + g);
                }
                break;
            }
            case TAPE_MAX:
            case TAPE_MIN: {
                // Only the element that was the largest (or the smallest)
                // gets the gradient. It is found by looking at the input
                // again, which holds the value it had when the reduction
                // was made.
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                if (!a->grad) {
                    causa_tensor_t* g = causa_tensor_new(a->arena, a->backend,
                                                           a->shape, a->ndim, &gopts);
                    if (!g) break;
                    a->grad = g;
                    causa_tensor_fill(a->grad, 0.0);
                }
                int64_t n = causa_tensor_numel(a);
                if (n <= 0) break;
                int64_t best = 0;
                float best_value = causa_tensor_load_f32(a, 0);
                for (int64_t k = 1; k < n; k++) {
                    float v = causa_tensor_load_f32(a, k);
                    if (e->op == TAPE_MAX ? (v > best_value) : (v < best_value)) {
                        best_value = v;
                        best = k;
                    }
                }
                int64_t gn = causa_tensor_numel(grad);
                float g = gn > 0 ? causa_tensor_load_f32(grad, 0) : 0.0f;
                causa_tensor_store_f32(a->grad, best, causa_tensor_load_f32(a->grad, best) + g);
                break;
            }
            case TAPE_TRANSPOSE: {
                // Gradient w.r.t. input is the transpose of the output gradient
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_t* t = causa_tensor_transpose(grad, e->dim0, e->dim1);
                if (!t) break;
                causa_tensor_accumulate_grad(a, t);
                break;
            }
            case TAPE_RESHAPE: {
                // Gradient w.r.t. input is the output gradient reshaped back
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_t* r = causa_tensor_reshape(grad, e->out_shape, e->out_ndim);
                if (!r) break;
                causa_tensor_accumulate_grad(a, r);
                break;
            }
            case TAPE_COPY: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_accumulate_grad(a, grad);
                break;
            }
            case TAPE_ADD_SCALAR: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_add_scalar_backward(a, e->scalar, grad);
                break;
            }
            case TAPE_SUB_SCALAR: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_sub_scalar_backward(a, e->scalar, grad);
                break;
            }
            case TAPE_MUL_SCALAR: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_mul_scalar_backward(a, e->scalar, grad);
                break;
            }
            case TAPE_DIV_SCALAR: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_div_scalar_backward(a, e->scalar, grad);
                break;
            }
            case TAPE_NEG: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_neg_backward(a, grad);
                break;
            }
            case TAPE_ABS: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_abs_backward(a, grad);
                break;
            }
            case TAPE_SQRT: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_sqrt_backward(a, grad);
                break;
            }
            case TAPE_EXP: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_exp_backward(a, grad);
                break;
            }
            case TAPE_LOG: {
                causa_tensor_t* a = ((PyTensorObject*)e->inputs[0])->tensor;
                causa_tensor_log_backward(a, grad);
                break;
            }
        }
    }

    // Return the gradient accumulated at the root leaf (the earliest input
    // reachable through the tape that has no further inputs).
    //
    // Only the entries this graph reaches are looked at. The tape is
    // shared, so the leaf of another graph would otherwise be found first
    // and its gradient returned.
    PyTensorObject* root = NULL;
    for (int i = 0; i <= start && i < g_tape_count; i++) {
        if (!g_tape[i].reached) continue;
        tape_entry_t* e = &g_tape[i];
        if (!e->output) continue;
        PyTensorObject* out_obj = (PyTensorObject*)e->output;
        if (out_obj->tape_idx != i) continue;
        for (int j = 0; j < e->num_inputs; j++) {
            PyTensorObject* in_obj = (PyTensorObject*)e->inputs[j];
            if (in_obj->tape_idx < 0 && in_obj->tensor->grad) {
                root = in_obj;
                break;
            }
        }
        if (root) break;
    }

    // The marks are only for this pass. Without clearing them the next
    // backward pass would find everything already reached and do nothing.
    for (int i = 0; i < g_tape_count; i++) g_tape[i].reached = false;

    if (!root) {
        root = self;
    }
    if (!root->tensor->grad) Py_RETURN_NONE;

    PyObject* result = causa_new_internal(root->tensor->grad, root->arena, root->backend,
                                           root->arena_cap, root->backend_cap);
    return result;
}

static PyObject* causa_repr(PyTensorObject* self) {
    if (!self->tensor) return PyUnicode_FromString("Tensor(None)");
    PyObject* shape = causa_shape(self, NULL);
    if (!shape) return NULL;
    PyObject* shape_str = PyObject_Repr(shape);
    Py_DECREF(shape);
    if (!shape_str) return NULL;

    PyObject* result = PyUnicode_FromFormat("Tensor(shape=%S, dtype=%s, device=%s, requires_grad=%s)",
        shape_str, causa_dtype_name(self->tensor->dtype),
        causa_device_name(self->tensor->device),
        self->tensor->requires_grad ? "True" : "False");
    Py_DECREF(shape_str);
    return result;
}

static PyObject* causa_str(PyTensorObject* self) {
    PyObject* repr = causa_repr(self);
    if (!repr) return NULL;
    PyObject* s = PyUnicode_FromFormat("%S", repr);
    Py_DECREF(repr);
    return s;
}

static PyMethodDef causa_methods[] = {
    {"tolist", (PyCFunction)causa_to_list, METH_NOARGS, "Return tensor data as a nested Python list"},
    {"backward", (PyCFunction)causa_backward, METH_VARARGS | METH_KEYWORDS, "Compute gradients via reverse-mode autodiff"},
    {"transpose", (PyCFunction)causa_transpose, METH_VARARGS, "Transpose two dimensions"},
    {"reshape", (PyCFunction)causa_reshape, METH_VARARGS, "Reshape the tensor"},
    {"relu", (PyCFunction)causa_relu, METH_NOARGS, "ReLU activation"},
    {"gelu", (PyCFunction)causa_gelu, METH_NOARGS, "GELU activation"},
    {"sigmoid", (PyCFunction)causa_sigmoid, METH_NOARGS, "Sigmoid activation"},
    {"tanh", (PyCFunction)causa_tanh, METH_NOARGS, "Tanh activation"},
    {"softmax", (PyCFunction)causa_softmax, METH_NOARGS, "Softmax activation"},
    {"sqrt", (PyCFunction)causa_sqrt, METH_NOARGS, "Square root"},
    {"exp", (PyCFunction)causa_exp, METH_NOARGS, "Exponential"},
    {"log", (PyCFunction)causa_log, METH_NOARGS, "Natural logarithm"},
    {"abs", (PyCFunction)causa_abs, METH_NOARGS, "Absolute value"},
    {"sum", (PyCFunction)causa_sum, METH_NOARGS, "Sum all elements"},
    {"mean", (PyCFunction)causa_mean, METH_NOARGS, "Mean of all elements"},
    {"max", (PyCFunction)causa_max, METH_NOARGS, "Maximum element"},
    {"min", (PyCFunction)causa_min, METH_NOARGS, "Minimum element"},
    {NULL}
};

static PyNumberMethods causa_as_number = {
    .nb_add = causa_add,
    .nb_subtract = causa_sub,
    .nb_multiply = causa_mul,
    .nb_true_divide = causa_div,
    .nb_negative = (unaryfunc)causa_neg,
    .nb_absolute = (unaryfunc)causa_abs,
    .nb_matrix_multiply = causa_matmul,
};

static PyMappingMethods causa_as_mapping = {
    .mp_length = NULL,
    .mp_subscript = (binaryfunc)causa_getitem,
    .mp_ass_subscript = (objobjargproc)causa_setitem,
};

static PySequenceMethods causa_as_sequence = {
    .sq_length = NULL,
    .sq_item = NULL,
};

static PyTypeObject PyTensor_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "causa.Tensor",
    .tp_basicsize = sizeof(PyTensorObject),
    .tp_dealloc = (destructor)causa_dealloc,
    .tp_repr = (reprfunc)causa_repr,
    .tp_str = (reprfunc)causa_str,
    .tp_as_number = &causa_as_number,
    .tp_as_mapping = &causa_as_mapping,
    .tp_as_sequence = &causa_as_sequence,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Tensor",
    .tp_methods = causa_methods,
    .tp_getset = causa_getset,
    .tp_new = causa_new,
};

// matmul method (__matmul__ special)
static PyObject* causa_matmul_method(PyObject* self, PyObject* other) {
    return causa_matmul(self, other);
}

// ============================================================
// Module-level functions
// ============================================================

static PyObject* py_zeros(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;
    static char* kwlist[] = {"shape", "requires_grad", NULL};
    PyObject* shape_obj;
    int requires_grad = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", kwlist, &shape_obj, &requires_grad)) return NULL;

    PyObject* seq = PySequence_Fast(shape_obj, "shape must be a sequence");
    if (!seq) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n <= 0 || n > CAUSA_MAX_DIMS) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "invalid shape");
        return NULL;
    }
    int64_t shape[CAUSA_MAX_DIMS];
    for (Py_ssize_t i = 0; i < n; i++) {
        shape[i] = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(seq, i));
        if (PyErr_Occurred()) { Py_DECREF(seq); return NULL; }
    }
    Py_DECREF(seq);

causa_arena_t* a;
causa_backend_t* b;
causa_new_arena(&a, &b);

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.requires_grad = requires_grad != 0;

    causa_tensor_t* t = causa_tensor_new(a, b, shape, (int32_t)n, &opts);
    if (!t) {
        causa_arena_free(a);
        causa_backend_free(b);
        PyErr_SetString(PyExc_RuntimeError, "tensor allocation failed");
        return NULL;
    }
    causa_tensor_fill(t, 0.0);

    return causa_new_owned(t, a, b);
}

static PyObject* py_full(PyObject* self, PyObject* args, PyObject* kwargs) {    (void)self;
    static char* kwlist[] = {"shape", "value", "requires_grad", NULL};
    PyObject* shape_obj;
    double value;
    int requires_grad = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Od|p", kwlist, &shape_obj, &value, &requires_grad)) return NULL;

    PyObject* seq = PySequence_Fast(shape_obj, "shape must be a sequence");
    if (!seq) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n <= 0 || n > CAUSA_MAX_DIMS) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "invalid shape");
        return NULL;
    }
    int64_t shape[CAUSA_MAX_DIMS];
    for (Py_ssize_t i = 0; i < n; i++) {
        shape[i] = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(seq, i));
        if (PyErr_Occurred()) { Py_DECREF(seq); return NULL; }
    }
    Py_DECREF(seq);

causa_arena_t* a;
causa_backend_t* b;
causa_new_arena(&a, &b);

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.requires_grad = requires_grad != 0;

    causa_tensor_t* t = causa_tensor_new(a, b, shape, (int32_t)n, &opts);
    if (!t) {
        causa_arena_free(a);
        causa_backend_free(b);
        PyErr_SetString(PyExc_RuntimeError, "tensor allocation failed");
        return NULL;
    }
    causa_tensor_fill(t, value);

    return causa_new_owned(t, a, b);
}

static PyObject* py_randn(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;
    static char* kwlist[] = {"shape", "requires_grad", NULL};
    PyObject* shape_obj;
    int requires_grad = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", kwlist, &shape_obj, &requires_grad)) return NULL;

    PyObject* seq = PySequence_Fast(shape_obj, "shape must be a sequence");
    if (!seq) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n <= 0 || n > CAUSA_MAX_DIMS) {
        Py_DECREF(seq);
        PyErr_SetString(PyExc_ValueError, "invalid shape");
        return NULL;
    }
    int64_t shape[CAUSA_MAX_DIMS];
    for (Py_ssize_t i = 0; i < n; i++) {
        shape[i] = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(seq, i));
        if (PyErr_Occurred()) { Py_DECREF(seq); return NULL; }
    }
    Py_DECREF(seq);

causa_arena_t* a;
causa_backend_t* b;
causa_new_arena(&a, &b);

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.requires_grad = requires_grad != 0;

    causa_tensor_t* t = causa_tensor_new(a, b, shape, (int32_t)n, &opts);
    if (!t) {
        causa_arena_free(a);
        causa_backend_free(b);
        PyErr_SetString(PyExc_RuntimeError, "tensor allocation failed");
        return NULL;
    }
    causa_tensor_init(t, CAUSA_INIT_NORMAL, 0.0, 1.0);

    return causa_new_owned(t, a, b);
}

static PyObject* py_tensor(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;
    static char* kwlist[] = {"data", "requires_grad", NULL};
    PyObject* data;
    int requires_grad = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", kwlist, &data, &requires_grad)) return NULL;

    // Flatten nested sequences
    PyObject* flat = PyList_New(0);
    if (!flat) return NULL;

    PyObject* stack[16];
    int stack_size = 0;

    // Recursive flatten via list of lists
    PyObject* items = PySequence_Fast(data, "data must be a sequence");
    if (!items) { Py_DECREF(flat); return NULL; }

    Py_ssize_t count = PySequence_Fast_GET_SIZE(items);
    int64_t ndim = 1;
    Py_ssize_t inner = -1;
    Py_ssize_t num_sequences = 0;
    Py_ssize_t num_scalars = 0;
    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject* item = PySequence_Fast_GET_ITEM(items, i);
        if (PySequence_Check(item) && !PyBytes_Check(item) && !PyUnicode_Check(item)) {
            PyObject* sub = PySequence_Fast(item, "nested sequence expected");
            if (!sub) { Py_DECREF(items); Py_DECREF(flat); return NULL; }
            if (inner < 0) inner = PySequence_Fast_GET_SIZE(sub);
            else if (inner != PySequence_Fast_GET_SIZE(sub)) {
                Py_DECREF(sub); Py_DECREF(items); Py_DECREF(flat);
                PyErr_SetString(PyExc_ValueError, "ragged sequences are not supported");
                return NULL;
            }
            num_sequences++;
            for (Py_ssize_t j = 0; j < PySequence_Fast_GET_SIZE(sub); j++) {
                double v = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(sub, j));
                if (v == -1.0 && PyErr_Occurred()) { Py_DECREF(sub); Py_DECREF(items); Py_DECREF(flat); return NULL; }
                PyObject* f = PyFloat_FromDouble(v);
                if (!f) { Py_DECREF(sub); Py_DECREF(items); Py_DECREF(flat); return NULL; }
                if (PyList_Append(flat, f) != 0) {
                    Py_DECREF(f); Py_DECREF(sub); Py_DECREF(items); Py_DECREF(flat);
                    return NULL;
                }
                Py_DECREF(f);
            }
            Py_DECREF(sub);
        } else {
            double v = PyFloat_AsDouble(item);
            if (v == -1.0 && PyErr_Occurred()) { Py_DECREF(items); Py_DECREF(flat); return NULL; }
            PyObject* f = PyFloat_FromDouble(v);
            if (!f) { Py_DECREF(items); Py_DECREF(flat); return NULL; }
            if (PyList_Append(flat, f) != 0) {
                Py_DECREF(f); Py_DECREF(items); Py_DECREF(flat);
                return NULL;
            }
            Py_DECREF(f);
            num_scalars++;
        }
    }
    Py_DECREF(items);

    // A list that holds both single numbers and sequences has no shape.
    // The number of values would not divide into the rows, and the tensor
    // would be made smaller than the data, so the copy would run past the
    // end of it.
    if (num_scalars > 0 && num_sequences > 0) {
        Py_DECREF(flat);
        PyErr_SetString(PyExc_ValueError, "the data mixes single numbers and sequences");
        return NULL;
    }

    Py_ssize_t total = PyList_GET_SIZE(flat);
    if (total == 0) {
        Py_DECREF(flat);
        PyErr_SetString(PyExc_ValueError, "empty data");
        return NULL;
    }

    // Determine shape: 1D if flat elements, else infer from nesting
    if (inner > 0) {
        // 2D
        int64_t shape[2];
        shape[0] = (int64_t)(total / inner);
        shape[1] = (int64_t)inner;
        ndim = 2;
        (void)stack; (void)stack_size;

causa_arena_t* a;
causa_backend_t* b;
causa_new_arena(&a, &b);
        causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        opts.requires_grad = requires_grad != 0;
        causa_tensor_t* t = causa_tensor_new(a, b, shape, 2, &opts);
        if (!t) { Py_DECREF(flat); causa_arena_free(a); causa_backend_free(b); PyErr_SetString(PyExc_RuntimeError, "allocation failed"); return NULL; }

        float* data_ptr = (float*)t->data;
        for (Py_ssize_t i = 0; i < total; i++) {
            data_ptr[i] = (float)PyFloat_AsDouble(PyList_GET_ITEM(flat, i));
        }
        Py_DECREF(flat);

        return causa_new_owned(t, a, b);
    }

    // 1D
    int64_t shape[1];
    shape[0] = (int64_t)total;

causa_arena_t* a;
causa_backend_t* b;
causa_new_arena(&a, &b);
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.requires_grad = requires_grad != 0;
    causa_tensor_t* t = causa_tensor_new(a, b, shape, 1, &opts);
    if (!t) { Py_DECREF(flat); causa_arena_free(a); causa_backend_free(b); PyErr_SetString(PyExc_RuntimeError, "allocation failed"); return NULL; }

    float* data_ptr = (float*)t->data;
    for (Py_ssize_t i = 0; i < total; i++) {
        data_ptr[i] = (float)PyFloat_AsDouble(PyList_GET_ITEM(flat, i));
    }
    Py_DECREF(flat);

    return causa_new_owned(t, a, b);
}

static PyObject* py_save(PyObject* self, PyObject* args) {
    (void)self;
    const char* path;
    PyObject* causa_obj;
    if (!PyArg_ParseTuple(args, "sO", &path, &causa_obj)) return NULL;
    if (!PyObject_TypeCheck(causa_obj, &PyTensor_Type)) {
        PyErr_SetString(PyExc_TypeError, "expected a Tensor");
        return NULL;
    }
    PyTensorObject* t = (PyTensorObject*)causa_obj;
    if (!causa_save_tensor(path, t->tensor)) {
        PyErr_SetString(PyExc_RuntimeError, "failed to save tensor");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* py_load(PyObject* self, PyObject* args) {
    (void)self;
    const char* path;
    if (!PyArg_ParseTuple(args, "s", &path)) return NULL;

causa_arena_t* a;
causa_backend_t* b;
causa_new_arena(&a, &b);

    causa_tensor_t* t = causa_load_tensor(path, a, b);
    if (!t) {
        causa_arena_free(a);
        causa_backend_free(b);
        PyErr_SetString(PyExc_RuntimeError, "failed to load tensor");
        return NULL;
    }

    return causa_new_owned(t, a, b);
}

static PyObject* py_version(PyObject* self, PyObject* args) {
    (void)self; (void)args;
    return PyUnicode_FromFormat("%d.%d.%d", CAUSA_VERSION_MAJOR, CAUSA_VERSION_MINOR, CAUSA_VERSION_PATCH);
}

static PyObject* py_has_cuda(PyObject* self, PyObject* args) {
    (void)self; (void)args;
    causa_backend_t* backend = causa_backend_cuda_new();
    bool available = (backend != NULL);
    if (backend) causa_backend_free(backend);
    return PyBool_FromLong(available);
}

// Python bindings for the causal and memory layer.
//
// The tensor bindings alone expose the substrate; these expose the part that
// makes it a causal system: hierarchical memory, structural causal models,
// interventions, counterfactuals and causal discovery.

typedef struct {
    PyObject_HEAD
    causa_memory_t* memory;
    causa_arena_t* arena;
    causa_backend_t* backend;
    PyObject* arena_cap;
    PyObject* backend_cap;
} PyMemoryObject;

typedef struct {
    PyObject_HEAD
    causa_scm_t* scm;
    causa_graph_t* graph;
    causa_arena_t* arena;
    causa_backend_t* backend;
    PyObject* arena_cap;
    PyObject* backend_cap;
} PySCMObject;

typedef struct {
    PyObject_HEAD
    causa_counterfactual_t* cf;
    causa_arena_t* arena;
    causa_backend_t* backend;
    PyObject* arena_cap;
    PyObject* backend_cap;
    PyObject* query;
    PyObject* intervention_var;
    PyObject* intervention_val;
    PyObject* evidence;
    PyObject* evidence_vars;
} PyCounterfactualObject;

typedef struct {
    PyObject_HEAD
    causa_discovery_t* discovery;
    causa_arena_t* arena;
    PyObject* arena_cap;
} PyDiscoveryObject;

static PyTypeObject PyMemory_Type;
static PyTypeObject PySCM_Type;
static PyTypeObject PyCounterfactual_Type;
static PyTypeObject PyDiscovery_Type;

// ============================================================
// Helpers
// ============================================================

static int memory_level_from_name(const char* name, causa_memory_level_t* out) {
    if (!name) { *out = CAUSA_MEMORY_WORKING; return 0; }
    if (strcmp(name, "working") == 0)    { *out = CAUSA_MEMORY_WORKING; return 0; }
    if (strcmp(name, "episodic") == 0)   { *out = CAUSA_MEMORY_EPISODIC; return 0; }
    if (strcmp(name, "semantic") == 0)   { *out = CAUSA_MEMORY_SEMANTIC; return 0; }
    if (strcmp(name, "procedural") == 0) { *out = CAUSA_MEMORY_PROCEDURAL; return 0; }
    PyErr_Format(PyExc_ValueError, "unknown memory level '%s'", name);
    return -1;
}

static const char* memory_level_name(causa_memory_level_t level) {
    switch (level) {
        case CAUSA_MEMORY_WORKING:    return "working";
        case CAUSA_MEMORY_EPISODIC:   return "episodic";
        case CAUSA_MEMORY_SEMANTIC:   return "semantic";
        case CAUSA_MEMORY_PROCEDURAL: return "procedural";
    }
    return "working";
}

// Build a tensor from a Python sequence of floats
static causa_tensor_t* sequence_to_tensor(
    causa_arena_t* arena,
    causa_backend_t* backend,
    PyObject* seq,
    int64_t ndim
) {
    PyObject* flat = PySequence_Fast(seq, "expected a sequence of numbers");
    if (!flat) return NULL;

    Py_ssize_t count = PySequence_Fast_GET_SIZE(flat);
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;

    int64_t shape[1] = {0};
    causa_tensor_t* t = NULL;

    if (ndim == 1) {
        shape[0] = count;
        t = causa_tensor_new(arena, backend, shape, 1, &opts);
    } else {
        // Two dimensions: the outer sequence holds rows
        Py_ssize_t rows = count;
        Py_ssize_t cols = 0;
        for (Py_ssize_t i = 0; i < rows; i++) {
            PyObject* row = PySequence_Fast_GET_ITEM(flat, i);
            Py_ssize_t n = PySequence_Size(row);
            if (i == 0) cols = n;
            else if (n != cols) {
                Py_DECREF(flat);
                PyErr_SetString(PyExc_ValueError, "ragged sequences are not supported");
                return NULL;
            }
        }
        int64_t shape2[2] = {rows, cols};
        t = causa_tensor_new(arena, backend, shape2, 2, &opts);
    }

    if (!t) {
        Py_DECREF(flat);
        PyErr_SetString(PyExc_RuntimeError, "tensor allocation failed");
        return NULL;
    }

    float* dst = (float*)t->data;
    Py_ssize_t w = 0;
    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject* item = PySequence_Fast_GET_ITEM(flat, i);
        if (ndim == 1) {
            double v = PyFloat_AsDouble(item);
            if (v == -1.0 && PyErr_Occurred()) { Py_DECREF(flat); return NULL; }
            dst[w++] = (float)v;
        } else {
            PyObject* row = PySequence_Fast(item, "expected a sequence of rows");
            if (!row) { Py_DECREF(flat); return NULL; }
            for (Py_ssize_t j = 0; j < PySequence_Fast_GET_SIZE(row); j++) {
                double v = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(row, j));
                if (v == -1.0 && PyErr_Occurred()) { Py_DECREF(row); Py_DECREF(flat); return NULL; }
                dst[w++] = (float)v;
            }
            Py_DECREF(row);
        }
    }
    Py_DECREF(flat);
    return t;
}

// ============================================================
// Memory
// ============================================================

static void memory_dealloc(PyMemoryObject* self) {
    Py_XDECREF(self->arena_cap);
    Py_XDECREF(self->backend_cap);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* memory_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    return PyType_GenericNew(type, args, kwargs);
}

static int memory_init(PyMemoryObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"key_dim", "value_dim", "capacity", NULL};
    int key_dim = 64;
    int value_dim = 64;
    PyObject* capacity = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiO", kwlist, &key_dim, &value_dim, &capacity)) {
        return -1;
    }

    causa_memory_config_t cfg = CAUSA_MEMORY_CONFIG_DEFAULTS;
    cfg.key_dim = key_dim;
    cfg.value_dim = value_dim;

    if (capacity) {
        PyObject* fast = PySequence_Fast(capacity, "capacity must be a sequence");
        if (!fast) return -1;
        Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
        for (Py_ssize_t i = 0; i < n && i < CAUSA_MEMORY_MAX_LEVELS; i++) {
            cfg.capacity[i] = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(fast, i));
            if (PyErr_Occurred()) { Py_DECREF(fast); return -1; }
        }
        Py_DECREF(fast);
    }

    causa_arena_t* arena = causa_arena_new(16 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();
    if (!arena || !backend) {
        PyErr_SetString(PyExc_RuntimeError, "could not create the memory");
        return -1;
    }

    self->memory = causa_memory_new(arena, backend, &cfg);
    if (!self->memory) {
        causa_arena_free(arena);
        causa_backend_free(backend);
        PyErr_SetString(PyExc_RuntimeError, "could not create the memory");
        return -1;
    }

    self->arena = arena;
    self->backend = backend;
    // An object that is initialised a second time gives up the arena it
    // had, or the first one would never be freed.
    Py_XDECREF(self->arena_cap);
    Py_XDECREF(self->backend_cap);
    self->arena_cap = PyCapsule_New((void*)arena, ARENA_CAPSULE, arena_capsule_destructor);
    self->backend_cap = PyCapsule_New((void*)backend, BACKEND_CAPSULE, backend_capsule_destructor);
    return 0;
}

static PyObject* memory_write(PyMemoryObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"key", "value", "level", NULL};
    PyObject* key_obj;
    PyObject* value_obj;
    const char* level_name = "episodic";

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|s", kwlist, &key_obj, &value_obj, &level_name)) {
        return NULL;
    }

    causa_memory_level_t level;
    if (memory_level_from_name(level_name, &level) < 0) return NULL;

    int64_t key_dim = causa_memory_config(self->memory)->key_dim;
    int64_t value_dim = causa_memory_config(self->memory)->value_dim;

    // The key and the value are only needed until the memory has copied
    // them into a slot, so they go in a scratch arena that is freed
    // straight away. Otherwise every write would grow the memory's own
    // arena for ever.
    causa_arena_t* scratch = causa_arena_new(64 * 1024);
    if (!scratch) {
        PyErr_SetString(PyExc_RuntimeError, "could not allocate");
        return NULL;
    }

    causa_tensor_t* key = sequence_to_tensor(scratch, self->backend, key_obj, 1);
    if (!key) { causa_arena_free(scratch); return NULL; }
    causa_tensor_t* value = sequence_to_tensor(scratch, self->backend, value_obj, 1);
    if (!value) { causa_arena_free(scratch); return NULL; }

    if (causa_tensor_numel(key) != key_dim) {
        PyErr_Format(PyExc_ValueError, "key has %lld values, expected %lld",
                     (long long)causa_tensor_numel(key), (long long)key_dim);
        causa_arena_free(scratch);
        return NULL;
    }
    if (causa_tensor_numel(value) != value_dim) {
        PyErr_Format(PyExc_ValueError, "value has %lld values, expected %lld",
                     (long long)causa_tensor_numel(value), (long long)value_dim);
        causa_arena_free(scratch);
        return NULL;
    }

    causa_tensor_t* stored = causa_memory_write(self->memory, key, value, level);
    causa_arena_free(scratch);

    if (!stored) {
        PyErr_SetString(PyExc_RuntimeError, "the write failed");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject* memory_read(PyMemoryObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"key", "level", NULL};
    PyObject* key_obj;
    const char* level_name = "episodic";

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|s", kwlist, &key_obj, &level_name)) {
        return NULL;
    }

    causa_memory_level_t level;
    if (memory_level_from_name(level_name, &level) < 0) return NULL;

    // The key is only needed to look the slot up, so it goes in a scratch
    // arena that is freed straight away. Otherwise every read would grow
    // the memory's own arena for ever.
    causa_arena_t* scratch = causa_arena_new(64 * 1024);
    if (!scratch) {
        PyErr_SetString(PyExc_RuntimeError, "could not allocate");
        return NULL;
    }

    causa_tensor_t* key = sequence_to_tensor(scratch, self->backend, key_obj, 1);
    if (!key) { causa_arena_free(scratch); return NULL; }

    causa_tensor_t* found = causa_memory_read(self->memory, key, level);
    causa_arena_free(scratch);

    if (!found) Py_RETURN_NONE;

    return causa_new_internal(found, self->arena, self->backend,
                               self->arena_cap, self->backend_cap);
}

static PyObject* memory_size(PyMemoryObject* self, PyObject* args) {
    const char* level_name = NULL;
    if (!PyArg_ParseTuple(args, "|s", &level_name)) return NULL;

    causa_memory_level_t level;
    if (memory_level_from_name(level_name, &level) < 0) return NULL;

    return PyLong_FromLongLong((long long)causa_memory_size(self->memory, level));
}

static PyObject* memory_usage(PyMemoryObject* self, PyObject* args) {
    const char* level_name = NULL;
    if (!PyArg_ParseTuple(args, "|s", &level_name)) return NULL;

    causa_memory_level_t level;
    if (memory_level_from_name(level_name, &level) < 0) return NULL;

    return PyFloat_FromDouble((double)causa_memory_usage(self->memory, level));
}

static PyObject* memory_collect(PyMemoryObject* self, PyObject* args) {
    const char* level_name = "episodic";
    if (!PyArg_ParseTuple(args, "|s", &level_name)) return NULL;

    causa_memory_level_t level;
    if (memory_level_from_name(level_name, &level) < 0) return NULL;

    int64_t dim = causa_memory_config(self->memory)->key_dim;
    int64_t count = causa_memory_size(self->memory, level);
    if (count <= 0) {
        PyErr_SetString(PyExc_ValueError, "there is nothing to collect");
        return NULL;
    }

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;

    int64_t shape[] = {count, dim};
    causa_tensor_t* out = causa_tensor_new(self->arena, self->backend, shape, 2, &opts);
    if (!out) {
        PyErr_SetString(PyExc_RuntimeError, "allocation failed");
        return NULL;
    }

    int64_t gathered = causa_memory_collect(self->memory, level, out);
    if (gathered < 0) {
        PyErr_SetString(PyExc_RuntimeError, "the collection failed");
        return NULL;
    }

    return causa_new_internal(out, self->arena, self->backend,
                               self->arena_cap, self->backend_cap);
}

static PyObject* memory_consolidate(PyMemoryObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"scm", "alpha", "threshold", "iterations", NULL};
    PyObject* scm_obj = NULL;
    double alpha = 0.1;
    double threshold = 0.3;
    int iterations = 200;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|Oddi", kwlist, &scm_obj, &alpha, &threshold, &iterations)) {
        return NULL;
    }

    causa_config_t cfg = CAUSA_CONFIG_DEFAULTS;
    cfg.discovery_alpha = (float)alpha;
    cfg.discovery_threshold = (float)threshold;
    cfg.max_iterations = iterations;

    causa_scm_t* scm = NULL;
    if (scm_obj && PyObject_TypeCheck(scm_obj, &PySCM_Type)) {
        scm = ((PySCMObject*)scm_obj)->scm;
    }

    int32_t laws = causa_memory_consolidate_causal(self->memory, scm, &cfg);
    if (laws < 0) {
        PyErr_SetString(PyExc_RuntimeError, "the consolidation failed");
        return NULL;
    }
    return PyLong_FromLong(laws);
}

static PyObject* memory_find_laws(PyMemoryObject* self, PyObject* args) {
    int child;
    if (!PyArg_ParseTuple(args, "i", &child)) return NULL;

    causa_tensor_t* found[64];
    int32_t n = causa_memory_find_causal_laws(self->memory, child, found, 64);
    if (n < 0) {
        PyErr_SetString(PyExc_RuntimeError, "the search failed");
        return NULL;
    }

    PyObject* list = PyList_New(n);
    if (!list) return NULL;
    for (int32_t i = 0; i < n; i++) {
        PyObject* t = causa_new_internal(found[i], self->arena, self->backend,
                                          self->arena_cap, self->backend_cap);
        if (!t) { Py_DECREF(list); return NULL; }
        PyList_SET_ITEM(list, i, t);
    }
    return list;
}

static PyObject* memory_decay(PyMemoryObject* self, PyObject* args) {
    double dt;
    if (!PyArg_ParseTuple(args, "d", &dt)) return NULL;
    causa_memory_decay(self->memory, (float)dt);
    Py_RETURN_NONE;
}

// Reason over the memory and fill in the variables that were not observed.
//
// The observation holds one value per variable, and a NaN marks a variable
// that was not observed. The prediction comes from the causal laws in
// semantic memory and the episodes in episodic memory.
static PyObject* memory_predict(PyMemoryObject* self, PyObject* args) {
    PyObject* obs_obj;
    if (!PyArg_ParseTuple(args, "O", &obs_obj)) return NULL;

    if (!PyObject_TypeCheck(obs_obj, &PyTensor_Type)) {
        PyErr_SetString(PyExc_TypeError, "the observation must be a tensor");
        return NULL;
    }

    causa_tensor_t* obs = ((PyTensorObject*)obs_obj)->tensor;

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;

    int64_t dim = causa_memory_config(self->memory)->key_dim;
    int64_t shape[] = {dim};
    causa_tensor_t* out = causa_tensor_new(self->arena, self->backend, shape, 1, &opts);
    if (!out) {
        PyErr_SetString(PyExc_RuntimeError, "allocation failed");
        return NULL;
    }

    int32_t predicted = causa_memory_predict(self->memory, obs, out);
    if (predicted < 0) {
        PyErr_SetString(PyExc_RuntimeError, "the prediction failed");
        return NULL;
    }

    return causa_new_internal(out, self->arena, self->backend,
                                self->arena_cap, self->backend_cap);
}

// Build a model out of what the memory has learned.
//
// The causal laws in semantic memory become the structural equations, so
// the model can be intervened on and reasoned about like any other.
static PyObject* memory_build_scm(PyMemoryObject* self, PyObject* args) {
    int num_variables;
    if (!PyArg_ParseTuple(args, "i", &num_variables)) return NULL;

    PySCMObject* scm = PyObject_New(PySCMObject, &PySCM_Type);
    if (!scm) return NULL;

    causa_arena_t* arena = causa_arena_new(16 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();
    if (!arena || !backend) {
        PyErr_SetString(PyExc_RuntimeError, "could not create the model");
        Py_DECREF(scm);
        return NULL;
    }

    scm->arena = arena;
    scm->backend = backend;
    scm->arena_cap = PyCapsule_New((void*)arena, ARENA_CAPSULE, arena_capsule_destructor);
    scm->backend_cap = PyCapsule_New((void*)backend, BACKEND_CAPSULE, backend_capsule_destructor);
    scm->graph = causa_graph_new(arena, backend);
    scm->scm = causa_scm_new(arena, backend, NULL);
    if (!scm->graph || !scm->scm) {
        PyErr_SetString(PyExc_RuntimeError, "could not create the model");
        Py_DECREF(scm);
        return NULL;
    }
    scm->scm->base_graph = scm->graph;

    int32_t equations = causa_memory_build_scm(self->memory, scm->scm, num_variables);
    if (equations < 0) {
        PyErr_SetString(PyExc_RuntimeError, "the model could not be built");
        Py_DECREF(scm);
        return NULL;
    }

    return (PyObject*)scm;
}

// Learn procedures from the episodes.
//
// The episodes are grouped: episodes that resemble each other become one
// procedure, whose trigger is the centre of the group and whose action is
// the average of what those episodes led to.
static PyObject* memory_learn_procedures(PyMemoryObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"radius", NULL};
    double radius = 0.5;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|d", kwlist, &radius)) {
        return NULL;
    }

    int32_t procedures = causa_memory_learn_procedures(self->memory, (float)radius);
    if (procedures < 0) {
        PyErr_SetString(PyExc_RuntimeError, "the learning failed");
        return NULL;
    }
    return PyLong_FromLong(procedures);
}

// Store a procedure: a trigger and an action.
static PyObject* memory_store_procedure(PyMemoryObject* self, PyObject* args) {
    PyObject* trigger_obj;
    PyObject* action_obj;
    if (!PyArg_ParseTuple(args, "OO", &trigger_obj, &action_obj)) return NULL;

    if (!PyObject_TypeCheck(trigger_obj, &PyTensor_Type) ||
        !PyObject_TypeCheck(action_obj, &PyTensor_Type)) {
        PyErr_SetString(PyExc_TypeError, "the trigger and the action must be tensors");
        return NULL;
    }

    causa_tensor_t* stored = causa_memory_store_procedure(
        self->memory,
        ((PyTensorObject*)trigger_obj)->tensor,
        ((PyTensorObject*)action_obj)->tensor);
    if (!stored) {
        PyErr_SetString(PyExc_RuntimeError, "the procedure was not stored");
        return NULL;
    }

    Py_RETURN_NONE;
}

// Find the procedures that apply to a state, the nearest first.
//
// Each procedure comes back as a pair of the trigger and the action.
static PyObject* memory_find_procedures(PyMemoryObject* self, PyObject* args) {
    PyObject* state_obj;
    if (!PyArg_ParseTuple(args, "O", &state_obj)) return NULL;

    if (!PyObject_TypeCheck(state_obj, &PyTensor_Type)) {
        PyErr_SetString(PyExc_TypeError, "the state must be a tensor");
        return NULL;
    }

    causa_tensor_t* state = ((PyTensorObject*)state_obj)->tensor;

    int64_t cap = causa_memory_config(self->memory)->capacity[CAUSA_MEMORY_PROCEDURAL];
    int32_t max_procedures = (int32_t)(cap > 0 ? cap : 1);

    causa_tensor_t** triggers = (causa_tensor_t**)calloc((size_t)max_procedures, sizeof(causa_tensor_t*));
    causa_tensor_t** actions = (causa_tensor_t**)calloc((size_t)max_procedures, sizeof(causa_tensor_t*));
    if (!triggers || !actions) {
        free(triggers); free(actions);
        PyErr_SetString(PyExc_RuntimeError, "allocation failed");
        return NULL;
    }

    int32_t found = causa_memory_find_procedures(self->memory, state, triggers, actions, max_procedures);
    if (found < 0) {
        free(triggers); free(actions);
        PyErr_SetString(PyExc_RuntimeError, "the search failed");
        return NULL;
    }

    PyObject* list = PyList_New(found);
    if (!list) { free(triggers); free(actions); return NULL; }

    for (int32_t i = 0; i < found; i++) {
        PyObject* pair = PyTuple_New(2);
        if (!pair) { Py_DECREF(list); free(triggers); free(actions); return NULL; }

        PyObject* trigger = causa_new_internal(triggers[i], self->arena, self->backend,
                                                self->arena_cap, self->backend_cap);
        PyObject* action = causa_new_internal(actions[i], self->arena, self->backend,
                                               self->arena_cap, self->backend_cap);
        if (!trigger || !action) {
            Py_XDECREF(trigger); Py_XDECREF(action);
            Py_DECREF(pair); Py_DECREF(list);
            free(triggers); free(actions);
            return NULL;
        }
        PyTuple_SET_ITEM(pair, 0, trigger);
        PyTuple_SET_ITEM(pair, 1, action);
        PyList_SET_ITEM(list, i, pair);
    }

    free(triggers); free(actions);
    return list;
}

// Work out what to do to reach a goal.
//
// The state says where things stand, and the goal says what is wanted: the
// goal holds the wanted value of the variable to be changed, and NaN for the
// rest.
//
// Returns the plan, which is the change to make to each variable, and the
// number of variables the plan acts on.
static PyObject* memory_plan_intervention(PyMemoryObject* self, PyObject* args) {
    PyObject* state_obj;
    PyObject* goal_obj;
    if (!PyArg_ParseTuple(args, "OO", &state_obj, &goal_obj)) return NULL;

    if (!PyObject_TypeCheck(state_obj, &PyTensor_Type) ||
        !PyObject_TypeCheck(goal_obj, &PyTensor_Type)) {
        PyErr_SetString(PyExc_TypeError, "the state and the goal must be tensors");
        return NULL;
    }

    int64_t dim = causa_memory_config(self->memory)->key_dim;

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;

    int64_t shape[] = {dim};
    causa_tensor_t* plan = causa_tensor_new(self->arena, self->backend, shape, 1, &opts);
    if (!plan) {
        PyErr_SetString(PyExc_RuntimeError, "allocation failed");
        return NULL;
    }

    int32_t count = 0;
    if (causa_memory_plan_intervention(self->memory,
                                        ((PyTensorObject*)state_obj)->tensor,
                                        ((PyTensorObject*)goal_obj)->tensor,
                                        plan, &count) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "the plan could not be made");
        return NULL;
    }

    PyObject* result = PyTuple_New(2);
    if (!result) return NULL;

    PyObject* plan_obj = causa_new_internal(plan, self->arena, self->backend,
                                             self->arena_cap, self->backend_cap);
    if (!plan_obj) { Py_DECREF(result); return NULL; }

    PyObject* count_obj = PyLong_FromLong(count);
    if (!count_obj) { Py_DECREF(plan_obj); Py_DECREF(result); return NULL; }

    PyTuple_SET_ITEM(result, 0, plan_obj);
    PyTuple_SET_ITEM(result, 1, count_obj);

    return result;
}

static PyMethodDef memory_methods[] = {
    {"write", (PyCFunction)memory_write, METH_VARARGS | METH_KEYWORDS, "Store a key and value at a memory level"},
    {"read", (PyCFunction)memory_read, METH_VARARGS | METH_KEYWORDS, "Look up the value stored under a key"},
    {"size", (PyCFunction)memory_size, METH_VARARGS, "Number of occupied slots at a level"},
    {"usage", (PyCFunction)memory_usage, METH_VARARGS, "Fraction of a level that is occupied"},
    {"collect", (PyCFunction)memory_collect, METH_VARARGS, "Gather the stored observations into a matrix"},
    {"consolidate_causal", (PyCFunction)memory_consolidate, METH_VARARGS | METH_KEYWORDS, "Learn causal laws from the stored observations"},
    {"find_causal_laws", (PyCFunction)memory_find_laws, METH_VARARGS, "Find the causal laws recorded for a variable"},
    {"predict", (PyCFunction)memory_predict, METH_VARARGS, "Predict the variables that were not observed"},
    {"build_scm", (PyCFunction)memory_build_scm, METH_VARARGS, "Build a model from the learned causal laws"},
    {"learn_procedures", (PyCFunction)memory_learn_procedures, METH_VARARGS | METH_KEYWORDS, "Group the episodes into procedures"},
    {"store_procedure", (PyCFunction)memory_store_procedure, METH_VARARGS, "Store a procedure"},
    {"find_procedures", (PyCFunction)memory_find_procedures, METH_VARARGS, "Find the procedures that apply to a state"},
    {"plan_intervention", (PyCFunction)memory_plan_intervention, METH_VARARGS, "Work out what to do to reach a goal"},
    {"decay", (PyCFunction)memory_decay, METH_VARARGS, "Age the memories by a time step"},
    {NULL}
};

static PyTypeObject PyMemory_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "causa.Memory",
    .tp_basicsize = sizeof(PyMemoryObject),
    .tp_dealloc = (destructor)memory_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Hierarchical memory with working, episodic, semantic and procedural levels",
    .tp_methods = memory_methods,
    .tp_new = memory_new,
    .tp_init = (initproc)memory_init,
};

// ============================================================
// SCM
// ============================================================

static void scm_dealloc(PySCMObject* self) {
    Py_XDECREF(self->arena_cap);
    Py_XDECREF(self->backend_cap);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* scm_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    return PyType_GenericNew(type, args, kwargs);
}

static int scm_init(PySCMObject* self, PyObject* args, PyObject* kwargs) {
    (void)args; (void)kwargs;

    causa_arena_t* arena = causa_arena_new(16 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();
    if (!arena || !backend) {
        PyErr_SetString(PyExc_RuntimeError, "could not create the model");
        return -1;
    }

    self->arena = arena;
    self->backend = backend;
    // An object that is initialised a second time gives up the arena it
    // had, or the first one would never be freed.
    Py_XDECREF(self->arena_cap);
    Py_XDECREF(self->backend_cap);
    self->arena_cap = PyCapsule_New((void*)arena, ARENA_CAPSULE, arena_capsule_destructor);
    self->backend_cap = PyCapsule_New((void*)backend, BACKEND_CAPSULE, backend_capsule_destructor);

    self->graph = causa_graph_new(arena, backend);
    self->scm = causa_scm_new(arena, backend, NULL);
    if (!self->graph || !self->scm) {
        PyErr_SetString(PyExc_RuntimeError, "could not create the model");
        return -1;
    }
    self->scm->base_graph = self->graph;
    return 0;
}

static PyObject* scm_variable(PySCMObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"name", "size", "trainable", NULL};
    const char* name = NULL;
    int size = 1;
    int trainable = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|sip", kwlist, &name, &size, &trainable)) {
        return NULL;
    }

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    opts.requires_grad = trainable != 0;
    if (name) opts.name = name;

    int64_t shape[] = {size};
    causa_tensor_t* t = causa_graph_new_tensor(self->graph, shape, 1, &opts);
    if (!t) {
        PyErr_SetString(PyExc_RuntimeError, "allocation failed");
        return NULL;
    }
    causa_tensor_fill(t, 0.0);

    return causa_new_internal(t, self->arena, self->backend,
                               self->arena_cap, self->backend_cap);
}

static PyObject* scm_add_equation(PySCMObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"target", "parents", "noise", "coefficient", NULL};
    PyObject* target_obj;
    PyObject* parents_obj;
    PyObject* noise_obj;
    double coefficient = 1.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|d", kwlist,
                                     &target_obj, &parents_obj, &noise_obj, &coefficient)) {
        return NULL;
    }

    if (!PyObject_TypeCheck(target_obj, &PyTensor_Type) ||
        !PyObject_TypeCheck(noise_obj, &PyTensor_Type)) {
        PyErr_SetString(PyExc_TypeError, "the target and the noise must be tensors");
        return NULL;
    }

    causa_tensor_t* target = ((PyTensorObject*)target_obj)->tensor;
    causa_tensor_t* noise = ((PyTensorObject*)noise_obj)->tensor;

    PyObject* fast = PySequence_Fast(parents_obj, "the parents must be a sequence");
    if (!fast) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    if (n < 1 || n > CAUSA_MAX_OP_INPUTS) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "an equation needs between one and eight parents");
        return NULL;
    }

    causa_tensor_t* parents[CAUSA_MAX_OP_INPUTS];
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* p = PySequence_Fast_GET_ITEM(fast, i);
        if (!PyObject_TypeCheck(p, &PyTensor_Type)) {
            Py_DECREF(fast);
            PyErr_SetString(PyExc_TypeError, "the parents must be tensors");
            return NULL;
        }
        parents[i] = ((PyTensorObject*)p)->tensor;
    }
    Py_DECREF(fast);

    // The equation is built as graph operations, so the model is
    // differentiable: target = coefficient * parent + noise
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    // The intermediate product carries a gradient when the noise does, and
    // the target carries one when the product or the noise does.
    opts.requires_grad = noise->requires_grad || parents[0]->requires_grad;

    // The table of variables has room for a fixed number of them. Writing
    // past the end of it would put the equation over the counters that
    // follow it.
    if (self->scm->num_variables >= CAUSA_MAX_VARIABLES) {
        PyErr_SetString(PyExc_ValueError, "the model has as many variables as it can hold");
        return NULL;
    }

    int64_t shape[] = {1};
    causa_tensor_t* coeff = causa_graph_new_tensor(self->graph, shape, 1, &opts);
    if (!coeff) {
        PyErr_SetString(PyExc_RuntimeError, "allocation failed");
        return NULL;
    }
    causa_tensor_fill(coeff, coefficient);

    // The equation is the coefficient times the sum of the parents, plus
    // the noise. Every parent has to take part, or the model would say a
    // variable is caused by parents it does not depend on.
    causa_tensor_t* accumulator = NULL;
    for (Py_ssize_t i = 0; i < n; i++) {
        causa_tensor_t* product = causa_graph_new_tensor(self->graph, shape, 1, &opts);
        if (!product) {
            PyErr_SetString(PyExc_RuntimeError, "allocation failed");
            return NULL;
        }
        causa_graph_add_op(self->graph, CAUSA_OP_MUL, "coefficient*parent",
            (causa_tensor_t*[]){coeff, parents[i]}, 2,
            (causa_tensor_t*[]){product}, 1, NULL, NULL, NULL, 0);

        if (!accumulator) {
            accumulator = product;
            continue;
        }
        causa_tensor_t* sum = causa_graph_new_tensor(self->graph, shape, 1, &opts);
        if (!sum) {
            PyErr_SetString(PyExc_RuntimeError, "allocation failed");
            return NULL;
        }
        causa_graph_add_op(self->graph, CAUSA_OP_ADD, "sum",
            (causa_tensor_t*[]){accumulator, product}, 2,
            (causa_tensor_t*[]){sum}, 1, NULL, NULL, NULL, 0);
        accumulator = sum;
    }

    causa_graph_add_op(self->graph, CAUSA_OP_ADD, "equation",
        (causa_tensor_t*[]){accumulator, noise}, 2,
        (causa_tensor_t*[]){target}, 1, NULL, NULL, NULL, 0);

    // The target is computed from the parents and the noise, so it carries a
    // gradient whenever one of them does. Without this the gradient would
    // stop at the equation.
    if (opts.requires_grad) {
        target->requires_grad = true;
    }

    // Register the equation with the SCM
    causa_scm_var_t* var = &self->scm->variables[self->scm->num_variables];
    var->variable = target;
    var->num_parents = (int32_t)n;
    var->parents = (causa_tensor_t**)causa_arena_alloc(self->arena,
        sizeof(causa_tensor_t*) * n, CAUSA_ALIGNOF(causa_tensor_t*));
    if (!var->parents) {
        PyErr_SetString(PyExc_RuntimeError, "allocation failed");
        return NULL;
    }
    for (Py_ssize_t i = 0; i < n; i++) var->parents[i] = parents[i];
    var->noise = noise;
    var->structural_fn = NULL;
    var->structural_bwd = NULL;
    var->is_intervened = false;
    var->intervention_value = NULL;
    self->scm->num_variables++;

    // The noise has to be registered as exogenous, so that the abduction
    // step can infer it.
    causa_scm_add_exogenous(self->scm, noise);

    self->scm->built = false;
    causa_scm_build(self->scm);

    Py_RETURN_NONE;
}

static PyObject* scm_add_observed(PySCMObject* self, PyObject* args) {
    PyObject* var_obj;
    if (!PyArg_ParseTuple(args, "O", &var_obj)) return NULL;
    if (!PyObject_TypeCheck(var_obj, &PyTensor_Type)) {
        PyErr_SetString(PyExc_TypeError, "expected a tensor");
        return NULL;
    }
    causa_scm_add_observed_variable(self->scm, ((PyTensorObject*)var_obj)->tensor);
    Py_RETURN_NONE;
}

static PyObject* scm_build(PySCMObject* self, PyObject* args) {
    (void)args;
    causa_scm_build(self->scm);
    Py_RETURN_NONE;
}

static PyObject* scm_do(PySCMObject* self, PyObject* args) {
    PyObject* target_obj;
    double value;
    if (!PyArg_ParseTuple(args, "Od", &target_obj, &value)) return NULL;

    if (!PyObject_TypeCheck(target_obj, &PyTensor_Type)) {
        PyErr_SetString(PyExc_TypeError, "expected a tensor");
        return NULL;
    }
    causa_tensor_t* target = ((PyTensorObject*)target_obj)->tensor;

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;

    int64_t shape[] = {1};
    causa_tensor_t* constant = causa_tensor_new(self->arena, self->backend, shape, 1, &opts);
    causa_tensor_fill(constant, value);

    causa_intervention_t* iv = causa_intervention_new(self->arena, target, constant);
    causa_scm_t* intervened = causa_scm_do(self->scm, iv);
    if (!intervened) {
        PyErr_SetString(PyExc_RuntimeError, "the intervention failed");
        return NULL;
    }

    // The intervened model is a model of its own, so it is handed back as
    // one. It shares the arena with the model it came from.
    PySCMObject* result = PyObject_New(PySCMObject, &PySCM_Type);
    if (!result) return NULL;

    result->scm = intervened;
    result->graph = intervened->intervened_graph ? intervened->intervened_graph : self->graph;
    result->arena = self->arena;
    result->backend = self->backend;
    result->arena_cap = self->arena_cap;
    result->backend_cap = self->backend_cap;
    Py_XINCREF(self->arena_cap);
    Py_XINCREF(self->backend_cap);

    return (PyObject*)result;
}

// Run the model and report the value of every variable.
//
// The variables are given in the order they were registered, which is the
// order the observed values have to be given in.
static PyObject* scm_evaluate(PySCMObject* self, PyObject* args) {
    (void)args;

    causa_scm_evaluate(self->scm, NULL, 0);

    int32_t n = self->scm->num_variables;
    PyObject* list = PyList_New(n);
    if (!list) return NULL;

    for (int32_t i = 0; i < n; i++) {
        causa_tensor_t* v = self->scm->variables[i].variable;
        if (!v) {
            PyErr_SetString(PyExc_RuntimeError, "a variable has no value");
            Py_DECREF(list);
            return NULL;
        }
        PyObject* t = causa_new_internal(v, self->arena, self->backend,
                                          self->arena_cap, self->backend_cap);
        if (!t) { Py_DECREF(list); return NULL; }
        PyList_SET_ITEM(list, i, t);
    }

    return list;
}

static PyObject* scm_sample(PySCMObject* self, PyObject* args) {
    (void)args;
    causa_scm_sample(self->scm, NULL, 0);
    Py_RETURN_NONE;
}

static PyMethodDef scm_methods[] = {
    {"variable", (PyCFunction)scm_variable, METH_VARARGS | METH_KEYWORDS, "Create a variable in the model"},
    {"add_equation", (PyCFunction)scm_add_equation, METH_VARARGS | METH_KEYWORDS, "Add a structural equation"},
    {"add_observed_variable", (PyCFunction)scm_add_observed, METH_VARARGS, "Register a variable that has no equation"},
    {"build", (PyCFunction)scm_build, METH_VARARGS, "Order the model for evaluation"},
    {"do", (PyCFunction)scm_do, METH_VARARGS, "Apply an intervention"},
    {"evaluate", (PyCFunction)scm_evaluate, METH_VARARGS, "Run the model and report the value of every variable"},
    {"sample", (PyCFunction)scm_sample, METH_VARARGS, "Draw fresh noise and evaluate"},
    {NULL}
};

static PyTypeObject PySCM_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "causa.SCM",
    .tp_basicsize = sizeof(PySCMObject),
    .tp_dealloc = (destructor)scm_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "A structural causal model",
    .tp_methods = scm_methods,
    .tp_new = scm_new,
    .tp_init = (initproc)scm_init,
};

// ============================================================
// Discovery
// ============================================================

static void discovery_dealloc(PyDiscoveryObject* self) {
    Py_XDECREF(self->arena_cap);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* discovery_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    return PyType_GenericNew(type, args, kwargs);
}

static int discovery_init(PyDiscoveryObject* self, PyObject* args, PyObject* kwargs) {
    (void)args; (void)kwargs;
    // An object that is initialised a second time gives up what it had,
    // or the old arena would never be released.
    Py_XDECREF(self->arena_cap);
    self->arena = causa_arena_new(1024 * 1024);
    self->arena_cap = PyCapsule_New((void*)self->arena, ARENA_CAPSULE, arena_capsule_destructor);
    self->discovery = NULL;
    return 0;
}

static PyObject* discovery_parents(PyDiscoveryObject* self, PyObject* args) {
    int var_idx;
    if (!PyArg_ParseTuple(args, "i", &var_idx)) return NULL;

    int32_t parents[CAUSA_MAX_PARENTS];
    int32_t n = causa_get_parents(self->discovery, var_idx, parents, CAUSA_MAX_PARENTS);
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "no such variable");
        return NULL;
    }

    PyObject* list = PyList_New(n);
    if (!list) return NULL;
    for (int32_t i = 0; i < n; i++) {
        PyObject* v = PyLong_FromLong(parents[i]);
        if (!v) { Py_DECREF(list); return NULL; }
        PyList_SET_ITEM(list, i, v);
    }
    return list;
}

static PyObject* discovery_confidence(PyDiscoveryObject* self, PyObject* args) {
    int var_idx;
    if (!PyArg_ParseTuple(args, "i", &var_idx)) return NULL;
    if (var_idx < 0 || var_idx >= self->discovery->num_variables) {
        PyErr_SetString(PyExc_ValueError, "no such variable");
        return NULL;
    }

    PyObject* dict = PyDict_New();
    if (!dict) return NULL;
    for (int32_t i = 0; i < self->discovery->num_variables; i++) {
        if (self->discovery->adjacency[var_idx][i]) {
            PyObject* v = PyFloat_FromDouble((double)self->discovery->confidence[var_idx][i]);
            if (!v) { Py_DECREF(dict); return NULL; }
            PyDict_SetItemString(dict, memory_level_name((causa_memory_level_t)0), v);
            Py_DECREF(v);
        }
    }
    return dict;
}

static PyMethodDef discovery_methods[] = {
    {"parents", (PyCFunction)discovery_parents, METH_VARARGS, "The discovered parents of a variable"},
    {NULL}
};

static PyTypeObject PyDiscovery_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "causa.Discovery",
    .tp_basicsize = sizeof(PyDiscoveryObject),
    .tp_dealloc = (destructor)discovery_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "The result of causal discovery",
    .tp_methods = discovery_methods,
    .tp_new = discovery_new,
    .tp_init = (initproc)discovery_init,
};

// ============================================================
// Counterfactual
// ============================================================

static void counterfactual_dealloc(PyCounterfactualObject* self) {
    Py_XDECREF(self->arena_cap);
    Py_XDECREF(self->backend_cap);
    Py_XDECREF(self->query);
    Py_XDECREF(self->intervention_var);
    Py_XDECREF(self->intervention_val);
    Py_XDECREF(self->evidence);
    Py_XDECREF(self->evidence_vars);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* counterfactual_new(PyTypeObject* type, PyObject* args, PyObject* kwargs) {
    return PyType_GenericNew(type, args, kwargs);
}

static int counterfactual_init(PyCounterfactualObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"query", "intervention_var", "intervention_val", "evidence", NULL};
    PyObject* query_obj;
    PyObject* var_obj;
    PyObject* val_obj;
    PyObject* evidence_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO", kwlist,
                                     &query_obj, &var_obj, &val_obj, &evidence_obj)) {
        return -1;
    }

    if (!PyObject_TypeCheck(query_obj, &PyTensor_Type) ||
        !PyObject_TypeCheck(var_obj, &PyTensor_Type) ||
        !PyObject_TypeCheck(val_obj, &PyTensor_Type) ||
        !PyObject_TypeCheck(evidence_obj, &PyTensor_Type)) {
        PyErr_SetString(PyExc_TypeError, "all four arguments must be tensors");
        return -1;
    }

    causa_arena_t* arena = causa_arena_new(4 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();
    if (!arena || !backend) {
        PyErr_SetString(PyExc_RuntimeError, "could not create the counterfactual");
        return -1;
    }

    causa_counterfactual_t* cf = causa_counterfactual_new(arena,
        ((PyTensorObject*)query_obj)->tensor,
        ((PyTensorObject*)var_obj)->tensor,
        ((PyTensorObject*)val_obj)->tensor,
        ((PyTensorObject*)evidence_obj)->tensor);
    if (!cf) {
        causa_arena_free(arena);
        causa_backend_free(backend);
        PyErr_SetString(PyExc_RuntimeError, "could not create the counterfactual");
        return -1;
    }

    // An object that is initialised a second time gives up what it had,
    // or the old arena and the old references would never be released.
    Py_XDECREF(self->arena_cap);
    Py_XDECREF(self->backend_cap);
    Py_XDECREF(self->query);
    Py_XDECREF(self->intervention_var);
    Py_XDECREF(self->intervention_val);
    Py_XDECREF(self->evidence);
    Py_XDECREF(self->evidence_vars);

    self->cf = cf;
    self->arena = arena;
    self->backend = backend;
    self->arena_cap = PyCapsule_New((void*)arena, ARENA_CAPSULE, arena_capsule_destructor);
    self->backend_cap = PyCapsule_New((void*)backend, BACKEND_CAPSULE, backend_capsule_destructor);
    self->query = query_obj; Py_INCREF(query_obj);
    self->intervention_var = var_obj; Py_INCREF(var_obj);
    self->intervention_val = val_obj; Py_INCREF(val_obj);
    self->evidence = evidence_obj; Py_INCREF(evidence_obj);
    self->evidence_vars = NULL;
    return 0;
}

static PyObject* counterfactual_set_evidence(PyCounterfactualObject* self, PyObject* args) {
    PyObject* list;
    if (!PyArg_ParseTuple(args, "O", &list)) return NULL;

    PyObject* fast = PySequence_Fast(list, "expected a sequence of tensors");
    if (!fast) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);

    causa_tensor_t** observed = (causa_tensor_t**)causa_arena_alloc(self->arena,
        sizeof(causa_tensor_t*) * (n ? n : 1), CAUSA_ALIGNOF(causa_tensor_t*));
    if (!observed) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_RuntimeError, "could not allocate");
        return NULL;
    }

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* item = PySequence_Fast_GET_ITEM(fast, i);
        if (!PyObject_TypeCheck(item, &PyTensor_Type)) {
            Py_DECREF(fast);
            PyErr_SetString(PyExc_TypeError, "the observed values must be tensors");
            return NULL;
        }
        observed[i] = ((PyTensorObject*)item)->tensor;
    }

    causa_counterfactual_set_evidence(self->cf, observed, (int32_t)n);

    // Keep the tensors alive for as long as the counterfactual is.
    //
    // The list is kept, not released and then used. PySequence_Fast
    // returns a list of its own when the argument is not already a list,
    // and that list would have been freed by now. The reference it came
    // with is the one that is kept.
    Py_XDECREF(self->evidence_vars);
    self->evidence_vars = fast;
    Py_RETURN_NONE;
}

static PyObject* counterfactual_compute(PyCounterfactualObject* self, PyObject* args) {
    PyObject* scm_obj;
    if (!PyArg_ParseTuple(args, "O", &scm_obj)) return NULL;
    if (!PyObject_TypeCheck(scm_obj, &PySCM_Type)) {
        PyErr_SetString(PyExc_TypeError, "expected an SCM");
        return NULL;
    }

    causa_scm_t* scm = ((PySCMObject*)scm_obj)->scm;
    causa_counterfactual_compute(scm, self->cf);

    if (!self->cf->result) {
        PyErr_SetString(PyExc_RuntimeError, "the counterfactual produced no result");
        return NULL;
    }

    return causa_new_internal(self->cf->result, self->arena, self->backend,
                               self->arena_cap, self->backend_cap);
}

static PyObject* counterfactual_result(PyCounterfactualObject* self, void* closure) {
    (void)closure;
    if (!self->cf || !self->cf->result) Py_RETURN_NONE;
    return causa_new_internal(self->cf->result, self->arena, self->backend,
                               self->arena_cap, self->backend_cap);
}

static PyGetSetDef counterfactual_getset[] = {
    {"result", (getter)counterfactual_result, NULL, "The counterfactual outcome", NULL},
    {NULL}
};

static PyMethodDef counterfactual_methods[] = {
    {"set_evidence", (PyCFunction)counterfactual_set_evidence, METH_VARARGS, "Give the observed value of every variable"},
    {"compute", (PyCFunction)counterfactual_compute, METH_VARARGS, "Run the counterfactual"},
    {NULL}
};

static PyTypeObject PyCounterfactual_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "causa.Counterfactual",
    .tp_basicsize = sizeof(PyCounterfactualObject),
    .tp_dealloc = (destructor)counterfactual_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "A counterfactual query",
    .tp_methods = counterfactual_methods,
    .tp_getset = counterfactual_getset,
    .tp_new = counterfactual_new,
    .tp_init = (initproc)counterfactual_init,
};

// ============================================================
// Module-level causal entry points
// ============================================================

static PyObject* py_discover(PyObject* self, PyObject* args, PyObject* kwargs) {
    (void)self;
    static char* kwlist[] = {"data", "alpha", "threshold", "iterations", NULL};
    PyObject* data_obj;
    double alpha = 0.1;
    double threshold = 0.3;
    int iterations = 200;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|ddi", kwlist,
                                     &data_obj, &alpha, &threshold, &iterations)) {
        return NULL;
    }

    if (!PyObject_TypeCheck(data_obj, &PyTensor_Type)) {
        PyErr_SetString(PyExc_TypeError, "the data must be a tensor");
        return NULL;
    }
    PyTensorObject* data = (PyTensorObject*)data_obj;
    if (data->tensor->ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "the data must be a matrix of samples and variables");
        return NULL;
    }

    causa_config_t cfg = CAUSA_CONFIG_DEFAULTS;
    cfg.discovery_alpha = (float)alpha;
    cfg.discovery_threshold = (float)threshold;
    cfg.max_iterations = iterations;

    causa_discovery_t* disc = causa_discover(data->tensor, NULL, NULL, &cfg);
    if (!disc) {
        PyErr_SetString(PyExc_RuntimeError, "the discovery failed");
        return NULL;
    }

    // The discovery is made in the arena of the data it was made from, so
    // it is copied into one of its own. Otherwise it would die with the
    // data, and the result would point at memory that is gone.
    causa_arena_t* arena = causa_arena_new(64 * 1024);
    if (!arena) {
        PyErr_SetString(PyExc_RuntimeError, "could not create the workspace");
        return NULL;
    }

    causa_discovery_t* copy = (causa_discovery_t*)causa_arena_alloc(
        arena, sizeof(causa_discovery_t), CAUSA_ALIGNOF(causa_discovery_t));
    if (!copy) {
        causa_arena_free(arena);
        PyErr_SetString(PyExc_RuntimeError, "could not create the workspace");
        return NULL;
    }
    memcpy(copy, disc, sizeof(causa_discovery_t));

    PyDiscoveryObject* result = PyObject_New(PyDiscoveryObject, &PyDiscovery_Type);
    if (!result) {
        causa_arena_free(arena);
        return NULL;
    }
    result->discovery = copy;
    result->arena = arena;
    result->arena_cap = PyCapsule_New((void*)arena, ARENA_CAPSULE, arena_capsule_destructor);
    return (PyObject*)result;
}

static PyMethodDef causal_methods[] = {
    {"discover", (PyCFunction)py_discover, METH_VARARGS | METH_KEYWORDS, "Learn a causal structure from data"},
    {NULL}
};

static PyMethodDef module_methods[] = {
    {"zeros", (PyCFunction)py_zeros, METH_VARARGS | METH_KEYWORDS, "Create a zero-filled tensor"},
    {"full", (PyCFunction)py_full, METH_VARARGS | METH_KEYWORDS, "Create a tensor filled with a value"},
    {"randn", (PyCFunction)py_randn, METH_VARARGS | METH_KEYWORDS, "Create a tensor with standard normal values"},
    {"tensor", (PyCFunction)py_tensor, METH_VARARGS | METH_KEYWORDS, "Create a tensor from Python data"},
    {"save", (PyCFunction)py_save, METH_VARARGS, "Save a tensor to disk"},
    {"load", (PyCFunction)py_load, METH_VARARGS, "Load a tensor from disk"},
    {"version", (PyCFunction)py_version, METH_NOARGS, "Return the library version"},
    {"has_cuda", (PyCFunction)py_has_cuda, METH_NOARGS, "Return True if a CUDA device is available"},
    {"discover", (PyCFunction)py_discover, METH_VARARGS | METH_KEYWORDS, "Learn a causal structure from data"},
    {NULL}
};

static void module_free(void* m) {
    (void)m;
    tape_clear();
}

static struct PyModuleDef causa_module = {
    PyModuleDef_HEAD_INIT,
    "causa",
    "Tensor computation with differentiable memory and causal reasoning",
    -1,
    module_methods,
    NULL, NULL, NULL, module_free
};

PyMODINIT_FUNC PyInit_causa(void) {
    if (PyType_Ready(&PyTensor_Type) < 0) return NULL;
    if (PyType_Ready(&PyMemory_Type) < 0) return NULL;
    if (PyType_Ready(&PySCM_Type) < 0) return NULL;
    if (PyType_Ready(&PyCounterfactual_Type) < 0) return NULL;
    if (PyType_Ready(&PyDiscovery_Type) < 0) return NULL;

    PyObject* m = PyModule_Create(&causa_module);
    if (!m) return NULL;

    Py_INCREF(&PyTensor_Type);
    PyModule_AddObject(m, "Tensor", (PyObject*)&PyTensor_Type);

    Py_INCREF(&PyMemory_Type);
    PyModule_AddObject(m, "Memory", (PyObject*)&PyMemory_Type);
    Py_INCREF(&PySCM_Type);
    PyModule_AddObject(m, "SCM", (PyObject*)&PySCM_Type);
    Py_INCREF(&PyCounterfactual_Type);
    PyModule_AddObject(m, "Counterfactual", (PyObject*)&PyCounterfactual_Type);
    Py_INCREF(&PyDiscovery_Type);
    PyModule_AddObject(m, "Discovery", (PyObject*)&PyDiscovery_Type);

    PyObject* version_str = PyUnicode_FromFormat("%d.%d.%d",
        CAUSA_VERSION_MAJOR, CAUSA_VERSION_MINOR, CAUSA_VERSION_PATCH);
    if (version_str) {
        PyModule_AddObject(m, "__version__", version_str);
    }

    return m;
}


