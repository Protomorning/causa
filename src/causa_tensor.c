#include "causa_tensor.h"
#include "causa_arena.h"
#include "causa_platform.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdarg.h>

static const size_t dtype_sizes[CAUSA_DTYPE_COUNT] = {
    [CAUSA_DTYPE_F64] = 8,
    [CAUSA_DTYPE_F32] = 4,
    [CAUSA_DTYPE_F16] = 2,
    [CAUSA_DTYPE_BF16] = 2,
    [CAUSA_DTYPE_I64] = 8,
    [CAUSA_DTYPE_I32] = 4,
    [CAUSA_DTYPE_I16] = 2,
    [CAUSA_DTYPE_I8] = 1,
    [CAUSA_DTYPE_U8] = 1,
    [CAUSA_DTYPE_BOOL] = 1,
    [CAUSA_DTYPE_SPIKE] = 1,
};

size_t causa_dtype_size(causa_dtype_t dtype) {
    if (dtype >= 0 && dtype < CAUSA_DTYPE_COUNT) {
        return dtype_sizes[dtype];
    }
    return 0;
}

static const char* dtype_names[CAUSA_DTYPE_COUNT] = {
    "f64", "f32", "f16", "bf16", "i64", "i32", "i16", "i8", "u8", "bool", "spike"
};

const char* causa_dtype_name(causa_dtype_t dtype) {
    if (dtype >= 0 && dtype < CAUSA_DTYPE_COUNT) {
        return dtype_names[dtype];
    }
    return "unknown";
}

static const char* device_names[CAUSA_DEVICE_COUNT] = {
    "cpu", "cuda", "metal", "vulkan", "neuromorphic", "bare"
};

const char* causa_device_name(causa_device_type_t device) {
    if (device >= 0 && device < CAUSA_DEVICE_COUNT) {
        return device_names[device];
    }
    return "unknown";
}

// ===== Element access =====
//
// The library has eleven element types but only float32 is used by the
// kernels that do arithmetic, so every kernel goes through these two
// helpers to read and write elements. A kernel that handles float32
// directly has to say so, because the gradient of an operation on a
// float64 tensor is a float64 tensor.

// Half-precision conversions.
//
// A half float is stored as an unsigned integer with the sign in bit 15,
// the exponent in bits 14 to 10 and the mantissa in bits 9 to 0, which is
// the layout the IEEE 754 binary16 format uses.
static uint16_t float_to_half(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));

    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;

    if (((bits >> 23) & 0xffu) == 0xffu) {
        // Infinity or a not-a-number
        return (uint16_t)(sign | 0x7c00u | (mantissa ? 0x200u : 0u));
    }

    if (exponent >= 0x1f) {
        // Too large: it becomes an infinity
        return (uint16_t)(sign | 0x7c00u);
    }

    if (exponent <= 0) {
        if (exponent < -10) {
            // Too small to be represented
            return (uint16_t)sign;
        }
        // A subnormal half float
        mantissa |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exponent);
        uint32_t sub = mantissa >> shift;
        if ((mantissa >> (shift - 1)) & 1u) sub++;   // round to nearest
        return (uint16_t)(sign | sub);
    }

    uint32_t half = (uint32_t)(exponent << 10) | (mantissa >> 13);
    if ((mantissa & 0x1fffu) > 0x1000u) half++;     // round to nearest
    return (uint16_t)(sign | half);
}

static float half_to_float(uint16_t half) {
    uint32_t sign = (uint32_t)(half & 0x8000u) << 16;
    uint32_t exponent = (half >> 10) & 0x1fu;
    uint32_t mantissa = half & 0x3ffu;

    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            // A subnormal half float is a normal float with an exponent of
            // its own
            int32_t e = -1;
            do {
                mantissa <<= 1;
                e++;
            } while (!(mantissa & 0x400u));
            mantissa &= 0x3ffu;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

// A bfloat16 keeps the float32 exponent and loses the low mantissa bits.
static uint16_t float_to_bfloat16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return (uint16_t)((bits >> 16) | 0x40u);   // a not-a-number
    }
    uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(rounded >> 16);
}

static float bfloat16_to_float(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

// Read one element of a tensor as a float.
//
// The index counts elements of the tensor's own type, so the caller has to
// have already worked out where the element sits.
float causa_tensor_load_f32(const causa_tensor_t* t, int64_t index) {
    if (!t) return 0.0f;
    const char* base = (const char*)t->data;
    switch (t->dtype) {
        case CAUSA_DTYPE_F32:  return *(const float*)(base + index * 4);
        case CAUSA_DTYPE_F64:  return (float)*(const double*)(base + index * 8);
        case CAUSA_DTYPE_F16:  return half_to_float(*(const uint16_t*)(base + index * 2));
        case CAUSA_DTYPE_BF16: return bfloat16_to_float(*(const uint16_t*)(base + index * 2));
        case CAUSA_DTYPE_I64:  return (float)*(const int64_t*)(base + index * 8);
        case CAUSA_DTYPE_I32:  return (float)*(const int32_t*)(base + index * 4);
        case CAUSA_DTYPE_I16:  return (float)*(const int16_t*)(base + index * 2);
        case CAUSA_DTYPE_I8:   return (float)*(const int8_t*)(base + index);
        case CAUSA_DTYPE_U8:   return (float)*(const uint8_t*)(base + index);
        case CAUSA_DTYPE_BOOL: return *(const uint8_t*)(base + index) ? 1.0f : 0.0f;
        default: return 0.0f;
    }
}

// Write one element of a tensor from a float.
void causa_tensor_store_f32(causa_tensor_t* t, int64_t index, float value) {
    if (!t) return;
    char* base = (char*)t->data;
    switch (t->dtype) {
        case CAUSA_DTYPE_F32:  *(float*)(base + index * 4) = value; break;
        case CAUSA_DTYPE_F64:  *(double*)(base + index * 8) = (double)value; break;
        case CAUSA_DTYPE_F16:  *(uint16_t*)(base + index * 2) = float_to_half(value); break;
        case CAUSA_DTYPE_BF16: *(uint16_t*)(base + index * 2) = float_to_bfloat16(value); break;
        case CAUSA_DTYPE_I64:  *(int64_t*)(base + index * 8) = (int64_t)value; break;
        case CAUSA_DTYPE_I32:  *(int32_t*)(base + index * 4) = (int32_t)value; break;
        case CAUSA_DTYPE_I16:  *(int16_t*)(base + index * 2) = (int16_t)value; break;
        case CAUSA_DTYPE_I8:   *(int8_t*)(base + index) = (int8_t)value; break;
        case CAUSA_DTYPE_U8:   *(uint8_t*)(base + index) = (uint8_t)value; break;
        case CAUSA_DTYPE_BOOL: *(uint8_t*)(base + index) = value != 0.0f ? 1 : 0; break;
        default: break;
    }
}

static void causa_default_dealloc(causa_tensor_t* tensor) {
    if (tensor->data && tensor->arena) {
        causa_arena_t* arena = tensor->arena;
        (void)arena;
    }
}

static void compute_strides_contiguous(int64_t* strides, const int64_t* shape, int32_t ndim, size_t elem_size) {
    if (ndim == 0) return;
    strides[ndim - 1] = elem_size;
    for (int32_t i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

static void compute_strides_channels_last(int64_t* strides, const int64_t* shape, int32_t ndim, size_t elem_size) {
    if (ndim < 3) {
        compute_strides_contiguous(strides, shape, ndim, elem_size);
        return;
    }
    strides[ndim - 1] = elem_size;
    strides[ndim - 3] = shape[ndim - 1] * elem_size;
    strides[ndim - 2] = shape[ndim - 3] * strides[ndim - 3];
    for (int32_t i = ndim - 4; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

causa_tensor_t* causa_tensor_new(
    causa_arena_t* arena,
    causa_backend_t* backend,
    const int64_t* shape,
    int32_t ndim,
    const causa_tensor_options_t* options
) {
    if (!arena || !shape || ndim < 0 || ndim > CAUSA_MAX_DIMS) return NULL;

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    if (options) opts = *options;

    size_t elem_size = causa_dtype_size(opts.dtype);
    int64_t numel = 1;
    for (int32_t i = 0; i < ndim; i++) {
        if (shape[i] < 0) return NULL;
        numel *= shape[i];
    }
    size_t nbytes = numel * elem_size;

    void* data = causa_arena_alloc(arena, nbytes, 64);
    if (!data) return NULL;

    causa_tensor_t* tensor = (causa_tensor_t*)causa_arena_alloc(arena, sizeof(causa_tensor_t), CAUSA_ALIGNOF(causa_tensor_t));
    if (!tensor) return NULL;

    tensor->data = data;
    tensor->grad = NULL;
    tensor->arena = arena;
    tensor->backend = backend;
    tensor->device = opts.device;
    tensor->dtype = opts.dtype;
    tensor->layout = opts.layout;
    tensor->ndim = ndim;
    tensor->offset = 0;
    tensor->refcount = 1;
    tensor->requires_grad = opts.requires_grad;
    tensor->is_view = false;
    tensor->dealloc = causa_default_dealloc;
    tensor->autograd_ctx = NULL;

    if (opts.name) {
        strncpy(tensor->name, opts.name, CAUSA_MAX_NAME - 1);
        tensor->name[CAUSA_MAX_NAME - 1] = '\0';
    } else {
        tensor->name[0] = '\0';
    }

    if (ndim > 0) memcpy(tensor->shape, shape, ndim * sizeof(int64_t));

    if (opts.layout == CAUSA_LAYOUT_CHANNELS_LAST) {
        compute_strides_channels_last(tensor->strides, shape, ndim, elem_size);
    } else {
        compute_strides_contiguous(tensor->strides, shape, ndim, elem_size);
    }

    return tensor;
}

causa_tensor_t* causa_tensor_new_from_data(
    causa_arena_t* arena,
    causa_backend_t* backend,
    void* data,
    const int64_t* shape,
    int32_t ndim,
    const causa_tensor_options_t* options
) {
    causa_tensor_t* tensor = causa_tensor_new(arena, backend, shape, ndim, options);
    if (!tensor) return NULL;

    size_t nbytes = causa_tensor_nbytes(tensor);
    memcpy(tensor->data, data, nbytes);
    return tensor;
}

causa_tensor_t* causa_tensor_view(
    causa_tensor_t* base,
    const int64_t* shape,
    int32_t ndim,
    const int64_t* strides,
    int64_t offset
) {
    if (!base || !shape || ndim <= 0 || ndim > CAUSA_MAX_DIMS) return NULL;

    int64_t numel = 1;
    for (int32_t i = 0; i < ndim; i++) {
        if (shape[i] < 0) return NULL;
        numel *= shape[i];
    }

    causa_tensor_t* view = (causa_tensor_t*)causa_arena_alloc(base->arena, sizeof(causa_tensor_t), CAUSA_ALIGNOF(causa_tensor_t));
    if (!view) return NULL;

    *view = *base;
    view->refcount = 1;
    view->is_view = true;
    view->ndim = ndim;

    // The view starts at the given offset into the buffer of the tensor it
    // views, so the pointer is moved rather than the offset kept beside
    // it. Every kernel indexes from the data pointer, so this is the one
    // place the offset has to be applied.
    view->data = (char*)base->data + offset;
    view->offset = 0;

    // The view has a gradient of its own. Sharing the gradient of the
    // tensor it views would make the two accumulate into the same buffer.
    view->grad = NULL;

    memcpy(view->shape, shape, ndim * sizeof(int64_t));
    if (strides) {
        memcpy(view->strides, strides, ndim * sizeof(int64_t));
    } else {
        compute_strides_contiguous(view->strides, shape, ndim, causa_dtype_size(base->dtype));
    }

    causa_tensor_retain(base);
    return view;
}

causa_tensor_t* causa_tensor_slice(
    causa_tensor_t* tensor,
    const int64_t* starts,
    const int64_t* ends,
    const int64_t* steps,
    int32_t ndim
) {
    if (!tensor || ndim != tensor->ndim) return NULL;

    int64_t shape[CAUSA_MAX_DIMS];
    int64_t strides[CAUSA_MAX_DIMS];
    int64_t offset = tensor->offset;

    for (int32_t i = 0; i < ndim; i++) {
        int64_t start = starts ? starts[i] : 0;
        int64_t end = ends ? ends[i] : tensor->shape[i];
        int64_t step = steps ? steps[i] : 1;

        if (step == 0) return NULL;

        if (start < 0) start += tensor->shape[i];
        if (start < 0) start = 0;
        if (start > tensor->shape[i]) start = tensor->shape[i];

        if (step > 0) {
            if (end < 0) end += tensor->shape[i];
            if (end < 0) end = 0;
            if (end > tensor->shape[i]) end = tensor->shape[i];
            if (start >= end) {
                shape[i] = 0;
            } else {
                shape[i] = (end - start + step - 1) / step;
            }
        } else {
            // A negative step counts backwards, so the end of the slice is
            // below the start. When it is not given, the slice runs to the
            // beginning of the dimension.
            if (ends) {
                if (end < 0) end += tensor->shape[i];
                if (end < 0) end = -1;
                if (end >= tensor->shape[i]) end = tensor->shape[i] - 1;
            } else {
                end = -1;
            }
            if (start <= end) {
                shape[i] = 0;
            } else {
                shape[i] = (start - end + (-step) - 1) / (-step);
            }
        }

        offset += start * tensor->strides[i];
        strides[i] = tensor->strides[i] * step;
    }

    return causa_tensor_view(tensor, shape, ndim, strides, offset);
}

causa_tensor_t* causa_tensor_transpose(causa_tensor_t* tensor, int32_t dim0, int32_t dim1) {
    if (!tensor || dim0 < 0 || dim0 >= tensor->ndim || dim1 < 0 || dim1 >= tensor->ndim) return NULL;
    if (dim0 == dim1) return causa_tensor_view(tensor, tensor->shape, tensor->ndim, tensor->strides, tensor->offset);

    int64_t shape[CAUSA_MAX_DIMS];
    int64_t strides[CAUSA_MAX_DIMS];
    memcpy(shape, tensor->shape, tensor->ndim * sizeof(int64_t));
    memcpy(strides, tensor->strides, tensor->ndim * sizeof(int64_t));

    int64_t tmp_shape = shape[dim0];
    shape[dim0] = shape[dim1];
    shape[dim1] = tmp_shape;

    int64_t tmp_stride = strides[dim0];
    strides[dim0] = strides[dim1];
    strides[dim1] = tmp_stride;

    return causa_tensor_view(tensor, shape, tensor->ndim, strides, tensor->offset);
}

causa_tensor_t* causa_tensor_permute(causa_tensor_t* tensor, const int32_t* dims, int32_t ndim) {
    if (!tensor || !dims || ndim != tensor->ndim) return NULL;

    int64_t shape[CAUSA_MAX_DIMS];
    int64_t strides[CAUSA_MAX_DIMS];
    bool seen[CAUSA_MAX_DIMS] = {false};

    for (int32_t i = 0; i < ndim; i++) {
        if (dims[i] < 0 || dims[i] >= ndim) return NULL;
        // The dimensions have to be a permutation. Two dimensions taking
        // the same one would make the view overlap itself.
        if (seen[dims[i]]) return NULL;
        seen[dims[i]] = true;
        shape[i] = tensor->shape[dims[i]];
        strides[i] = tensor->strides[dims[i]];
    }

    return causa_tensor_view(tensor, shape, ndim, strides, tensor->offset);
}

causa_tensor_t* causa_tensor_reshape(causa_tensor_t* tensor, const int64_t* shape, int32_t ndim) {
    if (!tensor || !causa_tensor_is_contiguous(tensor)) return NULL;

    int64_t numel_old = causa_tensor_numel(tensor);
    int64_t numel_new = 1;
    for (int32_t i = 0; i < ndim; i++) {
        if (shape[i] < 0) return NULL;
        numel_new *= shape[i];
    }
    if (numel_old != numel_new) return NULL;

    return causa_tensor_view(tensor, shape, ndim, NULL, tensor->offset);
}

causa_tensor_t* causa_tensor_squeeze(causa_tensor_t* tensor, int32_t dim) {
    if (!tensor || dim < 0 || dim >= tensor->ndim || tensor->shape[dim] != 1) return NULL;

    int64_t shape[CAUSA_MAX_DIMS];
    int64_t strides[CAUSA_MAX_DIMS];
    int32_t j = 0;
    for (int32_t i = 0; i < tensor->ndim; i++) {
        if (i != dim) {
            shape[j] = tensor->shape[i];
            strides[j] = tensor->strides[i];
            j++;
        }
    }
    return causa_tensor_view(tensor, shape, tensor->ndim - 1, strides, tensor->offset);
}

causa_tensor_t* causa_tensor_unsqueeze(causa_tensor_t* tensor, int32_t dim) {
    if (!tensor || dim < 0 || dim > tensor->ndim) return NULL;

    int64_t shape[CAUSA_MAX_DIMS];
    int64_t strides[CAUSA_MAX_DIMS];
    int32_t j = 0;
    for (int32_t i = 0; i <= tensor->ndim; i++) {
        if (i == dim) {
            shape[i] = 1;
            strides[i] = 0;
        } else {
            shape[i] = tensor->shape[j];
            strides[i] = tensor->strides[j];
            j++;
        }
    }
    return causa_tensor_view(tensor, shape, tensor->ndim + 1, strides, tensor->offset);
}

void causa_tensor_retain(causa_tensor_t* tensor) {
    if (tensor) {
        causa_atomic_inc(&tensor->refcount);
    }
}

void causa_tensor_release(causa_tensor_t* tensor) {
    if (!tensor) return;
    if (causa_atomic_dec(&tensor->refcount) == 0) {
        if (tensor->dealloc) tensor->dealloc(tensor);
        if (tensor->grad) causa_tensor_release((causa_tensor_t*)tensor->grad);
    }
}

void causa_tensor_zero_grad(causa_tensor_t* tensor) {
    if (!tensor || !tensor->grad) return;
    size_t nbytes = causa_tensor_nbytes((causa_tensor_t*)tensor->grad);
    memset(((causa_tensor_t*)tensor->grad)->data, 0, nbytes);
}

void causa_tensor_backward(causa_tensor_t* tensor, causa_tensor_t* grad_output) {
    (void)tensor;
    (void)grad_output;
}

bool causa_tensor_is_contiguous(const causa_tensor_t* tensor) {
    if (!tensor) return false;

    size_t elem_size = causa_dtype_size(tensor->dtype);
    int64_t expected_strides[CAUSA_MAX_DIMS];
    compute_strides_contiguous(expected_strides, tensor->shape, tensor->ndim, elem_size);

    // A view can be contiguous too: a slice of a row, or a reshape, both
    // keep the elements next to each other. Only the strides decide.
    for (int32_t i = 0; i < tensor->ndim; i++) {
        if (tensor->strides[i] != expected_strides[i]) return false;
    }
    return true;
}

int64_t causa_tensor_numel(const causa_tensor_t* tensor) {
    if (!tensor) return 0;
    int64_t n = 1;
    for (int32_t i = 0; i < tensor->ndim; i++) n *= tensor->shape[i];
    return n;
}

size_t causa_tensor_nbytes(const causa_tensor_t* tensor) {
    return causa_tensor_numel(tensor) * causa_dtype_size(tensor->dtype);
}

causa_tensor_t* causa_tensor_to_device(causa_tensor_t* tensor, causa_device_type_t device) {
    if (!tensor || tensor->device == device) return tensor;
    (void)device;
    return NULL;
}

causa_tensor_t* causa_tensor_to_dtype(causa_tensor_t* tensor, causa_dtype_t dtype) {
    if (!tensor || tensor->dtype == dtype) return tensor;
    (void)dtype;
    return NULL;
}

void causa_tensor_fill(causa_tensor_t* tensor, double value) {
    if (!tensor) return;
    int64_t n = causa_tensor_numel(tensor);
    float v = (float)value;

    // Every element type is written through the store helper, so a tensor
    // of a type the arithmetic kernels do not handle is still filled
    // properly rather than left with whatever the arena held.
    for (int64_t i = 0; i < n; i++) {
        causa_tensor_store_f32(tensor, i, v);
    }
}

void causa_tensor_copy(causa_tensor_t* dst, const causa_tensor_t* src) {
    if (!dst || !src) return;
    if (dst->dtype != src->dtype) return;

    int64_t dn = causa_tensor_numel(dst);
    int64_t sn = causa_tensor_numel(src);

    if (dn == sn) {
        if (causa_tensor_is_contiguous(src) && causa_tensor_is_contiguous(dst)) {
            memcpy(dst->data, src->data, causa_tensor_nbytes(src));
            return;
        }

        // A source that is not laid out in order is copied element by
        // element, following its strides.
        int64_t elem_size = (int64_t)causa_dtype_size(src->dtype);
        int64_t index[CAUSA_MAX_DIMS];
        for (int32_t i = 0; i < src->ndim; i++) index[i] = 0;

        for (int64_t n = 0; n < sn; n++) {
            int64_t offset = 0;
            for (int32_t i = 0; i < src->ndim; i++) {
                offset += index[i] * (src->strides[i] / elem_size);
            }
            causa_tensor_store_f32(dst, offset, causa_tensor_load_f32(src, offset));

            for (int32_t i = src->ndim - 1; i >= 0; i--) {
                if (++index[i] < src->shape[i]) break;
                index[i] = 0;
            }
        }
        return;
    }

    // A single value spreads over the whole destination, and a destination
    // with a single value takes the first value of the source. Without
    // these a copy between tensors of different sizes would silently do
    // nothing, and a model would quietly evaluate to zero.
    if (sn == 1) {
        causa_tensor_fill(dst, (double)causa_tensor_load_f32(src, 0));
        return;
    }

    if (dn == 1) {
        causa_tensor_store_f32(dst, 0, causa_tensor_load_f32(src, 0));
        return;
    }
}

void causa_tensor_init(causa_tensor_t* tensor, causa_init_t init, double param1, double param2) {
    if (!tensor) return;
    int64_t n = causa_tensor_numel(tensor);

    switch (init) {
        case CAUSA_INIT_ZEROS:
            causa_tensor_fill(tensor, 0.0);
            break;
        case CAUSA_INIT_ONES:
            causa_tensor_fill(tensor, 1.0);
            break;
        case CAUSA_INIT_UNIFORM: {
            float low = (float)param1;
            float high = (float)param2;
            for (int64_t i = 0; i < n; i++) {
                float v = low + (high - low) * ((float)rand() / RAND_MAX);
                causa_tensor_store_f32(tensor, i, v);
            }
            break;
        }
        case CAUSA_INIT_NORMAL: {
            float mean = (float)param1;
            float std = (float)param2;
            for (int64_t i = 0; i < n; i++) {
                float u1 = (float)rand() / RAND_MAX;
                float u2 = (float)rand() / RAND_MAX;
                float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265359f * u2);
                causa_tensor_store_f32(tensor, i, mean + std * z);
            }
            break;
        }
        case CAUSA_INIT_XAVIER_UNIFORM: {
            int64_t fan_in = 1, fan_out = 1;
            if (tensor->ndim >= 2) {
                fan_in = tensor->shape[tensor->ndim - 2];
                fan_out = tensor->shape[tensor->ndim - 1];
            }
            float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
            for (int64_t i = 0; i < n; i++) {
                float v = -limit + 2.0f * limit * ((float)rand() / RAND_MAX);
                causa_tensor_store_f32(tensor, i, v);
            }
            break;
        }
        case CAUSA_INIT_KAIMING_UNIFORM: {
            int64_t fan_in = 1;
            if (tensor->ndim >= 2) fan_in = tensor->shape[tensor->ndim - 2];
            float limit = sqrtf(3.0f / (float)fan_in);
            for (int64_t i = 0; i < n; i++) {
                float v = -limit + 2.0f * limit * ((float)rand() / RAND_MAX);
                causa_tensor_store_f32(tensor, i, v);
            }
            break;
        }
        default:
            causa_tensor_fill(tensor, 0.0);
            break;
    }
}

causa_tensor_t* causa_tensor_reduce(
    const causa_tensor_t* input,
    causa_reduce_op_t op,
    const int32_t* dims,
    int32_t num_dims,
    bool keepdim
) {
    if (!input || num_dims <= 0) return NULL;

    // Compute output shape
    int64_t out_shape[CAUSA_MAX_DIMS];
    int32_t out_ndim = 0;
    bool reduce_mask[CAUSA_MAX_DIMS] = {0};

    for (int32_t i = 0; i < num_dims; i++) {
        int32_t d = dims[i];
        if (d < 0) d += input->ndim;
        // A dimension that is not there is an error. Ignoring it would
        // reduce over something else and say so with the wrong shape.
        if (d < 0 || d >= input->ndim) return NULL;
        reduce_mask[d] = true;
    }

    for (int32_t i = 0; i < input->ndim; i++) {
        if (!reduce_mask[i]) {
            out_shape[out_ndim++] = input->shape[i];
        } else if (keepdim) {
            out_shape[out_ndim++] = 1;
        }
    }

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = input->dtype; opts.device = input->device; opts.layout = input->layout; opts.requires_grad = input->requires_grad;
    causa_tensor_t* out = causa_tensor_new(input->arena, input->backend, out_shape, out_ndim, &opts);
    if (!out) return NULL;

    if (input->dtype == CAUSA_DTYPE_F32) {
        int64_t in_strides[CAUSA_MAX_DIMS];
        memcpy(in_strides, input->strides, input->ndim * sizeof(int64_t));

        int64_t out_strides[CAUSA_MAX_DIMS];
        if (out_ndim > 0) {
            out_strides[out_ndim - 1] = 1;
            for (int32_t i = out_ndim - 2; i >= 0; i--) {
                out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
            }
        }

        int64_t numel_out = causa_tensor_numel(out);

        for (int64_t o = 0; o < numel_out; o++) {
            int64_t idx[CAUSA_MAX_DIMS];
            int64_t temp = o;
            for (int32_t i = out_ndim - 1; i >= 0; i--) {
                if (out_strides[i] > 0) {
                    idx[i] = temp / out_strides[i];
                    temp %= out_strides[i];
                } else {
                    idx[i] = 0;
                }
            }

            int64_t in_idx[CAUSA_MAX_DIMS];
            int32_t out_idx = 0;
            for (int32_t i = 0; i < input->ndim; i++) {
                if (reduce_mask[i]) {
                    in_idx[i] = 0;
                } else {
                    in_idx[i] = idx[out_idx++];
                }
            }

            float acc = 0.0f;
            bool first = true;

            int64_t reduce_size = 1;
            for (int32_t i = 0; i < input->ndim; i++) {
                if (reduce_mask[i]) reduce_size *= input->shape[i];
            }

            int64_t reduce_idx[CAUSA_MAX_DIMS];
            for (int32_t i = 0; i < input->ndim; i++) reduce_idx[i] = 0;

            for (int64_t r = 0; r < reduce_size; r++) {
                int64_t rest = r;
                for (int32_t i = input->ndim - 1; i >= 0; i--) {
                    if (reduce_mask[i]) {
                        int64_t dim_size = input->shape[i];
                        reduce_idx[i] = rest % dim_size;
                        rest /= dim_size;
                    }
                }

                int64_t offset = 0;
                for (int32_t i = 0; i < input->ndim; i++) {
                    int64_t i_idx = reduce_mask[i] ? reduce_idx[i] : in_idx[i];
                    offset += i_idx * (input->strides[i] / (int64_t)sizeof(float));
                }

                float val = ((const float*)input->data)[offset];

                switch (op) {
                    case CAUSA_REDUCE_SUM:
                        acc += val;
                        break;
                    case CAUSA_REDUCE_MEAN:
                        acc += val;
                        break;
                    case CAUSA_REDUCE_MAX:
                        if (first || val > acc) acc = val;
                        break;
                    case CAUSA_REDUCE_MIN:
                        if (first || val < acc) acc = val;
                        break;
                    case CAUSA_REDUCE_PROD:
                        if (first) acc = val;
                        else acc *= val;
                        break;
                    case CAUSA_REDUCE_NORM_L1:
                        acc += fabsf(val);
                        break;
                    case CAUSA_REDUCE_NORM_L2:
                        acc += val * val;
                        break;
                }
                first = false;
            }

            if (op == CAUSA_REDUCE_MEAN) {
                acc /= reduce_size;
            } else if (op == CAUSA_REDUCE_NORM_L2) {
                acc = sqrtf(acc);
            }

            ((float*)out->data)[o] = acc;
        }
    }

    return out;
}

causa_tensor_t* causa_tensor_compare(
    const causa_tensor_t* a,
    const causa_tensor_t* b,
    causa_cmp_op_t op
) {
    (void)a; (void)b; (void)op;
    return NULL;
}

causa_tensor_t* causa_tensor_activation(const causa_tensor_t* input, causa_activation_t act, double param) {
    if (!input) return NULL;

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = input->dtype;
    opts.device = input->device;
    opts.layout = input->layout;
    opts.requires_grad = input->requires_grad;

    causa_tensor_t* output = causa_tensor_new(input->arena, input->backend, input->shape, input->ndim, &opts);
    if (!output) return NULL;

    int64_t n = causa_tensor_numel(input);

    switch (act) {
        case CAUSA_ACT_RELU:
            for (int64_t i = 0; i < n; i++) causa_tensor_store_f32(output, i, fmaxf(0.0f, causa_tensor_load_f32(input, i)));
            break;
        case CAUSA_ACT_GELU:
            for (int64_t i = 0; i < n; i++) {
                float x = causa_tensor_load_f32(input, i);
                float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
                causa_tensor_store_f32(output, i, x * cdf);
            }
            break;
        case CAUSA_ACT_SILU:
            for (int64_t i = 0; i < n; i++) {
                float x = causa_tensor_load_f32(input, i);
                causa_tensor_store_f32(output, i, x / (1.0f + expf(-x)));
            }
            break;
        case CAUSA_ACT_TANH:
            for (int64_t i = 0; i < n; i++) causa_tensor_store_f32(output, i, tanhf(causa_tensor_load_f32(input, i)));
            break;
        case CAUSA_ACT_SIGMOID:
            for (int64_t i = 0; i < n; i++) causa_tensor_store_f32(output, i, 1.0f / (1.0f + expf(-causa_tensor_load_f32(input, i))));
            break;
        case CAUSA_ACT_SOFTMAX: {
            float max_val = causa_tensor_load_f32(input, 0);
            for (int64_t i = 1; i < n; i++) {
                float v = causa_tensor_load_f32(input, i);
                if (v > max_val) max_val = v;
            }
            float sum = 0.0f;
            for (int64_t i = 0; i < n; i++) {
                float e = expf(causa_tensor_load_f32(input, i) - max_val);
                causa_tensor_store_f32(output, i, e);
                sum += e;
            }
            for (int64_t i = 0; i < n; i++) causa_tensor_store_f32(output, i, causa_tensor_load_f32(output, i) / sum);
            break;
        }
        case CAUSA_ACT_LOG_SOFTMAX: {
            float max_val = causa_tensor_load_f32(input, 0);
            for (int64_t i = 1; i < n; i++) {
                float v = causa_tensor_load_f32(input, i);
                if (v > max_val) max_val = v;
            }
            float sum = 0.0f;
            for (int64_t i = 0; i < n; i++) {
                float e = expf(causa_tensor_load_f32(input, i) - max_val);
                causa_tensor_store_f32(output, i, e);
                sum += e;
            }
            float log_sum = logf(sum);
            for (int64_t i = 0; i < n; i++) {
                causa_tensor_store_f32(output, i, causa_tensor_load_f32(input, i) - max_val - log_sum);
            }
            break;
        }
        case CAUSA_ACT_ELU: {
            float alpha = (float)param;
            for (int64_t i = 0; i < n; i++) {
                float x = causa_tensor_load_f32(input, i);
                causa_tensor_store_f32(output, i, x >= 0 ? x : alpha * (expf(x) - 1.0f));
            }
            break;
        }
        case CAUSA_ACT_LEAKY_RELU: {
            float alpha = (float)param;
            for (int64_t i = 0; i < n; i++) {
                float x = causa_tensor_load_f32(input, i);
                causa_tensor_store_f32(output, i, x >= 0 ? x : alpha * x);
            }
            break;
        }
        case CAUSA_ACT_SWISH: {
            float beta = (float)param;
            for (int64_t i = 0; i < n; i++) {
                float x = causa_tensor_load_f32(input, i);
                causa_tensor_store_f32(output, i, x * (1.0f / (1.0f + expf(-beta * x))));
            }
            break;
        }
        case CAUSA_ACT_EXP:
            for (int64_t i = 0; i < n; i++) causa_tensor_store_f32(output, i, expf(causa_tensor_load_f32(input, i)));
            break;
        default:
            causa_tensor_copy(output, input);
            break;
    }

    return output;
}

causa_tensor_t* causa_tensor_matmul(const causa_tensor_t* a, const causa_tensor_t* b) {
    if (!a || !b || a->dtype != b->dtype) return NULL;
    if (a->ndim < 2 || b->ndim < 2) return NULL;
    if (a->shape[a->ndim - 1] != b->shape[b->ndim - 2]) return NULL;

    int64_t M = a->shape[a->ndim - 2];
    int64_t K = a->shape[a->ndim - 1];
    int64_t N = b->shape[b->ndim - 1];

    int64_t batch_a = 1;
    for (int32_t i = 0; i < a->ndim - 2; i++) batch_a *= a->shape[i];
    int64_t batch_b = 1;
    for (int32_t i = 0; i < b->ndim - 2; i++) batch_b *= b->shape[i];
    int64_t batch = batch_a > batch_b ? batch_a : batch_b;

    int64_t out_shape[CAUSA_MAX_DIMS];
    int32_t out_ndim = 0;
    for (int32_t i = 0; i < a->ndim - 2; i++) out_shape[out_ndim++] = a->shape[i];
    for (int32_t i = 0; i < b->ndim - 2; i++) out_shape[out_ndim++] = b->shape[i];
    out_shape[out_ndim++] = M;
    out_shape[out_ndim++] = N;

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = a->dtype; opts.device = a->device; opts.layout = a->layout; opts.requires_grad = a->requires_grad || b->requires_grad;
    causa_tensor_t* out = causa_tensor_new(a->arena, a->backend, out_shape, out_ndim, &opts);
    if (!out) return NULL;

    if (a->dtype == CAUSA_DTYPE_F32 && b->dtype == CAUSA_DTYPE_F32) {
        const float* A = (const float*)a->data;
        const float* B = (const float*)b->data;
        float* C = (float*)out->data;

        // The blocked loop accumulates partial sums across k-blocks, so the
        // output must start at zero. A batched output has more elements
        // than one matrix, so the whole output is cleared.
        memset(C, 0, (size_t)causa_tensor_numel(out) * sizeof(float));

        // Strides are stored in bytes; convert to element units for indexing
        const size_t esz = sizeof(float);
        int64_t stride_a_batch = (a->ndim > 2) ? a->strides[a->ndim - 3] / (int64_t)esz : 0;
        int64_t stride_b_batch = (b->ndim > 2) ? b->strides[b->ndim - 3] / (int64_t)esz : 0;
        int64_t stride_out_batch = (out_ndim > 2) ? out->strides[out_ndim - 3] / (int64_t)esz : 0;

        for (int64_t batch_idx = 0; batch_idx < batch; batch_idx++) {
            const float* a_ptr = A + batch_idx * stride_a_batch;
            const float* b_ptr = B + batch_idx * stride_b_batch;
            float* c_ptr = C + batch_idx * stride_out_batch;

            int64_t lda = a->strides[a->ndim - 2] / (int64_t)esz;
            int64_t ldb = b->strides[b->ndim - 2] / (int64_t)esz;
            int64_t ldc = out->strides[out_ndim - 2] / (int64_t)esz;

            const int64_t BLOCK = 32;
            for (int64_t i = 0; i < M; i += BLOCK) {
                int64_t i_max = (i + BLOCK < M) ? i + BLOCK : M;
                for (int64_t j = 0; j < N; j += BLOCK) {
                    int64_t j_max = (j + BLOCK < N) ? j + BLOCK : N;
                    for (int64_t k = 0; k < K; k += BLOCK) {
                        int64_t k_max = (k + BLOCK < K) ? k + BLOCK : K;
                        for (int64_t ii = i; ii < i_max; ii++) {
                            for (int64_t jj = j; jj < j_max; jj++) {
                                float sum = 0.0f;
                                for (int64_t kk = k; kk < k_max; kk++) {
                                    sum += a_ptr[ii * lda + kk] * b_ptr[kk * ldb + jj];
                                }
                                c_ptr[ii * ldc + jj] += sum;
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Any other element type goes through the generic path.
        const int64_t esz = (int64_t)causa_dtype_size(a->dtype);
        int64_t stride_a_batch = (a->ndim > 2) ? a->strides[a->ndim - 3] / esz : 0;
        int64_t stride_b_batch = (b->ndim > 2) ? b->strides[b->ndim - 3] / esz : 0;
        int64_t stride_out_batch = (out_ndim > 2) ? out->strides[out_ndim - 3] / esz : 0;
        int64_t lda = a->strides[a->ndim - 2] / esz;
        int64_t ldb = b->strides[b->ndim - 2] / esz;
        int64_t ldc = out->strides[out_ndim - 2] / esz;

        for (int64_t batch_idx = 0; batch_idx < batch; batch_idx++) {
            int64_t a_base = batch_idx * stride_a_batch;
            int64_t b_base = batch_idx * stride_b_batch;
            int64_t c_base = batch_idx * stride_out_batch;

            for (int64_t i = 0; i < M; i++) {
                for (int64_t j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int64_t k = 0; k < K; k++) {
                        sum += causa_tensor_load_f32(a, a_base + i * lda + k) *
                               causa_tensor_load_f32(b, b_base + k * ldb + j);
                    }
                    int64_t at = c_base + i * ldc + j;
                    causa_tensor_store_f32(out, at, causa_tensor_load_f32(out, at) + sum);
                }
            }
        }
    }

    return out;
}

causa_tensor_t* causa_tensor_bmm(const causa_tensor_t* a, const causa_tensor_t* b) {
    if (!a || !b || a->dtype != b->dtype) return NULL;
    if (a->ndim != 3 || b->ndim != 3) return NULL;
    if (a->shape[0] != b->shape[0]) return NULL;
    if (a->shape[2] != b->shape[1]) return NULL;

    int64_t batch = a->shape[0];
    int64_t M = a->shape[1];
    int64_t K = a->shape[2];
    int64_t N = b->shape[2];

    int64_t out_shape[] = {batch, M, N};

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = a->dtype; opts.device = a->device; opts.layout = a->layout; opts.requires_grad = a->requires_grad || b->requires_grad;
    causa_tensor_t* out = causa_tensor_new(a->arena, a->backend, out_shape, 3, &opts);
    if (!out) return NULL;

    if (a->dtype == CAUSA_DTYPE_F32) {
        const float* A = (const float*)a->data;
        const float* B = (const float*)b->data;
        float* C = (float*)out->data;

        const size_t esz = sizeof(float);
        int64_t stride_a_batch = a->strides[0] / (int64_t)esz;
        int64_t stride_b_batch = b->strides[0] / (int64_t)esz;
        int64_t stride_out_batch = out->strides[0] / (int64_t)esz;

        for (int64_t b_idx = 0; b_idx < batch; b_idx++) {
            const float* a_ptr = A + b_idx * stride_a_batch;
            const float* b_ptr = B + b_idx * stride_b_batch;
            float* c_ptr = C + b_idx * stride_out_batch;

            int64_t lda = a->strides[1] / (int64_t)esz;
            int64_t ldb = b->strides[1] / (int64_t)esz;
            int64_t ldc = out->strides[1] / (int64_t)esz;

            for (int64_t i = 0; i < M; i++) {
                for (int64_t j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int64_t k = 0; k < K; k++) {
                        sum += a_ptr[i * lda + k] * b_ptr[k * ldb + j];
                    }
                    c_ptr[i * ldc + j] = sum;
                }
            }
        }
    }

    return out;
}

causa_tensor_t* causa_tensor_conv1d(
    const causa_tensor_t* input,
    const causa_tensor_t* weight,
    const causa_tensor_t* bias,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t groups
) {
    if (!input || !weight || input->dtype != weight->dtype) return NULL;
    if (input->ndim != 3 || weight->ndim != 3) return NULL;

    int64_t N = input->shape[0];
    int64_t C_in = input->shape[1];
    int64_t L_in = input->shape[2];
    int64_t C_out = weight->shape[0];
    int64_t k = weight->shape[2];

    if (C_in % groups != 0 || C_out % groups != 0) return NULL;
    if (weight->shape[1] != C_in / groups) return NULL;

    int64_t L_out = (L_in + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    if (L_out <= 0) return NULL;

    int64_t out_shape[] = {N, C_out, L_out};

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = input->dtype; opts.device = input->device; opts.layout = input->layout; opts.requires_grad = input->requires_grad || weight->requires_grad;
    causa_tensor_t* out = causa_tensor_new(input->arena, input->backend, out_shape, 3, &opts);
    if (!out) return NULL;

    if (input->dtype == CAUSA_DTYPE_F32) {
        const float* in = (const float*)input->data;
        const float* w = (const float*)weight->data;
        const float* b = bias ? (const float*)bias->data : NULL;
        float* out_data = (float*)out->data;

        int64_t in_stride_n = input->strides[0] / (int64_t)sizeof(float);
        int64_t in_stride_c = input->strides[1] / (int64_t)sizeof(float);
        int64_t in_stride_l = input->strides[2] / (int64_t)sizeof(float);

        int64_t w_stride_c = weight->strides[1] / (int64_t)sizeof(float);
        int64_t w_stride_k = weight->strides[2] / (int64_t)sizeof(float);

        int64_t out_stride_n = out->strides[0] / (int64_t)sizeof(float);
        int64_t out_stride_c = out->strides[1] / (int64_t)sizeof(float);
        int64_t out_stride_l = out->strides[2] / (int64_t)sizeof(float);

        for (int64_t n = 0; n < N; n++) {
            for (int64_t c_out = 0; c_out < C_out; c_out++) {
                int64_t g = c_out / (C_out / groups);
                int64_t c_in_start = g * (C_in / groups);
                int64_t c_in_end = c_in_start + (C_in / groups);

                for (int64_t l_out = 0; l_out < L_out; l_out++) {
                    int64_t l_in_start = l_out * stride - padding;
                    float sum = 0.0f;

                    for (int64_t c_in = c_in_start; c_in < c_in_end; c_in++) {
                        for (int64_t kk = 0; kk < k; kk++) {
                            int64_t l_in = l_in_start + kk * dilation;
                            if (l_in >= 0 && l_in < L_in) {
                                float in_val = in[n * in_stride_n + c_in * in_stride_c + l_in * in_stride_l];
                                float w_val = w[c_out * (weight->strides[0] / (int64_t)sizeof(float)) + (c_in - c_in_start) * w_stride_c + kk * w_stride_k];
                                sum += in_val * w_val;
                            }
                        }
                    }

                    if (b) sum += b[c_out];
                    out_data[n * out_stride_n + c_out * out_stride_c + l_out * out_stride_l] = sum;
                }
            }
        }
    }

    return out;
}

causa_tensor_t* causa_tensor_conv2d(
    const causa_tensor_t* input,
    const causa_tensor_t* weight,
    const causa_tensor_t* bias,
    const int32_t stride[2],
    const int32_t padding[2],
    const int32_t dilation[2],
    int32_t groups
) {
    if (!input || !weight || input->dtype != weight->dtype) return NULL;
    if (input->ndim != 4 || weight->ndim != 4) return NULL;

    int64_t N = input->shape[0];
    int64_t C_in = input->shape[1];
    int64_t H_in = input->shape[2];
    int64_t W_in = input->shape[3];
    int64_t C_out = weight->shape[0];
    int64_t kH = weight->shape[2];
    int64_t kW = weight->shape[3];

    if (C_in % groups != 0 || C_out % groups != 0) return NULL;
    if (weight->shape[1] != C_in / groups) return NULL;

    int64_t H_out = (H_in + 2 * padding[0] - dilation[0] * (kH - 1) - 1) / stride[0] + 1;
    int64_t W_out = (W_in + 2 * padding[1] - dilation[1] * (kW - 1) - 1) / stride[1] + 1;
    if (H_out <= 0 || W_out <= 0) return NULL;

    int64_t out_shape[] = {N, C_out, H_out, W_out};

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = input->dtype; opts.device = input->device; opts.layout = input->layout; opts.requires_grad = input->requires_grad || weight->requires_grad;
    causa_tensor_t* out = causa_tensor_new(input->arena, input->backend, out_shape, 4, &opts);
    if (!out) return NULL;

    if (input->dtype == CAUSA_DTYPE_F32) {
        const float* in = (const float*)input->data;
        const float* w = (const float*)weight->data;
        const float* b = bias ? (const float*)bias->data : NULL;
        float* out_data = (float*)out->data;

        int64_t in_stride_n = input->strides[0] / (int64_t)sizeof(float);
        int64_t in_stride_c = input->strides[1] / (int64_t)sizeof(float);
        int64_t in_stride_h = input->strides[2] / (int64_t)sizeof(float);
        int64_t in_stride_w = input->strides[3] / (int64_t)sizeof(float);

        int64_t w_stride_c = weight->strides[1] / (int64_t)sizeof(float);
        int64_t w_stride_h = weight->strides[2] / (int64_t)sizeof(float);
        int64_t w_stride_w = weight->strides[3] / (int64_t)sizeof(float);

        int64_t out_stride_n = out->strides[0] / (int64_t)sizeof(float);
        int64_t out_stride_c = out->strides[1] / (int64_t)sizeof(float);
        int64_t out_stride_h = out->strides[2] / (int64_t)sizeof(float);
        int64_t out_stride_w = out->strides[3] / (int64_t)sizeof(float);

        for (int64_t n = 0; n < N; n++) {
            for (int64_t c_out = 0; c_out < C_out; c_out++) {
                int64_t g = c_out / (C_out / groups);
                int64_t c_in_start = g * (C_in / groups);
                int64_t c_in_end = c_in_start + (C_in / groups);

                for (int64_t h_out = 0; h_out < H_out; h_out++) {
                    for (int64_t w_out = 0; w_out < W_out; w_out++) {
                        int64_t h_in_start = h_out * stride[0] - padding[0];
                        int64_t w_in_start = w_out * stride[1] - padding[1];
                        float sum = 0.0f;

                        for (int64_t c_in = c_in_start; c_in < c_in_end; c_in++) {
                            for (int64_t kh = 0; kh < kH; kh++) {
                                for (int64_t kw = 0; kw < kW; kw++) {
                                    int64_t h_in = h_in_start + kh * dilation[0];
                                    int64_t w_in = w_in_start + kw * dilation[1];
                                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                        float in_val = in[n * in_stride_n + c_in * in_stride_c + h_in * in_stride_h + w_in * in_stride_w];
                                        float w_val = w[c_out * (weight->strides[0] / (int64_t)sizeof(float)) + (c_in - c_in_start) * w_stride_c + kh * w_stride_h + kw * w_stride_w];
                                        sum += in_val * w_val;
                                    }
                                }
                            }
                        }

                        if (b) sum += b[c_out];
                        out_data[n * out_stride_n + c_out * out_stride_c + h_out * out_stride_h + w_out * out_stride_w] = sum;
                    }
                }
            }
        }
    }

    return out;
}

bool causa_tensor_shapes_broadcastable(
    const int64_t* a_shape, int32_t a_ndim,
    const int64_t* b_shape, int32_t b_ndim
) {
    int32_t max_ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
    for (int32_t i = 1; i <= max_ndim; i++) {
        int64_t a_dim = (i <= a_ndim) ? a_shape[a_ndim - i] : 1;
        int64_t b_dim = (i <= b_ndim) ? b_shape[b_ndim - i] : 1;
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) return false;
    }
    return true;
}

int32_t causa_tensor_broadcast_shape(
    const int64_t* a_shape, int32_t a_ndim,
    const int64_t* b_shape, int32_t b_ndim,
    int64_t* out_shape
) {
    if (!causa_tensor_shapes_broadcastable(a_shape, a_ndim, b_shape, b_ndim)) return -1;

    int32_t max_ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
    for (int32_t i = 1; i <= max_ndim; i++) {
        int64_t a_dim = (i <= a_ndim) ? a_shape[a_ndim - i] : 1;
        int64_t b_dim = (i <= b_ndim) ? b_shape[b_ndim - i] : 1;
        out_shape[max_ndim - i] = (a_dim == 1) ? b_dim : a_dim;
    }
    return max_ndim;
}

// Compute per-dimension strides (in elements) for reading a tensor under
// broadcasting.
//
// Dimensions are indexed from the right: position 0 is the rightmost.
// A dimension of extent 1, or one beyond the tensor's rank, gets a stride of
// zero so the value repeats.
static void broadcast_strides(
    const causa_tensor_t* t,
    int32_t out_ndim,
    int64_t* out_strides
) {
    const size_t esz = causa_dtype_size(t->dtype);
    for (int32_t p = 0; p < out_ndim; p++) {
        int32_t t_dim = t->ndim - 1 - p;
        if (t_dim < 0) {
            out_strides[p] = 0;
        } else if (t->shape[t_dim] == 1) {
            out_strides[p] = 0;
        } else {
            out_strides[p] = t->strides[t_dim] / (int64_t)esz;
        }
    }
}

// A tensor viewed under broadcast semantics.
//
// bcast_strides are used to read the tensor's values at a position in the
// broadcast shape; dimensions of extent 1 repeat, so their stride is zero.
// own_strides are the tensor's actual element strides, used to write results
// back; a dimension of extent 1 always receives index 0.
typedef struct {
    const causa_tensor_t* t;
    int64_t bcast_strides[CAUSA_MAX_DIMS];
    int64_t own_strides[CAUSA_MAX_DIMS];
    int32_t out_ndim;
} broadcast_view_t;

static void broadcast_view_init(broadcast_view_t* v, const causa_tensor_t* t, int32_t out_ndim) {
    v->t = t;
    v->out_ndim = out_ndim;
    const size_t esz = causa_dtype_size(t->dtype);

    for (int32_t p = 0; p < out_ndim; p++) {
        int32_t t_dim = t->ndim - 1 - p;
        if (t_dim < 0) {
            v->bcast_strides[p] = 0;
            v->own_strides[p] = 0;
        } else {
            int64_t stride = t->strides[t_dim] / (int64_t)esz;
            v->own_strides[p] = stride;
            v->bcast_strides[p] = (t->shape[t_dim] == 1) ? 0 : stride;
        }
    }
}

// Offset for reading the tensor's value at the broadcast position
static int64_t broadcast_view_read(const broadcast_view_t* v, const int64_t* idx) {
    int64_t off = 0;
    for (int32_t p = 0; p < v->out_ndim; p++) {
        off += idx[p] * v->bcast_strides[p];
    }
    return off;
}

// Offset for writing to the tensor at the broadcast position. Dimensions of
// extent 1 always land on index 0, so contributions to them accumulate.
static int64_t broadcast_view_write(const broadcast_view_t* v, const int64_t* idx) {
    int64_t off = 0;
    for (int32_t p = 0; p < v->out_ndim; p++) {
        int32_t t_dim = v->t->ndim - 1 - p;
        if (t_dim < 0) continue;
        if (v->t->shape[t_dim] == 1) continue;
        off += idx[p] * v->own_strides[p];
    }
    return off;
}

// Walk a right-aligned index over an output shape, invoking the callback for
// each element. The stride arrays are right-aligned; the output is contiguous
// so its offset is the element counter.
typedef void (*causa_broadcast_walk_fn)(
    void* ctx, int64_t offset_a, int64_t offset_b, int64_t offset_out, const int64_t* idx);

static void broadcast_walk(
    const int64_t* out_shape,
    int32_t out_ndim,
    const int64_t* a_strides,
    const int64_t* b_strides,
    causa_broadcast_walk_fn fn,
    void* ctx
) {
    int64_t total = 1;
    for (int32_t i = 0; i < out_ndim; i++) total *= out_shape[i];

    int64_t idx[CAUSA_MAX_DIMS];
    for (int32_t p = 0; p < out_ndim; p++) idx[p] = 0;

    for (int64_t n = 0; n < total; n++) {
        int64_t a_off = 0, b_off = 0;
        for (int32_t p = 0; p < out_ndim; p++) {
            a_off += idx[p] * a_strides[p];
            b_off += idx[p] * b_strides[p];
        }
        fn(ctx, a_off, b_off, n, idx);

        for (int32_t p = 0; p < out_ndim; p++) {
            if (++idx[p] < out_shape[out_ndim - 1 - p]) break;
            idx[p] = 0;
        }
    }
}

// Elementwise binary operation with numpy-style broadcasting.
typedef struct {
    const causa_tensor_t* a;
    const causa_tensor_t* b;
    causa_tensor_t* out;
    char op;
} binary_op_ctx_t;

static void binary_op_walk(void* ctx, int64_t offset_a, int64_t offset_b, int64_t offset_out, const int64_t* idx) {
    (void)idx;
    binary_op_ctx_t* c = (binary_op_ctx_t*)ctx;
    float x = causa_tensor_load_f32(c->a, offset_a);
    float y = causa_tensor_load_f32(c->b, offset_b);
    float r;
    switch (c->op) {
        case '+': r = x + y; break;
        case '-': r = x - y; break;
        case '*': r = x * y; break;
        case '/': r = x / (y + 1e-8f); break;
        default: r = 0.0f; break;
    }
    causa_tensor_store_f32(c->out, offset_out, r);
}
#define BINARY_OP(name, opchar) \
causa_tensor_t* causa_tensor_##name(const causa_tensor_t* a, const causa_tensor_t* b) { \
    if (!a || !b || a->dtype != b->dtype) return NULL; \
    int64_t out_shape[CAUSA_MAX_DIMS]; \
    int32_t out_ndim = causa_tensor_broadcast_shape(a->shape, a->ndim, b->shape, b->ndim, out_shape); \
    if (out_ndim <= 0) return NULL; \
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS; \
    opts.dtype = a->dtype; opts.device = a->device; opts.layout = a->layout; opts.requires_grad = a->requires_grad || b->requires_grad; \
    causa_tensor_t* out = causa_tensor_new(a->arena, a->backend, out_shape, out_ndim, &opts); \
    if (!out) return NULL; \
    int64_t a_strides[CAUSA_MAX_DIMS], b_strides[CAUSA_MAX_DIMS]; \
    broadcast_strides(a, out_ndim, a_strides); \
    broadcast_strides(b, out_ndim, b_strides); \
    binary_op_ctx_t ctx = { a, b, out, opchar }; \
    broadcast_walk(out_shape, out_ndim, a_strides, b_strides, binary_op_walk, &ctx); \
    return out; \
}

BINARY_OP(add, '+')
BINARY_OP(sub, '-')
BINARY_OP(mul, '*')
BINARY_OP(div, '/')

#define SCALAR_OP(name, op) \
causa_tensor_t* causa_tensor_##name(const causa_tensor_t* a, double b) { \
    if (!a) return NULL; \
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS; \
    opts.dtype = a->dtype; opts.device = a->device; opts.layout = a->layout; opts.requires_grad = a->requires_grad; \
    causa_tensor_t* out = causa_tensor_new(a->arena, a->backend, a->shape, a->ndim, &opts); \
    if (!out) return NULL; \
    int64_t n = causa_tensor_numel(a); \
    float vb = (float)b; \
    for (int64_t i = 0; i < n; i++) causa_tensor_store_f32(out, i, causa_tensor_load_f32(a, i) op vb); \
    return out; \
}

SCALAR_OP(add_scalar, +)
SCALAR_OP(sub_scalar, -)
SCALAR_OP(mul_scalar, *)
SCALAR_OP(div_scalar, /)

// A unary operation on a tensor.
//
// The elements are read and written through the helpers, so the operation
// works on any element type rather than only on float32.
#define UNARY_OP(name, expr) \
causa_tensor_t* causa_tensor_##name(const causa_tensor_t* a) { \
    if (!a) return NULL; \
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS; \
    opts.dtype = a->dtype; opts.device = a->device; opts.layout = a->layout; opts.requires_grad = a->requires_grad; \
    causa_tensor_t* out = causa_tensor_new(a->arena, a->backend, a->shape, a->ndim, &opts); \
    if (!out) return NULL; \
    int64_t n = causa_tensor_numel(a); \
    for (int64_t i = 0; i < n; i++) { \
        float x = causa_tensor_load_f32(a, i); \
        (void)x; \
        causa_tensor_store_f32(out, i, (expr)); \
    } \
    return out; \
}

UNARY_OP(neg, -x)
UNARY_OP(abs, fabsf(x))
UNARY_OP(sqrt, sqrtf(fmaxf(0.0f, x)))
UNARY_OP(rsqrt, 1.0f / sqrtf(fmaxf(1e-8f, x)))
UNARY_OP(exp, expf(x))
UNARY_OP(log, logf(fmaxf(1e-8f, x)))

causa_tensor_t* causa_tensor_pow(const causa_tensor_t* a, double exp) {
    if (!a) return NULL;
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = a->dtype; opts.device = a->device; opts.layout = a->layout; opts.requires_grad = a->requires_grad;
    causa_tensor_t* out = causa_tensor_new(a->arena, a->backend, a->shape, a->ndim, &opts);
    if (!out) return NULL;
    int64_t n = causa_tensor_numel(a);
    float e = (float)exp;
    for (int64_t i = 0; i < n; i++) {
        float x = causa_tensor_load_f32(a, i);
        causa_tensor_store_f32(out, i, powf(fmaxf(0.0f, x), e));
    }
    return out;
}

int causa_tensor_print(const causa_tensor_t* tensor, int max_elements) {
    return causa_tensor_fprint(stdout, tensor, max_elements);
}

int causa_tensor_fprint(FILE* stream, const causa_tensor_t* tensor, int max_elements) {
    if (!tensor || !stream) return -1;

    fprintf(stream, "causa_tensor(%s", causa_dtype_name(tensor->dtype));
    if (tensor->name[0]) fprintf(stream, ", name=\"%s\"", tensor->name);
    fprintf(stream, ", shape=[");
    for (int32_t i = 0; i < tensor->ndim; i++) {
        if (i) fprintf(stream, ", ");
        fprintf(stream, "%ld", (long)tensor->shape[i]);
    }
    fprintf(stream, "], device=%s", causa_device_name(tensor->device));
    if (tensor->requires_grad) fprintf(stream, ", requires_grad=true");
    if (tensor->is_view) fprintf(stream, ", view=true");
    fprintf(stream, ")");

    if (max_elements > 0) {
        int64_t n = causa_tensor_numel(tensor);
        int64_t limit = n < max_elements ? n : max_elements;
        fprintf(stream, "\n  [");
        const float* data = (const float*)tensor->data;
        for (int64_t i = 0; i < limit; i++) {
            if (i) fprintf(stream, ", ");
            fprintf(stream, "%.4f", data[i]);
        }
        if (n > limit) fprintf(stream, ", ...");
        fprintf(stream, "]");
    }
    fprintf(stream, "\n");
    return 0;
}

// ===== BACKWARD PASS IMPLEMENTATIONS =====

static void ensure_grad(causa_tensor_t* tensor) {
    if (!tensor->grad && tensor->requires_grad) {
        causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        opts.dtype = tensor->dtype;
        opts.device = tensor->device;
        opts.layout = tensor->layout;
        opts.requires_grad = false;
        tensor->grad = causa_tensor_new(tensor->arena, tensor->backend, tensor->shape, tensor->ndim, &opts);
        if (tensor->grad) causa_tensor_fill(tensor->grad, 0.0f);
    }
}

static void accum_grad(causa_tensor_t* tensor, const causa_tensor_t* grad) {
    if (!tensor->requires_grad || !grad) return;
    // The gradient of a tensor that has not been given one yet is made
    // here. Without it the contribution would be dropped, and the gradient
    // of a product would be lost.
    ensure_grad(tensor);
    if (!tensor->grad) return;
    int64_t n = causa_tensor_numel(tensor);
    for (int64_t i = 0; i < n; i++) {
        causa_tensor_store_f32(tensor->grad, i, causa_tensor_load_f32(tensor->grad, i) + causa_tensor_load_f32(grad, i));
    }
}

static void accum_grad_scalar(causa_tensor_t* tensor, float grad) {
    if (!tensor->requires_grad) return;
    ensure_grad(tensor);
    if (!tensor->grad) return;
    int64_t n = causa_tensor_numel(tensor);
    for (int64_t i = 0; i < n; i++) {
        causa_tensor_store_f32(tensor->grad, i, causa_tensor_load_f32(tensor->grad, i) + grad);
    }
}

void causa_tensor_accumulate_grad(causa_tensor_t* t, const causa_tensor_t* grad) {
    if (!t || !grad) return;
    ensure_grad(t);
    accum_grad(t, grad);
}


void causa_tensor_transpose_backward(
    causa_tensor_t* a,
    const causa_tensor_t* grad_output,
    int32_t dim0,
    int32_t dim1
) {
    if (!a || !grad_output) return;

    // The gradient w.r.t. the input is the output gradient with the two
    // transposed dimensions swapped back.
    //
    // The transpose gives a view over the same buffer, so the gradient is
    // not modified by it and the pointer does not have to be mutable.
    causa_tensor_t* t = causa_tensor_transpose((causa_tensor_t*)grad_output, dim0, dim1);
    if (!t) return;
    causa_tensor_accumulate_grad(a, t);
}

causa_tensor_t* causa_tensor_diag(const causa_tensor_t* a) {
    if (!a || a->ndim != 2) return NULL;
    if (a->shape[0] != a->shape[1]) return NULL;

    int64_t d = a->shape[0];

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = a->dtype;
    opts.device = a->device;
    opts.layout = CAUSA_LAYOUT_CONTIGUOUS;
    opts.requires_grad = a->requires_grad;

    int64_t shape[1] = {d};
    causa_tensor_t* out = causa_tensor_new(a->arena, a->backend, shape, 1, &opts);
    if (!out) return NULL;

    const size_t esz = causa_dtype_size(a->dtype);
    int64_t stride0 = a->strides[0] / (int64_t)esz;
    int64_t stride1 = a->strides[1] / (int64_t)esz;

    if (a->dtype == CAUSA_DTYPE_F32) {
        const float* src = (const float*)a->data;
        float* dst = (float*)out->data;
        for (int64_t i = 0; i < d; i++) {
            dst[i] = src[i * stride0 + i * stride1];
        }
    } else {
        for (int64_t i = 0; i < d; i++) {
            causa_tensor_store_f32(out, i, causa_tensor_load_f32(a, i * stride0 + i * stride1));
        }
    }

    return out;
}

void causa_tensor_diag_backward(causa_tensor_t* a, const causa_tensor_t* grad_output) {
    if (!a || !grad_output || a->ndim != 2) return;
    if (a->shape[0] != a->shape[1]) return;
    // The gradient holds one value per diagonal entry
    if (causa_tensor_numel(grad_output) != a->shape[0]) return;

    ensure_grad(a);

    int64_t d = a->shape[0];
    const size_t esz = causa_dtype_size(a->dtype);
    int64_t stride0 = a->strides[0] / (int64_t)esz;
    int64_t stride1 = a->strides[1] / (int64_t)esz;

    if (a->dtype == CAUSA_DTYPE_F32) {
        float* dst = (float*)a->grad->data;
        const float* src = (const float*)grad_output->data;
        for (int64_t i = 0; i < d; i++) {
            dst[i * stride0 + i * stride1] += src[i];
        }
    } else {
        for (int64_t i = 0; i < d; i++) {
            int64_t at = i * stride0 + i * stride1;
            causa_tensor_store_f32(a->grad, at, causa_tensor_load_f32(a->grad, at) + causa_tensor_load_f32(grad_output, i));
        }
    }
}

static void accum_broadcast_grad(
    causa_tensor_t* input,
    const causa_tensor_t* grad,
    int32_t grad_ndim
) {
    if (!input || !grad || !input->requires_grad) return;

    // The gradient buffer is created here rather than by the caller, because
    // this function is the only place that knows the input needs one.
    ensure_grad(input);
    if (!input->grad) return;

    const size_t esz = causa_dtype_size(input->dtype);

    // Gradient strides, right-aligned
    int64_t g_strides[CAUSA_MAX_DIMS];
    for (int32_t p = 0; p < grad_ndim; p++) {
        g_strides[p] = grad->strides[grad_ndim - 1 - p] / (int64_t)esz;
    }

    // Input strides, right-aligned; broadcast dimensions get a stride of zero
    int64_t i_strides[CAUSA_MAX_DIMS];
    for (int32_t p = 0; p < grad_ndim; p++) {
        int32_t in_dim = input->ndim - 1 - p;
        if (in_dim < 0) {
            i_strides[p] = 0;
        } else if (input->shape[in_dim] == 1) {
            i_strides[p] = 0;
        } else {
            i_strides[p] = input->strides[in_dim] / (int64_t)esz;
        }
    }

    int64_t total = causa_tensor_numel(grad);
    int64_t idx[CAUSA_MAX_DIMS];
    for (int32_t p = 0; p < grad_ndim; p++) idx[p] = 0;

    if (input->dtype == CAUSA_DTYPE_F32 && grad->dtype == CAUSA_DTYPE_F32) {
        float* gi = (float*)input->grad->data;
        const float* gg = (const float*)grad->data;

        for (int64_t n = 0; n < total; n++) {
            int64_t g_off = 0, i_off = 0;
            for (int32_t p = 0; p < grad_ndim; p++) {
                g_off += idx[p] * g_strides[p];
                i_off += idx[p] * i_strides[p];
            }
            gi[i_off] += gg[g_off];

            for (int32_t p = 0; p < grad_ndim; p++) {
                if (++idx[p] < grad->shape[grad_ndim - 1 - p]) break;
                idx[p] = 0;
            }
        }
        return;
    }

    for (int64_t n = 0; n < total; n++) {
        int64_t g_off = 0, i_off = 0;
        for (int32_t p = 0; p < grad_ndim; p++) {
            g_off += idx[p] * g_strides[p];
            i_off += idx[p] * i_strides[p];
        }
        causa_tensor_store_f32(input->grad, i_off, causa_tensor_load_f32(input->grad, i_off) + causa_tensor_load_f32(grad, g_off));

        for (int32_t p = 0; p < grad_ndim; p++) {
            if (++idx[p] < grad->shape[grad_ndim - 1 - p]) break;
            idx[p] = 0;
        }
    }
}

void causa_tensor_add_backward(causa_tensor_t* a, causa_tensor_t* b, const causa_tensor_t* grad_output) {
    if (!grad_output) return;
    ensure_grad(a);
    ensure_grad(b);

    int64_t out_shape[CAUSA_MAX_DIMS];
    int32_t out_ndim = causa_tensor_broadcast_shape(a->shape, a->ndim, b->shape, b->ndim, out_shape);
    if (out_ndim <= 0) return;

    // The gradient of a sum broadcasts back over every input element.
    accum_broadcast_grad(a, grad_output, out_ndim);
    accum_broadcast_grad(b, grad_output, out_ndim);
}

void causa_tensor_sub_backward(causa_tensor_t* a, causa_tensor_t* b, const causa_tensor_t* grad_output) {
    if (!grad_output) return;
    ensure_grad(a);
    ensure_grad(b);

    int64_t out_shape[CAUSA_MAX_DIMS];
    int32_t out_ndim = causa_tensor_broadcast_shape(a->shape, a->ndim, b->shape, b->ndim, out_shape);
    if (out_ndim <= 0) return;

    accum_broadcast_grad(a, grad_output, out_ndim);

    // The gradient with respect to b carries the opposite sign.
    causa_tensor_t* neg_grad = causa_tensor_neg(grad_output);
    if (neg_grad) {
        accum_broadcast_grad(b, neg_grad, out_ndim);
        causa_tensor_release(neg_grad);
    }
}

// Reduce a broadcasted gradient back onto an input's shape.
// The gradient has the broadcast shape; the input may have smaller dimensions
// or extent-1 dimensions, which need their contributions summed.

void causa_tensor_mul_backward(causa_tensor_t* a, causa_tensor_t* b, const causa_tensor_t* grad_output) {
    if (!grad_output) return;
    ensure_grad(a);
    ensure_grad(b);

    int64_t out_shape[CAUSA_MAX_DIMS];
    int32_t out_ndim = causa_tensor_broadcast_shape(a->shape, a->ndim, b->shape, b->ndim, out_shape);
    if (out_ndim <= 0) return;

    broadcast_view_t va, vb;
    broadcast_view_init(&va, a, out_ndim);
    broadcast_view_init(&vb, b, out_ndim);

    int64_t a_total = causa_tensor_numel(a);
    int64_t b_total = causa_tensor_numel(b);
    float* a_acc = (float*)calloc((size_t)a_total, sizeof(float));
    float* b_acc = (float*)calloc((size_t)b_total, sizeof(float));
    if (!a_acc || !b_acc) { free(a_acc); free(b_acc); return; }

    int64_t total = 1;
    for (int32_t i = 0; i < out_ndim; i++) total *= out_shape[i];

    int64_t idx[CAUSA_MAX_DIMS];
    for (int32_t p = 0; p < out_ndim; p++) idx[p] = 0;

    for (int64_t n = 0; n < total; n++) {
        int64_t a_read = broadcast_view_read(&va, idx);
        int64_t b_read = broadcast_view_read(&vb, idx);
        int64_t a_write = broadcast_view_write(&va, idx);
        int64_t b_write = broadcast_view_write(&vb, idx);

        float g = causa_tensor_load_f32(grad_output, n);
        a_acc[a_write] += g * causa_tensor_load_f32(b, b_read);
        b_acc[b_write] += g * causa_tensor_load_f32(a, a_read);

        for (int32_t p = 0; p < out_ndim; p++) {
            if (++idx[p] < out_shape[out_ndim - 1 - p]) break;
            idx[p] = 0;
        }
    }

    if (a->requires_grad) {
        ensure_grad(a);
        if (a->grad) {
            for (int64_t i = 0; i < a_total; i++) {
                causa_tensor_store_f32(a->grad, i, causa_tensor_load_f32(a->grad, i) + a_acc[i]);
            }
        }
    }
    if (b->requires_grad) {
        ensure_grad(b);
        if (b->grad) {
            for (int64_t i = 0; i < b_total; i++) {
                causa_tensor_store_f32(b->grad, i, causa_tensor_load_f32(b->grad, i) + b_acc[i]);
            }
        }
    }

    free(a_acc);
    free(b_acc);
}

void causa_tensor_div_backward(causa_tensor_t* a, causa_tensor_t* b, const causa_tensor_t* grad_output) {
    if (!grad_output) return;


    int64_t out_shape[CAUSA_MAX_DIMS];
    int32_t out_ndim = causa_tensor_broadcast_shape(a->shape, a->ndim, b->shape, b->ndim, out_shape);
    if (out_ndim <= 0) return;

    broadcast_view_t va, vb;
    broadcast_view_init(&va, a, out_ndim);
    broadcast_view_init(&vb, b, out_ndim);

    int64_t a_total = causa_tensor_numel(a);
    int64_t b_total = causa_tensor_numel(b);
    float* a_acc = (float*)calloc((size_t)a_total, sizeof(float));
    float* b_acc = (float*)calloc((size_t)b_total, sizeof(float));
    if (!a_acc || !b_acc) { free(a_acc); free(b_acc); return; }

    int64_t total = 1;
    for (int32_t i = 0; i < out_ndim; i++) total *= out_shape[i];

    int64_t idx[CAUSA_MAX_DIMS];
    for (int32_t p = 0; p < out_ndim; p++) idx[p] = 0;

    for (int64_t n = 0; n < total; n++) {
        int64_t a_read = broadcast_view_read(&va, idx);
        int64_t b_read = broadcast_view_read(&vb, idx);
        int64_t a_write = broadcast_view_write(&va, idx);
        int64_t b_write = broadcast_view_write(&vb, idx);

        float g = causa_tensor_load_f32(grad_output, n);
        float bv = causa_tensor_load_f32(b, b_read);
        float b_sq = bv * bv + 1e-8f;
        a_acc[a_write] += g / bv;
        b_acc[b_write] += -g * causa_tensor_load_f32(a, a_read) / b_sq;

        for (int32_t p = 0; p < out_ndim; p++) {
            if (++idx[p] < out_shape[out_ndim - 1 - p]) break;
            idx[p] = 0;
        }
    }

    if (a->requires_grad) {
        ensure_grad(a);
        if (a->grad) {
            for (int64_t i = 0; i < a_total; i++) {
                causa_tensor_store_f32(a->grad, i, causa_tensor_load_f32(a->grad, i) + a_acc[i]);
            }
        }
    }
    if (b->requires_grad) {
        ensure_grad(b);
        if (b->grad) {
            for (int64_t i = 0; i < b_total; i++) {
                causa_tensor_store_f32(b->grad, i, causa_tensor_load_f32(b->grad, i) + b_acc[i]);
            }
        }
    }

    free(a_acc);
    free(b_acc);
}

void causa_tensor_add_scalar_backward(causa_tensor_t* a, double b, const causa_tensor_t* grad_output) {
    (void)b;
    if (!grad_output) return;
    ensure_grad(a);
    accum_grad(a, grad_output);
}

void causa_tensor_sub_scalar_backward(causa_tensor_t* a, double b, const causa_tensor_t* grad_output) {
    (void)b;
    if (!grad_output) return;
    ensure_grad(a);
    accum_grad(a, grad_output);
}

void causa_tensor_mul_scalar_backward(causa_tensor_t* a, double b, const causa_tensor_t* grad_output) {
    if (!grad_output) return;
    ensure_grad(a);
    float vb = (float)b;
    int64_t n = causa_tensor_numel(a);
    float* ga = a->grad ? (float*)a->grad->data : NULL;
    if (!ga) return;
    const float* go = (const float*)grad_output->data;
    for (int64_t i = 0; i < n; i++) ga[i] += go[i] * vb;
}

void causa_tensor_div_scalar_backward(causa_tensor_t* a, double b, const causa_tensor_t* grad_output) {
    if (!grad_output) return;
    ensure_grad(a);
    float vb = (float)b;
    int64_t n = causa_tensor_numel(a);
    float* ga = a->grad ? (float*)a->grad->data : NULL;
    if (!ga) return;
    const float* go = (const float*)grad_output->data;
    for (int64_t i = 0; i < n; i++) ga[i] += go[i] / vb;
}

void causa_tensor_neg_backward(causa_tensor_t* a, const causa_tensor_t* grad_output) {
    if (!grad_output) return;
    ensure_grad(a);
    causa_tensor_t* neg_grad = causa_tensor_neg(grad_output);
    accum_grad(a, neg_grad);
    causa_tensor_release(neg_grad);
}

void causa_tensor_abs_backward(causa_tensor_t* a, const causa_tensor_t* grad_output) {
    if (!grad_output) return;
    ensure_grad(a);

    int64_t n = causa_tensor_numel(a);
    float* ga = a->grad ? (float*)a->grad->data : NULL;
    if (!ga) return;
    const float* da = (const float*)a->data;
    const float* go = (const float*)grad_output->data;

    for (int64_t i = 0; i < n; i++) {
        float sign = (da[i] > 0) ? 1.0f : ((da[i] < 0) ? -1.0f : 0.0f);
        ga[i] += go[i] * sign;
    }
}

void causa_tensor_sqrt_backward(causa_tensor_t* a, const causa_tensor_t* grad_output) {
    if (!grad_output) return;
    ensure_grad(a);

    int64_t n = causa_tensor_numel(a);
    float* ga = a->grad ? (float*)a->grad->data : NULL;
    if (!ga) return;
    const float* da = (const float*)a->data;
    const float* go = (const float*)grad_output->data;

    for (int64_t i = 0; i < n; i++) {
        float val = fmaxf(1e-8f, da[i]);
        ga[i] += go[i] * 0.5f / sqrtf(val);
    }
}

void causa_tensor_rsqrt_backward(causa_tensor_t* a, const causa_tensor_t* grad_output) {
    if (!grad_output) return;
    ensure_grad(a);

    int64_t n = causa_tensor_numel(a);
    float* ga = a->grad ? (float*)a->grad->data : NULL;
    if (!ga) return;
    const float* da = (const float*)a->data;
    const float* go = (const float*)grad_output->data;

    for (int64_t i = 0; i < n; i++) {
        float val = fmaxf(1e-8f, da[i]);
        ga[i] += go[i] * -0.5f * powf(val, -1.5f);
    }
}

void causa_tensor_exp_backward(causa_tensor_t* a, const causa_tensor_t* grad_output) {
    if (!grad_output) return;
    ensure_grad(a);

    int64_t n = causa_tensor_numel(a);
    float* ga = a->grad ? (float*)a->grad->data : NULL;
    if (!ga) return;
    const float* da = (const float*)a->data;
    const float* go = (const float*)grad_output->data;

    for (int64_t i = 0; i < n; i++) {
        ga[i] += go[i] * expf(da[i]);
    }
}

void causa_tensor_log_backward(causa_tensor_t* a, const causa_tensor_t* grad_output) {
    if (!grad_output) return;
    ensure_grad(a);

    int64_t n = causa_tensor_numel(a);
    float* ga = a->grad ? (float*)a->grad->data : NULL;
    if (!ga) return;
    const float* da = (const float*)a->data;
    const float* go = (const float*)grad_output->data;

    for (int64_t i = 0; i < n; i++) {
        ga[i] += go[i] / fmaxf(1e-8f, da[i]);
    }
}

void causa_tensor_pow_backward(causa_tensor_t* a, double exp, const causa_tensor_t* grad_output) {
    if (!grad_output) return;
    ensure_grad(a);

    int64_t n = causa_tensor_numel(a);
    float* ga = a->grad ? (float*)a->grad->data : NULL;
    if (!ga) return;
    const float* da = (const float*)a->data;
    const float* go = (const float*)grad_output->data;
    float e = (float)exp;

    for (int64_t i = 0; i < n; i++) {
        float val = fmaxf(1e-8f, da[i]);
        ga[i] += go[i] * e * powf(val, e - 1.0f);
    }
}

void causa_tensor_activation_backward(causa_tensor_t* input, causa_activation_t act, double param, const causa_tensor_t* grad_output) {
    if (!grad_output || !input) return;
    ensure_grad(input);

    int64_t n = causa_tensor_numel(input);
    float* gi = (float*)input->grad->data;
    const float* di = (const float*)input->data;
    const float* go = (const float*)grad_output->data;

    switch (act) {
        case CAUSA_ACT_RELU:
            for (int64_t i = 0; i < n; i++) {
                gi[i] += go[i] * (di[i] > 0 ? 1.0f : 0.0f);
            }
            break;
        case CAUSA_ACT_GELU: {
            for (int64_t i = 0; i < n; i++) {
                float x = di[i];
                float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
                float pdf = 0.5f * (1.0f / sqrtf(2.0f * 3.14159265359f)) * expf(-0.5f * x * x);
                float gelu_grad = cdf + x * pdf;
                gi[i] += go[i] * gelu_grad;
            }
            break;
        }
        case CAUSA_ACT_SILU: {
            for (int64_t i = 0; i < n; i++) {
                float x = di[i];
                float sig = 1.0f / (1.0f + expf(-x));
                gi[i] += go[i] * sig * (1.0f + x * (1.0f - sig));
            }
            break;
        }
        case CAUSA_ACT_TANH:
            for (int64_t i = 0; i < n; i++) {
                float t = tanhf(di[i]);
                gi[i] += go[i] * (1.0f - t * t);
            }
            break;
        case CAUSA_ACT_SIGMOID:
            for (int64_t i = 0; i < n; i++) {
                float s = 1.0f / (1.0f + expf(-di[i]));
                gi[i] += go[i] * s * (1.0f - s);
            }
            break;
        case CAUSA_ACT_SOFTMAX: {
            for (int64_t i = 0; i < n; i++) {
                float s = expf(di[i]);
                float sum = 0;
                for (int64_t j = 0; j < n; j++) sum += expf(di[j]);
                s /= sum;
                gi[i] += go[i] * s * (1.0f - s);
            }
            break;
        }
        case CAUSA_ACT_LOG_SOFTMAX: {
            for (int64_t i = 0; i < n; i++) {
                float s = expf(di[i]);
                float sum = 0;
                for (int64_t j = 0; j < n; j++) sum += expf(di[j]);
                s /= sum;
                gi[i] += go[i] * (1.0f - s);
            }
            break;
        }
        case CAUSA_ACT_ELU: {
            float alpha = (float)param;
            for (int64_t i = 0; i < n; i++) {
                gi[i] += go[i] * (di[i] > 0 ? 1.0f : alpha * expf(di[i]));
            }
            break;
        }
        case CAUSA_ACT_LEAKY_RELU: {
            float alpha = (float)param;
            for (int64_t i = 0; i < n; i++) {
                gi[i] += go[i] * (di[i] > 0 ? 1.0f : alpha);
            }
            break;
        }
        case CAUSA_ACT_SWISH: {
            float beta = (float)param;
            for (int64_t i = 0; i < n; i++) {
                float x = di[i];
                float sig = 1.0f / (1.0f + expf(-beta * x));
                gi[i] += go[i] * sig * (1.0f + beta * x * (1.0f - sig));
            }
            break;
        }
        case CAUSA_ACT_EXP:
            for (int64_t i = 0; i < n; i++) {
                gi[i] += go[i] * expf(di[i]);
            }
            break;
        default:
            for (int64_t i = 0; i < n; i++) gi[i] += go[i];
            break;
    }
}

void causa_tensor_matmul_backward(causa_tensor_t* a, causa_tensor_t* b, const causa_tensor_t* grad_output) {
    if (!grad_output || !a || !b) return;

    if (a->ndim != 2 || b->ndim != 2) return;

    int64_t M = a->shape[0];
    int64_t K = a->shape[1];
    int64_t N = b->shape[1];

    if (K != b->shape[0]) return;
    if (grad_output->shape[0] != M || grad_output->shape[1] != N) return;

    const float* A = (const float*)a->data;
    const float* B = (const float*)b->data;
    const float* GO = (const float*)grad_output->data;

    // Each side is handled separately: an operand that does not require
    // gradients has no gradient buffer, but the other operand still does.
    if (a->requires_grad) {
        ensure_grad(a);
        if (a->grad) {
            float* GA = (float*)a->grad->data;
            // dA = GO * B^T
            for (int64_t i = 0; i < M; i++) {
                for (int64_t k = 0; k < K; k++) {
                    float sum = 0;
                    for (int64_t j = 0; j < N; j++) {
                        sum += GO[i * N + j] * B[k * N + j];
                    }
                    GA[i * K + k] += sum;
                }
            }
        }
    }

    if (b->requires_grad) {
        ensure_grad(b);
        if (b->grad) {
            float* GB = (float*)b->grad->data;
            // dB = A^T * GO
            for (int64_t k = 0; k < K; k++) {
                for (int64_t j = 0; j < N; j++) {
                    float sum = 0;
                    for (int64_t i = 0; i < M; i++) {
                        sum += A[i * K + k] * GO[i * N + j];
                    }
                    GB[k * N + j] += sum;
                }
            }
        }
    }
}


