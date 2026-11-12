#ifndef CAUSA_TENSOR_H
#define CAUSA_TENSOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CAUSA_VERSION_MAJOR 0
#define CAUSA_VERSION_MINOR 1
#define CAUSA_VERSION_PATCH 0

#define CAUSA_MAX_DIMS 8
#define CAUSA_MAX_NAME 64

typedef enum {
    CAUSA_DEVICE_CPU = 0,
    CAUSA_DEVICE_CUDA = 1,
    CAUSA_DEVICE_METAL = 2,
    CAUSA_DEVICE_VULKAN = 3,
    CAUSA_DEVICE_NEUROMORPHIC = 4,
    CAUSA_DEVICE_BARE = 5,
    CAUSA_DEVICE_COUNT
} causa_device_type_t;

typedef enum {
    CAUSA_DTYPE_F64 = 0,
    CAUSA_DTYPE_F32 = 1,
    CAUSA_DTYPE_F16 = 2,
    CAUSA_DTYPE_BF16 = 3,
    CAUSA_DTYPE_I64 = 4,
    CAUSA_DTYPE_I32 = 5,
    CAUSA_DTYPE_I16 = 6,
    CAUSA_DTYPE_I8 = 7,
    CAUSA_DTYPE_U8 = 8,
    CAUSA_DTYPE_BOOL = 9,
    CAUSA_DTYPE_SPIKE = 10,
    CAUSA_DTYPE_COUNT
} causa_dtype_t;

typedef enum {
    CAUSA_LAYOUT_STRIDED = 0,
    CAUSA_LAYOUT_CONTIGUOUS = 1,
    CAUSA_LAYOUT_CHANNELS_LAST = 2
} causa_layout_t;

typedef struct causa_arena_s causa_arena_t;
typedef struct causa_tensor_s causa_tensor_t;
typedef struct causa_backend_s causa_backend_t;

typedef void (*causa_tensor_dealloc_fn)(causa_tensor_t* tensor);

struct causa_tensor_s {
    void* data;
    causa_tensor_t* grad;
    causa_arena_t* arena;
    causa_backend_t* backend;
    causa_device_type_t device;
    causa_dtype_t dtype;
    causa_layout_t layout;
    int32_t ndim;
    int32_t refcount;
    int64_t shape[CAUSA_MAX_DIMS];
    int64_t strides[CAUSA_MAX_DIMS];
    int64_t offset;
    bool requires_grad;
    bool is_view;
    causa_tensor_dealloc_fn dealloc;
    char name[CAUSA_MAX_NAME];
    void* autograd_ctx;
};

typedef struct {
    causa_dtype_t dtype;
    causa_device_type_t device;
    causa_layout_t layout;
    bool requires_grad;
    const char* name;
} causa_tensor_options_t;

#define CAUSA_TENSOR_OPTIONS_DEFAULTS { \
    .dtype = CAUSA_DTYPE_F32, \
    .device = CAUSA_DEVICE_CPU, \
    .layout = CAUSA_LAYOUT_CONTIGUOUS, \
    .requires_grad = false, \
    .name = NULL \
}

size_t causa_dtype_size(causa_dtype_t dtype);
const char* causa_dtype_name(causa_dtype_t dtype);
const char* causa_device_name(causa_device_type_t device);

causa_tensor_t* causa_tensor_new(
    causa_arena_t* arena,
    causa_backend_t* backend,
    const int64_t* shape,
    int32_t ndim,
    const causa_tensor_options_t* options
);

causa_tensor_t* causa_tensor_new_from_data(
    causa_arena_t* arena,
    causa_backend_t* backend,
    void* data,
    const int64_t* shape,
    int32_t ndim,
    const causa_tensor_options_t* options
);

causa_tensor_t* causa_tensor_view(
    causa_tensor_t* base,
    const int64_t* shape,
    int32_t ndim,
    const int64_t* strides,
    int64_t offset
);

causa_tensor_t* causa_tensor_slice(
    causa_tensor_t* tensor,
    const int64_t* starts,
    const int64_t* ends,
    const int64_t* steps,
    int32_t ndim
);

causa_tensor_t* causa_tensor_transpose(causa_tensor_t* tensor, int32_t dim0, int32_t dim1);
causa_tensor_t* causa_tensor_permute(causa_tensor_t* tensor, const int32_t* dims, int32_t ndim);
causa_tensor_t* causa_tensor_reshape(causa_tensor_t* tensor, const int64_t* shape, int32_t ndim);
causa_tensor_t* causa_tensor_squeeze(causa_tensor_t* tensor, int32_t dim);
causa_tensor_t* causa_tensor_unsqueeze(causa_tensor_t* tensor, int32_t dim);

void causa_tensor_retain(causa_tensor_t* tensor);
void causa_tensor_release(causa_tensor_t* tensor);

void causa_tensor_zero_grad(causa_tensor_t* tensor);
void causa_tensor_backward(causa_tensor_t* tensor, causa_tensor_t* grad_output);

bool causa_tensor_is_contiguous(const causa_tensor_t* tensor);
int64_t causa_tensor_numel(const causa_tensor_t* tensor);
size_t causa_tensor_nbytes(const causa_tensor_t* tensor);

causa_tensor_t* causa_tensor_to_device(causa_tensor_t* tensor, causa_device_type_t device);
causa_tensor_t* causa_tensor_to_dtype(causa_tensor_t* tensor, causa_dtype_t dtype);

void causa_tensor_fill(causa_tensor_t* tensor, double value);
void causa_tensor_copy(causa_tensor_t* dst, const causa_tensor_t* src);

typedef enum {
    CAUSA_INIT_UNIFORM = 0,
    CAUSA_INIT_NORMAL = 1,
    CAUSA_INIT_XAVIER_UNIFORM = 2,
    CAUSA_INIT_XAVIER_NORMAL = 3,
    CAUSA_INIT_KAIMING_UNIFORM = 4,
    CAUSA_INIT_KAIMING_NORMAL = 5,
    CAUSA_INIT_ORTHOGONAL = 6,
    CAUSA_INIT_ZEROS = 7,
    CAUSA_INIT_ONES = 8,
    CAUSA_INIT_EYE = 9
} causa_init_t;

// Fill a tensor according to one of the initialisations.
//
// The parameters mean different things for different initialisations:
//
//   ZEROS, ONES, ORTHOGONAL and EYE take none.
//
//   UNIFORM takes the low and the high value.
//
//   NORMAL takes the mean and the standard deviation.
//
//   XAVIER_UNIFORM, XAVIER_NORMAL, KAIMING_UNIFORM and KAIMING_NORMAL take
//   the gain the scale is multiplied by and the mean. The scale itself
//   comes from the shape of the tensor: Xavier from the number of values
//   that come in and the number that go out, and Kaiming from the number
//   that come in.
void causa_tensor_init(causa_tensor_t* tensor, causa_init_t init, double param1, double param2);

typedef enum {
    CAUSA_REDUCE_SUM = 0,
    CAUSA_REDUCE_MEAN = 1,
    CAUSA_REDUCE_MAX = 2,
    CAUSA_REDUCE_MIN = 3,
    CAUSA_REDUCE_PROD = 4,
    CAUSA_REDUCE_NORM_L1 = 5,
    CAUSA_REDUCE_NORM_L2 = 6
} causa_reduce_op_t;

// Read one element of a tensor as a float.
//
// The index counts elements of the tensor's own type. The value is
// converted, so a tensor of any element type can be read this way.
float causa_tensor_load_f32(const causa_tensor_t* t, int64_t index);

// Write one element of a tensor from a float.
//
// The value is converted to the tensor's own type.
void causa_tensor_store_f32(causa_tensor_t* t, int64_t index, float value);

causa_tensor_t* causa_tensor_reduce(
    const causa_tensor_t* input,
    causa_reduce_op_t op,
    const int32_t* dims,
    int32_t num_dims,
    bool keepdim
);

typedef enum {
    CAUSA_CMP_EQ = 0,
    CAUSA_CMP_NE = 1,
    CAUSA_CMP_LT = 2,
    CAUSA_CMP_LE = 3,
    CAUSA_CMP_GT = 4,
    CAUSA_CMP_GE = 5
} causa_cmp_op_t;

// Compare two tensors element by element.
//
// The result has one byte per element: a byte that is not zero where the
// comparison holds and a byte that is zero where it does not. The two
// tensors have to have the same shape.
causa_tensor_t* causa_tensor_compare(
    const causa_tensor_t* a,
    const causa_tensor_t* b,
    causa_cmp_op_t op
);

typedef enum {
    CAUSA_ACT_RELU = 0,
    CAUSA_ACT_GELU = 1,
    CAUSA_ACT_SILU = 2,
    CAUSA_ACT_TANH = 3,
    CAUSA_ACT_SIGMOID = 4,
    CAUSA_ACT_SOFTMAX = 5,
    CAUSA_ACT_LOG_SOFTMAX = 6,
    CAUSA_ACT_ELU = 7,
    CAUSA_ACT_LEAKY_RELU = 8,
    CAUSA_ACT_SWISH = 9,
    CAUSA_ACT_EXP = 10
} causa_activation_t;

causa_tensor_t* causa_tensor_activation(const causa_tensor_t* input, causa_activation_t act, double param);

// Backward pass functions (accumulate gradients into input->grad)
// Note: input tensors are non-const because we accumulate into their grad buffers
void causa_tensor_add_backward(causa_tensor_t* a, causa_tensor_t* b, const causa_tensor_t* grad_output);
void causa_tensor_sub_backward(causa_tensor_t* a, causa_tensor_t* b, const causa_tensor_t* grad_output);
void causa_tensor_mul_backward(causa_tensor_t* a, causa_tensor_t* b, const causa_tensor_t* grad_output);
void causa_tensor_div_backward(causa_tensor_t* a, causa_tensor_t* b, const causa_tensor_t* grad_output);

void causa_tensor_add_scalar_backward(causa_tensor_t* a, double b, const causa_tensor_t* grad_output);
void causa_tensor_sub_scalar_backward(causa_tensor_t* a, double b, const causa_tensor_t* grad_output);
void causa_tensor_mul_scalar_backward(causa_tensor_t* a, double b, const causa_tensor_t* grad_output);
void causa_tensor_div_scalar_backward(causa_tensor_t* a, double b, const causa_tensor_t* grad_output);

void causa_tensor_neg_backward(causa_tensor_t* a, const causa_tensor_t* grad_output);
void causa_tensor_abs_backward(causa_tensor_t* a, const causa_tensor_t* grad_output);
void causa_tensor_sqrt_backward(causa_tensor_t* a, const causa_tensor_t* grad_output);
void causa_tensor_rsqrt_backward(causa_tensor_t* a, const causa_tensor_t* grad_output);
void causa_tensor_exp_backward(causa_tensor_t* a, const causa_tensor_t* grad_output);
void causa_tensor_log_backward(causa_tensor_t* a, const causa_tensor_t* grad_output);
void causa_tensor_pow_backward(causa_tensor_t* a, double exp, const causa_tensor_t* grad_output);

void causa_tensor_activation_backward(causa_tensor_t* input, causa_activation_t act, double param, const causa_tensor_t* grad_output);

void causa_tensor_matmul_backward(causa_tensor_t* a, causa_tensor_t* b, const causa_tensor_t* grad_output);

// Accumulate a gradient into a tensor's grad buffer (creates the buffer if needed)
void causa_tensor_accumulate_grad(causa_tensor_t* t, const causa_tensor_t* grad);

// Broadcast two shapes against each other, numpy style.
// Returns the number of dimensions of the result, or -1 if incompatible.
// out_shape must have room for CAUSA_MAX_DIMS entries.
int32_t causa_tensor_broadcast_shape(
    const int64_t* a_shape, int32_t a_ndim,
    const int64_t* b_shape, int32_t b_ndim,
    int64_t* out_shape
);

// True when the two shapes broadcast to a common shape.
bool causa_tensor_shapes_broadcastable(
    const int64_t* a_shape, int32_t a_ndim,
    const int64_t* b_shape, int32_t b_ndim
);

// Transpose backward: gradient w.r.t. the input is the output gradient transposed
void causa_tensor_transpose_backward(
    causa_tensor_t* a,
    const causa_tensor_t* grad_output,
    int32_t dim0,
    int32_t dim1
);

// Extract the diagonal of a square 2D tensor into a 1D tensor
causa_tensor_t* causa_tensor_diag(const causa_tensor_t* a);

// Diagonal backward: scatter the 1D gradient back onto the diagonal
void causa_tensor_diag_backward(causa_tensor_t* a, const causa_tensor_t* grad_output);

causa_tensor_t* causa_tensor_matmul(const causa_tensor_t* a, const causa_tensor_t* b);
causa_tensor_t* causa_tensor_bmm(const causa_tensor_t* a, const causa_tensor_t* b);

causa_tensor_t* causa_tensor_conv1d(
    const causa_tensor_t* input,
    const causa_tensor_t* weight,
    const causa_tensor_t* bias,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    int32_t groups
);

causa_tensor_t* causa_tensor_conv2d(
    const causa_tensor_t* input,
    const causa_tensor_t* weight,
    const causa_tensor_t* bias,
    const int32_t stride[2],
    const int32_t padding[2],
    const int32_t dilation[2],
    int32_t groups
);

causa_tensor_t* causa_tensor_add(const causa_tensor_t* a, const causa_tensor_t* b);
causa_tensor_t* causa_tensor_sub(const causa_tensor_t* a, const causa_tensor_t* b);
causa_tensor_t* causa_tensor_mul(const causa_tensor_t* a, const causa_tensor_t* b);
causa_tensor_t* causa_tensor_div(const causa_tensor_t* a, const causa_tensor_t* b);

causa_tensor_t* causa_tensor_add_scalar(const causa_tensor_t* a, double b);
causa_tensor_t* causa_tensor_sub_scalar(const causa_tensor_t* a, double b);
causa_tensor_t* causa_tensor_mul_scalar(const causa_tensor_t* a, double b);
causa_tensor_t* causa_tensor_div_scalar(const causa_tensor_t* a, double b);

causa_tensor_t* causa_tensor_neg(const causa_tensor_t* a);
causa_tensor_t* causa_tensor_abs(const causa_tensor_t* a);
causa_tensor_t* causa_tensor_sqrt(const causa_tensor_t* a);
causa_tensor_t* causa_tensor_rsqrt(const causa_tensor_t* a);
causa_tensor_t* causa_tensor_exp(const causa_tensor_t* a);
causa_tensor_t* causa_tensor_log(const causa_tensor_t* a);
causa_tensor_t* causa_tensor_pow(const causa_tensor_t* a, double exp);

causa_tensor_t* causa_tensor_sum(const causa_tensor_t* a, const causa_tensor_t* b);
causa_tensor_t* causa_tensor_mean(const causa_tensor_t* a, const causa_tensor_t* b);

int causa_tensor_print(const causa_tensor_t* tensor, int max_elements);
int causa_tensor_fprint(FILE* stream, const causa_tensor_t* tensor, int max_elements);

#ifdef __cplusplus
}
#endif

#endif