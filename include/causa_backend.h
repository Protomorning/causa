#ifndef CAUSA_BACKEND_H
#define CAUSA_BACKEND_H

#include "causa_tensor.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct causa_backend_s causa_backend_t;

typedef void (*causa_backend_init_fn)(causa_backend_t* backend);
typedef void (*causa_backend_deinit_fn)(causa_backend_t* backend);
typedef void* (*causa_backend_alloc_fn)(causa_backend_t* backend, size_t size, size_t alignment);
typedef void (*causa_backend_free_fn)(causa_backend_t* backend, void* ptr);
typedef void (*causa_backend_copy_fn)(causa_backend_t* backend, void* dst, const void* src, size_t nbytes);
typedef void (*causa_backend_sync_fn)(causa_backend_t* backend);

typedef void (*causa_backend_fill_fn)(causa_backend_t* backend, causa_tensor_t* tensor, double value);
typedef void (*causa_backend_copy_tensor_fn)(causa_backend_t* backend, causa_tensor_t* dst, const causa_tensor_t* src);

typedef causa_tensor_t* (*causa_backend_add_fn)(causa_backend_t* backend, const causa_tensor_t* a, const causa_tensor_t* b);
typedef causa_tensor_t* (*causa_backend_mul_fn)(causa_backend_t* backend, const causa_tensor_t* a, const causa_tensor_t* b);
typedef causa_tensor_t* (*causa_backend_matmul_fn)(causa_backend_t* backend, const causa_tensor_t* a, const causa_tensor_t* b);

typedef causa_tensor_t* (*causa_backend_activation_fn)(causa_backend_t* backend, const causa_tensor_t* input, causa_activation_t act, double param);
typedef causa_tensor_t* (*causa_backend_reduce_fn)(causa_backend_t* backend, const causa_tensor_t* input, causa_reduce_op_t op, const int32_t* dims, int32_t num_dims, bool keepdim);
typedef causa_tensor_t* (*causa_backend_conv2d_fn)(causa_backend_t* backend, const causa_tensor_t* input, const causa_tensor_t* weight, const causa_tensor_t* bias, const int32_t stride[2], const int32_t padding[2], const int32_t dilation[2], int32_t groups);

struct causa_backend_s {
    causa_device_type_t device_type;
    char name[64];

    causa_backend_init_fn init;
    causa_backend_deinit_fn deinit;
    causa_backend_alloc_fn alloc;
    causa_backend_free_fn free;
    causa_backend_copy_fn copy;
    causa_backend_sync_fn sync;

    causa_backend_fill_fn fill;
    causa_backend_copy_tensor_fn copy_tensor;

    causa_backend_add_fn add;
    causa_backend_mul_fn mul;
    causa_backend_matmul_fn matmul;

    causa_backend_activation_fn activation;
    causa_backend_reduce_fn reduce;
    causa_backend_conv2d_fn conv2d;

    void* device_ctx;
    void* user_ctx;
};

causa_backend_t* causa_backend_cpu_new(void);
causa_backend_t* causa_backend_cuda_new(void);
void causa_backend_free(causa_backend_t* backend);

int causa_backend_init(causa_backend_t* backend);
void causa_backend_deinit(causa_backend_t* backend);

void* causa_backend_alloc(causa_backend_t* backend, size_t size, size_t alignment);
void causa_backend_free_ptr(causa_backend_t* backend, void* ptr);
void causa_backend_copy(causa_backend_t* backend, void* dst, const void* src, size_t nbytes);
void causa_backend_sync(causa_backend_t* backend);

void causa_backend_fill(causa_backend_t* backend, causa_tensor_t* tensor, double value);
void causa_backend_copy_tensor(causa_backend_t* backend, causa_tensor_t* dst, const causa_tensor_t* src);

causa_tensor_t* causa_backend_add(causa_backend_t* backend, const causa_tensor_t* a, const causa_tensor_t* b);
causa_tensor_t* causa_backend_mul(causa_backend_t* backend, const causa_tensor_t* a, const causa_tensor_t* b);
causa_tensor_t* causa_backend_matmul(causa_backend_t* backend, const causa_tensor_t* a, const causa_tensor_t* b);

causa_tensor_t* causa_backend_activation(causa_backend_t* backend, const causa_tensor_t* input, causa_activation_t act, double param);
causa_tensor_t* causa_backend_reduce(causa_backend_t* backend, const causa_tensor_t* input, causa_reduce_op_t op, const int32_t* dims, int32_t num_dims, bool keepdim);
causa_tensor_t* causa_backend_conv2d(causa_backend_t* backend, const causa_tensor_t* input, const causa_tensor_t* weight, const causa_tensor_t* bias, const int32_t stride[2], const int32_t padding[2], const int32_t dilation[2], int32_t groups);

#ifdef __cplusplus
}
#endif

#endif