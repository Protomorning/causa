#include "causa_backend.h"
#include "causa_tensor.h"
#include "causa_arena.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

typedef struct {
    int dummy;
} cpu_backend_ctx_t;

static void cpu_backend_init(causa_backend_t* backend) {
    (void)backend;
}

static void cpu_backend_deinit(causa_backend_t* backend) {
    (void)backend;
}

static void* cpu_backend_alloc(causa_backend_t* backend, size_t size, size_t alignment) {
    (void)backend;
    void* ptr = NULL;
#if defined(_WIN32)
    ptr = _aligned_malloc(size, alignment);
#else
    int ret = posix_memalign(&ptr, alignment, size);
    if (ret != 0) ptr = NULL;
#endif
    return ptr;
}

static void cpu_backend_free(causa_backend_t* backend, void* ptr) {
    (void)backend;
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static void cpu_backend_copy(causa_backend_t* backend, void* dst, const void* src, size_t nbytes) {
    (void)backend;
    memcpy(dst, src, nbytes);
}

static void cpu_backend_sync(causa_backend_t* backend) {
    (void)backend;
}

static void cpu_backend_fill(causa_backend_t* backend, causa_tensor_t* tensor, double value) {
    (void)backend;
    causa_tensor_fill(tensor, value);
}

static void cpu_backend_copy_tensor(causa_backend_t* backend, causa_tensor_t* dst, const causa_tensor_t* src) {
    (void)backend;
    causa_tensor_copy(dst, src);
}

static void add_f32_avx(const float* a, const float* b, float* out, int64_t n) {
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }
    for (; i < n; i++) out[i] = a[i] + b[i];
}

static causa_tensor_t* cpu_backend_add(causa_backend_t* backend, const causa_tensor_t* a, const causa_tensor_t* b) {
    (void)backend;
    if (!a || !b || a->dtype != b->dtype || causa_tensor_numel(a) != causa_tensor_numel(b)) return NULL;

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = a->dtype;
    opts.device = a->device;
    opts.layout = a->layout;
    opts.requires_grad = a->requires_grad || b->requires_grad;
    causa_tensor_t* out = causa_tensor_new(a->arena, a->backend, a->shape, a->ndim, &opts);
    if (!out) return NULL;

    int64_t n = causa_tensor_numel(a);
    if (a->dtype == CAUSA_DTYPE_F32) {
        add_f32_avx((const float*)a->data, (const float*)b->data, (float*)out->data, n);
    } else {
        causa_tensor_t* tmp = causa_tensor_add(a, b);
        if (tmp) causa_tensor_copy(out, tmp);
    }
    return out;
}

static void mul_f32_avx(const float* a, const float* b, float* out, int64_t n) {
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out + i, vr);
    }
    for (; i < n; i++) out[i] = a[i] * b[i];
}

static causa_tensor_t* cpu_backend_mul(causa_backend_t* backend, const causa_tensor_t* a, const causa_tensor_t* b) {
    (void)backend;
    if (!a || !b || a->dtype != b->dtype || causa_tensor_numel(a) != causa_tensor_numel(b)) return NULL;

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = a->dtype;
    opts.device = a->device;
    opts.layout = a->layout;
    opts.requires_grad = a->requires_grad || b->requires_grad;
    causa_tensor_t* out = causa_tensor_new(a->arena, a->backend, a->shape, a->ndim, &opts);
    if (!out) return NULL;

    int64_t n = causa_tensor_numel(a);
    if (a->dtype == CAUSA_DTYPE_F32) {
        mul_f32_avx((const float*)a->data, (const float*)b->data, (float*)out->data, n);
    } else {
        causa_tensor_t* tmp = causa_tensor_mul(a, b);
        if (tmp) causa_tensor_copy(out, tmp);
    }
    return out;
}

static causa_tensor_t* cpu_backend_matmul(causa_backend_t* backend, const causa_tensor_t* a, const causa_tensor_t* b) {
    (void)backend;
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
    opts.dtype = a->dtype;
    opts.device = a->device;
    opts.layout = a->layout;
    opts.requires_grad = a->requires_grad || b->requires_grad;
    causa_tensor_t* out = causa_tensor_new(a->arena, a->backend, out_shape, out_ndim, &opts);
    if (!out) return NULL;

    if (a->dtype == CAUSA_DTYPE_F32) {
        const float* A = (const float*)a->data;
        const float* B = (const float*)b->data;
        float* C = (float*)out->data;

        int64_t stride_a_batch = (a->ndim > 2) ? a->strides[a->ndim - 3] : 0;
        int64_t stride_b_batch = (b->ndim > 2) ? b->strides[b->ndim - 3] : 0;
        int64_t stride_out_batch = (out_ndim > 2) ? out->strides[out_ndim - 3] : 0;

        for (int64_t batch_idx = 0; batch_idx < batch; batch_idx++) {
            const float* a_ptr = A + batch_idx * stride_a_batch;
            const float* b_ptr = B + batch_idx * stride_b_batch;
            float* c_ptr = C + batch_idx * stride_out_batch;

            int64_t lda = a->strides[a->ndim - 2];
            int64_t ldb = b->strides[b->ndim - 2];
            int64_t ldc = out->strides[out_ndim - 2];

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

static causa_tensor_t* cpu_backend_activation(causa_backend_t* backend, const causa_tensor_t* input, causa_activation_t act, double param) {
    (void)backend;
    return causa_tensor_activation(input, act, param);
}

static causa_tensor_t* cpu_backend_reduce(causa_backend_t* backend, const causa_tensor_t* input, causa_reduce_op_t op, const int32_t* dims, int32_t num_dims, bool keepdim) {
    (void)backend;
    return causa_tensor_reduce(input, op, dims, num_dims, keepdim);
}

static causa_tensor_t* cpu_backend_conv2d(causa_backend_t* backend, const causa_tensor_t* input, const causa_tensor_t* weight, const causa_tensor_t* bias, const int32_t stride[2], const int32_t padding[2], const int32_t dilation[2], int32_t groups) {
    (void)backend;
    return causa_tensor_conv2d(input, weight, bias, stride, padding, dilation, groups);
}

causa_backend_t* causa_backend_cpu_new(void) {
    causa_backend_t* backend = (causa_backend_t*)malloc(sizeof(causa_backend_t));
    if (!backend) return NULL;

    memset(backend, 0, sizeof(causa_backend_t));
    backend->device_type = CAUSA_DEVICE_CPU;
    strncpy(backend->name, "CPU", 63);

    backend->init = cpu_backend_init;
    backend->deinit = cpu_backend_deinit;
    backend->alloc = cpu_backend_alloc;
    backend->free = cpu_backend_free;
    backend->copy = cpu_backend_copy;
    backend->sync = cpu_backend_sync;

    backend->fill = cpu_backend_fill;
    backend->copy_tensor = cpu_backend_copy_tensor;

    backend->add = cpu_backend_add;
    backend->mul = cpu_backend_mul;
    backend->matmul = cpu_backend_matmul;

    backend->activation = cpu_backend_activation;
    backend->reduce = cpu_backend_reduce;
    backend->conv2d = cpu_backend_conv2d;

    backend->device_ctx = malloc(sizeof(cpu_backend_ctx_t));

    return backend;
}

void causa_backend_free(causa_backend_t* backend) {
    if (!backend) return;
    if (backend->deinit) backend->deinit(backend);
    free(backend->device_ctx);
    free(backend);
}

int causa_backend_init(causa_backend_t* backend) {
    if (!backend || !backend->init) return -1;
    backend->init(backend);
    return 0;
}

void causa_backend_deinit(causa_backend_t* backend) {
    if (backend && backend->deinit) backend->deinit(backend);
}

void* causa_backend_alloc(causa_backend_t* backend, size_t size, size_t alignment) {
    if (!backend || !backend->alloc) return NULL;
    return backend->alloc(backend, size, alignment);
}

void causa_backend_free_ptr(causa_backend_t* backend, void* ptr) {
    if (backend && backend->free) backend->free(backend, ptr);
}

void causa_backend_copy(causa_backend_t* backend, void* dst, const void* src, size_t nbytes) {
    if (backend && backend->copy) backend->copy(backend, dst, src, nbytes);
}

void causa_backend_sync(causa_backend_t* backend) {
    if (backend && backend->sync) backend->sync(backend);
}

void causa_backend_fill(causa_backend_t* backend, causa_tensor_t* tensor, double value) {
    if (backend && backend->fill) backend->fill(backend, tensor, value);
}

void causa_backend_copy_tensor(causa_backend_t* backend, causa_tensor_t* dst, const causa_tensor_t* src) {
    if (backend && backend->copy_tensor) backend->copy_tensor(backend, dst, src);
}

causa_tensor_t* causa_backend_add(causa_backend_t* backend, const causa_tensor_t* a, const causa_tensor_t* b) {
    if (!backend || !backend->add) return NULL;
    return backend->add(backend, a, b);
}

causa_tensor_t* causa_backend_mul(causa_backend_t* backend, const causa_tensor_t* a, const causa_tensor_t* b) {
    if (!backend || !backend->mul) return NULL;
    return backend->mul(backend, a, b);
}

causa_tensor_t* causa_backend_matmul(causa_backend_t* backend, const causa_tensor_t* a, const causa_tensor_t* b) {
    if (!backend || !backend->matmul) return NULL;
    return backend->matmul(backend, a, b);
}

causa_tensor_t* causa_backend_activation(causa_backend_t* backend, const causa_tensor_t* input, causa_activation_t act, double param) {
    if (!backend || !backend->activation) return NULL;
    return backend->activation(backend, input, act, param);
}

causa_tensor_t* causa_backend_reduce(causa_backend_t* backend, const causa_tensor_t* input, causa_reduce_op_t op, const int32_t* dims, int32_t num_dims, bool keepdim) {
    if (!backend || !backend->reduce) return NULL;
    return backend->reduce(backend, input, op, dims, num_dims, keepdim);
}

causa_tensor_t* causa_backend_conv2d(causa_backend_t* backend, const causa_tensor_t* input, const causa_tensor_t* weight, const causa_tensor_t* bias, const int32_t stride[2], const int32_t padding[2], const int32_t dilation[2], int32_t groups) {
    if (!backend || !backend->conv2d) return NULL;
    return backend->conv2d(backend, input, weight, bias, stride, padding, dilation, groups);
}