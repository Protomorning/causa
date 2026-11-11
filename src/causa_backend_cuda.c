#include "causa_cuda.h"
#include "causa_tensor.h"
#include "causa_arena.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

typedef struct {
    cublasHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;
    cudaStream_t stream;
    int device_id;
    size_t workspace_size;
    void* workspace;
} cuda_backend_ctx_t;

static cudaError_t cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
    }
    return err;
}

#define CUDA_CHECK(err) cuda_check(err, __FILE__, __LINE__)

static cublasStatus_t cublas_check(cublasStatus_t err, const char* file, int line) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", file, line, err);
    }
    return err;
}

#define CUBLAS_CHECK(err) cublas_check(err, __FILE__, __LINE__)

static cudnnStatus_t cudnn_check(cudnnStatus_t err, const char* file, int line) {
    if (err != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "cuDNN error at %s:%d: %d\n", file, line, err);
    }
    return err;
}

#define CUDNN_CHECK(err) cudnn_check(err, __FILE__, __LINE__)

__global__ void add_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

__global__ void mul_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}

__global__ void sub_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] - b[idx];
}

__global__ void div_kernel(const float* a, const float* b, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] / (b[idx] + 1e-8f);
}

__global__ void neg_kernel(const float* a, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = -a[idx];
}

__global__ void abs_kernel(const float* a, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fabsf(a[idx]);
}

__global__ void sqrt_kernel(const float* a, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = sqrtf(fmaxf(0.0f, a[idx]));
}

__global__ void rsqrt_kernel(const float* a, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = 1.0f / sqrtf(fmaxf(1e-8f, a[idx]));
}

__global__ void exp_kernel(const float* a, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = expf(a[idx]);
}

__global__ void log_kernel(const float* a, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = logf(fmaxf(1e-8f, a[idx]));
}

__global__ void relu_kernel(const float* a, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fmaxf(0.0f, a[idx]);
}

__global__ void gelu_kernel(const float* a, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        out[idx] = x * cdf;
    }
}

__global__ void silu_kernel(const float* a, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        out[idx] = x * sig;
    }
}

__global__ void tanh_kernel(const float* a, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = tanhf(a[idx]);
}

__global__ void sigmoid_kernel(const float* a, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = 1.0f / (1.0f + expf(-a[idx]));
}

__global__ void elu_kernel(const float* a, float* out, float alpha, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] >= 0 ? a[idx] : alpha * (expf(a[idx]) - 1.0f);
}

__global__ void leaky_relu_kernel(const float* a, float* out, float alpha, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] >= 0 ? a[idx] : alpha * a[idx];
}

__global__ void swish_kernel(const float* a, float* out, float beta, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        float sig = 1.0f / (1.0f + expf(-beta * x));
        out[idx] = x * sig;
    }
}

__global__ void fill_kernel(float* data, float value, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = value;
}

__global__ void copy_kernel(const float* src, float* dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

static inline int div_ceil(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

static void* cuda_alloc(cuda_backend_ctx_t* ctx, size_t size, size_t alignment) {
    void* ptr = NULL;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

static void cuda_free(cuda_backend_ctx_t* ctx, void* ptr) {
    if (ptr) CUDA_CHECK(cudaFree(ptr));
}

static void cuda_copy(cuda_backend_ctx_t* ctx, void* dst, const void* src, size_t nbytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice));
}

static void cuda_sync(cuda_backend_ctx_t* ctx) {
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

static void cuda_fill(causa_backend_t* backend, causa_tensor_t* tensor, double value) {
    cuda_backend_ctx_t* ctx = (cuda_backend_ctx_t*)backend->device_ctx;
    if (!tensor || tensor->device != CAUSA_DEVICE_CUDA) return;

    int64_t n = causa_tensor_numel(tensor);
    if (n == 0) return;

    int block = 256;
    int grid = div_ceil(n, block);
    float v = (float)value;

    fill_kernel<<<grid, block, 0, ctx->stream>>>((float*)tensor->data, v, n);
}

static void cuda_copy_tensor(causa_backend_t* backend, causa_tensor_t* dst, const causa_tensor_t* src) {
    cuda_backend_ctx_t* ctx = (cuda_backend_ctx_t*)backend->device_ctx;
    if (!dst || !src) return;
    if (causa_tensor_numel(dst) != causa_tensor_numel(src)) return;

    int64_t nbytes = causa_tensor_nbytes(src);
    cudaMemcpyAsync(dst->data, src->data, nbytes, cudaMemcpyDeviceToDevice, ctx->stream);
}

static causa_tensor_t* cuda_elementwise_binary(causa_backend_t* backend, const causa_tensor_t* a, const causa_tensor_t* b,
    void (*kernel)(const float*, const float*, float*, int64_t)) {
    if (!a || !b || a->dtype != b->dtype || causa_tensor_numel(a) != causa_tensor_numel(b)) return NULL;

    cuda_backend_ctx_t* ctx = (cuda_backend_ctx_t*)backend->device_ctx;
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = a->dtype;
    opts.device = a->device;
    opts.layout = a->layout;
    opts.requires_grad = a->requires_grad || b->requires_grad;
    causa_tensor_t* out = causa_tensor_new(a->arena, a->backend, a->shape, a->ndim, &opts);
    if (!out) return NULL;

    int64_t n = causa_tensor_numel(a);
    int block = 256;
    int grid = div_ceil(n, block);

    kernel<<<grid, block, 0, ctx->stream>>>((const float*)a->data, (const float*)b->data, (float*)out->data, n);
    return out;
}

static causa_tensor_t* cuda_elementwise_unary(causa_backend_t* backend, const causa_tensor_t* a,
    void (*kernel)(const float*, float*, int64_t)) {
    if (!a) return NULL;

    cuda_backend_ctx_t* ctx = (cuda_backend_ctx_t*)backend->device_ctx;
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = a->dtype;
    opts.device = a->device;
    opts.layout = a->layout;
    opts.requires_grad = a->requires_grad;
    causa_tensor_t* out = causa_tensor_new(a->arena, a->backend, a->shape, a->ndim, &opts);
    if (!out) return NULL;

    int64_t n = causa_tensor_numel(a);
    int block = 256;
    int grid = div_ceil(n, block);

    kernel<<<grid, block, 0, ctx->stream>>>((const float*)a->data, (float*)out->data, n);
    return out;
}

static causa_tensor_t* cuda_elementwise_unary_param(causa_backend_t* backend, const causa_tensor_t* a, float param,
    void (*kernel)(const float*, float*, float, int64_t)) {
    if (!a) return NULL;

    cuda_backend_ctx_t* ctx = (cuda_backend_ctx_t*)backend->device_ctx;
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = a->dtype;
    opts.device = a->device;
    opts.layout = a->layout;
    opts.requires_grad = a->requires_grad;
    causa_tensor_t* out = causa_tensor_new(a->arena, a->backend, a->shape, a->ndim, &opts);
    if (!out) return NULL;

    int64_t n = causa_tensor_numel(a);
    int block = 256;
    int grid = div_ceil(n, block);

    kernel<<<grid, block, 0, ctx->stream>>>((const float*)a->data, (float*)out->data, param, n);
    return out;
}

static causa_tensor_t* cuda_add(causa_backend_t* backend, const causa_tensor_t* a, const causa_tensor_t* b) {
    return cuda_elementwise_binary(backend, a, b, add_kernel);
}

static causa_tensor_t* cuda_mul(causa_backend_t* backend, const causa_tensor_t* a, const causa_tensor_t* b) {
    return cuda_elementwise_binary(backend, a, b, mul_kernel);
}

static causa_tensor_t* cuda_matmul(causa_backend_t* backend, const causa_tensor_t* a, const causa_tensor_t* b) {
    if (!a || !b || a->dtype != b->dtype) return NULL;
    if (a->ndim < 2 || b->ndim < 2) return NULL;
    if (a->shape[a->ndim - 1] != b->shape[b->ndim - 2]) return NULL;

    cuda_backend_ctx_t* ctx = (cuda_backend_ctx_t*)backend->device_ctx;

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

        float alpha = 1.0f;
        float beta = 0.0f;

        int64_t stride_a_batch = (a->ndim > 2) ? a->strides[a->ndim - 3] / sizeof(float) : 0;
        int64_t stride_b_batch = (b->ndim > 2) ? b->strides[b->ndim - 3] / sizeof(float) : 0;
        int64_t stride_out_batch = (out_ndim > 2) ? out->strides[out_ndim - 3] / sizeof(float) : 0;

        for (int64_t batch_idx = 0; batch_idx < batch; batch_idx++) {
            const float* a_ptr = A + batch_idx * stride_a_batch;
            const float* b_ptr = B + batch_idx * stride_b_batch;
            float* c_ptr = C + batch_idx * stride_out_batch;

            int lda = a->strides[a->ndim - 2] / sizeof(float);
            int ldb = b->strides[b->ndim - 2] / sizeof(float);
            int ldc = out->strides[out_ndim - 2] / sizeof(float);

            CUBLAS_CHECK(cublasSgemm(ctx->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, b_ptr, ldb, a_ptr, lda, &beta, c_ptr, ldc));
        }
    }

    return out;
}

static causa_tensor_t* cuda_activation(causa_backend_t* backend, const causa_tensor_t* input, causa_activation_t act, double param) {
    if (!input) return NULL;

    cuda_backend_ctx_t* ctx = (cuda_backend_ctx_t*)backend->device_ctx;
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = input->dtype;
    opts.device = input->device;
    opts.layout = input->layout;
    opts.requires_grad = input->requires_grad;
    causa_tensor_t* out = causa_tensor_new(input->arena, input->backend, input->shape, input->ndim, &opts);
    if (!out) return NULL;

    int64_t n = causa_tensor_numel(input);
    int block = 256;
    int grid = div_ceil(n, block);
    float p = (float)param;

    switch (act) {
        case CAUSA_ACT_RELU:
            relu_kernel<<<grid, block, 0, ctx->stream>>>((const float*)input->data, (float*)out->data, n);
            break;
        case CAUSA_ACT_GELU:
            gelu_kernel<<<grid, block, 0, ctx->stream>>>((const float*)input->data, (float*)out->data, n);
            break;
        case CAUSA_ACT_SILU:
            silu_kernel<<<grid, block, 0, ctx->stream>>>((const float*)input->data, (float*)out->data, n);
            break;
        case CAUSA_ACT_TANH:
            tanh_kernel<<<grid, block, 0, ctx->stream>>>((const float*)input->data, (float*)out->data, n);
            break;
        case CAUSA_ACT_SIGMOID:
            sigmoid_kernel<<<grid, block, 0, ctx->stream>>>((const float*)input->data, (float*)out->data, n);
            break;
        case CAUSA_ACT_ELU:
            elu_kernel<<<grid, block, 0, ctx->stream>>>((const float*)input->data, (float*)out->data, (float)param, n);
            break;
        case CAUSA_ACT_LEAKY_RELU:
            leaky_relu_kernel<<<grid, block, 0, ctx->stream>>>((const float*)input->data, (float*)out->data, (float)param, n);
            break;
        case CAUSA_ACT_SWISH:
            swish_kernel<<<grid, block, 0, ctx->stream>>>((const float*)input->data, (float*)out->data, (float)param, n);
            break;
        case CAUSA_ACT_SOFTMAX:
        case CAUSA_ACT_LOG_SOFTMAX:
        default:
            // Fallback to CPU for complex activations
            return causa_tensor_activation(input, act, param);
    }

    return out;
}

static causa_tensor_t* cuda_reduce(causa_backend_t* backend, const causa_tensor_t* input, causa_reduce_op_t op, const int32_t* dims, int32_t num_dims, bool keepdim) {
    // For now, fall back to CPU implementation
    (void)backend; (void)input; (void)op; (void)dims; (void)num_dims; (void)keepdim;
    return NULL;
}

static causa_tensor_t* cuda_conv2d(causa_backend_t* backend, const causa_tensor_t* input, const causa_tensor_t* weight, const causa_tensor_t* bias, const int32_t stride[2], const int32_t padding[2], const int32_t dilation[2], int32_t groups) {
    (void)backend; (void)input; (void)weight; (void)bias; (void)stride; (void)padding; (void)dilation; (void)groups;
    return NULL;
}

static void cuda_backend_init(causa_backend_t* backend) {
    cuda_backend_ctx_t* ctx = (cuda_backend_ctx_t*)malloc(sizeof(cuda_backend_ctx_t));
    memset(ctx, 0, sizeof(cuda_backend_ctx_t));
    ctx->device_id = 0;
    CUDA_CHECK(cudaSetDevice(ctx->device_id));
    CUDA_CHECK(cudaStreamCreate(&ctx->stream));
    CUBLAS_CHECK(cublasCreate(&ctx->cublas_handle));
    CUBLAS_CHECK(cublasSetStream(ctx->cublas_handle, ctx->stream));
    CUDNN_CHECK(cudnnCreate(&ctx->cudnn_handle));
    CUDNN_CHECK(cudnnSetStream(ctx->cudnn_handle, ctx->stream));
    backend->device_ctx = ctx;
}

static void cuda_backend_deinit(causa_backend_t* backend) {
    cuda_backend_ctx_t* ctx = (cuda_backend_ctx_t*)backend->device_ctx;
    if (ctx) {
        if (ctx->cublas_handle) cublasDestroy(ctx->cublas_handle);
        if (ctx->cudnn_handle) cudnnDestroy(ctx->cudnn_handle);
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
        free(ctx);
        backend->device_ctx = NULL;
    }
}

causa_backend_t* causa_backend_cuda_new(void) {
    causa_backend_t* backend = (causa_backend_t*)malloc(sizeof(causa_backend_t));
    if (!backend) return NULL;

    memset(backend, 0, sizeof(causa_backend_t));
    backend->device_type = CAUSA_DEVICE_CUDA;
    strncpy(backend->name, "CUDA", 63);

    backend->init = cuda_backend_init;
    backend->deinit = cuda_backend_deinit;
    backend->alloc = (void*(*)(causa_backend_t*, size_t, size_t))cuda_alloc;
    backend->free = (void(*)(causa_backend_t*, void*))cuda_free;
    backend->copy = (void(*)(causa_backend_t*, void*, const void*, size_t))cuda_copy;
    backend->sync = (void(*)(causa_backend_t*))cuda_sync;

    backend->fill = cuda_fill;
    backend->copy_tensor = cuda_copy_tensor;

    backend->add = cuda_add;
    backend->mul = cuda_mul;
    backend->matmul = cuda_matmul;

    backend->activation = cuda_activation;
    backend->reduce = cuda_reduce;
    backend->conv2d = cuda_conv2d;

    return backend;
}

#else
causa_backend_t* causa_backend_cuda_new(void) {
    return NULL;
}
#endif