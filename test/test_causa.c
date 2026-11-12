#include "causa.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>

// The number of failures.
//
// The suite tells the shell about them through the exit code. Without that,
// a suite with a failing test would still look like a suite that passed,
// and a broken build could not be told from a good one.
static int g_failures = 0;

// Report a failure.
//
// The message is printed the way it always was, so that the output does not
// change, but the failure is also counted.
static void test_fail(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    g_failures++;
}

void test_arena() {
    printf("Testing arena...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    if (!arena) {
        test_fail("arena creation\n");
        return;
    }

    int* a = causa_arena_alloc_type(arena, int);
    *a = 42;

    float* b = causa_arena_alloc_array(arena, float, 10);
    for (int i = 0; i < 10; i++) b[i] = (float)i;

    causa_arena_checkpoint_t cp = causa_arena_checkpoint(arena);
    int* c = causa_arena_alloc_type(arena, int);
    *c = 100;

    causa_arena_rewind(&cp);
    int* d = causa_arena_alloc_type(arena, int);
    *d = 200;

    if (*a != 42) test_fail("arena allocation\n");
    if (b[5] != 5.0f) test_fail("arena array\n");
    if (*c == 100) test_fail("c should be invalid after rewind (allocated after checkpoint)\n");
    if (*d != 200) test_fail("rewind\n");

    printf("  arena used: %zu bytes\n", causa_arena_used(arena));
    causa_arena_free(arena);
    printf("PASS: arena\n");
}

void test_tensor_basic() {
    printf("Testing tensor basic...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    int64_t shape[] = {2, 3};
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    opts.requires_grad = true;
    opts.name = "test_tensor";

    causa_tensor_t* t = causa_tensor_new(arena, backend, shape, 2, &opts);
    if (!t) {
        test_fail("tensor creation\n");
        return;
    }

    if (t->ndim != 2 || t->shape[0] != 2 || t->shape[1] != 3) {
        test_fail("tensor shape\n");
        return;
    }

    if (causa_tensor_numel(t) != 6) {
        test_fail("tensor numel\n");
        return;
    }

    causa_tensor_fill(t, 1.5f);
    float* data = (float*)t->data;
    if (fabsf(data[0] - 1.5f) > 1e-6) {
        test_fail("tensor fill\n");
        return;
    }

    causa_tensor_t* t2 = causa_tensor_new(arena, backend, shape, 2, &opts);
    causa_tensor_fill(t2, 2.5f);

    causa_tensor_t* t3 = causa_tensor_add(t, t2);
    if (!t3) {
        test_fail("tensor add\n");
        return;
    }

    float* d3 = (float*)t3->data;
    if (fabsf(d3[0] - 4.0f) > 1e-6) {
        test_fail("tensor add value\n");
        return;
    }

    causa_tensor_print(t3, 10);

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: tensor basic\n");
}

void test_tensor_operations() {
    printf("Testing tensor operations...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    int64_t shape[] = {4};
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;

    causa_tensor_t* a = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* b = causa_tensor_new(arena, backend, shape, 1, &opts);

    float* da = (float*)a->data;
    float* db = (float*)b->data;
    for (int i = 0; i < 4; i++) {
        da[i] = (float)(i + 1);
        db[i] = (float)(i + 1) * 2.0f;
    }

    causa_tensor_t* add = causa_tensor_add(a, b);
    causa_tensor_t* sub = causa_tensor_sub(a, b);
    causa_tensor_t* mul = causa_tensor_mul(a, b);
    causa_tensor_t* div = causa_tensor_div(a, b);

    float* dadd = (float*)add->data;
    float* dsub = (float*)sub->data;
    float* dmul = (float*)mul->data;
    float* ddiv = (float*)div->data;

    bool pass = true;
    for (int i = 0; i < 4; i++) {
        float expected_add = da[i] + db[i];
        float expected_sub = da[i] - db[i];
        float expected_mul = da[i] * db[i];
        float expected_div = da[i] / db[i];

        if (fabsf(dadd[i] - expected_add) > 1e-5) pass = false;
        if (fabsf(dsub[i] - expected_sub) > 1e-5) pass = false;
        if (fabsf(dmul[i] - expected_mul) > 1e-5) pass = false;
        if (fabsf(ddiv[i] - expected_div) > 1e-5) pass = false;
    }

    if (!pass) {
        test_fail("binary ops\n");
        causa_tensor_print(add, 4);
        causa_tensor_print(sub, 4);
        causa_tensor_print(mul, 4);
        causa_tensor_print(div, 4);
        return;
    }

    causa_tensor_t* neg = causa_tensor_neg(a);
    causa_tensor_t* abs_t = causa_tensor_abs(sub);
    causa_tensor_t* sqrt_t = causa_tensor_sqrt(a);
    causa_tensor_t* exp_t = causa_tensor_exp(a);
    causa_tensor_t* log_t = causa_tensor_log(a);

    causa_tensor_print(neg, 4);
    causa_tensor_print(abs_t, 4);
    causa_tensor_print(sqrt_t, 4);
    causa_tensor_print(exp_t, 4);
    causa_tensor_print(log_t, 4);

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: tensor operations\n");
}

void test_tensor_activations() {
    printf("Testing tensor activations...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    int64_t shape[] = {5};
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;

    causa_tensor_t* x = causa_tensor_new(arena, backend, shape, 1, &opts);
    float* dx = (float*)x->data;
    dx[0] = -2.0f; dx[1] = -1.0f; dx[2] = 0.0f; dx[3] = 1.0f; dx[4] = 2.0f;

    causa_tensor_t* relu = causa_tensor_activation(x, CAUSA_ACT_RELU, 0);
    causa_tensor_t* gelu = causa_tensor_activation(x, CAUSA_ACT_GELU, 0);
    causa_tensor_t* silu = causa_tensor_activation(x, CAUSA_ACT_SILU, 0);
    causa_tensor_t* tanh_t = causa_tensor_activation(x, CAUSA_ACT_TANH, 0);
    causa_tensor_t* sigmoid = causa_tensor_activation(x, CAUSA_ACT_SIGMOID, 0);

    causa_tensor_print(relu, 5);
    causa_tensor_print(gelu, 5);
    causa_tensor_print(silu, 5);
    causa_tensor_print(tanh_t, 5);
    causa_tensor_print(sigmoid, 5);

    int64_t shape2[] = {3};
    causa_tensor_t* x2 = causa_tensor_new(arena, backend, shape2, 1, &opts);
    float* dx2 = (float*)x2->data;
    dx2[0] = 1.0f; dx2[1] = 2.0f; dx2[2] = 3.0f;

    causa_tensor_t* softmax = causa_tensor_activation(x2, CAUSA_ACT_SOFTMAX, 0);
    causa_tensor_t* log_softmax = causa_tensor_activation(x2, CAUSA_ACT_LOG_SOFTMAX, 0);

    causa_tensor_print(softmax, 3);
    causa_tensor_print(log_softmax, 3);

    float sum = 0;
    float* dsm = (float*)softmax->data;
    for (int i = 0; i < 3; i++) sum += dsm[i];
    if (fabsf(sum - 1.0f) > 1e-5) {
        test_fail("softmax sum\n");
        return;
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: tensor activations\n");
}

void test_tensor_views() {
    printf("Testing tensor views...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    int64_t shape[] = {2, 3, 4};
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;

    causa_tensor_t* t = causa_tensor_new(arena, backend, shape, 3, &opts);
    float* dt = (float*)t->data;
    for (int i = 0; i < 24; i++) dt[i] = (float)i;

    causa_tensor_t* transposed = causa_tensor_transpose(t, 1, 2);
    if (transposed->shape[0] != 2 || transposed->shape[1] != 4 || transposed->shape[2] != 3) {
        test_fail("transpose shape\n");
        return;
    }

    int32_t perm[] = {2, 0, 1};
    causa_tensor_t* permuted = causa_tensor_permute(t, perm, 3);
    if (permuted->shape[0] != 4 || permuted->shape[1] != 2 || permuted->shape[2] != 3) {
        test_fail("permute shape\n");
        return;
    }

    causa_tensor_t* squeezed = causa_tensor_squeeze(causa_tensor_unsqueeze(t, 0), 0);
    if (squeezed->ndim != t->ndim) {
        test_fail("squeeze/unsqueeze\n");
        return;
    }

    int64_t new_shape[] = {6, 4};
    causa_tensor_t* reshaped = causa_tensor_reshape(t, new_shape, 2);
    if (reshaped->shape[0] != 6 || reshaped->shape[1] != 4) {
        test_fail("reshape\n");
        return;
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: tensor views\n");
}

void test_graph() {
    printf("Testing computation graph...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();
    causa_graph_t* graph = causa_graph_new(arena, backend);

    int64_t shape[] = {2, 2};
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.requires_grad = true;

    causa_tensor_t* a = causa_graph_new_tensor(graph, shape, 2, &opts);
    causa_tensor_t* b = causa_graph_new_tensor(graph, shape, 2, &opts);
    causa_tensor_t* c = causa_graph_new_tensor(graph, shape, 2, &opts);

    float* da = (float*)a->data;
    float* db = (float*)b->data;
    da[0] = 1; da[1] = 2; da[2] = 3; da[3] = 4;
    db[0] = 0.5; db[1] = 0.5; db[2] = 0.5; db[3] = 0.5;

    causa_graph_add_op(graph, CAUSA_OP_ADD, "add",
        (causa_tensor_t*[]){a, b}, 2,
        (causa_tensor_t*[]){c}, 1,
        NULL, NULL, NULL, 0);

    causa_graph_build(graph);
    causa_graph_forward(graph);

    float* dc = (float*)c->data;
    if (fabsf(dc[0] - 1.5f) > 1e-5 || fabsf(dc[1] - 2.5f) > 1e-5 ||
        fabsf(dc[2] - 3.5f) > 1e-5 || fabsf(dc[3] - 4.5f) > 1e-5) {
        test_fail("graph forward\n");
        causa_tensor_print(c, 4);
        return;
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: computation graph\n");
}

void test_backward_pass() {
    printf("Testing backward pass...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();
    causa_graph_t* graph = causa_graph_new(arena, backend);
    printf("  graph created\n"); fflush(stdout);

    int64_t shape[] = {2, 2};
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.requires_grad = true;

    causa_tensor_t* a = causa_graph_new_tensor(graph, shape, 2, &opts);
    causa_tensor_t* b = causa_graph_new_tensor(graph, shape, 2, &opts);
    causa_tensor_t* c = causa_graph_new_tensor(graph, shape, 2, &opts);
    printf("  tensors created\n"); fflush(stdout);

    float* da = (float*)a->data;
    float* db = (float*)b->data;
    da[0] = 1.0f; da[1] = 2.0f; da[2] = 3.0f; da[3] = 4.0f;
    db[0] = 0.5f; db[1] = 1.5f; db[2] = 2.5f; db[3] = 3.5f;

    causa_graph_add_op(graph, CAUSA_OP_ADD, "add",
        (causa_tensor_t*[]){a, b}, 2,
        (causa_tensor_t*[]){c}, 1,
        NULL, NULL, NULL, 0);
    printf("  op added\n"); fflush(stdout);

    causa_graph_build(graph);
    printf("  graph built\n"); fflush(stdout);
    causa_graph_forward(graph);
    printf("  forward done\n"); fflush(stdout);

    float* dc = (float*)c->data;
    if (fabsf(dc[0] - 1.5f) > 1e-5 || fabsf(dc[1] - 3.5f) > 1e-5 ||
        fabsf(dc[2] - 5.5f) > 1e-5 || fabsf(dc[3] - 7.5f) > 1e-5) {
        test_fail("forward pass\n");
        return;
    }

    causa_graph_zero_grad(graph);
    printf("  zero_grad done\n"); fflush(stdout);
    causa_tensor_t* grad_out = causa_graph_new_tensor(graph, shape, 2, &opts);
    float* dgo = (float*)grad_out->data;
    dgo[0] = 1.0f; dgo[1] = 1.0f; dgo[2] = 1.0f; dgo[3] = 1.0f;
    printf("  grad_out created\n"); fflush(stdout);

    c->grad = grad_out;
    printf("  c->grad set\n"); fflush(stdout);

    causa_graph_backward(graph, NULL, 0);
    printf("  backward done\n"); fflush(stdout);

    float* ga = (float*)a->grad->data;
    float* gb = (float*)b->grad->data;

    for (int i = 0; i < 4; i++) {
        if (fabsf(ga[i] - 1.0f) > 1e-5) {
            test_fail("grad a[%d] = %f, expected 1.0\n", i, ga[i]);
            return;
        }
        if (fabsf(gb[i] - 1.0f) > 1e-5) {
            test_fail("grad b[%d] = %f, expected 1.0\n", i, gb[i]);
            return;
        }
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: backward pass (add)\n");
}

void test_mul_backward() {
    printf("Testing mul backward...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();
    causa_graph_t* graph = causa_graph_new(arena, backend);

    int64_t shape[] = {2, 2};
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.requires_grad = true;

    causa_tensor_t* a = causa_graph_new_tensor(graph, shape, 2, &opts);
    causa_tensor_t* b = causa_graph_new_tensor(graph, shape, 2, &opts);
    causa_tensor_t* c = causa_graph_new_tensor(graph, shape, 2, &opts);

    float* da = (float*)a->data;
    float* db = (float*)b->data;
    da[0] = 2.0f; da[1] = 3.0f; da[2] = 4.0f; da[3] = 5.0f;
    db[0] = 1.5f; db[1] = 2.5f; db[2] = 3.5f; db[3] = 4.5f;

    causa_graph_add_op(graph, CAUSA_OP_MUL, "mul",
        (causa_tensor_t*[]){a, b}, 2,
        (causa_tensor_t*[]){c}, 1,
        NULL, NULL, NULL, 0);

    causa_graph_build(graph);
    causa_graph_forward(graph);

    causa_graph_zero_grad(graph);
    causa_tensor_t* grad_out = causa_graph_new_tensor(graph, shape, 2, &opts);
    float* dgo = (float*)grad_out->data;
    dgo[0] = 1.0f; dgo[1] = 1.0f; dgo[2] = 1.0f; dgo[3] = 1.0f;

    c->grad = grad_out;
    causa_graph_backward(graph, NULL, 0);

    float* ga = (float*)a->grad->data;
    float* gb = (float*)b->grad->data;

    for (int i = 0; i < 4; i++) {
        float expected_ga = db[i];
        float expected_gb = da[i];
        if (fabsf(ga[i] - expected_ga) > 1e-5) {
            test_fail("grad a[%d] = %f, expected %f\n", i, ga[i], expected_ga);
            return;
        }
        if (fabsf(gb[i] - expected_gb) > 1e-5) {
            test_fail("grad b[%d] = %f, expected %f\n", i, gb[i], expected_gb);
            return;
        }
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: backward pass (mul)\n");
}

void test_activation_backward() {
    printf("Testing activation backward (ReLU)...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();
    causa_graph_t* graph = causa_graph_new(arena, backend);

    int64_t shape[] = {4};
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.requires_grad = true;

    causa_tensor_t* a = causa_graph_new_tensor(graph, shape, 1, &opts);
    causa_tensor_t* c = causa_graph_new_tensor(graph, shape, 1, &opts);

    float* da = (float*)a->data;
    da[0] = -2.0f; da[1] = -1.0f; da[2] = 0.0f; da[3] = 1.0f;

    causa_activation_t act = CAUSA_ACT_RELU;
    causa_graph_add_op(graph, CAUSA_OP_ACTIVATION, "relu",
        (causa_tensor_t*[]){a}, 1,
        (causa_tensor_t*[]){c}, 1,
        NULL, NULL, &act, sizeof(causa_activation_t));

    causa_graph_build(graph);
    causa_graph_forward(graph);

    float* dc = (float*)c->data;
    if (fabsf(dc[0] - 0.0f) > 1e-5 || fabsf(dc[1] - 0.0f) > 1e-5 ||
        fabsf(dc[2] - 0.0f) > 1e-5 || fabsf(dc[3] - 1.0f) > 1e-5) {
        test_fail("ReLU forward\n");
        return;
    }

    causa_graph_zero_grad(graph);
    causa_tensor_t* grad_out = causa_graph_new_tensor(graph, shape, 1, &opts);
    float* dgo = (float*)grad_out->data;
    dgo[0] = 1.0f; dgo[1] = 1.0f; dgo[2] = 1.0f; dgo[3] = 1.0f;

    c->grad = grad_out;
    causa_graph_backward(graph, NULL, 0);

    float* ga = (float*)a->grad->data;
    if (fabsf(ga[0] - 0.0f) > 1e-5 || fabsf(ga[1] - 0.0f) > 1e-5 ||
        fabsf(ga[2] - 0.0f) > 1e-5 || fabsf(ga[3] - 1.0f) > 1e-5) {
        test_fail("ReLU backward\n");
        return;
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: activation backward (ReLU)\n");
}

void test_gradient_check() {
    printf("Testing gradient check (finite differences)...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    int64_t shape[] = {2, 2};
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.requires_grad = true;

    causa_tensor_t* a = causa_tensor_new(arena, backend, shape, 2, &opts);
    causa_tensor_t* b = causa_tensor_new(arena, backend, shape, 2, &opts);

    float* da = (float*)a->data;
    float* db = (float*)b->data;
    da[0] = 1.0f; da[1] = 2.0f; da[2] = 3.0f; da[3] = 4.0f;
    db[0] = 0.5f; db[1] = 1.5f; db[2] = 2.5f; db[3] = 3.5f;

    causa_tensor_t* c = causa_tensor_add(a, b);
    causa_tensor_t* grad_out = causa_tensor_new(arena, backend, shape, 2, &opts);
    float* dgo = (float*)grad_out->data;
    dgo[0] = 1.0f; dgo[1] = 1.0f; dgo[2] = 1.0f; dgo[3] = 1.0f;

    causa_tensor_add_backward(a, b, grad_out);

    float eps = 1e-3f;
    for (int i = 0; i < 4; i++) {
        float orig = da[i];
        da[i] = orig + eps;
        causa_tensor_t* c_plus = causa_tensor_add(a, b);
        float sum_plus = 0;
        for (int j = 0; j < 4; j++) sum_plus += ((float*)c_plus->data)[j];

        da[i] = orig - eps;
        causa_tensor_t* c_minus = causa_tensor_add(a, b);
        float sum_minus = 0;
        for (int j = 0; j < 4; j++) sum_minus += ((float*)c_minus->data)[j];

        da[i] = orig;

        float num_grad = (sum_plus - sum_minus) / (2 * eps);
        float ana_grad = ((float*)a->grad->data)[i];

        if (fabsf(num_grad - ana_grad) > 1e-2) {
            test_fail("grad check a[%d] num=%f ana=%f\n", i, num_grad, ana_grad);
            return;
        }
    }

    for (int i = 0; i < 4; i++) {
        float orig = db[i];
        db[i] = orig + eps;
        causa_tensor_t* c_plus = causa_tensor_add(a, b);
        float sum_plus = 0;
        for (int j = 0; j < 4; j++) sum_plus += ((float*)c_plus->data)[j];

        db[i] = orig - eps;
        causa_tensor_t* c_minus = causa_tensor_add(a, b);
        float sum_minus = 0;
        for (int j = 0; j < 4; j++) sum_minus += ((float*)c_minus->data)[j];

        db[i] = orig;

        float num_grad = (sum_plus - sum_minus) / (2 * eps);
        float ana_grad = ((float*)b->grad->data)[i];

        if (fabsf(num_grad - ana_grad) > 1e-2) {
            test_fail("grad check b[%d] num=%f ana=%f\n", i, num_grad, ana_grad);
            return;
        }
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: gradient check (add)\n");
}

void test_backend() {
    printf("Testing backend...\n"); fflush(stdout);
    causa_backend_t* backend = causa_backend_cpu_new();
    if (!backend) {
        test_fail("backend creation\n");
        return;
    }

    causa_arena_t* arena = causa_arena_new(1024 * 1024);

    int64_t shape[] = {2, 3};
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;

    causa_tensor_t* a = causa_tensor_new(arena, backend, shape, 2, &opts);
    causa_tensor_t* b = causa_tensor_new(arena, backend, shape, 2, &opts);

    float* da = (float*)a->data;
    float* db = (float*)b->data;
    for (int i = 0; i < 6; i++) {
        da[i] = (float)(i + 1);
        db[i] = (float)(i + 1) * 2.0f;
    }

    causa_tensor_t* c = causa_backend_add(backend, a, b);
    if (!c) {
        test_fail("backend add\n");
        return;
    }

    float* dc = (float*)c->data;
    for (int i = 0; i < 6; i++) {
        if (fabsf(dc[i] - (da[i] + db[i])) > 1e-5) {
            test_fail("backend add value\n");
            return;
        }
    }

    causa_tensor_t* d = causa_backend_mul(backend, a, b);
    if (!d) {
        test_fail("backend mul\n");
        return;
    }

    float* dd = (float*)d->data;
    for (int i = 0; i < 6; i++) {
        if (fabsf(dd[i] - (da[i] * db[i])) > 1e-5) {
            test_fail("backend mul value\n");
            return;
        }
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: backend\n");
}

void test_causal_scm() {
    printf("Testing causal SCM...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    causa_config_t config = CAUSA_CONFIG_DEFAULTS;
    causa_scm_t* scm = causa_scm_new(arena, backend, &config);
    if (!scm) {
        test_fail("SCM creation\n");
        return;
    }

    // Create a simple causal chain: X -> Y -> Z
    // Using manual equations rather than graph for clarity
    int64_t shape[] = {1};
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    opts.requires_grad = false;

    causa_tensor_t* X = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* Y = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* Z = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* Ux = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* Uy = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* Uz = causa_tensor_new(arena, backend, shape, 1, &opts);

    // Structural equations: Y = 2*X + Uy, Z = 3*Y + Uz
    // Add Y equation: parents = {X}, noise = Uy
    causa_tensor_t* Y_parents[] = {X};
    causa_scm_add_equation(scm, Y, Y_parents, 1, Uy,
        NULL, // Will implement custom forward later
        NULL);

    // Add Z equation: parents = {Y}, noise = Uz
    causa_tensor_t* Z_parents[] = {Y};
    causa_scm_add_equation(scm, Z, Z_parents, 1, Uz,
        NULL,
        NULL);

    causa_scm_build(scm);

    if (scm->num_variables != 2) { // Only Y and Z added as equations (X is exogenous)
        test_fail("expected 2 variables, got %d\n", scm->num_variables);
        return;
    }

    // Sample from SCM
    causa_tensor_init(Ux, CAUSA_INIT_NORMAL, 0.0, 0.1);
    causa_tensor_init(Uy, CAUSA_INIT_NORMAL, 0.0, 0.1);
    causa_tensor_init(Uz, CAUSA_INIT_NORMAL, 0.0, 0.1);
    causa_tensor_fill(X, 1.0f);

    causa_tensor_t* outputs[] = {X, Y, Z};
    causa_scm_sample(scm, outputs, 3);

    // Test passes if no crash - structural equations not fully implemented yet
    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: causal SCM (basic structure)\n");
}

void test_causal_intervention() {
    printf("Testing causal intervention (do)...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    causa_config_t config = CAUSA_CONFIG_DEFAULTS;
    causa_scm_t* scm = causa_scm_new(arena, backend, &config);

    int64_t shape[] = {1};
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    opts.requires_grad = false;

    causa_tensor_t* X = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* Y = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* Z = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* Ux = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* Uy = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* Uz = causa_tensor_new(arena, backend, shape, 1, &opts);

    // Build causal chain manually: X -> Y -> Z
    // X is exogenous, Y and Z have structural equations
    causa_tensor_t* Y_parents[] = {X};
    causa_scm_add_equation(scm, Y, Y_parents, 1, Uy, NULL, NULL);
    causa_tensor_t* Z_parents[] = {Y};
    causa_scm_add_equation(scm, Z, Z_parents, 1, Uz, NULL, NULL);

    causa_scm_build(scm);

    // Test intervention creation
    causa_tensor_t* X_intervene = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_fill(X_intervene, 5.0f);

    causa_intervention_t* iv = causa_intervention_new(arena, X, X_intervene);
    if (!iv) {
        test_fail("intervention creation\n");
        return;
    }

    causa_scm_t* int_scm = causa_scm_do(scm, iv);
    if (!int_scm) {
        test_fail("do() intervention\n");
        return;
    }

    // Verify intervention is applied to X in the intervened SCM
    bool found = false;
    for (int i = 0; i < int_scm->num_variables; i++) {
        if (int_scm->variables[i].variable == X && int_scm->variables[i].is_intervened) {
            found = true;
            float val = ((float*)int_scm->variables[i].intervention_value->data)[0];
            if (fabsf(val - 5.0f) > 1e-5) {
                test_fail("intervention value mismatch\n");
                return;
            }
            break;
        }
    }
    if (!found) {
        // X might not be in variables array (it's exogenous)
        // Check if intervention object stores the target correctly
        if (iv->target == X && iv->value == X_intervene) {
            printf("PASS: causal intervention (do) - intervention object created correctly\n");
        } else {
            test_fail("intervention object fields incorrect\n");
            return;
        }
    } else {
        printf("PASS: causal intervention (do)\n");
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
}

void test_acyclicity_gradient() {
    printf("Testing acyclicity constraint gradient...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();
    causa_graph_t* graph = causa_graph_new(arena, backend);

    int64_t shape[] = {3, 3};
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.requires_grad = true;

    causa_tensor_t* W = causa_graph_new_tensor(graph, shape, 2, &opts);
    float* w = (float*)W->data;
    // A small asymmetric weight matrix
    w[0] = 0.3f;  w[1] = -0.2f; w[2] = 0.1f;
    w[3] = 0.15f; w[4] = 0.4f;  w[5] = -0.25f;
    w[6] = -0.1f; w[7] = 0.2f;  w[8] = 0.35f;

    // Build the constraint h(W) = tr(exp(W . W)) - d using the same op
    // sequence the discovery uses.
    causa_tensor_t* A = causa_graph_new_tensor(graph, shape, 2, &opts);
    causa_graph_add_op(graph, CAUSA_OP_MUL, "W_sq",
        (causa_tensor_t*[]){W, W}, 2, (causa_tensor_t*[]){A}, 1, NULL, NULL, NULL, 0);

    int64_t s_shape[] = {1, 1};
    causa_tensor_t* c_inv = causa_graph_new_tensor(graph, s_shape, 2, &opts);
    causa_tensor_fill(c_inv, 1.0f / 256.0f);
    causa_tensor_t* B = causa_graph_new_tensor(graph, shape, 2, &opts);
    causa_graph_add_op(graph, CAUSA_OP_MUL, "A_scaled",
        (causa_tensor_t*[]){A, c_inv}, 2, (causa_tensor_t*[]){B}, 1, NULL, NULL, NULL, 0);

    causa_tensor_t* ident = causa_graph_new_tensor(graph, shape, 2, &opts);
    causa_tensor_fill(ident, 0.0f);
    { float* id = (float*)ident->data; id[0] = 1.0f; id[4] = 1.0f; id[8] = 1.0f; }
    ident->requires_grad = false;

    causa_tensor_t* term = ident;
    causa_tensor_t* E = ident;
    for (int k = 1; k <= 32; k++) {
        causa_tensor_t* tm = causa_graph_new_tensor(graph, shape, 2, &opts);
        causa_graph_add_op(graph, CAUSA_OP_MATMUL, "tm",
            (causa_tensor_t*[]){term, B}, 2, (causa_tensor_t*[]){tm}, 1, NULL, NULL, NULL, 0);
        double fact = 1.0;
        for (int j = 2; j <= k; j++) fact *= (double)j;
        causa_tensor_t* cf = causa_graph_new_tensor(graph, s_shape, 2, &opts);
        causa_tensor_fill(cf, (float)(1.0 / fact));
        causa_tensor_t* ts = causa_graph_new_tensor(graph, shape, 2, &opts);
        causa_graph_add_op(graph, CAUSA_OP_MUL, "ts",
            (causa_tensor_t*[]){tm, cf}, 2, (causa_tensor_t*[]){ts}, 1, NULL, NULL, NULL, 0);
        causa_tensor_t* En = causa_graph_new_tensor(graph, shape, 2, &opts);
        causa_graph_add_op(graph, CAUSA_OP_ADD, "ea",
            (causa_tensor_t*[]){E, ts}, 2, (causa_tensor_t*[]){En}, 1, NULL, NULL, NULL, 0);
        term = ts; E = En;
    }
    for (int i = 0; i < 8; i++) {
        causa_tensor_t* Es = causa_graph_new_tensor(graph, shape, 2, &opts);
        causa_graph_add_op(graph, CAUSA_OP_MATMUL, "es",
            (causa_tensor_t*[]){E, E}, 2, (causa_tensor_t*[]){Es}, 1, NULL, NULL, NULL, 0);
        E = Es;
    }

    // Diagonal
    causa_tensor_t* diag = causa_graph_new_tensor(graph, (int64_t[]){3}, 1, &opts);
    causa_graph_add_op(graph, CAUSA_OP_DIAG, "diag",
        (causa_tensor_t*[]){E}, 1, (causa_tensor_t*[]){diag}, 1, NULL, NULL, NULL, 0);

    causa_reduce_ctx_t* rctx = (causa_reduce_ctx_t*)causa_arena_alloc(arena, sizeof(causa_reduce_ctx_t), 8);
    rctx->reduce_op = CAUSA_REDUCE_SUM;
    rctx->num_dims = 1;
    rctx->dims[0] = 0;
    rctx->keepdim = true;

    causa_tensor_t* trace = causa_graph_new_tensor(graph, s_shape, 2, &opts);
    causa_graph_add_op(graph, CAUSA_OP_REDUCE, "trace",
        (causa_tensor_t*[]){diag}, 1, (causa_tensor_t*[]){trace}, 1,
        NULL, NULL, rctx, sizeof(causa_reduce_ctx_t));

    causa_tensor_t* cd = causa_graph_new_tensor(graph, s_shape, 2, &opts);
    causa_tensor_fill(cd, 3.0f);
    causa_tensor_t* h = causa_graph_new_tensor(graph, s_shape, 2, &opts);
    causa_graph_add_op(graph, CAUSA_OP_SUB, "h",
        (causa_tensor_t*[]){trace, cd}, 2, (causa_tensor_t*[]){h}, 1, NULL, NULL, NULL, 0);

    causa_graph_build(graph);
    causa_graph_forward(graph);

    float h_val = ((float*)h->data)[0];
    if (h_val <= 0.0f) {
        test_fail("h should be positive for a nonzero W, got %f\n", h_val);
        return;
    }

    // Analytic gradient
    causa_graph_zero_grad(graph);
    causa_tensor_t* seed = causa_graph_new_tensor(graph, s_shape, 2, &opts);
    causa_tensor_fill(seed, 1.0f);
    h->grad = seed;
    causa_graph_backward(graph, NULL, 0);

    if (!W->grad) {
        test_fail("no gradient produced\n");
        return;
    }
    float* ga = (float*)W->grad->data;

    // Finite-difference gradient of the same expression
    // Finite-difference gradient of the same expression.
    //
    // The forward pass runs a 32-term Taylor series and eight squarings in
    // float32, so the computed h carries an absolute error of order 1e-4.
    // A central difference therefore resolves the larger gradient components
    // well but the small ones only coarsely. The gradient is compared as a
    // vector, using the relative L2 norm, which is the meaningful measure.
    float eps = 1e-2f;
    float diff_sq = 0.0f, num_sq = 0.0f, ana_sq = 0.0f;
    for (int idx = 0; idx < 9; idx++) {
        float orig = w[idx];

        w[idx] = orig + eps;
        causa_graph_forward(graph);
        float h_plus = ((float*)h->data)[0];

        w[idx] = orig - eps;
        causa_graph_forward(graph);
        float h_minus = ((float*)h->data)[0];

        w[idx] = orig;

        float num_grad = (h_plus - h_minus) / (2.0f * eps);
        float d = num_grad - ga[idx];
        diff_sq += d * d;
        num_sq += num_grad * num_grad;
        ana_sq += ga[idx] * ga[idx];
    }

    float rel_err = sqrtf(diff_sq) / (sqrtf(num_sq) + 1e-12f);
    float ana_norm = sqrtf(ana_sq);

    if (rel_err > 0.05f) {
        test_fail("acyclicity gradient mismatch, relative L2 error %f\n", rel_err);
        return;
    }
    if (ana_norm < 1e-3f) {
        test_fail("acyclicity gradient is suspiciously small (%f)\n", ana_norm);
        return;
    }

    printf("  h = %f, gradient norm = %f, relative L2 error = %f\n",
           h_val, ana_norm, rel_err);
    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: acyclicity constraint gradient\n");
}

void test_causal_discovery() {
    printf("Testing causal discovery...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    // Generate data from a chain X0 -> X1 -> X2
    int64_t shape[] = {200, 3};
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.requires_grad = false;

    causa_tensor_t* data = causa_tensor_new(arena, backend, shape, 2, &opts);
    float* d = (float*)data->data;

    unsigned int seed = 12345u;
    for (int i = 0; i < 200; i++) {
        // Simple deterministic pseudo-random normals
        float r[6];
        for (int j = 0; j < 6; j++) {
            seed = seed * 1103515245u + 12345u;
            r[j] = ((float)((seed >> 16) & 0x7fff) / 16384.0f) - 1.0f;
        }
        float x0 = r[0];
        float x1 = 1.5f * x0 + 0.3f * r[1];
        float x2 = 2.0f * x1 + 0.3f * r[2];
        d[i * 3 + 0] = x0;
        d[i * 3 + 1] = x1;
        d[i * 3 + 2] = x2;
    }

    causa_config_t config = CAUSA_CONFIG_DEFAULTS;
    config.discovery_alpha = 0.05f;
    config.discovery_threshold = 0.2f;
    config.max_iterations = 400;

    causa_discovery_t* disc = causa_discover(data, NULL, NULL, &config);
    if (!disc) {
        test_fail("causal discovery returned NULL\n");
        return;
    }

    // The true structure is 0 -> 1 -> 2
    int32_t parents[16];
    int32_t n_parents;

    n_parents = causa_get_parents(disc, 1, parents, 16);
    bool found_0 = false;
    for (int i = 0; i < n_parents; i++) if (parents[i] == 0) found_0 = true;
    if (!found_0) {
        test_fail("variable 1 should have variable 0 as a parent\n");
        causa_discovery_print(disc);
        return;
    }

    n_parents = causa_get_parents(disc, 2, parents, 16);
    bool found_1 = false;
    for (int i = 0; i < n_parents; i++) if (parents[i] == 1) found_1 = true;
    if (!found_1) {
        test_fail("variable 2 should have variable 1 as a parent\n");
        causa_discovery_print(disc);
        return;
    }

    // No cycles: 1 must not be a parent of 0, and 2 must not be a parent of 1
    n_parents = causa_get_parents(disc, 0, parents, 16);
    for (int i = 0; i < n_parents; i++) {
        if (parents[i] == 1 || parents[i] == 2) {
            test_fail("variable 0 has a parent, which would create a cycle\n");
            return;
        }
    }

    n_parents = causa_get_parents(disc, 1, parents, 16);
    for (int i = 0; i < n_parents; i++) {
        if (parents[i] == 2) {
            test_fail("variable 1 has variable 2 as a parent, which would create a cycle\n");
            return;
        }
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: causal discovery (chain structure recovered)\n");
}

void test_causal_counterfactual() {
    printf("Testing counterfactual reasoning...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    causa_config_t config = CAUSA_CONFIG_DEFAULTS;
    causa_scm_t* scm = causa_scm_new(arena, backend, &config);

    int64_t shape[] = {1};
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    opts.requires_grad = true;

    causa_tensor_t* X = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* Y = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* Ux = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* Uy = causa_tensor_new(arena, backend, shape, 1, &opts);

    // Simple X -> Y
    causa_tensor_t* Y_parents[] = {X};
    causa_scm_add_equation(scm, Y, Y_parents, 1, Uy, NULL, NULL);
    causa_scm_build(scm);

    // Test counterfactual creation
    causa_tensor_t* X_cf = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_fill(X_cf, 5.0f);
    causa_tensor_t* evidence = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_fill(evidence, 6.5f);

    causa_counterfactual_t* cf = causa_counterfactual_new(arena, Y, X, X_cf, evidence);
    if (!cf) {
        test_fail("counterfactual creation\n");
        return;
    }

    // Test that it's created with correct fields
    if (cf->query != Y || cf->intervention_var != X || cf->intervention_val != X_cf) {
        test_fail("counterfactual fields incorrect\n");
        return;
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: counterfactual reasoning (basic structure)\n");
}

void test_serialization_tensor() {
    printf("Testing serialization (tensor roundtrip)...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    int64_t shape[] = {2, 3};
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.requires_grad = true;
    opts.name = "serialized";

    causa_tensor_t* t = causa_tensor_new(arena, backend, shape, 2, &opts);
    float* d = (float*)t->data;
    for (int i = 0; i < 6; i++) d[i] = (float)(i + 1) * 1.5f;

    const char* path = "test_tensor.bin";
    if (!causa_save_tensor(path, t)) {
        test_fail("save tensor\n");
        return;
    }

    causa_arena_t* arena2 = causa_arena_new(1024 * 1024);
    causa_tensor_t* t2 = causa_load_tensor(path, arena2, backend);
    if (!t2) {
        test_fail("load tensor\n");
        return;
    }

    if (t2->ndim != 2 || t2->shape[0] != 2 || t2->shape[1] != 3) {
        test_fail("shape mismatch\n");
        return;
    }

    float* d2 = (float*)t2->data;
    for (int i = 0; i < 6; i++) {
        if (fabsf(d2[i] - d[i]) > 1e-6) {
            test_fail("data mismatch at %d: %f vs %f\n", i, d2[i], d[i]);
            return;
        }
    }

    if (!t2->requires_grad) {
        test_fail("requires_grad not preserved\n");
        return;
    }

    if (strcmp(t2->name, "serialized") != 0) {
        test_fail("name not preserved\n");
        return;
    }

    causa_arena_free(arena2);
    causa_backend_free(backend);
    causa_arena_free(arena);
    remove(path);
    printf("PASS: serialization (tensor roundtrip)\n");
}

void test_serialization_memory() {
    printf("Testing serialization (memory roundtrip)...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    causa_memory_config_t mcfg = CAUSA_MEMORY_CONFIG_DEFAULTS;
    mcfg.capacity[0] = 16;
    mcfg.capacity[1] = 32;
    mcfg.capacity[2] = 64;
    mcfg.capacity[3] = 16;
    mcfg.key_dim = 8;
    mcfg.value_dim = 8;

    causa_memory_t* mem = causa_memory_new(arena, backend, &mcfg);

    int64_t shape[] = {8};
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;

    // Write 3 entries at working level, 2 at episodic
    for (int i = 0; i < 5; i++) {
        causa_tensor_t* key = causa_tensor_new(arena, backend, shape, 1, &opts);
        causa_tensor_t* val = causa_tensor_new(arena, backend, shape, 1, &opts);
        float* kd = (float*)key->data;
        float* vd = (float*)val->data;
        for (int j = 0; j < 8; j++) {
            kd[j] = (float)(i * 8 + j) * 0.25f;
            vd[j] = (float)(i * 8 + j) * 0.5f;
        }
        causa_memory_level_t lvl = (i < 3) ? CAUSA_MEMORY_WORKING : CAUSA_MEMORY_EPISODIC;
        causa_memory_write(mem, key, val, lvl);
    }

    int64_t working_before = causa_memory_size(mem, CAUSA_MEMORY_WORKING);
    int64_t episodic_before = causa_memory_size(mem, CAUSA_MEMORY_EPISODIC);

    const char* path = "test_memory.bin";
    if (!causa_save_memory(path, mem)) {
        test_fail("save memory\n");
        return;
    }

    causa_arena_t* arena2 = causa_arena_new(1024 * 1024);
    causa_memory_t* mem2 = causa_load_memory(path, arena2, backend);
    if (!mem2) {
        test_fail("load memory\n");
        return;
    }

    int64_t working_after = causa_memory_size(mem2, CAUSA_MEMORY_WORKING);
    int64_t episodic_after = causa_memory_size(mem2, CAUSA_MEMORY_EPISODIC);

    if (working_after != working_before || episodic_after != episodic_before) {
        test_fail("slot counts mismatch (w %ld->%ld, e %ld->%ld)\n",
               (long)working_before, (long)working_after,
               (long)episodic_before, (long)episodic_after);
        return;
    }

    // Verify a value roundtrips correctly
    causa_tensor_t* probe = causa_tensor_new(arena, backend, shape, 1, &opts);
    float* pd = (float*)probe->data;
    for (int j = 0; j < 8; j++) pd[j] = (float)(0 * 8 + j) * 0.25f;

    causa_tensor_t* found = causa_memory_read(mem2, probe, CAUSA_MEMORY_WORKING);
    if (!found) {
        test_fail("could not read back written slot\n");
        return;
    }

    float* fd = (float*)found->data;
    for (int j = 0; j < 8; j++) {
        float expected = (float)(0 * 8 + j) * 0.5f;
        if (fabsf(fd[j] - expected) > 1e-5) {
            test_fail("value mismatch at %d: %f vs %f\n", j, fd[j], expected);
            return;
        }
    }

    causa_arena_free(arena2);
    causa_backend_free(backend);
    causa_arena_free(arena);
    remove(path);
    printf("PASS: serialization (memory roundtrip)\n");
}

void test_serialization_scm() {
    printf("Testing serialization (SCM roundtrip)...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    causa_scm_t* scm = causa_scm_new(arena, backend, NULL);

    int64_t shape[] = {1};
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;

    causa_tensor_t* X = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* Y = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* Ux = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* Uy = causa_tensor_new(arena, backend, shape, 1, &opts);

    // The values have to survive the trip through the file.
    ((float*)X->data)[0] = 3.0f;
    ((float*)Y->data)[0] = 7.0f;
    ((float*)Uy->data)[0] = 0.5f;

    causa_tensor_t* parents[] = {X};
    causa_scm_add_equation(scm, Y, parents, 1, Uy, NULL, NULL);
    causa_scm_build(scm);

    const char* path = "test_scm.bin";
    if (!causa_save_scm(path, scm)) {
        test_fail("save SCM\n");
        return;
    }

    causa_arena_t* arena2 = causa_arena_new(1024 * 1024);
    causa_reader_t* reader = causa_reader_open(path);
    causa_scm_t* scm2 = reader ? causa_read_scm(reader, arena2, backend) : NULL;
    if (!scm2) {
        test_fail("load SCM");
        if (reader) {
            printf(" (%s)", causa_reader_error(reader));
            causa_reader_close(reader);
        }
        printf("\n");
        return;
    }
    causa_reader_close(reader);

    if (scm2->num_variables != scm->num_variables) {
        test_fail("variable count mismatch\n");
        return;
    }

    if (scm2->variables[0].num_parents != 1) {
        test_fail("parent count not preserved\n");
        return;
    }

    // The parent has to be a real tensor again, and the noise as well.
    // Without them the model would be a table of numbers rather than
    // something that can be run.
    if (!scm2->variables[0].parents[0]) {
        test_fail("the parent was not restored\n");
        return;
    }
    if (!scm2->variables[0].noise) {
        test_fail("the noise was not restored\n");
        return;
    }
    if (scm2->variables[0].parents[0] == scm2->variables[0].variable) {
        test_fail("the parent is the variable itself\n");
        return;
    }
    // The parent has to be the same tensor the model was built with. The
    // reader makes new tensors, so the values are what tell them apart.
    if (fabsf(((float*)scm2->variables[0].parents[0]->data)[0] - 3.0f) > 1e-6f) {
        test_fail("the parent did not keep its value\n");
        return;
    }

    // The model has to still run. A model that loads but cannot be
    // evaluated is no better than one that failed to load.
    {
        causa_tensor_t* out = causa_tensor_new(arena2, backend, shape, 1, &opts);
        causa_tensor_t* outputs[1] = {out};
        causa_tensor_t* result = causa_scm_evaluate(scm2, outputs, 1);
        if (!result) {
            test_fail("the loaded model could not be evaluated\n");
            return;
        }
    }

    // The variables have to keep their values.
    if (fabsf(((float*)scm2->variables[0].variable->data)[0] - 7.0f) > 1e-6f) {
        test_fail("the variable did not keep its value\n");
        return;
    }
    if (fabsf(((float*)scm2->variables[0].noise->data)[0] - 0.5f) > 1e-6f) {
        test_fail("the noise did not keep its value\n");
        return;
    }

    causa_arena_free(arena2);
    causa_backend_free(backend);
    causa_arena_free(arena);
    remove(path);
    printf("PASS: serialization (SCM roundtrip)\n");
}

void test_kernels() {
    printf("Testing kernels (matmul/bmm/conv/reduce)...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;

    // --- matmul ---
    {
        int64_t shape[] = {2, 2};
        causa_tensor_t* a = causa_tensor_new(arena, backend, shape, 2, &opts);
        causa_tensor_t* b = causa_tensor_new(arena, backend, shape, 2, &opts);
        float* da = (float*)a->data;
        float* db = (float*)b->data;
        da[0]=1; da[1]=2; da[2]=3; da[3]=4;
        db[0]=5; db[1]=6; db[2]=7; db[3]=8;

        causa_tensor_t* c = causa_tensor_matmul(a, b);
        if (!c) { test_fail("matmul returned NULL\n"); return; }
        float* dc = (float*)c->data;
        float expected[4] = {19, 22, 43, 50};
        for (int i = 0; i < 4; i++) {
            if (fabsf(dc[i] - expected[i]) > 1e-4) {
                test_fail("matmul[%d] = %f, expected %f\n", i, dc[i], expected[i]);
                return;
            }
        }
    }

    // --- batched matmul ---
    {
        int64_t shape[] = {2, 2, 2};  // [batch=2, M=2, K=2]
        causa_tensor_t* a = causa_tensor_new(arena, backend, shape, 3, &opts);
        causa_tensor_t* b = causa_tensor_new(arena, backend, shape, 3, &opts);
        float* da = (float*)a->data;
        float* db = (float*)b->data;
        for (int i = 0; i < 8; i++) { da[i] = (float)(i + 1); db[i] = (float)(i + 1); }

        causa_tensor_t* c = causa_tensor_bmm(a, b);
        if (!c) { test_fail("bmm returned NULL\n"); return; }
        float* dc = (float*)c->data;
        // batch 0: [[1,2],[3,4]] @ [[1,2],[3,4]] = [[7,10],[15,22]]
        if (fabsf(dc[0] - 7.0f) > 1e-4 || fabsf(dc[1] - 10.0f) > 1e-4 ||
            fabsf(dc[2] - 15.0f) > 1e-4 || fabsf(dc[3] - 22.0f) > 1e-4) {
            test_fail("bmm batch 0: %f %f %f %f\n", dc[0], dc[1], dc[2], dc[3]);
            return;
        }
        // batch 1: [[5,6],[7,8]] @ [[5,6],[7,8]] = [[67,78],[91,106]]
        if (fabsf(dc[4] - 67.0f) > 1e-4 || fabsf(dc[5] - 78.0f) > 1e-4 ||
            fabsf(dc[6] - 91.0f) > 1e-4 || fabsf(dc[7] - 106.0f) > 1e-4) {
            test_fail("bmm batch 1: %f %f %f %f\n", dc[4], dc[5], dc[6], dc[7]);
            return;
        }
    }

    // --- reductions ---
    {
        int64_t shape[] = {2, 3};
        causa_tensor_t* t = causa_tensor_new(arena, backend, shape, 2, &opts);
        float* dt = (float*)t->data;
        for (int i = 0; i < 6; i++) dt[i] = (float)(i + 1);  // 1..6

        int32_t dims[] = {0, 1};
        causa_tensor_t* total = causa_tensor_reduce(t, CAUSA_REDUCE_SUM, dims, 2, true);
        if (!total) { test_fail("reduce returned NULL\n"); return; }
        float sum = ((float*)total->data)[0];
        if (fabsf(sum - 21.0f) > 1e-4) {
            test_fail("reduce sum = %f, expected 21\n", sum);
            return;
        }

        causa_tensor_t* mean = causa_tensor_reduce(t, CAUSA_REDUCE_MEAN, dims, 2, true);
        float avg = ((float*)mean->data)[0];
        if (fabsf(avg - 3.5f) > 1e-4) {
            test_fail("reduce mean = %f, expected 3.5\n", avg);
            return;
        }

        causa_tensor_t* mx = causa_tensor_reduce(t, CAUSA_REDUCE_MAX, dims, 2, true);
        float maximum = ((float*)mx->data)[0];
        if (fabsf(maximum - 6.0f) > 1e-4) {
            test_fail("reduce max = %f, expected 6\n", maximum);
            return;
        }
    }

    // --- conv1d ---
    {
        int64_t in_shape[] = {1, 1, 5};      // [N=1, C=1, L=5]
        int64_t w_shape[] = {1, 1, 3};       // [C_out=1, C_in=1, k=3]
        causa_tensor_t* in = causa_tensor_new(arena, backend, in_shape, 3, &opts);
        causa_tensor_t* w = causa_tensor_new(arena, backend, w_shape, 3, &opts);
        float* di = (float*)in->data;
        float* dw = (float*)w->data;
        for (int i = 0; i < 5; i++) di[i] = (float)(i + 1);
        dw[0] = 1.0f; dw[1] = 0.0f; dw[2] = 0.0f;  // identity kernel

        causa_tensor_t* out = causa_tensor_conv1d(in, w, NULL, 1, 0, 1, 1);
        if (!out) { test_fail("conv1d returned NULL\n"); return; }
        float* dout = (float*)out->data;
        // L_out = 3, identity kernel -> [1,2,3]
        if (out->shape[2] != 3) {
            test_fail("conv1d output length = %ld, expected 3\n", (long)out->shape[2]);
            return;
        }
        for (int i = 0; i < 3; i++) {
            if (fabsf(dout[i] - (float)(i + 1)) > 1e-4) {
                test_fail("conv1d[%d] = %f, expected %d\n", i, dout[i], i + 1);
                return;
            }
        }
    }

    // --- conv2d ---
    {
        int64_t in_shape[] = {1, 1, 3, 3};   // [N=1, C=1, H=3, W=3]
        int64_t w_shape[] = {1, 1, 2, 2};    // [C_out=1, C_in=1, 2x2]
        causa_tensor_t* in = causa_tensor_new(arena, backend, in_shape, 4, &opts);
        causa_tensor_t* w = causa_tensor_new(arena, backend, w_shape, 4, &opts);
        float* di = (float*)in->data;
        float* dw = (float*)w->data;
        for (int i = 0; i < 9; i++) di[i] = (float)(i + 1);
        // all-ones 2x2 kernel: output is the sum of each 2x2 window
        dw[0] = 1.0f; dw[1] = 1.0f; dw[2] = 1.0f; dw[3] = 1.0f;

        int32_t stride[2] = {1, 1};
        int32_t padding[2] = {0, 0};
        int32_t dilation[2] = {1, 1};
        causa_tensor_t* out = causa_tensor_conv2d(in, w, NULL, stride, padding, dilation, 1);
        if (!out) { test_fail("conv2d returned NULL\n"); return; }
        if (out->shape[2] != 2 || out->shape[3] != 2) {
            test_fail("conv2d output shape\n");
            return;
        }
        float* dout = (float*)out->data;
        // window [0,0]: 1+2+4+5 = 12; [0,1]: 2+3+5+6 = 16; [1,0]: 4+5+7+8 = 24; [1,1]: 5+6+8+9 = 28
        if (fabsf(dout[0] - 12.0f) > 1e-4 || fabsf(dout[1] - 16.0f) > 1e-4 ||
            fabsf(dout[2] - 24.0f) > 1e-4 || fabsf(dout[3] - 28.0f) > 1e-4) {
            test_fail("conv2d: %f %f %f %f (expected 12 16 24 28)\n",
                   dout[0], dout[1], dout[2], dout[3]);
            return;
        }
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: kernels (matmul/bmm/conv/reduce)\n");
}

void test_causal_memory_integration() {
    printf("Testing causal-memory integration...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(4 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    // Three variables: X0 -> X1 -> X2
    const int32_t D = 3;

    causa_memory_config_t mcfg = CAUSA_MEMORY_CONFIG_DEFAULTS;
    mcfg.capacity[CAUSA_MEMORY_WORKING] = 16;
    mcfg.capacity[CAUSA_MEMORY_EPISODIC] = 512;
    mcfg.capacity[CAUSA_MEMORY_SEMANTIC] = 64;
    mcfg.capacity[CAUSA_MEMORY_PROCEDURAL] = 16;
    mcfg.key_dim = D;
    mcfg.value_dim = D;
    mcfg.working_decay_rate = 0.0f;
    mcfg.episodic_decay_rate = 0.0f;

    causa_memory_t* mem = causa_memory_new(arena, backend, &mcfg);
    if (!mem) { test_fail("memory creation\n"); return; }

    // Record episodes: each key holds one observation of the three variables
    unsigned int seed = 98765u;
    for (int i = 0; i < 300; i++) {
        float r[6];
        for (int j = 0; j < 6; j++) {
            seed = seed * 1103515245u + 12345u;
            r[j] = ((float)((seed >> 16) & 0x7fff) / 16384.0f) - 1.0f;
        }
        float x0 = r[0];
        float x1 = 1.5f * x0 + 0.25f * r[1];
        float x2 = 2.0f * x1 + 0.25f * r[2];

        causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        opts.dtype = CAUSA_DTYPE_F32;
        opts.device = CAUSA_DEVICE_CPU;

        int64_t shape[] = {D};
        causa_tensor_t* key = causa_tensor_new(arena, backend, shape, 1, &opts);
        causa_tensor_t* value = causa_tensor_new(arena, backend, shape, 1, &opts);
        float* kd = (float*)key->data;
        float* vd = (float*)value->data;
        kd[0] = x0; kd[1] = x1; kd[2] = x2;
        vd[0] = x1; vd[1] = x2; vd[2] = 0.0f;

        causa_memory_write(mem, key, value, CAUSA_MEMORY_EPISODIC);
    }

    int64_t episodic = causa_memory_size(mem, CAUSA_MEMORY_EPISODIC);
    if (episodic != 300) {
        test_fail("expected 300 episodes, got %ld\n", (long)episodic);
        return;
    }

    // Gather the observations
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;

    int64_t obs_shape[] = {300, D};
    causa_tensor_t* observations = causa_tensor_new(arena, backend, obs_shape, 2, &opts);
    int64_t gathered = causa_memory_collect(mem, CAUSA_MEMORY_EPISODIC, observations);
    if (gathered != 300) {
        test_fail("collected %ld observations, expected 300\n", (long)gathered);
        return;
    }

    // The collected observations should still show the generating relationship
    // x1 = 1.5 * x0
    float* od = (float*)observations->data;
    float sum_x0 = 0.0f, sum_x1 = 0.0f;
    for (int i = 0; i < 300; i++) {
        sum_x0 += od[i * D + 0];
        sum_x1 += od[i * D + 1];
    }
    if (fabsf(sum_x0) < 1e-3f) {
        test_fail("collected observations are degenerate\n");
        return;
    }
    float ratio = sum_x1 / sum_x0;
    if (fabsf(ratio - 1.5f) > 0.15f) {
        test_fail("collected observations lost the generating ratio, got %f\n", ratio);
        return;
    }

    // Consolidate: episodic observations -> semantic causal laws
    causa_config_t ccfg = CAUSA_CONFIG_DEFAULTS;
    ccfg.discovery_alpha = 0.05f;
    ccfg.discovery_threshold = 0.25f;
    ccfg.max_iterations = 300;

    causa_scm_t* scm = causa_scm_new(arena, backend, &ccfg);
    int32_t laws = causa_memory_consolidate_causal(mem, scm, &ccfg);
    if (laws <= 0) {
        test_fail("consolidation produced %d laws\n", laws);
        return;
    }

    // The learned laws should be recoverable from semantic memory
    causa_tensor_t* found[16];
    int32_t n = causa_memory_find_causal_laws(mem, 1, found, 16);
    if (n <= 0) {
        test_fail("no causal law found for variable 1\n");
        return;
    }

    // The law for variable 1 should name variable 0 as a parent
    const float* coeffs = (const float*)found[0]->data;
    bool has_parent_0 = fabsf(coeffs[0]) > 1e-3f;

    int32_t n2 = causa_memory_find_causal_laws(mem, 2, found, 16);
    if (n2 <= 0) {
        test_fail("no causal law found for variable 2\n");
        return;
    }
    const float* coeffs2 = (const float*)found[0]->data;
    bool has_parent_1 = fabsf(coeffs2[1]) > 1e-3f;

    if (!has_parent_0) {
        test_fail("the law for variable 1 does not name variable 0 as a parent\n");
        return;
    }
    if (!has_parent_1) {
        test_fail("the law for variable 2 does not name variable 1 as a parent\n");
        return;
    }

    printf("  %d laws consolidated, %d for variable 1, %d for variable 2\n", laws, n, n2);
    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: causal-memory integration\n");
}

void test_abduction() {
    printf("Testing counterfactual abduction...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(4 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    // A linear SCM:  X1 = 1.5 * X0 + U1
    //                X2 = 2.0 * X1 + U2
    // The abduction must recover U1 and U2 from the observed values.
    causa_graph_t* graph = causa_graph_new(arena, backend);

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    opts.requires_grad = true;

    int64_t shape[] = {1};

    causa_tensor_t* X0 = causa_graph_new_tensor(graph, shape, 1, &opts);
    causa_tensor_t* X1 = causa_graph_new_tensor(graph, shape, 1, &opts);
    causa_tensor_t* X2 = causa_graph_new_tensor(graph, shape, 1, &opts);
    causa_tensor_t* U1 = causa_graph_new_tensor(graph, shape, 1, &opts);
    causa_tensor_t* U2 = causa_graph_new_tensor(graph, shape, 1, &opts);

    // U0 is the exogenous noise of X0; X0 itself is observed, so it has no
    // equation to infer.
    causa_tensor_t* U0 = causa_graph_new_tensor(graph, shape, 1, &opts);

    causa_tensor_t* c15 = causa_graph_new_tensor(graph, shape, 1, &opts);
    causa_tensor_fill(c15, 1.5f);
    causa_tensor_t* c20 = causa_graph_new_tensor(graph, shape, 1, &opts);
    causa_tensor_fill(c20, 2.0f);

    // X1 = 1.5 * X0 + U1
    causa_tensor_t* m1 = causa_graph_new_tensor(graph, shape, 1, &opts);
    causa_graph_add_op(graph, CAUSA_OP_MUL, "1.5*X0",
        (causa_tensor_t*[]){c15, X0}, 2, (causa_tensor_t*[]){m1}, 1, NULL, NULL, NULL, 0);
    causa_graph_add_op(graph, CAUSA_OP_ADD, "X1",
        (causa_tensor_t*[]){m1, U1}, 2, (causa_tensor_t*[]){X1}, 1, NULL, NULL, NULL, 0);

    // X2 = 2.0 * X1 + U2
    causa_tensor_t* m2 = causa_graph_new_tensor(graph, shape, 1, &opts);
    causa_graph_add_op(graph, CAUSA_OP_MUL, "2.0*X1",
        (causa_tensor_t*[]){c20, X1}, 2, (causa_tensor_t*[]){m2}, 1, NULL, NULL, NULL, 0);
    causa_graph_add_op(graph, CAUSA_OP_ADD, "X2",
        (causa_tensor_t*[]){m2, U2}, 2, (causa_tensor_t*[]){X2}, 1, NULL, NULL, NULL, 0);

    causa_graph_build(graph);

    causa_scm_t* scm = causa_scm_new(arena, backend, NULL);
    scm->base_graph = graph;
    scm->num_exogenous = 3;
    scm->exogenous_noise = (causa_tensor_t**)causa_arena_alloc(arena, sizeof(causa_tensor_t*) * 3, CAUSA_ALIGNOF(causa_tensor_t*));
    scm->exogenous_noise[0] = U0;
    scm->exogenous_noise[1] = U1;
    scm->exogenous_noise[2] = U2;

    // Equations, recorded so the SCM knows the causal structure.
    // X0 is registered first as an observed variable, so that the SCM's
    // variable indices line up with the observed values given to abduction.
    {
        causa_tensor_t* parents1[] = {X0};
        causa_tensor_t* parents2[] = {X1};
        causa_scm_var_t* v;

        causa_scm_add_observed_variable(scm, X0);

        v = &scm->variables[scm->num_variables++];
        v->variable = X1; v->num_parents = 1; v->parents = parents1;
        v->noise = U1; v->is_intervened = false; v->intervention_value = NULL;
        v = &scm->variables[scm->num_variables++];
        v->variable = X2; v->num_parents = 1; v->parents = parents2;
        v->noise = U2; v->is_intervened = false; v->intervention_value = NULL;
    }
    causa_scm_build(scm);

    // The ground truth: X0 = 2, U1 = 0.5, U2 = -0.25
    // so X1 = 1.5*2 + 0.5 = 3.5 and X2 = 2*3.5 - 0.25 = 6.75
    ((float*)X0->data)[0] = 2.0f;
    ((float*)U1->data)[0] = 0.5f;
    ((float*)U2->data)[0] = -0.25f;

    causa_graph_forward(graph);
    float x1_true = ((float*)X1->data)[0];
    float x2_true = ((float*)X2->data)[0];
    if (fabsf(x1_true - 3.5f) > 1e-4f || fabsf(x2_true - 6.75f) > 1e-4f) {
        test_fail("the model did not evaluate as expected (%f, %f)\n", x1_true, x2_true);
        return;
    }

    // Now hide the noise and let abduction recover it.
    // Observed values are what the abduction must explain.
    causa_tensor_t* obs0 = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* obs1 = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* obs2 = causa_tensor_new(arena, backend, shape, 1, &opts);
    ((float*)obs0->data)[0] = 2.0f;
    ((float*)obs1->data)[0] = 3.5f;
    ((float*)obs2->data)[0] = 6.75f;

    // Start the noise away from the truth
    ((float*)U1->data)[0] = 0.0f;
    ((float*)U2->data)[0] = 0.0f;

    causa_tensor_t* evidence[] = {obs0, obs1, obs2};
    causa_counterfactual_t* cf = causa_counterfactual_new(arena, X2, X0, X0, obs2);
    causa_counterfactual_set_evidence(cf, evidence, 3);

    // The abduction minimises the reconstruction error over the equations.
    // The learning rate has to stay below 2/lambda_max of the Hessian, which
    // is about 21 for this model, so 0.01 is safely inside the stable range.
    int rc = causa_counterfactual_abduction(cf, scm, 0.01f, 2000);
    if (rc != 0) {
        test_fail("abduction returned %d\n", rc);
        return;
    }

    float u1 = ((float*)U1->data)[0];
    float u2 = ((float*)U2->data)[0];
    printf("  inferred U1 = %f (true 0.5), U2 = %f (true -0.25)\n", u1, u2);

    if (!(u1 == u1) || !(u2 == u2)) {
        test_fail("abduction diverged\n");
        return;
    }
    if (fabsf(u1 - 0.5f) > 1e-2f) {
        test_fail("U1 was not recovered\n");
        return;
    }
    if (fabsf(u2 + 0.25f) > 1e-2f) {
        test_fail("U2 was not recovered\n");
        return;
    }

    // The full counterfactual: what would X2 have been had X0 been 5?
    // X1 = 1.5 * 5 + U1 = 8.0
    // X2 = 2.0 * 8.0 + U2 = 15.75
    causa_tensor_t* x0_cf = causa_tensor_new(arena, backend, shape, 1, &opts);
    ((float*)x0_cf->data)[0] = 5.0f;

    causa_counterfactual_t* cf2 = causa_counterfactual_new(arena, X2, X0, x0_cf, obs2);
    causa_counterfactual_set_evidence(cf2, evidence, 3);

    causa_counterfactual_compute(scm, cf2);

    if (!cf2->result) {
        test_fail("counterfactual produced no result\n");
        return;
    }

    float y_cf = ((float*)cf2->result->data)[0];
    printf("  counterfactual X2 given do(X0=5) = %f (expected 15.75)\n", y_cf);
    if (!(y_cf == y_cf)) {
        test_fail("counterfactual diverged\n");
        return;
    }
    if (fabsf(y_cf - 15.75f) > 0.05f) {
        test_fail("counterfactual value is wrong\n");
        return;
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: counterfactual abduction\n");
}

void test_memory_reasoning() {
    printf("Testing reasoning over memory...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(16 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    // Three variables: X0 -> X1 -> X2
    //
    // The relations carry an offset, which a linear law cannot express.
    // The law gives the slope and the episodes give the offset, so the
    // prediction has to use both.
    const int32_t D = 3;

    causa_memory_config_t mcfg = CAUSA_MEMORY_CONFIG_DEFAULTS;
    mcfg.capacity[CAUSA_MEMORY_WORKING] = 16;
    mcfg.capacity[CAUSA_MEMORY_EPISODIC] = 512;
    mcfg.capacity[CAUSA_MEMORY_SEMANTIC] = 64;
    mcfg.capacity[CAUSA_MEMORY_PROCEDURAL] = 16;
    mcfg.key_dim = D;
    mcfg.value_dim = D;
    mcfg.working_decay_rate = 0.0f;
    mcfg.episodic_decay_rate = 0.0f;

    causa_memory_t* mem = causa_memory_new(arena, backend, &mcfg);
    if (!mem) { test_fail("memory creation\n"); return; }

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    int64_t shape[] = {D};

    unsigned int seed = 24680u;
    for (int i = 0; i < 400; i++) {
        float r[6];
        for (int j = 0; j < 6; j++) {
            seed = seed * 1103515245u + 12345u;
            r[j] = ((float)((seed >> 16) & 0x7fff) / 16384.0f) - 1.0f;
        }
        float x0 = r[0];
        float x1 = 1.5f * x0 + 0.5f + 0.25f * r[1];
        float x2 = 2.0f * x1 + 0.25f + 0.25f * r[2];

        causa_tensor_t* key = causa_tensor_new(arena, backend, shape, 1, &opts);
        causa_tensor_t* value = causa_tensor_new(arena, backend, shape, 1, &opts);
        float* kd = (float*)key->data;
        float* vd = (float*)value->data;
        kd[0] = x0; kd[1] = x1; kd[2] = x2;
        vd[0] = x1; vd[1] = x2; vd[2] = 0.0f;

        causa_memory_write(mem, key, value, CAUSA_MEMORY_EPISODIC);
    }

    causa_config_t ccfg = CAUSA_CONFIG_DEFAULTS;
    ccfg.discovery_alpha = 0.05f;
    ccfg.discovery_threshold = 0.25f;
    ccfg.max_iterations = 300;

    causa_scm_t* scm = causa_scm_new(arena, backend, &ccfg);
    int32_t laws = causa_memory_consolidate_causal(mem, scm, &ccfg);
    if (laws <= 0) {
        test_fail("consolidation produced %d laws\n", laws);
        return;
    }

    // ---- Predicting an unobserved variable ----

    // Only X0 is observed. X1 and X2 have to be predicted.
    causa_tensor_t* observation = causa_tensor_new(arena, backend, shape, 1, &opts);
    float* od = (float*)observation->data;
    od[0] = 0.8f;
    od[1] = NAN;
    od[2] = NAN;

    causa_tensor_t* prediction = causa_tensor_new(arena, backend, shape, 1, &opts);
    int32_t predicted = causa_memory_predict(mem, observation, prediction);
    if (predicted != 2) {
        test_fail("expected 2 variables to be predicted, got %d\n", predicted);
        return;
    }

    float* pd = (float*)prediction->data;
    float expected_x1 = 1.5f * 0.8f + 0.5f;   // 1.7
    float expected_x2 = 2.0f * expected_x1 + 0.25f;  // 3.65

    printf("  predicted X1 = %f (expected %f)\n", pd[1], expected_x1);
    printf("  predicted X2 = %f (expected %f)\n", pd[2], expected_x2);

    if (!(pd[1] == pd[1]) || !(pd[2] == pd[2])) {
        test_fail("the prediction is not a number\n");
        return;
    }
    if (fabsf(pd[1] - expected_x1) > 0.25f) {
        test_fail("X1 was predicted as %f, expected %f\n", pd[1], expected_x1);
        return;
    }
    if (fabsf(pd[2] - expected_x2) > 0.45f) {
        test_fail("X2 was predicted as %f, expected %f\n", pd[2], expected_x2);
        return;
    }

    // The observed variable has to come through untouched.
    if (fabsf(pd[0] - 0.8f) > 1e-6f) {
        test_fail("the observed variable was changed\n");
        return;
    }

    // ---- Building a model from what was learned ----

    causa_scm_t* learned = causa_scm_new(arena, backend, &ccfg);
    int32_t equations = causa_memory_build_scm(mem, learned, D);
    if (equations <= 0) {
        test_fail("no equations were built from the laws\n");
        return;
    }
    printf("  built %d equations from the laws\n", equations);

    // The model has to reproduce the relation it was built from. The
    // variables of the model are scalars, so the evidence is given as
    // scalars too.
    int64_t scalar_shape[] = {1};
    causa_tensor_t* evidence[3];
    for (int i = 0; i < 3; i++) {
        evidence[i] = causa_tensor_new(arena, backend, scalar_shape, 1, &opts);
    }
    ((float*)evidence[0]->data)[0] = 1.0f;
    ((float*)evidence[1]->data)[0] = 2.0f;
    ((float*)evidence[2]->data)[0] = 4.0f;

    // The counterfactual runs on the model that came out of memory. The
    // coefficients are the ones that were learned, so the answer is close
    // to the truth rather than exact.
    causa_tensor_t* intervention = causa_tensor_new(arena, backend, scalar_shape, 1, &opts);
    ((float*)intervention->data)[0] = 5.0f;

    causa_counterfactual_t* cf = causa_counterfactual_new(
        arena,
        learned->variables[2].variable,   // the query: X2
        learned->variables[0].variable,   // the intervention: X0
        intervention,
        evidence[2]);
    causa_counterfactual_set_evidence(cf, evidence, 3);

    causa_counterfactual_compute(learned, cf);

    if (!cf->result) {
        test_fail("the counterfactual on the learned model produced no result\n");
        return;
    }

    // The truth: X1 = 1.5 * 5 + 0.5 = 8.0, X2 = 2 * 8.0 + 0.25 = 16.25.
    // The model that came out of memory has no intercept, so the constant
    // terms are carried by the noise and the answer is a little different.
    float y_cf = ((float*)cf->result->data)[0];
    printf("  counterfactual X2 given do(X0=5) = %f (expected about 16)\n", y_cf);

    if (!(y_cf == y_cf)) {
        test_fail("the counterfactual diverged\n");
        return;
    }
    if (fabsf(y_cf - 16.0f) > 1.0f) {
        test_fail("the counterfactual on the learned model is wrong\n");
        return;
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: reasoning over memory\n");
}

// A model that came out of memory has to run.
//
// The equations of such a model are operations of a graph rather than
// functions attached to the variables, so a model that only knows how to
// call those functions would compute nothing at all.
void test_learned_model_runs() {
    printf("Testing that a learned model runs...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(16 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    const int32_t D = 3;

    causa_memory_config_t mcfg = CAUSA_MEMORY_CONFIG_DEFAULTS;
    mcfg.capacity[CAUSA_MEMORY_WORKING] = 16;
    mcfg.capacity[CAUSA_MEMORY_EPISODIC] = 512;
    mcfg.capacity[CAUSA_MEMORY_SEMANTIC] = 64;
    mcfg.capacity[CAUSA_MEMORY_PROCEDURAL] = 16;
    mcfg.key_dim = D;
    mcfg.value_dim = D;
    mcfg.working_decay_rate = 0.0f;
    mcfg.episodic_decay_rate = 0.0f;

    causa_memory_t* mem = causa_memory_new(arena, backend, &mcfg);
    if (!mem) { test_fail("memory creation\n"); return; }

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    int64_t shape[] = {D};

    // X0 -> X1 -> X2, with no constant terms, so that the law is the whole
    // story.
    unsigned int seed = 13579u;
    for (int i = 0; i < 400; i++) {
        float r[6];
        for (int j = 0; j < 6; j++) {
            seed = seed * 1103515245u + 12345u;
            r[j] = ((float)((seed >> 16) & 0x7fff) / 16384.0f) - 1.0f;
        }
        float x0 = r[0];
        float x1 = 1.5f * x0 + 0.25f * r[1];
        float x2 = 2.0f * x1 + 0.25f * r[2];

        causa_tensor_t* key = causa_tensor_new(arena, backend, shape, 1, &opts);
        causa_tensor_t* value = causa_tensor_new(arena, backend, shape, 1, &opts);
        ((float*)key->data)[0] = x0;
        ((float*)key->data)[1] = x1;
        ((float*)key->data)[2] = x2;
        ((float*)value->data)[0] = x1;
        ((float*)value->data)[1] = x2;
        ((float*)value->data)[2] = 0.0f;

        causa_memory_write(mem, key, value, CAUSA_MEMORY_EPISODIC);
    }

    causa_config_t ccfg = CAUSA_CONFIG_DEFAULTS;
    ccfg.discovery_alpha = 0.05f;
    ccfg.discovery_threshold = 0.25f;
    ccfg.max_iterations = 300;

    causa_scm_t* scm = causa_scm_new(arena, backend, &ccfg);
    causa_memory_consolidate_causal(mem, scm, &ccfg);

    causa_scm_t* learned = causa_scm_new(arena, backend, &ccfg);
    int32_t equations = causa_memory_build_scm(mem, learned, D);
    if (equations <= 0) {
        test_fail("no equations were built from the laws\n");
        return;
    }

    // The first variable is an input, so its value is set by hand.
    ((float*)learned->variables[0].variable->data)[0] = 1.0f;

    causa_tensor_t* outputs[3] = {
        learned->variables[0].variable,
        learned->variables[1].variable,
        learned->variables[2].variable
    };

    causa_tensor_t* result = causa_scm_evaluate(learned, outputs, 3);
    if (!result) {
        test_fail("the learned model could not be evaluated\n");
        return;
    }

    float x1 = ((float*)learned->variables[1].variable->data)[0];
    float x2 = ((float*)learned->variables[2].variable->data)[0];

    printf("  the model gives X1 = %f and X2 = %f\n", x1, x2);

    // The law is X1 = 1.5 * X0 and X2 = 2.0 * X1, and the noise carries no
    // constant, so with X0 = 1 the answer is 1.5 and 3.0. The coefficients
    // are learned, so the answers are close rather than exact.
    if (!(x1 == x1) || !(x2 == x2)) {
        test_fail("the learned model produced a number that is not a number\n");
        return;
    }
    if (fabsf(x1 - 1.5f) > 0.2f) {
        test_fail("the learned model gave X1 = %f, expected about 1.5\n", x1);
        return;
    }
    if (fabsf(x2 - 3.0f) > 0.4f) {
        test_fail("the learned model gave X2 = %f, expected about 3.0\n", x2);
        return;
    }

    // Running the model again has to give the same answer. A model that
    // only worked once would be no use.
    ((float*)learned->variables[0].variable->data)[0] = 1.0f;
    causa_scm_evaluate(learned, outputs, 3);
    float x1_again = ((float*)learned->variables[1].variable->data)[0];
    if (fabsf(x1_again - x1) > 1e-5f) {
        test_fail("the model gave %f the second time and %f the first\n", x1_again, x1);
        return;
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: a learned model runs\n");
}

void test_procedural_memory() {
    printf("Testing procedural memory...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(16 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    const int32_t D = 3;

    causa_memory_config_t mcfg = CAUSA_MEMORY_CONFIG_DEFAULTS;
    mcfg.capacity[CAUSA_MEMORY_WORKING] = 16;
    mcfg.capacity[CAUSA_MEMORY_EPISODIC] = 256;
    mcfg.capacity[CAUSA_MEMORY_SEMANTIC] = 64;
    mcfg.capacity[CAUSA_MEMORY_PROCEDURAL] = 32;
    mcfg.key_dim = D;
    mcfg.value_dim = D;
    mcfg.working_decay_rate = 0.0f;
    mcfg.episodic_decay_rate = 0.0f;

    causa_memory_t* mem = causa_memory_new(arena, backend, &mcfg);
    if (!mem) { test_fail("memory creation\n"); return; }

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    int64_t shape[] = {D};

    // A hundred different states, each with what came of it
    for (int i = 0; i < 100; i++) {
        float x0 = -1.0f + 0.02f * (float)i;
        float x1 = 1.5f * x0 + 0.5f;
        float x2 = 2.0f * x1 + 0.25f;

        causa_tensor_t* key = causa_tensor_new(arena, backend, shape, 1, &opts);
        causa_tensor_t* value = causa_tensor_new(arena, backend, shape, 1, &opts);
        float* kd = (float*)key->data;
        float* vd = (float*)value->data;
        kd[0] = x0; kd[1] = x1; kd[2] = x2;
        vd[0] = x1; vd[1] = x2; vd[2] = 0.0f;

        causa_memory_write(mem, key, value, CAUSA_MEMORY_EPISODIC);
    }

    // The episodes are grouped into procedures
    int32_t procedures = causa_memory_learn_procedures(mem, 0.5f);
    printf("  %d procedures from 100 episodes\n", procedures);

    if (procedures <= 0) {
        test_fail("no procedures were learned\n");
        return;
    }
    if (procedures >= 100) {
        test_fail("the episodes were not grouped into procedures\n");
        return;
    }

    // The procedure that applies to a state is the nearest one
    causa_tensor_t* state = causa_tensor_new(arena, backend, shape, 1, &opts);
    float* st = (float*)state->data;
    st[0] = 0.5f; st[1] = 1.25f; st[2] = 2.75f;

    causa_tensor_t* triggers[8];
    causa_tensor_t* actions[8];
    int32_t found = causa_memory_find_procedures(mem, state, triggers, actions, 8);
    if (found <= 0) {
        test_fail("no procedure was found\n");
        return;
    }

    // The nearest trigger has to be within the radius that was used
    const float* td = (const float*)triggers[0]->data;
    float distance = 0.0f;
    for (int32_t j = 0; j < D; j++) {
        float diff = td[j] - st[j];
        distance += diff * diff;
    }
    distance = sqrtf(distance);

    printf("  nearest procedure is %.3f away\n", distance);
    if (distance > 0.5f) {
        test_fail("the nearest procedure is too far away\n");
        return;
    }

    // A procedure that was stored by hand is found as well
    causa_tensor_t* trigger = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* action = causa_tensor_new(arena, backend, shape, 1, &opts);
    float* tr = (float*)trigger->data;
    float* ac = (float*)action->data;
    tr[0] = 100.0f; tr[1] = 100.0f; tr[2] = 100.0f;
    ac[0] = 1.0f; ac[1] = 2.0f; ac[2] = 3.0f;

    if (!causa_memory_store_procedure(mem, trigger, action)) {
        test_fail("the procedure was not stored\n");
        return;
    }

    causa_tensor_t* far_state = causa_tensor_new(arena, backend, shape, 1, &opts);
    ((float*)far_state->data)[0] = 99.0f;
    ((float*)far_state->data)[1] = 99.0f;
    ((float*)far_state->data)[2] = 99.0f;

    found = causa_memory_find_procedures(mem, far_state, triggers, actions, 8);
    if (found <= 0) {
        test_fail("the stored procedure was not found\n");
        return;
    }
    const float* ad = (const float*)actions[0]->data;
    if (fabsf(ad[0] - 1.0f) > 1e-5f) {
        test_fail("the wrong action came back\n");
        return;
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: procedural memory\n");
}

void test_intervention_planning() {
    printf("Testing intervention planning...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(16 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    const int32_t D = 3;

    causa_memory_config_t mcfg = CAUSA_MEMORY_CONFIG_DEFAULTS;
    mcfg.capacity[CAUSA_MEMORY_WORKING] = 16;
    mcfg.capacity[CAUSA_MEMORY_EPISODIC] = 512;
    mcfg.capacity[CAUSA_MEMORY_SEMANTIC] = 64;
    mcfg.capacity[CAUSA_MEMORY_PROCEDURAL] = 32;
    mcfg.key_dim = D;
    mcfg.value_dim = D;
    mcfg.working_decay_rate = 0.0f;
    mcfg.episodic_decay_rate = 0.0f;

    causa_memory_t* mem = causa_memory_new(arena, backend, &mcfg);
    if (!mem) { test_fail("memory creation\n"); return; }

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    int64_t shape[] = {D};

    unsigned int seed = 13579u;
    for (int i = 0; i < 400; i++) {
        float r[3];
        for (int j = 0; j < 3; j++) {
            seed = seed * 1103515245u + 12345u;
            r[j] = ((float)((seed >> 16) & 0x7fff) / 16384.0f) - 1.0f;
        }
        float x0 = r[0];
        float x1 = 1.5f * x0 + 0.5f + 0.25f * r[1];
        float x2 = 2.0f * x1 + 0.25f + 0.25f * r[2];

        causa_tensor_t* key = causa_tensor_new(arena, backend, shape, 1, &opts);
        causa_tensor_t* value = causa_tensor_new(arena, backend, shape, 1, &opts);
        float* kd = (float*)key->data;
        float* vd = (float*)value->data;
        kd[0] = x0; kd[1] = x1; kd[2] = x2;
        vd[0] = x1; vd[1] = x2; vd[2] = 0.0f;

        causa_memory_write(mem, key, value, CAUSA_MEMORY_EPISODIC);
    }

    causa_config_t ccfg = CAUSA_CONFIG_DEFAULTS;
    ccfg.discovery_alpha = 0.05f;
    ccfg.discovery_threshold = 0.25f;
    ccfg.max_iterations = 300;

    causa_scm_t* scm = causa_scm_new(arena, backend, &ccfg);
    int32_t laws = causa_memory_consolidate_causal(mem, scm, &ccfg);
    if (laws <= 0) {
        test_fail("consolidation produced %d laws\n", laws);
        return;
    }

    // The state: only x0 is known.
    // The goal: x2 should be 10.
    causa_tensor_t* state = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* goal = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* plan = causa_tensor_new(arena, backend, shape, 1, &opts);

    float* sd = (float*)state->data;
    sd[0] = 1.0f; sd[1] = NAN; sd[2] = NAN;

    float* gd = (float*)goal->data;
    gd[0] = NAN; gd[1] = NAN; gd[2] = 10.0f;

    int32_t count = 0;
    if (causa_memory_plan_intervention(mem, state, goal, plan, &count) != 0) {
        test_fail("the intervention could not be planned\n");
        return;
    }

    float* pd = (float*)plan->data;
    printf("  plan: change x0 by %f, x1 by %f, x2 by %f\n", pd[0], pd[1], pd[2]);

    if (count != 1) {
        test_fail("the plan acts on %d variables, expected 1\n", count);
        return;
    }

    // x2 stands at 2 * (1.5 * 1 + 0.5) + 0.25 = 4.25, so it has to move by
    // 5.75. The strongest cause of x2 is x1, which moves it by 2 for 1, so
    // x1 has to move by 2.875.
    if (fabsf(pd[1] - 2.875f) > 0.35f) {
        test_fail("the plan says x1 moves by %f, expected about 2.875\n", pd[1]);
        return;
    }

    // Applying the plan has to reach the goal. x1 stands at 1.5 * 1 + 0.5
    // = 2.0, and x2 follows from it.
    float achieved = 2.0f * (2.0f + pd[1]) + 0.25f;
    printf("  applying the plan puts x2 at %f\n", achieved);
    if (fabsf(achieved - 10.0f) > 0.7f) {
        test_fail("the plan does not reach the goal\n");
        return;
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: intervention planning\n");
}

// A goal that needs several variables changed at once.
//
// The plan has to act on more than one variable, and it has to act on both
// of them: reaching one goal and missing the other would be no use.
void test_intervention_planning_several_goals() {
    printf("Testing intervention planning with several goals...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(16 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    const int32_t D = 3;

    causa_memory_config_t mcfg = CAUSA_MEMORY_CONFIG_DEFAULTS;
    mcfg.capacity[CAUSA_MEMORY_WORKING] = 16;
    mcfg.capacity[CAUSA_MEMORY_EPISODIC] = 512;
    mcfg.capacity[CAUSA_MEMORY_SEMANTIC] = 64;
    mcfg.capacity[CAUSA_MEMORY_PROCEDURAL] = 32;
    mcfg.key_dim = D;
    mcfg.value_dim = D;
    mcfg.working_decay_rate = 0.0f;
    mcfg.episodic_decay_rate = 0.0f;

    causa_memory_t* mem = causa_memory_new(arena, backend, &mcfg);
    if (!mem) { test_fail("memory creation\n"); return; }

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    int64_t shape[] = {D};

    // Two separate chains, so that reaching one goal does not move the
    // other: x0 -> x1 and x2 on its own.
    unsigned int seed = 97531u;
    for (int i = 0; i < 400; i++) {
        float r[3];
        for (int j = 0; j < 3; j++) {
            seed = seed * 1103515245u + 12345u;
            r[j] = ((float)((seed >> 16) & 0x7fff) / 16384.0f) - 1.0f;
        }
        float x0 = r[0];
        float x1 = 1.5f * x0 + 0.5f + 0.25f * r[1];
        float x2 = 3.0f * r[2];

        causa_tensor_t* key = causa_tensor_new(arena, backend, shape, 1, &opts);
        causa_tensor_t* value = causa_tensor_new(arena, backend, shape, 1, &opts);
        float* kd = (float*)key->data;
        float* vd = (float*)value->data;
        kd[0] = x0; kd[1] = x1; kd[2] = x2;
        vd[0] = x1; vd[1] = x2; vd[2] = 0.0f;

        causa_memory_write(mem, key, value, CAUSA_MEMORY_EPISODIC);
    }

    causa_config_t ccfg = CAUSA_CONFIG_DEFAULTS;
    ccfg.discovery_alpha = 0.05f;
    ccfg.discovery_threshold = 0.25f;
    ccfg.max_iterations = 300;

    causa_scm_t* scm = causa_scm_new(arena, backend, &ccfg);
    int32_t laws = causa_memory_consolidate_causal(mem, scm, &ccfg);
    if (laws <= 0) {
        test_fail("consolidation produced %d laws\n", laws);
        return;
    }

    causa_tensor_t* state = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* goal = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* plan = causa_tensor_new(arena, backend, shape, 1, &opts);

    float* sd = (float*)state->data;
    sd[0] = 1.0f; sd[1] = NAN; sd[2] = NAN;

    // Two goals at once: x1 should be 4 and x2 should be 9.
    float* gd = (float*)goal->data;
    gd[0] = NAN; gd[1] = 4.0f; gd[2] = 9.0f;

    int32_t count = 0;
    if (causa_memory_plan_intervention(mem, state, goal, plan, &count) != 0) {
        test_fail("the intervention could not be planned\n");
        return;
    }

    float* pd = (float*)plan->data;
    printf("  plan: change x0 by %f, x1 by %f, x2 by %f\n", pd[0], pd[1], pd[2]);

    // Both goals have to be reached, so the plan has to act on more than one
    // variable.
    if (count < 2) {
        test_fail("the plan acts on %d variables, expected at least 2\n", count);
        return;
    }

    // x1 follows from x0, so it is x0 that moves. x1 stands at
    // 1.5 * 1 + 0.5 = 2, and it has to reach 4, so x0 moves by
    // (4 - 2) / 1.5 = 1.333.
    float x1_after = 1.5f * (1.0f + pd[0]) + 0.5f;
    printf("  applying the plan puts x1 at %f and x2 at %f\n", x1_after, pd[2]);
    if (fabsf(x1_after - 4.0f) > 0.4f) {
        test_fail("the plan puts x1 at %f, expected about 4\n", x1_after);
        return;
    }

    // x2 has no cause, so it is set directly.
    if (fabsf(pd[2] - 9.0f) > 1.5f) {
        test_fail("the plan says x2 moves by %f, expected about 9\n", pd[2]);
        return;
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: intervention planning with several goals\n");
}

// A counterfactual kept in memory.
//
// The counterfactual has to be found again for the variable it answers, and
// not for the others. Were the query variable ignored, a question about one
// variable would be answered with another's result.
void test_counterfactuals_in_memory() {
    printf("Testing counterfactuals in memory...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(16 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    const int32_t D = 3;

    causa_memory_config_t mcfg = CAUSA_MEMORY_CONFIG_DEFAULTS;
    mcfg.capacity[CAUSA_MEMORY_WORKING] = 16;
    mcfg.capacity[CAUSA_MEMORY_EPISODIC] = 64;
    mcfg.capacity[CAUSA_MEMORY_SEMANTIC] = 16;
    mcfg.capacity[CAUSA_MEMORY_PROCEDURAL] = 16;
    mcfg.key_dim = D;
    mcfg.value_dim = D;

    causa_memory_t* mem = causa_memory_new(arena, backend, &mcfg);
    if (!mem) { test_fail("memory creation\n"); return; }

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    int64_t shape[] = {D};

    causa_tensor_t* context = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* result_x1 = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* result_x2 = causa_tensor_new(arena, backend, shape, 1, &opts);
    if (!context || !result_x1 || !result_x2) {
        test_fail("tensor creation\n");
        return;
    }

    ((float*)context->data)[0] = 2.0f;
    ((float*)context->data)[1] = 3.5f;
    ((float*)context->data)[2] = 0.0f;

    ((float*)result_x1->data)[0] = 8.0f;
    ((float*)result_x2->data)[0] = 15.75f;

    if (causa_memory_keep_counterfactual(mem, "x1", context, result_x1) < 0) {
        test_fail("the counterfactual was not kept\n");
        return;
    }
    if (causa_memory_keep_counterfactual(mem, "x2", context, result_x2) < 0) {
        test_fail("the counterfactual was not kept\n");
        return;
    }

    // Asking about one variable must give that variable's answer and not the
    // other's.
    causa_tensor_t* found[4];
    int32_t n = causa_memory_find_counterfactuals(mem, "x1", context, 4, found);
    if (n != 1) {
        test_fail("found %d counterfactuals for x1, expected 1\n", n);
        return;
    }
    if (fabsf(((float*)found[0]->data)[0] - 8.0f) > 1e-5f) {
        test_fail("the counterfactual for x1 is wrong\n");
        return;
    }

    n = causa_memory_find_counterfactuals(mem, "x2", context, 4, found);
    if (n != 1) {
        test_fail("found %d counterfactuals for x2, expected 1\n", n);
        return;
    }
    if (fabsf(((float*)found[0]->data)[0] - 15.75f) > 1e-5f) {
        test_fail("the counterfactual for x2 is wrong\n");
        return;
    }

    // A variable that was never asked about has no counterfactuals.
    n = causa_memory_find_counterfactuals(mem, "x3", context, 4, found);
    if (n != 0) {
        test_fail("found %d counterfactuals for x3, expected none\n", n);
        return;
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: counterfactuals in memory\n");
}

void test_tensor_views_and_dtypes() {
    printf("Testing views and element types...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(16 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;

    /* A slice starts at its own offset, so every kernel has to read from
       there rather than from the beginning of the buffer. */
    int64_t shape[] = {6};
    causa_tensor_t* t = causa_tensor_new(arena, backend, shape, 1, &opts);
    float* td = (float*)t->data;
    for (int i = 0; i < 6; i++) td[i] = (float)(i + 1);   /* 1 2 3 4 5 6 */

    int64_t starts[] = {2};
    int64_t ends[] = {5};
    int64_t steps[] = {1};
    causa_tensor_t* s = causa_tensor_slice(t, starts, ends, steps, 1);
    if (!s) { test_fail("the slice was not created\n"); return; }

    int32_t dims[] = {0};
    causa_tensor_t* sum = causa_tensor_reduce(s, CAUSA_REDUCE_SUM, dims, 1, true);
    if (!sum || fabsf(((const float*)sum->data)[0] - 12.0f) > 1e-4f) {
        test_fail("the sum of the slice is %f, expected 12\n",
               sum ? ((const float*)sum->data)[0] : -999.0);
        return;
    }

    /* A reversed slice counts backwards */
    int64_t rstarts[] = {5};
    int64_t rends[] = {0};
    int64_t rsteps[] = {-1};
    causa_tensor_t* rev = causa_tensor_slice(t, rstarts, rends, rsteps, 1);
    causa_tensor_t* rsum = rev ? causa_tensor_reduce(rev, CAUSA_REDUCE_SUM, dims, 1, true) : NULL;
    if (!rsum || fabsf(((const float*)rsum->data)[0] - 20.0f) > 1e-4f) {
        test_fail("the sum of the reversed slice is %f, expected 20\n",
               rsum ? ((const float*)rsum->data)[0] : -999.0);
        return;
    }

    /* A matrix multiplied by rows of itself */
    int64_t mshape[] = {3, 2};
    causa_tensor_t* m = causa_tensor_new(arena, backend, mshape, 2, &opts);
    float* md = (float*)m->data;
    for (int i = 0; i < 6; i++) md[i] = (float)(i + 1);

    int64_t rowstarts[] = {1, 0};
    int64_t rowends[] = {3, 2};
    int64_t rowsteps[] = {1, 1};
    causa_tensor_t* rows = causa_tensor_slice(m, rowstarts, rowends, rowsteps, 2);

    int64_t ishape[] = {2, 2};
    causa_tensor_t* ident = causa_tensor_new(arena, backend, ishape, 2, &opts);
    float* id = (float*)ident->data;
    id[0] = 1.0f; id[1] = 0.0f; id[2] = 0.0f; id[3] = 1.0f;

    causa_tensor_t* mm = causa_tensor_matmul(rows, ident);
    if (!mm) { test_fail("the matrix product failed\n"); return; }
    const float* mmd = (const float*)mm->data;
    if (mmd[0] != 3.0f || mmd[1] != 4.0f || mmd[2] != 5.0f || mmd[3] != 6.0f) {
        test_fail("the product of the sliced rows is %g %g %g %g, expected 3 4 5 6\n",
               mmd[0], mmd[1], mmd[2], mmd[3]);
        return;
    }

    /* A permute has to be a permutation */
    int64_t pshape[] = {2, 3};
    causa_tensor_t* p = causa_tensor_new(arena, backend, pshape, 2, &opts);
    int32_t bad_dims[] = {0, 0};
    if (causa_tensor_permute(p, bad_dims, 2) != NULL) {
        test_fail("a permute that repeats a dimension was accepted\n");
        return;
    }

    /* A dimension that is not there is an error, not a smaller reduction */
    int32_t far_dims[] = {7};
    if (causa_tensor_reduce(t, CAUSA_REDUCE_SUM, far_dims, 1, true) != NULL) {
        test_fail("a reduction over a dimension that is not there was accepted\n");
        return;
    }

    /* A view can be reshaped again */
    int64_t rshape[] = {2, 3};
    causa_tensor_t* r = causa_tensor_reshape(t, rshape, 2);
    if (!r || !causa_tensor_reshape(r, rshape, 2)) {
        test_fail("a reshaped tensor could not be reshaped again\n");
        return;
    }

    /* A batched product clears the whole output, not only the first
       matrix, so the other batches do not keep what the arena held. */
    for (int i = 0; i < 4000; i++) {
        int64_t junk_shape[] = {64};
        causa_tensor_t* junk = causa_tensor_new(arena, backend, junk_shape, 1, &opts);
        float* jd = (float*)junk->data;
        for (int k = 0; k < 64; k++) jd[k] = 12345.0f;
    }
    causa_arena_reset(arena);

    int64_t bshape[] = {3, 2, 2};
    causa_tensor_t* batch = causa_tensor_new(arena, backend, bshape, 3, &opts);
    float* bd = (float*)batch->data;
    for (int i = 0; i < 12; i++) bd[i] = (float)(i + 1);

    causa_tensor_t* ident2 = causa_tensor_new(arena, backend, ishape, 2, &opts);
    float* id2 = (float*)ident2->data;
    id2[0] = 1.0f; id2[1] = 0.0f; id2[2] = 0.0f; id2[3] = 1.0f;

    causa_tensor_t* bmm = causa_tensor_matmul(batch, ident2);
    if (!bmm) { test_fail("the batched product failed\n"); return; }
    const float* bmd = (const float*)bmm->data;
    if (bmd[4] != 5.0f || bmd[5] != 6.0f || bmd[6] != 7.0f || bmd[7] != 8.0f) {
        test_fail("the second batch of the product is %g %g %g %g, expected 5 6 7 8\n",
               bmd[4], bmd[5], bmd[6], bmd[7]);
        return;
    }

    /* Every element type can be filled and initialised. */
    int64_t four[] = {4};
    causa_tensor_options_t u8opts = opts;
    u8opts.dtype = CAUSA_DTYPE_U8;
    causa_tensor_t* bytes = causa_tensor_new(arena, backend, four, 1, &u8opts);
    causa_tensor_init(bytes, CAUSA_INIT_UNIFORM, 0.0, 10.0);
    const uint8_t* bd2 = (const uint8_t*)bytes->data;
    if (bd2[0] > 10) {
        test_fail("a byte tensor was initialised with %d, expected at most 10\n", bd2[0]);
        return;
    }

    causa_tensor_options_t f64opts = opts;
    f64opts.dtype = CAUSA_DTYPE_F64;
    causa_tensor_t* doubles = causa_tensor_new(arena, backend, four, 1, &f64opts);
    causa_tensor_fill(doubles, 2.5);
    const double* dd = (const double*)doubles->data;
    if (dd[0] != 2.5 || dd[3] != 2.5) {
        test_fail("a double tensor holds %g and %g, expected 2.5\n", dd[0], dd[3]);
        return;
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: views and element types\n");
}

void test_memory_prediction_nonlinear() {
    printf("Testing prediction on a relation that is not linear...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(16 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    const int32_t D = 2;

    causa_memory_config_t mcfg = CAUSA_MEMORY_CONFIG_DEFAULTS;
    mcfg.capacity[CAUSA_MEMORY_WORKING] = 16;
    mcfg.capacity[CAUSA_MEMORY_EPISODIC] = 1024;
    mcfg.capacity[CAUSA_MEMORY_SEMANTIC] = 64;
    mcfg.key_dim = D;
    mcfg.value_dim = D;
    mcfg.working_decay_rate = 0.0f;
    mcfg.episodic_decay_rate = 0.0f;

    causa_memory_t* mem = causa_memory_new(arena, backend, &mcfg);
    if (!mem) { test_fail("memory creation\n"); return; }

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    int64_t shape[] = {D};

    // The relation curves, so the part of it that a linear law cannot
    // express grows with x0. The prediction has to use the episodes that
    // resemble the observation, not all of them.
    unsigned int seed = 999u;
    for (int i = 0; i < 400; i++) {
        seed = seed * 1103515245u + 12345u;
        float r0 = ((float)((seed >> 16) & 0x7fff) / 16384.0f) - 1.0f;
        seed = seed * 1103515245u + 12345u;
        float r1 = ((float)((seed >> 16) & 0x7fff) / 16384.0f) - 1.0f;

        float x0 = r0;
        float x1 = 1.5f * x0 + 0.5f * x0 * x0 + 0.1f * r1;

        causa_tensor_t* key = causa_tensor_new(arena, backend, shape, 1, &opts);
        causa_tensor_t* value = causa_tensor_new(arena, backend, shape, 1, &opts);
        ((float*)key->data)[0] = x0;
        ((float*)key->data)[1] = x1;
        ((float*)value->data)[0] = x1;
        ((float*)value->data)[1] = 0.0f;

        causa_memory_write(mem, key, value, CAUSA_MEMORY_EPISODIC);
    }

    causa_config_t ccfg = CAUSA_CONFIG_DEFAULTS;
    ccfg.discovery_alpha = 0.05f;
    ccfg.discovery_threshold = 0.25f;
    ccfg.max_iterations = 300;

    causa_scm_t* scm = causa_scm_new(arena, backend, &ccfg);
    causa_memory_consolidate_causal(mem, scm, &ccfg);

    causa_tensor_t* observation = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* prediction = causa_tensor_new(arena, backend, shape, 1, &opts);
    ((float*)observation->data)[0] = 0.8f;
    ((float*)observation->data)[1] = NAN;

    causa_memory_predict(mem, observation, prediction);

    float got = ((float*)prediction->data)[1];
    float truth = 1.5f * 0.8f + 0.5f * 0.8f * 0.8f;

    printf("  predicted x1 = %f (expected %f)\n", got, truth);

    if (fabsf(got - truth) > 0.06f) {
        test_fail("the prediction is %f, expected %f\n", got, truth);
        return;
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: prediction on a relation that is not linear\n");
}

void test_serialization_refuses_bad_files() {
    printf("Testing serialization (bad files are refused)...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;

    int64_t shape[] = {2, 3};
    causa_tensor_t* t = causa_tensor_new(arena, backend, shape, 2, &opts);
    float* d = (float*)t->data;
    for (int i = 0; i < 6; i++) d[i] = (float)(i + 1);

    const char* path = "test_bad.bin";
    if (!causa_save_tensor(path, t)) {
        test_fail("save tensor\n");
        return;
    }

    // The file is read into memory so that it can be spoiled.
    FILE* f = fopen(path, "rb");
    if (!f) { test_fail("could not reopen the file\n"); return; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* bytes = (uint8_t*)malloc((size_t)size);
    if (!bytes) { fclose(f); test_fail("allocation\n"); return; }
    if (fread(bytes, 1, (size_t)size, f) != (size_t)size) {
        fclose(f); free(bytes); test_fail("could not read the file\n"); return;
    }
    fclose(f);

    // The byte count of the data is somewhere in the file. Spoiling it has
    // to make the reader refuse the file rather than copy more than the
    // tensor holds.
    bool refused = false;
    for (long i = 8; i < size - 8 && !refused; i++) {
        uint8_t saved = bytes[i];
        bytes[i] = 0xFF;
        bytes[i + 1] = 0xFF;
        bytes[i + 2] = 0xFF;
        bytes[i + 3] = 0xFF;

        FILE* out = fopen(path, "wb");
        if (out) {
            fwrite(bytes, 1, (size_t)size, out);
            fclose(out);
        }

        causa_arena_t* arena2 = causa_arena_new(1024 * 1024);
        causa_reader_t* r = causa_reader_open(path);
        causa_tensor_t* bad = r ? causa_read_tensor(r, arena2, backend) : NULL;
        if (r) {
            if (!bad) refused = true;
            causa_reader_close(r);
        }
        causa_arena_free(arena2);
        bytes[i] = saved;
    }

    // A file that stops in the middle has to be refused too.
    FILE* out = fopen(path, "wb");
    if (out) {
        fwrite(bytes, 1, (size_t)(size / 2), out);
        fclose(out);
    }
    {
        causa_arena_t* arena2 = causa_arena_new(1024 * 1024);
        causa_reader_t* r = causa_reader_open(path);
        causa_tensor_t* bad = r ? causa_read_tensor(r, arena2, backend) : NULL;
        if (r) {
            if (!bad) refused = true;
            causa_reader_close(r);
        }
        causa_arena_free(arena2);
    }

    free(bytes);
    remove(path);

    if (!refused) {
        test_fail("a spoiled file was accepted\n");
        return;
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: serialization (bad files are refused)\n");
}

void test_gradients() {
    printf("Testing gradients...\n"); fflush(stdout);
    causa_arena_t* arena = causa_arena_new(16 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;

    /* The gradient of a product has to reach an input that has not been
       given a gradient yet. It used to be dropped, because the backward
       only added to a gradient that was already there. */
    int64_t shape[] = {2};
    causa_tensor_t* x = causa_tensor_new(arena, backend, shape, 1, &opts);
    opts.requires_grad = true;
    causa_tensor_t* xr = causa_tensor_new(arena, backend, shape, 1, &opts);
    float* xd = (float*)xr->data;
    xd[0] = 3.0f; xd[1] = 4.0f;

    causa_tensor_t* sq = causa_tensor_mul(xr, xr);
    if (!sq) { test_fail("the product failed\n"); return; }

    causa_graph_t* g = causa_graph_new(arena, backend);
    causa_op_t* op = causa_graph_add_op(g, CAUSA_OP_MUL, "square",
        (causa_tensor_t*[]){xr, xr}, 2, (causa_tensor_t*[]){sq}, 1, NULL, NULL, NULL, 0);
    if (!op) { test_fail("the operation was not added\n"); return; }
    causa_graph_build(g);

    causa_tensor_t* grad = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_fill(grad, 1.0f);

    // The gradient belongs to the output of the operation
    sq->grad = grad;
    causa_graph_backward(g, NULL, 0);

    if (!xr->grad) {
        test_fail("the input has no gradient\n");
        return;
    }
    const float* gd = (const float*)xr->grad->data;
    printf("  the gradient of x*x is %f, %f (expected 6, 8)\n", gd[0], gd[1]);
    if (fabsf(gd[0] - 6.0f) > 1e-4f || fabsf(gd[1] - 8.0f) > 1e-4f) {
        test_fail("the gradient of a product is wrong\n");
        return;
    }

    /* A gradient that arrives by two paths is the sum of them */
    causa_arena_t* arena2 = causa_arena_new(16 * 1024 * 1024);
    opts.requires_grad = true;
    causa_tensor_t* y = causa_tensor_new(arena2, backend, shape, 1, &opts);
    float* yd = (float*)y->data;
    yd[0] = 3.0f; yd[1] = 4.0f;

    causa_tensor_t* sq2 = causa_tensor_mul(y, y);
    causa_tensor_t* total = causa_tensor_add(sq2, y);
    causa_tensor_t* grad2 = causa_tensor_new(arena2, backend, shape, 1, &opts);
    causa_tensor_fill(grad2, 1.0f);

    causa_graph_t* g2 = causa_graph_new(arena2, backend);
    causa_graph_add_op(g2, CAUSA_OP_MUL, "square",
        (causa_tensor_t*[]){y, y}, 2, (causa_tensor_t*[]){sq2}, 1, NULL, NULL, NULL, 0);
    causa_graph_add_op(g2, CAUSA_OP_ADD, "total",
        (causa_tensor_t*[]){sq2, y}, 2, (causa_tensor_t*[]){total}, 1, NULL, NULL, NULL, 0);
    causa_graph_build(g2);
    total->grad = grad2;
    causa_graph_backward(g2, NULL, 0);

    if (!y->grad) {
        test_fail("the input has no gradient\n");
        return;
    }
    const float* gd2 = (const float*)y->grad->data;
    printf("  the gradient of x*x + x is %f, %f (expected 7, 9)\n", gd2[0], gd2[1]);
    if (fabsf(gd2[0] - 7.0f) > 1e-4f || fabsf(gd2[1] - 9.0f) > 1e-4f) {
        test_fail("the gradients of two paths were not added\n");
        return;
    }

    causa_arena_free(arena2);
    causa_backend_free(backend);
    causa_arena_free(arena);
    printf("PASS: gradients\n");
}

// The kernels have to work on every element type.
//
// The arithmetic kernels were written for float32 only, and a tensor of
// another type came back with whatever the arena held in it. The values
// below are the ones a float32 tensor would give.
void test_dtype_kernels() {
    printf("Testing kernels on other element types...\n"); fflush(stdout);

    causa_arena_t* arena = causa_arena_new(4 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();
    bool failed = false;

    // The sum of four doubles is ten.
    {
        causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        opts.dtype = CAUSA_DTYPE_F64;
        int64_t shape[] = {4};
        causa_tensor_t* d = causa_tensor_new(arena, backend, shape, 1, &opts);
        if (!d) {
            test_fail("a double tensor could not be made\n");
            failed = true;
        } else {
            double* dd = (double*)d->data;
            for (int i = 0; i < 4; i++) dd[i] = (double)(i + 1);

            int32_t dims[] = {0};
            causa_tensor_t* r = causa_tensor_reduce(d, CAUSA_REDUCE_SUM, dims, 1, false);
            if (!r) {
                test_fail("the reduction of a double tensor failed\n");
                failed = true;
            } else {
                double v = ((const double*)r->data)[0];
                if (fabs(v - 10.0) > 1e-9) {
                    test_fail("the sum of the doubles is %g, expected 10\n", v);
                    failed = true;
                }
            }
        }
    }

    // A batched product of doubles gives the same answer as the float one.
    {
        causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        opts.dtype = CAUSA_DTYPE_F64;
        int64_t shape[] = {2, 2, 2};
        causa_tensor_t* a = causa_tensor_new(arena, backend, shape, 3, &opts);
        causa_tensor_t* b = causa_tensor_new(arena, backend, shape, 3, &opts);
        if (!a || !b) {
            test_fail("the double tensors could not be made\n");
            failed = true;
        } else {
            double* ad = (double*)a->data;
            double* bd = (double*)b->data;
            for (int i = 0; i < 8; i++) { ad[i] = 1.0; bd[i] = 2.0; }

            causa_tensor_t* c = causa_tensor_bmm(a, b);
            if (!c) {
                test_fail("the batched product of doubles failed\n");
                failed = true;
            } else {
                double v = ((const double*)c->data)[0];
                if (fabs(v - 4.0) > 1e-9) {
                    test_fail("the batched product of doubles is %g, expected 4\n", v);
                    failed = true;
                }
            }
        }
    }

    // The backend product has to agree with the general one. The backend
    // read the strides as if they were element counts, so it stepped four
    // times too far and gave a different answer.
    {
        causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        int64_t shape[] = {2, 2};
        causa_tensor_t* a = causa_tensor_new(arena, backend, shape, 2, &opts);
        causa_tensor_t* b = causa_tensor_new(arena, backend, shape, 2, &opts);
        float* af = (float*)a->data;
        float* bf = (float*)b->data;
        for (int i = 0; i < 4; i++) { af[i] = (float)(i + 1); bf[i] = (float)(i + 5); }

        causa_tensor_t* via_tensor = causa_tensor_matmul(a, b);
        causa_tensor_t* via_backend = causa_backend_matmul(backend, a, b);
        if (!via_tensor || !via_backend) {
            test_fail("the matrix products failed\n");
            failed = true;
        } else {
            const float* t = (const float*)via_tensor->data;
            const float* k = (const float*)via_backend->data;
            for (int i = 0; i < 4; i++) {
                if (fabsf(t[i] - k[i]) > 1e-5f) {
                    test_fail("the backend product is %g where the general one is %g\n", k[i], t[i]);
                    failed = true;
                    break;
                }
            }
        }
    }

    // A number of groups that is not a positive count would divide by
    // zero, so it is refused.
    {
        causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        int64_t shape[] = {1, 2, 4};
        int64_t wshape[] = {2, 2, 3};
        causa_tensor_t* input = causa_tensor_new(arena, backend, shape, 3, &opts);
        causa_tensor_t* w = causa_tensor_new(arena, backend, wshape, 3, &opts);
        if (!input || !w) {
            test_fail("the convolution tensors could not be made\n");
            failed = true;
        } else if (causa_tensor_conv1d(input, w, NULL, 1, 0, 1, 0)) {
            test_fail("a convolution with no groups was accepted\n");
            failed = true;
        }
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    if (failed) return;
    printf("PASS: kernels on other element types\n");
}

// Write a tensor file with a given shape and strides.
//
// The file is written by hand rather than through the writer, because the
// point of the test is a file that the writer would never make: strides
// that point outside the buffer, and a shape whose count of elements
// overflows.
static void write_raw_tensor_file(const char* path, uint32_t dtype,
                                   int64_t ndim, const int64_t* shape,
                                   const int64_t* strides, uint64_t nbytes) {
    FILE* f = fopen(path, "wb");
    if (!f) return;

    uint32_t magic = CAUSA_IO_MAGIC;
    uint32_t version = CAUSA_IO_VERSION;
    uint32_t section = CAUSA_IO_SECTION_TENSORS;
    uint32_t count = 1;
    uint32_t device = CAUSA_DEVICE_CPU;
    uint32_t layout = CAUSA_LAYOUT_CONTIGUOUS;
    uint8_t flag = 0;
    uint32_t name_len = 0;

    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&section, sizeof(section), 1, f);
    fwrite(&count, sizeof(count), 1, f);

    fwrite(&dtype, sizeof(dtype), 1, f);
    fwrite(&device, sizeof(device), 1, f);
    fwrite(&layout, sizeof(layout), 1, f);
    fwrite(&ndim, sizeof(ndim), 1, f);
    for (int32_t i = 0; i < ndim; i++) fwrite(&shape[i], sizeof(int64_t), 1, f);
    for (int32_t i = 0; i < ndim; i++) fwrite(&strides[i], sizeof(int64_t), 1, f);
    int64_t offset = 0;
    fwrite(&offset, sizeof(offset), 1, f);
    fwrite(&flag, sizeof(flag), 1, f);
    fwrite(&name_len, sizeof(name_len), 1, f);
    fwrite(&nbytes, sizeof(nbytes), 1, f);
    for (uint64_t i = 0; i < nbytes; i++) {
        uint8_t zero = 0;
        fwrite(&zero, sizeof(zero), 1, f);
    }
    fwrite(&flag, sizeof(flag), 1, f);

    fclose(f);
}

// A file that describes a tensor the reader cannot make has to be refused.
//
// The strides say where each element sits, so a file that claims a stride
// of a billion would make every kernel read outside the buffer. A shape
// whose count of elements overflows would ask for a buffer of four bytes
// and a tensor of four thousand million million elements.
void test_serialization_refuses_bad_geometry() {
    printf("Testing serialization (a file with a bad geometry is refused)...\n"); fflush(stdout);

    causa_backend_t* backend = causa_backend_cpu_new();
    const char* path = "test_geometry.bin";
    bool failed = false;

    // Strides that point outside the buffer.
    {
        int64_t shape[] = {4};
        int64_t strides[] = {1LL << 40};
        write_raw_tensor_file(path, CAUSA_DTYPE_F32, 1, shape, strides, 16);

        causa_arena_t* arena = causa_arena_new(1024 * 1024);
        causa_reader_t* r = causa_reader_open(path);
        causa_tensor_t* t = r ? causa_read_tensor(r, arena, backend) : NULL;
        if (t) {
            test_fail("a file with absurd strides was accepted\n");
            failed = true;
        }
        if (r) causa_reader_close(r);
        causa_arena_free(arena);
    }

    // A shape whose count of elements overflows.
    {
        int64_t shape[] = {(1LL << 62) + 1, 1};
        int64_t strides[] = {4, 4};
        write_raw_tensor_file(path, CAUSA_DTYPE_F32, 2, shape, strides, 4);

        causa_arena_t* arena = causa_arena_new(1024 * 1024);
        causa_reader_t* r = causa_reader_open(path);
        causa_tensor_t* t = r ? causa_read_tensor(r, arena, backend) : NULL;
        if (t) {
            test_fail("a file with an overflowing shape was accepted\n");
            failed = true;
        }
        if (r) causa_reader_close(r);
        causa_arena_free(arena);
    }

    // An ordinary tensor still loads, so the refusals are not simply the
    // reader refusing everything.
    {
        int64_t shape[] = {4};
        int64_t strides[] = {4};
        write_raw_tensor_file(path, CAUSA_DTYPE_F32, 1, shape, strides, 16);

        causa_arena_t* arena = causa_arena_new(1024 * 1024);
        causa_reader_t* r = causa_reader_open(path);
        causa_tensor_t* t = r ? causa_read_tensor(r, arena, backend) : NULL;
        if (!t) {
            test_fail("an ordinary file was refused\n");
            failed = true;
        }
        if (r) causa_reader_close(r);
        causa_arena_free(arena);
    }

    remove(path);
    causa_backend_free(backend);
    if (failed) return;
    printf("PASS: serialization (a file with a bad geometry is refused)\n");
}

// The counts that decide how much memory is set aside have to be checked.
void test_scm_and_memory_bounds() {
    printf("Testing the bounds on counts...\n"); fflush(stdout);

    causa_arena_t* arena = causa_arena_new(4 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();
    bool failed = false;

    // More interventions than the model has room for is refused. The count
    // decides how much room the surgery needs on the stack.
    {
        causa_scm_t* scm = causa_scm_new(arena, backend, NULL);
        if (!scm) {
            test_fail("the model could not be made\n");
            failed = true;
        } else {
            causa_tensor_t* targets[CAUSA_MAX_VARIABLES + 8];
            causa_tensor_t* values[CAUSA_MAX_VARIABLES + 8];
            for (int32_t i = 0; i < CAUSA_MAX_VARIABLES + 8; i++) {
                int64_t shape[] = {1};
                targets[i] = causa_tensor_new(arena, backend, shape, 1, NULL);
                values[i] = causa_tensor_new(arena, backend, shape, 1, NULL);
            }
            if (causa_scm_do_multi(scm, targets, values, CAUSA_MAX_VARIABLES + 1)) {
                test_fail("too many interventions were accepted\n");
                failed = true;
            }
        }
    }

    // More noise terms than the model has room for is refused as well.
    {
        causa_scm_t* scm = causa_scm_new(arena, backend, NULL);
        if (!scm) {
            test_fail("the model could not be made\n");
            failed = true;
        } else {
            int64_t shape[] = {1};
            for (int32_t i = 0; i < CAUSA_MAX_VARIABLES + 4; i++) {
                causa_tensor_t* n = causa_tensor_new(arena, backend, shape, 1, NULL);
                if (causa_scm_add_exogenous(scm, n) != 0) break;
            }
            if (scm->num_exogenous > CAUSA_MAX_VARIABLES) {
                test_fail("the model took %d noise terms, more than the %d it has room for\n",
                          scm->num_exogenous, CAUSA_MAX_VARIABLES);
                failed = true;
            }
        }
    }

    // Collecting into a matrix with too few rows is refused. A matrix with
    // too few rows would be written past its end.
    {
        causa_memory_config_t cfg = CAUSA_MEMORY_CONFIG_DEFAULTS;
        cfg.key_dim = 3;
        cfg.value_dim = 3;
        causa_memory_t* mem = causa_memory_new(arena, backend, &cfg);
        if (!mem) {
            test_fail("the memory could not be made\n");
            failed = true;
        } else {
            for (int i = 0; i < 5; i++) {
                float key[3] = {(float)i, 0.0f, 0.0f};
                float value[3] = {0.0f, 0.0f, 0.0f};
                causa_tensor_t* k = causa_tensor_new(arena, backend, (int64_t[]){3}, 1, NULL);
                causa_tensor_t* v = causa_tensor_new(arena, backend, (int64_t[]){3}, 1, NULL);
                memcpy(k->data, key, sizeof(key));
                memcpy(v->data, value, sizeof(value));
                causa_memory_write(mem, k, v, CAUSA_MEMORY_EPISODIC);
            }

            int64_t too_small[] = {2, 3};
            causa_tensor_t* out = causa_tensor_new(arena, backend, too_small, 2, NULL);
            if (causa_memory_collect(mem, CAUSA_MEMORY_EPISODIC, out) >= 0) {
                test_fail("a matrix with too few rows was accepted\n");
                failed = true;
            }

            int64_t big_enough[] = {5, 3};
            causa_tensor_t* good = causa_tensor_new(arena, backend, big_enough, 2, NULL);
            if (causa_memory_collect(mem, CAUSA_MEMORY_EPISODIC, good) != 5) {
                test_fail("the collection did not gather every observation\n");
                failed = true;
            }
        }
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    if (failed) return;
    printf("PASS: the bounds on counts\n");
}

// An intervention on a model that was built as a graph has to be carried
// out.
//
// The surgery that copies the model for an intervention looks for the
// tensors the operations read, and the graph is only told about the
// tensors an operation writes. The parents and the noise therefore have to
// be registered with the graph as well, or the surgery fails and the
// intervention is quietly dropped.
void test_intervention_on_a_graph_model() {
    printf("Testing an intervention on a model built as a graph...\n"); fflush(stdout);

    causa_arena_t* arena = causa_arena_new(4 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();
    bool failed = false;

    causa_scm_t* scm = causa_scm_new(arena, backend, NULL);
    if (!scm) {
        test_fail("the model could not be made\n");
        causa_backend_free(backend);
        causa_arena_free(arena);
        return;
    }

    int64_t shape[] = {1};
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;

    causa_tensor_t* x0 = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* x1 = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* u1 = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* value = causa_tensor_new(arena, backend, shape, 1, &opts);
    if (!x0 || !x1 || !u1 || !value) {
        test_fail("the tensors could not be made\n");
        failed = true;
    } else {
        ((float*)x0->data)[0] = 1.0f;
        ((float*)u1->data)[0] = 0.5f;
        ((float*)value->data)[0] = 5.0f;

        // x0 is an observation and u1 is the noise of x1.
        causa_scm_add_observed_variable(scm, x0);
        causa_scm_add_exogenous(scm, u1);

        // x1 = x0 + u1, written as one operation over the parent and the
        // noise.
        causa_tensor_t* parents[] = {x0};
        if (causa_scm_add_equation_op(scm, x1, parents, 1, u1, CAUSA_OP_ADD, NULL, 0) != 0) {
            test_fail("the equation could not be added\n");
            failed = true;
        } else {
            causa_scm_build(scm);

            causa_intervention_t* iv = causa_intervention_new(arena, x0, value);
            causa_scm_t* int_scm = iv ? causa_scm_do(scm, iv) : NULL;
            if (!int_scm) {
                test_fail("the intervention failed\n");
                failed = true;
            } else {
                causa_tensor_t* outputs[1] = {x1};
                causa_scm_evaluate(int_scm, outputs, 1);

                // The values belong to the intervened model. The model
                // that was intervened on keeps the values it had.
                causa_tensor_t* got = int_scm->variables[1].variable;
                float v = got ? ((const float*)got->data)[0] : 0.0f;
                // 5.0 for the intervention, plus 0.5 for the noise.
                if (fabsf(v - 5.5f) > 1e-4f) {
                    test_fail("the intervention gave x1 = %g, expected 5.5\n", v);
                    failed = true;
                }

                // The model that was intervened on is left as it was.
                float before = ((const float*)x1->data)[0];
                if (fabsf(before - 0.0f) > 1e-6f) {
                    test_fail("the intervention changed the original model: x1 = %g, expected 0\n", before);
                    failed = true;
                }
            }
        }
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    if (failed) return;
    printf("PASS: an intervention on a model built as a graph\n");
}

// The gradients of the operations that work on more than one element.
//
// The backward of a matrix product used to give nothing back unless both
// sides were plain matrices, so the gradient of a batched product was
// dropped without a word. The backward of a reduction used to give the
// gradient to every element of the reduced group, where it belongs only to
// the largest one.
//
// The gradient buffers are laid out in order, whatever the layout of the
// tensor they belong to, so a view of a larger tensor has to be handled by
// the position and not by the strides.
void test_backward_of_shapes() {
    printf("Testing the gradients of batched and reduced operations...\n"); fflush(stdout);

    causa_arena_t* arena = causa_arena_new(4 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();
    bool failed = false;

    // The gradient of a batched product is the product of the gradients.
    {
        causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        opts.dtype = CAUSA_DTYPE_F32;
        opts.device = CAUSA_DEVICE_CPU;
        opts.requires_grad = true;

        int64_t shape[] = {2, 2, 2};
        causa_tensor_t* a = causa_tensor_new(arena, backend, shape, 3, &opts);
        causa_tensor_t* b = causa_tensor_new(arena, backend, shape, 3, &opts);
        if (!a || !b) {
            test_fail("the tensors could not be made\n");
            failed = true;
        } else {
            float* ad = (float*)a->data;
            float* bd = (float*)b->data;
            for (int i = 0; i < 8; i++) { ad[i] = 1.0f; bd[i] = 1.0f; }

            causa_tensor_t* c = causa_tensor_matmul(a, b);
            if (!c) {
                test_fail("the batched product failed\n");
                failed = true;
            } else {
                causa_tensor_options_t sopts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
                sopts.dtype = CAUSA_DTYPE_F32;
                sopts.device = CAUSA_DEVICE_CPU;
                causa_tensor_t* seed = causa_tensor_new(arena, backend, shape, 3, &sopts);
                causa_tensor_fill(seed, 1.0f);

                causa_tensor_matmul_backward(a, b, seed);

                if (!a->grad || !b->grad) {
                    test_fail("the backward of a batched product gave no gradient\n");
                    failed = true;
                } else {
                    // Every entry of the gradient is the sum of two ones.
                    const float* ga = (const float*)a->grad->data;
                    const float* gb = (const float*)b->grad->data;
                    for (int i = 0; i < 8; i++) {
                        if (fabsf(ga[i] - 2.0f) > 1e-5f || fabsf(gb[i] - 2.0f) > 1e-5f) {
                            test_fail("the gradient of the batched product is %g and %g, expected 2 and 2\n",
                                      ga[i], gb[i]);
                            failed = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    // The gradient of the largest element belongs to that element alone.
    {
        causa_graph_t* graph = causa_graph_new(arena, backend);
        causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        opts.dtype = CAUSA_DTYPE_F32;
        opts.device = CAUSA_DEVICE_CPU;
        opts.requires_grad = true;

        causa_tensor_t* x = causa_graph_new_tensor(graph, (int64_t[]){4}, 1, &opts);
        float* xd = (float*)x->data;
        xd[0] = 1.0f; xd[1] = 5.0f; xd[2] = 2.0f; xd[3] = 0.0f;

        causa_reduce_ctx_t* rctx = (causa_reduce_ctx_t*)causa_arena_alloc(arena, sizeof(causa_reduce_ctx_t), 8);
        rctx->reduce_op = CAUSA_REDUCE_MAX;
        rctx->num_dims = 1;
        rctx->dims[0] = 0;
        rctx->keepdim = false;

        causa_tensor_t* out = causa_graph_new_tensor(graph, (int64_t[]){1}, 1, &opts);
        causa_graph_add_op(graph, CAUSA_OP_REDUCE, "max",
            (causa_tensor_t*[]){x}, 1, (causa_tensor_t*[]){out}, 1,
            NULL, NULL, rctx, sizeof(causa_reduce_ctx_t));

        causa_graph_build(graph);
        causa_graph_forward(graph);

        // The gradient of the output is the seed of the backward pass. It
        // is put in place of the output's own gradient buffer, the same way
        // the abduction step does it.
        causa_tensor_options_t sopts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        sopts.dtype = CAUSA_DTYPE_F32;
        sopts.device = CAUSA_DEVICE_CPU;
        causa_tensor_t* seed = causa_tensor_new(arena, backend, (int64_t[]){1}, 1, &sopts);
        causa_tensor_fill(seed, 1.0f);
        out->grad = seed;

        causa_graph_backward(graph, NULL, 0);

        const float* gx = x->grad ? (const float*)x->grad->data : NULL;
        if (!gx) {
            test_fail("the reduction gave no gradient\n");
            failed = true;
        } else {
            float sum = gx[0] + gx[1] + gx[2] + gx[3];
            if (fabsf(gx[1] - 1.0f) > 1e-5f || fabsf(sum - 1.0f) > 1e-5f) {
                test_fail("the gradient of the largest is [%g %g %g %g], expected it on the second one alone\n",
                          gx[0], gx[1], gx[2], gx[3]);
                failed = true;
            }
        }
    }

    // The gradient of a view has to land in the view and nowhere else. A
    // column of a matrix has strides that reach past its own elements, so
    // a kernel that used them would write past the end of the gradient.
    {
        causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        opts.dtype = CAUSA_DTYPE_F32;
        opts.device = CAUSA_DEVICE_CPU;

        int64_t shape[] = {2, 3};
        causa_tensor_t* t = causa_tensor_new(arena, backend, shape, 2, &opts);
        float* td = (float*)t->data;
        for (int i = 0; i < 6; i++) td[i] = (float)(i + 1);

        causa_tensor_t* column = causa_tensor_slice(t, (int64_t[]){0, 1}, (int64_t[]){2, 2}, NULL, 2);
        if (!column) {
            test_fail("the column could not be taken\n");
            failed = true;
        } else {
            column->requires_grad = true;

            causa_tensor_options_t o2 = CAUSA_TENSOR_OPTIONS_DEFAULTS;
            o2.dtype = CAUSA_DTYPE_F32;
            o2.device = CAUSA_DEVICE_CPU;
            causa_tensor_t* b = causa_tensor_new(arena, backend, (int64_t[]){2, 1}, 2, &o2);
            causa_tensor_fill(b, 1.0f);
            causa_tensor_t* seed = causa_tensor_new(arena, backend, (int64_t[]){2, 1}, 2, &o2);
            causa_tensor_fill(seed, 1.0f);

            causa_tensor_mul_backward(column, b, seed);

            if (!column->grad) {
                test_fail("the view gave no gradient\n");
                failed = true;
            } else {
                const float* gc = (const float*)column->grad->data;
                if (fabsf(gc[0] - 1.0f) > 1e-5f || fabsf(gc[1] - 1.0f) > 1e-5f) {
                    test_fail("the gradient of the view is [%g %g], expected [1 1]\n", gc[0], gc[1]);
                    failed = true;
                }
            }
        }
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    if (failed) return;
    printf("PASS: the gradients of batched and reduced operations\n");
}

// The guards against a shape that does not fit.
//
// Every one of these used to read past the end of something: a key that is
// narrower than the memory expects, a batch whose two halves do not have
// the same width, and data that is not a matrix of samples and variables.
void test_guards_against_bad_shapes() {
    printf("Testing the guards against a shape that does not fit...\n"); fflush(stdout);

    causa_arena_t* arena = causa_arena_new(4 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();
    bool failed = false;

    // A key that is narrower than the memory expects is not a key.
    {
        causa_memory_config_t cfg = CAUSA_MEMORY_CONFIG_DEFAULTS;
        cfg.key_dim = 4;
        cfg.value_dim = 4;
        causa_memory_t* mem = causa_memory_new(arena, backend, &cfg);
        if (!mem) {
            test_fail("the memory could not be made\n");
            failed = true;
        } else {
            causa_tensor_t* key = causa_tensor_new(arena, backend, (int64_t[]){4}, 1, NULL);
            causa_tensor_t* value = causa_tensor_new(arena, backend, (int64_t[]){4}, 1, NULL);
            float* kd = (float*)key->data;
            for (int i = 0; i < 3; i++) kd[i] = (float)(i + 1);
            // The last element is a not-a-number. A comparison against one
            // is never greater than anything, so a key that is read one
            // element too far would match this one on every element.
            kd[3] = NAN;
            causa_tensor_fill(value, 1.0f);
            causa_memory_write(mem, key, value, CAUSA_MEMORY_EPISODIC);

            // A key that is one element short. Reading it as a key of four
            // would read past its end.
            causa_tensor_t* narrow = causa_tensor_new(arena, backend, (int64_t[]){3}, 1, NULL);
            float* nd = (float*)narrow->data;
            nd[0] = 1.0f; nd[1] = 2.0f; nd[2] = 3.0f;

            if (causa_memory_read(mem, narrow, CAUSA_MEMORY_EPISODIC)) {
                test_fail("a key of the wrong width was accepted\n");
                failed = true;
            }
        }
    }

    // A batch whose two halves are not the same width is not applied. The
    // rule would read past the narrower half.
    {
        causa_plasticity_config_t pcfg = CAUSA_PLASTICITY_CONFIG_DEFAULTS;
        pcfg.type = CAUSA_PLASTICITY_HEBBIAN;
        causa_plasticity_rule_t* rule = causa_plasticity_rule_new(arena, &pcfg);
        if (!rule) {
            test_fail("the rule could not be made\n");
            failed = true;
        } else {
            causa_tensor_t* pre = causa_tensor_new(arena, backend, (int64_t[]){2, 3}, 2, NULL);
            causa_tensor_t* post = causa_tensor_new(arena, backend, (int64_t[]){2, 1}, 2, NULL);
            causa_tensor_t* weight = causa_tensor_new(arena, backend, (int64_t[]){3}, 1, NULL);
            causa_tensor_fill(pre, 1.0f);
            causa_tensor_fill(post, 1.0f);
            causa_tensor_fill(weight, 0.0f);

            causa_plasticity_apply_batch(rule, pre, post, weight);

            const float* wd = (const float*)weight->data;
            if (fabsf(wd[0]) > 1e-6f || fabsf(wd[1]) > 1e-6f || fabsf(wd[2]) > 1e-6f) {
                test_fail("a batch of the wrong width was applied: [%g %g %g]\n", wd[0], wd[1], wd[2]);
                failed = true;
            }
        }
    }

    // Data that is not a matrix of samples and variables is refused. A
    // tensor with no dimensions has no last dimension to read.
    {
        causa_tensor_t* flat = causa_tensor_new(arena, backend, (int64_t[]){1}, 0, NULL);
        if (!flat) {
            test_fail("the tensor could not be made\n");
            failed = true;
        } else if (causa_discover(flat, NULL, NULL, NULL)) {
            test_fail("data that is not a matrix was accepted\n");
            failed = true;
        }
    }

    // Asking for the parents of a variable without somewhere to put them
    // is refused.
    {
        causa_discovery_t* disc = causa_discovery_new(arena, 3);
        if (!disc) {
            test_fail("the discovery could not be made\n");
            failed = true;
        } else if (causa_get_parents(disc, 0, NULL, 4) != -1) {
            test_fail("a request for parents with nowhere to put them was accepted\n");
            failed = true;
        }
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    if (failed) return;
    printf("PASS: the guards against a shape that does not fit\n");
}

// The operations that used to say nothing when they should have said
// something.
//
// A copy between tensors of different sizes did nothing at all, the memory
// took a key of any width and stored it as one of its own, a plasticity
// rule changed the key the memory is looked up by, four of the ways of
// filling a tensor filled it with zeros, and two of the operations on
// tensors gave nothing back at all.
void test_the_operations_that_were_silent() {
    printf("Testing the operations that used to say nothing...\n"); fflush(stdout);

    causa_arena_t* arena = causa_arena_new(4 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();
    bool failed = false;

    // A copy between tensors of different sizes copies the part that both
    // of them have.
    {
        causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        opts.dtype = CAUSA_DTYPE_F32;
        opts.device = CAUSA_DEVICE_CPU;

        causa_tensor_t* src = causa_tensor_new(arena, backend, (int64_t[]){4}, 1, &opts);
        causa_tensor_t* dst = causa_tensor_new(arena, backend, (int64_t[]){2}, 1, &opts);
        float* sd = (float*)src->data;
        for (int i = 0; i < 4; i++) sd[i] = (float)(i + 1);

        causa_tensor_copy(dst, src);

        const float* dd = (const float*)dst->data;
        if (fabsf(dd[0] - 1.0f) > 1e-6f || fabsf(dd[1] - 2.0f) > 1e-6f) {
            test_fail("the copy gave [%g %g], expected [1 2]\n", dd[0], dd[1]);
            failed = true;
        }
    }

    // A copy into a tensor that is not laid out in order puts the values in
    // the right places.
    {
        causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        opts.dtype = CAUSA_DTYPE_F32;
        opts.device = CAUSA_DEVICE_CPU;

        causa_tensor_t* src = causa_tensor_new(arena, backend, (int64_t[]){2, 1}, 2, &opts);
        causa_tensor_t* dst = causa_tensor_new(arena, backend, (int64_t[]){2, 2}, 2, &opts);
        float* sd = (float*)src->data;
        sd[0] = 7.0f; sd[1] = 8.0f;

        // A column of the destination, which is not laid out in order.
        causa_tensor_t* column = causa_tensor_slice(dst, (int64_t[]){0, 0}, (int64_t[]){2, 1}, NULL, 2);
        if (!column) {
            test_fail("the column could not be taken\n");
            failed = true;
        } else {
            causa_tensor_copy(column, src);
            const float* dd = (const float*)dst->data;
            if (fabsf(dd[0] - 7.0f) > 1e-6f || fabsf(dd[2] - 8.0f) > 1e-6f ||
                fabsf(dd[1]) > 1e-6f || fabsf(dd[3]) > 1e-6f) {
                test_fail("the copy into the column gave [%g %g %g %g], expected [7 0 8 0]\n",
                          dd[0], dd[1], dd[2], dd[3]);
                failed = true;
            }
        }
    }

    // The ways of filling a tensor that were declared but not done.
    {
        causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        opts.dtype = CAUSA_DTYPE_F32;
        opts.device = CAUSA_DEVICE_CPU;

        // The identity.
        causa_tensor_t* eye = causa_tensor_new(arena, backend, (int64_t[]){3, 3}, 2, &opts);
        causa_tensor_init(eye, CAUSA_INIT_EYE, 0.0, 0.0);
        const float* ed = (const float*)eye->data;
        bool right = fabsf(ed[0] - 1.0f) < 1e-6f && fabsf(ed[4] - 1.0f) < 1e-6f &&
                     fabsf(ed[8] - 1.0f) < 1e-6f && fabsf(ed[1]) < 1e-6f &&
                     fabsf(ed[2]) < 1e-6f && fabsf(ed[3]) < 1e-6f &&
                     fabsf(ed[5]) < 1e-6f && fabsf(ed[6]) < 1e-6f && fabsf(ed[7]) < 1e-6f;
        if (!right) {
            test_fail("the identity was not made\n");
            failed = true;
        }

        // A matrix whose rows are at right angles to each other.
        causa_tensor_t* orth = causa_tensor_new(arena, backend, (int64_t[]){3, 3}, 2, &opts);
        causa_tensor_init(orth, CAUSA_INIT_ORTHOGONAL, 0.0, 0.0);
        const float* od = (const float*)orth->data;
        float worst = 0.0f;
        for (int r = 0; r < 3; r++) {
            for (int s = 0; s < 3; s++) {
                float dot = 0.0f;
                for (int c = 0; c < 3; c++) dot += od[r * 3 + c] * od[s * 3 + c];
                float want = (r == s) ? 1.0f : 0.0f;
                float off = fabsf(dot - want);
                if (off > worst) worst = off;
            }
        }
        if (worst > 1e-4f) {
            test_fail("the rows of the orthogonal matrix are not at right angles (off by %g)\n", worst);
            failed = true;
        }

        // The two ways of filling a matrix with numbers that come from the
        // shape of the matrix.
        causa_tensor_t* xavier = causa_tensor_new(arena, backend, (int64_t[]){4, 4}, 2, &opts);
        causa_tensor_init(xavier, CAUSA_INIT_XAVIER_NORMAL, 1.0, 0.0);
        const float* xd = (const float*)xavier->data;
        bool all_zero = true;
        for (int i = 0; i < 16; i++) {
            if (xd[i] != 0.0f) all_zero = false;
        }
        if (all_zero) {
            test_fail("the Xavier fill gave nothing but zeros\n");
            failed = true;
        }

        causa_tensor_t* kaiming = causa_tensor_new(arena, backend, (int64_t[]){4, 4}, 2, &opts);
        causa_tensor_init(kaiming, CAUSA_INIT_KAIMING_NORMAL, 1.0, 0.0);
        const float* kd = (const float*)kaiming->data;
        all_zero = true;
        for (int i = 0; i < 16; i++) {
            if (kd[i] != 0.0f) all_zero = false;
        }
        if (all_zero) {
            test_fail("the Kaiming fill gave nothing but zeros\n");
            failed = true;
        }
    }

    // The comparison of two tensors.
    {
        causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        opts.dtype = CAUSA_DTYPE_F32;
        opts.device = CAUSA_DEVICE_CPU;

        causa_tensor_t* a = causa_tensor_new(arena, backend, (int64_t[]){3}, 1, &opts);
        causa_tensor_t* b = causa_tensor_new(arena, backend, (int64_t[]){3}, 1, &opts);
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        ad[0] = 1.0f; ad[1] = 2.0f; ad[2] = 3.0f;
        bd[0] = 1.0f; bd[1] = 5.0f; bd[2] = 3.0f;

        causa_tensor_t* same = causa_tensor_compare(a, b, CAUSA_CMP_EQ);
        if (!same) {
            test_fail("the comparison gave nothing back\n");
            failed = true;
        } else {
            // The result of a comparison is a byte per element, so it is
            // read through the loader and not as floats.
            float s0 = causa_tensor_load_f32(same, 0);
            float s1 = causa_tensor_load_f32(same, 1);
            float s2 = causa_tensor_load_f32(same, 2);
            if (s0 != 1.0f || s1 != 0.0f || s2 != 1.0f) {
                test_fail("the comparison gave [%g %g %g], expected [1 0 1]\n", s0, s1, s2);
                failed = true;
            }
        }

        causa_tensor_t* bigger = causa_tensor_compare(a, b, CAUSA_CMP_LT);
        if (!bigger) {
            test_fail("the comparison gave nothing back\n");
            failed = true;
        } else {
            float s0 = causa_tensor_load_f32(bigger, 0);
            float s1 = causa_tensor_load_f32(bigger, 1);
            float s2 = causa_tensor_load_f32(bigger, 2);
            if (s0 != 0.0f || s1 != 1.0f || s2 != 0.0f) {
                test_fail("the comparison gave [%g %g %g], expected [0 1 0]\n", s0, s1, s2);
                failed = true;
            }
        }
    }

    // The change of element type.
    {
        causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        opts.dtype = CAUSA_DTYPE_F32;
        opts.device = CAUSA_DEVICE_CPU;

        causa_tensor_t* a = causa_tensor_new(arena, backend, (int64_t[]){2}, 1, &opts);
        float* ad = (float*)a->data;
        ad[0] = 1.5f; ad[1] = 2.5f;

        causa_tensor_t* wide = causa_tensor_to_dtype(a, CAUSA_DTYPE_F64);
        if (!wide) {
            test_fail("the change of type gave nothing back\n");
            failed = true;
        } else {
            const double* wd = (const double*)wide->data;
            if (fabs(wd[0] - 1.5) > 1e-9 || fabs(wd[1] - 2.5) > 1e-9) {
                test_fail("the change of type gave [%g %g], expected [1.5 2.5]\n", wd[0], wd[1]);
                failed = true;
            }
        }
    }

    // The memory refuses a key or a value that is not the right width, and
    // a plasticity rule does not change the key the memory is looked up by.
    {
        causa_memory_config_t cfg = CAUSA_MEMORY_CONFIG_DEFAULTS;
        cfg.key_dim = 3;
        cfg.value_dim = 3;
        causa_memory_t* mem = causa_memory_new(arena, backend, &cfg);
        if (!mem) {
            test_fail("the memory could not be made\n");
            failed = true;
        } else {
            causa_tensor_t* narrow = causa_tensor_new(arena, backend, (int64_t[]){2}, 1, NULL);
            causa_tensor_t* value = causa_tensor_new(arena, backend, (int64_t[]){3}, 1, NULL);
            causa_tensor_fill(narrow, 1.0f);
            causa_tensor_fill(value, 1.0f);

            if (causa_memory_write(mem, narrow, value, CAUSA_MEMORY_EPISODIC)) {
                test_fail("a key of the wrong width was stored\n");
                failed = true;
            }

            // A rule that would change the key if it were applied to it.
            causa_plasticity_config_t pcfg = CAUSA_PLASTICITY_CONFIG_DEFAULTS;
            pcfg.type = CAUSA_PLASTICITY_HEBBIAN;
            pcfg.learning_rate = 1.0f;
            causa_plasticity_rule_t* rule = causa_plasticity_rule_new(arena, &pcfg);
            causa_memory_set_plasticity(mem, CAUSA_MEMORY_EPISODIC, rule);

            causa_tensor_t* key = causa_tensor_new(arena, backend, (int64_t[]){3}, 1, NULL);
            float* kd = (float*)key->data;
            kd[0] = 1.0f; kd[1] = 2.0f; kd[2] = 3.0f;

            causa_memory_write(mem, key, value, CAUSA_MEMORY_EPISODIC);
            causa_memory_write(mem, key, value, CAUSA_MEMORY_EPISODIC);

            // The same key twice is one entry, not two. A rule that changed
            // the key would make the second write miss.
            if (causa_memory_occupied(mem, CAUSA_MEMORY_EPISODIC) != 1) {
                test_fail("the memory holds %d entries, expected one\n",
                          causa_memory_occupied(mem, CAUSA_MEMORY_EPISODIC));
                failed = true;
            }
        }
    }

    // The laws of a child are counted, however many there are.
    {
        causa_memory_config_t cfg = CAUSA_MEMORY_CONFIG_DEFAULTS;
        cfg.key_dim = 8;
        cfg.value_dim = 8;
        causa_memory_t* mem = causa_memory_new(arena, backend, &cfg);
        if (!mem) {
            test_fail("the memory could not be made\n");
            failed = true;
        } else {
            // Seventy laws of the same child, each with a different set of
            // parents. A count that stopped at the size of a list would say
            // that there were only sixty-four of them.
            const int32_t child = 7;
            const int32_t wanted = 70;

            for (int32_t n = 0; n < wanted; n++) {
                causa_tensor_t* mask = causa_tensor_new(arena, backend, (int64_t[]){8}, 1, NULL);
                causa_tensor_t* coeff = causa_tensor_new(arena, backend, (int64_t[]){8}, 1, NULL);
                float* md = (float*)mask->data;
                float* cd = (float*)coeff->data;
                for (int32_t bit = 0; bit < 7; bit++) {
                    bool on = (n & (1 << bit)) != 0;
                    md[bit] = on ? 1.0f : 0.0f;
                    cd[bit] = on ? (float)(bit + 1) : 0.0f;
                }
                md[child] = 0.0f;
                cd[child] = 0.0f;
                causa_memory_store_causal_law(mem, child, mask, coeff);
            }

            int32_t count = causa_memory_count_causal_laws(mem, child);
            if (count != wanted) {
                test_fail("the child was found to have %d laws, expected %d\n", count, wanted);
                failed = true;
            }
        }
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    if (failed) return;
    printf("PASS: the operations that used to say nothing\n");
}

// The lifetime of an intervened model.
//
// An intervention makes a copy of the model. The copy has to be its own in
// every way: the values it computes must not land in the model it came
// from, and a second intervention on it must keep the first.
//
// A memory whose capacity is more than it can hold is refused rather than
// asked for.
void test_the_lifetime_of_an_intervened_model() {
    printf("Testing the lifetime of an intervened model...\n"); fflush(stdout);

    causa_arena_t* arena = causa_arena_new(4 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();
    bool failed = false;

    // x0 is an observation and x1 = x0 + u1.
    causa_scm_t* scm = causa_scm_new(arena, backend, NULL);
    if (!scm) {
        test_fail("the model could not be made\n");
        failed = true;
    } else {
        causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        opts.dtype = CAUSA_DTYPE_F32;
        opts.device = CAUSA_DEVICE_CPU;

        int64_t shape[] = {1};
        causa_tensor_t* x0 = causa_tensor_new(arena, backend, shape, 1, &opts);
        causa_tensor_t* x1 = causa_tensor_new(arena, backend, shape, 1, &opts);
        causa_tensor_t* u1 = causa_tensor_new(arena, backend, shape, 1, &opts);
        causa_tensor_t* value = causa_tensor_new(arena, backend, shape, 1, &opts);
        if (!x0 || !x1 || !u1 || !value) {
            test_fail("the tensors could not be made\n");
            failed = true;
        } else {
            ((float*)x0->data)[0] = 1.0f;
            ((float*)u1->data)[0] = 0.5f;
            ((float*)value->data)[0] = 5.0f;

            causa_scm_add_observed_variable(scm, x0);
            causa_scm_add_exogenous(scm, u1);

            causa_tensor_t* parents[] = {x0};
            if (causa_scm_add_equation_op(scm, x1, parents, 1, u1, CAUSA_OP_ADD, NULL, 0) != 0) {
                test_fail("the equation could not be added\n");
                failed = true;
            } else {
                causa_scm_build(scm);

                causa_intervention_t* iv = causa_intervention_new(arena, x0, value);
                causa_scm_t* first = iv ? causa_scm_do(scm, iv) : NULL;
                if (!first) {
                    test_fail("the intervention failed\n");
                    failed = true;
                } else {
                    causa_scm_evaluate(first, NULL, 0);

                    float x1_after = ((const float*)first->variables[1].variable->data)[0];
                    if (fabsf(x1_after - 5.5f) > 1e-4f) {
                        test_fail("the intervention gave x1 = %g, expected 5.5\n", x1_after);
                        failed = true;
                    }

                    // A second intervention on the first one. The first
                    // intervention has to survive it.
                    causa_tensor_t* second_value = causa_tensor_new(arena, backend, shape, 1, &opts);
                    ((float*)second_value->data)[0] = 2.0f;
                    causa_intervention_t* iv2 = causa_intervention_new(arena, x1, second_value);
                    causa_scm_t* second = iv2 ? causa_scm_do(first, iv2) : NULL;
                    if (!second) {
                        test_fail("the second intervention failed\n");
                        failed = true;
                    } else {
                        causa_scm_evaluate(second, NULL, 0);

                        float x0_after = ((const float*)second->variables[0].variable->data)[0];
                        float x1_after2 = ((const float*)second->variables[1].variable->data)[0];
                        if (fabsf(x0_after - 5.0f) > 1e-4f) {
                            test_fail("the second intervention lost the first: x0 = %g, expected 5\n", x0_after);
                            failed = true;
                        }
                        if (fabsf(x1_after2 - 2.0f) > 1e-4f) {
                            test_fail("the second intervention gave x1 = %g, expected 2\n", x1_after2);
                            failed = true;
                        }

                        // The model the interventions started from keeps the
                        // values it had.
                        float x0_original = ((const float*)x0->data)[0];
                        float x1_original = ((const float*)x1->data)[0];
                        if (fabsf(x0_original - 1.0f) > 1e-6f || fabsf(x1_original - 0.0f) > 1e-6f) {
                            test_fail("the interventions changed the original model: x0 = %g, x1 = %g\n",
                                      x0_original, x1_original);
                            failed = true;
                        }
                    }
                }
            }
        }
    }

    // A capacity that is more than the memory can hold is refused.
    {
        causa_memory_config_t cfg = CAUSA_MEMORY_CONFIG_DEFAULTS;
        cfg.capacity[CAUSA_MEMORY_SEMANTIC] = (int64_t)1 << 40;
        causa_memory_t* mem = causa_memory_new(arena, backend, &cfg);
        if (mem) {
            test_fail("a capacity of a billion slots was accepted\n");
            failed = true;
        }
    }

    // A capacity that is not a count of slots at all is refused.
    {
        causa_memory_config_t cfg = CAUSA_MEMORY_CONFIG_DEFAULTS;
        cfg.capacity[CAUSA_MEMORY_WORKING] = -5;
        causa_memory_t* mem = causa_memory_new(arena, backend, &cfg);
        if (mem) {
            test_fail("a negative capacity was accepted\n");
            failed = true;
        }
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    if (failed) return;
    printf("PASS: the lifetime of an intervened model\n");
}

int main() {
    printf("=== Tensor Library Tests ===\n\n");
    fflush(stdout);

    test_arena();
    fflush(stdout);
    test_tensor_basic();
    fflush(stdout);
    test_tensor_operations();
    fflush(stdout);
    test_tensor_activations();
    fflush(stdout);
    test_tensor_views();
    fflush(stdout);
    test_graph();
    fflush(stdout);
    test_backward_pass();
    fflush(stdout);
    test_mul_backward();
    fflush(stdout);
    test_activation_backward();
    fflush(stdout);
    test_gradient_check();
    fflush(stdout);
    test_backend();
    fflush(stdout);
    test_kernels();
    fflush(stdout);
    test_causal_scm();
    fflush(stdout);
    test_causal_intervention();
    fflush(stdout);
    test_acyclicity_gradient();
    fflush(stdout);
    test_causal_discovery();
    fflush(stdout);
    test_causal_counterfactual();
    fflush(stdout);
    test_abduction();
    fflush(stdout);
    test_causal_memory_integration();
    fflush(stdout);
    test_memory_reasoning();
    fflush(stdout);
    test_learned_model_runs();
    fflush(stdout);
    test_memory_prediction_nonlinear();
    fflush(stdout);
    test_tensor_views_and_dtypes();
    fflush(stdout);
    test_gradients();
    fflush(stdout);
    test_procedural_memory();
    fflush(stdout);
    test_intervention_planning();
    fflush(stdout);
    test_intervention_planning_several_goals();
    fflush(stdout);
    test_counterfactuals_in_memory();
    fflush(stdout);
    fflush(stdout);
    test_serialization_tensor();
    fflush(stdout);
    test_serialization_refuses_bad_files();
    fflush(stdout);
    test_serialization_memory();
    fflush(stdout);
    test_serialization_scm();
    fflush(stdout);
    test_dtype_kernels();
    fflush(stdout);
    test_serialization_refuses_bad_geometry();
    fflush(stdout);
    test_scm_and_memory_bounds();
    fflush(stdout);
    test_intervention_on_a_graph_model();
    fflush(stdout);
    test_backward_of_shapes();
    fflush(stdout);
    test_guards_against_bad_shapes();
    fflush(stdout);
    test_the_operations_that_were_silent();
    fflush(stdout);
    test_the_lifetime_of_an_intervened_model();
    fflush(stdout);

    printf("\n=== All tests completed ===\n");
    fflush(stdout);

    if (g_failures > 0) {
        printf("=== %d test(s) FAILED ===\n", g_failures);
        fflush(stdout);
        return 1;
    }

    return 0;
}