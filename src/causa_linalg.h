#ifndef CAUSA_LINALG_H
#define CAUSA_LINALG_H

// Small dense linear algebra.
//
// The systems here are small: a model has at most a few dozen variables, so
// a dense solve is the simplest thing that works and the fastest.

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <stdlib.h>

// Solve a small dense system by Gaussian elimination with partial pivoting.
//
// The matrix is n x n and the right-hand side is b. The solution is written
// into x. Returns false when the system has no unique solution.
//
// The matrix and the right-hand side are both changed, so a copy is made
// when they are still needed afterwards.
static inline bool causa_solve_dense(float* a, float* b, int32_t n, float* x) {
    for (int32_t col = 0; col < n; col++) {
        int32_t pivot = col;
        float best = fabsf(a[col * n + col]);
        for (int32_t r = col + 1; r < n; r++) {
            float v = fabsf(a[r * n + col]);
            if (v > best) { best = v; pivot = r; }
        }
        if (best < 1e-12f) return false;

        if (pivot != col) {
            for (int32_t c = 0; c < n; c++) {
                float tmp = a[col * n + c];
                a[col * n + c] = a[pivot * n + c];
                a[pivot * n + c] = tmp;
            }
            float tmp = b[col];
            b[col] = b[pivot];
            b[pivot] = tmp;
        }

        float diag = a[col * n + col];
        for (int32_t r = col + 1; r < n; r++) {
            float factor = a[r * n + col] / diag;
            if (factor == 0.0f) continue;
            for (int32_t c = col; c < n; c++) a[r * n + c] -= factor * a[col * n + c];
            b[r] -= factor * b[col];
        }
    }

    for (int32_t r = n - 1; r >= 0; r--) {
        float sum = b[r];
        for (int32_t c = r + 1; c < n; c++) sum -= a[r * n + c] * x[c];
        float diag = a[r * n + r];
        if (fabsf(diag) < 1e-12f) return false;
        x[r] = sum / diag;
    }

    return true;
}

// Solve a system in the least squares sense.
//
// The system is m equations in n unknowns. It may have more equations than
// unknowns or fewer, and the answer is the one that fits best: where there
// are more equations than unknowns the squared error is as small as it can
// be, and where there are fewer the change is as small as it can be.
//
// A small ridge term keeps the system solvable when the equations do not
// pin the unknowns down. That happens when two goals ask for changes that
// cannot both be made, and when a goal cannot be reached at all.
//
// The matrix is m rows of n columns and the right-hand side is m values.
// The solution is written into x.
//
// Returns false when the system cannot be solved.
static inline bool causa_solve_least_squares(
    const float* mtx,
    const float* rhs,
    int32_t m,
    int32_t n,
    float* x
) {
    if (!mtx || !rhs || !x || m <= 0 || n <= 0) return false;

    // The normal equations: (M' M) x = M' b.
    //
    // The matrix is small, so the products are worked out directly.
    float* normal = (float*)calloc((size_t)n * (size_t)n, sizeof(float));
    float* moved = (float*)calloc((size_t)n, sizeof(float));
    if (!normal || !moved) {
        free(normal);
        free(moved);
        return false;
    }

    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int32_t k = 0; k < m; k++) sum += mtx[k * n + i] * mtx[k * n + j];
            normal[i * n + j] = sum;
        }
        float sum = 0.0f;
        for (int32_t k = 0; k < m; k++) sum += mtx[k * n + i] * rhs[k];
        moved[i] = sum;
    }

    // The ridge term, which is small enough not to move the answer of a
    // system that is already pinned down.
    float scale = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        if (normal[i * n + i] > scale) scale = normal[i * n + i];
    }
    float ridge = 1e-6f * (scale + 1.0f);
    for (int32_t i = 0; i < n; i++) normal[i * n + i] += ridge;

    bool ok = causa_solve_dense(normal, moved, n, x);

    free(normal);
    free(moved);
    return ok;
}

#endif
