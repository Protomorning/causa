/*
 * A worked example of the causal memory system in C.
 *
 * Episodes become causal laws, the laws become a model, and the model can be
 * used to predict, to intervene, and to work out what to do.
 *
 * Build it with:
 *     cmake -S . -B build && cmake --build build
 * or against the static library the build script makes:
 *     cl /Iinclude examples\causal_memory.c build\causa.lib
 */

#include "causa.h"
#include <stdio.h>
#include <math.h>

#define DIM 3

/* A generator of observations from a chain X0 -> X1 -> X2. */
typedef struct { unsigned int seed; } generator_t;

static float generator_next(generator_t* g) {
    g->seed = g->seed * 1103515245u + 12345u;
    return ((float)((g->seed >> 16) & 0x7fff) / 16384.0f) - 1.0f;
}

int main(void) {
    causa_arena_t* arena = causa_arena_new(32 * 1024 * 1024);
    causa_backend_t* backend = causa_backend_cpu_new();

    causa_memory_config_t mcfg = CAUSA_MEMORY_CONFIG_DEFAULTS;
    mcfg.capacity[CAUSA_MEMORY_WORKING] = 16;
    mcfg.capacity[CAUSA_MEMORY_EPISODIC] = 1024;
    mcfg.capacity[CAUSA_MEMORY_SEMANTIC] = 64;
    mcfg.capacity[CAUSA_MEMORY_PROCEDURAL] = 64;
    mcfg.key_dim = DIM;
    mcfg.value_dim = DIM;
    mcfg.working_decay_rate = 0.0f;
    mcfg.episodic_decay_rate = 0.0f;

    causa_memory_t* memory = causa_memory_new(arena, backend, &mcfg);
    if (!memory) {
        fprintf(stderr, "could not create the memory\n");
        return 1;
    }

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    int64_t shape[] = {DIM};

    /* ---- 1. Experience -------------------------------------------------- */
    generator_t g = { 20240917u };

    for (int i = 0; i < 400; i++) {
        float x0 = generator_next(&g);
        float x1 = 1.5f * x0 + 0.5f + 0.25f * generator_next(&g);
        float x2 = 2.0f * x1 + 0.25f + 0.25f * generator_next(&g);

        causa_tensor_t* key = causa_tensor_new(arena, backend, shape, 1, &opts);
        causa_tensor_t* value = causa_tensor_new(arena, backend, shape, 1, &opts);

        float* k = (float*)key->data;
        float* v = (float*)value->data;
        k[0] = x0; k[1] = x1; k[2] = x2;
        v[0] = x1; v[1] = x2; v[2] = 0.0f;

        causa_memory_write(memory, key, value, CAUSA_MEMORY_EPISODIC);
    }

    printf("episodes recorded: %ld\n", (long)causa_memory_size(memory, CAUSA_MEMORY_EPISODIC));

    /* ---- 2. Learning ---------------------------------------------------- */
    causa_config_t ccfg = CAUSA_CONFIG_DEFAULTS;
    ccfg.discovery_alpha = 0.05f;
    ccfg.discovery_threshold = 0.25f;
    ccfg.max_iterations = 300;

    causa_scm_t* scm = causa_scm_new(arena, backend, &ccfg);
    int32_t laws = causa_memory_consolidate_causal(memory, scm, &ccfg);
    printf("causal laws learned: %d\n", laws);

    for (int32_t child = 0; child < DIM; child++) {
        causa_tensor_t* found[16];
        int32_t n = causa_memory_find_causal_laws(memory, child, found, 16);
        for (int32_t e = 0; e < n; e++) {
            const float* c = (const float*)found[e]->data;
            printf("  x%d =", child);
            for (int32_t j = 0; j < DIM; j++) {
                if (fabsf(c[j]) > 1e-3f) printf(" x%d * %.3f", j, c[j]);
            }
            printf("\n");
        }
    }

    /* ---- 3. Prediction -------------------------------------------------- */
    causa_tensor_t* partial = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* prediction = causa_tensor_new(arena, backend, shape, 1, &opts);

    float* p = (float*)partial->data;
    p[0] = 0.8f; p[1] = NAN; p[2] = NAN;

    causa_memory_predict(memory, partial, prediction);

    const float* pred = (const float*)prediction->data;
    printf("\nprediction from x0 = 0.8\n");
    printf("  x1 = %.3f   (expected %.3f)\n", pred[1], 1.5f * 0.8f + 0.5f);
    printf("  x2 = %.3f   (expected %.3f)\n", pred[2], 2.0f * (1.5f * 0.8f + 0.5f) + 0.25f);

    /* ---- 4. Planning ---------------------------------------------------- */
    causa_tensor_t* state = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* goal = causa_tensor_new(arena, backend, shape, 1, &opts);
    causa_tensor_t* plan = causa_tensor_new(arena, backend, shape, 1, &opts);

    float* s = (float*)state->data;
    float* go = (float*)goal->data;
    s[0] = 1.0f; s[1] = NAN; s[2] = NAN;
    go[0] = NAN; go[1] = NAN; go[2] = 10.0f;

    int32_t count = 0;
    causa_memory_plan_intervention(memory, state, goal, plan, &count);

    const float* changes = (const float*)plan->data;
    printf("\nplan to reach x2 = 10:\n");
    for (int32_t i = 0; i < DIM; i++) {
        if (changes[i] != 0.0f) printf("  change x%d by %+.3f\n", i, changes[i]);
    }

    /* ---- 5. Procedures -------------------------------------------------- */
    int32_t procedures = causa_memory_learn_procedures(memory, 0.5f);
    printf("\nprocedures learned: %d\n", procedures);

    causa_tensor_t* query = causa_tensor_new(arena, backend, shape, 1, &opts);
    float* q = (float*)query->data;
    q[0] = 0.5f; q[1] = 1.25f; q[2] = 2.75f;

    causa_tensor_t* triggers[8];
    causa_tensor_t* actions[8];
    int32_t found = causa_memory_find_procedures(memory, query, triggers, actions, 8);
    if (found > 0) {
        const float* t = (const float*)triggers[0]->data;
        printf("nearest procedure: [%.3f, %.3f, %.3f]\n", t[0], t[1], t[2]);
    }

    causa_backend_free(backend);
    causa_arena_free(arena);
    return 0;
}
