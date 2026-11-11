#include "causa_scm.h"
#include "causa_arena.h"
#include "causa_tensor.h"
#include "causa_graph.h"
#include "causa_memory.h"
#include "causa_backend.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ===== Helper Functions =====

static int scm_find_var_index(causa_scm_t* scm, causa_tensor_t* var) {
    for (int32_t i = 0; i < scm->num_variables; i++) {
        if (scm->variables[i].variable == var) return i;
    }
    return -1;
}

static void scm_compute_topo(causa_scm_t* scm) {
    // Simple Kahn's algorithm on the causal graph
    int32_t indegree[CAUSA_MAX_VARIABLES] = {0};
    int32_t adj[CAUSA_MAX_VARIABLES][CAUSA_MAX_VARIABLES] = {{0}};

    for (int32_t i = 0; i < scm->num_variables; i++) {
        for (int32_t p = 0; p < scm->variables[i].num_parents; p++) {
            int32_t parent_idx = scm_find_var_index(scm, scm->variables[i].parents[p]);
            if (parent_idx >= 0) {
                adj[parent_idx][i] = 1;
                indegree[i]++;
            }
        }
    }

    int32_t queue[CAUSA_MAX_VARIABLES];
    int32_t qhead = 0, qtail = 0;

    for (int32_t i = 0; i < scm->num_variables; i++) {
        if (indegree[i] == 0) queue[qtail++] = i;
    }

    scm->num_topo = 0;
    while (qhead < qtail) {
        int32_t v = queue[qhead++];
        scm->topo_order[scm->num_topo++] = v;
        for (int32_t i = 0; i < scm->num_variables; i++) {
            if (adj[v][i]) {
                if (--indegree[i] == 0) queue[qtail++] = i;
            }
        }
    }
}

// ===== SCM Construction =====

causa_scm_t* causa_scm_new(
    causa_arena_t* arena,
    causa_backend_t* backend,
    const causa_config_t* config
) {
    if (!arena || !backend) return NULL;

    causa_scm_t* scm = (causa_scm_t*)causa_arena_alloc(arena, sizeof(causa_scm_t), CAUSA_ALIGNOF(causa_scm_t));
    if (!scm) return NULL;

    scm->arena = arena;
    scm->backend = backend;
    scm->base_graph = NULL;
    scm->intervened_graph = NULL;
    scm->intervened_map = NULL;
    scm->num_intervened_map = 0;
    scm->num_variables = 0;
    scm->exogenous_noise = NULL;
    scm->num_exogenous = 0;
    scm->num_model_ops = -1;
    scm->num_topo = 0;
    {
        causa_config_t defaults = CAUSA_CONFIG_DEFAULTS;
        scm->config = defaults;
    }
    if (config) scm->config = *config;
    scm->built = false;

    for (int32_t i = 0; i < CAUSA_MAX_VARIABLES; i++) {
        scm->variables[i].variable = NULL;
        scm->variables[i].parents = NULL;
        scm->variables[i].num_parents = 0;
        scm->variables[i].noise = NULL;
        scm->variables[i].structural_fn = NULL;
        scm->variables[i].structural_bwd = NULL;
        scm->variables[i].is_intervened = false;
        scm->variables[i].intervention_value = NULL;
    }

    return scm;
}

void causa_scm_free(causa_scm_t* scm) {
    (void)scm;
}

int causa_scm_from_graph(
    causa_scm_t* scm,
    causa_graph_t* graph,
    causa_tensor_t** exogenous_noise,
    int32_t num_exogenous
) {
    if (!scm || !graph || !graph->built) return -1;

    scm->base_graph = graph;
    scm->exogenous_noise = exogenous_noise;
    scm->num_exogenous = num_exogenous;

    // Each op output becomes a causal variable
    // Parents = op inputs (simplified: assumes graph structure = causal structure)
    for (int32_t i = 0; i < graph->num_ops; i++) {
        causa_op_t* op = graph->ops[i];
        for (int32_t o = 0; o < op->num_outputs; o++) {
            if (scm->num_variables >= CAUSA_MAX_VARIABLES) break;

            causa_tensor_t* target = op->outputs[o];
            causa_scm_var_t* var = &scm->variables[scm->num_variables];

            var->variable = target;
            var->num_parents = op->num_inputs;
            var->parents = (causa_tensor_t**)causa_arena_alloc(scm->arena, sizeof(causa_tensor_t*) * op->num_inputs, CAUSA_ALIGNOF(causa_tensor_t*));
            for (int32_t p = 0; p < op->num_inputs; p++) {
                var->parents[p] = op->inputs[p];
            }

            // Assign exogenous noise (round-robin for simplicity)
            if (num_exogenous > 0) {
                var->noise = exogenous_noise[scm->num_variables % num_exogenous];
            }

            var->structural_fn = op->forward;
            var->structural_bwd = op->backward;
            var->is_intervened = false;
            var->intervention_value = NULL;

            scm->num_variables++;
        }
    }

    return causa_scm_build(scm);
}

int causa_scm_add_equation(
    causa_scm_t* scm,
    causa_tensor_t* target,
    causa_tensor_t** parents,
    int32_t num_parents,
    causa_tensor_t* noise,
    causa_op_forward_fn structural_fn,
    causa_op_backward_fn structural_bwd
) {
    if (!scm || scm->num_variables >= CAUSA_MAX_VARIABLES) return -1;

    causa_scm_var_t* var = &scm->variables[scm->num_variables];
    var->variable = target;
    var->num_parents = num_parents;
    var->parents = parents;
    var->noise = noise;
    var->structural_fn = structural_fn;
    var->structural_bwd = structural_bwd;
    var->is_intervened = false;
    var->intervention_value = NULL;

    scm->num_variables++;
    return causa_scm_build(scm);
}

int causa_scm_add_observed_variable(causa_scm_t* scm, causa_tensor_t* variable) {
    if (!scm || !variable) return -1;
    if (scm->num_variables >= CAUSA_MAX_VARIABLES) return -1;

    causa_scm_var_t* var = &scm->variables[scm->num_variables];
    var->variable = variable;
    var->num_parents = 0;
    var->parents = NULL;
    var->noise = NULL;
    var->structural_fn = NULL;
    var->structural_bwd = NULL;
    var->is_intervened = false;
    var->intervention_value = NULL;
    scm->num_variables++;

    scm->built = false;
    return causa_scm_build(scm);
}

int causa_scm_add_exogenous(causa_scm_t* scm, causa_tensor_t* noise) {
    if (!scm || !noise) return -1;

    for (int32_t i = 0; i < scm->num_exogenous; i++) {
        if (scm->exogenous_noise[i] == noise) return 0;
    }

    int32_t n = scm->num_exogenous + 1;
    causa_tensor_t** grown = (causa_tensor_t**)causa_arena_alloc(scm->arena,
        sizeof(causa_tensor_t*) * n, CAUSA_ALIGNOF(causa_tensor_t*));
    if (!grown) return -1;

    if (scm->num_exogenous > 0 && scm->exogenous_noise) {
        memcpy(grown, scm->exogenous_noise, (size_t)scm->num_exogenous * sizeof(causa_tensor_t*));
    }
    grown[scm->num_exogenous] = noise;
    scm->exogenous_noise = grown;
    scm->num_exogenous = n;

    return 0;
}

int causa_scm_build(causa_scm_t* scm) {
    if (!scm) return -1;

    scm_compute_topo(scm);

    // Build intervened graph (initially same as base)
    // Will be modified by do() operations
    scm->intervened_graph = scm->base_graph;

    // If the SCM came from a graph, make sure the graph is ordered so that
    // gradients flow through the structural equations.
    if (scm->base_graph && !scm->base_graph->built) {
        causa_graph_build(scm->base_graph);
    }

    scm->built = true;
    return 0;
}

int causa_scm_add_equation_op(
    causa_scm_t* scm,
    causa_tensor_t* target,
    causa_tensor_t** parents,
    int32_t num_parents,
    causa_tensor_t* noise,
    causa_op_type_t op_type,
    const void* op_ctx,
    size_t op_ctx_size
) {
    if (!scm || !target || !noise) return -1;
    if (num_parents < 0 || num_parents > CAUSA_MAX_OP_INPUTS) return -1;

    causa_graph_t* g = scm->base_graph;
    if (!g) {
        // The SCM needs a graph to hold the equation
        g = causa_graph_new(scm->arena, scm->backend);
        scm->base_graph = g;
    }

    // The op consumes the parents and the noise and writes the target.
    causa_tensor_t* inputs[CAUSA_MAX_OP_INPUTS];
    int32_t n = 0;
    for (int32_t i = 0; i < num_parents; i++) {
        inputs[n++] = parents[i];
    }
    inputs[n++] = noise;

    // The context has to outlive the op, so it is copied into the arena.
    void* ctx = NULL;
    if (op_ctx && op_ctx_size > 0) {
        ctx = causa_arena_alloc(scm->arena, op_ctx_size, 8);
        if (!ctx) return -1;
        memcpy(ctx, op_ctx, op_ctx_size);
    }

    causa_op_t* op = causa_graph_add_op(g, op_type, "equation",
        inputs, n,
        (causa_tensor_t*[]){target}, 1,
        NULL, NULL, ctx, op_ctx_size);

    if (!op) return -1;

    // The target is computed from the parents and the noise, so it carries a
    // gradient whenever one of them does. Without this the gradient would
    // stop at the equation.
    for (int32_t i = 0; i < n; i++) {
        if (inputs[i] && inputs[i]->requires_grad) {
            target->requires_grad = true;
            break;
        }
    }

    // Record the equation in the SCM's variable table.
    if (scm->num_variables >= CAUSA_MAX_VARIABLES) return -1;

    causa_scm_var_t* var = &scm->variables[scm->num_variables];
    var->variable = target;
    var->num_parents = num_parents;
    var->parents = parents;
    var->noise = noise;
    var->structural_fn = NULL;
    var->structural_bwd = NULL;
    var->is_intervened = false;
    var->intervention_value = NULL;
    scm->num_variables++;

    scm->built = false;
    return causa_scm_build(scm);
}

causa_tensor_t* causa_scm_add_reconstruction_loss(
    causa_scm_t* scm,
    causa_tensor_t** observed,
    int32_t num_observed
) {
    if (!scm || !observed || num_observed <= 0) return NULL;

    causa_graph_t* g = scm->base_graph;
    if (!g) return NULL;

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    opts.requires_grad = true;

    causa_tensor_t* total = NULL;

    // The operations added here are the reconstruction loss, not the model.
    // Their inputs are the observed values, which are not model tensors, so
    // they must not be copied into an intervened graph.
    if (scm->num_model_ops < 0) {
        scm->num_model_ops = g->num_ops;
    }

    for (int32_t i = 0; i < num_observed && i < scm->num_variables; i++) {
        causa_tensor_t* var = scm->variables[i].variable;
        causa_tensor_t* obs = observed[i];
        if (!var || !obs) continue;

        // The observed value has to be the same shape as the variable it
        // belongs to. The difference is taken element by element, so a
        // different shape would make the loss something other than the
        // error, and the abduction would then fit noise.
        if (obs->ndim != var->ndim) {
            return NULL;
        }
        for (int32_t d = 0; d < var->ndim; d++) {
            if (obs->shape[d] != var->shape[d]) return NULL;
        }

        causa_tensor_t* diff = causa_graph_new_tensor(g, var->shape, var->ndim, &opts);
        causa_graph_add_op(g, CAUSA_OP_SUB, "obs_diff",
            (causa_tensor_t*[]){var, obs}, 2,
            (causa_tensor_t*[]){diff}, 1, NULL, NULL, NULL, 0);

        causa_tensor_t* sq = causa_graph_new_tensor(g, var->shape, var->ndim, &opts);
        causa_graph_add_op(g, CAUSA_OP_MUL, "obs_diff_sq",
            (causa_tensor_t*[]){diff, diff}, 2,
            (causa_tensor_t*[]){sq}, 1, NULL, NULL, NULL, 0);

        // The reduction sums every element of the variable and keeps the
        // dimensions, so the result has one element per dimension.
        int64_t term_shape[CAUSA_MAX_DIMS];
        for (int32_t d = 0; d < var->ndim; d++) term_shape[d] = 1;

        causa_reduce_ctx_t* rctx = (causa_reduce_ctx_t*)causa_arena_alloc(scm->arena, sizeof(causa_reduce_ctx_t), 8);
        if (!rctx) return NULL;
        rctx->reduce_op = CAUSA_REDUCE_SUM;
        rctx->num_dims = var->ndim;
        for (int32_t d = 0; d < var->ndim; d++) rctx->dims[d] = d;
        rctx->keepdim = true;

        causa_tensor_t* term = causa_graph_new_tensor(g, term_shape, var->ndim, &opts);
        causa_graph_add_op(g, CAUSA_OP_REDUCE, "obs_term",
            (causa_tensor_t*[]){sq}, 1,
            (causa_tensor_t*[]){term}, 1,
            NULL, NULL, rctx, sizeof(causa_reduce_ctx_t));

        if (!total) {
            total = term;
        } else {
            causa_tensor_t* next = causa_graph_new_tensor(g, total->shape, total->ndim, &opts);
            causa_graph_add_op(g, CAUSA_OP_ADD, "obs_total",
                (causa_tensor_t*[]){total, term}, 2,
                (causa_tensor_t*[]){next}, 1, NULL, NULL, NULL, 0);
            total = next;
        }
    }

    if (!total) return NULL;

    // Adding the ops invalidated the graph, so it has to be reordered before
    // it runs. causa_graph_add_op clears the built flag, so the rebuild is
    // unconditional here.
    g->built = false;
    causa_graph_build(g);

    return total;
}

// Ancestral sampling from SCM
// Evaluate the SCM with the noise values currently stored.
//
// Nothing is resampled: the exogenous noise is taken as it is, which is what
// a counterfactual needs, because the noise has been fixed by the abduction
// step.
// Run the operations of a graph that belong to the model.
//
// The abduction step adds the reconstruction loss to the model's graph, and
// that loss refers to observed values rather than to model tensors. Running
// it when the model is evaluated would read values that are no longer
// current, so only the model's own operations are run.
static void scm_forward_model_ops(causa_graph_t* graph, int32_t num_model_ops) {
    if (!graph || !graph->built) return;

    if (num_model_ops < 0 || num_model_ops >= graph->num_ops) {
        causa_graph_forward(graph);
        return;
    }

    for (int32_t i = 0; i < graph->num_nodes; i++) {
        causa_op_t* op = graph->nodes[i]->op;
        bool is_model = false;
        for (int32_t k = 0; k < num_model_ops; k++) {
            if (graph->ops[k] == op) { is_model = true; break; }
        }
        if (!is_model) continue;
        if (op->forward) op->forward(op);
    }
}

causa_tensor_t* causa_scm_evaluate(
    causa_scm_t* scm,
    causa_tensor_t** outputs,
    int32_t num_outputs
) {
    if (!scm || !scm->built) return NULL;

    // The values are computed into the model's own variable tensors. The
    // outputs array is optional; when it is given, its first entry is
    // returned so that the caller can chain the call.
    causa_tensor_t* result = (outputs && num_outputs > 0) ? outputs[0] : NULL;

    // An intervened SCM built from a graph has its own copy of the graph, in
    // which the intervened variables take their constant values. Running it
    // gives the counterfactual values directly.
    if (scm->intervened_graph && scm->intervened_map) {
        causa_graph_forward(scm->intervened_graph);

        // Copy the values back onto the SCM's own tensors, so that the
        // caller sees them where it expects them.
        for (int32_t v = 0; v < scm->num_variables; v++) {
            causa_tensor_t* var = scm->variables[v].variable;
            if (!var) continue;

            for (int32_t k = 0; k < scm->num_intervened_map; k++) {
                // The map is indexed by the source graph's tensor order.
                if (scm->intervened_map[k] && scm->base_graph &&
                    k < scm->base_graph->num_tensors &&
                    scm->base_graph->tensors[k] == var) {
                    causa_tensor_copy(var, scm->intervened_map[k]);
                    break;
                }
            }
        }

        return result;
    }

    // A model whose equations are operations of a graph is run by running
    // the graph. The variables of such a model are the outputs of those
    // operations, so the values land where the caller expects them.
    //
    // Without this the model would be a table of numbers: the equations
    // would be there, but nothing would ever compute them.
    if (scm->base_graph && scm->base_graph->built && scm->base_graph->num_ops > 0) {
        scm_forward_model_ops(scm->base_graph, scm->num_model_ops);
        return result;
    }

    for (int32_t t = 0; t < scm->num_topo; t++) {
        int32_t idx = scm->topo_order[t];
        causa_scm_var_t* var = &scm->variables[idx];
        if (var->is_intervened && var->intervention_value) {
            // The intervened variable takes the constant value
            causa_tensor_copy(var->variable, var->intervention_value);
        } else if (var->structural_fn) {
            causa_op_t tmp_op = {0};
            tmp_op.inputs[0] = var->parents[0];
            tmp_op.num_inputs = var->num_parents;
            tmp_op.outputs[0] = var->variable;
            tmp_op.num_outputs = 1;
            tmp_op.forward = var->structural_fn;
            var->structural_fn(&tmp_op);
        }
    }

    return result;
}

// Sample from SCM (ancestral sampling)
//
// The exogenous noise is drawn afresh, then the model is evaluated with it.
causa_tensor_t* causa_scm_sample(
    causa_scm_t* scm,
    causa_tensor_t** outputs,
    int32_t num_outputs
) {
    if (!scm || !scm->built) return NULL;

    // Draw fresh exogenous noise
    for (int32_t i = 0; i < scm->num_exogenous; i++) {
        if (scm->exogenous_noise[i]) {
            causa_tensor_init(scm->exogenous_noise[i], CAUSA_INIT_NORMAL, 0.0, 1.0);
        }
    }

    // Draw noise for any variable whose noise is not one of the exogenous
    // tensors, so that it is not sampled twice.
    for (int32_t v = 0; v < scm->num_variables; v++) {
        causa_tensor_t* noise = scm->variables[v].noise;
        if (!noise) continue;

        bool is_exogenous = false;
        for (int32_t i = 0; i < scm->num_exogenous; i++) {
            if (scm->exogenous_noise[i] == noise) { is_exogenous = true; break; }
        }
        if (!is_exogenous) {
            causa_tensor_init(noise, CAUSA_INIT_NORMAL, 0.0, 1.0);
        }
    }

    return causa_scm_evaluate(scm, outputs, num_outputs);
}

// Solve a small dense system by Gaussian elimination with partial pivoting.
//
// The matrix is n x n and the right-hand side is b. The solution is written
// into x. Returns false when the system has no unique solution.
static bool causal_solve_dense(float* a, float* b, int32_t n, float* x) {
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

// Regress a child on its parents and read off the coefficients.
//
// The regression has an intercept, which is not part of a structural
// equation: the intercept belongs to the noise. A small ridge term keeps
// the fit stable when the parents are close to collinear, which they are
// whenever one of them is a descendant of another.
//
// The arithmetic is done on standardised data and the coefficients are put
// back on the original scale afterwards. Doing it on the raw data would
// lose the precision that decides whether a parent has an effect of its
// own when the variables are large.
//
// The coefficients are written to out_coefficients, one per parent.
// Returns false when the fit could not be made.
static bool causal_refit_coefficients(
    const float* data,
    int32_t N,
    int32_t D,
    int32_t child,
    const int32_t* parents,
    int32_t num_parents,
    const float* mean,
    const float* stddev,
    float* out_coefficients
) {
    if (!data || !parents || !out_coefficients || num_parents <= 0) return false;
    if (num_parents > CAUSA_MAX_PARENTS) return false;

    int32_t k = num_parents;
    int32_t n = k + 1;  // the intercept and the parents

    float* a = (float*)calloc((size_t)(n * n), sizeof(float));
    float* b = (float*)calloc((size_t)n, sizeof(float));
    float* x = (float*)calloc((size_t)n, sizeof(float));
    if (!a || !b || !x) {
        free(a); free(b); free(x);
        return false;
    }

    for (int32_t i = 0; i < N; i++) {
        float row[CAUSA_MAX_PARENTS + 1];
        row[0] = 1.0f;
        for (int32_t p = 0; p < k; p++) {
            int32_t parent = parents[p];
            row[p + 1] = (data[(int64_t)i * D + parent] - mean[parent]) / stddev[parent];
        }

        float y = (data[(int64_t)i * D + child] - mean[child]) / stddev[child];

        for (int32_t r = 0; r < n; r++) {
            b[r] += row[r] * y;
            for (int32_t c = 0; c < n; c++) a[r * n + c] += row[r] * row[c];
        }
    }

    // The ridge applies to the slopes only, so the intercept stays free.
    for (int32_t p = 1; p < n; p++) a[p * n + p] += 1e-3f;

    bool ok = causal_solve_dense(a, b, n, x);
    if (ok) {
        for (int32_t p = 0; p < k; p++) {
            // Back onto the scale of the data.
            out_coefficients[p] = x[p + 1] * (stddev[child] / stddev[parents[p]]);
        }
    }

    free(a); free(b); free(x);
    return ok;
}

// ===== Building a model from memory =====

// The law of one variable: which variables are its parents and what the
// coefficient of each one is.
typedef struct {
    int32_t num_parents;
    int32_t parents[CAUSA_MAX_PARENTS];
    float coefficients[CAUSA_MAX_PARENTS];
    bool found;
} causal_law_t;

// Collect the law of a variable from the laws in semantic memory.
//
// When several laws were recorded for the same variable the one with the
// most parents is taken, because it describes the variable in the most
// detail.
static bool causal_collect_law(
    causa_memory_t* mem,
    int32_t child,
    int32_t dim,
    causal_law_t* law
) {
    if (!mem || !law) return false;
    if (child < 0 || child >= dim) return false;

    causa_tensor_t* keys[16];
    causa_tensor_t* values[16];
    int32_t n = causa_memory_find_causal_law_entries(mem, child, keys, values, 16);
    if (n <= 0) return false;

    bool have = false;

    for (int32_t e = 0; e < n; e++) {
        const float* role = (const float*)keys[e]->data;
        const float* coeff = (const float*)values[e]->data;

        causal_law_t candidate;
        memset(&candidate, 0, sizeof(candidate));
        candidate.found = true;

        for (int64_t j = 0; j < dim && candidate.num_parents < CAUSA_MAX_PARENTS; j++) {
            // The role vector marks the child with 2 and the parents with 1.
            if (role[j] == 1.0f) {
                candidate.parents[candidate.num_parents] = (int32_t)j;
                candidate.coefficients[candidate.num_parents] = coeff[j];
                candidate.num_parents++;
            }
        }

        if (!have || candidate.num_parents > law->num_parents) {
            *law = candidate;
            have = true;
        }
    }

    return have;
}

int32_t causa_memory_build_scm(
    causa_memory_t* mem,
    causa_scm_t* scm,
    int32_t num_variables
) {
    if (!mem || !scm) return -1;
    if (num_variables <= 0 || num_variables > CAUSA_MAX_VARIABLES) return -1;

    const causa_memory_config_t* mcfg = causa_memory_config(mem);
    if (!mcfg) return -1;

    int64_t dim = mcfg->key_dim;
    if (dim <= 0 || dim > CAUSA_MAX_VARIABLES) return -1;

    causa_graph_t* g = scm->base_graph;
    if (!g) {
        g = causa_graph_new(scm->arena, scm->backend);
        if (!g) return -1;
        scm->base_graph = g;
    }

    causal_law_t* laws = (causal_law_t*)calloc((size_t)num_variables, sizeof(causal_law_t));
    causa_tensor_t** variables = (causa_tensor_t**)calloc((size_t)num_variables, sizeof(causa_tensor_t*));
    causa_tensor_t** noises = (causa_tensor_t**)calloc((size_t)num_variables, sizeof(causa_tensor_t*));
    if (!laws || !variables || !noises) {
        free(laws); free(variables); free(noises);
        return -1;
    }

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;

    // The intermediates of an equation carry a gradient, because the noise
    // they depend on is inferred by optimisation. Without this the gradient
    // would stop at the first operation of the equation.
    causa_tensor_options_t gopts = opts;
    gopts.requires_grad = true;

    int64_t shape[] = {1};
    int32_t equations = 0;

    for (int32_t c = 0; c < num_variables; c++) {
        causal_collect_law(mem, c, (int32_t)dim, &laws[c]);
    }

    // The variable tensors are made first, because an equation needs its
    // parents to exist before it can refer to them.
    for (int32_t c = 0; c < num_variables; c++) {
        variables[c] = causa_graph_new_tensor(g, shape, 1, laws[c].found ? &gopts : &opts);
        if (!variables[c]) {
            free(laws); free(variables); free(noises);
            return -1;
        }

        if (laws[c].found) {
            // The noise carries a gradient, because the abduction step
            // infers it by optimisation.
            noises[c] = causa_graph_new_tensor(g, shape, 1, &gopts);
            if (!noises[c]) {
                free(laws); free(variables); free(noises);
                return -1;
            }
            causa_tensor_fill(noises[c], 0.0f);
            causa_scm_add_exogenous(scm, noises[c]);
        }
    }

    // The variables are registered in index order, because the reconstruction
    // loss pairs them with the observed values by position.
    for (int32_t c = 0; c < num_variables; c++) {
        if (laws[c].found && laws[c].num_parents > 0) {
            // The SCM keeps the pointer to the parent list, so it has to
            // outlive this function.
            causa_tensor_t** parents = (causa_tensor_t**)causa_arena_alloc(
                scm->arena,
                sizeof(causa_tensor_t*) * laws[c].num_parents,
                CAUSA_ALIGNOF(causa_tensor_t*));
            if (!parents) {
                free(laws); free(variables); free(noises);
                return -1;
            }
            for (int32_t p = 0; p < laws[c].num_parents; p++) {
                parents[p] = variables[laws[c].parents[p]];
            }

            causa_scm_add_equation(scm, variables[c], parents, laws[c].num_parents,
                                    noises[c], NULL, NULL);
            equations++;
        } else {
            // A variable without a law is an exogenous input: its value is
            // fixed by the observation.
            causa_scm_add_observed_variable(scm, variables[c]);
        }
    }

    // The equations become graph operations, so the model is differentiable.
    // The graph orders them itself, so they can be added in any order.
    for (int32_t c = 0; c < num_variables; c++) {
        if (!laws[c].found) continue;

        if (laws[c].num_parents == 0) {
            // The variable is the noise itself.
            causa_graph_add_op(g, CAUSA_OP_COPY, "equation",
                (causa_tensor_t*[]){noises[c]}, 1,
                (causa_tensor_t*[]){variables[c]}, 1, NULL, NULL, NULL, 0);
            continue;
        }

        causa_tensor_t* accumulator = NULL;

        for (int32_t p = 0; p < laws[c].num_parents; p++) {
            causa_tensor_t* coefficient = causa_graph_new_tensor(g, shape, 1, &opts);
            if (!coefficient) {
                free(laws); free(variables); free(noises);
                return -1;
            }
            causa_tensor_fill(coefficient, laws[c].coefficients[p]);

            causa_tensor_t* product = causa_graph_new_tensor(g, shape, 1, &gopts);
            if (!product) {
                free(laws); free(variables); free(noises);
                return -1;
            }

            causa_graph_add_op(g, CAUSA_OP_MUL, "coefficient*parent",
                (causa_tensor_t*[]){coefficient, variables[laws[c].parents[p]]}, 2,
                (causa_tensor_t*[]){product}, 1, NULL, NULL, NULL, 0);

            if (!accumulator) {
                accumulator = product;
                continue;
            }

            causa_tensor_t* sum = causa_graph_new_tensor(g, shape, 1, &gopts);
            if (!sum) {
                free(laws); free(variables); free(noises);
                return -1;
            }
            causa_graph_add_op(g, CAUSA_OP_ADD, "sum",
                (causa_tensor_t*[]){accumulator, product}, 2,
                (causa_tensor_t*[]){sum}, 1, NULL, NULL, NULL, 0);
            accumulator = sum;
        }

        // The noise is added last, with a coefficient of one.
        causa_tensor_t* result = causa_graph_new_tensor(g, shape, 1, &gopts);
        if (!result) {
            free(laws); free(variables); free(noises);
            return -1;
        }
        causa_graph_add_op(g, CAUSA_OP_ADD, "equation",
            (causa_tensor_t*[]){accumulator, noises[c]}, 2,
            (causa_tensor_t*[]){result}, 1, NULL, NULL, NULL, 0);

        // The result is the variable itself.
        causa_graph_add_op(g, CAUSA_OP_COPY, "variable",
            (causa_tensor_t*[]){result}, 1,
            (causa_tensor_t*[]){variables[c]}, 1, NULL, NULL, NULL, 0);
    }

    free(laws);
    free(variables);
    free(noises);

    scm->built = false;
    causa_scm_build(scm);

    return equations;
}

// ===== Intervention: do(X = x) =====

causa_intervention_t* causa_intervention_new(
    causa_arena_t* arena,
    causa_tensor_t* target,
    causa_tensor_t* value
) {
    if (!arena || !target || !value) return NULL;

    causa_intervention_t* iv = (causa_intervention_t*)causa_arena_alloc(arena, sizeof(causa_intervention_t), CAUSA_ALIGNOF(causa_intervention_t));
    if (!iv) return NULL;

    iv->target = target;
    iv->value = value;
    iv->original_scm = NULL;
    iv->intervened_scm = NULL;

    return iv;
}

// Build a copy of the SCM's graph in which the intervened variables take
// constant values instead of being produced by their structural equations.
//
// This is the graph form of do(X = x): the variable's equation is replaced
// by a copy of the constant, and everything downstream of it then sees the
// intervened value.
//
// The map from the source graph's tensors to the copy's tensors is returned
// so the caller can read the values back.
static causa_graph_t* scm_build_intervened_graph(
    causa_scm_t* scm,
    causa_tensor_t** targets,
    causa_tensor_t** constants,
    int32_t num_interventions,
    causa_tensor_t*** out_map,
    int32_t* out_map_count
) {
    causa_graph_t* src = scm->base_graph;
    if (!src || !src->built || num_interventions <= 0) return NULL;

    causa_arena_t* arena = scm->arena;
    causa_backend_t* backend = scm->backend;

    causa_graph_t* g = causa_graph_new(arena, backend);

    int32_t n = src->num_tensors;
    causa_tensor_t** map = (causa_tensor_t**)causa_arena_alloc(arena, sizeof(causa_tensor_t*) * (n ? n : 1), CAUSA_ALIGNOF(causa_tensor_t*));

    // One replacement tensor per source tensor
    for (int32_t i = 0; i < n; i++) {
        causa_tensor_t* s = src->tensors[i];
        causa_tensor_options_t o = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        o.dtype = s->dtype;
        o.device = s->device;
        o.layout = s->layout;
        o.requires_grad = s->requires_grad;
        map[i] = causa_tensor_new(arena, backend, s->shape, s->ndim, &o);
        if (!map[i]) return NULL;

        // Carry the value over. Constants such as a coefficient live in the
        // source graph and are not produced by any operation, so without this
        // the copy would start from whatever the arena held.
        causa_tensor_copy(map[i], s);
    }

    // Find which intervened variables are produced by an operation. A variable
    // with no operation of its own, such as an observed exogenous variable, has
    // the copy added rather than an operation replaced.
    bool produced[CAUSA_MAX_VARIABLES];
    for (int32_t iv = 0; iv < num_interventions; iv++) produced[iv] = false;

    int32_t model_ops = scm->num_model_ops >= 0 ? scm->num_model_ops : src->num_ops;
    if (model_ops > src->num_ops) model_ops = src->num_ops;

    for (int32_t i = 0; i < model_ops; i++) {
        causa_op_t* op = src->ops[i];
        for (int32_t j = 0; j < op->num_outputs; j++) {
            for (int32_t iv = 0; iv < num_interventions; iv++) {
                if (op->outputs[j] == targets[iv]) produced[iv] = true;
            }
        }
    }

    // Copy the operations, replacing the ones that produce an intervened
    // variable. Only the model's own operations are copied; the
    // reconstruction loss belongs to the abduction step and refers to
    // observed values that are not part of the model.
    int32_t ops_to_copy = scm->num_model_ops >= 0 ? scm->num_model_ops : src->num_ops;
    if (ops_to_copy > src->num_ops) ops_to_copy = src->num_ops;

    for (int32_t i = 0; i < ops_to_copy; i++) {
        causa_op_t* op = src->ops[i];

        causa_tensor_t* inputs[CAUSA_MAX_OP_INPUTS];
        for (int32_t j = 0; j < op->num_inputs && j < CAUSA_MAX_OP_INPUTS; j++) {
            inputs[j] = NULL;
            for (int32_t k = 0; k < n; k++) {
                if (src->tensors[k] == op->inputs[j]) { inputs[j] = map[k]; break; }
            }
            if (!inputs[j]) {
                // An input that is not a graph tensor cannot be copied.
                return NULL;
            }
        }

        causa_tensor_t* outputs[CAUSA_MAX_OP_OUTPUTS];
        int32_t replaced_by = -1;
        for (int32_t j = 0; j < op->num_outputs && j < CAUSA_MAX_OP_OUTPUTS; j++) {
            outputs[j] = NULL;
            for (int32_t k = 0; k < n; k++) {
                if (src->tensors[k] == op->outputs[j]) { outputs[j] = map[k]; break; }
            }
            for (int32_t iv = 0; iv < num_interventions; iv++) {
                if (op->outputs[j] == targets[iv]) { replaced_by = iv; break; }
            }
        }

        // The op context has to outlive the op, so it is copied.
        void* ctx = NULL;
        if (op->ctx && op->ctx_size > 0) {
            ctx = causa_arena_alloc(arena, op->ctx_size, 8);
            if (!ctx) return NULL;
            memcpy(ctx, op->ctx, op->ctx_size);
        }

        if (replaced_by >= 0 && op->num_outputs > 0) {
            // The intervened variable takes the constant value
            causa_graph_add_op(g, CAUSA_OP_COPY, "intervened",
                (causa_tensor_t*[]){constants[replaced_by]}, 1,
                (causa_tensor_t*[]){outputs[0]}, 1,
                NULL, NULL, NULL, 0);
        } else {
            causa_graph_add_op(g, op->type, op->name,
                inputs, op->num_inputs,
                outputs, op->num_outputs,
                op->forward, op->backward, ctx, op->ctx_size);
        }
    }

    // An intervened variable that no operation produced gets a copy added, so
    // that the consumers of it see the intervened value.
    for (int32_t iv = 0; iv < num_interventions; iv++) {
        if (produced[iv]) continue;

        for (int32_t k = 0; k < n; k++) {
            if (src->tensors[k] == targets[iv]) {
                causa_graph_add_op(g, CAUSA_OP_COPY, "intervened",
                    (causa_tensor_t*[]){constants[iv]}, 1,
                    (causa_tensor_t*[]){map[k]}, 1,
                    NULL, NULL, NULL, 0);
                break;
            }
        }
    }

    causa_graph_build(g);

    if (out_map) *out_map = map;
    if (out_map_count) *out_map_count = n;
    return g;
}

// Apply an intervention to a copy of the SCM.
static causa_scm_t* scm_do_internal(
    causa_scm_t* scm,
    causa_tensor_t** targets,
    causa_tensor_t** values,
    int32_t num_interventions
) {
    causa_scm_t* new_scm = causa_scm_new(scm->arena, scm->backend, NULL);
    if (!new_scm) return NULL;

    new_scm->num_variables = scm->num_variables;
    new_scm->num_exogenous = scm->num_exogenous;
    new_scm->exogenous_noise = scm->exogenous_noise;
    new_scm->base_graph = scm->base_graph;

    for (int32_t i = 0; i < scm->num_variables; i++) {
        new_scm->variables[i] = scm->variables[i];
    }

    for (int32_t iv = 0; iv < num_interventions; iv++) {
        for (int32_t i = 0; i < new_scm->num_variables; i++) {
            if (new_scm->variables[i].variable == targets[iv]) {
                new_scm->variables[i].is_intervened = true;
                new_scm->variables[i].intervention_value = values[iv];
                break;
            }
        }
    }

    memcpy(new_scm->topo_order, scm->topo_order, sizeof(int32_t) * scm->num_topo);
    new_scm->num_topo = scm->num_topo;

    // For a graph-based SCM, build a copy of the graph in which the
    // intervened variables take their constant values.
    if (scm->base_graph && scm->base_graph->built && num_interventions > 0) {
        causa_tensor_t** map = NULL;
        int32_t map_count = 0;

        causa_graph_t* ig = scm_build_intervened_graph(scm, targets, values,
            num_interventions, &map, &map_count);

        if (ig) {
            new_scm->intervened_graph = ig;
            new_scm->intervened_map = map;
            new_scm->num_intervened_map = map_count;
        }
    }

    new_scm->built = true;
    return new_scm;
}

causa_scm_t* causa_scm_do(
    causa_scm_t* scm,
    causa_intervention_t* intervention
) {
    if (!scm || !intervention || !scm->built) return NULL;

    causa_tensor_t* targets[1] = {intervention->target};
    causa_tensor_t* values[1] = {intervention->value};

    causa_scm_t* new_scm = scm_do_internal(scm, targets, values, 1);
    if (!new_scm) return NULL;

    intervention->original_scm = scm;
    intervention->intervened_scm = new_scm;
    return new_scm;
}

causa_scm_t* causa_scm_do_multi(
    causa_scm_t* scm,
    causa_tensor_t** targets,
    causa_tensor_t** values,
    int32_t num_interventions
) {
    if (!scm || !targets || !values || num_interventions <= 0 || !scm->built) return NULL;
    return scm_do_internal(scm, targets, values, num_interventions);
}

// ===== Counterfactual Reasoning =====

causa_counterfactual_t* causa_counterfactual_new(
    causa_arena_t* arena,
    causa_tensor_t* query,
    causa_tensor_t* intervention_var,
    causa_tensor_t* intervention_val,
    causa_tensor_t* evidence
) {
    if (!arena || !query || !intervention_var || !intervention_val || !evidence) return NULL;

    causa_counterfactual_t* cf = (causa_counterfactual_t*)causa_arena_alloc(arena, sizeof(causa_counterfactual_t), CAUSA_ALIGNOF(causa_counterfactual_t));
    if (!cf) return NULL;

    cf->query = query;
    cf->intervention_var = intervention_var;
    cf->intervention_val = intervention_val;
    cf->evidence = evidence;
    cf->evidence_vars = NULL;
    cf->num_evidence_vars = 0;
    cf->inferred_noise = NULL;
    cf->num_inferred_noise = 0;
    cf->result = NULL;
    cf->abduction_error = 0.0f;
    cf->abduction_steps = 0;
    cf->intervened_scm = NULL;
    cf->abduction_done = false;
    cf->action_done = false;
    cf->prediction_done = false;

    return cf;
}

void causa_counterfactual_set_evidence(
    causa_counterfactual_t* cf,
    causa_tensor_t** observed,
    int32_t count
) {
    if (!cf || !observed || count <= 0) return;
    cf->evidence_vars = observed;
    cf->num_evidence_vars = count;
}

// Step 1: Abduction - infer U from evidence
//
// The abduction solves for the exogenous noise that best explains the
// observed values: minimise || SCM(U) - evidence ||^2. When the SCM was
// built from a graph, the loss is added to that graph and the noise is found
// by gradient descent, so the inference is exact rather than approximate.
int causa_counterfactual_abduction(
    causa_counterfactual_t* cf,
    causa_scm_t* scm,
    float lr,
    int32_t steps
) {
    if (!cf || !scm || !cf->evidence) return -1;

    causa_graph_t* g = scm->base_graph;
    if (!g || !g->built) return -1;

    // The observed value for each variable
    if (!cf->evidence_vars || cf->num_evidence_vars <= 0) return -1;

    causa_tensor_t* loss = causa_scm_add_reconstruction_loss(scm, cf->evidence_vars, cf->num_evidence_vars);
    if (!loss) return -1;

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;

    // The seed for the loss gradient
    causa_tensor_t* seed = causa_graph_new_tensor(g, (int64_t[]){1, 1}, 2, &opts);
    if (!seed) return -1;
    causa_tensor_fill(seed, 1.0f);

    // The noise is found by gradient descent on the reconstruction error.
    //
    // The error is a sum of squares along a curved valley, so the curvature
    // along the valley is much smaller than across it. Plain descent is
    // limited by the larger curvature, which is why the iteration count has
    // to be generous: the error falls by a small factor per step.

    for (int32_t step = 0; step < steps; step++) {
        causa_graph_zero_grad(g);

        // A variable with no equation of its own is pinned to its observed
        // value, because that is the only information there is about it.
        for (int32_t i = 0; i < scm->num_variables && i < cf->num_evidence_vars; i++) {
        const causa_scm_var_t* var = &scm->variables[i];
            if (var->num_parents == 0 && !var->noise && var->variable && cf->evidence_vars[i]) {
                causa_tensor_copy(var->variable, cf->evidence_vars[i]);
            }
        }

        causa_graph_forward(g);

        // The loss gradient is the seed. Zeroing the graph clears the
        // contents of every gradient buffer, so the seed is refilled here.
        loss->grad = seed;
        causa_tensor_fill(seed, 1.0f);

        causa_graph_backward(g, NULL, 0);

        // Take a step on the exogenous noise
        for (int32_t i = 0; i < scm->num_exogenous; i++) {
            causa_tensor_t* noise = scm->exogenous_noise[i];
            if (!noise || !noise->grad) continue;

            float* nd = (float*)noise->data;
            float* gd = (float*)noise->grad->data;
            int64_t n = causa_tensor_numel(noise);

            for (int64_t j = 0; j < n; j++) {
                nd[j] -= lr * gd[j];
            }
        }
    }

    // Record the inferred noise
    cf->num_inferred_noise = scm->num_exogenous;
    cf->inferred_noise = (causa_tensor_t**)causa_arena_alloc(scm->arena, sizeof(causa_tensor_t*) * scm->num_exogenous, CAUSA_ALIGNOF(causa_tensor_t*));
    for (int32_t i = 0; i < scm->num_exogenous; i++) {
        cf->inferred_noise[i] = scm->exogenous_noise[i];
    }

    // How well the fit turned out. A caller that does not believe the
    // counterfactual can look here to see that the inference did not
    // settle.
    cf->abduction_error = loss && causa_tensor_numel(loss) > 0
        ? ((const float*)loss->data)[0]
        : 0.0f;
    cf->abduction_steps = steps;

    cf->abduction_done = true;
    return 0;
}

// Step 2: Action - apply intervention
int causa_counterfactual_action(
    causa_counterfactual_t* cf,
    causa_scm_t* scm
) {
    if (!cf || !scm || !cf->abduction_done) return -1;

    causa_intervention_t* iv = causa_intervention_new(scm->arena, cf->intervention_var, cf->intervention_val);
    causa_scm_t* int_scm = causa_scm_do(scm, iv);
    if (!int_scm) return -1;

    // The model the intervention made is kept, so that the prediction step
    // uses it rather than building a second one.
    cf->intervened_scm = int_scm;
    cf->action_done = true;
    return 0;
}

// Step 3: Prediction - run with inferred U
int causa_counterfactual_prediction(
    causa_counterfactual_t* cf,
    causa_scm_t* scm
) {
    if (!cf || !scm || !cf->action_done) return -1;

    // Set inferred noise into SCM
    for (int32_t i = 0; i < scm->num_exogenous; i++) {
        if (cf->inferred_noise[i] && scm->exogenous_noise[i]) {
            causa_tensor_copy(scm->exogenous_noise[i], cf->inferred_noise[i]);
        }
    }

    // The model the action step built is used here. Building a second one
    // would give the same answer, but it would do the work twice.
    causa_scm_t* int_scm = cf->intervened_scm;
    if (!int_scm) {
        causa_intervention_t* iv = causa_intervention_new(scm->arena, cf->intervention_var, cf->intervention_val);
        int_scm = causa_scm_do(scm, iv);
        if (!int_scm) return -1;
        cf->intervened_scm = int_scm;
    }

    // Forward pass
    causa_scm_evaluate(int_scm, &cf->query, 1);
    cf->result = cf->query;

    cf->prediction_done = true;
    return 0;
}

// Full counterfactual computation (all three steps)
causa_tensor_t* causa_counterfactual_compute(
    causa_scm_t* scm,
    causa_counterfactual_t* cf
) {
    if (!scm || !cf) return NULL;

    // Step 1: Abduction
    causa_config_t config = CAUSA_CONFIG_DEFAULTS;
    causa_counterfactual_abduction(cf, scm, config.counterfactual_lr, config.counterfactual_steps);

    // Step 2: Action
    causa_counterfactual_action(cf, scm);

    // Step 3: Prediction
    causa_counterfactual_prediction(cf, scm);

    return cf->result;
}

// ===== Counterfactual Gradients (Unrolled Abduction) =====

// Compute counterfactual with gradient tracking by unrolling abduction as computation graph
int causa_counterfactual_compute_with_grad(
    causa_scm_t* scm,
    causa_counterfactual_t* cf,
    causa_graph_t* grad_graph
) {
    if (!scm || !cf || !grad_graph) return -1;

    causa_config_t config = CAUSA_CONFIG_DEFAULTS;
    int32_t steps = config.counterfactual_steps;
    float lr = config.counterfactual_lr;

    if (steps <= 0) steps = 10;
    if (lr <= 0) lr = 0.01f;

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    opts.requires_grad = true;

    // Create optimizable noise variables (one per exogenous)
    cf->num_inferred_noise = scm->num_exogenous;
    cf->inferred_noise = (causa_tensor_t**)causa_arena_alloc(scm->arena, sizeof(causa_tensor_t*) * scm->num_exogenous, CAUSA_ALIGNOF(causa_tensor_t*));

    for (int32_t i = 0; i < scm->num_exogenous; i++) {
        int64_t shape[] = {causa_tensor_numel(scm->exogenous_noise[i])};
        cf->inferred_noise[i] = causa_graph_new_tensor(grad_graph, shape, 1, &opts);
        causa_tensor_init(cf->inferred_noise[i], CAUSA_INIT_NORMAL, 0.0, 1.0);
    }

    // Create the evidence tensor in the grad graph
    causa_tensor_t* evidence_node = causa_graph_new_tensor(grad_graph, cf->evidence->shape, cf->evidence->ndim, &opts);
    causa_tensor_copy(evidence_node, cf->evidence);

    // Unroll abduction gradient descent steps
    causa_tensor_t* current_noise[CAUSA_MAX_VARIABLES];
    for (int32_t i = 0; i < scm->num_exogenous; i++) {
        current_noise[i] = cf->inferred_noise[i];
    }

    for (int32_t step = 0; step < steps; step++) {
        // Set exogenous noise in SCM from current noise variables
        for (int32_t i = 0; i < scm->num_exogenous; i++) {
            if (scm->exogenous_noise[i] && current_noise[i]) {
                causa_tensor_copy(scm->exogenous_noise[i], current_noise[i]);
            }
        }

        // Set variable noise
        for (int32_t v = 0; v < scm->num_variables; v++) {
            if (scm->variables[v].noise) {
                int32_t ext_idx = v % scm->num_exogenous;
                if (current_noise[ext_idx]) {
                    causa_tensor_copy(scm->variables[v].noise, current_noise[ext_idx]);
                }
            }
        }

        // Forward pass through SCM
        causa_scm_sample(scm, NULL, 0);

        // Get prediction (query variable)
        causa_tensor_t* pred = cf->query;

        // Compute loss: ||pred - evidence||^2
        int64_t n = causa_tensor_numel(pred);
        if (n != causa_tensor_numel(evidence_node)) continue;

        // loss = sum((pred - evidence)^2)
        // In graph: pred - evidence, then square, then sum
        causa_tensor_t* diff = causa_graph_new_tensor(grad_graph, pred->shape, pred->ndim, &opts);
        causa_tensor_t* sq = causa_graph_new_tensor(grad_graph, pred->shape, pred->ndim, &opts);
        causa_tensor_t* loss = causa_graph_new_tensor(grad_graph, (int64_t[]){1}, 1, &opts);

        causa_graph_add_op(grad_graph, CAUSA_OP_SUB, "cf_diff",
            (causa_tensor_t*[]){pred, evidence_node}, 2,
            (causa_tensor_t*[]){diff}, 1, NULL, NULL, NULL, 0);

        causa_graph_add_op(grad_graph, CAUSA_OP_MUL, "cf_sq",
            (causa_tensor_t*[]){diff, diff}, 2,
            (causa_tensor_t*[]){sq}, 1, NULL, NULL, NULL, 0);

        // Sum reduction
        int32_t dims[] = {0};
        causa_graph_add_op(grad_graph, CAUSA_OP_REDUCE, "cf_loss",
            (causa_tensor_t*[]){sq}, 1,
            (causa_tensor_t*[]){loss}, 1, NULL, NULL, dims, sizeof(dims));

        // Backward pass through graph to get gradients wrt current_noise
        causa_graph_zero_grad(grad_graph);

        // Set loss gradient = 1
        causa_tensor_t* loss_grad = causa_graph_new_tensor(grad_graph, (int64_t[]){1}, 1, &opts);
        causa_tensor_fill(loss_grad, 1.0f);
        loss->grad = loss_grad;

        causa_graph_backward(grad_graph, NULL, 0);

        // Update noise variables: noise = noise - lr * grad
        for (int32_t i = 0; i < scm->num_exogenous; i++) {
            if (current_noise[i] && current_noise[i]->grad) {
                causa_tensor_t* new_noise = causa_graph_new_tensor(grad_graph, current_noise[i]->shape, current_noise[i]->ndim, &opts);
                causa_tensor_t* lr_tensor = causa_graph_new_tensor(grad_graph, (int64_t[]){1}, 1, &opts);
                causa_tensor_fill(lr_tensor, lr);

                causa_tensor_t* grad_scaled = causa_graph_new_tensor(grad_graph, current_noise[i]->shape, current_noise[i]->ndim, &opts);
                causa_graph_add_op(grad_graph, CAUSA_OP_MUL, "grad_scaled",
                    (causa_tensor_t*[]){current_noise[i]->grad, lr_tensor}, 2,
                    (causa_tensor_t*[]){grad_scaled}, 1, NULL, NULL, NULL, 0);

                causa_graph_add_op(grad_graph, CAUSA_OP_SUB, "noise_update",
                    (causa_tensor_t*[]){current_noise[i], grad_scaled}, 2,
                    (causa_tensor_t*[]){new_noise}, 1, NULL, NULL, NULL, 0);

                current_noise[i] = new_noise;
            }
        }
    }

    // After unrolled abduction, current_noise holds inferred noise
    cf->inferred_noise = current_noise;
    cf->abduction_done = true;

    // Step 2: Action - apply intervention (creates intervened SCM)
    causa_counterfactual_action(cf, scm);

    // Step 3: Prediction - run with inferred noise
    // Copy inferred noise to SCM
    for (int32_t i = 0; i < scm->num_exogenous; i++) {
        if (cf->inferred_noise[i] && scm->exogenous_noise[i]) {
            causa_tensor_copy(scm->exogenous_noise[i], cf->inferred_noise[i]);
        }
    }

    causa_intervention_t* iv = causa_intervention_new(scm->arena, cf->intervention_var, cf->intervention_val);
    causa_scm_t* int_scm = causa_scm_do(scm, iv);

    // Forward pass
    causa_scm_evaluate(int_scm, &cf->query, 1);
    cf->result = cf->query;

    cf->prediction_done = true;
    return 0;
}

// Gradient of counterfactual w.r.t. intervention value
causa_tensor_t* causa_counterfactual_grad_wrt_intervention(
    causa_scm_t* scm,
    causa_counterfactual_t* cf
) {
    if (!scm || !cf || !cf->intervention_val) return NULL;

    // Create a grad graph for the full counterfactual computation
    causa_arena_t* arena = causa_arena_new(1024 * 1024);
    causa_graph_t* grad_graph = causa_graph_new(arena, scm->backend);

    // Build counterfactual with gradient tracking
    causa_counterfactual_compute_with_grad(scm, cf, grad_graph);

    // Build the graph
    causa_graph_build(grad_graph);

    // Forward pass
    causa_graph_forward(grad_graph);

    // Set output gradient = 1
    if (cf->result && cf->result->grad) {
        causa_tensor_fill(cf->result->grad, 1.0f);
    }

    // Backward pass
    causa_graph_backward(grad_graph, NULL, 0);

    // Return gradient w.r.t intervention value
    if (cf->intervention_val && cf->intervention_val->grad) {
        causa_tensor_t* grad = causa_tensor_new(arena, scm->backend, cf->intervention_val->shape, cf->intervention_val->ndim,
            &(causa_tensor_options_t){.dtype=CAUSA_DTYPE_F32, .device=CAUSA_DEVICE_CPU, .requires_grad=false});
        causa_tensor_copy(grad, cf->intervention_val->grad);
        causa_graph_free(grad_graph);
        causa_arena_free(arena);
        return grad;
    }

    causa_graph_free(grad_graph);
    causa_arena_free(arena);
    return NULL;
}

// Counterfactual gradient: d Y_x(u) / d x
causa_tensor_t* causa_counterfactual_gradient(
    causa_scm_t* scm,
    causa_counterfactual_t* cf,
    causa_tensor_t* wrt
) {
    (void)wrt;
    // For now, compute gradient w.r.t. intervention value
    return causa_counterfactual_grad_wrt_intervention(scm, cf);
}

// ===== Causal Discovery (NOTEARS) =====

causa_discovery_t* causa_discovery_new(
    causa_arena_t* arena,
    int32_t num_variables
) {
    if (!arena || num_variables <= 0 || num_variables > CAUSA_MAX_VARIABLES) return NULL;

    causa_discovery_t* disc = (causa_discovery_t*)causa_arena_alloc(arena, sizeof(causa_discovery_t), CAUSA_ALIGNOF(causa_discovery_t));
    if (!disc) return NULL;

    memset(disc->adjacency, 0, sizeof(disc->adjacency));
    memset(disc->confidence, 0, sizeof(disc->confidence));
    memset(disc->weights, 0, sizeof(disc->weights));
    memset(disc->parents, -1, sizeof(disc->parents));
    memset(disc->num_parents, 0, sizeof(disc->num_parents));
    disc->num_variables = num_variables;

    return disc;
}

void causa_discovery_free(causa_discovery_t* discovery) {
    (void)discovery;
}

// Acyclicity constraint: h(W) = tr(exp(W . W)) - d
//
// The matrix exponential is built from matmul and add ops inside the
// computation graph, so the gradient of the constraint comes from the same
// reverse-mode autodiff as the rest of the library. Scaling-and-squaring is
// used for the exponential: A is divided by 2^n, a truncated Taylor series
// evaluates exp(A / 2^n), and the result is squared n times.
causa_discovery_t* causa_discover(
    causa_tensor_t* observational_data,
    causa_tensor_t* interventional_data,
    causa_tensor_t* intervention_targets,
    const causa_config_t* config
) {
    if (!observational_data) return NULL;

    causa_config_t cfg = CAUSA_CONFIG_DEFAULTS;
    if (config) cfg = *config;

    int32_t D = (int32_t)observational_data->shape[observational_data->ndim - 1]; // variables
    int32_t N = (int32_t)observational_data->shape[0]; // samples

    // The data has to be a matrix of samples and variables, and it has to
    // hold something. A sample count of zero would divide by zero when the
    // mean is worked out, and the whole fit would be a not-a-number.
    if (observational_data->ndim != 2) return NULL;
    if (D <= 0 || N <= 0) return NULL;
    if (D > CAUSA_MAX_VARIABLES) return NULL;

    causa_arena_t* arena = observational_data->arena;
    causa_discovery_t* disc = causa_discovery_new(arena, D);
    if (!disc) return NULL;

    // Optimization graph
    causa_arena_t* opt_arena = causa_arena_new(32 * 1024 * 1024);
    causa_backend_t* backend = observational_data->backend;
    causa_graph_t* opt_graph = causa_graph_new(opt_arena, backend);

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    opts.requires_grad = true;

    // The mean and the standard deviation of every variable.
    //
    // The weights that come out of the optimisation are not the
    // coefficients of the structural equations: the acyclicity constraint
    // and the sparsity penalty both pull them away from the fit they
    // started from. The coefficients are therefore estimated again
    // afterwards, by regressing each child on the parents that were
    // selected. The standard deviations are needed to compare a weight
    // with the threshold on a scale that is the same for every variable.
    float mean[CAUSA_MAX_VARIABLES];
    float stddev[CAUSA_MAX_VARIABLES];
    {
        const float* od = (const float*)observational_data->data;
        for (int32_t c = 0; c < D; c++) {
            double sum = 0.0;
            for (int32_t n = 0; n < N; n++) sum += (double)od[n * D + c];
            mean[c] = (float)(sum / (double)N);

            double variance = 0.0;
            for (int32_t n = 0; n < N; n++) {
                double diff = (double)od[n * D + c] - (double)mean[c];
                variance += diff * diff;
            }
            float s = (float)sqrt(variance / (double)N);
            stddev[c] = s > 1e-6f ? s : 1.0f;
        }
    }

    // The data is centred and scaled by one common factor before the
    // optimisation.
    //
    // Centring takes away the constant terms, so the fit is not pulled
    // away from the true coefficients by an intercept that the model does
    // not have. Scaling keeps the weights in a range the acyclicity
    // constraint can handle: the constraint compares the weights with each
    // other, and a weight of a thousand would make it explode. One common
    // factor is used rather than one per variable, because the relation
    // between the variables is what tells the structure apart, and
    // standardising each of them separately would hide it.
    float scale = 1.0f;
    {
        float total = 0.0f;
        for (int32_t c = 0; c < D; c++) total += stddev[c];
        float mean_std = total / (float)D;
        if (mean_std > 1e-6f) scale = mean_std;
    }

    // Weight matrix W [D, D]. W[child][parent] holds the coefficient of the
    // parent in the child's structural equation.
    int64_t w_shape[] = {D, D};
    causa_tensor_t* W = causa_graph_new_tensor(opt_graph, w_shape, 2, &opts);
    causa_tensor_fill(W, 0.0f);

    // The data, centred and scaled
    causa_tensor_t* X = causa_graph_new_tensor(opt_graph, observational_data->shape, observational_data->ndim, &opts);
    {
        const float* od = (const float*)observational_data->data;
        float* xd = (float*)X->data;
        for (int32_t n = 0; n < N; n++) {
            for (int32_t c = 0; c < D; c++) {
                xd[n * D + c] = (od[n * D + c] - mean[c]) / scale;
            }
        }
    }
    X->requires_grad = false;

    // Reduced values are [1,1] because reductions keep their dimensions
    int64_t s_shape[] = {1, 1};
    causa_tensor_t* loss_mse   = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);
    causa_tensor_t* l1_norm    = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);
    causa_tensor_t* h_val      = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);
    causa_tensor_t* h_sq       = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);
    causa_tensor_t* l1_term    = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);
    causa_tensor_t* acyc_term  = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);
    causa_tensor_t* beta_term  = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);
    causa_tensor_t* sum_a      = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);
    causa_tensor_t* sum_b      = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);
    causa_tensor_t* total_loss = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);

    // Constants; the augmented Lagrangian multiplier is updated in place
    causa_tensor_t* c_mse_scale = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);
    causa_tensor_fill(c_mse_scale, 1.0f / (2.0f * (float)N));
    causa_tensor_t* c_alpha = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);
    causa_tensor_fill(c_alpha, cfg.discovery_alpha);
    causa_tensor_t* c_rho = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);
    causa_tensor_fill(c_rho, 0.5f);
    causa_tensor_t* c_beta = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);
    causa_tensor_fill(c_beta, 0.0f);
    causa_tensor_t* c_d = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);
    causa_tensor_fill(c_d, (float)D);

    // ---- Least squares loss: ||X - XW||^2 / (2N) ----
    causa_tensor_t* XW = causa_graph_new_tensor(opt_graph, observational_data->shape, observational_data->ndim, &opts);
    causa_tensor_t* residual = causa_graph_new_tensor(opt_graph, observational_data->shape, observational_data->ndim, &opts);
    causa_tensor_t* residual_sq = causa_graph_new_tensor(opt_graph, observational_data->shape, observational_data->ndim, &opts);
    causa_tensor_t* mse_raw = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);

    causa_graph_add_op(opt_graph, CAUSA_OP_MATMUL, "XW",
        (causa_tensor_t*[]){X, W}, 2, (causa_tensor_t*[]){XW}, 1, NULL, NULL, NULL, 0);
    causa_graph_add_op(opt_graph, CAUSA_OP_SUB, "residual",
        (causa_tensor_t*[]){X, XW}, 2, (causa_tensor_t*[]){residual}, 1, NULL, NULL, NULL, 0);
    causa_graph_add_op(opt_graph, CAUSA_OP_MUL, "residual_sq",
        (causa_tensor_t*[]){residual, residual}, 2, (causa_tensor_t*[]){residual_sq}, 1, NULL, NULL, NULL, 0);

    causa_reduce_ctx_t* red_ctx = (causa_reduce_ctx_t*)causa_arena_alloc(opt_arena, sizeof(causa_reduce_ctx_t), 8);
    red_ctx->reduce_op = CAUSA_REDUCE_SUM;
    red_ctx->num_dims = 2;
    red_ctx->dims[0] = 0;
    red_ctx->dims[1] = 1;
    red_ctx->keepdim = true;

    causa_graph_add_op(opt_graph, CAUSA_OP_REDUCE, "mse_sum",
        (causa_tensor_t*[]){residual_sq}, 1, (causa_tensor_t*[]){mse_raw}, 1,
        NULL, NULL, red_ctx, sizeof(causa_reduce_ctx_t));
    causa_graph_add_op(opt_graph, CAUSA_OP_MUL, "mse_scaled",
        (causa_tensor_t*[]){mse_raw, c_mse_scale}, 2, (causa_tensor_t*[]){loss_mse}, 1, NULL, NULL, NULL, 0);

    // ---- L1 penalty on W ----
    causa_tensor_t* W_abs = causa_graph_new_tensor(opt_graph, w_shape, 2, &opts);
    causa_graph_add_op(opt_graph, CAUSA_OP_ABS, "W_abs",
        (causa_tensor_t*[]){W}, 1, (causa_tensor_t*[]){W_abs}, 1, NULL, NULL, NULL, 0);
    causa_graph_add_op(opt_graph, CAUSA_OP_REDUCE, "l1_sum",
        (causa_tensor_t*[]){W_abs}, 1, (causa_tensor_t*[]){l1_norm}, 1,
        NULL, NULL, red_ctx, sizeof(causa_reduce_ctx_t));

    // ---- Acyclicity constraint: h(W) = tr(exp(W . W)) - d ----
    //
    // A = W . W (elementwise square)
    causa_tensor_t* A = causa_graph_new_tensor(opt_graph, w_shape, 2, &opts);
    causa_graph_add_op(opt_graph, CAUSA_OP_MUL, "W_sq",
        (causa_tensor_t*[]){W, W}, 2, (causa_tensor_t*[]){A}, 1, NULL, NULL, NULL, 0);

    // Scale A by 1/2^n for the Taylor evaluation
    #define CAUSA_EXP_SCALE 8
    #define CAUSA_EXP_TERMS 32
    causa_tensor_t* c_inv_scale = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);
    causa_tensor_fill(c_inv_scale, 1.0f / (float)(1 << CAUSA_EXP_SCALE));
    causa_tensor_t* B = causa_graph_new_tensor(opt_graph, w_shape, 2, &opts);
    causa_graph_add_op(opt_graph, CAUSA_OP_MUL, "A_scaled",
        (causa_tensor_t*[]){A, c_inv_scale}, 2, (causa_tensor_t*[]){B}, 1, NULL, NULL, NULL, 0);

    // Identity matrix, the starting point of the Taylor series
    causa_tensor_t* ident = causa_graph_new_tensor(opt_graph, w_shape, 2, &opts);
    causa_tensor_fill(ident, 0.0f);
    {
        float* id = (float*)ident->data;
        for (int32_t i = 0; i < D; i++) id[i * D + i] = 1.0f;
    }
    ident->requires_grad = false;

    // Taylor series: E = I + B + B^2/2! + ... + B^k/k!
    causa_tensor_t* term = ident;
    causa_tensor_t* E = ident;
    for (int32_t k = 1; k <= CAUSA_EXP_TERMS; k++) {
        causa_tensor_t* term_mul = causa_graph_new_tensor(opt_graph, w_shape, 2, &opts);
        causa_graph_add_op(opt_graph, CAUSA_OP_MATMUL, "exp_term_mul",
            (causa_tensor_t*[]){term, B}, 2, (causa_tensor_t*[]){term_mul}, 1, NULL, NULL, NULL, 0);

        double fact = 1.0;
        for (int32_t j = 2; j <= k; j++) fact *= (double)j;
        causa_tensor_t* c_fact = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);
        causa_tensor_fill(c_fact, (float)(1.0 / fact));

        causa_tensor_t* term_scaled = causa_graph_new_tensor(opt_graph, w_shape, 2, &opts);
        causa_graph_add_op(opt_graph, CAUSA_OP_MUL, "exp_term_scaled",
            (causa_tensor_t*[]){term_mul, c_fact}, 2, (causa_tensor_t*[]){term_scaled}, 1, NULL, NULL, NULL, 0);

        causa_tensor_t* E_next = causa_graph_new_tensor(opt_graph, w_shape, 2, &opts);
        causa_graph_add_op(opt_graph, CAUSA_OP_ADD, "exp_accum",
            (causa_tensor_t*[]){E, term_scaled}, 2, (causa_tensor_t*[]){E_next}, 1, NULL, NULL, NULL, 0);

        term = term_scaled;
        E = E_next;
    }

    // Squaring: exp(A) = (exp(A / 2^n))^(2^n)
    for (int32_t i = 0; i < CAUSA_EXP_SCALE; i++) {
        causa_tensor_t* E_sq = causa_graph_new_tensor(opt_graph, w_shape, 2, &opts);
        causa_graph_add_op(opt_graph, CAUSA_OP_MATMUL, "exp_square",
            (causa_tensor_t*[]){E, E}, 2, (causa_tensor_t*[]){E_sq}, 1, NULL, NULL, NULL, 0);
        E = E_sq;
    }

    // trace(E) = sum of the diagonal
    causa_tensor_t* diag = causa_graph_new_tensor(opt_graph, (int64_t[]){D}, 1, &opts);
    causa_graph_add_op(opt_graph, CAUSA_OP_DIAG, "diag",
        (causa_tensor_t*[]){E}, 1, (causa_tensor_t*[]){diag}, 1, NULL, NULL, NULL, 0);

    causa_reduce_ctx_t* trace_ctx = (causa_reduce_ctx_t*)causa_arena_alloc(opt_arena, sizeof(causa_reduce_ctx_t), 8);
    trace_ctx->reduce_op = CAUSA_REDUCE_SUM;
    trace_ctx->num_dims = 1;
    trace_ctx->dims[0] = 0;
    trace_ctx->keepdim = true;

    causa_tensor_t* trace_raw = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);
    causa_graph_add_op(opt_graph, CAUSA_OP_REDUCE, "trace",
        (causa_tensor_t*[]){diag}, 1, (causa_tensor_t*[]){trace_raw}, 1,
        NULL, NULL, trace_ctx, sizeof(causa_reduce_ctx_t));

    // h = trace - d
    causa_graph_add_op(opt_graph, CAUSA_OP_SUB, "h",
        (causa_tensor_t*[]){trace_raw, c_d}, 2, (causa_tensor_t*[]){h_val}, 1, NULL, NULL, NULL, 0);

    // h^2
    causa_graph_add_op(opt_graph, CAUSA_OP_MUL, "h_sq",
        (causa_tensor_t*[]){h_val, h_val}, 2, (causa_tensor_t*[]){h_sq}, 1, NULL, NULL, NULL, 0);

    // ---- Augmented Lagrangian terms ----
    causa_graph_add_op(opt_graph, CAUSA_OP_MUL, "l1_term",
        (causa_tensor_t*[]){c_alpha, l1_norm}, 2, (causa_tensor_t*[]){l1_term}, 1, NULL, NULL, NULL, 0);
    causa_graph_add_op(opt_graph, CAUSA_OP_MUL, "acyc_term",
        (causa_tensor_t*[]){c_rho, h_sq}, 2, (causa_tensor_t*[]){acyc_term}, 1, NULL, NULL, NULL, 0);
    causa_graph_add_op(opt_graph, CAUSA_OP_MUL, "beta_term",
        (causa_tensor_t*[]){c_beta, h_val}, 2, (causa_tensor_t*[]){beta_term}, 1, NULL, NULL, NULL, 0);

    causa_graph_add_op(opt_graph, CAUSA_OP_ADD, "loss_sum_a",
        (causa_tensor_t*[]){loss_mse, l1_term}, 2, (causa_tensor_t*[]){sum_a}, 1, NULL, NULL, NULL, 0);
    causa_graph_add_op(opt_graph, CAUSA_OP_ADD, "loss_sum_b",
        (causa_tensor_t*[]){sum_a, acyc_term}, 2, (causa_tensor_t*[]){sum_b}, 1, NULL, NULL, NULL, 0);
    causa_graph_add_op(opt_graph, CAUSA_OP_ADD, "loss_total",
        (causa_tensor_t*[]){sum_b, beta_term}, 2, (causa_tensor_t*[]){total_loss}, 1, NULL, NULL, NULL, 0);

    causa_graph_build(opt_graph);

    // ---- Optimization ----
    //
    // Adam, which is what makes NOTEARS converge in practice. The augmented
    // Lagrangian multiplier is updated from the current constraint violation.
    int32_t max_iter = cfg.max_iterations > 0 ? cfg.max_iterations : 200;
    float lr = 0.01f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    float rho = 1.0f;
    float beta = 0.0f;

    float* m = (float*)calloc((size_t)D * D, sizeof(float));
    float* v = (float*)calloc((size_t)D * D, sizeof(float));
    if (!m || !v) {
        free(m); free(v);
        causa_graph_free(opt_graph);
        causa_arena_free(opt_arena);
        return disc;
    }

    float* w_data = (float*)W->data;

    for (int32_t iter = 0; iter < max_iter; iter++) {
        causa_graph_zero_grad(opt_graph);
        causa_graph_forward(opt_graph);

        // Update the augmented Lagrangian multiplier
        float h_now = ((float*)h_val->data)[0];
        beta += rho * h_now;
        causa_tensor_fill(c_beta, beta);

        // Backward pass
        causa_tensor_t* loss_grad = causa_graph_new_tensor(opt_graph, s_shape, 2, &opts);
        causa_tensor_fill(loss_grad, 1.0f);
        total_loss->grad = loss_grad;
        causa_graph_backward(opt_graph, NULL, 0);

        if (!W->grad) continue;

        float* g = (float*)W->grad->data;
        float bias1 = 1.0f - powf(beta1, (float)(iter + 1));
        float bias2 = 1.0f - powf(beta2, (float)(iter + 1));

        for (int32_t i = 0; i < D * D; i++) {
            float gi = g[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * gi;
            v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;
            float mh = m[i] / bias1;
            float vh = v[i] / bias2;
            w_data[i] -= lr * mh / (sqrtf(vh) + eps);
        }

        // A variable cannot be its own cause, so the diagonal of the weight
        // matrix is held at zero. Without this the fit would explain a
        // variable by itself, and the acyclicity constraint would be spent
        // on a self-loop that cannot exist.
        for (int32_t i = 0; i < D; i++) {
            w_data[i * D + i] = 0.0f;
        }
    }

    free(m);
    free(v);

    // ---- Extract the learned structure ----
    //
    // The model is X ~= X W, so (X W)[n][child] = sum_p X[n][parent] * W[parent][child].
    // A nonzero W[parent][child] therefore means parent -> child.
    //
    // The weights are compared with the threshold on the scale of the
    // standard deviations, so that the threshold means the same thing for
    // every variable: a weight of w moves the child by w standard
    // deviations when the parent moves by one.
    for (int32_t parent = 0; parent < D; parent++) {
        for (int32_t child = 0; child < D; child++) {
            if (parent == child) continue;
            float w = w_data[parent * D + child];
            float magnitude = fabsf(w) * (stddev[parent] / stddev[child]);
            disc->weights[parent][child] = w;
            if (magnitude > cfg.discovery_threshold) {
                disc->adjacency[parent][child] = 1;
                disc->confidence[parent][child] = magnitude;
                if (disc->num_parents[child] < CAUSA_MAX_PARENTS) {
                    disc->parents[child][disc->num_parents[child]++] = parent;
                }
            }
        }
    }

    // ---- Estimate the coefficients ----
    //
    // The optimisation finds the structure, but the weights it leaves
    // behind are not the coefficients of the structural equations: the
    // acyclicity constraint and the sparsity penalty both pull them away
    // from the fit. The coefficients are therefore estimated again, by
    // regressing each child on the parents that were selected.
    {
        const float* data = (const float*)observational_data->data;

        for (int32_t child = 0; child < D; child++) {
            int32_t num = disc->num_parents[child];
            if (num <= 0) continue;

            int32_t parents[CAUSA_MAX_PARENTS];
            float coefficients[CAUSA_MAX_PARENTS];
            for (int32_t p = 0; p < num; p++) parents[p] = disc->parents[child][p];

            if (!causal_refit_coefficients(data, N, D, child, parents, num, mean, stddev, coefficients)) {
                continue;
            }

            // A parent whose effect is negligible is not a cause. The fit
            // is repeated without it, so the coefficients that remain are
            // not distorted by a parent that does nothing.
            int32_t kept = 0;
            for (int32_t p = 0; p < num; p++) {
                float magnitude = fabsf(coefficients[p]) * (stddev[parents[p]] / stddev[child]);
                if (magnitude > cfg.discovery_threshold) kept++;
            }

            if (kept < num) {
                int32_t fewer[CAUSA_MAX_PARENTS];
                float fewer_coefficients[CAUSA_MAX_PARENTS];
                int32_t count = 0;
                for (int32_t p = 0; p < num; p++) {
                    float magnitude = fabsf(coefficients[p]) * (stddev[parents[p]] / stddev[child]);
                    if (magnitude > cfg.discovery_threshold) {
                        fewer[count] = parents[p];
                        count++;
                    }
                }
                if (count > 0 && causal_refit_coefficients(data, N, D, child, fewer, count, mean, stddev, fewer_coefficients)) {
                    for (int32_t p = 0; p < count; p++) {
                        parents[p] = fewer[p];
                        coefficients[p] = fewer_coefficients[p];
                    }
                    num = count;
                }
            }

            disc->num_parents[child] = num;
            for (int32_t p = 0; p < num; p++) {
                int32_t parent = parents[p];
                disc->parents[child][p] = parent;
                disc->weights[parent][child] = coefficients[p];
                disc->confidence[parent][child] = fabsf(coefficients[p]) * (stddev[parent] / stddev[child]);
            }
        }
    }

    // ---- Orient ambiguous edges using interventional data ----
    //
    // If intervening on variable a moves variable b, then a is an ancestor of
    // b. Where the observational structure had the edge the other way round,
    // correct it.
    if (cfg.use_interventional && interventional_data && intervention_targets) {
        int32_t M = (int32_t)interventional_data->shape[0];

        float obs_mean[CAUSA_MAX_VARIABLES];
        {
            const float* od = (const float*)observational_data->data;
            for (int32_t i = 0; i < D; i++) obs_mean[i] = 0.0f;
            for (int32_t n = 0; n < N; n++) {
                for (int32_t i = 0; i < D; i++) obs_mean[i] += od[n * D + i];
            }
            for (int32_t i = 0; i < D; i++) obs_mean[i] /= (float)N;
        }

        const float* idata = (const float*)interventional_data->data;
        const float* itargets = (const float*)intervention_targets->data;

        for (int32_t row = 0; row < M; row++) {
            int32_t target = (int32_t)itargets[row];
            if (target < 0 || target >= D) continue;

            for (int32_t i = 0; i < D; i++) {
                if (i == target) continue;

                float int_mean = 0.0f;
                for (int32_t n = 0; n < N; n++) {
                    int_mean += idata[row * D + i];
                }
                int_mean /= (float)N;

                if (fabsf(int_mean - obs_mean[i]) <= cfg.discovery_threshold) continue;

                // i responds to do(target): target is an ancestor of i.
                if (disc->adjacency[i][target] && !disc->adjacency[target][i]) {
                    disc->adjacency[i][target] = 0;
                    disc->adjacency[target][i] = 1;
                    disc->confidence[target][i] = disc->confidence[i][target];
                    disc->confidence[i][target] = 0.0f;
                    disc->weights[target][i] = disc->weights[i][target];
                    disc->weights[i][target] = 0.0f;

                    for (int32_t k = 0; k < disc->num_parents[i]; k++) {
                        if (disc->parents[i][k] == target) {
                            disc->parents[i][k] = disc->parents[i][disc->num_parents[i] - 1];
                            disc->num_parents[i]--;
                            break;
                        }
                    }
                    if (disc->num_parents[target] < CAUSA_MAX_PARENTS) {
                        disc->parents[target][disc->num_parents[target]++] = i;
                    }
                }
            }
        }
    }

    causa_graph_free(opt_graph);
    causa_arena_free(opt_arena);

    return disc;
}

void causa_apply_discovery(
    causa_scm_t* scm,
    causa_discovery_t* discovery,
    float confidence_threshold
) {
    if (!scm || !discovery) return;

    // Remove edges below threshold
    for (int32_t i = 0; i < discovery->num_variables; i++) {
        for (int32_t j = 0; j < discovery->num_variables; j++) {
            if (discovery->confidence[i][j] < confidence_threshold) {
                discovery->adjacency[i][j] = 0;
            }
        }
    }

    // The parent lists have to follow the pruned adjacency, otherwise a
    // parent that was dropped still shows up in the list and the law that
    // is built from it is wrong.
    for (int32_t child = 0; child < discovery->num_variables; child++) {
        int32_t kept = 0;
        for (int32_t p = 0; p < discovery->num_parents[child]; p++) {
            int32_t parent = discovery->parents[child][p];
            if (parent < 0 || parent >= discovery->num_variables) continue;
            if (!discovery->adjacency[parent][child]) continue;
            discovery->parents[child][kept++] = parent;
        }
        discovery->num_parents[child] = kept;
    }
}

int32_t causa_get_parents(
    causa_discovery_t* discovery,
    int32_t var_idx,
    int32_t* out_parents,
    int32_t max_parents
) {
    if (!discovery || var_idx < 0 || var_idx >= discovery->num_variables) return -1;

    int32_t count = 0;
    for (int32_t i = 0; i < discovery->num_parents[var_idx] && count < max_parents; i++) {
        out_parents[count++] = discovery->parents[var_idx][i];
    }
    return count;
}

// ===== Transportability =====

causa_tensor_t* causa_transport(
    causa_scm_t* source_scm,
    causa_scm_t* target_scm,
    causa_tensor_t* source_intervention,
    causa_tensor_t* target_observations,
    causa_tensor_t* selection_vars
) {
    // Placeholder: transport P_source(Y|do(X)) to P_target(Y|do(X))
    // Using selection diagram adjustment
    (void)source_scm; (void)target_scm; (void)source_intervention;
    (void)target_observations; (void)selection_vars;
    return NULL;
}

// ===== Memory Integration =====

// Consolidate what episodic memory has seen into causal laws in semantic
// memory.
//
// Episodic slots are expected to hold one observation each, with the key
// holding the observed variables. Those observations are gathered into a
// matrix, causal discovery is run over them, and the resulting parent sets
// and coefficients are written to semantic memory as causal laws.
//
// A law for child c is stored with a role vector: 2 at the child's position,
// 1 at each parent's position, 0 elsewhere. The value holds the coefficient
// of every variable in the child's equation.
//
// Returns the number of laws written, or -1 on error.
int32_t causa_memory_consolidate_causal(
    causa_memory_t* memory,
    causa_scm_t* scm,
    const causa_config_t* config
) {
    if (!memory) return -1;

    causa_config_t cfg = CAUSA_CONFIG_DEFAULTS;
    if (config) cfg = *config;

    const causa_memory_config_t* mcfg = causa_memory_config(memory);
    if (!mcfg) return -1;

    int64_t dim = mcfg->key_dim;
    if (dim <= 0 || dim > CAUSA_MAX_VARIABLES) return -1;

    // Gather the episodic observations
    int64_t episodic_count = causa_memory_size(memory, CAUSA_MEMORY_EPISODIC);
    if (episodic_count <= 0) return 0;

    causa_arena_t* arena = causa_memory_arena(memory);
    causa_backend_t* backend = causa_memory_backend(memory);
    if (!arena || !backend) return -1;

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;

    int64_t obs_shape[] = {episodic_count, dim};
    causa_tensor_t* observations = causa_tensor_new(arena, backend, obs_shape, 2, &opts);
    if (!observations) return -1;

    int64_t gathered = causa_memory_collect(memory, CAUSA_MEMORY_EPISODIC, observations);
    if (gathered < 0) return -1;
    if (gathered != episodic_count) {
        // The matrix was sized for the full count; trim the row count
        observations->shape[0] = gathered;
    }
    if (gathered < 3) return 0;   // not enough data to identify a structure

    // Run causal discovery over the observations
    causa_discovery_t* discovery = causa_discover(observations, NULL, NULL, &cfg);
    if (!discovery) return -1;

    // Write the discovered laws into semantic memory
    int32_t laws = 0;
    int64_t coeff_shape[] = {dim};

    for (int32_t child = 0; child < discovery->num_variables; child++) {
        int32_t num_parents = discovery->num_parents[child];
        if (num_parents == 0) continue;

        causa_tensor_t* mask = causa_tensor_new(arena, backend, coeff_shape, 1, &opts);
        causa_tensor_t* coeffs = causa_tensor_new(arena, backend, coeff_shape, 1, &opts);
        if (!mask || !coeffs) continue;

        float* m = (float*)mask->data;
        float* c = (float*)coeffs->data;
        for (int32_t i = 0; i < dim; i++) { m[i] = 0.0f; c[i] = 0.0f; }

        for (int32_t p = 0; p < num_parents; p++) {
            int32_t parent = discovery->parents[child][p];
            if (parent < 0 || parent >= dim) continue;
            m[parent] = 1.0f;
            // The signed weight is the coefficient in the child's equation.
            c[parent] = discovery->weights[parent][child];
        }

        if (causa_memory_store_causal_law(memory, child, mask, coeffs)) {
            laws++;
        }
    }

    // If the SCM was supplied, prune the edges that the data does not support
    if (scm && scm->num_variables > 0) {
        causa_apply_discovery(scm, discovery, cfg.discovery_threshold);
    }

    causa_discovery_free(discovery);
    return laws;
}

void causa_memory_causal_consolidate(
    causa_memory_t* memory,
    causa_scm_t* scm,
    causa_discovery_t* discovery
) {
    if (!memory || !scm || !discovery) return;

    int64_t dim = causa_memory_config(memory)->key_dim;
    if (dim <= 0 || dim > CAUSA_MAX_VARIABLES) return;

    causa_arena_t* arena = causa_memory_arena(memory);
    causa_backend_t* backend = causa_memory_backend(memory);
    if (!arena || !backend) return;

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;

    int64_t coeff_shape[] = {dim};

    for (int32_t child = 0; child < discovery->num_variables; child++) {
        int32_t num_parents = discovery->num_parents[child];
        if (num_parents == 0) continue;

        causa_tensor_t* mask = causa_tensor_new(arena, backend, coeff_shape, 1, &opts);
        causa_tensor_t* coeffs = causa_tensor_new(arena, backend, coeff_shape, 1, &opts);
        if (!mask || !coeffs) continue;

        float* m = (float*)mask->data;
        float* c = (float*)coeffs->data;
        for (int32_t i = 0; i < dim; i++) { m[i] = 0.0f; c[i] = 0.0f; }

        for (int32_t p = 0; p < num_parents; p++) {
            int32_t parent = discovery->parents[child][p];
            if (parent < 0 || parent >= dim) continue;
            m[parent] = 1.0f;
            c[parent] = discovery->weights[parent][child];
        }

        causa_memory_store_causal_law(memory, child, mask, coeffs);
    }

    causa_apply_discovery(scm, discovery, 0.0f);
}

causa_tensor_t* causa_memory_store_counterfactual(
    causa_memory_t* memory,
    causa_counterfactual_t* cf
) {
    if (!memory || !cf || !cf->result) return NULL;

    // Create key from (intervention_var, intervention_val, query)
    // Store in episodic memory
    return causa_memory_write(memory, cf->intervention_var, cf->result, CAUSA_MEMORY_EPISODIC);
}

causa_tensor_t* causa_memory_retrieve_counterfactuals(
    causa_memory_t* memory,
    causa_tensor_t* query,
    causa_tensor_t* context,
    int32_t k
) {
    if (!memory || !query || !context) return NULL;

    // Search episodic memory for similar counterfactuals
    causa_tensor_t* keys[16];
    causa_tensor_t* values[16];
    return causa_memory_read_topk(memory, context, CAUSA_MEMORY_EPISODIC, k, keys, values);
}

// ===== Utility =====

void causa_scm_print(const causa_scm_t* scm) {
    if (!scm) return;
    printf("SCM: %d variables, %d exogenous, topo=%d\n", scm->num_variables, scm->num_exogenous, scm->num_topo);
    for (int32_t i = 0; i < scm->num_variables; i++) {
        const causa_scm_var_t* var = &scm->variables[i];
        printf("  %s <- ", var->variable->name[0] ? var->variable->name : "unnamed");
        for (int32_t p = 0; p < var->num_parents; p++) {
            printf("%s ", var->parents[p]->name[0] ? var->parents[p]->name : "?");
        }
        if (var->is_intervened) printf(" [DO=%p]", (void*)var->intervention_value);
        printf("\n");
    }
}

void causa_intervention_print(const causa_intervention_t* intervention) {
    if (!intervention) return;
    printf("do(%s = %p)\n", intervention->target->name[0] ? intervention->target->name : "?", (void*)intervention->value);
}

void causa_counterfactual_print(const causa_counterfactual_t* cf) {
    if (!cf) return;
    printf("Counterfactual: %s_{%s=%p} | evidence=%p | done: A=%d Act=%d P=%d\n",
           cf->query->name[0] ? cf->query->name : "Y",
           cf->intervention_var->name[0] ? cf->intervention_var->name : "X",
           (void*)cf->intervention_val,
           (void*)cf->evidence,
           cf->abduction_done, cf->action_done, cf->prediction_done);
}

void causa_discovery_print(const causa_discovery_t* discovery) {
    if (!discovery) return;
    printf("Causal Discovery: %d variables\n", discovery->num_variables);
    for (int32_t i = 0; i < discovery->num_variables; i++) {
        if (discovery->num_parents[i] > 0) {
            printf("  %d <- ", i);
            for (int32_t p = 0; p < discovery->num_parents[i]; p++) {
                printf("%d (%.2f) ", discovery->parents[i][p], discovery->confidence[i][discovery->parents[i][p]]);
            }
            printf("\n");
        }
    }
}
