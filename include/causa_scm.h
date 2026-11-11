#ifndef CAUSA_SCM_H
#define CAUSA_SCM_H

#include "causa_tensor.h"
#include "causa_graph.h"
#include "causa_memory.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CAUSA_MAX_VARIABLES 64
#define CAUSA_MAX_PARENTS 16

// Forward declarations
typedef struct causa_scm_s causa_scm_t;
typedef struct causa_intervention_s causa_intervention_t;
typedef struct causa_counterfactual_s causa_counterfactual_t;
typedef struct causa_discovery_s causa_discovery_t;

// Causal variable in SCM
typedef struct {
    causa_tensor_t* variable;           // The variable X
    causa_tensor_t** parents;           // Pa(X) - direct causal parents
    int32_t num_parents;
    causa_tensor_t* noise;              // Exogenous noise U_X
    causa_op_forward_fn structural_fn;  // f(Pa(X), U_X) -> X
    causa_op_backward_fn structural_bwd; // Optional: for counterfactual gradients
    causa_tensor_t* intervention_value; // The value x in do(X=x)
    bool is_intervened;                  // True if do(X=x) applied
} causa_scm_var_t;

// Configuration
typedef struct {
    float discovery_alpha;      // Sparsity penalty for NOTEARS
    float discovery_threshold;  // Edge threshold
    int32_t max_iterations;     // Max optimization iterations
    bool use_interventional;    // Use interventional data if available
    float counterfactual_lr;    // Learning rate for counterfactual optimization
    int32_t counterfactual_steps; // Steps for abduction optimization
} causa_config_t;

#define CAUSA_CONFIG_DEFAULTS { \
    .discovery_alpha = 0.1f, \
    .discovery_threshold = 0.3f, \
    .max_iterations = 200, \
    .use_interventional = true, \
    .counterfactual_lr = 0.01f, \
    .counterfactual_steps = 2000 \
}

// Structural Causal Model
struct causa_scm_s {
    causa_arena_t* arena;
    causa_backend_t* backend;
    causa_graph_t* base_graph;          // Original observational graph
    causa_graph_t* intervened_graph;    // Graph after do() interventions

    // For an intervened SCM built from a graph, this maps the source graph's
    // tensors onto the intervened graph's tensors, so that values can be
    // read back after the intervened graph runs.
    causa_tensor_t** intervened_map;
    int32_t num_intervened_map;

    causa_scm_var_t variables[CAUSA_MAX_VARIABLES];
    int32_t num_variables;

    causa_tensor_t** exogenous_noise;   // U variables for all nodes
    int32_t num_exogenous;

    // The number of operations that belong to the model itself. The
    // reconstruction loss added by the abduction step is not part of the
    // model and must not be copied into an intervened graph, because its
    // inputs are the observed values rather than model tensors.
    int32_t num_model_ops;

    // Topological order for ancestral sampling
    int32_t topo_order[CAUSA_MAX_VARIABLES];
    int32_t num_topo;

    // The settings the model was made with. The counterfactual step reads
    // the learning rate and the iteration count from here.
    causa_config_t config;

    bool built;
};
// Intervention: do(X = x)
struct causa_intervention_s {
    causa_tensor_t* target;      // Variable X
    causa_tensor_t* value;       // Value x
    causa_scm_t* original_scm;   // Reference to original
    causa_scm_t* intervened_scm; // Resulting SCM
};

// Counterfactual query Y_x(u)
struct causa_counterfactual_s {
    causa_tensor_t* query;           // Y (what we want to know)
    causa_tensor_t* intervention_var; // X (what we intervene on)
    causa_tensor_t* intervention_val; // x (the intervention value)
    causa_tensor_t* evidence;        // Observed factual data
    causa_tensor_t** evidence_vars;  // Observed value per variable, used by abduction
    int32_t num_evidence_vars;
    causa_tensor_t** inferred_noise;  // Array of U (abduced from evidence), one per exogenous
    int32_t num_inferred_noise;
    causa_tensor_t* result;          // Y_x(u) - the counterfactual outcome

    // How well the abduction fitted the evidence. A caller that gets a
    // counterfactual it does not believe can see here that the inference
    // did not settle.
    float abduction_error;
    int32_t abduction_steps;

    // The model the action step built. It is kept so that the prediction
    // step uses the model the action made rather than building a second
    // one.
    causa_scm_t* intervened_scm;

    // Three-step state
    bool abduction_done;
    bool action_done;
    bool prediction_done;
};

// Causal discovery result
struct causa_discovery_s {
    // Adjacency matrix: discovered[i][j] = 1 if j -> i (j is parent of i)
    int32_t adjacency[CAUSA_MAX_VARIABLES][CAUSA_MAX_VARIABLES];
    float confidence[CAUSA_MAX_VARIABLES][CAUSA_MAX_VARIABLES];

    // The signed weight of every edge: weights[parent][child] is the
    // coefficient of the parent in the child's equation. The confidence
    // matrix holds the magnitude of the same weight, which is what the
    // edge threshold is applied to. The sign is kept separately because a
    // negative effect is as real a causal law as a positive one.
    float weights[CAUSA_MAX_VARIABLES][CAUSA_MAX_VARIABLES];

    int32_t num_variables;

    // For each variable, list of discovered parents
    int32_t parents[CAUSA_MAX_VARIABLES][CAUSA_MAX_PARENTS];
    int32_t num_parents[CAUSA_MAX_VARIABLES];
};

// ===== SCM Construction =====

causa_scm_t* causa_scm_new(
    causa_arena_t* arena,
    causa_backend_t* backend,
    const causa_config_t* config
);

void causa_scm_free(causa_scm_t* scm);

// Build SCM from computation graph (each node becomes structural equation)
int causa_scm_from_graph(
    causa_scm_t* scm,
    causa_graph_t* graph,
    causa_tensor_t** exogenous_noise,
    int32_t num_exogenous
);

// Add structural equation manually: X := f(Pa(X), U_X)
int causa_scm_add_equation(
    causa_scm_t* scm,
    causa_tensor_t* target,
    causa_tensor_t** parents,
    int32_t num_parents,
    causa_tensor_t* noise,
    causa_op_forward_fn structural_fn,
    causa_op_backward_fn structural_bwd
);

// Add a structural equation as an operation in the SCM's graph.
//
// The equation becomes a real graph operation, which means the model is
// differentiable: gradients flow from the variables back to the exogenous
// noise, and the noise can be inferred by optimisation.
//
// The op takes the parents and the noise as inputs and writes the target.
// For the common linear case, use CAUSA_OP_MUL with a coefficient tensor in
// ctx, followed by CAUSA_OP_ADD with the noise.
//
// Returns the number of equations added.
int causa_scm_add_equation_op(
    causa_scm_t* scm,
    causa_tensor_t* target,
    causa_tensor_t** parents,
    int32_t num_parents,
    causa_tensor_t* noise,
    causa_op_type_t op_type,
    const void* op_ctx,
    size_t op_ctx_size
);

// Add a reconstruction loss to the SCM's graph.
//
// The loss is the sum over the given variables of the squared difference
// between the value the model computes and the observed value. It is the
// objective the abduction step minimises.
//
// Requires the SCM to have been built from a graph. Returns the loss tensor.
causa_tensor_t* causa_scm_add_reconstruction_loss(
    causa_scm_t* scm,
    causa_tensor_t** observed,
    int32_t num_observed
);

// Register a variable that has no structural equation, such as an observed
// exogenous variable.
//
// Such a variable takes part in the reconstruction loss but its value is
// fixed. Registering it keeps the SCM's variable indices aligned with the
// observed values that the abduction step is given.
int causa_scm_add_observed_variable(causa_scm_t* scm, causa_tensor_t* variable);

// Register an exogenous noise tensor.
//
// The abduction step infers the registered noise by minimising the
// reconstruction error, so every noise term that should be inferred has to
// be registered.
int causa_scm_add_exogenous(causa_scm_t* scm, causa_tensor_t* noise);

// Build topological order for ancestral sampling
int causa_scm_build(causa_scm_t* scm);

// Sample from SCM (ancestral sampling)
causa_tensor_t* causa_scm_sample(
    causa_scm_t* scm,
    causa_tensor_t** outputs,
    int32_t num_outputs
);

// Evaluate the SCM with the noise values currently stored, without drawing
// new noise. A counterfactual needs this, because the abduction step has
// already fixed the noise.
causa_tensor_t* causa_scm_evaluate(
    causa_scm_t* scm,
    causa_tensor_t** outputs,
    int32_t num_outputs
);

// ===== Building a model from memory =====

// Build a structural causal model from the causal laws in semantic memory.
//
// This is the link that turns a memory into a model: the laws that were
// consolidated from the episodes become the structural equations, and the
// model can then be intervened on or reasoned about like any other.
//
// Each variable that has a law becomes an equation over its parents, with
// an exogenous noise term of its own. A variable without a law is an
// exogenous input.
//
// The equations are added to the SCM's own graph, so the model is
// differentiable and the abduction step can infer the parts of it that were
// not observed.
//
// The variables are registered in index order, which is the order the
// observed values have to be given in.
//
// num_variables is the size of the variable space. Returns the number of
// equations that were added, or -1 on error.
int32_t causa_memory_build_scm(
    causa_memory_t* mem,
    causa_scm_t* scm,
    int32_t num_variables
);

// ===== Intervention: do(X = x) =====

// Single intervention
causa_intervention_t* causa_intervention_new(
    causa_arena_t* arena,
    causa_tensor_t* target,
    causa_tensor_t* value
);

// Apply do(X=x) - returns new intervened SCM
causa_scm_t* causa_scm_do(
    causa_scm_t* scm,
    causa_intervention_t* intervention
);

// Multiple interventions do(X1=x1, X2=x2, ...)
causa_scm_t* causa_scm_do_multi(
    causa_scm_t* scm,
    causa_tensor_t** targets,
    causa_tensor_t** values,
    int32_t num_interventions
);

// ===== Counterfactual Reasoning =====

causa_counterfactual_t* causa_counterfactual_new(
    causa_arena_t* arena,
    causa_tensor_t* query,           // Y
    causa_tensor_t* intervention_var, // X
    causa_tensor_t* intervention_val, // x
    causa_tensor_t* evidence         // observed data
);

// Attach the observed value of every variable, which is what the abduction
// step needs to infer the exogenous noise.
void causa_counterfactual_set_evidence(
    causa_counterfactual_t* cf,
    causa_tensor_t** observed,
    int32_t count
);

// Three-step counterfactual:
// 1. Abduction: infer U from evidence (optimization)
// 2. Action: apply do(X=x) to SCM
// 3. Prediction: compute Y in modified SCM with inferred U
causa_tensor_t* causa_counterfactual_compute(
    causa_scm_t* scm,
    causa_counterfactual_t* cf
);

// Step 1: Abduction - infer exogenous noise U from observed evidence
// Minimizes ||SCM(U) - evidence||^2 via gradient descent
int causa_counterfactual_abduction(
    causa_counterfactual_t* cf,
    causa_scm_t* scm,
    float lr,
    int32_t steps
);

// Step 2: Action - create intervened SCM (uses causa_scm_do)
int causa_counterfactual_action(
    causa_counterfactual_t* cf,
    causa_scm_t* scm
);

// Step 3: Prediction - run intervened SCM with inferred U
int causa_counterfactual_prediction(
    causa_counterfactual_t* cf,
    causa_scm_t* scm
);

// Counterfactual gradient: d Y_x(u) / d x
// Enables gradient-based counterfactual optimization
causa_tensor_t* causa_counterfactual_gradient(
    causa_scm_t* scm,
    causa_counterfactual_t* cf,
    causa_tensor_t* wrt  // variable to differentiate wrt (usually intervention_val)
);

// Compute counterfactual with gradient tracking enabled
// Returns counterfactual result and builds computation graph for gradients
int causa_counterfactual_compute_with_grad(
    causa_scm_t* scm,
    causa_counterfactual_t* cf,
    causa_graph_t* grad_graph
);

// Gradient of counterfactual loss w.r.t. intervention value
causa_tensor_t* causa_counterfactual_grad_wrt_intervention(
    causa_scm_t* scm,
    causa_counterfactual_t* cf
);

// ===== Causal Discovery =====

causa_discovery_t* causa_discovery_new(
    causa_arena_t* arena,
    int32_t num_variables
);

void causa_discovery_free(causa_discovery_t* discovery);

// NOTEARS-based causal discovery from observational + interventional data
// Minimizes: L(W) + alpha * ||W||_1 + rho/2 * h(W)^2
// where h(W) = tr(exp(W⊙W)) - d (acyclicity constraint)
causa_discovery_t* causa_discover(
    causa_tensor_t* observational_data,   // [N, D] - observational samples
    causa_tensor_t* interventional_data,  // [M, D] - interventional samples (with do() markers)
    causa_tensor_t* intervention_targets, // [M] - which variable was intervened on (or -1)
    const causa_config_t* config
);

// Apply discovered structure to SCM (prune edges below threshold)
void causa_apply_discovery(
    causa_scm_t* scm,
    causa_discovery_t* discovery,
    float confidence_threshold
);

// Get discovered parents for variable
int32_t causa_get_parents(
    causa_discovery_t* discovery,
    int32_t var_idx,
    int32_t* out_parents,
    int32_t max_parents
);

// ===== Transportability / Domain Adaptation =====

// Transport causal effect from source to target domain
// Using selection diagrams (Pearl & Bareinboim)
// Returns P_target(Y | do(X)) estimated from source + target observations
causa_tensor_t* causa_transport(
    causa_scm_t* source_scm,
    causa_scm_t* target_scm,
    causa_tensor_t* source_intervention,  // do(X) in source
    causa_tensor_t* target_observations,  // P_target(V) observations
    causa_tensor_t* selection_vars        // Variables where domains differ
);

// ===== Memory Integration =====

// Consolidate what episodic memory has seen into causal laws in semantic
// memory.
//
// Episodic slots are expected to hold one observation each, with the key
// holding the observed variables. Those observations are gathered into a
// matrix, causal discovery is run over them, and the resulting parent sets
// and coefficients are written to semantic memory.
//
// Returns the number of laws written, or -1 on error.
int32_t causa_memory_consolidate_causal(
    causa_memory_t* memory,
    causa_scm_t* scm,
    const causa_config_t* config
);

// Causal consolidation: episodic patterns -> semantic causal laws
// 1. Extract frequent patterns from episodic memory
// 2. Run causal discovery on patterns
// 3. Write discovered causal laws to semantic memory
void causa_memory_causal_consolidate(
    causa_memory_t* memory,
    causa_scm_t* scm,
    causa_discovery_t* discovery
);

// Store counterfactual in episodic memory
causa_tensor_t* causa_memory_store_counterfactual(
    causa_memory_t* memory,
    causa_counterfactual_t* cf
);

// Retrieve relevant counterfactuals for query
causa_tensor_t* causa_memory_retrieve_counterfactuals(
    causa_memory_t* memory,
    causa_tensor_t* query,      // Y
    causa_tensor_t* context,    // X context
    int32_t k                    // top-k
);

// ===== Utility =====

// Print SCM structure
void causa_scm_print(const causa_scm_t* scm);

// Print intervention
void causa_intervention_print(const causa_intervention_t* intervention);

// Print counterfactual
void causa_counterfactual_print(const causa_counterfactual_t* cf);

// Print discovery result
void causa_discovery_print(const causa_discovery_t* discovery);

#ifdef __cplusplus
}
#endif

#endif