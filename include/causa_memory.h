#ifndef CAUSA_MEMORY_H
#define CAUSA_MEMORY_H

#include "causa_tensor.h"
#include "causa_graph.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CAUSA_MEMORY_MAX_LEVELS 4

// The largest number of slots a level may hold. A level has one slot
// struct for each of its slots, so the number decides how much memory is
// set aside before anything is stored. A capacity above this is refused.
#define CAUSA_MEMORY_MAX_SLOTS 1048576

typedef enum {
    CAUSA_MEMORY_WORKING = 0,    // Fast, small, volatile
    CAUSA_MEMORY_EPISODIC = 1,   // Medium, event-based
    CAUSA_MEMORY_SEMANTIC = 2,   // Slow, large, consolidated
    CAUSA_MEMORY_PROCEDURAL = 3  // Skills, implicit
} causa_memory_level_t;

typedef enum {
    CAUSA_CONSOLIDATE_SLOW = 0,   // Gradual consolidation
    CAUSA_CONSOLIDATE_FAST = 1,   // Rapid consolidation (sleep-like)
    CAUSA_CONSOLIDATE_REPLAY = 2  // Experience replay
} causa_consolidate_mode_t;

typedef struct causa_memory_s causa_memory_t;
typedef struct causa_memory_slot_s causa_memory_slot_t;
typedef struct causa_memory_config_s causa_memory_config_t;

// A counterfactual kept in memory: what would have happened, in what
// situation, and to which variable.
//
// The counterfactuals are kept apart from the episodes, because an episode
// is something that happened and a counterfactual is something that would
// have happened. Were they mixed, the discovery would learn from things
// that never occurred.
typedef struct {
    char query_name[CAUSA_MAX_NAME];
    causa_tensor_t* context;
    causa_tensor_t* result;
    float timestamp;
} causa_memory_counterfactual_t;

typedef void (*causa_memory_write_fn)(causa_memory_t* mem, causa_tensor_t* key, causa_tensor_t* value, causa_memory_level_t level);
typedef causa_tensor_t* (*causa_memory_read_fn)(causa_memory_t* mem, causa_tensor_t* key, causa_memory_level_t level);
typedef void (*causa_memory_consolidate_fn)(causa_memory_t* mem, causa_consolidate_mode_t mode);
typedef void (*causa_memory_forget_fn)(causa_memory_t* mem, causa_tensor_t* key, causa_memory_level_t level);

struct causa_memory_config_s {
    int64_t capacity[CAUSA_MEMORY_MAX_LEVELS];
    int64_t key_dim;
    int64_t value_dim;
    float working_decay_rate;
    float episodic_decay_rate;
    float consolidation_threshold;
    bool differentiable;
};
#define CAUSA_MEMORY_CONFIG_DEFAULTS { \
    .capacity = {128, 1024, 8192, 1024}, \
    .key_dim = 64, \
    .value_dim = 64, \
    .working_decay_rate = 0.1f, \
    .episodic_decay_rate = 0.01f, \
    .consolidation_threshold = 0.5f, \
    .differentiable = true \
}

causa_memory_t* causa_memory_new(
    causa_arena_t* arena,
    causa_backend_t* backend,
    const causa_memory_config_t* config
);

void causa_memory_free(causa_memory_t* mem);

// Write a key and a value into a level of memory.
//
// The key has to be as wide as the memory's keys are and the value as wide
// as its values are. A key or a value of another width is refused, because
// the slot it would go into is of a fixed width and the rest of it would
// hold whatever was there before.
//
// The value that was written is given back. It is the memory's own tensor,
// so it must not be changed by the caller.
causa_tensor_t* causa_memory_write(
    causa_memory_t* mem,
    causa_tensor_t* key,
    causa_tensor_t* value,
    causa_memory_level_t level
);

// The tensors a memory hands back are the memory's own.
//
// They are given back so that the caller does not have to copy them, and
// they must not be changed: a caller that writes to one changes what the
// memory holds. They stay valid as long as the memory does, and they are
// gone when it is freed.
//
// The tensors that a memory was given are copied into it, so the caller
// keeps its own and can change them afterwards.
causa_tensor_t* causa_memory_read(
    causa_memory_t* mem,
    causa_tensor_t* key,
    causa_memory_level_t level
);

// The k keys and values that are most like the given one, the nearest
// first.
//
// The tensors written to the out arrays are the memory's own, and must not
// be changed.
causa_tensor_t* causa_memory_read_topk(
    causa_memory_t* mem,
    causa_tensor_t* key,
    causa_memory_level_t level,
    int32_t k,
    causa_tensor_t** out_keys,
    causa_tensor_t** out_values
);

void causa_memory_consolidate(
    causa_memory_t* mem,
    causa_consolidate_mode_t mode
);

void causa_memory_forget(
    causa_memory_t* mem,
    causa_tensor_t* key,
    causa_memory_level_t level
);

void causa_memory_decay(causa_memory_t* mem, float dt);

int64_t causa_memory_size(const causa_memory_t* mem, causa_memory_level_t level);
int64_t causa_memory_capacity(const causa_memory_t* mem, causa_memory_level_t level);
float causa_memory_usage(const causa_memory_t* mem, causa_memory_level_t level);

typedef struct {
    causa_tensor_t* key;
    causa_tensor_t* value;
    float timestamp;
    float strength;
    int32_t access_count;
    causa_memory_level_t level;
} causa_memory_slot_info_t;

// The key and the value of a slot.
//
// They are the memory's own tensors, and must not be changed.
causa_memory_slot_info_t causa_memory_inspect(const causa_memory_t* mem, causa_memory_level_t level, int64_t index);

// The plasticity rules.
//
// The spike timing rule and the neurogenesis rule are declared but are not
// implemented: the first needs the time a spike happened and the second
// needs room to be made, and neither fits the model as it is. A rule of
// either kind does nothing.
typedef enum {
    CAUSA_PLASTICITY_HEBBIAN = 0,
    CAUSA_PLASTICITY_OJA = 1,
    CAUSA_PLASTICITY_BCM = 2,
    CAUSA_PLASTICITY_STDP = 3,
    CAUSA_PLASTICITY_EWC = 4,
    CAUSA_PLASTICITY_SYNAPTIC_SCALING = 5,
    CAUSA_PLASTICITY_NEUROGENESIS = 6,
    CAUSA_PLASTICITY_CUSTOM = 7
} causa_plasticity_type_t;

typedef struct causa_plasticity_rule_s causa_plasticity_rule_t;
typedef struct causa_plasticity_config_s causa_plasticity_config_t;

struct causa_plasticity_config_s {
    causa_plasticity_type_t type;
    float learning_rate;
    float decay_rate;
    float threshold;
    void* params;
    size_t params_size;
};

#define CAUSA_PLASTICITY_CONFIG_DEFAULTS { \
    .type = CAUSA_PLASTICITY_HEBBIAN, \
    .learning_rate = 0.01f, \
    .decay_rate = 0.0001f, \
    .threshold = 0.0f, \
    .params = NULL, \
    .params_size = 0 \
}

causa_plasticity_rule_t* causa_plasticity_rule_new(
    causa_arena_t* arena,
    const causa_plasticity_config_t* config
);

void causa_plasticity_apply(
    causa_plasticity_rule_t* rule,
    causa_tensor_t* pre,
    causa_tensor_t* post,
    causa_tensor_t* weight
);

void causa_plasticity_apply_batch(
    causa_plasticity_rule_t* rule,
    causa_tensor_t* pre_batch,
    causa_tensor_t* post_batch,
    causa_tensor_t* weight
);

void causa_memory_set_plasticity(
    causa_memory_t* mem,
    causa_memory_level_t level,
    causa_plasticity_rule_t* rule
);

causa_plasticity_rule_t* causa_memory_get_plasticity(
    const causa_memory_t* mem,
    causa_memory_level_t level
);

causa_graph_t* causa_memory_to_graph(
    causa_memory_t* mem,
    causa_arena_t* arena,
    causa_backend_t* backend
);

// ===== Introspection accessors (used by serialization) =====

// Retrieve the configuration of a memory structure
const causa_memory_config_t* causa_memory_config(const causa_memory_t* mem);

// The arena and backend the memory allocates from
causa_arena_t* causa_memory_arena(const causa_memory_t* mem);
causa_backend_t* causa_memory_backend(const causa_memory_t* mem);

// Current simulation time of the memory
float causa_memory_time(const causa_memory_t* mem);

// Override the simulation time
void causa_memory_set_time(causa_memory_t* mem, float t);

// Number of occupied slots at a level
int64_t causa_memory_occupied(const causa_memory_t* mem, causa_memory_level_t level);

// Read the key/value tensors of an occupied slot without touching access stats
bool causa_memory_slot_tensors(
    const causa_memory_t* mem,
    causa_memory_level_t level,
    int64_t index,
    causa_tensor_t** out_key,
    causa_tensor_t** out_value
);

// Write directly into a slot index, bypassing key lookup
// Used to restore serialized state faithfully
bool causa_memory_restore_slot(
    causa_memory_t* mem,
    causa_memory_level_t level,
    int64_t index,
    causa_tensor_t* key,
    causa_tensor_t* value,
    float strength,
    float timestamp,
    int32_t access_count
);

// ===== Structured episode storage =====

// Gather the keys of every occupied slot at a level into a matrix.
//
// Episodic slots are expected to hold one observation each: the key is the
// vector of observed variables. The result has shape [count, key_dim], which
// is the layout the causal discovery routine consumes.
//
// Returns the number of observations gathered, or -1 on error.
int64_t causa_memory_collect(
    const causa_memory_t* mem,
    causa_memory_level_t level,
    causa_tensor_t* out
);

// ===== Causal law storage =====

// A causal law is stored in semantic memory as a key/value pair.
//
// The key is a role vector over the variables: 2 marks the child the law
// describes, 1 marks each of its parents, and 0 marks the rest. The value
// holds the coefficient of each variable in the child's equation.
//
// The value has to be as wide as the key, because the coefficient of a
// parent is read from it. A memory whose values are narrower cannot hold
// the law, and the call gives nothing back.
//
// Storing a law returns the slot that holds it.
causa_tensor_t* causa_memory_store_causal_law(
    causa_memory_t* mem,
    int32_t child,
    const causa_tensor_t* parent_mask,
    const causa_tensor_t* coefficients
);

// Find every causal law recorded for a child in semantic memory.
//
// Writes the coefficient vectors to out_values and returns how many were
// found, or -1 on error.
int32_t causa_memory_find_causal_laws(
    const causa_memory_t* mem,
    int32_t child,
    causa_tensor_t** out_values,
    int32_t max_values
);

// Find every causal law recorded for a child, together with the role vector
// that says which variables are the parents.
//
// The role vector holds 2 at the child, 1 at each parent and 0 elsewhere, so
// the entries say both what the law is and which variables it relates.
//
// Writes to out_keys and out_values and returns how many were found, or -1
// on error.
int32_t causa_memory_find_causal_law_entries(
    const causa_memory_t* mem,
    int32_t child,
    causa_tensor_t** out_keys,
    causa_tensor_t** out_values,
    int32_t max_entries
);

// Count the causal laws recorded for a child in semantic memory.
int32_t causa_memory_count_causal_laws(
    const causa_memory_t* mem,
    int32_t child
);

// ===== Counterfactuals in memory =====

// Store a counterfactual: what would have happened, in a given situation.
//
// A counterfactual is what would have happened to a variable if something
// else had been done. It is kept apart from the episodes, because an
// episode is something that happened and a counterfactual is something
// that would have happened. Were they mixed, the discovery would learn
// from things that never occurred.
//
// The query_name says which variable the question was about, so that only
// the counterfactuals about that variable are returned later. The context
// is the situation the question was asked in, and the result is what would
// have happened.
//
// The counterfactuals are kept in the order they were added, up to the
// capacity of the episodic level, and the oldest are dropped first.
//
// Returns the number of counterfactuals kept, or -1 on error.
int32_t causa_memory_keep_counterfactual(
    causa_memory_t* mem,
    const char* query_name,
    const causa_tensor_t* context,
    const causa_tensor_t* result
);

// The counterfactuals about one variable, in the situations most like the
// given one.
//
// The distance between two situations is the straight-line distance
// between them, so it is in the units of the situation.
//
// Writes the results to out_results and returns how many were found, or
// -1 on error.
//
// The results are the memory's own tensors, and must not be changed.
int32_t causa_memory_find_counterfactuals(
    const causa_memory_t* mem,
    const char* query_name,
    const causa_tensor_t* context,
    int32_t max_results,
    causa_tensor_t** out_results
);

// ===== Reasoning over memory =====

// Predict the variables that were not observed.
//
// The observation holds one value per variable. A NaN marks a variable that
// was not observed. The prediction is built from two places:
//
//   - the causal laws in semantic memory give the structure, so an
//     unobserved variable is predicted from its parents;
//   - the episodes in episodic memory give the residual, because a law is
//     an average over the episodes and what actually happened differs from
//     that average.
//
// The law provides the structure and the episodes provide the offset.
//
// The prediction is written to out, which needs room for one value per
// variable. Observed variables are copied through unchanged.
//
// Returns the number of variables that were predicted, or -1 on error. A
// variable whose value could not be worked out, because the laws are in a
// cycle, is reported as an error rather than as a value that is not a
// number.
int32_t causa_memory_predict(
    causa_memory_t* mem,
    const causa_tensor_t* observation,
    causa_tensor_t* out
);

// Predict a single variable from the laws and the episodes.
//
// The observation holds one value per variable, with NaN for the variables
// that were not observed. The prediction of the given variable is written
// to out.
//
// Returns 0 on success, or -1 on error.
int32_t causa_memory_predict_variable(
    causa_memory_t* mem,
    const causa_tensor_t* observation,
    int32_t variable,
    causa_tensor_t* out
);

// ===== Procedural memory =====

// A procedure is a policy: a trigger and an action.
//
// The trigger is the state the procedure applies to, and the action is what
// to do there. The two are held in procedural memory as a key and a value,
// like everything else.
//
// The procedural level is where the memory stops describing the world and
// starts acting on it. The semantic level holds the laws, which say what
// causes what; a procedure says what to do in a state of the world.

// Store a procedure.
//
// The trigger and the action are copied, so the caller keeps its own
// tensors. Returns the value that holds the action, or NULL on error.
causa_tensor_t* causa_memory_store_procedure(
    causa_memory_t* mem,
    const causa_tensor_t* trigger,
    const causa_tensor_t* action
);

// Find the procedures that apply to a state, the nearest first.
//
// The distance between two states is the straight-line distance between
// them, so it is in the units of the state.
//
// Writes the triggers and the actions to the out arrays and returns how
// many were found, or -1 on error.
//
// The tensors written to the out arrays are the memory's own, and must not
// be changed.
int32_t causa_memory_find_procedures(
    const causa_memory_t* mem,
    const causa_tensor_t* state,
    causa_tensor_t** out_triggers,
    causa_tensor_t** out_actions,
    int32_t max_procedures
);

// Learn procedures from the episodes.
//
// Each episode is a state and what came of it. Episodes that resemble each
// other are grouped into one procedure: the trigger is the centre of the
// group and the action is the average of what those episodes led to.
//
// An episode joins the nearest procedure when it falls within the radius of
// that procedure's centre. Otherwise it starts a procedure of its own.
//
// Returns the number of procedures in procedural memory afterwards, or -1 on
// error.
int32_t causa_memory_learn_procedures(
    causa_memory_t* mem,
    float radius
);

// ===== Acting on what is known =====

// Work out what to do to reach a goal.
//
// The state says where things stand and the goal says what is wanted. The
// goal holds the wanted value of each variable to be changed, and NaN for
// the rest, in the same way the observation marks what was not observed.
//
// The laws say which variables cause the goal and how much, so the
// intervention that reaches the goal follows from them. The strongest cause
// is the one that is acted on, because it moves the goal the most for the
// least change.
//
// Several goals can be wanted at once. They are solved together, so that a
// change made for one goal does not undo another. Where the goals cannot
// all be reached, the change that fits them best is given.
//
// The intervention is written to out as a change to each variable. A
// variable that does not need to change is left at zero.
//
// Writes the number of variables the intervention acts on to out_count. A
// count of zero means the goal cannot be reached by changing anything.
//
// Returns 0 on success, or -1 on error.
int32_t causa_memory_plan_intervention(
    causa_memory_t* mem,
    const causa_tensor_t* state,
    const causa_tensor_t* goal,
    causa_tensor_t* out,
    int32_t* out_count
);

#ifdef __cplusplus
}
#endif

#endif
