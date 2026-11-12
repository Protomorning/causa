#include "causa_memory.h"
#include "causa_arena.h"
#include "causa_tensor.h"
#include "causa_graph.h"
#include "causa_linalg.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

struct causa_memory_slot_s {
    causa_tensor_t* key;
    causa_tensor_t* value;
    // The weight the plasticity rule works on. It is kept apart from the
    // key, because the key is what the memory is looked up by.
    causa_tensor_t* weight;
    float timestamp;
    float strength;
    int32_t access_count;
    // How many episodes a procedure stands for. It is what the running
    // mean needs to fold one more episode into the procedure.
    int32_t group_count;
    bool occupied;
};

struct causa_memory_s {
    causa_arena_t* arena;
    causa_backend_t* backend;
    causa_memory_config_t config;

    causa_memory_slot_t** slots[CAUSA_MEMORY_MAX_LEVELS];
    int64_t num_slots[CAUSA_MEMORY_MAX_LEVELS];
    int64_t next_slot[CAUSA_MEMORY_MAX_LEVELS];

    // The counterfactuals that were computed and kept.
    //
    // They are kept here rather than in a level, because a level holds what
    // happened and a counterfactual holds what would have happened. Were
    // they in the same place, the discovery would learn from things that
    // never occurred.
    causa_memory_counterfactual_t* counterfactuals;
    int32_t num_counterfactuals;
    int32_t capacity_counterfactuals;

    float global_time;
    float last_consolidation_time;

    causa_plasticity_rule_t* plasticity_rules[CAUSA_MEMORY_MAX_LEVELS];
};

static bool memory_init_level(causa_memory_t* mem, causa_memory_level_t level) {
    int64_t cap = mem->config.capacity[level];

    // A capacity that is not a count of slots is refused. A negative one
    // would ask for memory that cannot be asked for, and a very large one
    // would ask for more than there is.
    if (cap < 0 || cap > CAUSA_MEMORY_MAX_SLOTS) return false;

    mem->slots[level] = (causa_memory_slot_t**)causa_arena_alloc(mem->arena, sizeof(causa_memory_slot_t*) * (cap ? cap : 1), CAUSA_ALIGNOF(causa_memory_slot_t*));
    if (!mem->slots[level]) return false;

    for (int64_t i = 0; i < cap; i++) {
        mem->slots[level][i] = (causa_memory_slot_t*)causa_arena_alloc(mem->arena, sizeof(causa_memory_slot_t), CAUSA_ALIGNOF(causa_memory_slot_t));
        if (!mem->slots[level][i]) return false;
        mem->slots[level][i]->occupied = false;
        mem->slots[level][i]->key = NULL;
        mem->slots[level][i]->value = NULL;
        mem->slots[level][i]->weight = NULL;
        mem->slots[level][i]->timestamp = 0.0f;
        mem->slots[level][i]->strength = 0.0f;
        mem->slots[level][i]->access_count = 0;
        mem->slots[level][i]->group_count = 0;
    }
    mem->num_slots[level] = cap;
    mem->next_slot[level] = 0;
    return true;
}

causa_memory_t* causa_memory_new(
    causa_arena_t* arena,
    causa_backend_t* backend,
    const causa_memory_config_t* config
) {
    if (!arena || !backend) return NULL;

    causa_memory_config_t cfg = CAUSA_MEMORY_CONFIG_DEFAULTS;
    if (config) cfg = *config;

    causa_memory_t* mem = (causa_memory_t*)causa_arena_alloc(arena, sizeof(causa_memory_t), CAUSA_ALIGNOF(causa_memory_t));
    if (!mem) return NULL;

    mem->arena = arena;
    mem->backend = backend;
    mem->config = cfg;
    mem->global_time = 0.0f;
    mem->last_consolidation_time = 0.0f;

    for (int i = 0; i < CAUSA_MEMORY_MAX_LEVELS; i++) {
        mem->slots[i] = NULL;
        mem->num_slots[i] = 0;
        mem->next_slot[i] = 0;
        mem->plasticity_rules[i] = NULL;
    }

    for (int i = 0; i < CAUSA_MEMORY_MAX_LEVELS; i++) {
        if (!memory_init_level(mem, (causa_memory_level_t)i)) {
            return NULL;
        }
    }

    // The counterfactuals are kept apart from the levels, because they are
    // what would have happened rather than what did. They are given the room
    // of the episodic level, which is where experience is kept.
    mem->counterfactuals = NULL;
    mem->capacity_counterfactuals = (int32_t)cfg.capacity[CAUSA_MEMORY_EPISODIC];
    if (mem->capacity_counterfactuals > 0) {
        mem->counterfactuals = (causa_memory_counterfactual_t*)causa_arena_alloc(
            arena,
            sizeof(causa_memory_counterfactual_t) * (size_t)mem->capacity_counterfactuals,
            CAUSA_ALIGNOF(causa_memory_counterfactual_t));
        if (!mem->counterfactuals) {
            mem->capacity_counterfactuals = 0;
            return NULL;
        }
    }
    mem->num_counterfactuals = 0;

    return mem;
}

void causa_memory_free(causa_memory_t* mem) {
    (void)mem;
}

static int64_t memory_find_slot(causa_memory_t* mem, causa_memory_level_t level, const causa_tensor_t* key) {
    if (!mem || level >= CAUSA_MEMORY_MAX_LEVELS) return -1;
    if (!key) return -1;

    int64_t cap = mem->config.capacity[level];
    int64_t key_dim = mem->config.key_dim;

    // The key is read up to its width, so it has to be that wide. A key
    // that is shorter would be read past its end.
    if (causa_tensor_numel(key) < key_dim) return -1;

    float* key_data = (float*)key->data;

    for (int64_t i = 0; i < cap; i++) {
        causa_memory_slot_t* slot = mem->slots[level][i];
        if (slot->occupied && slot->key) {
            float* slot_key = (float*)slot->key->data;
            bool match = true;
            for (int64_t j = 0; j < key_dim; j++) {
                if (fabsf(slot_key[j] - key_data[j]) > 1e-6) {
                    match = false;
                    break;
                }
            }
            if (match) return i;
        }
    }
    return -1;
}

static int64_t memory_find_empty_slot(causa_memory_t* mem, causa_memory_level_t level) {
    int64_t cap = mem->config.capacity[level];
    for (int64_t i = 0; i < cap; i++) {
        if (!mem->slots[level][i]->occupied) return i;
    }
    return -1;
}

static int64_t memory_find_weakest_slot(causa_memory_t* mem, causa_memory_level_t level) {
    int64_t cap = mem->config.capacity[level];
    int64_t weakest = -1;
    float min_strength = FLT_MAX;

    for (int64_t i = 0; i < cap; i++) {
        causa_memory_slot_t* slot = mem->slots[level][i];
        if (slot->occupied && slot->strength < min_strength) {
            min_strength = slot->strength;
            weakest = i;
        }
    }
    return weakest;
}

static float memory_compute_similarity(causa_memory_t* mem, causa_tensor_t* key, causa_tensor_t* slot_key) {
    if (!key || !slot_key) return 0.0f;

    int64_t dim = mem->config.key_dim;

    // Both are read up to the width of a key, so both have to be that wide.
    if (causa_tensor_numel(key) < dim || causa_tensor_numel(slot_key) < dim) return 0.0f;

    float* key_data = (float*)key->data;
    float* slot_data = (float*)slot_key->data;

    float dot = 0.0f, norm_key = 0.0f, norm_slot = 0.0f;
    for (int64_t i = 0; i < dim; i++) {
        dot += key_data[i] * slot_data[i];
        norm_key += key_data[i] * key_data[i];
        norm_slot += slot_data[i] * slot_data[i];
    }

    norm_key = sqrtf(norm_key + 1e-8f);
    norm_slot = sqrtf(norm_slot + 1e-8f);
    return dot / (norm_key * norm_slot);
}

causa_tensor_t* causa_memory_write(
    causa_memory_t* mem,
    causa_tensor_t* key,
    causa_tensor_t* value,
    causa_memory_level_t level
) {
    if (!mem || !key || !value || level >= CAUSA_MEMORY_MAX_LEVELS) return NULL;

    // The key and the value are copied into slots of a fixed width, so they
    // have to be that wide. A key that is narrower would leave the rest of
    // the slot holding whatever the arena had, and the memory would hold a
    // value that was never written.
    if (causa_tensor_numel(key) != mem->config.key_dim) return NULL;
    if (causa_tensor_numel(value) != mem->config.value_dim) return NULL;

    int64_t existing = memory_find_slot(mem, level, key);
    causa_memory_slot_t* slot;

    if (existing >= 0) {
        slot = mem->slots[level][existing];
    } else {
        int64_t empty = memory_find_empty_slot(mem, level);
        if (empty >= 0) {
            slot = mem->slots[level][empty];
        } else {
            int64_t weakest = memory_find_weakest_slot(mem, level);
            if (weakest >= 0) {
                slot = mem->slots[level][weakest];
            } else {
                return NULL;
            }
        }
    }

    if (!slot->key) {
        causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        opts.dtype = CAUSA_DTYPE_F32;
        opts.device = CAUSA_DEVICE_CPU;
        opts.requires_grad = mem->config.differentiable;
        int64_t key_shape[] = {mem->config.key_dim};
        int64_t val_shape[] = {mem->config.value_dim};
        slot->key = causa_tensor_new(mem->arena, mem->backend, key_shape, 1, &opts);
        slot->value = causa_tensor_new(mem->arena, mem->backend, val_shape, 1, &opts);
    }

    causa_tensor_copy(slot->key, key);
    causa_tensor_copy(slot->value, value);
    slot->timestamp = mem->global_time;
    slot->strength = 1.0f;
    slot->access_count = 0;
    slot->occupied = true;

    // The rule is applied to a weight of its own. The key is what the
    // memory is looked up by, so a rule that changed it would make the
    // next write of the same key miss and fill the memory with copies of
    // the same thing.
    if (mem->plasticity_rules[level]) {
        if (!slot->weight) {
            causa_tensor_options_t wopts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
            wopts.dtype = CAUSA_DTYPE_F32;
            wopts.device = CAUSA_DEVICE_CPU;
            wopts.requires_grad = mem->config.differentiable;
            int64_t weight_shape[] = {mem->config.key_dim};
            slot->weight = causa_tensor_new(mem->arena, mem->backend, weight_shape, 1, &wopts);
        }
        if (slot->weight) {
            causa_plasticity_apply(mem->plasticity_rules[level], key, value, slot->weight);
        }
    }

    return slot->value;
}

causa_tensor_t* causa_memory_read(
    causa_memory_t* mem,
    causa_tensor_t* key,
    causa_memory_level_t level
) {
    if (!mem || !key || level >= CAUSA_MEMORY_MAX_LEVELS) return NULL;

    int64_t existing = memory_find_slot(mem, level, key);
    if (existing >= 0) {
        causa_memory_slot_t* slot = mem->slots[level][existing];
        slot->access_count++;
        slot->strength = fminf(1.0f, slot->strength + 0.1f);
        return slot->value;
    }

    return NULL;
}

causa_tensor_t* causa_memory_read_topk(
    causa_memory_t* mem,
    causa_tensor_t* key,
    causa_memory_level_t level,
    int32_t k,
    causa_tensor_t** out_keys,
    causa_tensor_t** out_values
) {
    if (!mem || !key || level >= CAUSA_MEMORY_MAX_LEVELS || k <= 0) return NULL;

    int64_t cap = mem->config.capacity[level];
    int64_t valid_count = 0;

    // The similarities are held in a buffer of this call's own. A buffer
    // shared by the whole memory would be written by two calls at once,
    // and the two would mix their numbers together.
    float* sims = (float*)malloc((size_t)(cap > 0 ? cap : 1) * sizeof(float));
    if (!sims) return NULL;

    for (int64_t i = 0; i < cap; i++) {
        causa_memory_slot_t* slot = mem->slots[level][i];
        if (slot->occupied && slot->key) {
            float sim = memory_compute_similarity(mem, key, slot->key);
            sims[valid_count] = sim;
            valid_count++;
        }
    }

    if (valid_count == 0) { free(sims); return NULL; }

    int32_t actual_k = (k < valid_count) ? k : (int32_t)valid_count;

    int64_t* indices = (int64_t*)malloc((size_t)valid_count * sizeof(int64_t));
    if (!indices) { free(sims); return NULL; }
    for (int64_t i = 0; i < valid_count; i++) indices[i] = i;

    for (int32_t i = 0; i < actual_k; i++) {
        int64_t best_idx = i;
        for (int64_t j = i + 1; j < valid_count; j++) {
            if (sims[indices[j]] > sims[indices[best_idx]]) {
                best_idx = j;
            }
        }
        int64_t temp = indices[i];
        indices[i] = indices[best_idx];
        indices[best_idx] = temp;
    }

    int64_t slot_idx = 0;
    for (int64_t i = 0; i < cap && slot_idx < valid_count; i++) {
        if (mem->slots[level][i]->occupied) {
            if (slot_idx == indices[0]) {
                for (int32_t k_idx = 0; k_idx < actual_k; k_idx++) {
                    int64_t idx = indices[k_idx];
                    int64_t count = 0;
                    for (int64_t s = 0; s < cap; s++) {
                        if (mem->slots[level][s]->occupied) {
                            if (count == idx) {
                                if (out_keys) out_keys[k_idx] = mem->slots[level][s]->key;
                                if (out_values) out_values[k_idx] = mem->slots[level][s]->value;
                                mem->slots[level][s]->access_count++;
                                mem->slots[level][s]->strength = fminf(1.0f, mem->slots[level][s]->strength + 0.05f);
                                break;
                            }
                            count++;
                        }
                    }
                }
                break;
            }
            slot_idx++;
        }
    }

    free(indices);
    free(sims);

    return out_values ? out_values[0] : NULL;
}

void causa_memory_consolidate(
    causa_memory_t* mem,
    causa_consolidate_mode_t mode
) {
    if (!mem) return;

    float current_time = mem->global_time;

    for (int level = 0; level < CAUSA_MEMORY_MAX_LEVELS - 1; level++) {
        int64_t cap = mem->config.capacity[level];

        for (int64_t i = 0; i < cap; i++) {
            causa_memory_slot_t* slot = mem->slots[level][i];
            if (!slot->occupied) continue;

            float age = current_time - slot->timestamp;
            bool should_consolidate = false;

            switch (mode) {
                case CAUSA_CONSOLIDATE_SLOW:
                    should_consolidate = (slot->strength > mem->config.consolidation_threshold &&
                                         slot->access_count > 2 && age > 100.0f);
                    break;
                case CAUSA_CONSOLIDATE_FAST:
                    should_consolidate = (slot->strength > mem->config.consolidation_threshold * 0.5f);
                    break;
                case CAUSA_CONSOLIDATE_REPLAY:
                    should_consolidate = (slot->access_count > 0);
                    break;
            }

            if (should_consolidate) {
                int64_t empty = memory_find_empty_slot(mem, level + 1);
                if (empty >= 0) {
                    causa_memory_slot_t* target = mem->slots[level + 1][empty];
                    if (!target->key) {
                        causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
                        opts.dtype = CAUSA_DTYPE_F32;
                        opts.device = CAUSA_DEVICE_CPU;
                        opts.requires_grad = mem->config.differentiable;
                        int64_t key_shape[] = {mem->config.key_dim};
                        int64_t val_shape[] = {mem->config.value_dim};
                        target->key = causa_tensor_new(mem->arena, mem->backend, key_shape, 1, &opts);
                        target->value = causa_tensor_new(mem->arena, mem->backend, val_shape, 1, &opts);
                    }
                    causa_tensor_copy(target->key, slot->key);
                    causa_tensor_copy(target->value, slot->value);
                    target->timestamp = slot->timestamp;
                    target->strength = slot->strength * 0.9f;
                    target->access_count = slot->access_count;
                    target->occupied = true;

                    if (mode == CAUSA_CONSOLIDATE_FAST) {
                        slot->strength *= 0.5f;
                    }
                }
            }
        }
    }

    mem->last_consolidation_time = current_time;
}

void causa_memory_forget(
    causa_memory_t* mem,
    causa_tensor_t* key,
    causa_memory_level_t level
) {
    if (!mem || !key || level >= CAUSA_MEMORY_MAX_LEVELS) return;

    int64_t existing = memory_find_slot(mem, level, key);
    if (existing >= 0) {
        causa_memory_slot_t* slot = mem->slots[level][existing];
        slot->occupied = false;
        slot->strength = 0.0f;
    }
}

void causa_memory_decay(causa_memory_t* mem, float dt) {
    if (!mem) return;

    mem->global_time += dt;

    for (int level = 0; level < CAUSA_MEMORY_MAX_LEVELS; level++) {
        float decay_rate = (level == CAUSA_MEMORY_WORKING) ? mem->config.working_decay_rate :
                          (level == CAUSA_MEMORY_EPISODIC) ? mem->config.episodic_decay_rate : 0.0f;

        if (decay_rate > 0.0f) {
            int64_t cap = mem->config.capacity[level];
            for (int64_t i = 0; i < cap; i++) {
                causa_memory_slot_t* slot = mem->slots[level][i];
                if (slot->occupied) {
                    slot->strength *= expf(-decay_rate * dt);
                    if (slot->strength < 0.01f) {
                        slot->occupied = false;
                        slot->strength = 0.0f;
                    }
                }
            }
        }
    }
}

int64_t causa_memory_size(const causa_memory_t* mem, causa_memory_level_t level) {
    if (!mem || level >= CAUSA_MEMORY_MAX_LEVELS) return 0;

    int64_t count = 0;
    int64_t cap = mem->config.capacity[level];
    for (int64_t i = 0; i < cap; i++) {
        if (mem->slots[level][i]->occupied) count++;
    }
    return count;
}

int64_t causa_memory_capacity(const causa_memory_t* mem, causa_memory_level_t level) {
    if (!mem || level >= CAUSA_MEMORY_MAX_LEVELS) return 0;
    return mem->config.capacity[level];
}

float causa_memory_usage(const causa_memory_t* mem, causa_memory_level_t level) {
    if (!mem || level >= CAUSA_MEMORY_MAX_LEVELS) return 0.0f;
    int64_t cap = mem->config.capacity[level];
    if (cap == 0) return 0.0f;
    return (float)causa_memory_size(mem, level) / (float)cap;
}

causa_memory_slot_info_t causa_memory_inspect(const causa_memory_t* mem, causa_memory_level_t level, int64_t index) {
    causa_memory_slot_info_t info = {0};
    if (!mem || level >= CAUSA_MEMORY_MAX_LEVELS) return info;

    int64_t cap = mem->config.capacity[level];
    if (index < 0 || index >= cap) return info;

    causa_memory_slot_t* slot = mem->slots[level][index];
    if (!slot->occupied) return info;

    info.key = slot->key;
    info.value = slot->value;
    info.timestamp = slot->timestamp;
    info.strength = slot->strength;
    info.access_count = slot->access_count;
    info.level = level;

    return info;
}

struct causa_plasticity_rule_s {
    causa_plasticity_type_t type;
    float learning_rate;
    float decay_rate;
    float threshold;
    void* params;
    size_t params_size;
};

causa_plasticity_rule_t* causa_plasticity_rule_new(
    causa_arena_t* arena,
    const causa_plasticity_config_t* config
) {
    if (!arena) return NULL;

    causa_plasticity_config_t cfg = CAUSA_PLASTICITY_CONFIG_DEFAULTS;
    if (config) cfg = *config;

    causa_plasticity_rule_t* rule = (causa_plasticity_rule_t*)causa_arena_alloc(arena, sizeof(causa_plasticity_rule_t), CAUSA_ALIGNOF(causa_plasticity_rule_t));
    if (!rule) return NULL;

    rule->type = cfg.type;
    rule->learning_rate = cfg.learning_rate;
    rule->decay_rate = cfg.decay_rate;
    rule->threshold = cfg.threshold;
    rule->params = cfg.params;
    rule->params_size = cfg.params_size;

    return rule;
}

static void plasticity_hebbian(causa_plasticity_rule_t* rule, causa_tensor_t* pre, causa_tensor_t* post, causa_tensor_t* weight) {
    int64_t n = causa_tensor_numel(pre);

    // All three are read and written up to the length of the first one, so
    // all three have to be that long.
    if (causa_tensor_numel(post) < n || causa_tensor_numel(weight) < n) return;

    float* pre_data = (float*)pre->data;
    float* post_data = (float*)post->data;
    float* weight_data = (float*)weight->data;

    for (int64_t i = 0; i < n; i++) {
        weight_data[i] += rule->learning_rate * pre_data[i] * post_data[i];
        weight_data[i] -= rule->decay_rate * weight_data[i];
    }
}

static void plasticity_oja(causa_plasticity_rule_t* rule, causa_tensor_t* pre, causa_tensor_t* post, causa_tensor_t* weight) {
    int64_t n = causa_tensor_numel(pre);

    // All three are read and written up to the length of the first one, so
    // all three have to be that long.
    if (causa_tensor_numel(post) < n || causa_tensor_numel(weight) < n) return;

    float* pre_data = (float*)pre->data;
    float* post_data = (float*)post->data;
    float* weight_data = (float*)weight->data;

    // The output of the weights is the same for every element, so it is
    // worked out once and not once per element.
    float y = 0.0f;
    for (int64_t j = 0; j < n; j++) {
        y += weight_data[j] * pre_data[j];
    }

    for (int64_t i = 0; i < n; i++) {
        weight_data[i] += rule->learning_rate * (post_data[i] * pre_data[i] - y * y * weight_data[i]);
    }
}

static void plasticity_bcm(causa_plasticity_rule_t* rule, causa_tensor_t* pre, causa_tensor_t* post, causa_tensor_t* weight) {
    int64_t n = causa_tensor_numel(pre);

    // All three are read and written up to the length of the first one, so
    // all three have to be that long.
    if (causa_tensor_numel(post) < n || causa_tensor_numel(weight) < n) return;

    float* pre_data = (float*)pre->data;
    float* post_data = (float*)post->data;
    float* weight_data = (float*)weight->data;

    for (int64_t i = 0; i < n; i++) {
        float y = post_data[i];
        float theta = rule->threshold;
        float phi = y * (y - theta);
        weight_data[i] += rule->learning_rate * phi * pre_data[i];
        weight_data[i] -= rule->decay_rate * weight_data[i];
    }
}

static void plasticity_ewc(causa_plasticity_rule_t* rule, causa_tensor_t* pre, causa_tensor_t* post, causa_tensor_t* weight) {
    // The Fisher information and the previous weights are kept in the
    // rule's parameters. The pre and post activities are not used.
    (void)pre; (void)post;
    int64_t n = causa_tensor_numel(weight);
    float* weight_data = (float*)weight->data;

    // The parameters hold the Fisher information and then the previous
    // weights, so there have to be two of each. Without them the rule
    // would read past the end of what it was given.
    bool have_params = rule->params && rule->params_size >= (size_t)(2 * n * sizeof(float));

    for (int64_t i = 0; i < n; i++) {
        float fisher = have_params ? ((const float*)rule->params)[i] : 1.0f;
        float target = have_params ? ((const float*)rule->params)[n + i] : weight_data[i];
        weight_data[i] -= rule->learning_rate * fisher * (weight_data[i] - target);
    }
}

static void plasticity_synaptic_scaling(causa_plasticity_rule_t* rule, causa_tensor_t* pre, causa_tensor_t* post, causa_tensor_t* weight) {
    // The rule scales the weights so that their mean is the target. The
    // activities are not used.
    (void)pre; (void)post;
    int64_t n = causa_tensor_numel(weight);
    if (n <= 0) return;

    float* weight_data = (float*)weight->data;

    float mean_weight = 0.0f;
    for (int64_t i = 0; i < n; i++) mean_weight += weight_data[i];
    mean_weight /= (float)n;

    float target_mean = rule->threshold > 0 ? rule->threshold : 1.0f;

    // The mean of the weights may be negative, and a scale that is computed
    // from it would then turn every weight around. The sign of the weights
    // is kept, and only their size is changed.
    float scale = 1.0f;
    if (fabsf(mean_weight) > 1e-8f) {
        scale = target_mean / fabsf(mean_weight);
    }

    for (int64_t i = 0; i < n; i++) {
        weight_data[i] *= scale;
    }
}

void causa_plasticity_apply(
    causa_plasticity_rule_t* rule,
    causa_tensor_t* pre,
    causa_tensor_t* post,
    causa_tensor_t* weight
) {
    if (!rule || !pre || !post || !weight) return;

    switch (rule->type) {
        case CAUSA_PLASTICITY_HEBBIAN:
            plasticity_hebbian(rule, pre, post, weight);
            break;
        case CAUSA_PLASTICITY_OJA:
            plasticity_oja(rule, pre, post, weight);
            break;
        case CAUSA_PLASTICITY_BCM:
            plasticity_bcm(rule, pre, post, weight);
            break;
        case CAUSA_PLASTICITY_EWC:
            plasticity_ewc(rule, pre, post, weight);
            break;
        case CAUSA_PLASTICITY_SYNAPTIC_SCALING:
            plasticity_synaptic_scaling(rule, pre, post, weight);
            break;
        default:
            // The spike timing rule and the neurogenesis rule are declared
            // but have no meaning in this model: the first needs the time
            // a spike happened and the second needs room to be made. They
            // are listed in the header so that they are not mistaken for
            // something that works.
            break;
    }
}

void causa_plasticity_apply_batch(
    causa_plasticity_rule_t* rule,
    causa_tensor_t* pre_batch,
    causa_tensor_t* post_batch,
    causa_tensor_t* weight
) {
    if (!rule || !pre_batch || !post_batch || !weight) return;

    // The batch is a matrix, and its size divides the count of elements to
    // find the width of one row. A batch with no rows has nothing to divide
    // by.
    if (pre_batch->ndim < 1 || post_batch->ndim < 1) return;

    int64_t batch_size = pre_batch->shape[0];
    if (batch_size <= 0) return;

    int64_t pre_total = causa_tensor_numel(pre_batch);
    int64_t post_total = causa_tensor_numel(post_batch);
    if (pre_total % batch_size != 0 || post_total % batch_size != 0) return;

    int64_t pre_dim = pre_total / batch_size;
    int64_t post_dim = post_total / batch_size;

    for (int64_t b = 0; b < batch_size; b++) {
        causa_tensor_t* pre = causa_tensor_slice(pre_batch, (int64_t[]){b, 0}, (int64_t[]){b+1, pre_dim}, NULL, 2);
        causa_tensor_t* post = causa_tensor_slice(post_batch, (int64_t[]){b, 0}, (int64_t[]){b+1, post_dim}, NULL, 2);
        if (!pre || !post) {
            // The views are made in the arena, which is never given back,
            // so they are released as they are used.
            if (pre) causa_tensor_release(pre);
            if (post) causa_tensor_release(post);
            return;
        }
        causa_plasticity_apply(rule, pre, post, weight);
        causa_tensor_release(pre);
        causa_tensor_release(post);
    }
}

void causa_memory_set_plasticity(
    causa_memory_t* mem,
    causa_memory_level_t level,
    causa_plasticity_rule_t* rule
) {
    if (mem && level < CAUSA_MEMORY_MAX_LEVELS) {
        mem->plasticity_rules[level] = rule;
    }
}

causa_plasticity_rule_t* causa_memory_get_plasticity(
    const causa_memory_t* mem,
    causa_memory_level_t level
) {
    if (mem && level < CAUSA_MEMORY_MAX_LEVELS) {
        return mem->plasticity_rules[level];
    }
    return NULL;
}

causa_graph_t* causa_memory_to_graph(
    causa_memory_t* mem,
    causa_arena_t* arena,
    causa_backend_t* backend
) {
    if (!mem || !arena || !backend) return NULL;

    causa_graph_t* graph = causa_graph_new(arena, backend);

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    opts.requires_grad = true;

    int64_t key_shape[] = {mem->config.key_dim};
    int64_t val_shape[] = {mem->config.value_dim};

    causa_tensor_t* key_input = causa_graph_new_tensor(graph, key_shape, 1, &opts);
    causa_tensor_t* value_input = causa_graph_new_tensor(graph, val_shape, 1, &opts);

    causa_tensor_t* working_read = causa_graph_new_tensor(graph, val_shape, 1, &opts);
    causa_tensor_t* episodic_read = causa_graph_new_tensor(graph, val_shape, 1, &opts);
    causa_tensor_t* semantic_read = causa_graph_new_tensor(graph, val_shape, 1, &opts);

    causa_tensor_t* consolidated = causa_graph_new_tensor(graph, val_shape, 1, &opts);

    causa_tensor_t* output = causa_graph_new_tensor(graph, val_shape, 1, &opts);

    causa_graph_add_op(graph, CAUSA_OP_CUSTOM, "memory_read_working",
        (causa_tensor_t*[]){key_input}, 1,
        (causa_tensor_t*[]){working_read}, 1,
        NULL, NULL, mem, sizeof(causa_memory_t*));

    causa_graph_add_op(graph, CAUSA_OP_CUSTOM, "memory_read_episodic",
        (causa_tensor_t*[]){key_input}, 1,
        (causa_tensor_t*[]){episodic_read}, 1,
        NULL, NULL, mem, sizeof(causa_memory_t*));

    causa_graph_add_op(graph, CAUSA_OP_CUSTOM, "memory_read_semantic",
        (causa_tensor_t*[]){key_input}, 1,
        (causa_tensor_t*[]){semantic_read}, 1,
        NULL, NULL, mem, sizeof(causa_memory_t*));

    causa_graph_add_op(graph, CAUSA_OP_ADD, "combine_reads",
        (causa_tensor_t*[]){working_read, episodic_read, semantic_read}, 3,
        (causa_tensor_t*[]){consolidated}, 1,
        NULL, NULL, NULL, 0);

    causa_graph_add_op(graph, CAUSA_OP_ADD, "output_with_input",
        (causa_tensor_t*[]){consolidated, value_input}, 2,
        (causa_tensor_t*[]){output}, 1,
        NULL, NULL, NULL, 0);

    causa_graph_build(graph);
    return graph;
}

// ===== Introspection accessors =====

const causa_memory_config_t* causa_memory_config(const causa_memory_t* mem) {
    return mem ? &mem->config : NULL;
}

causa_arena_t* causa_memory_arena(const causa_memory_t* mem) {
    return mem ? mem->arena : NULL;
}

causa_backend_t* causa_memory_backend(const causa_memory_t* mem) {
    return mem ? mem->backend : NULL;
}

float causa_memory_time(const causa_memory_t* mem) {
    return mem ? mem->global_time : 0.0f;
}

void causa_memory_set_time(causa_memory_t* mem, float t) {
    if (mem) mem->global_time = t;
}

int64_t causa_memory_occupied(const causa_memory_t* mem, causa_memory_level_t level) {
    return causa_memory_size(mem, level);
}

bool causa_memory_slot_tensors(
    const causa_memory_t* mem,
    causa_memory_level_t level,
    int64_t index,
    causa_tensor_t** out_key,
    causa_tensor_t** out_value
) {
    if (!mem || level >= CAUSA_MEMORY_MAX_LEVELS) return false;
    int64_t cap = mem->config.capacity[level];
    if (index < 0 || index >= cap) return false;

    causa_memory_slot_t* slot = mem->slots[level][index];
    if (!slot->occupied) return false;

    if (out_key) *out_key = slot->key;
    if (out_value) *out_value = slot->value;
    return true;
}

bool causa_memory_restore_slot(
    causa_memory_t* mem,
    causa_memory_level_t level,
    int64_t index,
    causa_tensor_t* key,
    causa_tensor_t* value,
    float strength,
    float timestamp,
    int32_t access_count
) {
    if (!mem || !key || !value || level >= CAUSA_MEMORY_MAX_LEVELS) return false;
    int64_t cap = mem->config.capacity[level];
    if (index < 0 || index >= cap) return false;

    causa_memory_slot_t* slot = mem->slots[level][index];

    if (!slot->key) {
        causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        opts.dtype = CAUSA_DTYPE_F32;
        opts.device = CAUSA_DEVICE_CPU;
        opts.requires_grad = mem->config.differentiable;
        int64_t key_shape[] = {mem->config.key_dim};
        int64_t val_shape[] = {mem->config.value_dim};
        slot->key = causa_tensor_new(mem->arena, mem->backend, key_shape, 1, &opts);
        slot->value = causa_tensor_new(mem->arena, mem->backend, val_shape, 1, &opts);
        if (!slot->key || !slot->value) return false;
    }

    causa_tensor_copy(slot->key, key);
    causa_tensor_copy(slot->value, value);
    slot->occupied = true;
    slot->strength = strength;
    slot->timestamp = timestamp;
    slot->access_count = access_count;
    return true;
}

int64_t causa_memory_collect(
    const causa_memory_t* mem,
    causa_memory_level_t level,
    causa_tensor_t* out
) {
    if (!mem || !out || level >= CAUSA_MEMORY_MAX_LEVELS) return -1;

    int64_t dim = mem->config.key_dim;
    int64_t count = causa_memory_size(mem, level);
    if (count <= 0) return 0;

    // The output must be a matrix with one row per observation. The number
    // of rows is checked as well as the number of columns, because a matrix
    // with too few rows would be written past its end.
    if (out->ndim != 2 || out->shape[1] != dim) return -1;
    if (out->shape[0] < count) return -1;

    float* dst = (float*)out->data;
    int64_t row = 0;
    int64_t cap = mem->config.capacity[level];

    for (int64_t i = 0; i < cap && row < count; i++) {
        causa_tensor_t* key = NULL;
        causa_tensor_t* value = NULL;
        if (!causa_memory_slot_tensors(mem, level, i, &key, &value)) continue;

        const float* src = (const float*)key->data;
        for (int64_t j = 0; j < dim; j++) {
            dst[row * dim + j] = src[j];
        }
        row++;
    }

    return row;
}

causa_tensor_t* causa_memory_store_causal_law(
    causa_memory_t* mem,
    int32_t child,
    const causa_tensor_t* parent_mask,
    const causa_tensor_t* coefficients
) {
    if (!mem || !parent_mask || !coefficients) return NULL;

    int64_t dim = mem->config.key_dim;
    if (causa_tensor_numel(parent_mask) < dim) return NULL;
    if (causa_tensor_numel(coefficients) < dim) return NULL;

    // The law is read back one coefficient per variable, so the value has
    // to be as wide as the key. A memory whose values are narrower cannot
    // hold the law, and saying so is better than holding a law that has
    // lost some of its coefficients.
    if (mem->config.value_dim < dim) return NULL;

    // Build the role vector: 2 marks the child, 1 marks each parent.
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;

    int64_t shape[] = {dim};
    causa_tensor_t* role = causa_tensor_new(mem->arena, mem->backend, shape, 1, &opts);
    if (!role) return NULL;

    float* r = (float*)role->data;
    const float* mask = (const float*)parent_mask->data;
    for (int64_t i = 0; i < dim; i++) {
        r[i] = (mask[i] > 0.5f) ? 1.0f : 0.0f;
    }
    if (child >= 0 && child < dim) {
        r[child] = 2.0f;
    }

    // The value is as wide as the memory's values are, so that it is
    // stored like any other value.
    int64_t value_shape[] = {mem->config.value_dim};
    causa_tensor_t* slot_value = causa_tensor_new(mem->arena, mem->backend, value_shape, 1, &opts);
    if (!slot_value) return NULL;
    causa_tensor_copy(slot_value, coefficients);

    causa_tensor_t* stored = causa_memory_write(mem, role, slot_value, CAUSA_MEMORY_SEMANTIC);
    if (!stored) return NULL;

    return stored;
}

int32_t causa_memory_find_causal_laws(
    const causa_memory_t* mem,
    int32_t child,
    causa_tensor_t** out_values,
    int32_t max_values
) {
    if (!mem || !out_values) return -1;

    int64_t dim = mem->config.key_dim;
    int64_t cap = mem->config.capacity[CAUSA_MEMORY_SEMANTIC];
    int32_t found = 0;

    for (int64_t i = 0; i < cap && found < max_values; i++) {
        causa_tensor_t* key = NULL;
        causa_tensor_t* value = NULL;
        if (!causa_memory_slot_tensors(mem, CAUSA_MEMORY_SEMANTIC, i, &key, &value)) continue;

        const float* k = (const float*)key->data;
        if (child < 0 || child >= dim) continue;
        if (k[child] != 2.0f) continue;

        out_values[found++] = value;
    }

    return found;
}

int32_t causa_memory_count_causal_laws(
    const causa_memory_t* mem,
    int32_t child
) {
    if (!mem) return 0;

    int64_t dim = mem->config.key_dim;
    int64_t cap = mem->config.capacity[CAUSA_MEMORY_SEMANTIC];
    if (child < 0 || child >= dim) return 0;

    // The laws are counted where they are, rather than through a list that
    // has room for a fixed number of them. A count that stops at the size
    // of the list would say that a child has fewer laws than it has.
    int32_t count = 0;
    for (int64_t i = 0; i < cap; i++) {
        causa_tensor_t* key = NULL;
        causa_tensor_t* value = NULL;
        if (!causa_memory_slot_tensors(mem, CAUSA_MEMORY_SEMANTIC, i, &key, &value)) continue;

        const float* k = (const float*)key->data;
        if (k[child] != 2.0f) continue;
        count++;
    }
    return count;
}

int32_t causa_memory_find_causal_law_entries(
    const causa_memory_t* mem,
    int32_t child,
    causa_tensor_t** out_keys,
    causa_tensor_t** out_values,
    int32_t max_entries
) {
    if (!mem || !out_keys || !out_values || max_entries <= 0) return -1;

    int64_t dim = mem->config.key_dim;
    int64_t cap = mem->config.capacity[CAUSA_MEMORY_SEMANTIC];
    int32_t found = 0;

    if (child < 0 || child >= dim) return 0;

    for (int64_t i = 0; i < cap && found < max_entries; i++) {
        causa_tensor_t* key = NULL;
        causa_tensor_t* value = NULL;
        if (!causa_memory_slot_tensors(mem, CAUSA_MEMORY_SEMANTIC, i, &key, &value)) continue;

        const float* k = (const float*)key->data;
        if (k[child] != 2.0f) continue;

        out_keys[found] = key;
        out_values[found] = value;
        found++;
    }

    return found;
}

// ===== Procedural memory =====

// The straight-line distance between two states.
static float memory_distance(
    const causa_tensor_t* a,
    const causa_tensor_t* b,
    int64_t dim
) {
    const float* ad = (const float*)a->data;
    const float* bd = (const float*)b->data;

    float total = 0.0f;
    for (int64_t j = 0; j < dim; j++) {
        float diff = ad[j] - bd[j];
        total += diff * diff;
    }
    return sqrtf(total);
}

// Make sure a slot has the tensors it needs to hold a key and a value.
static bool memory_prepare_slot(
    causa_memory_t* mem,
    causa_memory_slot_t* slot
) {
    if (slot->key && slot->value) return true;

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;
    opts.requires_grad = mem->config.differentiable;

    int64_t key_shape[] = {mem->config.key_dim};
    int64_t val_shape[] = {mem->config.value_dim};

    if (!slot->key) {
        slot->key = causa_tensor_new(mem->arena, mem->backend, key_shape, 1, &opts);
        if (!slot->key) return false;
    }
    if (!slot->value) {
        slot->value = causa_tensor_new(mem->arena, mem->backend, val_shape, 1, &opts);
        if (!slot->value) return false;
    }
    return true;
}

causa_tensor_t* causa_memory_store_procedure(
    causa_memory_t* mem,
    const causa_tensor_t* trigger,
    const causa_tensor_t* action
) {
    if (!mem || !trigger || !action) return NULL;

    int64_t dim = mem->config.key_dim;
    if (causa_tensor_numel(trigger) < dim) return NULL;
    if (causa_tensor_numel(action) < mem->config.value_dim) return NULL;

    // A procedure with the same trigger is the same procedure, so it is
    // updated rather than stored again.
    int64_t existing = memory_find_slot(mem, CAUSA_MEMORY_PROCEDURAL, trigger);
    causa_memory_slot_t* slot;

    if (existing >= 0) {
        slot = mem->slots[CAUSA_MEMORY_PROCEDURAL][existing];
    } else {
        int64_t empty = memory_find_empty_slot(mem, CAUSA_MEMORY_PROCEDURAL);
        if (empty < 0) return NULL;
        slot = mem->slots[CAUSA_MEMORY_PROCEDURAL][empty];
    }

    if (!memory_prepare_slot(mem, slot)) return NULL;

    causa_tensor_copy(slot->key, trigger);
    causa_tensor_copy(slot->value, action);
    slot->timestamp = mem->global_time;
    slot->strength = 1.0f;
    slot->occupied = true;
    if (slot->group_count <= 0) slot->group_count = 1;

    return slot->value;
}

// Select the indices of the n smallest distances.
//
// The distances are indexed by slot. The selection is a partial sort, so it
// costs n passes over the slots rather than a full sort.
static void memory_select_nearest(
    const float* distances,
    int64_t count,
    int32_t n,
    int64_t* out
) {
    for (int32_t i = 0; i < n; i++) {
        int64_t best = i;
        for (int64_t j = i + 1; j < count; j++) {
            if (distances[j] < distances[best]) best = j;
        }
        int64_t tmp = out[i];
        out[i] = out[best];
        out[best] = tmp;
    }
}

int32_t causa_memory_find_procedures(
    const causa_memory_t* mem,
    const causa_tensor_t* state,
    causa_tensor_t** out_triggers,
    causa_tensor_t** out_actions,
    int32_t max_procedures
) {
    if (!mem || !state || !out_triggers || !out_actions || max_procedures <= 0) return -1;

    int64_t dim = mem->config.key_dim;
    if (causa_tensor_numel(state) < dim) return -1;

    int64_t cap = mem->config.capacity[CAUSA_MEMORY_PROCEDURAL];

    // The distances go in a buffer of this call's own. A buffer shared by
    // the whole memory would be written by two calls at once, and the two
    // would mix their numbers together. The call is also allowed to change
    // the memory it is given, so it must not write to it.
    float* distances = (float*)malloc((size_t)(cap > 0 ? cap : 1) * sizeof(float));
    int64_t* slots = (int64_t*)malloc((size_t)(cap > 0 ? cap : 1) * sizeof(int64_t));
    if (!distances || !slots) { free(distances); free(slots); return -1; }

    int64_t found = 0;
    for (int64_t i = 0; i < cap; i++) {
        causa_memory_slot_t* slot = mem->slots[CAUSA_MEMORY_PROCEDURAL][i];
        slots[i] = i;
        if (!slot->occupied || !slot->key) {
            distances[i] = FLT_MAX;
            continue;
        }
        distances[i] = memory_distance(state, slot->key, dim);
        found++;
    }

    if (found == 0) { free(distances); free(slots); return 0; }

    int32_t actual = (max_procedures < found) ? max_procedures : (int32_t)found;
    memory_select_nearest(distances, cap, actual, slots);

    for (int32_t i = 0; i < actual; i++) {
        causa_memory_slot_t* slot = mem->slots[CAUSA_MEMORY_PROCEDURAL][slots[i]];
        out_triggers[i] = slot->key;
        out_actions[i] = slot->value;
    }

    free(distances);
    free(slots);

    return actual;
}

int32_t causa_memory_learn_procedures(
    causa_memory_t* mem,
    float radius
) {
    if (!mem) return -1;
    if (radius <= 0.0f) radius = 0.5f;

    int64_t dim = mem->config.key_dim;
    int64_t value_dim = mem->config.value_dim;
    int64_t cap = mem->config.capacity[CAUSA_MEMORY_PROCEDURAL];
    int64_t episodic_cap = mem->config.capacity[CAUSA_MEMORY_EPISODIC];

    for (int64_t e = 0; e < episodic_cap; e++) {
        causa_tensor_t* key = NULL;
        causa_tensor_t* value = NULL;
        if (!causa_memory_slot_tensors(mem, CAUSA_MEMORY_EPISODIC, e, &key, &value)) continue;
        if (!key || !value) continue;

        // The nearest procedure.
        int64_t best = -1;
        float best_distance = 0.0f;
        for (int64_t p = 0; p < cap; p++) {
            causa_memory_slot_t* slot = mem->slots[CAUSA_MEMORY_PROCEDURAL][p];
            if (!slot->occupied || !slot->key) continue;
            float d = memory_distance(key, slot->key, dim);
            if (best < 0 || d < best_distance) {
                best = p;
                best_distance = d;
            }
        }

        if (best >= 0 && best_distance <= radius) {
            // Fold the episode into the procedure: the centre and the
            // action both move a little towards what this episode shows.
            causa_memory_slot_t* slot = mem->slots[CAUSA_MEMORY_PROCEDURAL][best];
            int32_t n = slot->group_count > 0 ? slot->group_count : 1;
            float weight = 1.0f / (float)(n + 1);

            float* kd = (float*)slot->key->data;
            const float* ek = (const float*)key->data;
            for (int64_t j = 0; j < dim; j++) kd[j] += (ek[j] - kd[j]) * weight;

            float* vd = (float*)slot->value->data;
            const float* ev = (const float*)value->data;
            for (int64_t j = 0; j < value_dim && j < causa_tensor_numel(value); j++) {
                vd[j] += (ev[j] - vd[j]) * weight;
            }

            slot->group_count = n + 1;
            slot->timestamp = mem->global_time;
            slot->strength = fminf(1.0f, slot->strength + 0.05f);
            continue;
        }

        // The episode starts a procedure of its own.
        int64_t empty = memory_find_empty_slot(mem, CAUSA_MEMORY_PROCEDURAL);
        if (empty < 0) break;

        causa_memory_slot_t* slot = mem->slots[CAUSA_MEMORY_PROCEDURAL][empty];
        if (!memory_prepare_slot(mem, slot)) break;

        causa_tensor_copy(slot->key, key);
        causa_tensor_copy(slot->value, value);
        slot->timestamp = mem->global_time;
        slot->strength = 1.0f;
        slot->occupied = true;
        slot->group_count = 1;
    }

    int32_t total = 0;
    for (int64_t p = 0; p < cap; p++) {
        if (mem->slots[CAUSA_MEMORY_PROCEDURAL][p]->occupied) total++;
    }
    return total;
}

// ===== Counterfactuals in memory =====

int32_t causa_memory_keep_counterfactual(
    causa_memory_t* mem,
    const char* query_name,
    const causa_tensor_t* context,
    const causa_tensor_t* result
) {
    if (!mem || !query_name || !context || !result) return -1;
    if (mem->capacity_counterfactuals <= 0) return -1;

    int64_t dim = mem->config.key_dim;
    if (causa_tensor_numel(context) < dim) return -1;

    // The room is full, so the oldest goes. What was learned first is the
    // least useful once there is more.
    //
    // The tensors of the one that goes are kept and used again, so that the
    // memory does not grow for every counterfactual that is kept.
    causa_tensor_t* spare_context = NULL;
    causa_tensor_t* spare_result = NULL;
    if (mem->num_counterfactuals >= mem->capacity_counterfactuals) {
        causa_memory_counterfactual_t* dropped = &mem->counterfactuals[0];
        spare_context = dropped->context;
        spare_result = dropped->result;
        dropped->context = NULL;
        dropped->result = NULL;
        memmove(&mem->counterfactuals[0], &mem->counterfactuals[1],
                sizeof(causa_memory_counterfactual_t) * (size_t)(mem->capacity_counterfactuals - 1));
        mem->num_counterfactuals--;
    }

    causa_memory_counterfactual_t* entry = &mem->counterfactuals[mem->num_counterfactuals];

    // The context and the result are copied, so the caller keeps its own
    // tensors and can change them afterwards.
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;

    int64_t shape[] = {dim};
    entry->context = spare_context ? spare_context : causa_tensor_new(mem->arena, mem->backend, shape, 1, &opts);
    entry->result = spare_result ? spare_result : causa_tensor_new(mem->arena, mem->backend, shape, 1, &opts);
    if (!entry->context || !entry->result) {
        entry->context = NULL;
        entry->result = NULL;
        return -1;
    }

    causa_tensor_copy(entry->context, context);
    causa_tensor_copy(entry->result, result);

    strncpy(entry->query_name, query_name, CAUSA_MAX_NAME - 1);
    entry->query_name[CAUSA_MAX_NAME - 1] = '\0';
    entry->timestamp = mem->global_time;

    mem->num_counterfactuals++;
    return mem->num_counterfactuals;
}

int32_t causa_memory_find_counterfactuals(
    const causa_memory_t* mem,
    const char* query_name,
    const causa_tensor_t* context,
    int32_t max_results,
    causa_tensor_t** out_results
) {
    if (!mem || !query_name || !context || max_results <= 0 || !out_results) return -1;

    int64_t dim = mem->config.key_dim;
    if (causa_tensor_numel(context) < dim) return -1;

    const float* ctx = (const float*)context->data;

    // The counterfactuals about the variable, with their distance from the
    // given situation.
    int32_t found = 0;
    float* distances = (float*)malloc(sizeof(float) * (size_t)(mem->num_counterfactuals ? mem->num_counterfactuals : 1));
    int32_t* indices = (int32_t*)malloc(sizeof(int32_t) * (size_t)(mem->num_counterfactuals ? mem->num_counterfactuals : 1));
    if (!distances || !indices) {
        free(distances);
        free(indices);
        return -1;
    }

    for (int32_t i = 0; i < mem->num_counterfactuals; i++) {
        const causa_memory_counterfactual_t* entry = &mem->counterfactuals[i];
        if (strcmp(entry->query_name, query_name) != 0) continue;
        if (!entry->context || !entry->result) continue;

        const float* c = (const float*)entry->context->data;
        float total = 0.0f;
        for (int64_t j = 0; j < dim; j++) {
            // A dimension that was not observed is left out of the distance,
            // the same way the prediction leaves it out. A distance that
            // counted it would be a number that is not a number, and the
            // order of the counterfactuals would be decided by it.
            if (isnan(ctx[j]) || isnan(c[j])) continue;
            float diff = ctx[j] - c[j];
            total += diff * diff;
        }
        distances[found] = total;
        indices[found] = i;
        found++;
    }

    // The nearest ones first.
    //
    // The distances are held in the order the counterfactuals were found,
    // so the comparison uses the position in that list and not the index of
    // the counterfactual. Using the index would read past the end of the
    // distances whenever a counterfactual that does not match came first.
    for (int32_t i = 0; i < found; i++) {
        int32_t best = i;
        for (int32_t j = i + 1; j < found; j++) {
            if (distances[j] < distances[best]) best = j;
        }
        float tmp_d = distances[i];
        distances[i] = distances[best];
        distances[best] = tmp_d;
        int32_t tmp_i = indices[i];
        indices[i] = indices[best];
        indices[best] = tmp_i;
    }

    int32_t count = found < max_results ? found : max_results;
    for (int32_t i = 0; i < count; i++) {
        out_results[i] = mem->counterfactuals[indices[i]].result;
    }

    free(distances);
    free(indices);
    return count;
}

// ===== Reasoning over memory =====

// The learned structure of a variable: which variables are its parents and
// what the coefficient of each one is.
typedef struct {
    int32_t child;
    int32_t num_parents;
    int32_t parents[64];
    float coefficients[64];
    bool found;
} memory_law_t;

// Collect the law of a variable from semantic memory.
//
// When several laws were recorded for the same child the one with the most
// parents is taken, because it describes the variable in the most detail.
static bool memory_collect_law(
    const causa_memory_t* mem,
    int32_t child,
    memory_law_t* law
) {
    if (!mem || !law) return false;

    int64_t dim = mem->config.key_dim;
    if (child < 0 || child >= dim) return false;

    // Every law of the child is looked at. A list with room for a fixed
    // number of them would leave the rest unseen, and the law that was
    // missed might be the one with the most parents.
    int64_t cap = mem->config.capacity[CAUSA_MEMORY_SEMANTIC];
    if (cap <= 0) return false;

    causa_tensor_t** keys = (causa_tensor_t**)malloc((size_t)cap * sizeof(causa_tensor_t*));
    causa_tensor_t** values = (causa_tensor_t**)malloc((size_t)cap * sizeof(causa_tensor_t*));
    if (!keys || !values) { free(keys); free(values); return false; }

    int32_t n = causa_memory_find_causal_law_entries(mem, child, keys, values, (int32_t)cap);
    if (n <= 0) { free(keys); free(values); return false; }

    bool have = false;

    for (int32_t e = 0; e < n; e++) {
        const float* role = (const float*)keys[e]->data;
        const float* coeff = (const float*)values[e]->data;
        int64_t vdim = causa_tensor_numel(values[e]);

        memory_law_t candidate;
        memset(&candidate, 0, sizeof(candidate));
        candidate.child = child;
        candidate.found = true;

        // The coefficient of a parent is read from the value, so the value
        // has to be wide enough for it.
        for (int64_t j = 0; j < dim && j < vdim && candidate.num_parents < 64; j++) {
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

    free(keys);
    free(values);
    return have;
}

// Order the variables so that a parent always comes before its child.
//
// The laws come out of the discovery step, which enforces acyclicity, so the
// order always exists. A cycle would mean the discovery produced something
// inconsistent. The order is then not complete: the variables that take
// part in the cycle are left out, so that the caller can tell that the
// prediction cannot be worked out rather than being given a number that is
// not a number.
//
// The number of variables that were placed is returned.
static int32_t memory_order_variables(
    const memory_law_t* laws,
    int32_t num_variables,
    int32_t* order
) {
    int32_t* pending = (int32_t*)calloc((size_t)num_variables, sizeof(int32_t));
    bool* placed = (bool*)calloc((size_t)num_variables, sizeof(bool));
    if (!pending || !placed) {
        free(pending);
        free(placed);
        for (int32_t i = 0; i < num_variables; i++) order[i] = i;
        return num_variables;
    }

    for (int32_t i = 0; i < num_variables; i++) {
        pending[i] = laws[i].found ? laws[i].num_parents : 0;
    }

    int32_t written = 0;
    for (int32_t step = 0; step < num_variables; step++) {
        int32_t chosen = -1;
        for (int32_t i = 0; i < num_variables; i++) {
            if (placed[i] || pending[i] != 0) continue;
            chosen = i;
            break;
        }

        if (chosen < 0) {
            // A cycle, or a variable whose parent is missing. The
            // variables that take part in it are left out of the order, so
            // that the caller can tell that they cannot be worked out.
            break;
        }

        placed[chosen] = true;
        order[written++] = chosen;

        // The children of this variable can now be placed.
        for (int32_t c = 0; c < num_variables; c++) {
            if (placed[c] || !laws[c].found) continue;
            for (int32_t p = 0; p < laws[c].num_parents; p++) {
                if (laws[c].parents[p] == chosen) {
                    pending[c]--;
                    break;
                }
            }
        }
    }

    free(pending);
    free(placed);
    return written;
}

// The residual of an episode: what the law does not explain.
//
// The law predicts the child from its parents. The episode holds the whole
// observation, so the difference between what happened and what the law
// predicts is the part the law misses. This is where an offset that the
// linear law cannot express, such as a constant term, is picked up.
static float memory_episode_residual(
    const causa_memory_t* mem,
    const causa_tensor_t* key,
    const memory_law_t* law
) {
    if (!mem || !key || !law) return 0.0f;

    const float* kd = (const float*)key->data;

    float predicted = 0.0f;
    for (int32_t p = 0; p < law->num_parents; p++) {
        predicted += law->coefficients[p] * kd[law->parents[p]];
    }

    return kd[law->child] - predicted;
}

// The weight of an episode for a partial observation.
//
// The closer the episode is to the observation, the more it says about
// this case, so the weight falls off with the distance.
static float memory_episode_weight(
    const float* observation,
    const bool* observed,
    const float* key,
    int64_t dim
) {
    float total = 0.0f;
    for (int64_t j = 0; j < dim; j++) {
        if (!observed[j]) continue;
        float diff = observation[j] - key[j];
        total += diff * diff;
    }
    return 1.0f / (1.0f + total);
}

// How many of the nearest episodes speak to a case.
//
// The nearest ones are the ones that say something about it. A cosine
// would not pick them out: with one observed variable it is either 1 or
// -1, so every episode on the same side would count the same however far
// away it was, and the residual would be an average over half of them
// rather than over the ones that resemble the observation.
#define MEMORY_NEAREST_EPISODES 8

// The episodes that resemble a partial observation, the nearest first.
//
// The slot indices are written to out and the number found is returned.
static int32_t memory_nearest_episodes(
    const causa_memory_t* mem,
    const float* observation,
    const bool* observed,
    int64_t dim,
    int64_t* out_slots
) {
    int64_t cap = mem->config.capacity[CAUSA_MEMORY_EPISODIC];

    // The distances go in a buffer of this call's own. A buffer shared by
    // the whole memory would be written by two calls at once, and the two
    // would mix their numbers together.
    float* distances = (float*)malloc((size_t)(cap > 0 ? cap : 1) * sizeof(float));
    if (!distances) return 0;
    int64_t found = 0;

    for (int64_t i = 0; i < cap; i++) {
        causa_memory_slot_t* slot = mem->slots[CAUSA_MEMORY_EPISODIC][i];
        out_slots[i] = i;
        if (!slot->occupied || !slot->key) {
            distances[i] = FLT_MAX;
            continue;
        }
        const float* kd = (const float*)slot->key->data;
        float total = 0.0f;
        for (int64_t j = 0; j < dim; j++) {
            if (!observed[j]) continue;
            float diff = observation[j] - kd[j];
            total += diff * diff;
        }
        distances[i] = total;
        found++;
    }

    if (found == 0) { free(distances); return 0; }

    int32_t actual = (MEMORY_NEAREST_EPISODES < found) ? MEMORY_NEAREST_EPISODES : (int32_t)found;

    // A partial selection: the nearest ones are moved to the front.
    for (int32_t i = 0; i < actual; i++) {
        int64_t best = i;
        for (int64_t j = i + 1; j < cap; j++) {
            if (distances[j] < distances[best]) best = j;
        }
        float tmp = distances[i]; distances[i] = distances[best]; distances[best] = tmp;
        int64_t slot = out_slots[i]; out_slots[i] = out_slots[best]; out_slots[best] = slot;
    }

    free(distances);

    return actual;
}

int32_t causa_memory_predict(
    causa_memory_t* mem,
    const causa_tensor_t* observation,
    causa_tensor_t* out
) {
    if (!mem || !observation || !out) return -1;

    int64_t dim = mem->config.key_dim;
    if (causa_tensor_numel(observation) < dim) return -1;
    if (causa_tensor_numel(out) < dim) return -1;

    const float* obs = (const float*)observation->data;
    float* dst = (float*)out->data;

    bool* observed = (bool*)calloc((size_t)dim, sizeof(bool));
    float* values = (float*)calloc((size_t)dim, sizeof(float));
    memory_law_t* laws = (memory_law_t*)calloc((size_t)dim, sizeof(memory_law_t));
    int32_t* order = (int32_t*)calloc((size_t)dim, sizeof(int32_t));

    if (!observed || !values || !laws || !order) {
        free(observed); free(values); free(laws); free(order);
        return -1;
    }

    int32_t predicted = 0;

    for (int64_t j = 0; j < dim; j++) {
        observed[j] = !isnan(obs[j]);
        values[j] = obs[j];
    }

    // The structure comes from the laws in semantic memory.
    for (int64_t j = 0; j < dim; j++) {
        memory_collect_law(mem, (int32_t)j, &laws[j]);
    }
    int32_t ordered = memory_order_variables(laws, (int32_t)dim, order);

    // The offset comes from the episodes in episodic memory.
    int64_t episodic_cap = mem->config.capacity[CAUSA_MEMORY_EPISODIC];

    // Only the variables that are in the order are worked out. A variable
    // that is not in it depends on a variable that depends on it, so there
    // is no order to work it out in and it is left as it was.
    for (int32_t oi = 0; oi < ordered; oi++) {
        int32_t c = order[oi];
        if (observed[c]) continue;

        const memory_law_t* law = &laws[c];
        float estimate = 0.0f;

        if (law->found && law->num_parents > 0) {
            // The law predicts the variable from its parents.
            for (int32_t p = 0; p < law->num_parents; p++) {
                int32_t parent = law->parents[p];
                if (parent >= 0 && parent < dim) {
                    estimate += law->coefficients[p] * values[parent];
                }
            }

            // The episodes say what the law leaves out. The episodes that
            // resemble the observation the most are the ones that speak to
            // this case, so their residuals are averaged with weights that
            // grow with the closeness.
            {
                int64_t* nearest = (int64_t*)malloc(sizeof(int64_t) * (size_t)mem->config.capacity[CAUSA_MEMORY_EPISODIC]);
                if (!nearest) return -1;

                float weight_total = 0.0f;
                float residual_total = 0.0f;

                int32_t count = memory_nearest_episodes(mem, obs, observed, dim, nearest);
                for (int32_t e = 0; e < count; e++) {
                    causa_memory_slot_t* slot = mem->slots[CAUSA_MEMORY_EPISODIC][nearest[e]];
                    if (!slot->key) continue;

                    const float* kd = (const float*)slot->key->data;
                    float weight = memory_episode_weight(obs, observed, kd, dim);
                    if (weight <= 0.0f) continue;

                    float residual = memory_episode_residual(mem, slot->key, law);
                    residual_total += weight * residual;
                    weight_total += weight;
                }

                free(nearest);

                if (weight_total > 0.0f) {
                    estimate += residual_total / weight_total;
                }
            }
        } else {
            // Without a law the marginal mean is the best that can be said.
            float total = 0.0f;
            int64_t count = 0;
            for (int64_t e = 0; e < episodic_cap; e++) {
                causa_tensor_t* key = NULL;
                causa_tensor_t* value = NULL;
                if (!causa_memory_slot_tensors(mem, CAUSA_MEMORY_EPISODIC, e, &key, &value)) continue;
                if (!key) continue;
                total += ((const float*)key->data)[c];
                count++;
            }
            if (count > 0) estimate = total / (float)count;
        }

        values[c] = estimate;
        predicted++;
    }

    // A value that is not a number is a value that could not be worked
    // out. The laws are in a cycle, so a variable depends on a variable
    // that depends on it. The prediction is reported as a failure rather
    // than as a number that is not a number.
    bool failed = false;
    for (int64_t j = 0; j < dim; j++) {
        if (!observed[j] && isnan(values[j])) {
            failed = true;
        }
        dst[j] = values[j];
    }

    free(observed); free(values); free(laws); free(order);
    return failed ? -1 : predicted;
}

int32_t causa_memory_predict_variable(
    causa_memory_t* mem,
    const causa_tensor_t* observation,
    int32_t variable,
    causa_tensor_t* out
) {
    if (!mem || !observation || !out) return -1;

    int64_t dim = mem->config.key_dim;
    if (variable < 0 || variable >= dim) return -1;
    if (causa_tensor_numel(out) < 1) return -1;

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;

    int64_t shape[] = {dim};
    causa_tensor_t* full = causa_tensor_new(mem->arena, mem->backend, shape, 1, &opts);
    causa_tensor_t* result = causa_tensor_new(mem->arena, mem->backend, shape, 1, &opts);
    if (!full || !result) return -1;

    const float* src = (const float*)observation->data;
    float* fd = (float*)full->data;
    for (int64_t j = 0; j < dim; j++) fd[j] = src[j];

    int32_t predicted = causa_memory_predict(mem, full, result);
    if (predicted < 0) return -1;

    ((float*)out->data)[0] = ((const float*)result->data)[variable];
    return 0;
}
// ===== Acting on what is known =====

int32_t causa_memory_plan_intervention(
    causa_memory_t* mem,
    const causa_tensor_t* state,
    const causa_tensor_t* goal,
    causa_tensor_t* out,
    int32_t* out_count
) {
    if (!mem || !state || !goal || !out) return -1;

    int64_t dim = mem->config.key_dim;
    if (causa_tensor_numel(state) < dim) return -1;
    if (causa_tensor_numel(goal) < dim) return -1;
    if (causa_tensor_numel(out) < dim) return -1;

    const float* sd = (const float*)state->data;
    const float* gd = (const float*)goal->data;
    float* od = (float*)out->data;

    // The goal names the variables to be changed: each holds a wanted value,
    // and the others are not wanted.
    int32_t targets[CAUSA_MAX_VARIABLES];
    float wanted[CAUSA_MAX_VARIABLES];
    int32_t num_targets = 0;
    for (int64_t j = 0; j < dim; j++) {
        if (isnan(gd[j])) continue;
        if (num_targets >= CAUSA_MAX_VARIABLES) return -1;
        targets[num_targets] = (int32_t)j;
        wanted[num_targets] = gd[j];
        num_targets++;
    }
    if (num_targets == 0) return -1;

    for (int64_t j = 0; j < dim; j++) od[j] = 0.0f;
    if (out_count) *out_count = 0;

    // Where the goal variables stand now, according to what is known.
    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = CAUSA_DTYPE_F32;
    opts.device = CAUSA_DEVICE_CPU;

    int64_t shape[] = {dim};
    causa_tensor_t* observation = causa_tensor_new(mem->arena, mem->backend, shape, 1, &opts);
    causa_tensor_t* prediction = causa_tensor_new(mem->arena, mem->backend, shape, 1, &opts);
    if (!observation || !prediction) return -1;

    float* ob = (float*)observation->data;
    for (int64_t j = 0; j < dim; j++) {
        // The goal variables are left out, so that they are predicted from
        // what is known rather than read off the state.
        bool wanted_now = false;
        for (int32_t t = 0; t < num_targets; t++) {
            if (targets[t] == (int32_t)j) { wanted_now = true; break; }
        }
        ob[j] = wanted_now ? NAN : sd[j];
    }

    if (causa_memory_predict(mem, observation, prediction) < 0) return -1;
    const float* pd = (const float*)prediction->data;

    // The law of each variable says what moves it.
    memory_law_t* laws = (memory_law_t*)calloc((size_t)dim, sizeof(memory_law_t));
    if (!laws) return -1;
    for (int64_t j = 0; j < dim; j++) {
        memory_collect_law(mem, (int32_t)j, &laws[j]);
    }

    // The variables that can be acted on.
    //
    // For each goal the strongest cause is the one acted on, because it
    // moves the goal the most for the least change. A goal with no cause
    // is exogenous, so it is set directly.
    int32_t* candidates = (int32_t*)calloc((size_t)dim, sizeof(int32_t));
    int32_t num_candidates = 0;
    if (!candidates) { free(laws); return -1; }

    for (int32_t t = 0; t < num_targets; t++) {
        int32_t g = targets[t];
        const memory_law_t* law = &laws[g];

        int32_t action = -1;
        if (law->found && law->num_parents > 0) {
            float best = 0.0f;
            for (int32_t p = 0; p < law->num_parents; p++) {
                float magnitude = fabsf(law->coefficients[p]);
                if (magnitude > best) {
                    best = magnitude;
                    action = law->parents[p];
                }
            }
        } else {
            action = g;
        }

        if (action < 0 || action >= dim) continue;

        bool known = false;
        for (int32_t c = 0; c < num_candidates; c++) {
            if (candidates[c] == action) { known = true; break; }
        }
        if (!known) candidates[num_candidates++] = action;
    }

    // The effect of each action on each goal.
    //
    // A change to a variable moves everything downstream of it, so the
    // effect is the sum over the paths from the action to the goal of the
    // product of the coefficients along the path. The matrix is worked out
    // in the order the variables come in, so that a parent is settled
    // before the child that depends on it.
    float* effects = (float*)calloc((size_t)dim * (size_t)num_candidates, sizeof(float));
    if (!effects) { free(laws); free(candidates); return -1; }

    for (int32_t c = 0; c < num_candidates; c++) {
        // An action changes the variable it is applied to.
        effects[candidates[c] * num_candidates + c] = 1.0f;
    }

    int32_t* order = (int32_t*)calloc((size_t)dim, sizeof(int32_t));
    if (!order) { free(laws); free(candidates); free(effects); return -1; }
    memory_order_variables(laws, (int32_t)dim, order);

    for (int32_t oi = 0; oi < dim; oi++) {
        int32_t v = order[oi];
        const memory_law_t* law = &laws[v];
        if (!law->found) continue;

        for (int32_t p = 0; p < law->num_parents; p++) {
            int32_t parent = law->parents[p];
            if (parent < 0 || parent >= dim) continue;
            float coef = law->coefficients[p];
            for (int32_t c = 0; c < num_candidates; c++) {
                effects[v * num_candidates + c] += coef * effects[parent * num_candidates + c];
            }
        }
    }
    free(order);

    // The change each goal needs.
    float* needed = (float*)calloc((size_t)num_targets, sizeof(float));
    if (!needed) { free(laws); free(candidates); free(effects); return -1; }

    bool reachable = true;
    for (int32_t t = 0; t < num_targets; t++) {
        int32_t g = targets[t];
        float current = pd[g];
        if (isnan(current)) { reachable = false; break; }
        needed[t] = wanted[t] - current;
    }

    if (!reachable) {
        free(laws); free(candidates); free(effects); free(needed);
        return -1;
    }

    // Solve for the change to each action.
    //
    // The goals are solved together, so that a change made for one goal
    // does not undo another. Where the goals cannot all be reached, the
    // change that fits them best is given.
    float* changes = (float*)calloc((size_t)num_candidates, sizeof(float));
    if (!changes) { free(laws); free(candidates); free(effects); free(needed); return -1; }

    if (num_candidates > 0) {
        // Only the rows of the goals are used.
        float* rows = (float*)calloc((size_t)num_targets * (size_t)num_candidates, sizeof(float));
        float* solution = (float*)calloc((size_t)num_candidates, sizeof(float));
        if (!rows || !solution) {
            free(rows); free(solution); free(laws); free(candidates);
            free(effects); free(needed); free(changes);
            return -1;
        }

        for (int32_t t = 0; t < num_targets; t++) {
            int32_t g = targets[t];
            for (int32_t c = 0; c < num_candidates; c++) {
                rows[t * num_candidates + c] = effects[g * num_candidates + c];
            }
        }

        if (causa_solve_least_squares(rows, needed, num_targets, num_candidates, solution)) {
            memcpy(changes, solution, sizeof(float) * (size_t)num_candidates);
        }

        free(rows);
        free(solution);
    }

    // The change to each variable.
    int32_t count = 0;
    for (int32_t c = 0; c < num_candidates; c++) {
        int32_t action = candidates[c];
        if (fabsf(changes[c]) < 1e-6f) continue;
        od[action] += changes[c];
        count++;
    }

    free(laws);
    free(candidates);
    free(effects);
    free(needed);
    free(changes);

    if (out_count) *out_count = count;
    return 0;
}


