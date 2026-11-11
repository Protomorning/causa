#include "causa_graph.h"
#include "causa_arena.h"
#include "causa_tensor.h"
#include <stdlib.h>
#include <string.h>

// Forward declarations for standard op implementations
static void op_add_forward(causa_op_t* op);
static void op_add_backward(causa_op_t* op, causa_tensor_t* grad_output);
static void op_sub_forward(causa_op_t* op);
static void op_sub_backward(causa_op_t* op, causa_tensor_t* grad_output);
static void op_mul_forward(causa_op_t* op);
static void op_mul_backward(causa_op_t* op, causa_tensor_t* grad_output);
static void op_div_forward(causa_op_t* op);
static void op_div_backward(causa_op_t* op, causa_tensor_t* grad_output);
static void op_neg_forward(causa_op_t* op);
static void op_neg_backward(causa_op_t* op, causa_tensor_t* grad_output);
static void op_abs_forward(causa_op_t* op);
static void op_abs_backward(causa_op_t* op, causa_tensor_t* grad_output);
static void op_sqrt_forward(causa_op_t* op);
static void op_sqrt_backward(causa_op_t* op, causa_tensor_t* grad_output);
static void op_rsqrt_forward(causa_op_t* op);
static void op_rsqrt_backward(causa_op_t* op, causa_tensor_t* grad_output);
static void op_exp_forward(causa_op_t* op);
static void op_exp_backward(causa_op_t* op, causa_tensor_t* grad_output);
static void op_log_forward(causa_op_t* op);
static void op_log_backward(causa_op_t* op, causa_tensor_t* grad_output);
static void op_pow_forward(causa_op_t* op);
static void op_pow_backward(causa_op_t* op, causa_tensor_t* grad_output);
static void op_activation_forward(causa_op_t* op);
static void op_activation_backward(causa_op_t* op, causa_tensor_t* grad_output);
static void op_matmul_forward(causa_op_t* op);
static void op_matmul_backward(causa_op_t* op, causa_tensor_t* grad_output);
static void op_copy_forward(causa_op_t* op);
static void op_copy_backward(causa_op_t* op, causa_tensor_t* grad_output);
static void op_reduce_forward(causa_op_t* op);
static void op_reduce_backward(causa_op_t* op, causa_tensor_t* grad_output);
static void op_transpose_forward(causa_op_t* op);
static void op_transpose_backward(causa_op_t* op, causa_tensor_t* grad_output);
static void op_diag_forward(causa_op_t* op);
static void op_diag_backward(causa_op_t* op, causa_tensor_t* grad_output);

// Get standard forward/backward for op type
static void get_standard_op_fns(causa_op_type_t type, causa_op_forward_fn* forward, causa_op_backward_fn* backward) {
    *forward = NULL;
    *backward = NULL;

    switch (type) {
        case CAUSA_OP_ADD:
            *forward = op_add_forward; *backward = op_add_backward; break;
        case CAUSA_OP_SUB:
            *forward = op_sub_forward; *backward = op_sub_backward; break;
        case CAUSA_OP_MUL:
            *forward = op_mul_forward; *backward = op_mul_backward; break;
        case CAUSA_OP_DIV:
            *forward = op_div_forward; *backward = op_div_backward; break;
        case CAUSA_OP_NEG:
            *forward = op_neg_forward; *backward = op_neg_backward; break;
        case CAUSA_OP_ABS:
            *forward = op_abs_forward; *backward = op_abs_backward; break;
        case CAUSA_OP_SQRT:
            *forward = op_sqrt_forward; *backward = op_sqrt_backward; break;
        case CAUSA_OP_RSQRT:
            *forward = op_rsqrt_forward; *backward = op_rsqrt_backward; break;
        case CAUSA_OP_EXP:
            *forward = op_exp_forward; *backward = op_exp_backward; break;
        case CAUSA_OP_LOG:
            *forward = op_log_forward; *backward = op_log_backward; break;
        case CAUSA_OP_POW:
            *forward = op_pow_forward; *backward = op_pow_backward; break;
        case CAUSA_OP_ACTIVATION:
            *forward = op_activation_forward; *backward = op_activation_backward; break;
        case CAUSA_OP_MATMUL:
            *forward = op_matmul_forward; *backward = op_matmul_backward; break;
        case CAUSA_OP_COPY:
            *forward = op_copy_forward; *backward = op_copy_backward; break;
        case CAUSA_OP_REDUCE:
            *forward = op_reduce_forward; *backward = op_reduce_backward; break;
        case CAUSA_OP_TRANSPOSE:
            *forward = op_transpose_forward; *backward = op_transpose_backward; break;
        case CAUSA_OP_DIAG:
            *forward = op_diag_forward; *backward = op_diag_backward; break;
        default:
            break;
    }
}

static void graph_grow_nodes(causa_graph_t* graph) {
    int32_t new_cap = graph->capacity_nodes ? graph->capacity_nodes * 2 : 64;
    causa_node_t** new_nodes = (causa_node_t**)causa_arena_alloc(graph->arena, sizeof(causa_node_t*) * new_cap, CAUSA_ALIGNOF(causa_node_t*));
    if (graph->nodes) {
        memcpy(new_nodes, graph->nodes, graph->num_nodes * sizeof(causa_node_t*));
    }
    graph->nodes = new_nodes;
    graph->capacity_nodes = new_cap;
}

static void graph_grow_tensors(causa_graph_t* graph) {
    int32_t new_cap = graph->capacity_tensors ? graph->capacity_tensors * 2 : 64;
    causa_tensor_t** new_tensors = (causa_tensor_t**)causa_arena_alloc(graph->arena, sizeof(causa_tensor_t*) * new_cap, CAUSA_ALIGNOF(causa_tensor_t*));
    if (graph->tensors) {
        memcpy(new_tensors, graph->tensors, graph->num_tensors * sizeof(causa_tensor_t*));
    }
    graph->tensors = new_tensors;
    graph->capacity_tensors = new_cap;
}

static void graph_grow_ops(causa_graph_t* graph) {
    int32_t new_cap = graph->capacity_ops ? graph->capacity_ops * 2 : 64;
    causa_op_t** new_ops = (causa_op_t**)causa_arena_alloc(graph->arena, sizeof(causa_op_t*) * new_cap, CAUSA_ALIGNOF(causa_op_t*));
    if (graph->ops) {
        memcpy(new_ops, graph->ops, graph->num_ops * sizeof(causa_op_t*));
    }
    graph->ops = new_ops;
    graph->capacity_ops = new_cap;
}

causa_graph_t* causa_graph_new(causa_arena_t* arena, causa_backend_t* backend) {
    if (!arena) return NULL;

    causa_graph_t* graph = (causa_graph_t*)causa_arena_alloc(arena, sizeof(causa_graph_t), CAUSA_ALIGNOF(causa_graph_t));
    if (!graph) return NULL;

    graph->arena = arena;
    graph->backend = backend;
    graph->nodes = NULL;
    graph->num_nodes = 0;
    graph->capacity_nodes = 0;
    graph->tensors = NULL;
    graph->num_tensors = 0;
    graph->capacity_tensors = 0;
    graph->ops = NULL;
    graph->num_ops = 0;
    graph->capacity_ops = 0;
    graph->inputs = NULL;
    graph->num_inputs = 0;
    graph->outputs = NULL;
    graph->num_outputs = 0;
    graph->built = false;
    graph->training = true;

    return graph;
}

void causa_graph_free(causa_graph_t* graph) {
    (void)graph;
}

causa_tensor_t* causa_graph_add_tensor(causa_graph_t* graph, causa_tensor_t* tensor) {
    if (!graph || !tensor) return NULL;
    if (graph->num_tensors >= graph->capacity_tensors) graph_grow_tensors(graph);
    causa_tensor_retain(tensor);
    graph->tensors[graph->num_tensors++] = tensor;
    return tensor;
}

causa_tensor_t* causa_graph_new_tensor(
    causa_graph_t* graph,
    const int64_t* shape,
    int32_t ndim,
    const causa_tensor_options_t* options
) {
    if (!graph) return NULL;
    causa_tensor_t* tensor = causa_tensor_new(graph->arena, graph->backend, shape, ndim, options);
    if (!tensor) return NULL;
    return causa_graph_add_tensor(graph, tensor);
}

causa_op_t* causa_op_new(
    causa_arena_t* arena,
    causa_op_type_t type,
    const char* name,
    causa_tensor_t** inputs,
    int32_t num_inputs,
    causa_tensor_t** outputs,
    int32_t num_outputs,
    causa_op_forward_fn forward,
    causa_op_backward_fn backward,
    void* ctx,
    size_t ctx_size
) {
    if (!arena || num_inputs > CAUSA_MAX_OP_INPUTS || num_outputs > CAUSA_MAX_OP_OUTPUTS) return NULL;

    causa_op_t* op = (causa_op_t*)causa_arena_alloc(arena, sizeof(causa_op_t), CAUSA_ALIGNOF(causa_op_t));
    if (!op) return NULL;

    op->type = type;
    if (name) {
        strncpy(op->name, name, CAUSA_MAX_NAME - 1);
        op->name[CAUSA_MAX_NAME - 1] = '\0';
    } else {
        op->name[0] = '\0';
    }

    op->num_inputs = num_inputs;
    op->num_outputs = num_outputs;
    for (int32_t i = 0; i < num_inputs; i++) {
        op->inputs[i] = inputs[i];
        causa_tensor_retain(inputs[i]);
    }
    for (int32_t i = 0; i < num_outputs; i++) {
        op->outputs[i] = outputs[i];
        causa_tensor_retain(outputs[i]);
    }

    op->forward = forward;
    op->backward = backward;
    op->ctx = ctx;
    op->ctx_size = ctx_size;
    op->node = NULL;

    return op;
}

causa_op_t* causa_graph_add_op(
    causa_graph_t* graph,
    causa_op_type_t type,
    const char* name,
    causa_tensor_t** inputs,
    int32_t num_inputs,
    causa_tensor_t** outputs,
    int32_t num_outputs,
    causa_op_forward_fn forward,
    causa_op_backward_fn backward,
    void* ctx,
    size_t ctx_size
) {
    if (!graph) return NULL;

    // Use standard forward/backward if not provided
    if (!forward || !backward) {
        causa_op_forward_fn std_forward = NULL;
        causa_op_backward_fn std_backward = NULL;
        get_standard_op_fns(type, &std_forward, &std_backward);
        if (!forward) forward = std_forward;
        if (!backward) backward = std_backward;
    }

    causa_op_t* op = causa_op_new(graph->arena, type, name, inputs, num_inputs, outputs, num_outputs, forward, backward, ctx, ctx_size);
    if (!op) return NULL;

    if (graph->num_ops >= graph->capacity_ops) graph_grow_ops(graph);
    graph->ops[graph->num_ops++] = op;

    for (int32_t i = 0; i < num_outputs; i++) {
        // A tensor that is already in the graph is not put in again. It
        // would otherwise be listed twice, and a graph that is written and
        // read would grow every time it was read.
        bool known = false;
        for (int32_t k = 0; k < graph->num_tensors; k++) {
            if (graph->tensors[k] == outputs[i]) { known = true; break; }
        }
        if (!known) {
            causa_graph_add_tensor(graph, outputs[i]);
        }
    }

    graph->built = false;
    return op;
}

static void topological_sort_dfs(causa_node_t* node, causa_node_t** sorted, int32_t* index) {
    if (node->visited) return;
    node->visited = true;

    for (int32_t i = 0; i < node->num_successors; i++) {
        topological_sort_dfs(node->successors[i], sorted, index);
    }

    sorted[(*index)++] = node;
}

// Kahn's algorithm: produces an order in which every node precedes the nodes
// that consume its output, which is the order the forward pass needs.
static int32_t topological_sort_kahn(causa_graph_t* graph, causa_node_t** sorted) {
    int32_t n = graph->num_nodes;
    if (n == 0) return 0;

    int32_t* in_degree = (int32_t*)causa_arena_alloc(graph->arena, sizeof(int32_t) * n, CAUSA_ALIGNOF(int32_t));
    causa_node_t** queue = (causa_node_t**)causa_arena_alloc(graph->arena, sizeof(causa_node_t*) * n, CAUSA_ALIGNOF(causa_node_t*));
    if (!in_degree || !queue) return -1;

    // Remember each node's position so a successor can be located in O(1).
    for (int32_t i = 0; i < n; i++) {
        graph->nodes[i]->topo_index = i;
        in_degree[i] = graph->nodes[i]->num_predecessors;
    }

    int32_t head = 0, tail = 0;
    for (int32_t i = 0; i < n; i++) {
        if (in_degree[i] == 0) queue[tail++] = graph->nodes[i];
    }

    int32_t count = 0;
    while (head < tail) {
        causa_node_t* node = queue[head++];
        sorted[count++] = node;

        for (int32_t i = 0; i < node->num_successors; i++) {
            causa_node_t* succ = node->successors[i];
            if (--in_degree[succ->topo_index] == 0) {
                queue[tail++] = succ;
            }
        }
    }

    // Restore the topological index, which now means "position in execution order".
    for (int32_t i = 0; i < count; i++) {
        sorted[i]->topo_index = i;
    }

    return count == n ? 0 : -1;
}

int causa_graph_build(causa_graph_t* graph) {
    if (!graph || graph->built) return 0;

    if (graph->num_ops == 0) {
        graph->built = true;
        return 0;
    }

    if (graph->num_nodes < graph->num_ops) {
        causa_node_t** new_nodes = (causa_node_t**)causa_arena_alloc(graph->arena, sizeof(causa_node_t*) * graph->num_ops, CAUSA_ALIGNOF(causa_node_t*));
        if (graph->nodes) {
            memcpy(new_nodes, graph->nodes, graph->num_nodes * sizeof(causa_node_t*));
        }
        graph->nodes = new_nodes;
        graph->capacity_nodes = graph->num_ops;
    }

    graph->num_nodes = 0;
    for (int32_t i = 0; i < graph->num_ops; i++) {
        causa_op_t* op = graph->ops[i];
        causa_node_t* node = (causa_node_t*)causa_arena_alloc(graph->arena, sizeof(causa_node_t), CAUSA_ALIGNOF(causa_node_t));
        if (!node) return -1;

        node->op = op;
        node->predecessors = NULL;
        node->num_predecessors = 0;
        node->successors = NULL;
        node->num_successors = 0;
        node->topo_index = -1;
        node->visited = false;
        op->node = node;

        graph->nodes[graph->num_nodes++] = node;
    }

    // The build has to know, for every input of an operation, which
    // operation produced it. Looking for it by scanning every operation is
    // quadratic, and the build happens again every time the graph grows, so
    // the producers are put in a table first.
    //
    // The table is open-addressed with linear probing. Its size is a power
    // of two so the index is a mask rather than a division.
    int32_t table_size = 8;
    while (table_size < graph->num_ops * 4) table_size *= 2;
    int64_t table_mask = table_size - 1;

    causa_tensor_t** keys = (causa_tensor_t**)causa_arena_alloc(
        graph->arena, sizeof(causa_tensor_t*) * table_size, CAUSA_ALIGNOF(causa_tensor_t*));
    causa_node_t** values = (causa_node_t**)causa_arena_alloc(
        graph->arena, sizeof(causa_node_t*) * table_size, CAUSA_ALIGNOF(causa_node_t*));
    if (!keys || !values) return -1;
    for (int32_t i = 0; i < table_size; i++) keys[i] = NULL;

    // The producer of a tensor is the operation that writes it.
    for (int32_t k = 0; k < graph->num_ops; k++) {
        causa_op_t* producer = graph->ops[k];
        for (int32_t o = 0; o < producer->num_outputs; o++) {
            causa_tensor_t* key = producer->outputs[o];
            if (!key) continue;
            int64_t slot = (int64_t)(((uintptr_t)key >> 4) & (uint64_t)table_mask);
            while (keys[slot] && keys[slot] != key) {
                slot = (slot + 1) & table_mask;
            }
            keys[slot] = key;
            values[slot] = graph->nodes[k];
        }
    }

    for (int32_t i = 0; i < graph->num_ops; i++) {
        causa_node_t* node = graph->nodes[i];
        causa_op_t* op = node->op;

        int32_t pred_cap = 0;
        causa_node_t** preds = NULL;

        for (int32_t j = 0; j < op->num_inputs; j++) {
            causa_tensor_t* input = op->inputs[j];
            causa_node_t* producer = NULL;
            if (input) {
                int64_t slot = (int64_t)(((uintptr_t)input >> 4) & (uint64_t)table_mask);
                while (keys[slot]) {
                    if (keys[slot] == input) { producer = values[slot]; break; }
                    slot = (slot + 1) & table_mask;
                }
            }
            if (producer) {
                if (node->num_predecessors >= pred_cap) {
                    pred_cap = pred_cap ? pred_cap * 2 : 4;
                    causa_node_t** new_preds = (causa_node_t**)causa_arena_alloc(graph->arena, sizeof(causa_node_t*) * pred_cap, CAUSA_ALIGNOF(causa_node_t*));
                    if (!new_preds) return -1;
                    if (preds) memcpy(new_preds, preds, node->num_predecessors * sizeof(causa_node_t*));
                    preds = new_preds;
                }
                preds[node->num_predecessors++] = producer;
            }
        }
        node->predecessors = preds;
    }

    // The successors of a node are the operations that consume its output.
    // They are found by walking the inputs of every operation and adding
    // the operation to the node that produced the input, so the whole pass
    // costs one step per input rather than a scan of every operation.
    for (int32_t i = 0; i < graph->num_ops; i++) {
        causa_node_t* node = graph->nodes[i];
        causa_op_t* op = node->op;

        for (int32_t j = 0; j < op->num_inputs; j++) {
            causa_tensor_t* input = op->inputs[j];
            if (!input) continue;

            causa_node_t* producer = NULL;
            int64_t slot = (int64_t)(((uintptr_t)input >> 4) & (uint64_t)table_mask);
            while (keys[slot]) {
                if (keys[slot] == input) { producer = values[slot]; break; }
                slot = (slot + 1) & table_mask;
            }
            if (!producer) continue;

            if (producer->num_successors >= producer->succ_cap) {
                producer->succ_cap = producer->succ_cap ? producer->succ_cap * 2 : 4;
                causa_node_t** new_succs = (causa_node_t**)causa_arena_alloc(
                    graph->arena, sizeof(causa_node_t*) * producer->succ_cap, CAUSA_ALIGNOF(causa_node_t*));
                if (!new_succs) return -1;
                if (producer->successors) {
                    memcpy(new_succs, producer->successors,
                           producer->num_successors * sizeof(causa_node_t*));
                }
                producer->successors = new_succs;
            }
            producer->successors[producer->num_successors++] = node;
        }
    }

    causa_node_t** sorted = (causa_node_t**)causa_arena_alloc(graph->arena, sizeof(causa_node_t*) * graph->num_nodes, CAUSA_ALIGNOF(causa_node_t*));

    for (int32_t i = 0; i < graph->num_nodes; i++) {
        graph->nodes[i]->visited = false;
    }

    if (topological_sort_kahn(graph, sorted) != 0) {
        // A cycle in the graph: fall back to the depth-first order so the
        // caller still gets a usable, if not optimal, ordering.
        int32_t sorted_idx = 0;
        for (int32_t i = 0; i < graph->num_nodes; i++) {
            topological_sort_dfs(graph->nodes[i], sorted, &sorted_idx);
        }
    }

    for (int32_t i = 0; i < graph->num_nodes; i++) {
        graph->nodes[i] = sorted[i];
    }

    graph->built = true;
    return 0;
}

void causa_graph_forward(causa_graph_t* graph) {
    if (!graph || !graph->built) return;

    // The graph always computes. Whether the model is in training mode is
    // a question for the operations that behave differently in the two
    // modes, not for the graph itself.
    for (int32_t i = 0; i < graph->num_nodes; i++) {
        causa_op_t* op = graph->nodes[i]->op;
        if (op->forward) {
            op->forward(op);
        }
    }
}

void causa_graph_backward(causa_graph_t* graph, causa_tensor_t** grad_outputs, int32_t num_grad_outputs) {
    if (!graph || !graph->built) return;

    for (int32_t i = graph->num_nodes - 1; i >= 0; i--) {
        causa_op_t* op = graph->nodes[i]->op;
        if (op->backward) {
            causa_tensor_t* grad_output = NULL;
            if (op->num_outputs > 0) {
                if (grad_outputs && num_grad_outputs > 0) {
                    for (int32_t j = 0; j < num_grad_outputs; j++) {
                        if (grad_outputs[j] == op->outputs[0]) {
                            grad_output = grad_outputs[j];
                            break;
                        }
                    }
                }
                if (!grad_output) {
                    grad_output = op->outputs[0]->grad;
                }
            }
            if (grad_output) {
                op->backward(op, grad_output);
            }
        }
    }
}

void causa_graph_zero_grad(causa_graph_t* graph) {
    if (!graph) return;
    for (int32_t i = 0; i < graph->num_tensors; i++) {
        causa_tensor_zero_grad(graph->tensors[i]);
    }
}

void causa_graph_set_training(causa_graph_t* graph, bool training) {
    if (graph) graph->training = training;
}

bool causa_graph_is_training(const causa_graph_t* graph) {
    return graph ? graph->training : false;
}

void causa_op_forward(causa_op_t* op) {
    if (op && op->forward) op->forward(op);
}

void causa_op_backward(causa_op_t* op, causa_tensor_t* grad_output) {
    if (op && op->backward) op->backward(op, grad_output);
}

// ===== STANDARD OP IMPLEMENTATIONS =====

// Put the result of an operation into its output.
//
// If the operation could not be computed the output is set to zero. It used
// to keep whatever the arena held, which made a failure look like a result.
static void op_store(causa_tensor_t* out, causa_tensor_t* result) {
    if (!out) return;
    if (result) {
        causa_tensor_copy(out, result);
    } else {
        causa_tensor_fill(out, 0.0);
    }
}

static void op_add_forward(causa_op_t* op) {
    causa_tensor_t* a = op->inputs[0];
    causa_tensor_t* b = op->inputs[1];
    causa_tensor_t* out = op->outputs[0];
    causa_tensor_t* tmp = causa_tensor_add(a, b);
    op_store(out, tmp);
}

static void op_add_backward(causa_op_t* op, causa_tensor_t* grad_output) {
    causa_tensor_add_backward(op->inputs[0], op->inputs[1], grad_output);
}

static void op_sub_forward(causa_op_t* op) {
    causa_tensor_t* a = op->inputs[0];
    causa_tensor_t* b = op->inputs[1];
    causa_tensor_t* out = op->outputs[0];
    causa_tensor_t* tmp = causa_tensor_sub(a, b);
    op_store(out, tmp);
}

static void op_sub_backward(causa_op_t* op, causa_tensor_t* grad_output) {
    causa_tensor_sub_backward(op->inputs[0], op->inputs[1], grad_output);
}

static void op_mul_forward(causa_op_t* op) {
    causa_tensor_t* a = op->inputs[0];
    causa_tensor_t* b = op->inputs[1];
    causa_tensor_t* out = op->outputs[0];
    causa_tensor_t* tmp = causa_tensor_mul(a, b);
    op_store(out, tmp);
}

static void op_mul_backward(causa_op_t* op, causa_tensor_t* grad_output) {
    causa_tensor_mul_backward(op->inputs[0], op->inputs[1], grad_output);
}

static void op_div_forward(causa_op_t* op) {
    causa_tensor_t* a = op->inputs[0];
    causa_tensor_t* b = op->inputs[1];
    causa_tensor_t* out = op->outputs[0];
    causa_tensor_t* tmp = causa_tensor_div(a, b);
    op_store(out, tmp);
}

static void op_div_backward(causa_op_t* op, causa_tensor_t* grad_output) {
    causa_tensor_div_backward(op->inputs[0], op->inputs[1], grad_output);
}

static void op_neg_forward(causa_op_t* op) {
    causa_tensor_t* a = op->inputs[0];
    causa_tensor_t* out = op->outputs[0];
    causa_tensor_t* tmp = causa_tensor_neg(a);
    op_store(out, tmp);
}

static void op_neg_backward(causa_op_t* op, causa_tensor_t* grad_output) {
    causa_tensor_neg_backward(op->inputs[0], grad_output);
}

static void op_abs_forward(causa_op_t* op) {
    causa_tensor_t* a = op->inputs[0];
    causa_tensor_t* out = op->outputs[0];
    causa_tensor_t* tmp = causa_tensor_abs(a);
    op_store(out, tmp);
}

static void op_abs_backward(causa_op_t* op, causa_tensor_t* grad_output) {
    causa_tensor_abs_backward(op->inputs[0], grad_output);
}

static void op_sqrt_forward(causa_op_t* op) {
    causa_tensor_t* a = op->inputs[0];
    causa_tensor_t* out = op->outputs[0];
    causa_tensor_t* tmp = causa_tensor_sqrt(a);
    op_store(out, tmp);
}

static void op_sqrt_backward(causa_op_t* op, causa_tensor_t* grad_output) {
    causa_tensor_sqrt_backward(op->inputs[0], grad_output);
}

static void op_rsqrt_forward(causa_op_t* op) {
    causa_tensor_t* a = op->inputs[0];
    causa_tensor_t* out = op->outputs[0];
    causa_tensor_t* tmp = causa_tensor_rsqrt(a);
    op_store(out, tmp);
}

static void op_rsqrt_backward(causa_op_t* op, causa_tensor_t* grad_output) {
    causa_tensor_rsqrt_backward(op->inputs[0], grad_output);
}

static void op_exp_forward(causa_op_t* op) {
    causa_tensor_t* a = op->inputs[0];
    causa_tensor_t* out = op->outputs[0];
    causa_tensor_t* tmp = causa_tensor_exp(a);
    op_store(out, tmp);
}

static void op_exp_backward(causa_op_t* op, causa_tensor_t* grad_output) {
    causa_tensor_exp_backward(op->inputs[0], grad_output);
}

static void op_log_forward(causa_op_t* op) {
    causa_tensor_t* a = op->inputs[0];
    causa_tensor_t* out = op->outputs[0];
    causa_tensor_t* tmp = causa_tensor_log(a);
    op_store(out, tmp);
}

static void op_log_backward(causa_op_t* op, causa_tensor_t* grad_output) {
    causa_tensor_log_backward(op->inputs[0], grad_output);
}

static void op_pow_forward(causa_op_t* op) {
    causa_tensor_t* a = op->inputs[0];
    causa_tensor_t* out = op->outputs[0];
    double exp = *(double*)op->ctx;
    causa_tensor_t* tmp = causa_tensor_pow(a, exp);
    op_store(out, tmp);
}

static void op_pow_backward(causa_op_t* op, causa_tensor_t* grad_output) {
    double exp = *(double*)op->ctx;
    causa_tensor_pow_backward(op->inputs[0], exp, grad_output);
}

static void op_activation_forward(causa_op_t* op) {
    causa_tensor_t* a = op->inputs[0];
    causa_tensor_t* out = op->outputs[0];
    causa_activation_t act = *(causa_activation_t*)op->ctx;
    double param = *(double*)((char*)op->ctx + sizeof(causa_activation_t));
    causa_tensor_t* tmp = causa_tensor_activation(a, act, param);
    op_store(out, tmp);
}

static void op_activation_backward(causa_op_t* op, causa_tensor_t* grad_output) {
    // The context holds the activation and its parameter. An operation
    // without one has nothing to differentiate, so there is nothing to do.
    if (!op->ctx) return;
    causa_activation_t act = *(causa_activation_t*)op->ctx;
    double param = *(double*)((char*)op->ctx + sizeof(causa_activation_t));
    causa_tensor_activation_backward(op->inputs[0], act, param, grad_output);
}

static void op_matmul_forward(causa_op_t* op) {
    causa_tensor_t* a = op->inputs[0];
    causa_tensor_t* b = op->inputs[1];
    causa_tensor_t* out = op->outputs[0];
    causa_tensor_t* tmp = causa_tensor_matmul(a, b);
    op_store(out, tmp);
}

static void op_matmul_backward(causa_op_t* op, causa_tensor_t* grad_output) {
    causa_tensor_matmul_backward(op->inputs[0], op->inputs[1], grad_output);
}

static void op_copy_forward(causa_op_t* op) {
    causa_tensor_copy(op->outputs[0], op->inputs[0]);
}

static void op_copy_backward(causa_op_t* op, causa_tensor_t* grad_output) {
    causa_tensor_accumulate_grad(op->inputs[0], grad_output);
}

static void op_reduce_forward(causa_op_t* op) {
    const causa_reduce_ctx_t* ctx = (const causa_reduce_ctx_t*)op->ctx;
    if (!ctx) return;
    causa_tensor_t* r = causa_tensor_reduce(op->inputs[0], ctx->reduce_op,
                                              ctx->dims, ctx->num_dims, ctx->keepdim);
    op_store(op->outputs[0], r);
}

static void op_reduce_backward(causa_op_t* op, causa_tensor_t* grad_output) {
    const causa_reduce_ctx_t* ctx = (const causa_reduce_ctx_t*)op->ctx;
    if (!ctx || !grad_output) return;

    causa_tensor_t* input = op->inputs[0];
    if (!input || !input->requires_grad) return;

    // Allocate the gradient buffer for the input if needed
    if (!input->grad) {
        causa_tensor_options_t gopts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        gopts.dtype = input->dtype;
        gopts.device = input->device;
        gopts.layout = input->layout;
        input->grad = causa_tensor_new(input->arena, input->backend,
                                        input->shape, input->ndim, &gopts);
        if (!input->grad) return;
        causa_tensor_fill(input->grad, 0.0f);
    }

    // The gradient of a reduction broadcasts the output gradient back over the
    // reduced dimensions. With keepdim the reduced axes are size 1, so the
    // output index for an input element is its own index with the reduced
    // axes replaced by 0.
    bool reduce_mask[CAUSA_MAX_DIMS] = {false};
    for (int32_t i = 0; i < ctx->num_dims; i++) {
        int32_t d = ctx->dims[i];
        if (d < 0) d += input->ndim;
        if (d >= 0 && d < input->ndim) reduce_mask[d] = true;
    }

    // Scale for MEAN: each input element contributes 1/N of the output gradient
    float scale = 1.0f;
    if (ctx->reduce_op == CAUSA_REDUCE_MEAN) {
        int64_t n = causa_tensor_numel(input);
        scale = 1.0f / (float)(n ? n : 1);
    }

    const size_t esz = causa_dtype_size(input->dtype);
    float* gi = (float*)input->grad->data;
    const float* go = (const float*)grad_output->data;

    // Walk every input element and find the matching output gradient.
    int64_t total = causa_tensor_numel(input);
    int64_t go_numel = causa_tensor_numel(grad_output);

    // Strides of the output (contiguous, matching the reduced tensor)
    int64_t out_strides[CAUSA_MAX_DIMS];
    int32_t out_ndim = grad_output->ndim;
    if (out_ndim > 0) {
        out_strides[out_ndim - 1] = 1;
        for (int32_t i = out_ndim - 2; i >= 0; i--) {
            out_strides[i] = out_strides[i + 1] * grad_output->shape[i + 1];
        }
    }

    int64_t idx[CAUSA_MAX_DIMS];
    for (int32_t i = 0; i < input->ndim; i++) idx[i] = 0;

    for (int64_t linear = 0; linear < total; linear++) {
        // Decompose the linear index into per-dimension indices
        int64_t rem = linear;
        for (int32_t i = input->ndim - 1; i >= 0; i--) {
            int64_t dim_size = input->shape[i];
            idx[i] = rem % dim_size;
            rem /= dim_size;
        }

        // Map to the output index: reduced axes collapse to 0
        int64_t out_index = 0;
        int32_t out_dim = 0;
        for (int32_t i = 0; i < input->ndim; i++) {
            if (reduce_mask[i]) {
                if (ctx->keepdim) {
                    out_index += 0 * out_strides[out_dim];
                    out_dim++;
                }
            } else {
                out_index += idx[i] * out_strides[out_dim];
                out_dim++;
            }
        }

        // Input offset in elements
        int64_t offset = 0;
        for (int32_t i = 0; i < input->ndim; i++) {
            offset += idx[i] * (input->strides[i] / (int64_t)esz);
        }

        float g = (out_index < go_numel) ? go[out_index] : 0.0f;
        gi[offset] += g * scale;
    }
}

static void op_transpose_forward(causa_op_t* op) {
    const int32_t* dims = (const int32_t*)op->ctx;
    int32_t d0 = dims ? dims[0] : 0;
    int32_t d1 = dims ? dims[1] : 1;
    causa_tensor_t* t = causa_tensor_transpose(op->inputs[0], d0, d1);
    if (t) causa_tensor_copy(op->outputs[0], t);
}

static void op_transpose_backward(causa_op_t* op, causa_tensor_t* grad_output) {
    const int32_t* dims = (const int32_t*)op->ctx;
    int32_t d0 = dims ? dims[0] : 0;
    int32_t d1 = dims ? dims[1] : 1;
    causa_tensor_transpose_backward(op->inputs[0], grad_output, d0, d1);
}

static void op_diag_forward(causa_op_t* op) {
    causa_tensor_t* d = causa_tensor_diag(op->inputs[0]);
    if (d) causa_tensor_copy(op->outputs[0], d);
}

static void op_diag_backward(causa_op_t* op, causa_tensor_t* grad_output) {
    causa_tensor_diag_backward(op->inputs[0], grad_output);
}