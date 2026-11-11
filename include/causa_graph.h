#ifndef CAUSA_GRAPH_H
#define CAUSA_GRAPH_H

#include "causa_tensor.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CAUSA_MAX_OP_INPUTS 8
#define CAUSA_MAX_OP_OUTPUTS 4

typedef enum {
    CAUSA_OP_NONE = 0,
    CAUSA_OP_ADD,
    CAUSA_OP_SUB,
    CAUSA_OP_MUL,
    CAUSA_OP_DIV,
    CAUSA_OP_MATMUL,
    CAUSA_OP_BMM,
    CAUSA_OP_CONV1D,
    CAUSA_OP_CONV2D,
    CAUSA_OP_TRANSPOSE,
    CAUSA_OP_RESHAPE,
    CAUSA_OP_SLICE,
    CAUSA_OP_REDUCE,
    CAUSA_OP_ACTIVATION,
    CAUSA_OP_COPY,
    CAUSA_OP_NEG,
    CAUSA_OP_ABS,
    CAUSA_OP_SQRT,
    CAUSA_OP_RSQRT,
    CAUSA_OP_EXP,
    CAUSA_OP_LOG,
    CAUSA_OP_POW,
    CAUSA_OP_DIAG,
    CAUSA_OP_CUSTOM
} causa_op_type_t;

typedef struct causa_op_s causa_op_t;
typedef struct causa_graph_s causa_graph_t;
typedef struct causa_node_s causa_node_t;

// Context carried by a REDUCE op: which reduction, and which dimensions
typedef struct {
    causa_reduce_op_t reduce_op;
    int32_t num_dims;
    bool keepdim;
    int32_t dims[CAUSA_MAX_DIMS];
} causa_reduce_ctx_t;

typedef void (*causa_op_forward_fn)(causa_op_t* op);
typedef void (*causa_op_backward_fn)(causa_op_t* op, causa_tensor_t* grad_output);

struct causa_op_s {
    causa_op_type_t type;
    char name[CAUSA_MAX_NAME];

    causa_tensor_t* inputs[CAUSA_MAX_OP_INPUTS];
    int32_t num_inputs;

    causa_tensor_t* outputs[CAUSA_MAX_OP_OUTPUTS];
    int32_t num_outputs;

    causa_op_forward_fn forward;
    causa_op_backward_fn backward;

    void* ctx;
    size_t ctx_size;

    causa_node_t* node;
};

struct causa_node_s {
    causa_op_t* op;
    causa_node_t** predecessors;
    int32_t num_predecessors;
    causa_node_t** successors;
    int32_t num_successors;
    // How many successors the array has room for
    int32_t succ_cap;
    int32_t topo_index;
    bool visited;
};

struct causa_graph_s {
    causa_arena_t* arena;
    causa_backend_t* backend;

    causa_node_t** nodes;
    int32_t num_nodes;
    int32_t capacity_nodes;

    causa_tensor_t** tensors;
    int32_t num_tensors;
    int32_t capacity_tensors;

    causa_op_t** ops;
    int32_t num_ops;
    int32_t capacity_ops;

    causa_tensor_t** inputs;
    int32_t num_inputs;
    causa_tensor_t** outputs;
    int32_t num_outputs;

    bool built;
    bool training;
};

causa_graph_t* causa_graph_new(causa_arena_t* arena, causa_backend_t* backend);
void causa_graph_free(causa_graph_t* graph);

causa_tensor_t* causa_graph_add_tensor(causa_graph_t* graph, causa_tensor_t* tensor);
causa_tensor_t* causa_graph_new_tensor(
    causa_graph_t* graph,
    const int64_t* shape,
    int32_t ndim,
    const causa_tensor_options_t* options
);

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
);

int causa_graph_build(causa_graph_t* graph);
void causa_graph_forward(causa_graph_t* graph);
void causa_graph_backward(causa_graph_t* graph, causa_tensor_t** grad_outputs, int32_t num_grad_outputs);
void causa_graph_zero_grad(causa_graph_t* graph);

void causa_graph_set_training(causa_graph_t* graph, bool training);
bool causa_graph_is_training(const causa_graph_t* graph);

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
);

void causa_op_forward(causa_op_t* op);
void causa_op_backward(causa_op_t* op, causa_tensor_t* grad_output);

#define CAUSA_GRAPH_ADD_UNARY(graph, type, name, input, output, forward, backward, ctx, ctx_size) \
    causa_graph_add_op(graph, type, name, (causa_tensor_t*[]){input}, 1, (causa_tensor_t*[]){output}, 1, forward, backward, ctx, ctx_size)

#define CAUSA_GRAPH_ADD_BINARY(graph, type, name, a, b, output, forward, backward, ctx, ctx_size) \
    causa_graph_add_op(graph, type, name, (causa_tensor_t*[]){a, b}, 2, (causa_tensor_t*[]){output}, 1, forward, backward, ctx, ctx_size)

#ifdef __cplusplus
}
#endif

#endif