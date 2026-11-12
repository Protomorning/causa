#include "causa_io.h"
#include "causa_arena.h"
#include "causa_backend.h"
#include "causa_platform.h"
#include <stdlib.h>
#include <string.h>

// Writer state
struct causa_writer_s {
    FILE* fp;
    bool ok;
    bool header_written;
    char error[128];
};

// Reader state
struct causa_reader_s {
    FILE* fp;
    bool ok;
    bool header_read;
    char error[128];
};

static void writer_fail(causa_writer_t* w, const char* msg) {
    if (w->ok) {
        w->ok = false;
        snprintf(w->error, sizeof(w->error), "%s", msg);
    }
}

static void reader_fail(causa_reader_t* r, const char* msg) {
    if (r->ok) {
        r->ok = false;
        snprintf(r->error, sizeof(r->error), "%s", msg);
    }
}

// ===== Low-level write helpers =====

static bool w_u32(causa_writer_t* w, uint32_t v) {
    if (!w->ok) return false;
    if (fwrite(&v, sizeof(v), 1, w->fp) != 1) {
        writer_fail(w, "write failed");
        return false;
    }
    return true;
}

static bool w_u64(causa_writer_t* w, uint64_t v) {
    if (!w->ok) return false;
    if (fwrite(&v, sizeof(v), 1, w->fp) != 1) {
        writer_fail(w, "write failed");
        return false;
    }
    return true;
}

static bool w_i64(causa_writer_t* w, int64_t v) {
    if (!w->ok) return false;
    if (fwrite(&v, sizeof(v), 1, w->fp) != 1) {
        writer_fail(w, "write failed");
        return false;
    }
    return true;
}

static bool w_f32(causa_writer_t* w, float v) {
    if (!w->ok) return false;
    if (fwrite(&v, sizeof(v), 1, w->fp) != 1) {
        writer_fail(w, "write failed");
        return false;
    }
    return true;
}

static bool w_bool(causa_writer_t* w, bool v) {
    uint8_t b = v ? 1 : 0;
    if (!w->ok) return false;
    if (fwrite(&b, 1, 1, w->fp) != 1) {
        writer_fail(w, "write failed");
        return false;
    }
    return true;
}

static bool w_bytes(causa_writer_t* w, const void* data, size_t nbytes) {
    if (!w->ok) return false;
    if (nbytes == 0) return true;
    if (fwrite(data, 1, nbytes, w->fp) != nbytes) {
        writer_fail(w, "write failed");
        return false;
    }
    return true;
}

static bool w_string(causa_writer_t* w, const char* s) {
    uint32_t len = s ? (uint32_t)strlen(s) : 0;
    if (len >= CAUSA_IO_MAX_STRING) len = CAUSA_IO_MAX_STRING - 1;
    if (!w_u32(w, len)) return false;
    if (len > 0) return w_bytes(w, s, len);
    return true;
}

// ===== Low-level read helpers =====

static bool r_u32(causa_reader_t* r, uint32_t* out) {
    if (!r->ok) return false;
    if (fread(out, sizeof(*out), 1, r->fp) != 1) {
        reader_fail(r, "read failed");
        return false;
    }
    return true;
}

static bool r_u64(causa_reader_t* r, uint64_t* out) {
    if (!r->ok) return false;
    if (fread(out, sizeof(*out), 1, r->fp) != 1) {
        reader_fail(r, "read failed");
        return false;
    }
    return true;
}

static bool r_i64(causa_reader_t* r, int64_t* out) {
    if (!r->ok) return false;
    if (fread(out, sizeof(*out), 1, r->fp) != 1) {
        reader_fail(r, "read failed");
        return false;
    }
    return true;
}

static bool r_f32(causa_reader_t* r, float* out) {
    if (!r->ok) return false;
    if (fread(out, sizeof(*out), 1, r->fp) != 1) {
        reader_fail(r, "read failed");
        return false;
    }
    return true;
}

static bool r_bool(causa_reader_t* r, bool* out) {
    uint8_t b = 0;
    if (!r->ok) return false;
    if (fread(&b, 1, 1, r->fp) != 1) {
        reader_fail(r, "read failed");
        return false;
    }
    *out = (b != 0);
    return true;
}

static bool r_bytes(causa_reader_t* r, void* data, size_t nbytes) {
    if (!r->ok) return false;
    if (nbytes == 0) return true;
    if (fread(data, 1, nbytes, r->fp) != nbytes) {
        reader_fail(r, "read failed");
        return false;
    }
    return true;
}

static bool r_string(causa_reader_t* r, char* buf, size_t buf_size) {
    uint32_t len = 0;
    if (!r_u32(r, &len)) return false;
    if (len >= buf_size) {
        reader_fail(r, "string too long");
        return false;
    }
    if (len > 0 && !r_bytes(r, buf, len)) return false;
    buf[len] = '\0';
    return true;
}

// ===== Writer lifecycle =====

causa_writer_t* causa_writer_open(const char* path) {
    if (!path) return NULL;
    causa_writer_t* w = (causa_writer_t*)calloc(1, sizeof(causa_writer_t));
    if (!w) return NULL;
    w->fp = fopen(path, "wb");
    if (!w->fp) {
        snprintf(w->error, sizeof(w->error), "cannot open %s for writing", path);
        w->ok = false;
        return w;
    }
    w->ok = true;
    return w;
}

void causa_writer_close(causa_writer_t* writer) {
    if (!writer) return;
    if (writer->fp) {
        fclose(writer->fp);
        writer->fp = NULL;
    }
    free(writer);
}

bool causa_writer_is_ok(const causa_writer_t* writer) {
    return writer && writer->ok;
}

const char* causa_writer_error(const causa_writer_t* writer) {
    return writer ? writer->error : "null writer";
}

static bool writer_header(causa_writer_t* w) {
    if (w->header_written) return true;
    if (!w_u32(w, CAUSA_IO_MAGIC)) return false;
    if (!w_u32(w, CAUSA_IO_VERSION)) return false;
    w->header_written = true;
    return true;
}

// ===== Tensor serialization =====

// Serialize one tensor's metadata + payload
static bool write_tensor_record(causa_writer_t* w, const causa_tensor_t* t) {
    if (!w_u32(w, (uint32_t)t->dtype)) return false;
    if (!w_u32(w, (uint32_t)t->device)) return false;
    if (!w_u32(w, (uint32_t)t->layout)) return false;
    if (!w_i64(w, t->ndim)) return false;
    for (int32_t i = 0; i < t->ndim; i++) {
        if (!w_i64(w, t->shape[i])) return false;
    }
    for (int32_t i = 0; i < t->ndim; i++) {
        if (!w_i64(w, t->strides[i])) return false;
    }
    if (!w_i64(w, t->offset)) return false;
    if (!w_bool(w, t->requires_grad)) return false;
    if (!w_string(w, t->name)) return false;

    size_t nbytes = causa_tensor_nbytes(t);
    if (!w_u64(w, (uint64_t)nbytes)) return false;
    if (!w_bytes(w, t->data, nbytes)) return false;

    // Gradient payload if present
    if (t->grad) {
        if (!w_bool(w, true)) return false;
        size_t gbytes = causa_tensor_nbytes(t->grad);
        if (!w_u64(w, (uint64_t)gbytes)) return false;
        if (!w_bytes(w, t->grad->data, gbytes)) return false;
    } else {
        if (!w_bool(w, false)) return false;
    }
    return true;
}

// The number of bytes the context of an operation has to hold.
//
// An operation reads its context without checking how much of it there is,
// so a file that gives less than the operation reads would make the
// operation read past the end of its allocation.
static size_t op_ctx_size_required(causa_op_type_t type) {
    switch (type) {
        case CAUSA_OP_POW:        return sizeof(double);
        case CAUSA_OP_ACTIVATION: return sizeof(causa_activation_t) + sizeof(double);
        case CAUSA_OP_REDUCE:     return sizeof(causa_reduce_ctx_t);
        case CAUSA_OP_TRANSPOSE:  return 2 * sizeof(int32_t);
        default:                  return 0;
    }
}

// Check that the strides of a tensor that came from a file can be trusted.
//
// The strides say where each element sits in the buffer. A file that claims
// a stride of a billion would make every kernel that follows the strides
// read outside the buffer, so such a file is refused rather than believed.
//
// The rules are:
//
//   - every stride is a whole number of elements, because the kernels
//     divide the stride by the element size to get to an element;
//   - every element the shape can reach lies inside the buffer.
//
// A view of a larger buffer breaks the second rule, because the view only
// knows its own elements and not the buffer it sits in. Such a file is
// refused: the loader cannot rebuild it faithfully, and a tensor that looks
// right and is not is worse than one that is refused.
static bool tensor_strides_are_sane(
    const int64_t* shape,
    int32_t ndim,
    const int64_t* strides,
    size_t elem_size,
    size_t nbytes
) {
    if (ndim <= 0) return true;

    int64_t low = 0;
    int64_t high = 0;

    for (int32_t i = 0; i < ndim; i++) {
        int64_t stride = strides[i];

        // The stride has to be a whole number of elements.
        if (stride % (int64_t)elem_size != 0) return false;

        // The distance the last element of this dimension sits at has to be
        // worked out without overflowing.
        int64_t span = (shape[i] > 0) ? (shape[i] - 1) : 0;
        if (stride == INT64_MIN) return false;
        int64_t magnitude = (stride < 0) ? -stride : stride;
        if (magnitude != 0 && span > INT64_MAX / magnitude) return false;

        int64_t reach = span * stride;
        if (reach < 0) low += reach; else high += reach;
    }

    if (low < 0) return false;
    if (high < 0 || (uint64_t)high >= (uint64_t)nbytes) return false;
    return true;
}

static bool read_tensor_record(causa_reader_t* r, causa_tensor_t** out, causa_arena_t* arena, causa_backend_t* backend) {
    uint32_t dtype, device, layout;
    int64_t ndim;
    int64_t shape[CAUSA_MAX_DIMS];
    int64_t strides[CAUSA_MAX_DIMS];
    int64_t offset;
    bool requires_grad;
    char name[CAUSA_MAX_NAME];

    if (!r_u32(r, &dtype)) return false;
    if (!r_u32(r, &device)) return false;
    if (!r_u32(r, &layout)) return false;
    if (!r_i64(r, &ndim)) return false;
    if (ndim < 0 || ndim > CAUSA_MAX_DIMS) {
        reader_fail(r, "invalid ndim");
        return false;
    }
    for (int32_t i = 0; i < ndim; i++) {
        if (!r_i64(r, &shape[i])) return false;
        // A shape that is not made of whole numbers, or one that is so
        // large that the count of elements overflows, would make the
        // allocation below wrong.
        if (shape[i] < 0) {
            reader_fail(r, "invalid shape");
            return false;
        }
    }
    for (int32_t i = 0; i < ndim; i++) {
        if (!r_i64(r, &strides[i])) return false;
    }
    if (!r_i64(r, &offset)) return false;
    (void)offset;   // A view carries its offset in the data pointer now.
    if (!r_bool(r, &requires_grad)) return false;
    if (!r_string(r, name, CAUSA_MAX_NAME)) return false;

    uint64_t nbytes;
    if (!r_u64(r, &nbytes)) return false;

    // The count of elements decides how much memory is set aside, so it is
    // checked against a bound before anything is allocated. A shape that
    // multiplies out to more than the bound would ask for more memory than
    // the file could ever hold.
    int64_t numel = 1;
    for (int32_t i = 0; i < ndim; i++) {
        if (shape[i] != 0 && numel > CAUSA_IO_MAX_ELEMENTS / shape[i]) {
            reader_fail(r, "the tensor has more elements than the format allows");
            return false;
        }
        numel *= shape[i];
    }

    causa_tensor_options_t opts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
    opts.dtype = (causa_dtype_t)dtype;
    opts.device = (causa_device_type_t)device;
    opts.layout = (causa_layout_t)layout;
    opts.requires_grad = requires_grad;
    opts.name = name;

    // An element type that does not exist has no size, so the tensor cannot
    // be built. The file is refused rather than read into nothing.
    if (opts.dtype < 0 || opts.dtype >= CAUSA_DTYPE_COUNT) {
        reader_fail(r, "unknown element type");
        return false;
    }

    causa_tensor_t* t = causa_tensor_new(arena, backend, shape, (int32_t)ndim, &opts);
    if (!t) {
        reader_fail(r, "tensor allocation failed");
        return false;
    }

    // The file says how many bytes the data is. It has to be what the
    // tensor holds, or the copy would run past the end of the buffer.
    size_t expected = causa_tensor_nbytes(t);
    if (nbytes != (uint64_t)expected) {
        reader_fail(r, "the tensor size in the file does not match its shape");
        return false;
    }

    // The strides say where each element sits. They are only restored when
    // they can be trusted, because a kernel that follows them reads the
    // buffer wherever they point.
    if (!tensor_strides_are_sane(shape, (int32_t)ndim, strides,
                                 causa_dtype_size(opts.dtype), expected)) {
        reader_fail(r, "the strides in the file do not match its shape");
        return false;
    }

    // Restore strides exactly (views may have non-contiguous layout)
    for (int32_t i = 0; i < ndim; i++) {
        t->strides[i] = strides[i];
    }

    if (!r_bytes(r, t->data, (size_t)nbytes)) return false;

    bool has_grad;
    if (!r_bool(r, &has_grad)) return false;
    if (has_grad) {
        uint64_t gbytes;
        if (!r_u64(r, &gbytes)) return false;
        causa_tensor_options_t gopts = CAUSA_TENSOR_OPTIONS_DEFAULTS;
        gopts.dtype = t->dtype;
        gopts.device = t->device;
        gopts.layout = t->layout;
        t->grad = causa_tensor_new(arena, backend, shape, (int32_t)ndim, &gopts);
        if (!t->grad) {
            reader_fail(r, "grad allocation failed");
            return false;
        }
        if (gbytes != (uint64_t)causa_tensor_nbytes(t->grad)) {
            reader_fail(r, "the gradient size in the file does not match");
            return false;
        }
        if (!r_bytes(r, t->grad->data, (size_t)gbytes)) return false;
    }

    *out = t;
    return true;
}

// ===== Public writer API =====

bool causa_write_tensor(causa_writer_t* writer, const causa_tensor_t* tensor) {
    if (!writer || !writer->ok || !tensor) return false;
    if (!writer_header(writer)) return false;
    if (!w_u32(writer, CAUSA_IO_SECTION_TENSORS)) return false;
    if (!w_u32(writer, 1)) return false;  // count
    return write_tensor_record(writer, tensor);
}

bool causa_write_tensor_group(causa_writer_t* writer, causa_tensor_t** tensors, int32_t count) {
    if (!writer || !writer->ok || !tensors || count <= 0) return false;
    if (!writer_header(writer)) return false;
    if (!w_u32(writer, CAUSA_IO_SECTION_TENSORS)) return false;
    if (!w_u32(writer, (uint32_t)count)) return false;
    for (int32_t i = 0; i < count; i++) {
        if (!write_tensor_record(writer, tensors[i])) return false;
    }
    return true;
}

// Write the operations of a graph.
//
// The inputs and the outputs are written as indices into the tensor list
// that was written before them, so that the reader can put the graph back
// together.
static bool write_graph_ops(
    causa_writer_t* writer,
    const causa_graph_t* graph,
    causa_tensor_t** list,
    int32_t list_count
) {
    if (!w_u32(writer, (uint32_t)graph->num_ops)) return false;

    for (int32_t i = 0; i < graph->num_ops; i++) {
        const causa_op_t* op = graph->ops[i];
        if (!w_u32(writer, (uint32_t)op->type)) return false;
        if (!w_string(writer, op->name)) return false;

        if (!w_u32(writer, (uint32_t)op->num_inputs)) return false;
        for (int32_t j = 0; j < op->num_inputs; j++) {
            int32_t idx = -1;
            for (int32_t k = 0; k < list_count; k++) {
                if (list[k] == op->inputs[j]) { idx = k; break; }
            }
            if (!w_i64(writer, idx)) return false;
        }

        if (!w_u32(writer, (uint32_t)op->num_outputs)) return false;
        for (int32_t j = 0; j < op->num_outputs; j++) {
            int32_t idx = -1;
            for (int32_t k = 0; k < list_count; k++) {
                if (list[k] == op->outputs[j]) { idx = k; break; }
            }
            if (!w_i64(writer, idx)) return false;
        }

        // Context payload for ops that carry parameters (activation type, etc.)
        if (!w_u32(writer, (uint32_t)op->ctx_size)) return false;
        if (op->ctx_size > 0) {
            // A context without a pointer to go with it would be read from
            // nothing.
            if (!op->ctx) return false;
            if (!w_bytes(writer, op->ctx, op->ctx_size)) return false;
        }
    }
    return true;
}

bool causa_write_graph(causa_writer_t* writer, const causa_graph_t* graph) {
    if (!writer || !writer->ok || !graph) return false;
    if (!writer_header(writer)) return false;

    // Collect all tensors referenced by the graph
    int32_t causa_count = graph->num_tensors;
    if (causa_count > 0) {
        if (!w_u32(writer, CAUSA_IO_SECTION_TENSORS)) return false;
        if (!w_u32(writer, (uint32_t)causa_count)) return false;
        for (int32_t i = 0; i < causa_count; i++) {
            if (!write_tensor_record(writer, graph->tensors[i])) return false;
        }
    }

    // Write op structure
    if (!w_u32(writer, CAUSA_IO_SECTION_GRAPH)) return false;
    return write_graph_ops(writer, graph, graph->tensors, graph->num_tensors);
}

bool causa_write_scm(causa_writer_t* writer, const causa_scm_t* scm) {
    if (!writer || !writer->ok || !scm) return false;
    if (!writer_header(writer)) return false;

    // The tensors that go into the file are the variables, the exogenous
    // noise, the parents and whatever else the graph of the model holds.
    // The graph is what makes the model a model rather than a table of
    // numbers, so it has to be written with it.
    //
    // The list is built here, because the indices in the operations are
    // into this list and not into the graph's own.
    const causa_graph_t* graph = scm->base_graph;
    int32_t parent_count = 0;
    for (int32_t i = 0; i < scm->num_variables; i++) {
        parent_count += scm->variables[i].num_parents;
    }
    int32_t capacity = scm->num_variables + scm->num_exogenous + parent_count + scm->num_variables + (graph ? graph->num_tensors : 0);
    if (capacity <= 0) capacity = 1;

    causa_tensor_t** list = (causa_tensor_t**)malloc((size_t)capacity * sizeof(causa_tensor_t*));
    if (!list) return false;

    int32_t count = 0;
    for (int32_t i = 0; i < scm->num_variables; i++) {
        if (!scm->variables[i].variable) { free(list); return false; }
        list[count++] = scm->variables[i].variable;
    }

    // The noise comes next, because the reader counts on the variables
    // being the first tensors and the noise the ones after them.
    //
    // The registered noise comes first and then the noise the variables
    // point at, so that a model whose noise was never registered as
    // exogenous still keeps it.
    int32_t num_noise = 0;
    for (int32_t i = 0; i < scm->num_exogenous; i++) {
        if (!scm->exogenous_noise[i]) { free(list); return false; }
        bool known = false;
        for (int32_t k = 0; k < count; k++) {
            if (list[k] == scm->exogenous_noise[i]) { known = true; break; }
        }
        if (known) continue;
        list[count++] = scm->exogenous_noise[i];
        num_noise++;
    }
    for (int32_t i = 0; i < scm->num_variables; i++) {
        causa_tensor_t* noise = scm->variables[i].noise;
        if (!noise) continue;
        bool known = false;
        for (int32_t k = 0; k < count; k++) {
            if (list[k] == noise) { known = true; break; }
        }
        if (known) continue;
        list[count++] = noise;
        num_noise++;
    }

    // The parents come after the variables and the noise, because the
    // reader counts on the variables being the first ones and the noise
    // the ones after them.
    for (int32_t i = 0; i < scm->num_variables; i++) {
        for (int32_t p = 0; p < scm->variables[i].num_parents; p++) {
            causa_tensor_t* parent = scm->variables[i].parents[p];
            if (!parent) { free(list); return false; }
            bool known = false;
            for (int32_t k = 0; k < count; k++) {
                if (list[k] == parent) { known = true; break; }
            }
            if (known) continue;
            list[count++] = parent;
        }
    }
    if (graph) {
        for (int32_t i = 0; i < graph->num_tensors; i++) {
            causa_tensor_t* t = graph->tensors[i];
            bool known = false;
            for (int32_t k = 0; k < count; k++) {
                if (list[k] == t) { known = true; break; }
            }
            if (known) continue;
            if (count >= capacity) break;
            list[count++] = t;
        }
    }

    if (!w_u32(writer, CAUSA_IO_SECTION_TENSORS)) { free(list); return false; }
    if (!w_u32(writer, (uint32_t)count)) { free(list); return false; }
    for (int32_t i = 0; i < count; i++) {
        if (!write_tensor_record(writer, list[i])) { free(list); return false; }
    }

    if (!w_u32(writer, CAUSA_IO_SECTION_SCM)) { free(list); return false; }
    if (!w_u32(writer, (uint32_t)scm->num_variables)) { free(list); return false; }
    if (!w_u32(writer, (uint32_t)num_noise)) { free(list); return false; }

    for (int32_t i = 0; i < scm->num_variables; i++) {
        const causa_scm_var_t* v = &scm->variables[i];

        if (!w_u32(writer, (uint32_t)v->num_parents)) { free(list); return false; }
        for (int32_t p = 0; p < v->num_parents; p++) {
            int32_t idx = -1;
            for (int32_t k = 0; k < count; k++) {
                if (list[k] == v->parents[p]) { idx = k; break; }
            }
            if (!w_i64(writer, idx)) { free(list); return false; }
        }

        // Where the noise is in the list of tensors, -1 if there is none
        int32_t noise_idx = -1;
        if (v->noise) {
            for (int32_t k = 0; k < count; k++) {
                if (list[k] == v->noise) { noise_idx = k; break; }
            }
        }
        if (!w_i64(writer, noise_idx)) { free(list); return false; }
        if (!w_bool(writer, v->is_intervened)) { free(list); return false; }
    }

    // The equations of the model
    if (graph) {
        if (!w_u32(writer, CAUSA_IO_SECTION_GRAPH)) { free(list); return false; }
        if (!write_graph_ops(writer, graph, list, count)) { free(list); return false; }
    }

    free(list);
    return true;
}

bool causa_write_memory(causa_writer_t* writer, const causa_memory_t* memory) {
    if (!writer || !writer->ok || !memory) return false;
    if (!writer_header(writer)) return false;

    const causa_memory_config_t* cfg = causa_memory_config(memory);

    // Count occupied slots and collect their tensors
    int32_t causa_count = 0;
    for (int level = 0; level < CAUSA_MEMORY_MAX_LEVELS; level++) {
        int64_t cap = cfg->capacity[level];
        for (int64_t i = 0; i < cap; i++) {
            causa_tensor_t* key = NULL;
            causa_tensor_t* value = NULL;
            if (causa_memory_slot_tensors(memory, (causa_memory_level_t)level, i, &key, &value)) {
                if (key) causa_count++;
                if (value) causa_count++;
            }
        }
    }

    // Write all slot tensors first
    if (causa_count > 0) {
        if (!w_u32(writer, CAUSA_IO_SECTION_TENSORS)) return false;
        if (!w_u32(writer, (uint32_t)causa_count)) return false;
        for (int level = 0; level < CAUSA_MEMORY_MAX_LEVELS; level++) {
            int64_t cap = cfg->capacity[level];
            for (int64_t i = 0; i < cap; i++) {
                causa_tensor_t* key = NULL;
                causa_tensor_t* value = NULL;
                if (causa_memory_slot_tensors(memory, (causa_memory_level_t)level, i, &key, &value)) {
                    if (key && !write_tensor_record(writer, key)) return false;
                    if (value && !write_tensor_record(writer, value)) return false;
                }
            }
        }
    }

    // Write memory structure
    if (!w_u32(writer, CAUSA_IO_SECTION_MEMORY)) return false;
    // The dimensions are whole numbers, so they are written as such. A
    // dimension above two to the twenty fourth would lose its last digits
    // as a float.
    if (!w_i64(writer, cfg->key_dim)) return false;
    if (!w_i64(writer, cfg->value_dim)) return false;
    if (!w_f32(writer, cfg->working_decay_rate)) return false;
    if (!w_f32(writer, cfg->episodic_decay_rate)) return false;
    if (!w_f32(writer, cfg->consolidation_threshold)) return false;
    if (!w_bool(writer, cfg->differentiable)) return false;
    if (!w_f32(writer, causa_memory_time(memory))) return false;

    for (int level = 0; level < CAUSA_MEMORY_MAX_LEVELS; level++) {
        if (!w_i64(writer, cfg->capacity[level])) return false;
    }

    for (int level = 0; level < CAUSA_MEMORY_MAX_LEVELS; level++) {
        int64_t cap = cfg->capacity[level];
        if (!w_i64(writer, cap)) return false;
        for (int64_t i = 0; i < cap; i++) {
            causa_tensor_t* key = NULL;
            causa_tensor_t* value = NULL;
            bool occupied = causa_memory_slot_tensors(memory, (causa_memory_level_t)level, i, &key, &value);
            if (!w_bool(writer, occupied)) return false;
            if (occupied) {
                causa_memory_slot_info_t info = causa_memory_inspect(memory, (causa_memory_level_t)level, i);
                if (!w_f32(writer, info.strength)) return false;
                if (!w_f32(writer, info.timestamp)) return false;
                if (!w_i64(writer, info.access_count)) return false;
            }
        }
    }
    return true;
}

// ===== Reader lifecycle =====

causa_reader_t* causa_reader_open(const char* path) {
    if (!path) return NULL;
    causa_reader_t* r = (causa_reader_t*)calloc(1, sizeof(causa_reader_t));
    if (!r) return NULL;
    r->fp = fopen(path, "rb");
    if (!r->fp) {
        snprintf(r->error, sizeof(r->error), "cannot open %s for reading", path);
        r->ok = false;
        return r;
    }
    r->ok = true;

    // Validate header
    uint32_t magic, version;
    if (!r_u32(r, &magic) || magic != CAUSA_IO_MAGIC) {
        reader_fail(r, "bad magic - not a tensor file");
        return r;
    }
    if (!r_u32(r, &version) || version != CAUSA_IO_VERSION) {
        reader_fail(r, "unsupported version");
        return r;
    }
    r->header_read = true;
    return r;
}

void causa_reader_close(causa_reader_t* reader) {
    if (!reader) return;
    if (reader->fp) {
        fclose(reader->fp);
        reader->fp = NULL;
    }
    free(reader);
}

bool causa_reader_is_ok(const causa_reader_t* reader) {
    return reader && reader->ok;
}

const char* causa_reader_error(const causa_reader_t* reader) {
    return reader ? reader->error : "null reader";
}

causa_tensor_t* causa_read_tensor(
    causa_reader_t* reader,
    causa_arena_t* arena,
    causa_backend_t* backend
) {
    if (!reader || !reader->ok || !arena || !backend) return NULL;

    uint32_t section, count;
    if (!r_u32(reader, &section)) return NULL;
    if (section != CAUSA_IO_SECTION_TENSORS) {
        reader_fail(reader, "expected tensor section");
        return NULL;
    }
    if (!r_u32(reader, &count)) return NULL;
    if (count > CAUSA_IO_MAX_ITEMS) {
        reader_fail(reader, "too many tensors in the file");
        return NULL;
    }

    causa_tensor_t* out = NULL;
    for (uint32_t i = 0; i < count; i++) {
        causa_tensor_t* t = NULL;
        if (!read_tensor_record(reader, &t, arena, backend)) return NULL;
        if (i == 0) out = t;
    }
    return out;
}

int32_t causa_read_tensor_group(
    causa_reader_t* reader,
    causa_arena_t* arena,
    causa_backend_t* backend,
    causa_tensor_t** out_tensors,
    int32_t max_count
) {
    if (!reader || !reader->ok || !arena || !backend || !out_tensors) return -1;

    uint32_t section, count;
    if (!r_u32(reader, &section)) return -1;
    if (section != CAUSA_IO_SECTION_TENSORS) {
        reader_fail(reader, "expected tensor section");
        return -1;
    }
    if (!r_u32(reader, &count)) return -1;
    if (count > CAUSA_IO_MAX_ITEMS) {
        reader_fail(reader, "too many tensors in the file");
        return -1;
    }

    int32_t n = 0;
    for (uint32_t i = 0; i < count; i++) {
        causa_tensor_t* t = NULL;
        if (!read_tensor_record(reader, &t, arena, backend)) return -1;
        // Every record is read, even the ones that do not fit. Stopping
        // early would leave the reader in the middle of the section, and
        // whatever came next would be read from the wrong place.
        if (n < max_count) out_tensors[n++] = t;
    }
    return n;
}

// Read the operations of a graph.
//
// The inputs and the outputs are indices into the tensor list that was
// read before them, so that the graph is put back together the way it
// was.
static bool read_graph_ops(
    causa_reader_t* reader,
    causa_graph_t* graph,
    causa_tensor_t** tensors,
    int32_t num_tensors,
    causa_arena_t* arena
) {
    uint32_t num_ops;
    if (!r_u32(reader, &num_ops)) return false;
    if (num_ops > CAUSA_IO_MAX_ITEMS) {
        reader_fail(reader, "too many operations in the file");
        return false;
    }

    for (uint32_t i = 0; i < num_ops; i++) {
        uint32_t op_type;
        char name[CAUSA_MAX_NAME];
        if (!r_u32(reader, &op_type)) return false;
        if (!r_string(reader, name, CAUSA_MAX_NAME)) return false;

        uint32_t num_inputs;
        if (!r_u32(reader, &num_inputs)) return false;
        if (num_inputs > CAUSA_MAX_OP_INPUTS) {
            reader_fail(reader, "too many inputs on an operation");
            return false;
        }
        causa_tensor_t* inputs[CAUSA_MAX_OP_INPUTS];
        for (uint32_t j = 0; j < num_inputs; j++) {
            int64_t idx;
            if (!r_i64(reader, &idx)) return false;
            inputs[j] = (idx >= 0 && idx < num_tensors) ? tensors[idx] : NULL;
        }

        uint32_t num_outputs;
        if (!r_u32(reader, &num_outputs)) return false;
        if (num_outputs > CAUSA_MAX_OP_OUTPUTS) {
            reader_fail(reader, "too many outputs on an operation");
            return false;
        }
        causa_tensor_t* outputs[CAUSA_MAX_OP_OUTPUTS];
        for (uint32_t j = 0; j < num_outputs; j++) {
            int64_t idx;
            if (!r_i64(reader, &idx)) return false;
            outputs[j] = (idx >= 0 && idx < num_tensors) ? tensors[idx] : NULL;
        }

        uint32_t ctx_size;
        if (!r_u32(reader, &ctx_size)) return false;
        void* ctx = NULL;
        if (ctx_size > 0) {
            if (ctx_size > CAUSA_IO_MAX_ITEMS) {
                reader_fail(reader, "the context of an operation is too large");
                return false;
            }
            // The operation reads its context without checking how much of
            // it there is, so the file has to give at least as much as the
            // operation reads.
            size_t required = op_ctx_size_required((causa_op_type_t)op_type);
            if ((size_t)ctx_size < required) {
                reader_fail(reader, "the context of an operation is too small for it");
                return false;
            }
            ctx = causa_arena_alloc(arena, ctx_size, 8);
            if (!ctx || !r_bytes(reader, ctx, ctx_size)) return false;
        }

        causa_graph_add_op(graph, (causa_op_type_t)op_type, name,
            inputs, (int32_t)num_inputs,
            outputs, (int32_t)num_outputs,
            NULL, NULL, ctx, ctx_size);
    }

    return true;
}

causa_graph_t* causa_read_graph(
    causa_reader_t* reader,
    causa_arena_t* arena,
    causa_backend_t* backend
) {
    if (!reader || !reader->ok || !arena || !backend) return NULL;

    // Read tensors section (may be absent)
    causa_tensor_t** tensors = NULL;
    int32_t num_tensors = 0;

    uint32_t section, count;
    if (!r_u32(reader, &section)) return NULL;

    if (section == CAUSA_IO_SECTION_TENSORS) {
        if (!r_u32(reader, &count)) return NULL;
        if (count > CAUSA_IO_MAX_ITEMS) {
            reader_fail(reader, "too many tensors in the file");
            return NULL;
        }
        tensors = (causa_tensor_t**)causa_arena_alloc(arena, sizeof(causa_tensor_t*) * (count ? count : 1), CAUSA_ALIGNOF(causa_tensor_t*));
        if (!tensors) {
            reader_fail(reader, "could not hold the tensors");
            return NULL;
        }
        for (uint32_t i = 0; i < count; i++) {
            causa_tensor_t* t = NULL;
            if (!read_tensor_record(reader, &t, arena, backend)) return NULL;
            tensors[num_tensors++] = t;
        }
        if (!r_u32(reader, &section)) return NULL;
    }

    if (section != CAUSA_IO_SECTION_GRAPH) {
        reader_fail(reader, "expected graph section");
        return NULL;
    }

    causa_graph_t* graph = causa_graph_new(arena, backend);
    if (!graph) {
        reader_fail(reader, "could not create the graph");
        return NULL;
    }

    // Register tensors with the graph, in the order they were written, so
    // that the indices in the operations still point at the right ones.
    for (int32_t i = 0; i < num_tensors; i++) {
        causa_graph_add_tensor(graph, tensors[i]);
    }

    if (!read_graph_ops(reader, graph, tensors, num_tensors, arena)) return NULL;

    causa_graph_build(graph);
    return graph;
}

causa_scm_t* causa_read_scm(
    causa_reader_t* reader,
    causa_arena_t* arena,
    causa_backend_t* backend
) {
    if (!reader || !reader->ok || !arena || !backend) return NULL;

    uint32_t section, count;
    if (!r_u32(reader, &section)) return NULL;

    causa_tensor_t** tensors = NULL;
    int32_t num_tensors = 0;

    if (section == CAUSA_IO_SECTION_TENSORS) {
        if (!r_u32(reader, &count)) return NULL;
        if (count > CAUSA_IO_MAX_ITEMS) {
            reader_fail(reader, "too many tensors in the file");
            return NULL;
        }
        tensors = (causa_tensor_t**)causa_arena_alloc(arena, sizeof(causa_tensor_t*) * (count ? count : 1), CAUSA_ALIGNOF(causa_tensor_t*));
        if (!tensors) {
            reader_fail(reader, "could not hold the tensors");
            return NULL;
        }
        for (uint32_t i = 0; i < count; i++) {
            causa_tensor_t* t = NULL;
            if (!read_tensor_record(reader, &t, arena, backend)) return NULL;
            tensors[num_tensors++] = t;
        }
        if (!r_u32(reader, &section)) return NULL;
    }

    if (section != CAUSA_IO_SECTION_SCM) {
        reader_fail(reader, "expected SCM section");
        return NULL;
    }

    uint32_t num_vars, num_exog;
    if (!r_u32(reader, &num_vars)) return NULL;
    if (!r_u32(reader, &num_exog)) return NULL;

    if (num_vars > CAUSA_MAX_VARIABLES) {
        reader_fail(reader, "too many SCM variables");
        return NULL;
    }
    if (num_exog > CAUSA_MAX_VARIABLES) {
        reader_fail(reader, "too many exogenous variables");
        return NULL;
    }

    causa_scm_t* scm = causa_scm_new(arena, backend, NULL);
    if (!scm) {
        reader_fail(reader, "could not create the model");
        return NULL;
    }
    scm->num_exogenous = (int32_t)num_exog;

    // The variables are the first tensors of the file and the noise the
    // ones after them. The graph's other tensors come after those.
    int64_t needed = (int64_t)num_vars + (int64_t)num_exog;
    if ((int64_t)num_tensors < needed) {
        reader_fail(reader, "not enough tensors for SCM");
        return NULL;
    }

    scm->exogenous_noise = (causa_tensor_t**)causa_arena_alloc(arena, sizeof(causa_tensor_t*) * (num_exog ? num_exog : 1), CAUSA_ALIGNOF(causa_tensor_t*));
    if (!scm->exogenous_noise) {
        reader_fail(reader, "could not hold the exogenous variables");
        return NULL;
    }
    for (uint32_t i = 0; i < num_exog; i++) {
        scm->exogenous_noise[i] = tensors[num_vars + i];
    }

    for (uint32_t i = 0; i < num_vars; i++) {
        uint32_t num_parents;
        if (!r_u32(reader, &num_parents)) return NULL;
        if (num_parents > CAUSA_MAX_PARENTS) {
            reader_fail(reader, "too many parents on a variable");
            return NULL;
        }

        causa_tensor_t** parents = NULL;
        if (num_parents > 0) {
            parents = (causa_tensor_t**)causa_arena_alloc(arena, sizeof(causa_tensor_t*) * num_parents, CAUSA_ALIGNOF(causa_tensor_t*));
            if (!parents) {
                reader_fail(reader, "could not hold the parents");
                return NULL;
            }
            for (uint32_t p = 0; p < num_parents; p++) {
                int64_t idx;
                if (!r_i64(reader, &idx)) return NULL;
                if (idx < 0 || idx >= num_tensors) {
                    reader_fail(reader, "a parent points outside the tensors");
                    return NULL;
                }
                parents[p] = tensors[idx];
            }
        }

        int64_t noise_idx;
        if (!r_i64(reader, &noise_idx)) return NULL;

        bool intervened;
        if (!r_bool(reader, &intervened)) return NULL;

        causa_tensor_t* noise = (noise_idx >= 0 && noise_idx < num_tensors) ? tensors[noise_idx] : NULL;

        causa_scm_add_equation(scm, tensors[i], parents, (int32_t)num_parents, noise, NULL, NULL);
        scm->variables[scm->num_variables - 1].is_intervened = intervened;
    }

    // The equations of the model, if they were written.
    //
    // Without them the model would be a table of numbers rather than
    // something that can be run, so the graph is put back together and
    // the variables are pointed at the tensors it holds.
    if (!r_u32(reader, &section)) return NULL;
    if (section == CAUSA_IO_SECTION_GRAPH) {
        causa_graph_t* graph = causa_graph_new(arena, backend);
        if (!graph) {
            reader_fail(reader, "could not create the graph");
            return NULL;
        }
        for (int32_t i = 0; i < num_tensors; i++) {
            causa_graph_add_tensor(graph, tensors[i]);
        }
        if (!read_graph_ops(reader, graph, tensors, num_tensors, arena)) return NULL;
        causa_graph_build(graph);
        scm->base_graph = graph;
    }

    causa_scm_build(scm);
    return scm;
}

causa_memory_t* causa_read_memory(
    causa_reader_t* reader,
    causa_arena_t* arena,
    causa_backend_t* backend
) {
    if (!reader || !reader->ok || !arena || !backend) return NULL;

    uint32_t section, count;
    if (!r_u32(reader, &section)) return NULL;

    causa_tensor_t** tensors = NULL;
    int32_t num_tensors = 0;

    if (section == CAUSA_IO_SECTION_TENSORS) {
        if (!r_u32(reader, &count)) return NULL;
        if (count > CAUSA_IO_MAX_ITEMS) {
            reader_fail(reader, "too many tensors in the file");
            return NULL;
        }
        tensors = (causa_tensor_t**)causa_arena_alloc(arena, sizeof(causa_tensor_t*) * (count ? count : 1), CAUSA_ALIGNOF(causa_tensor_t*));
        if (!tensors) {
            reader_fail(reader, "could not hold the tensors");
            return NULL;
        }
        for (uint32_t i = 0; i < count; i++) {
            causa_tensor_t* t = NULL;
            if (!read_tensor_record(reader, &t, arena, backend)) return NULL;
            tensors[num_tensors++] = t;
        }
        if (!r_u32(reader, &section)) return NULL;
    }

    if (section != CAUSA_IO_SECTION_MEMORY) {
        reader_fail(reader, "expected memory section");
        return NULL;
    }

    int64_t key_dim, value_dim;
    float working_decay, episodic_decay, consolidation_threshold;
    bool differentiable;
    float global_time;

    if (!r_i64(reader, &key_dim)) return NULL;
    if (!r_i64(reader, &value_dim)) return NULL;
    if (!r_f32(reader, &working_decay)) return NULL;
    if (!r_f32(reader, &episodic_decay)) return NULL;
    if (!r_f32(reader, &consolidation_threshold)) return NULL;
    if (!r_bool(reader, &differentiable)) return NULL;
    if (!r_f32(reader, &global_time)) return NULL;

    if (key_dim <= 0 || value_dim <= 0 || key_dim > CAUSA_MAX_DIMS * 1024 || value_dim > CAUSA_MAX_DIMS * 1024) {
        reader_fail(reader, "invalid memory dimensions");
        return NULL;
    }

    causa_memory_config_t cfg = CAUSA_MEMORY_CONFIG_DEFAULTS;
    cfg.key_dim = key_dim;
    cfg.value_dim = value_dim;
    cfg.working_decay_rate = working_decay;
    cfg.episodic_decay_rate = episodic_decay;
    cfg.consolidation_threshold = consolidation_threshold;
    cfg.differentiable = differentiable;

    int64_t capacities[CAUSA_MEMORY_MAX_LEVELS];
    for (int level = 0; level < CAUSA_MEMORY_MAX_LEVELS; level++) {
        int64_t cap;
        if (!r_i64(reader, &cap)) return NULL;
        // A capacity decides how much memory is set aside, so a file that
        // asks for an enormous one is refused rather than believed.
        if (cap < 0 || cap > CAUSA_IO_MAX_ITEMS) {
            reader_fail(reader, "invalid memory capacity");
            return NULL;
        }
        cfg.capacity[level] = cap;
        capacities[level] = cap;
    }

    causa_memory_t* memory = causa_memory_new(arena, backend, &cfg);
    if (!memory) {
        reader_fail(reader, "memory allocation failed");
        return NULL;
    }
    causa_memory_set_time(memory, global_time);

    // Read slot descriptors, matching tensors in order
    int32_t causa_idx = 0;
    for (int level = 0; level < CAUSA_MEMORY_MAX_LEVELS; level++) {
        int64_t cap;
        if (!r_i64(reader, &cap)) return NULL;
        // The capacities are written twice, so that the slot list can be
        // read without looking at the configuration first. The two have to
        // agree, or the stream would fall out of step.
        if (cap != capacities[level]) {
            reader_fail(reader, "the memory capacities do not match");
            return NULL;
        }
        for (int64_t i = 0; i < cap; i++) {
            bool occupied;
            if (!r_bool(reader, &occupied)) return NULL;
            if (!occupied) continue;

            float strength, timestamp;
            int64_t access_count;
            if (!r_f32(reader, &strength)) return NULL;
            if (!r_f32(reader, &timestamp)) return NULL;
            if (!r_i64(reader, &access_count)) return NULL;

            if (causa_idx + 2 > num_tensors) {
                reader_fail(reader, "ran out of tensors for memory slots");
                return NULL;
            }
            causa_tensor_t* key = tensors[causa_idx++];
            causa_tensor_t* value = tensors[causa_idx++];

            if (!causa_memory_restore_slot(memory, (causa_memory_level_t)level, i,
                    key, value, strength, timestamp, (int32_t)access_count)) {
                reader_fail(reader, "failed to restore memory slot");
                return NULL;
            }
        }
    }

    return memory;
}

// ===== Convenience whole-file API =====

bool causa_save_tensor(const char* path, const causa_tensor_t* tensor) {
    causa_writer_t* w = causa_writer_open(path);
    if (!w) return false;
    bool ok = causa_write_tensor(w, tensor);
    if (ok) ok = w_u32(w, CAUSA_IO_SECTION_END);
    causa_writer_close(w);
    return ok;
}

causa_tensor_t* causa_load_tensor(
    const char* path,
    causa_arena_t* arena,
    causa_backend_t* backend
) {
    causa_reader_t* r = causa_reader_open(path);
    if (!r || !causa_reader_is_ok(r)) {
        if (r) causa_reader_close(r);
        return NULL;
    }
    causa_tensor_t* t = causa_read_tensor(r, arena, backend);
    causa_reader_close(r);
    return t;
}

bool causa_save_memory(const char* path, const causa_memory_t* memory) {
    causa_writer_t* w = causa_writer_open(path);
    if (!w) return false;
    bool ok = causa_write_memory(w, memory);
    if (ok) ok = w_u32(w, CAUSA_IO_SECTION_END);
    causa_writer_close(w);
    return ok;
}

causa_memory_t* causa_load_memory(
    const char* path,
    causa_arena_t* arena,
    causa_backend_t* backend
) {
    causa_reader_t* r = causa_reader_open(path);
    if (!r || !causa_reader_is_ok(r)) {
        if (r) causa_reader_close(r);
        return NULL;
    }
    causa_memory_t* m = causa_read_memory(r, arena, backend);
    causa_reader_close(r);
    return m;
}

bool causa_save_scm(const char* path, const causa_scm_t* scm) {
    causa_writer_t* w = causa_writer_open(path);
    if (!w) return false;
    bool ok = causa_write_scm(w, scm);
    if (ok) ok = w_u32(w, CAUSA_IO_SECTION_END);
    causa_writer_close(w);
    return ok;
}

causa_scm_t* causa_load_scm(
    const char* path,
    causa_arena_t* arena,
    causa_backend_t* backend
) {
    causa_reader_t* r = causa_reader_open(path);
    if (!r || !causa_reader_is_ok(r)) {
        if (r) causa_reader_close(r);
        return NULL;
    }
    causa_scm_t* s = causa_read_scm(r, arena, backend);
    causa_reader_close(r);
    return s;
}
